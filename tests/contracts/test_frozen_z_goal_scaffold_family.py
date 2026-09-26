"""Contracts for the FROZEN (not dead) z_goal condition in the scaffold-warmed family.

WHAT THE CONDITION IS. ~31 experiment scripts drive their warmup through
`experiments/scaffolded_sd054_onboarding.py`'s `ScaffoldedSD054OnboardingScheduler`,
which calls `agent.update_z_goal(...)` on Stage-0 / P1 / P2 steps, and then hand the
warmed agent to a HAND-ROLLED measurement loop that never calls it again. Two substrate
facts make that consequential:

  - `REEAgent.reset()` resets ~47 subsystems but NOT `goal_state` (its only documented
    exception is residue). So the per-episode reset in the measurement loop does not
    clear the goal.
  - `GoalState`'s decay lives INSIDE `GoalState.update()`, reachable only through
    `update_z_goal`. No call -> no decay.

So z_goal FREEZES at its post-warmup value for the whole measurement phase.
`is_active()` stays True and every goal consumer keeps firing against a goal that no
longer tracks the episode. This is a DIFFERENT and milder defect than the dead-zero
stream gated by `validate_experiments.dead_z_goal_stream_lint`, which discharges this
family on purpose via `_uses_a_z_goal_driving_helper` -- read that docstring first.

WHY THE FREEZE IS NOT THE SCAFFOLD'S OWN "FROZEN GOAL PIPELINE". The scheduler has a
purpose-built primitive for holding the goal still, `_set_goal_pipeline_frozen(agent,
frozen)`, used by Stage-0b consolidation, P0 and Stage-H. It is a PAIR: skip
`update_z_goal` AND short-circuit the MECH-295 liking bridge + MECH-307 conjunction, so
the held goal cannot drive MECH-295/307. (CORRECTION, 2026-07-27 follow-on triage: this
file originally said the pair means "the held goal cannot drive behaviour". It does not.
The pair silences the goal WRITE paths only; the two goal READ paths -- the E3
`goal_weight * goal_proximity` term and E1 goal-conditioning -- are gated independently
and stay LIVE inside every frozen stage. Measured on the 460c dry-run config over 3
seeds: goal_active_frac 1.000 in Stage-0b / P0 / Stage-H, and removing the E3 goal term
counterfactually moves the cost-argmin candidate on 39% / 19% / 38% of those stages'
ticks. See `scaffold_goal_freeze_e3_read_path_triage_2026-07-27.md` and the
`..._does_not_touch_the_read_paths` pin below. FOLLOW-ON, same day: the strict form the
triage chipped is now BUILT as an opt-in, default-OFF knob --
`ScaffoldedSD054OnboardingConfig.scaffold_strict_goal_isolation`, which makes the frozen
stages additionally zero `e3.goal_weight` and clear `e1_goal_conditioned`, restoring the
saved priors on unfreeze. Default False keeps every landed run bit-identical; the
`..._strict_goal_isolation_...` tests below pin both halves.) `run_p1` sets `frozen=False`
and nothing sets it
back, so the agent reaches the measurement phase with the goal held still but the
consumers LIVE -- a combination the scaffold itself never constructs. The measurement
phase inherits exactly half of the primitive, silently: no script in the family
documents its measurement-phase goal state at all.

TRIAGE OUTCOME (2026-07-27). Not a retro-fix. All 28 build the curriculum ONCE per seed
and evaluate every arm from a copy of that one build, so the frozen z_goal is
bit-identical across arms and cannot produce a between-arm difference. Of the 28 landed
manifests, 22 are `non_contributory`, 3 `superseded`, 1 `mixed` (diagnostic), 3 carry no
direction (diagnostic), and exactly ONE -- V3-EXQ-466e, SD-034 -- is `supports`. 466e's
criteria are existence thresholds on the ON arm (n_closures >= 1, discharge_events >= 1)
plus a structural negative control on an OFF clone that has no closure operator at all,
none of which reads z_goal; the frozen goal is arm-symmetric there exactly as the inert
goal term is arm-symmetric for V3-EXQ-615 in the dead-stream lint's own carrier table.

RETROFIT IS NOT FREE. Adding `update_z_goal` to one of these scripts is not a wiring
fix: the call is ALSO the SD-024 benefit-attractor producer (it calls
`ResidueField.accumulate_benefit` ahead of the `goal_state` guard), so it populates
`benefit_rbf_field` and un-zeroes the SD-025 curiosity bonus in
`HippocampalModule._curiosity_bonus`. For THIS family that specific path is gated off --
`residue.benefit_terrain_live_producer` defaults False and none of the 31 (nor the
scheduler) sets it -- but the retrofit still swaps a constant goal for 0.5%/step decay
plus contact reseeding, which moves the E3 goal term on every tick. Either way a patched
script is not comparable to the runs that came before it.

FAMILY GROWTH (2026-08-12). Two new members: V3-EXQ-464e and V3-EXQ-467e (ree-v3
`ed98d7b`), lettered bug-fix iterations of 464d/467d fixing a `_clone_for_arm()`
goal_state-drop bug plus a `salience_affinity_input_cap` substrate calibration -- neither
change touches how the measurement phase handles z_goal. Checked directly rather than
assumed: both build the curriculum ONCE per seed and evaluate every arm via
`_clone_for_arm` (exactly the structural property the 2026-07-27 triage above rests on,
and precisely the function the 464e/467e bug fix targets); neither sets `goal_weight` or
`residue.benefit_terrain_live_producer`; neither calls `update_z_goal` or
`_set_goal_pipeline_frozen`. So the frozen-goal condition applies unchanged and the
TRIAGE OUTCOME's reasoning (arm-symmetric, cannot produce a between-arm difference)
extends to them without re-deriving it -- a fixed goal is the deliberate, inherited state,
consistent with every other member of the family, and per the family's own convention
(no member documents this per-script; it is a family-wide property recorded once here)
neither script carries its own comment about it. `_FROZEN_FAMILY_SIZE` -> 30; the
28-landed-manifest evidence-direction breakdown above is unchanged since neither has a
manifest yet (both `claimed` in the queue, no run landed as of this note).

FAMILY GROWTH (2026-08-14). One new member: V3-EXQ-934
(`v3_exq_934_mech266_cap_sweep_mode_occupancy.py`, ree-v3 `c0998a6`), the GOV-FANOUT-1
leg-H1 `affinity_input_cap` sweep spun out of the 464e/467e cluster autopsy. It arrived
with `_FROZEN_FAMILY_SIZE` still at 30, so trunk's contract gate was red from `c0998a6`
until this note. VERDICT: a fixed goal is INTENDED -- no retrofit, pin 30 -> 31.

Checked directly rather than assumed, same four questions as the 464e/467e note above.
934 builds the curriculum ONCE per seed (one `agent` through
run_stage0_nursery / run_stage0b_consolidation / run_p0 / run_hazard_avoidance / run_p1 /
run_p2) and evaluates every one of its 10 CAP x ARM cells from a `_clone_for_arm()` copy
of that single trained agent -- and that clone explicitly carries `goal_state` across
(`agent.goal_state.load_state_dict(trained_agent.goal_state.state_dict())`, the 464e
fix), so all ten cells enter measurement with a BIT-IDENTICAL frozen z_goal. It sets
neither `goal_weight` nor `residue.benefit_terrain_live_producer`, and calls neither
`update_z_goal` nor `_set_goal_pipeline_frozen`. So the freeze is the deliberate
inherited state, and per the family's own convention (recorded once here, never
per-script) 934 carries no comment of its own about it.

ONE RESPECT IN WHICH 934 IS STRONGER THAN THE REST OF THE FAMILY, stated so it is not
rediscovered as a defect. For 464e/467e the frozen goal reaches behaviour only through
the two generic read paths (the E3 `goal_weight * goal_proximity` term and E1
goal-conditioning). 934 sets `use_external_task_drive=True`, and that injection
(`agent.py:6933-6960`) is gated on `goal_state.is_active()` and adds
`external_task_drive_proximity_weight * goal_proximity(z_world)` -- so the frozen goal
sits directly on the causal path of 934's PRIMARY DV, external_task mode occupancy. Two
things keep that from threatening the H1 read. (i) It is a fixed TARGET, not a frozen
SIGNAL: `goal_proximity` is recomputed per tick against the live `z_world`, so the
engagement scalar still varies within an episode -- the freeze does not flatten the
signal 934 is measuring. (ii) The H1 verdict is a WITHIN-seed comparison across caps
(`sym_graded` = is symmetric-arm occupancy graded rather than saturated across
CAP_SWEEP), and the goal state is identical across all cells of a seed, so it cannot
produce a between-cell difference -- the same arm-symmetry argument the TRIAGE OUTCOME
above rests on, extended over the cap axis.

FAMILY GROWTH (2026-08-18). One new member: V3-EXQ-935
(`v3_exq_935_mech266_margin_normalised_cap_rule.py`, ree-v3 `cbe407e5`, landed
2026-08-16), the MECH-266/SD-032a margin-normalised cap rule spun out of the same
GOV-FANOUT-1 leg as 934. It arrived with `_FROZEN_FAMILY_SIZE` still at 31, so trunk's
contract gate was red from `cbe407e5` until this note. VERDICT: a fixed goal is INTENDED
-- no retrofit, pin 31 -> 32.

Checked directly rather than assumed, same four questions as the 934 note above. 935
builds the curriculum ONCE per seed (`_run_seed` constructs one `agent` and drives it
through run_stage0_nursery / run_stage0b_consolidation / run_p0 / run_hazard_avoidance /
run_p1 / run_p2) and evaluates every cell -- the calibration cell, each ARM_NORM cell of
the r-sweep, and the ARM_ABS cell -- from a `_clone_for_arm()` copy of that single
trained agent, which carries `goal_state` across explicitly
(`agent.goal_state.load_state_dict(trained_agent.goal_state.state_dict())`, the 464e
fix, and 935's own docstring for that function names the gating this addresses). So
every cell of a seed enters measurement with a BIT-IDENTICAL frozen z_goal. It sets
neither `goal_weight` nor `residue.benefit_terrain_live_producer`, and calls neither
`update_z_goal` nor `_set_goal_pipeline_frozen`. The freeze is therefore the deliberate
inherited state, and per the family's own convention (recorded once here, never
per-script) 935 carries no comment of its own about it.

935 inherits 934's "one respect in which it is stronger" verbatim, and for the same
reason: it too sets `use_external_task_drive=True`, so the frozen goal sits on the
causal path of its primary DV (external_task mode occupancy). Both of 934's containment
arguments carry over unchanged -- (i) it is a fixed TARGET, not a frozen SIGNAL, since
`goal_proximity` is recomputed per tick against the live `z_world`; and (ii) 935's H1/C2
read is a WITHIN-seed comparison across the r-sweep (is occupancy graded across
normalised caps), and the goal state is identical across all cells of a seed, so it
cannot produce a between-cell difference. 935 additionally instruments the goal stream
directly (`ZGoalStreamAccumulator`, `_ZG.observe()` at the trained agent and at every
evaluated cell, reported as `z_goal_stream_stats`), so its manifests record the frozen
value rather than leaving it implicit -- the one member of the family that does. The one absolute-threshold criterion
(`margin_engaged`, max continuous external_task margin > MARGIN_FLOOR) does read a level
rather than a contrast, and the frozen post-curriculum goal is exactly the state the
banked cap=2.0 464e/467e reference was measured in -- which is what makes 934 comparable
to it, so driving or re-freezing the goal here would BREAK the comparison this run
exists to make, not repair it.

The 28-landed-manifest evidence-direction breakdown above is again unchanged: 934 is a
DIAGNOSTIC with no manifest yet (`claimed` by ree-cloud-3 as of this note).

FAMILY GROWTH (2026-09-16). One new member: V3-EXQ-935a
(`v3_exq_935a_mech266_margin_normalised_cap_rule.py`, ree-v3 `a8a61c2`, landed
2026-09-14), the corrected re-test of V3-EXQ-935 that fixes the four measurement
defects `failure_autopsy_V3-EXQ-935_2026-08-18.md` Section 5a found (missing H-KNIFE
routing branch, a hardcoded/false `route_reason`, an unchecked cross-substrate R_STAR
import, and a prose-only `ANCHOR_REACHABILITY_EXEMPT` claim) and re-earns the result on
fresh, out-of-sample seeds (47-51, vs 935's 42-46). It arrived with `_FROZEN_FAMILY_SIZE`
still at 32, so trunk's contract gate was red from `a8a61c2` until this note. VERDICT:
a fixed goal is INTENDED -- no retrofit, pin 32 -> 33.

Checked directly rather than assumed, same four questions as the 934/935 notes above.
935a builds the curriculum ONCE per seed (its own `_run_seed`) and evaluates every cell
-- the calibration cell, each cell of the extended r-sweep [2.25, 2.45, 2.65, 2.85,
3.05], and the absolute-cap cell -- from a `_clone_for_arm()` copy of that single trained
agent, which still carries `goal_state` across explicitly
(`agent.goal_state.load_state_dict(trained_agent.goal_state.state_dict())`, unchanged
from 935). So every cell of a seed enters measurement with a BIT-IDENTICAL frozen
z_goal. Like 935, it sets neither `goal_weight` nor
`residue.benefit_terrain_live_producer`, and calls neither `update_z_goal` nor
`_set_goal_pipeline_frozen`. The freeze is therefore the deliberate inherited state,
and per the family's own convention (recorded once here, never per-script) 935a carries
no comment of its own about it.

935a inherits 934/935's "one respect in which it is stronger" verbatim and for the same
reason: it too sets `use_external_task_drive=True` (line 563), so the frozen goal sits
on the causal path of its primary DV (external_task mode occupancy / the H-RULE vs
H-IDIO vs H-KNIFE routing). Both containment arguments carry over unchanged -- (i) it is
a fixed TARGET, not a frozen SIGNAL, since `goal_proximity` is recomputed per tick
against the live `z_world`; and (ii) 935a's read is a WITHIN-seed comparison across the
r-sweep (does occupancy grade at a single pre-registered r, simultaneously, across
fresh seeds), and the goal state is identical across all cells of a seed, so it cannot
produce a between-cell difference. 935a also keeps 935's direct goal-stream
instrumentation (`ZGoalStreamAccumulator`, `_ZG.observe()` at the trained agent and at
every evaluated cell, reported as `z_goal_stream_stats`) -- unchanged from 935, so the
frozen value is recorded rather than left implicit here too.

935a's own manifest landed 2026-09-16T09:58:09Z: `outcome=FAIL`,
`evidence_direction=non_contributory` (R_STAR=2.45 did not clear the pre-registered bar
on the fresh seeds; H-KNIFE routing -- newly wired by this run -- determines whether that
is idiosyncrasy or a further-correctable rule, per the autopsy's routing). The frozen
z_goal plays no role in that outcome for the same arm-symmetry reason as every other
member of this family: whatever the goal freezes at, it is identical across every cell
of a given seed, so it cannot explain a between-cell or between-seed difference in
occupancy grading.

FAMILY GROWTH (2026-09-24). One new member: V3-EXQ-1090
(`v3_exq_1090_mech449_endogenous_safety_veto_validation.py`), the MECH-449 endogenous
safety-producer validation (chip-20260918-mech449-endogenous-safety-veto-producer).
VERDICT: a fixed goal is INTENDED -- no retrofit, pin 34 -> 35. Checked directly: it
drives each (arm, seed) cell through run_stage0_nursery / run_stage0b_consolidation /
run_p0 / run_hazard_avoidance / run_p1, then a hand-rolled Stage-H eval loop that calls
neither `update_z_goal` nor `_set_goal_pipeline_frozen`, and sets neither `goal_weight`
nor `residue.benefit_terrain_live_producer`. UNLIKE the clone-per-arm members, its two
arms (ARM_HARM_ON / ARM_HARM_OFF_CONTROL) train SEPARATE curricula, so the frozen goal
is not arm-symmetric. That does not matter here because no criterion contrasts the arms:
C1 (safety No-Go applied), C2 (fire rate) and C3 (within-tick paired ground-truth
hazard difference) are within-arm, and C4 is an absolute ceiling on the control arm's
fire rate. The frozen goal enters only through F (the E3 goal term) and therefore only
through which candidates the F envelope admits -- a fixed TARGET recomputed per tick
against the live z_world, the same containment argument (i) as 934/935/935a.

FAMILY GROWTH (2026-09-22). One new member: V3-EXQ-1067
(`v3_exq_1067_mech266_squash_vs_clamp_cap_sweep.py`, ree-v3 `c817881` authored,
`01bfae2` + `a4f9650` red-team passes), the MECH-266/SD-032a squash-vs-clamp cap sweep
-- the next leg of the same 934/935/935a lineage. It arrived with `_FROZEN_FAMILY_SIZE`
still at 33, and `a4f9650`'s own commit message ("re-smoke pending") is the admission
that the corpus gate had not been re-run, so trunk's contract gate was red from
`a4f9650` (2026-09-20T01:36:31Z) until this note. VERDICT: a fixed goal is INTENDED --
no retrofit, pin 33 -> 34.

Checked directly rather than assumed, same four questions as the 934/935/935a notes
above. 1067 builds the curriculum ONCE per seed (`_run_seed` drives one `agent` through
run_stage0_nursery / run_stage0b_consolidation / run_p0 / run_hazard_avoidance / run_p1
/ run_p2) and evaluates every one of its CAP x RAIL_ARM x BOUND_ARM cells from a
`_clone_for_arm()` copy of that single trained agent -- one call site, line 1078 -- which
still carries `goal_state` across explicitly
(`agent.goal_state.load_state_dict(trained_agent.goal_state.state_dict())`, the 464e
fix, and 1067's own docstring for that function names the
`external_task_drive_require_goal_active` hard-gate this addresses). It sets neither
`goal_weight` nor `residue.benefit_terrain_live_producer`, and calls neither
`update_z_goal` nor `_set_goal_pipeline_frozen`. The freeze is therefore the deliberate
inherited state, and per the family's own convention (recorded once here, never
per-script) 1067 carries no comment of its own about it.

1067 inherits 934/935/935a's "one respect in which it is stronger" verbatim: it sets
`use_external_task_drive=True` (line 672) with
`external_task_drive_proximity_weight=1.0` (line 676), so the frozen goal sits directly
on the causal path of its primary DV (external_task mode occupancy). Both containment
arguments carry over unchanged -- (i) it is a fixed TARGET, not a frozen SIGNAL, since
`goal_proximity` is recomputed per tick against the live `z_world`; and (ii) the read is
a WITHIN-seed comparison across the cap x arm grid, and the goal state is identical
across every cell of a seed, so it cannot produce a between-cell difference.

TWO RESPECTS IN WHICH 1067 IS STRONGER THAN 935/935a, stated so they are not
rediscovered as defects. (1) Its cells are RNG- AND ENV-PAIRED on top of the shared
clone (red-team findings 2/3): the per-cell seed `cell_rng_seed(seed, cap, arm_label)`
excludes the bound arm, and each cell gets its own identically-seeded env, so the three
bound arms at one (cap, rail_arm) start from bit-identical RNG state as well as a
bit-identical frozen z_goal -- arm-symmetry here is enforced on three axes, not one.
(2) It is the first member to GATE on the frozen goal's magnitude rather than merely
record it: the per-seed guard requires `p2.z_goal_norm_at_contact_peak > P2_ZGOAL_GATE`
(0.4, line 1059) before the measurement phase is allowed to count, so a degenerate or
near-zero frozen goal aborts the seed instead of silently producing an uninterpretable
cell grid. It also keeps 935/935a's direct instrumentation (`ZGoalStreamAccumulator`,
`_ZG.observe()` at the trained agent on line 1061 and at every evaluated cell on line
1088, reported as `z_goal_stream_stats`), so the frozen value is recorded, not implicit.

AND THE FREEZE IS LOAD-BEARING FOR 1067'S COMPARISON, so driving or re-freezing the goal
here would BREAK the run rather than repair it -- the same point 935's note makes about
its banked cap=2.0 reference, but sharper. 1067's clamp arm exists to reproduce 934's
banked cells, which is why it keeps 934's SEEDS = [42, 43, 44] DELIBERATELY (the usual
substitute-seed-44 caution is overridden for that recorded reason) and holds training at
AFFINITY_INPUT_CAP_TRAIN = 2.0 under the clamp so the trained substrate is
bit-comparable to the banked reference. The frozen post-curriculum goal is exactly the
state that reference was measured in. Its one absolute-threshold criterion
(`margin_engaged`, max `ext_margin_mean` > MARGIN_FLOOR) does read a level rather than a
contrast, as 935's did -- but 1067 already hardened that in red-team pass 2 (F4): the
margin is floored by `external_task_bias = 1.0` and so reads 0.33-0.50 even with the
drive at exactly 0.0, so the drive's own engagement is asserted separately as a
zero-test (`drive_engaged`, max `et_drive_mean` > 0.0).

STATUS AT THE TIME OF THIS NOTE: V3-EXQ-1067 was `claimed` by DLAPTOP
(2026-09-20T02:07:22Z) and RUNNING -- the measurement was in flight while this note was
written, which is why the verdict was reached by reading the driver rather than by
editing it. Nothing in the script was touched. No manifest yet, so the
28-landed-manifest evidence-direction breakdown above is again unchanged.

FAMILY GROWTH (2026-09-25). One new member: V3-EXQ-1107
(`v3_exq_1107_sd032a_trained_mode_reversal_drive.py`, ree-v3 `70fb624`,
chip-20260925-mode-switch-trained-agent-run), the SD-032a trained-agent mode-reversal
diagnostic: external_task_drive OFF vs ON on the 935a curriculum. It arrived with
`_FROZEN_FAMILY_SIZE` still at 35, so trunk's contract gate was red from `70fb624`
(2026-09-25T15:10:25Z) until this note (chip-20260925-main-red-contracts-fix). VERDICT:
a fixed goal is INTENDED -- no retrofit, pin 35 -> 36.

Checked directly rather than assumed, same four questions as the 934/935/935a/1067 notes
above. (1) 1107 builds the curriculum ONCE per seed (`_run_seed` drives one `agent`
through run_stage0_nursery / run_stage0b_consolidation / run_p0 / run_hazard_avoidance
/ run_p1 / run_p2) and evaluates both arms -- ARM_DRIVE_OFF and ARM_DRIVE_ON, at every
training checkpoint and at the final eval -- from `_clone_for_arm()` copies of that
single agent (`_run_pair`), which carry `goal_state` across explicitly
(`agent.goal_state.load_state_dict(trained_agent.goal_state.state_dict())`, VERBATIM
from 935a apart from the drive toggle). So both arms of a seed enter measurement with a
BIT-IDENTICAL frozen z_goal, the same env seed and the same RNG state. (2) It sets
neither `goal_weight` nor `residue.benefit_terrain_live_producer`. (3) It calls neither
`update_z_goal` nor `_set_goal_pipeline_frozen`. The freeze is therefore the deliberate
inherited state, and per the family's own convention 1107 carries no comment of its own
about the z_goal freeze as such.

ONE NEW SHAPE, stated so it is not mistaken for a goal write or for a defect. 1107 is
the first member to feed the NATIVE drive scalar while leaving z_goal frozen: every eval
step it sets `agent.goal_state._last_drive_level` (SD-012 clip(1 - energy)) -- red-team
F1's repair, because that scalar's only writer is `update_z_goal`, so under a frozen
goal it sat at 0 in 935a's harness too and made a return to external_task
arithmetically impossible in the OFF arm. That is not a z_goal write:
`validate_experiments._writes_z_goal_directly` correctly does not count it (none of its
four discharges -- update_z_goal, a goal-receiver .update, cue_pull, a `_z_goal`
assignment -- fires), which is why 1107 is IN the family rather than exempted from it.

(4) The frozen goal sits on the causal path of the ON arm (external_task_drive reads
`goal_proximity`, and 935a's goal gate requires an active z_goal) -- 934's "one respect
in which it is stronger", inherited verbatim. Both containment arguments carry over:
(i) it is a fixed TARGET recomputed per tick against the live z_world, not a frozen
SIGNAL; (ii) the goal state is identical across the two arms of a seed, so it cannot
produce a between-arm difference except through the drive itself, which IS the
manipulated variable. The load-bearing criterion (C1, ARM_DRIVE_ON REVERSING on >= 2/3
testable seeds) and the structural control (C2, ARM_DRIVE_OFF ONE_WAY) are both
within-arm. Driving or re-freezing the goal would break the design rather than repair
it: 1107 exists to ask whether 935a's TRAINED agent, in 935a's measurement state, needs
the drive. Like 1067 it GATES on the frozen goal's magnitude (P2 guard,
`z_goal_norm_at_contact_peak > P2_ZGOAL_GATE` = 0.4) and records it
(`ZGoalStreamAccumulator`, `_ZG.observe()` at the trained agent and at every clone).

STATUS AT THE TIME OF THIS NOTE: 1107's manifest landed 2026-09-25T17:53:06Z,
`outcome=FAIL`, `evidence_direction=non_contributory`. Nothing in the script was
touched.

FAMILY GROWTH (2026-09-26). One new member: V3-EXQ-1109
(`v3_exq_1109_pag_freeze_veto_earliest_edge.py`, ree-v3 `89976eb`,
chip-20260926-pag-freeze-lock-confirmer, DCD2 probe F), the freeze/veto earliest-edge
diagnostic on the 1107 harness: F0 z_harm_a information content (scripted random walk),
F1 PAG freeze ON vs OFF, F3 MECH-449 veto ON vs OFF under freeze OFF. It arrived with
`_FROZEN_FAMILY_SIZE` still at 36, so trunk's contract gate was red from `89976eb`
until this note (reported by a peer session, igw-224). VERDICT: a fixed goal is
INTENDED -- no retrofit, pin 36 -> 37.

Same four questions. (1) Like 1107 it builds the curriculum once per seed and evaluates
every arm from clones that carry `goal_state` across explicitly
(`agent.goal_state.load_state_dict(trained.goal_state.state_dict())`), so all arms of a
seed enter measurement with a bit-identical frozen z_goal. (2) It sets neither
`goal_weight` nor `residue.benefit_terrain_live_producer`. (3) It calls neither
`update_z_goal` nor `_set_goal_pipeline_frozen`; like 1107 it feeds only the native
`goal_state._last_drive_level` scalar each eval step, which is not a z_goal write.
(4) Its _make_config inherits 1107's ON-arm `use_external_task_drive=True`, so the
frozen goal sits on that drive's path in every arm alike; the manipulated variables are
the freeze gate, the veto producer and the scripted walk, none of which is the goal, so
the identical-across-arms goal cannot produce a between-arm difference. It records the
goal stream (`_ZG.observe`). Nothing in the script was touched.
"""
import ast
import sys
from pathlib import Path

import torch

from ree_core.agent import REEAgent
from ree_core.utils.config import REEConfig

REPO_ROOT = Path(__file__).resolve().parents[2]  # ree-v3/
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

import validate_experiments as V  # noqa: E402

EXPERIMENTS_DIR = REPO_ROOT / "experiments"
SCAFFOLD = EXPERIMENTS_DIR / "scaffolded_sd054_onboarding.py"


def _make_agent(**overrides):
    kwargs = dict(
        body_obs_dim=12,
        world_obs_dim=54,
        action_dim=4,
        self_dim=16,
        world_dim=16,
        z_goal_enabled=True,
    )
    kwargs.update(overrides)
    return REEAgent(REEConfig.from_dims(**kwargs))


# ---- (1) the behavioural fact the whole triage rests on ----------------------------
# The sibling AST assertion in test_dead_z_goal_stream_lint.py pins that
# `REEAgent.reset()` contains no `goal_state.reset()` CALL. That is a source-shape
# check; this is the RUNTIME consequence, which is what the family actually depends on.

def test_agent_reset_leaves_a_seeded_z_goal_bit_identical():
    agent = _make_agent()
    assert agent.goal_state is not None

    z_world = torch.ones(1, agent.config.latent.world_dim)
    agent.goal_state.update(
        z_world_current=z_world, benefit_exposure=1.0, drive_level=1.0
    )
    seeded = agent.goal_state.z_goal.clone()
    assert agent.goal_state.is_active(), "seed did not fire -- test premise broken"

    for _ in range(5):
        agent.reset()

    assert torch.equal(agent.goal_state.z_goal, seeded), (
        "REEAgent.reset() now changes z_goal. The scaffold-warmed family's measurement "
        "phase depends on the goal surviving the per-episode reset unchanged; if this "
        "flipped, the family is no longer FROZEN and this whole contract file plus "
        "_uses_a_z_goal_driving_helper's discharge need re-reading.")
    assert agent.goal_state.is_active()


def test_z_goal_does_not_decay_without_update_z_goal():
    """Decay lives inside GoalState.update -- the second half of the freeze."""
    agent = _make_agent()
    z_world = torch.ones(1, agent.config.latent.world_dim)
    agent.goal_state.update(
        z_world_current=z_world, benefit_exposure=1.0, drive_level=1.0
    )
    seeded = agent.goal_state.z_goal.clone()

    # A measurement loop calls reset() per episode and never calls update_z_goal.
    for _ in range(50):
        agent.reset()
    assert torch.equal(agent.goal_state.z_goal, seeded)

    # ...whereas a single decay-only update DOES move it, which is the counterfactual
    # the family's measurement phase never applies.
    agent.goal_state.update(
        z_world_current=z_world, benefit_exposure=0.0, drive_level=0.0
    )
    assert not torch.equal(agent.goal_state.z_goal, seeded)


# ---- (2) the frozen goal is NOT inert ---------------------------------------------

def test_frozen_goal_still_drives_e3_for_the_family_config():
    """`goal_weight` resolves LIVE for this family, so a stale goal biases selection.

    None of the 31 sets `goal_weight`, and `E3Config.goal_weight`'s dataclass default is
    0.0 -- which reads as "the goal term is off, so who cares if z_goal is stale". That
    reading is wrong: `REEConfig.from_dims` carries its OWN default of 1.0 and assigns
    `config.e3.goal_weight`, so the E3 goal term (gated on `goal_weight > 0` AND
    `goal_state.is_active()`) fires on every E3 tick of the measurement phase.
    """
    agent = _make_agent()
    assert agent.e3.config.goal_weight > 0.0
    assert agent.config.goal.e1_goal_conditioned is True


# ---- (3) the scaffold's own freeze primitive is a PAIR ------------------------------

def _scaffold_tree():
    return ast.parse(SCAFFOLD.read_text(encoding="utf-8"))


def test_goal_pipeline_freeze_helper_silences_the_consumers():
    """`_set_goal_pipeline_frozen(frozen=True)` must also short-circuit MECH-295/307.

    This pairing is what stops a held goal from driving the MECH-295 liking bridge and
    the MECH-307 conjunction during the scaffold's own freeze windows. If a future edit
    drops this half, those stages lose even that much. It is NOT full goal isolation --
    see `test_goal_pipeline_freeze_does_not_touch_the_read_paths` immediately below,
    which pins the deliberate scope limit.
    """
    tree = _scaffold_tree()
    fn = next(
        (n for n in ast.walk(tree)
         if isinstance(n, ast.FunctionDef) and n.name == "_set_goal_pipeline_frozen"),
        None)
    assert fn is not None, "_set_goal_pipeline_frozen not found in the scheduler"

    written = {
        t.attr
        for n in ast.walk(fn) if isinstance(n, ast.Assign)
        for t in n.targets if isinstance(t, ast.Attribute)
    }
    assert "use_mech295_liking_bridge" in written
    assert "use_mech307_conjunction" in written


def _scaffold_module():
    """Import the scaffold module (not just its AST) for behavioural assertions."""
    import scaffolded_sd054_onboarding as S  # noqa: E402  (sys.path set above)
    return S


def test_goal_pipeline_freeze_does_not_touch_the_read_paths():
    """The freeze's DEFAULT scope limit, pinned as a deliberate decision.

    (2026-07-27 triage; AMENDED 2026-07-27 when the opt-in strict knob landed. The
    assertion used to be an AST equality on the helper's write set -- writes are
    EXACTLY the two MECH flags. That equality is now wrong by design: strict mode adds
    gated writes to the two read paths. What must still hold, and is what the triage
    actually cared about, is that the DEFAULT path -- `strict` unset, i.e. every one of
    the 78 landed scaffold importers -- leaves both read paths untouched. That is now
    asserted BEHAVIOURALLY, which is strictly stronger than the old source-shape check:
    an AST equality could be satisfied by a helper that reached the same flags through
    a helper call, whereas this fails unless the values are genuinely unchanged.)

    So: `_set_goal_pipeline_frozen(agent, frozen=True)` with no `strict=` argument does
    NOT zero `E3Config.goal_weight` and does NOT clear `GoalConfig.e1_goal_conditioned`.
    The E3 `goal_weight * goal_proximity` term and E1 goal-conditioning stay live inside
    every "frozen" stage once Stage-0 has seeded z_goal -- which is why
    `run_hazard_avoidance`'s docstring no longer claims survival is learned "without the
    goal pipeline".

    Widening the DEFAULT remains rejected: it changes E3 selection in three stages for
    all 78 scaffold importers and breaks comparability with every landed scaffold run,
    and no landed manifest's recorded conclusion rests on strict isolation. The strict
    form is opt-in per experiment (`scaffold_strict_goal_isolation`), never a default.
    """
    S = _scaffold_module()
    agent = _make_agent()

    # Premise: both read paths live before the freeze (the family-config fact).
    assert agent.e3.config.goal_weight > 0.0
    assert agent.config.goal.e1_goal_conditioned is True
    goal_weight_before = float(agent.e3.config.goal_weight)

    # The DEFAULT call -- exactly what every landed scaffold run executes.
    S._set_goal_pipeline_frozen(agent, frozen=True)

    assert agent.config.use_mech295_liking_bridge is False
    assert agent.config.use_mech307_conjunction is False
    assert agent.e3.config.goal_weight == goal_weight_before, (
        "the DEFAULT freeze path now zeroes e3.goal_weight. That is a behaviour change "
        "for all 78 scaffold importers and breaks comparability with every landed "
        "scaffold run -- the strict form must stay opt-in via "
        "scaffold_strict_goal_isolation. See this file's docstring.")
    assert agent.config.goal.e1_goal_conditioned is True, (
        "the DEFAULT freeze path now clears e1_goal_conditioned -- same objection as "
        "for goal_weight above.")
    assert not hasattr(agent, "_scaffold_strict_goal_isolation_saved"), (
        "the DEFAULT freeze path created strict-isolation save state; it must not "
        "enter strict mode at all.")

    # The write set is still confined to a known allowlist, so an unrelated new
    # mutation cannot ride in unremarked. (Subset, not equality: the strict-only
    # writes are legitimate members.)
    tree = _scaffold_tree()
    fn = next(
        (n for n in ast.walk(tree)
         if isinstance(n, ast.FunctionDef) and n.name == "_set_goal_pipeline_frozen"),
        None)
    assert fn is not None
    written = {
        t.attr
        for n in ast.walk(fn) if isinstance(n, ast.Assign)
        for t in n.targets if isinstance(t, ast.Attribute)
    }
    assert written == {"use_mech295_liking_bridge", "use_mech307_conjunction"}, (
        f"_set_goal_pipeline_frozen's own write set changed to {sorted(written)}. The "
        "strict-mode read-path writes live in _enter_strict_goal_isolation / "
        "_exit_strict_goal_isolation, NOT inline here -- keeping them out of this "
        "function is what makes the default path auditable at a glance.")


def test_default_freeze_path_is_equivalent_to_the_pre_knob_helper():
    """Bit-identity of the DEFAULT path, proven by equivalence rather than argued.

    The pre-knob helper body was exactly two assignments (below, verbatim). This runs
    the new helper on one agent and that replica on a seed-identical twin, then compares
    everything a curriculum could possibly read downstream: every goal-relevant config
    field, the full parameter state_dict bitwise, and the torch / numpy / stdlib-random
    RNG states. If the added code consumed a single RNG draw or touched one byte of
    state, the streams would diverge from the next tick onward and this fails.

    (Why not an end-to-end curriculum A/B instead: the 460c dry-run curriculum is NOT
    reproducible across processes -- two byte-identical checkouts diverge at Stage-0 even
    with torch, numpy AND stdlib random seeded -- so a run-vs-run diff cannot resolve a
    no-op change. Measured on ree-cloud-2, 2026-07-27.)
    """
    import random as _random

    import numpy as _np

    S = _scaffold_module()

    def _pre_knob_replica(agent, frozen):
        if frozen:
            agent.config.use_mech295_liking_bridge = False
            agent.config.use_mech307_conjunction = False
        else:
            agent.config.use_mech295_liking_bridge = True
            agent.config.use_mech307_conjunction = True

    def _fingerprint(agent):
        return {
            "goal_weight": float(agent.e3.config.goal_weight),
            "e1_goal_conditioned": bool(agent.config.goal.e1_goal_conditioned),
            "mech295": bool(agent.config.use_mech295_liking_bridge),
            "mech307": bool(agent.config.use_mech307_conjunction),
            "params": {
                k: v.detach().cpu().numpy().tobytes()
                for k, v in sorted(agent.state_dict().items())
                if hasattr(v, "detach")
            },
            "torch_rng": torch.get_rng_state().numpy().tobytes(),
            "np_rng": repr(_np.random.get_state()),
            "py_rng": repr(_random.getstate()),
        }

    prints = []
    for apply_freeze in (S._set_goal_pipeline_frozen, _pre_knob_replica):
        torch.manual_seed(1234)
        _np.random.seed(1234)
        _random.seed(1234)
        agent = _make_agent()
        # Both freeze and unfreeze, in the order a curriculum uses them.
        apply_freeze(agent, frozen=True)
        apply_freeze(agent, frozen=False)
        apply_freeze(agent, frozen=True)
        prints.append(_fingerprint(agent))

    new_fp, old_fp = prints
    assert new_fp["params"] == old_fp["params"], "a parameter tensor changed"
    for key in ("goal_weight", "e1_goal_conditioned", "mech295", "mech307",
                "torch_rng", "np_rng", "py_rng"):
        assert new_fp[key] == old_fp[key], (
            f"the DEFAULT freeze path diverged from the pre-knob helper on {key!r}: "
            f"{new_fp[key]!r} vs {old_fp[key]!r}. The knob must be bit-identical when "
            "scaffold_strict_goal_isolation is unset.")


# ---- (3b) the OPT-IN strict form (2026-07-27) ---------------------------------------
# The knob the triage chipped: a future experiment that genuinely needs a goal-free
# Stage-H can now get one, without moving the default for anybody else.

def test_strict_goal_isolation_defaults_off():
    S = _scaffold_module()
    cfg = S.ScaffoldedSD054OnboardingConfig()
    assert cfg.scaffold_strict_goal_isolation is False, (
        "scaffold_strict_goal_isolation must default False. Flipping this default "
        "silently changes E3 selection in three stages for all 78 scaffold importers.")


def test_strict_goal_isolation_silences_both_read_paths():
    """strict=True must silence BOTH read paths, not just the E3 one."""
    S = _scaffold_module()
    agent = _make_agent()
    assert agent.e3.config.goal_weight > 0.0
    assert agent.config.goal.e1_goal_conditioned is True

    S._set_goal_pipeline_frozen(agent, frozen=True, strict=True)

    # E3: the gate is `goal_weight > 0.0`, so zero SKIPS the term rather than
    # scaling it -- compute_goal_score is not called at all.
    assert agent.e3.config.goal_weight == 0.0
    # E1: sense() then passes z_goal=None, the same path E1 takes when the goal
    # is inactive.
    assert agent.config.goal.e1_goal_conditioned is False
    # The write paths are still frozen -- strict is additive, not a replacement.
    assert agent.config.use_mech295_liking_bridge is False
    assert agent.config.use_mech307_conjunction is False


def test_strict_goal_isolation_restores_the_saved_prior_values():
    """Unfreeze restores what was there, NOT a hardcoded 1.0/True.

    An experiment may set a non-default goal_weight; restoring 1.0 would silently
    rewrite its config mid-curriculum.
    """
    S = _scaffold_module()
    agent = _make_agent()
    agent.e3.config.goal_weight = 0.37  # deliberately non-default
    agent.config.goal.e1_goal_conditioned = True

    S._set_goal_pipeline_frozen(agent, frozen=True, strict=True)
    assert agent.e3.config.goal_weight == 0.0

    S._set_goal_pipeline_frozen(agent, frozen=False, strict=True)
    assert agent.e3.config.goal_weight == 0.37, (
        "unfreeze did not restore the SAVED goal_weight -- a non-default value set by "
        "the experiment was overwritten.")
    assert agent.config.goal.e1_goal_conditioned is True
    assert not hasattr(agent, "_scaffold_strict_goal_isolation_saved")


def test_strict_goal_isolation_is_idempotent_and_unfreeze_needs_no_strict_flag():
    """Double-freeze must not save the zeroed value over the real one, and an
    unfreeze that forgets strict=True must still restore (the saved state, not the
    caller's flag, drives the restore)."""
    S = _scaffold_module()
    agent = _make_agent()
    original = float(agent.e3.config.goal_weight)

    S._set_goal_pipeline_frozen(agent, frozen=True, strict=True)
    S._set_goal_pipeline_frozen(agent, frozen=True, strict=True)  # second freeze
    assert agent.e3.config.goal_weight == 0.0

    S._set_goal_pipeline_frozen(agent, frozen=False)  # note: no strict=
    assert agent.e3.config.goal_weight == original
    assert agent.config.goal.e1_goal_conditioned is True

    # And an unfreeze with no prior strict freeze is a no-op, not an exception.
    S._set_goal_pipeline_frozen(agent, frozen=False)
    assert agent.e3.config.goal_weight == original


def test_every_freeze_call_site_threads_the_strict_knob():
    """The knob is useless if a stage forgets to pass it -- pin all call sites.

    Stage-0b / P0 / Stage-H are the frozen stages that must silence the read paths;
    run_stage0_nursery / run_p1 unfreeze and must restore them. All five read the
    same cfg field, so no stage can end up half-isolated.
    """
    tree = _scaffold_tree()
    seen = {}
    for n in ast.walk(tree):
        if not isinstance(n, ast.FunctionDef):
            continue
        for c in ast.walk(n):
            if (isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
                    and c.func.id == "_set_goal_pipeline_frozen"):
                kws = {kw.arg: kw.value for kw in c.keywords}
                strict = kws.get("strict")
                seen[n.name] = (
                    ast.unparse(strict) if strict is not None else None)

    assert set(seen) == {
        "run_stage0_nursery", "run_stage0b_consolidation", "run_p0",
        "run_hazard_avoidance", "run_p1",
    }, f"freeze call sites moved: {sorted(seen)}"
    for name, expr in seen.items():
        assert expr == "self.cfg.scaffold_strict_goal_isolation", (
            f"{name} passes strict={expr!r}; every call site must thread the config "
            "knob, or a stage silently keeps the goal read paths live while its "
            "siblings silence them.")


def test_scaffold_hands_off_with_the_goal_consumers_unfrozen():
    """The half-inherited freeze, pinned as a fact rather than left to be rediscovered.

    `run_p1` is the last stage to set the freeze state and it UNFREEZES; `run_p2` does
    not touch it. So the measurement loop receives an agent whose z_goal is held still
    (nobody calls update_z_goal) while MECH-295/307 are live. If a future edit makes
    run_p2 re-freeze, or makes run_p1 leave it frozen, this triage's conclusion changes
    and the docstring above needs revisiting.
    """
    tree = _scaffold_tree()
    methods = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef):
            for c in ast.walk(n):
                if (isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
                        and c.func.id == "_set_goal_pipeline_frozen"):
                    for kw in c.keywords:
                        if kw.arg == "frozen":
                            methods.setdefault(n.name, []).append(
                                ast.literal_eval(kw.value))

    assert methods.get("run_stage0b_consolidation") == [True]
    assert methods.get("run_p0") == [True]
    assert methods.get("run_hazard_avoidance") == [True]
    assert methods.get("run_p1") == [False], (
        "run_p1 no longer unfreezes the goal pipeline -- re-read this file's docstring")
    assert "run_p2" not in methods, (
        "run_p2 now sets the freeze state; the hand-off condition to the measurement "
        "phase has changed")


# ---- (4) family membership ---------------------------------------------------------
# Re-derived 2026-07-27 rather than transcribed: a scaffold importer that enables a
# z_goal-dependent knob, never writes z_goal itself, and hand-rolls a measurement loop
# (its own `select_action` + env `step`, as opposed to delegating every step to the
# scheduler). The 22 other scaffold importers that enable z_goal do NOT hand-roll a loop
# -- the scheduler drives every step for them, so it keeps calling update_z_goal and
# they never freeze.
#
# 28 -> 30 (2026-08-12): V3-EXQ-464e and V3-EXQ-467e landed as lettered bug-fix
# iterations of already-family members 464d/467d (see the FAMILY GROWTH addendum in
# this file's module docstring for the confirmation that both inherit the same frozen,
# arm-symmetric goal state -- neither needed a retrofit).
#
# 30 -> 31 (2026-08-14): V3-EXQ-934, the GOV-FANOUT-1 leg-H1 affinity_input_cap sweep.
# A new NUMBER rather than a lettered iteration, but structurally the same member shape
# (train once per seed, evaluate every CAP x ARM cell from a goal_state-carrying
# _clone_for_arm copy). A fixed goal is INTENDED -- including for the external_task_drive
# path, which unlike the rest of the family reads the goal directly into 934's primary
# DV. Full derivation in the second FAMILY GROWTH addendum in this file's docstring.
#
# 31 -> 32 (2026-08-18): V3-EXQ-935, the MECH-266/SD-032a margin-normalised cap rule.
# Structurally 934's sibling and the same member shape (train once per seed, evaluate
# every calibration / ARM_NORM-sweep / ARM_ABS cell from a goal_state-carrying
# _clone_for_arm copy). A fixed goal is INTENDED. Full derivation in the third FAMILY
# GROWTH addendum in this file's docstring.
#
# 34 -> 35 (2026-09-24): V3-EXQ-1090, the MECH-449 endogenous safety-producer
# validation. NOT the clone-per-arm shape: its two arms (harm pathway trained / not)
# each train their OWN curriculum, so the frozen goal is not arm-symmetric. A fixed goal
# is INTENDED anyway, because no criterion is a between-arm contrast: C1-C3 are
# within-arm (C3 within-tick paired against an env ground truth) and C4 is an absolute
# bound on the control. Full derivation in the FAMILY GROWTH (2026-09-24) addendum.
#
# 35 -> 36 (2026-09-25): V3-EXQ-1107, the SD-032a trained-agent mode-reversal
# diagnostic (external_task_drive OFF vs ON on the 935a curriculum). The clone-per-arm
# shape: one curriculum per seed, both arms from goal_state-carrying _clone_for_arm
# copies. It feeds the native drive_level scalar each eval step but never writes z_goal.
# A fixed goal is INTENDED. Full derivation in the FAMILY GROWTH (2026-09-25) addendum.
#
# 36 -> 37 (2026-09-26): V3-EXQ-1109, the DCD2 freeze/veto earliest-edge diagnostic on
# the 1107 harness (freeze ON/OFF, veto ON/OFF under freeze OFF, scripted walk). Same
# clone-per-arm goal_state carry as 1107; never writes z_goal. A fixed goal is INTENDED.
# Full derivation in the FAMILY GROWTH (2026-09-26) addendum.
_FROZEN_FAMILY_SIZE = 37


def test_frozen_z_goal_family_size_is_pinned():
    fam = []
    for p in sorted(EXPERIMENTS_DIR.glob("v3_exq_*.py")):
        src = p.read_text(encoding="utf-8", errors="replace")
        if "ScaffoldedSD054OnboardingScheduler" not in src:
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        if not V._sets_knob_truthy(tree, V._DEAD_ZGOAL_TRIGGER_KNOBS):
            continue
        if V._writes_z_goal_directly(tree):
            continue
        calls = {n.func.attr for n in ast.walk(tree)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
        if {"select_action", "step"} <= calls:
            fam.append(p.name)

    assert len(fam) == _FROZEN_FAMILY_SIZE, (
        f"frozen-z_goal family size moved: {len(fam)} vs pinned {_FROZEN_FAMILY_SIZE}. "
        f"A NEW member means a new script inherited the half-freeze -- have it choose "
        f"and DOCUMENT its measurement-phase goal state (drive the goal, re-freeze the "
        f"pipeline via _set_goal_pipeline_frozen, or state that a fixed goal is "
        f"intended). Note the SD-024 side effect before adding update_z_goal. "
        f"Members: {fam}")
