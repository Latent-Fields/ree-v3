"""
V3-EXQ-932a -- z_goal / residue_wanting -> behaviour observational coupling,
RE-INSTRUMENTED after the V3-EXQ-932 coupling-read autopsy.

Claims: None (diagnostic / observational; promotes nothing, weights no governance)

EXPERIMENT_PURPOSE = "diagnostic"

LETTERED ITERATION, NOT A NEW NUMBER. Same scientific question as V3-EXQ-932
(does REE's internal goal/wanting signal PREDICT what the organism does next?),
broken instrument -- so an alphabetic suffix per the CLAUDE.md EXQ versioning
policy. Source:
`REE_assembly/evidence/planning/failure_autopsy_931-932-wanting-authority-cluster_2026-08-16.md`
(human gate CONFIRMED 2026-08-16T18:41:10Z, governance-applied 2026-08-16),
Section 8 "DRAFT routing -- V3-EXQ-932: queue-experiment (V3-EXQ-932a)".

--------------------------------------------------------------------------
WHAT STANDS FROM V3-EXQ-932 AND IS NOT RE-LITIGATED HERE
--------------------------------------------------------------------------
932's PASS gates MEASUREMENT VALIDITY only, by its own explicit
`combination_rule`, and that PASS is real: `residue_wanting` is genuinely live
for the first time in the 906/916 lineage (nonzero on 100% of 1013 pooled eval
steps, std 0.56), confirming the V3-EXQ-916a orphaned-writer fix. This script
does not re-test that and does not supersede 932's manifest.

Two of the autopsy's five repair items -- per-seed coupling breakdown (item 3)
and partial correlations against z_goal (item 4) -- were shown to be RECOVERABLE
FROM 932's COMMITTED EPISODE LOG with no re-run, and the autopsy says so
explicitly. They are therefore NOT the justification for this run's compute.
They are nonetheless IMPLEMENTED here, because they are the correct instrument
and cost nothing once the driver is running: the recomputation the autopsy did
by hand becomes a first-class, pre-registered manifest field.

This run's compute is justified by the three things the committed log CANNOT
answer, which are exactly the autopsy's items 1, 2 and 5:

  (A) getting z_goal live on more than ONE seed  (item 5);
  (B) whether `approach` mode can fire at all, or must be dropped (item 1);
  (C) variance-based rather than n-based admissibility on the rare-event DVs
      (item 2).

--------------------------------------------------------------------------
WHAT DID NOT STAND -- the four defects this instrument repairs
--------------------------------------------------------------------------
Of the four couplings 932's manifest flagged NON-TRIVIAL, autopsy re-analysis
of the committed episode log found ONE survives:

  SURVIVES  wanting -> moved(t+1): r=+0.373, partial r|z_goal = +0.432 (it
            STRENGTHENS under partialling), replicated in all 3 seeds
            (0.28/0.55/0.28), 103 positives.
  COLLAPSES wanting -> benefit_exposure: +0.151 -> +0.044 under partialling on
            z_goal; its Spearman rho (0.086) was already below the
            pre-registered floor.
  ARTIFACT  BOTH z_goal couplings (zgoal->benefitexp +0.180; wanting<->zgoal
            contemporaneous +0.653) are BETWEEN-SEED artifacts. z_goal is
            identically ZERO in seeds 0 and 1 -- all 97 active ticks are in
            seed 2 (active_frac 0.0087) -- so the within-seed r is UNDEFINED
            for two of three seeds and pooling manufactures the association.
  VACUOUS   BOTH *_to_approach_t1 couplings are STRUCTURALLY UNSETTABLE, not
            measured nulls: `approach` mode fires 0/1013 under a harm-saturated
            affect-precedence chain (shelter 618 / avoid 218 / freeze 176), and
            `_pearson_r`/`_spearman_r` return exactly 0.0 on zero-variance input
            BY DESIGN. Their `powered: true` flag certified n, NOT variance.

Two further DVs sit at rare-event base rates the n>=200 gate cannot see:
`harm_signal>0` is 6/1013 (0.59%) and `reef_exit` is 10/1013 (0.99%).

--------------------------------------------------------------------------
THE FOUR DESIGN CHANGES (autopsy items 1-5, mapped one-to-one)
--------------------------------------------------------------------------

(1) PER-SEED NON-DEGENERACY, replacing MAX-ACROSS-SEEDS  [autopsy item 3]
    932 computed `chan_max_std = {k: max(r["chan_std"][k] for r in seed_results)}`
    and gated on it. That is the SPECIFIC defect that let ONE seed certify a
    channel for ALL THREE: `chan_max_std_z_goal = 0.0780` was exactly seed 2's
    standard deviation while seeds 0 and 1 were structurally flat.
    Here the load-bearing predicate is `chan_min_std` -- the MINIMUM across
    seeds -- so a channel is non-degenerate only if it varies in EVERY seed.
    `chan_max_std` is still emitted, REPORTED-ONLY, for direct comparability
    with 932/916a's numbers. Both, plus the full per-seed vector
    (`chan_std_by_seed`), are in the manifest.

(2) `powered` CERTIFIES VARIANCE, NOT JUST n  [autopsy item 2]
    932's `criteria_non_degenerate` marked every coupling `_powered: true`
    purely on `n >= MIN_COUPLING_N`, so a zero-variance DV passed with an exact
    0.0 correlation that reads as a measured null. Here every coupling is put
    through `_coupling_admissibility()` and lands in exactly one status:

      settled                  -- n, DV variance and X variance all adequate;
                                  r/rho are REAL NUMBERS and interpretable.
      underpowered_n           -- fewer than MIN_COUPLING_N (200) pairs.
      unsettable_dv_degenerate -- the DV cannot move: a binary DV below
                                  MIN_DV_POSITIVES (20) positives or outside the
                                  [MIN_DV_BASE_RATE, 1-MIN_DV_BASE_RATE] band
                                  (0.05), or a continuous DV below
                                  CHANNEL_STD_FLOOR.
      unsettable_x_degenerate  -- the AFFECT CHANNEL cannot move (std below
                                  CHANNEL_STD_FLOOR) -- 932's z_goal case.

    A coupling that is not `settled` emits `r: null` / `rho: null` (JSON null),
    NOT 0.0. This is the load-bearing change: after it, an exact 0.0 can no
    longer be produced by a degenerate input at all, so "0.0" in this manifest
    always means "measured, and it is zero".

(3) PRE-REGISTERED BASE-RATE FLOORS FOR RARE-EVENT DVs  [autopsy item 2]
    MIN_DV_POSITIVES = 20 and MIN_DV_BASE_RATE = 0.05, declared below before any
    run. Against 932's realised rates these floors REFUSE `harm_signal>0`
    (6 positives, 0.59%) and `reef_exit` (10, 0.99%) as unsettable, which is the
    intended behaviour: 932 reported both as if measured.

    The PER-SEED estimates use lower absolute floors (MIN_COUPLING_N_PER_SEED = 50,
    MIN_DV_POSITIVES_PER_SEED = 10; the base-rate floor is unchanged, a rate being
    scale-free). This is not a loosening -- reusing the pooled floors per seed was
    tried and would have refused seed 2 of 932's OWN run (n = 186 pairs), making
    the within-seed estimator the autopsy specifically asked for unmeasurable.
    This script's extended smoke caught exactly that: a coupling settled POOLED
    with n_seeds_defined = 0/2 behind it.

(4) THE `approach` DV -- DROPPED FROM THE GATING SET, WITH THE EVIDENCE
    [autopsy item 1]
    The autopsy sanctions either arranging a regime where `approach` can fire,
    or dropping the DV explicitly and saying why. This script DROPS it, and
    makes the drop empirical rather than asserted, in three parts:

      (a) `mode == "approach"` is RETAINED verbatim (906c/916a comparability) but
          is REPORTED-ONLY and can never gate. Its admissibility status is
          expected `unsettable_dv_degenerate`, which is the honest label 932
          could not produce.
      (b) A NEW precedence-free variant `approach_raw` (`harm_signal > 0.01`,
          the same consummatory-approach event WITHOUT the freeze > assert >
          shelter > avoid precedence chain that masks it) is measured alongside,
          so the drop is justified by the DV's own base rate rather than by the
          mode classifier's precedence.
      (c) A direct probe of this env config (2026-08-19, seed 0, 66 steps,
          `harm_signal > 0.01` fired 1/66 = 1.5%) establishes that even the
          precedence-free variant sits below MIN_DV_BASE_RATE. So `approach` is
          not measurable in the reef regime at ANY precedence, and the drop is a
          property of the ENVIRONMENT, not of the mode classifier.

    The alternative -- re-tuning ENV_KWARGS until `approach` fires -- was
    REJECTED: it would change the substrate under study and break comparability
    with 932/916a/906c on EVERY other coupling, to rescue one DV that the
    autopsy already classifies as unsettable. Changing the environment is a
    larger design change than the repair calls for.

(5) THE z_goal REGIME -- a two-arm PRE-REGISTERED DOSE on ONE DOCUMENTED KNOB
    [autopsy item 5]
    The autopsy's residual open question is `complex (probe-gated) /
    puzzle (known rules)`: "under what conditions does z_goal form at all in an
    emergent, unforced regime?" It offers two routes -- adopt 931's forced
    formation, or declare the emergent question a separate spike. This script
    takes NEITHER, and answers the puzzle with the same compute, because a
    direct probe made the mechanism knowable rather than mysterious:

      MEASURED (2026-08-19, this env config, seed 0, 66 steps):
        benefit_exposure > 0 on 86% of steps, but its MAXIMUM is 0.0409 and its
        nonzero mean is 0.0072. `GoalState.update` fires only when
            effective_benefit = benefit_exposure
                                * goal.z_goal_seeding_gain      (default 1.0)
                                * (1 + goal.drive_weight * drive_trace)
          > goal.benefit_threshold                              (default 0.1)
        With gain 1.0 and drive_weight 2.0, the ceiling of effective_benefit is
        0.0409 * 1 * 3 = 0.123 -- reachable ONLY at near-maximal drive. Observed
        directly: 66 writer calls, 57 benefit-positive steps, final z_goal norm
        exactly 0.0.

      So z_goal formation in the reef regime is STRUCTURALLY NEAR-UNREACHABLE,
      not stochastic, and 932's "1 of 3 seeds" was a threshold-crossing accident.
      That converts the puzzle from "unknown rules" to a known, quantified gate.

    The repair is therefore a DOSE on the one knob designed for exactly this
    (MECH-187 `z_goal_seeding_gain`, "apply seeding gain before drive
    modulation"; gain=1.0 is identity and fully backward compatible). Two arms,
    identical in every other respect:

      ARM_G1_EMERGENT (gain 1.0)  -- V3-EXQ-932's operating point, byte-for-byte.
                                     The comparability arm. z_goal is EXPECTED to
                                     stay flat; that is a RESULT here, not a
                                     failure, and the per-DV gate in (2) reports
                                     it as `unsettable_x_degenerate` rather than
                                     as a near-null coupling.
      ARM_G4_SEEDED   (gain 4.0)  -- lifts the gate into range while PRESERVING
                                     its selectivity: at gain 4 a peak contact
                                     reaches 0.0409*4*3 = 0.49 (fires) while the
                                     nonzero-mean trickle reaches 0.0072*4*3 =
                                     0.086 (does not). So z_goal still forms on
                                     genuine contact, not on every tick. This is
                                     NOT 931's forced `update_z_goal(0.5, 0.9)`
                                     every-tick write, which was deliberately
                                     avoided: a force-seeded z_goal is a lagged
                                     z_world EMA and its coupling to behaviour
                                     would be an artifact of the forcing.

    4.0 is pre-registered below as `Z_GOAL_SEEDING_GAIN_SEEDED` from the measured
    ceiling above, before any run.

--------------------------------------------------------------------------
NO ARM MAY VACATE ANOTHER (GOV-FANOUT-1 / the V3-EXQ-785 lesson)
--------------------------------------------------------------------------
ARM_G1_EMERGENT is EXPECTED to have a flat z_goal. Under 932's whole-run
conjunction (`measurement_valid = wanting_nondegen AND zgoal_nondegen AND ...`)
that would fail the WHOLE RUN and vacate ARM_G4_SEEDED's valid measurement --
precisely the V3-EXQ-785 defect. So the z_goal non-degeneracy precondition is
`applies_to`-SCOPED to the seeded arm only (it is not a meaningful readiness
question for the arm whose entire purpose is to measure whether z_goal forms),
and the run's `non_degenerate` is `aggregate_arm_gates`' ANY-arm-green, never
ALL-arms-green. `experiments/_lib/precondition_gate.py` is used rather than a
hand-rolled gate, exactly as the skill requires.

--------------------------------------------------------------------------
DV-SYMMETRY INVARIANCE DECLARATION (required, per arm)
--------------------------------------------------------------------------
The manipulation is `goal.z_goal_seeding_gain` (1.0 -> 4.0). It enters
`GoalState.update` as a MULTIPLIER on `benefit_exposure` inside a STRICT
THRESHOLD COMPARISON (`effective_benefit > benefit_threshold`), and the DVs are
(i) per-step behaviour indicators (moved / benefit_exposure / reef_exit / mode)
and (ii) the z_goal channel's own per-step magnitude.

  ARM_G1_EMERGENT and ARM_G4_SEEDED, both arms: the manipulation is NOT a
  broadcast additive constant across candidates (it never enters candidate
  scoring at all), NOT a monotone rescaling of a rank-based DV (the DVs are
  threshold-crossing counts and magnitudes, not orderings), and NOT a
  permutation of interchangeable units. A positive multiplier inside a strict
  inequality changes the SET of firing steps -- which is the DV -- so no
  symmetry of any DV here annihilates it. The z_goal channel magnitude is
  likewise not invariant: gain changes which steps seed, hence the realised
  z_goal trajectory.

  One honest asymmetry, stated rather than hidden: the gain acts ONLY through
  that threshold. If NO step's `benefit_exposure * gain * (1 + 2*drive)` cleared
  0.1 even at gain 4.0, both arms would be bit-identical and the contrast would
  be vacuous. The pre-registered `z_goal_dose_separation` precondition below
  measures exactly that (the seeded arm's z_goal active fraction must exceed the
  emergent arm's by a floor), so a vacuous dose is DETECTED rather than reported
  as a null.

--------------------------------------------------------------------------
OBSERVATIONAL, NOT CAUSAL
--------------------------------------------------------------------------
A lagged correlation between an internal signal and a subsequent behaviour is
consistent with, but does not establish, that the signal DRIVES the behaviour
(common-cause confounds abound -- proximity to a resource raises both wanting and
approach). The causal-necessity test is the SEPARATE ablation V3-EXQ-931, which
this driver deliberately does not duplicate. Do not read any coupling here as
causal. The partial correlations added in (item 4) reduce ONE specific confound
(a shared z_goal driver); they do not make the design causal.

--------------------------------------------------------------------------
SUBSTRATE-PATH OVERLAP (queue-experiment Step 2.5c), settled by PROBE
--------------------------------------------------------------------------
Two OPEN `severity: corrupting` substrate_queue entries name paths this driver
imports. Both were checked by direct runtime probe (2026-08-19) rather than by
module-name matching, and NEITHER code path is exercised:

  mode-governance-engagement (`ree_core/agent.py`, `ree_core/utils/config.py`):
    the defect is in `SalienceCoordinator.tick()`'s affinity-input box clamp and
    the `_et_commit` boolean latch, governing the external_task /
    internal_planning mode axis. Probed: `config.use_salience_coordinator` is
    False and `agent.salience_coordinator` is None under this config, so the
    coordinator is never instantiated. This driver's `mode` DV is its OWN
    `_classify_mode` affect-precedence chain (freeze/assert/shelter/avoid/
    approach/explore), a different quantity entirely.

  contextmemory-write-path-addressing-degeneracy (`ree_core/predictors/e1_deep.py`):
    the defect is `ContextMemory.write()`'s hard-argmin addressing. Probed by
    monkeypatch-counting over a real 66-step run of this exact config:
    `ContextMemory.write()` was called 0 times (`read()` 132 times). The
    defective path is not on this driver's execution path.

Recorded here because a later reader must be able to re-check the claim; the
probe is cheap to repeat.

--------------------------------------------------------------------------
ETHICS PREFLIGHT (queue-experiment Step 2.6)
--------------------------------------------------------------------------
All involvement flags false; decision: allow. V3, no live self-model in E3, no
autobiographical memory, no social mind, no language, no human or clinical data.
The harm streams / residue / accumulators present are pre-ethical
instrumentation (SENT-0 boundary). Observational read-only instrument over an
existing config; the one manipulation is a goal-seeding gain.

--------------------------------------------------------------------------
OUTPUT
--------------------------------------------------------------------------
  evidence/experiments/v3_exq_932a_zgoal_wanting_coupling_reinstrument_<ts>_v3.json
  evidence/experiments/v3_exq_932a_zgoal_wanting_coupling_reinstrument/
    ..._<ts>_episode_log.json     (real runs only -- NOT written on --dry-run;
                                   932 wrote it unconditionally, which left a
                                   toy episode log under evidence/ after every
                                   smoke test)

Estimated runtime: ~110 min on cloud CPU (2 arms x 3 seeds x (50 warmup + 5 eval)
x 200 steps; 2x V3-EXQ-932's ~55 min for the same per-cell work).
"""

import sys
import math
import random
import statistics
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig
from ree_core.residue.field import (
    VALENCE_WANTING, VALENCE_LIKING, VALENCE_SURPRISE,
    VALENCE_POSITIVE_SURPRISE, VALENCE_NEGATIVE_SURPRISE,
)
from experiment_protocol import emit_outcome
from experiments.pack_writer import (  # noqa: E402
    write_flat_manifest, resolve_evidence_experiments_dir,
)
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator
from experiments._lib.arm_fingerprint import arm_cell
from experiments._lib.precondition_gate import (
    PreconditionSpec, assert_no_structurally_unsatisfiable_gate,
    evaluate_arm_gate, aggregate_arm_gates, arm_criteria_non_degenerate,
)


EXPERIMENT_TYPE    = "v3_exq_932a_zgoal_wanting_coupling_reinstrument"
EXPERIMENT_PURPOSE = "diagnostic"
CLAIM_IDS          = []
SUPERSEDES         = None   # 932's MEASUREMENT-VALIDITY PASS stands; see docstring.

# The readiness anchors use the lineage-standard non-degeneracy floor
# (std > CHANNEL_STD_FLOOR = 1e-4) -- the SAME definition 664/916/932 apply to
# every channel, not a hand-narrowed predicate. The one change is the QUANTIFIER
# (min-across-seeds instead of max-across-seeds), not the floor.
ANCHOR_REACHABILITY_EXEMPT = (
    "Readiness anchors use the lineage-standard std > CHANNEL_STD_FLOOR (1e-4) "
    "non-degeneracy floor -- the same predicate 664/916/932 apply to every "
    "channel. V3-EXQ-932's landed run measured residue_wanting std 0.568 in "
    "every one of its 3 seeds (0.468/0.288/0.568), so the min-across-seeds form "
    "is reachable by construction on the load-bearing channel."
)

# ===========================================================================
# PRE-REGISTERED ANALYSIS PARAMETERS -- all defined before any run.
# ===========================================================================

# --- unchanged from V3-EXQ-932, for direct comparability -------------------
COUPLING_FLOOR_R = 0.15    # |r|/|rho| a SETTLED coupling must exceed to be "non-trivial"
MIN_COUPLING_N   = 200     # min paired obs before a coupling is interpretable at all
LAG_APPROACH_H   = 1       # affect[t] -> approach at t+1
LAG_BENEFIT_H    = 3       # affect[t] -> resource contact within t+1..t+3
LAG_MOVED_H      = 1       # affect[t] -> locomotion at t+1
LAG_REEFEXIT_H   = 1       # affect[t] -> departure-from-refuge at t+1

# --- NEW: variance / base-rate admissibility (autopsy items 1 and 2) -------
# A binary DV must have at least MIN_DV_POSITIVES positives AND a base rate
# inside [MIN_DV_BASE_RATE, 1 - MIN_DV_BASE_RATE]. Against V3-EXQ-932's realised
# rates these REFUSE approach (0/1013), harm_signal>0 (6/1013 = 0.59%) and
# reef_exit (10/1013 = 0.99%), and ADMIT moved (103/1013 = 10.2%) and
# benefit_exposure>0 (220/1013 = 21.7%) -- i.e. they separate exactly the DVs
# the autopsy separated by hand.
MIN_DV_POSITIVES   = 20
MIN_DV_BASE_RATE   = 0.05
CHANNEL_STD_FLOOR  = 1e-4   # a channel (X, or a continuous DV) must vary by at least this

# The PER-SEED estimates are held to LOWER n / positives floors than the pooled
# figure, deliberately and pre-registered. Reusing the pooled floors per seed was
# tried first and is WRONG: V3-EXQ-932's own realised per-seed pair counts were
# 595 / 232 / 186, so a per-seed n >= 200 floor would have refused seed 2
# outright and the within-seed estimates the autopsy Section 2(c) actually
# computed and relied on (0.28 / 0.55 / 0.28) could never have been produced.
# Caught by this script's own extended smoke, which reported n_seeds_defined 0/2
# on a coupling whose POOLED figure was settled -- i.e. the repair for 932's
# between-seed artifact would itself have been unmeasurable.
#
# The per-seed estimates are secondary (they feed the Fisher-z within-seed pool
# and the no-within-seed-support flag, never the arm gate), and Fisher-z already
# weights each seed by (n - 3), so a small seed is down-weighted rather than
# treated as equal. The BASE-RATE floor is unchanged across levels because a rate
# is scale-free; only the absolute counts move.
MIN_COUPLING_N_PER_SEED  = 50
MIN_DV_POSITIVES_PER_SEED = 10

# --- NEW: the z_goal seeding dose (autopsy item 5) -------------------------
# See the docstring's item (5): benefit_exposure peaks at 0.0409 in this env and
# GoalState.update fires on benefit_exposure * gain * (1 + 2*drive) > 0.1.
Z_GOAL_SEEDING_GAIN_EMERGENT = 1.0   # V3-EXQ-932's operating point, unchanged
Z_GOAL_SEEDING_GAIN_SEEDED   = 4.0   # lifts peak contacts over the gate, not the trickle
# The seeded arm must actually separate from the emergent arm on z_goal
# occupancy, or the dose is vacuous (see the DV-symmetry declaration).
Z_GOAL_DOSE_SEPARATION_FLOOR = 0.05  # seeded active_frac minus emergent active_frac

ARM_EMERGENT = "g1_emergent"
ARM_SEEDED   = "g4_seeded"
ARM_GAINS: Dict[str, float] = {
    ARM_EMERGENT: Z_GOAL_SEEDING_GAIN_EMERGENT,
    ARM_SEEDED:   Z_GOAL_SEEDING_GAIN_SEEDED,
}
ARM_IDS: List[str] = [ARM_EMERGENT, ARM_SEEDED]

# `harm_signal > APPROACH_RAW_THRESH` is the precedence-free approach event --
# the SAME numeric test _classify_mode's approach branch applies, lifted out of
# the freeze > assert > shelter > avoid chain that masks it (docstring item 4b).
APPROACH_RAW_THRESH = 0.01

ENV_KWARGS = dict(
    size=12,
    num_hazards=4,
    num_resources=5,
    hazard_harm=0.05,
    env_drift_interval=5,
    env_drift_prob=0.1,
    proximity_harm_scale=0.1,
    proximity_benefit_scale=0.05,
    proximity_approach_threshold=0.2,
    hazard_field_decay=0.5,
    resource_respawn_on_consume=True,
    toroidal=False,
    # SD-011 second source: rolling harm-history window
    harm_history_len=10,
    # SD-054: reef enrichment substrate (ARM_1_reef_food config from EXQ-522)
    reef_enabled=True,
    n_reef_patches=3,
    reef_patch_radius=2,
    hazard_food_attraction=0.7,
    # MECH-353 feed: sparse external action blocks so z_block (frustration) rises
    scheduled_action_block_enabled=True,
    scheduled_action_block_interval=10,
    scheduled_action_block_prob=0.4,
    # 916a orphaned-writer fix: only with use_proxy_fields=True does
    # CausalGridWorldV2 populate info["benefit_exposure"].
    use_proxy_fields=True,
)

WARMUP_EPISODES   = 50
EVAL_EPISODES     = 5
STEPS_PER_EPISODE = 200
WORLD_DIM         = 32
SELF_DIM          = 32
HARM_DIM          = 32
HARM_A_DIM        = 16
HARM_HISTORY_LEN  = 10
PAG_MAX_FREEZE    = 8   # MECH-279 cap so the freeze gate never permanently locks

WF_BUF_MAX        = 2000
HARM_EVAL_BUF_MAX = 2000
BATCH_SIZE        = 32
LR_E1             = 1e-4
LR_E2_WF          = 3e-4
LR_E3_HARM        = 1e-3
LR_ENC_AUX        = 5e-4

HARM_MODE_THRESH    = 0.25
EXPLORE_ERR_THRESH  = 0.10
SHELTER_HARM_THRESH = 0.15   # z_harm_norm floor for shelter mode while in reef
ASSERT_THRESH       = 0.10   # z_block_assert floor for the 'assert' (frustration) mode

# Core channels whose non-degeneracy defines a valid showcase (664-inherited).
CORE_CHANNELS = ["z_harm_a", "z_harm_un", "drive"]
def _action_to_onehot(idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, idx] = 1.0
    return v


def _obs_harm(obs_dict) -> Optional[torch.Tensor]:
    return obs_dict.get("harm_obs")


def _obs_harm_a(obs_dict) -> Optional[torch.Tensor]:
    return obs_dict.get("harm_obs_a")


def _obs_harm_history(obs_dict) -> Optional[torch.Tensor]:
    return obs_dict.get("harm_history")


def _obs_accum(obs_dict) -> float:
    v = obs_dict.get("accumulated_harm")
    return float(v) if v is not None else 0.0


def _obs_resource_prox(obs_dict) -> float:
    rv = obs_dict.get("resource_field_view")
    if rv is None:
        return 0.0
    return float(rv.max().item()) if isinstance(rv, torch.Tensor) else float(np.max(rv))


def _norm(t: Optional[torch.Tensor]) -> Optional[float]:
    if t is None:
        return None
    return float(t.norm().item())


def _make_agent_and_env(
    seed: int, z_goal_seeding_gain: float
) -> Tuple[REEAgent, CausalGridWorldV2]:
    """V3-EXQ-932's agent/env construction, VERBATIM, plus exactly one
    pre-registered arm knob: `goal.z_goal_seeding_gain` (MECH-187). At the
    default 1.0 this is bit-identical to 932 -- gain is applied before drive
    modulation and 1.0 is the identity, so ARM_G1_EMERGENT reproduces 932's
    operating point exactly. See docstring item (5)."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    env = CausalGridWorldV2(seed=seed, **ENV_KWARGS)
    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
        harm_dim=HARM_DIM,
        alpha_world=0.9,
        alpha_self=0.3,
        reafference_action_dim=env.action_dim,
        # SD-010 / SD-011
        use_harm_stream=True,
        z_harm_dim=HARM_DIM,
        use_affective_harm_stream=True,
        z_harm_a_dim=HARM_A_DIM,
        harm_history_len=HARM_HISTORY_LEN,
        # SD-018: resource-prox supervision on z_world
        use_resource_proximity_head=True,
        resource_proximity_weight=0.5,
        # SD-012: homeostatic drive-modulated benefit + goal system
        benefit_eval_enabled=True,
        benefit_weight=1.0,
        z_goal_enabled=True,
        goal_weight=0.5,
        drive_weight=2.0,
        e1_goal_conditioned=True,
        # --- affective register (telemetry, inherited from 664) ---
        use_tonic_vigor=True,             # MECH-320
        use_blocked_agency=True,          # MECH-353
        use_pag_freeze_gate=True,         # MECH-279
        # --- relief / safety register (this experiment's addition) ---
        use_suffering_derivative_comparator=True,   # MECH-302
        use_conditioned_safety_store=True,           # MECH-304
        use_contextual_safety_terrain=True,          # MECH-303
        # --- residue_wanting orphaned-writer fix (916a vs 916, THIS script) ---
        # SD-014/MECH-203 serotonin master switch. Without this,
        # update_benefit_salience() below is a no-op (self.serotonin.enabled
        # gate) regardless of whether it is called -- see module docstring.
        tonic_5ht_enabled=True,
    )
    # Non-from_dims config tweaks (set directly on the dataclass / sub-configs).
    config.e3.commitment_threshold = 0.5           # MECH-090 realistic threshold
    config.heartbeat.beta_gate_bistable = True     # MECH-090 bistable latch
    config.harm_descending_mod_enabled = True      # SD-021
    config.descending_attenuation_factor = 0.5

    # SD-019a: unpleasantness channel (middle harm tier).
    config.latent.use_harm_un = True

    # MECH-279: cap freeze duration so the gate never permanently locks the fish.
    config.pag_max_freeze_duration = PAG_MAX_FREEZE

    # MECH-320 tonic vigor reads its true substrate value (no artificial floor).
    # In this reward-negative reef env the avg-reward-rate EWMA stays negative, so
    # vigor sits low -- a faithful "no surplus drive-to-act under sustained threat"
    # reading. The channel is emitted for the viz but is not load-bearing.

    # SD-037: orexin broadcast-override (arousal recruitment under drive + threat).
    config.use_broadcast_override = True

    # MECH-307 + MECH-205: split-surprise valence channels for excite / dread.
    config.surprise_gated_replay = True
    config.use_mech307_split_surprise = True

    # control-vector telemetry so the tonic-vigor v_t is inspectable each tick.
    config.use_control_vector_logging = True

    # MECH-302/303/304 knobs left at REEConfig defaults (suffering_window_length=5,
    # suffering_drop_threshold=0.10, safety_store_threshold=0.5,
    # contextual_safety_release_threshold=1.0, ...) -- this run's purpose is to
    # confirm the DEFAULT operating point is reachable/observable, not to tune it.

    # --- THE ONLY ARM KNOB (V3-EXQ-932a). MECH-187 seeding gain, applied in
    # GoalState.update BEFORE drive modulation:
    #   effective_benefit = benefit_exposure * z_goal_seeding_gain
    #                       * (1 + drive_weight * drive_trace)   > benefit_threshold
    # gain=1.0 is the identity (bit-identical to 932); see docstring item (5)
    # for the measured benefit_exposure ceiling that sets the pre-registered dose.
    config.goal.z_goal_seeding_gain = float(z_goal_seeding_gain)

    agent = REEAgent(config)
    return agent, env


# ---------------------------------------------------------------------------
# Affect readout
# ---------------------------------------------------------------------------

def _read_affect(
    agent: REEAgent, latent, obs_body,
    relief_event: bool = False, safety_cue_signal: float = 0.0,
) -> Dict:
    """Read the protoemotional register off the agent + latent after select_action.

    `relief_event` / `safety_cue_signal` are NOT read here -- they must be captured
    by the caller immediately after agent.sense() and before agent.select_action(),
    since select_action() consumes/clears both (see module docstring TIMING NOTE).
    They are threaded through as arguments purely so this function's return dict
    stays the single place step_rec is assembled from.
    """
    z_world = latent.z_world

    # Nociceptive cascade
    z_harm_s  = _norm(latent.z_harm)
    z_harm_un = _norm(getattr(latent, "z_harm_un", None))
    z_harm_a  = _norm(latent.z_harm_a)

    # Drive & wanting
    try:
        drive = float(REEAgent.compute_drive_level(obs_body))
    except Exception:
        drive = None
    z_goal = None
    if getattr(agent, "goal_state", None) is not None:
        try:
            z_goal = float(agent.goal_state.goal_norm())
        except Exception:
            z_goal = None

    # Tonic vigor (MECH-320) via control-vector telemetry
    vigor = None
    cv = getattr(agent, "_last_control_vector", None)
    if isinstance(cv, dict):
        shared = cv.get("shared", {})
        v = shared.get("tonic_vigor_v_t")
        if v is not None:
            vigor = float(v)

    # Orexin override (SD-037)
    override = None
    bo = getattr(agent, "broadcast_override", None)
    if bo is not None and getattr(bo, "override_signal", None) is not None:
        override = float(bo.override_signal)

    # PAG freeze (MECH-279)
    freeze = False
    pag_out = getattr(agent, "_pag_last_output", None)
    if pag_out is not None:
        freeze = bool(getattr(pag_out, "freeze_active", False))

    # Blocked-agency assert pole (MECH-353)
    z_block = 0.0
    ba = getattr(agent, "blocked_agency", None)
    if ba is not None and getattr(ba, "_last_output", None) is not None:
        z_block = float(getattr(ba._last_output, "z_block_assert", 0.0))
    elif getattr(latent, "z_block", None) is not None:
        z_block = float(latent.z_block.abs().mean().item())

    # MECH-307 anticipatory valence (excitement / dread) from residue, plus the
    # unsigned prediction-error magnitude (VALENCE_SURPRISE, index 3) both are split
    # from, and the residue-map wanting/liking channels (indices 0/1) -- all six
    # components are computed every step but were previously read no further than 4/5.
    # NOTE: `residue_wanting` (VALENCE_WANTING) is distinct from `z_goal` above --
    # z_goal is the frontal goal-attractor's own wanting signal (goal_state.goal_norm()),
    # not the hippocampal-map residue channel read here.
    excite, dread, surprise, residue_wanting, liking = None, None, None, None, None
    try:
        val = agent.residue_field.evaluate_valence(z_world)
        if val is not None and val.shape[-1] > VALENCE_NEGATIVE_SURPRISE:
            excite          = float(val[0, VALENCE_POSITIVE_SURPRISE].item())
            dread           = float(val[0, VALENCE_NEGATIVE_SURPRISE].item())
            surprise        = float(val[0, VALENCE_SURPRISE].item())
            residue_wanting = float(val[0, VALENCE_WANTING].item())
            liking          = float(val[0, VALENCE_LIKING].item())
    except Exception:
        excite, dread, surprise, residue_wanting, liking = None, None, None, None, None

    # MECH-303: contextual safety terrain read (stateless RBF query -- safe to
    # call at any point after sense(), unlike relief_event/safety_cue_signal above).
    safety_terrain_read = None
    try:
        rf = getattr(agent, "residue_field", None)
        if rf is not None and getattr(rf, "safety_terrain_enabled", False):
            safety_pred = rf.evaluate_safety(z_world.detach())
            safety_terrain_read = float(safety_pred.mean().item())
    except Exception:
        safety_terrain_read = None

    return {
        "z_harm_s":  z_harm_s,
        "z_harm_un": z_harm_un,
        "z_harm_a":  z_harm_a,
        "drive":     drive,
        "z_goal":    z_goal,
        "vigor":     vigor,
        "override":  override,
        "freeze":    freeze,
        "z_block":   z_block,
        "excite":    excite,
        "dread":     dread,
        "residue_wanting": residue_wanting,
        "liking":    liking,
        "surprise":  surprise,
        # MECH-302/303/304 (this experiment's addition)
        "relief_event":        1.0 if relief_event else 0.0,
        "safety_cue_signal":   float(safety_cue_signal),
        "safety_terrain_read": safety_terrain_read,
    }


def _classify_mode(
    z_harm_norm: float,
    world_change_norm: float,
    harm_signal: float,
    in_reef: bool,
    freeze: bool,
    z_block_assert: float,
) -> str:
    """Behavioural mode with affect precedence: freeze > assert > shelter > avoid > approach > explore > neutral."""
    if freeze:
        return "freeze"
    if z_block_assert is not None and z_block_assert > ASSERT_THRESH:
        return "assert"
    if in_reef and z_harm_norm > SHELTER_HARM_THRESH:
        return "shelter"
    if z_harm_norm > HARM_MODE_THRESH:
        return "avoid"
    if harm_signal > 0.01:
        return "approach"
    if world_change_norm > EXPLORE_ERR_THRESH:
        return "explore"
    return "neutral"


def _get_reef_cells(env: CausalGridWorldV2) -> List[List[int]]:
    raw: Set = getattr(env, "_reef_cells", set())
    return [[int(x), int(y)] for x, y in sorted(raw)]


# ===========================================================================
# V3-EXQ-932a COUPLING INSTRUMENT
#
# `_pearson_r` / `_lagged_pairs` / `_contemporaneous_pairs` keep V3-EXQ-906c's
# verbatim pairing and estimator semantics, so a SETTLED coupling here is
# directly comparable to 906c's and 932's numbers. Everything else in this
# section is new and exists to repair the four defects the V3-EXQ-932 autopsy
# found (see the module docstring):
#
#   * `_coupling_admissibility` -- variance and base-rate admissibility, so a
#     degenerate input can no longer masquerade as a measured null. THIS is
#     what makes an emitted 0.0 mean "measured zero" rather than "estimator
#     degraded gracefully".
#   * `_measure_coupling` -- returns r/rho as None (JSON null) unless SETTLED.
#   * per-seed estimates + a Fisher-z within-seed pooled estimate beside every
#     pooled figure, so a between-seed artifact is visible in the manifest.
#   * `_partial_r` -- partial correlations against the third channel.
# ===========================================================================

def _pearson_r(xs: Sequence[float], ys: Sequence[float]) -> Tuple[Optional[float], int]:
    """Pooled Pearson r. Returns (None, n) -- NOT (0.0, n) -- on n<2 or
    zero-variance input.

    This is the ONE semantic change to 906c's estimator, and it is the point of
    the repair: 906c/932 returned exactly 0.0 on degenerate input "by design",
    which is indistinguishable in the manifest from a genuine null. None cannot
    be misread. Callers must not coerce it back to 0.0.
    """
    n = len(xs)
    if n < 2:
        return None, n
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if float(np.std(x)) < 1e-12 or float(np.std(y)) < 1e-12:
        return None, n
    r = float(np.corrcoef(x, y)[0, 1])
    return (None if np.isnan(r) else r), n


def _avg_ranks(a: np.ndarray) -> np.ndarray:
    """Average ranks within tie groups (matches scipy's default). No scipy dep."""
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(1, len(a) + 1, dtype=float)
    sa = a[order]
    i = 0
    while i < len(sa):
        j = i
        while j + 1 < len(sa) and sa[j + 1] == sa[i]:
            j += 1
        if j > i:
            avg = (ranks[order[i]] + ranks[order[j]]) / 2.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg
        i = j + 1
    return ranks


def _spearman_r(xs: Sequence[float], ys: Sequence[float]) -> Tuple[Optional[float], int]:
    """Spearman rho = Pearson r on rank-transformed inputs. Same None-on-degenerate
    contract as `_pearson_r`."""
    n = len(xs)
    if n < 2:
        return None, n
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if float(np.std(x)) < 1e-12 or float(np.std(y)) < 1e-12:
        return None, n
    rho = float(np.corrcoef(_avg_ranks(x), _avg_ranks(y))[0, 1])
    return (None if np.isnan(rho) else rho), n


def _partial_r(xs: Sequence[float], ys: Sequence[float],
               zs: Sequence[float]) -> Optional[float]:
    """Partial correlation of x and y controlling for z:

        r_xy.z = (r_xy - r_xz * r_yz) / sqrt((1 - r_xz^2) * (1 - r_yz^2))

    None when any constituent correlation is undefined (a flat channel) or the
    denominator vanishes. This is autopsy item 4, which the 932 driver computed
    NOWHERE despite reporting wanting <-> z_goal r = 0.65 in the same manifest.
    """
    if not (len(xs) == len(ys) == len(zs)) or len(xs) < 4:
        return None
    r_xy, _ = _pearson_r(xs, ys)
    r_xz, _ = _pearson_r(xs, zs)
    r_yz, _ = _pearson_r(ys, zs)
    if r_xy is None or r_xz is None or r_yz is None:
        return None
    denom = math.sqrt(max(0.0, (1.0 - r_xz ** 2) * (1.0 - r_yz ** 2)))
    if denom < 1e-12:
        return None
    val = (r_xy - r_xz * r_yz) / denom
    return None if math.isnan(val) else float(max(-1.0, min(1.0, val)))


def _fisher_pool(rs: Sequence[Optional[float]],
                 ns: Sequence[int]) -> Tuple[Optional[float], int]:
    """WITHIN-SEED pooled estimate: Fisher-z average over the seeds where the
    coupling is DEFINED, weighted by (n - 3), back-transformed.

    This is the estimator the autopsy's Section 2(c) shows is the right one: for
    `wanting -> benefit_exposure` EVERY within-seed r (0.19-0.28) exceeded the
    pooled r (0.151), i.e. pooling across seeds ATTENUATED it, so the pooled
    figure was the wrong estimator rather than the effect being zero. Reported
    beside the pooled figure, never instead of it.
    """
    zs: List[float] = []
    ws: List[float] = []
    for r, n in zip(rs, ns):
        if r is None or n < 4:
            continue
        rc = max(-0.999999, min(0.999999, float(r)))
        zs.append(math.atanh(rc))
        ws.append(float(n - 3))
    if not zs:
        return None, 0
    z_bar = sum(z * w for z, w in zip(zs, ws)) / sum(ws)
    return float(math.tanh(z_bar)), len(zs)


def _coupling_admissibility(xs: Sequence[float], ys: Sequence[float],
                            y_is_binary: bool,
                            min_n: int = MIN_COUPLING_N,
                            min_pos: int = MIN_DV_POSITIVES,
                            level: str = "pooled") -> Dict[str, Any]:
    """Decide whether a coupling can be MEASURED at all, before looking at r.

    Replaces V3-EXQ-932's `coupling_n_ok = n >= MIN_COUPLING_N`, which certified
    SAMPLE SIZE and nothing else -- so both `*_to_approach_t1` couplings were
    marked `_powered: true` on n=998 while the approach indicator had ZERO
    variance (0 positives in 1013 steps) and the estimator was returning a
    structural 0.0. Power certifies n; it does not certify variance.

    Status is exactly one of:
      settled                  -- measurable; r/rho are real numbers.
      underpowered_n           -- fewer than MIN_COUPLING_N pairs.
      unsettable_x_degenerate  -- the AFFECT CHANNEL cannot move (932's z_goal).
      unsettable_dv_degenerate -- the BEHAVIOUR DV cannot move (932's approach,
                                  harm_signal>0 at 0.59%, reef_exit at 0.99%).

    x-degeneracy is tested BEFORE dv-degeneracy so a run whose affect channel is
    flat is labelled by its true cause rather than by a co-occurring rare DV.
    """
    n = len(xs)
    x_std = float(np.std(np.asarray(xs, dtype=float))) if n >= 2 else 0.0
    out: Dict[str, Any] = {
        "n": int(n),
        "x_std": x_std,
        "y_is_binary": bool(y_is_binary),
        # Which floors this verdict was reached under, so a reader is never
        # puzzled by a coupling that is 'settled' pooled and not per seed.
        "level": level,
        "min_n_applied": int(min_n),
        "min_dv_positives_applied": int(min_pos),
    }
    if y_is_binary:
        pos = int(sum(1 for v in ys if float(v) > 0.5))
        out["y_positives"] = pos
        out["y_base_rate"] = (float(pos) / n) if n else 0.0
        out["y_std"] = float(np.std(np.asarray(ys, dtype=float))) if n >= 2 else 0.0
        dv_ok = (
            pos >= min_pos
            and n - pos >= min_pos
            and MIN_DV_BASE_RATE <= out["y_base_rate"] <= (1.0 - MIN_DV_BASE_RATE)
        )
    else:
        out["y_positives"] = None
        out["y_base_rate"] = None
        out["y_std"] = float(np.std(np.asarray(ys, dtype=float))) if n >= 2 else 0.0
        dv_ok = out["y_std"] > CHANNEL_STD_FLOOR

    if n < min_n:
        status = "underpowered_n"
    elif x_std <= CHANNEL_STD_FLOOR:
        status = "unsettable_x_degenerate"
    elif not dv_ok:
        status = "unsettable_dv_degenerate"
    else:
        status = "settled"
    out["status"] = status
    out["settled"] = status == "settled"
    return out


def _measure_coupling(xs: Sequence[float], ys: Sequence[float],
                      y_is_binary: bool, per_seed: bool = False) -> Dict[str, Any]:
    """Admissibility first, estimate only if SETTLED. A non-settled coupling
    carries r=None / rho=None so no downstream reader can mistake a structural
    zero for a measured one.

    `per_seed=True` applies the lower per-seed floors (see their definition above
    for why they must differ from the pooled ones)."""
    adm = _coupling_admissibility(
        xs, ys, y_is_binary,
        min_n=MIN_COUPLING_N_PER_SEED if per_seed else MIN_COUPLING_N,
        min_pos=MIN_DV_POSITIVES_PER_SEED if per_seed else MIN_DV_POSITIVES,
        level="per_seed" if per_seed else "pooled",
    )
    if adm["settled"]:
        r, _ = _pearson_r(xs, ys)
        rho, _ = _spearman_r(xs, ys)
    else:
        r, rho = None, None
    adm["r"] = r
    adm["rho"] = rho
    adm["nontrivial"] = bool(
        adm["settled"]
        and max(abs(r) if r is not None else 0.0,
                abs(rho) if rho is not None else 0.0) >= COUPLING_FLOOR_R
    )
    return adm


def _lagged_pairs(all_episode_steps: Sequence[Sequence[Dict]],
                  x_key: str,
                  y_positive_fn: Callable[[Dict], bool],
                  horizon: int) -> Tuple[List[float], List[float]]:
    """Pool (x_t, y_t) pairs: x_t = steps[t][x_key]; y_t = 1.0 if y_positive_fn
    holds for any step in t+1..min(t+horizon, n-1). WITHIN-EPISODE ONLY (no
    cross-boundary lag). Ported verbatim from 906c/932."""
    xs: List[float] = []
    ys: List[float] = []
    for steps in all_episode_steps:
        n = len(steps)
        for t in range(n - 1):
            xv = steps[t].get(x_key)
            if xv is None:
                continue
            window_end = min(t + horizon, n - 1)
            hit = any(y_positive_fn(steps[k]) for k in range(t + 1, window_end + 1))
            xs.append(float(xv))
            ys.append(1.0 if hit else 0.0)
    return xs, ys


def _contemporaneous_pairs(all_episode_steps: Sequence[Sequence[Dict]],
                           x_key: str, y_key: str) -> Tuple[List[float], List[float]]:
    """Pool same-step (x_t, y_t) pairs (ported verbatim from 906c/932)."""
    xs: List[float] = []
    ys: List[float] = []
    for steps in all_episode_steps:
        for s in steps:
            xv, yv = s.get(x_key), s.get(y_key)
            if xv is None or yv is None:
                continue
            xs.append(float(xv))
            ys.append(float(yv))
    return xs, ys


def _aligned_triple(all_episode_steps: Sequence[Sequence[Dict]],
                    x_key: str, y_positive_fn: Callable[[Dict], bool],
                    horizon: int, z_key: str) -> Tuple[List[float], List[float], List[float]]:
    """`_lagged_pairs` plus the control channel z sampled at the SAME t, so the
    partial correlation is computed on exactly the pairs the bivariate r used."""
    xs: List[float] = []
    ys: List[float] = []
    zs: List[float] = []
    for steps in all_episode_steps:
        n = len(steps)
        for t in range(n - 1):
            xv = steps[t].get(x_key)
            zv = steps[t].get(z_key)
            if xv is None or zv is None:
                continue
            window_end = min(t + horizon, n - 1)
            hit = any(y_positive_fn(steps[k]) for k in range(t + 1, window_end + 1))
            xs.append(float(xv))
            ys.append(1.0 if hit else 0.0)
            zs.append(float(zv))
    return xs, ys, zs


# --- behaviour-readout predicates ------------------------------------------

def _is_approach(s: Dict) -> bool:
    """906c/932's mode-based approach. RETAINED for comparability, REPORTED-ONLY:
    it can never gate this run (docstring item 4a)."""
    return s.get("mode") == "approach"


def _is_approach_raw(s: Dict) -> bool:
    """V3-EXQ-932a precedence-free approach (docstring item 4b)."""
    return bool(s.get("approach_raw"))


def _has_benefit(s: Dict) -> bool:
    v = s.get("benefit_exposure")
    return v is not None and float(v) > 0.0


def _has_harm_signal_pos(s: Dict) -> bool:
    """906c's exact "benefit" definition (harm_signal>0), kept VERBATIM for
    comparability. 906c's own naming quirk; the clean resource-contact signal is
    `benefit_exposure`. At 6/1013 in 932 this now fails the base-rate floor,
    which is the intended repair."""
    v = s.get("harm_signal")
    return v is not None and float(v) > 0.0


def _has_moved(s: Dict) -> bool:
    return bool(s.get("moved"))


def _is_reef_exit(s: Dict) -> bool:
    return s.get("transition_type") == "reef_exit"


# --- the coupling registry --------------------------------------------------
# `gating` marks whether a coupling may contribute to the arm's
# "at least one coupling settled" readiness criterion. The two mode-based
# approach couplings are gating=False per docstring item (4a): they are retained
# for 906c comparability and must never make or break this run.
COUPLINGS: List[Dict[str, Any]] = [
    # --- z_goal, 906c-matched definitions (direct comparability) ---
    {"name": "zgoal_t_to_approach_t1", "x": "z_goal", "y_fn": _is_approach,
     "h": LAG_APPROACH_H, "binary": True, "gating": False, "control": "residue_wanting"},
    {"name": "zgoal_t_to_approachraw_t1", "x": "z_goal", "y_fn": _is_approach_raw,
     "h": LAG_APPROACH_H, "binary": True, "gating": False, "control": "residue_wanting"},
    {"name": "zgoal_t_to_benefit_t1t3", "x": "z_goal", "y_fn": _has_harm_signal_pos,
     "h": LAG_BENEFIT_H, "binary": True, "gating": True, "control": "residue_wanting"},
    # --- z_goal, real resource-contact definition (916a fix enables) ---
    {"name": "zgoal_t_to_benefitexp_t1t3", "x": "z_goal", "y_fn": _has_benefit,
     "h": LAG_BENEFIT_H, "binary": True, "gating": True, "control": "residue_wanting"},
    # --- residue_wanting -> behaviour ---
    {"name": "wanting_t_to_approach_t1", "x": "residue_wanting", "y_fn": _is_approach,
     "h": LAG_APPROACH_H, "binary": True, "gating": False, "control": "z_goal"},
    {"name": "wanting_t_to_approachraw_t1", "x": "residue_wanting", "y_fn": _is_approach_raw,
     "h": LAG_APPROACH_H, "binary": True, "gating": False, "control": "z_goal"},
    {"name": "wanting_t_to_benefitexp_t1t3", "x": "residue_wanting", "y_fn": _has_benefit,
     "h": LAG_BENEFIT_H, "binary": True, "gating": True, "control": "z_goal"},
    {"name": "wanting_t_to_moved_t1", "x": "residue_wanting", "y_fn": _has_moved,
     "h": LAG_MOVED_H, "binary": True, "gating": True, "control": "z_goal"},
    {"name": "wanting_t_to_reefexit_t1", "x": "residue_wanting", "y_fn": _is_reef_exit,
     "h": LAG_REEFEXIT_H, "binary": True, "gating": True, "control": "z_goal"},
]
COUPLING_NAMES: List[str] = [c["name"] for c in COUPLINGS]
GATING_COUPLING_NAMES: List[str] = [c["name"] for c in COUPLINGS if c["gating"]]
CONTEMPORANEOUS_NAME = "wanting_zgoal_contemporaneous"


def _compute_coupling_block(
    episodes_by_seed: Dict[int, List[List[Dict]]],
) -> Dict[str, Any]:
    """The full coupling payload for ONE arm.

    For every coupling: the pooled estimate (906c-comparable), the PER-SEED
    estimates, the Fisher-z within-seed pooled estimate, and the partial
    correlation against the third channel -- each with its own admissibility
    status, so a between-seed artifact (932's z_goal case: undefined in 2 of 3
    seeds, pooled +0.18) is legible directly from the manifest.
    """
    all_steps: List[List[Dict]] = [ep for eps in episodes_by_seed.values() for ep in eps]
    seeds_sorted = sorted(episodes_by_seed.keys())
    block: Dict[str, Any] = {}

    for spec in COUPLINGS:
        name, x_key, y_fn, h = spec["name"], spec["x"], spec["y_fn"], spec["h"]
        binary, z_key = spec["binary"], spec["control"]

        xs, ys = _lagged_pairs(all_steps, x_key, y_fn, h)
        pooled = _measure_coupling(xs, ys, binary)

        xs3, ys3, zs3 = _aligned_triple(all_steps, x_key, y_fn, h, z_key)
        pooled["partial_r_given_" + z_key] = (
            _partial_r(xs3, ys3, zs3) if pooled["settled"] else None
        )
        pooled["partial_control"] = z_key

        per_seed: Dict[str, Any] = {}
        rs: List[Optional[float]] = []
        ns: List[int] = []
        for s in seeds_sorted:
            sx, sy = _lagged_pairs(episodes_by_seed[s], x_key, y_fn, h)
            m = _measure_coupling(sx, sy, binary, per_seed=True)
            sx3, sy3, sz3 = _aligned_triple(episodes_by_seed[s], x_key, y_fn, h, z_key)
            m["partial_r_given_" + z_key] = _partial_r(sx3, sy3, sz3) if m["settled"] else None
            per_seed[str(s)] = m
            rs.append(m["r"])
            ns.append(m["n"])
        within_r, n_defined = _fisher_pool(rs, ns)
        pooled["per_seed"] = per_seed
        pooled["within_seed_pooled_r"] = within_r
        pooled["n_seeds_defined"] = int(n_defined)
        pooled["n_seeds_total"] = len(seeds_sorted)
        # A pooled figure with 0 or 1 within-seed estimates behind it has NO
        # within-seed replication -- exactly V3-EXQ-932's z_goal case (pooled
        # +0.18, undefined in 2 of 3 seeds). Named for what it asserts rather than
        # for the one instance of it: 0 defined seeds is the same warning as 1.
        pooled["pooled_without_within_seed_support"] = bool(
            pooled["settled"] and n_defined <= 1 and len(seeds_sorted) > 1
        )
        block[name] = pooled

    # --- contemporaneous: are the two "wanting" signals even related? ---
    cx, cy = _contemporaneous_pairs(all_steps, "residue_wanting", "z_goal")
    cm = _measure_coupling(cx, cy, y_is_binary=False)
    c_per_seed: Dict[str, Any] = {}
    crs: List[Optional[float]] = []
    cns: List[int] = []
    for s in seeds_sorted:
        sx, sy = _contemporaneous_pairs(episodes_by_seed[s], "residue_wanting", "z_goal")
        m = _measure_coupling(sx, sy, y_is_binary=False, per_seed=True)
        c_per_seed[str(s)] = m
        crs.append(m["r"])
        cns.append(m["n"])
    cw, cnd = _fisher_pool(crs, cns)
    cm["per_seed"] = c_per_seed
    cm["within_seed_pooled_r"] = cw
    cm["n_seeds_defined"] = int(cnd)
    cm["n_seeds_total"] = len(seeds_sorted)
    cm["pooled_without_within_seed_support"] = bool(
        cm["settled"] and cnd <= 1 and len(seeds_sorted) > 1
    )
    cm["partial_control"] = None
    block[CONTEMPORANEOUS_NAME] = cm
    return block
# ---------------------------------------------------------------------------
# Phase 0: Warmup training with dual-stream auxiliary losses
# ---------------------------------------------------------------------------

def _warmup_train(
    agent: REEAgent,
    env: CausalGridWorldV2,
    num_episodes: int,
    steps_per_episode: int,
    seed: int,
) -> Dict:
    device     = agent.device
    action_dim = env.action_dim

    e1_optimizer = optim.Adam(agent.e1.parameters(), lr=LR_E1)
    e2_wf_optimizer = optim.Adam(
        list(agent.e2.world_transition.parameters()) +
        list(agent.e2.world_action_encoder.parameters()),
        lr=LR_E2_WF,
    )
    harm_eval_optimizer = optim.Adam(
        agent.e3.harm_eval_head.parameters(), lr=LR_E3_HARM
    )

    aux_params: List[torch.nn.Parameter] = list(agent.latent_stack.parameters())
    aux_optimizer = optim.Adam(aux_params, lr=LR_ENC_AUX)

    wf_buf:        List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    harm_eval_buf: List[Tuple[torch.Tensor, torch.Tensor]]               = []
    reward_log:    List[float] = []

    agent.train()

    for ep in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        # residue_wanting orphaned-writer fix (916a vs 916): benefit_exposure
        # is populated into `info`, not `obs_dict` (env.reset() returns no
        # `info` at all) -- see module docstring. Default {} so the first
        # step's benefit_exposure read (before this episode's first env.step())
        # safely evaluates to 0.0, matching "no exposure recorded yet".
        info: Dict = {}

        z_world_prev: Optional[torch.Tensor] = None
        z_self_prev:  Optional[torch.Tensor] = None
        action_prev:  Optional[torch.Tensor] = None
        ep_reward = 0.0

        for _ in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            obs_h     = _obs_harm(obs_dict)
            obs_h_a   = _obs_harm_a(obs_dict)
            obs_h_h   = _obs_harm_history(obs_dict)
            prox_t    = _obs_resource_prox(obs_dict)
            accum_t   = _obs_accum(obs_dict)

            latent = agent.sense(
                obs_body, obs_world,
                obs_harm=obs_h, obs_harm_a=obs_h_a, obs_harm_history=obs_h_h,
            )
            z_world_curr = latent.z_world.detach()

            aux_terms: List[torch.Tensor] = []
            prox_target_t = torch.tensor([[prox_t]], device=device)
            prox_loss = agent.compute_resource_proximity_loss(prox_target_t, latent)
            if prox_loss is not None and prox_loss.requires_grad:
                aux_terms.append(prox_loss)
            accum_target_t = torch.tensor([[accum_t]], device=device)
            harm_accum_loss = agent.compute_harm_accum_loss(accum_target_t, latent)
            if harm_accum_loss is not None and harm_accum_loss.requires_grad:
                aux_terms.append(harm_accum_loss)
            if aux_terms:
                aux_loss = sum(aux_terms)
                aux_optimizer.zero_grad()
                aux_loss.backward(retain_graph=False)
                torch.nn.utils.clip_grad_norm_(aux_params, 1.0)
                aux_optimizer.step()

            if z_self_prev is not None and action_prev is not None:
                agent.record_transition(z_self_prev, action_prev, latent.z_self.detach())

            latent = agent.sense(
                obs_body, obs_world,
                obs_harm=obs_h, obs_harm_a=obs_h_a, obs_harm_history=obs_h_h,
            )

            ticks    = agent.clock.advance()
            e1_prior = (
                agent._e1_tick(latent) if ticks.get("e1_tick", False)
                else torch.zeros(1, WORLD_DIM, device=device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)

            drive_level      = REEAgent.compute_drive_level(obs_body)
            benefit_exposure = max(0.0, float(info.get("benefit_exposure", 0.0)))
            agent.update_z_goal(benefit_exposure=benefit_exposure, drive_level=drive_level)

            # residue_wanting orphaned-writer fix (916a vs 916) -- canonical
            # call site per _harness.py / v3_exq_263. update_benefit_salience
            # no-ops internally unless serotonin is enabled (tonic_5ht_enabled
            # above); update_schema_wanting no-ops unless schema_wanting_enabled
            # is set (left at its default False here, matching _harness.py's
            # own "default-off guard" idiom -- MECH-216 wired but inert).
            if benefit_exposure > 0:
                agent.update_benefit_salience(benefit_exposure)
            if bool(getattr(agent.config.e1, "schema_wanting_enabled", False)):
                agent.update_schema_wanting(drive_level=drive_level)

            action = agent.select_action(candidates, ticks, temperature=1.0)
            if action is None:
                action = _action_to_onehot(random.randint(0, action_dim - 1), action_dim, device)
                agent._last_action = action

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)
            ep_reward += float(harm_signal)

            # Populate residue (incl. MECH-307 split-surprise valence) so excite /
            # dread are non-degenerate at eval time. Also feeds MECH-302/303/304
            # accumulation state during warmup (residue update, safety terrain).
            agent.update_residue(float(harm_signal))

            if z_world_prev is not None and action_prev is not None:
                wf_buf.append((z_world_prev.cpu(), action_prev.cpu(), z_world_curr.cpu()))
                if len(wf_buf) > WF_BUF_MAX:
                    wf_buf = wf_buf[-WF_BUF_MAX:]

            harm_target = abs(float(harm_signal)) if float(harm_signal) < 0 else 0.0
            harm_eval_buf.append((z_world_curr.cpu(), torch.tensor([harm_target])))
            if len(harm_eval_buf) > HARM_EVAL_BUF_MAX:
                harm_eval_buf = harm_eval_buf[-HARM_EVAL_BUF_MAX:]

            if len(wf_buf) >= BATCH_SIZE:
                idxs  = torch.randperm(len(wf_buf))[:BATCH_SIZE].tolist()
                zw_b  = torch.cat([wf_buf[i][0] for i in idxs]).to(device)
                a_b   = torch.cat([wf_buf[i][1] for i in idxs]).to(device)
                zw1_b = torch.cat([wf_buf[i][2] for i in idxs]).to(device)
                wf_pred = agent.e2.world_forward(zw_b, a_b)
                wf_loss = F.mse_loss(wf_pred, zw1_b)
                if wf_loss.requires_grad:
                    e2_wf_optimizer.zero_grad()
                    wf_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(agent.e2.world_transition.parameters()) +
                        list(agent.e2.world_action_encoder.parameters()), 1.0,
                    )
                    e2_wf_optimizer.step()
                with torch.no_grad():
                    agent.e3.update_running_variance((wf_pred.detach() - zw1_b).detach())

            if len(harm_eval_buf) >= BATCH_SIZE:
                idxs = torch.randperm(len(harm_eval_buf))[:BATCH_SIZE].tolist()
                zw_b = torch.cat([harm_eval_buf[i][0] for i in idxs]).to(device)
                ht_b = torch.cat([harm_eval_buf[i][1] for i in idxs]).to(device)
                hp   = agent.e3.harm_eval(zw_b)
                he_loss = F.mse_loss(hp.squeeze(), ht_b.squeeze())
                if he_loss.requires_grad:
                    harm_eval_optimizer.zero_grad()
                    he_loss.backward()
                    harm_eval_optimizer.step()

            if len(agent._world_experience_buffer) >= 2:
                e1_loss = agent.compute_prediction_loss()
                if e1_loss.requires_grad:
                    e1_optimizer.zero_grad()
                    e1_loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.e1.parameters(), 1.0)
                    e1_optimizer.step()

            z_world_prev = z_world_curr
            z_self_prev  = latent.z_self.detach()
            action_prev  = action.detach()
            if done:
                break

        reward_log.append(ep_reward)

        if (ep + 1) % 10 == 0 or ep == num_episodes - 1:
            print(
                f"  [warmup] seed={seed} ep {ep+1}/{num_episodes}"
                f"  rv={agent.e3._running_variance:.4f}  ep_reward={ep_reward:.4f}",
                flush=True,
            )

    first10 = float(np.mean(reward_log[:10]))  if len(reward_log) >= 10 else float(np.mean(reward_log))
    last10  = float(np.mean(reward_log[-10:])) if len(reward_log) >= 10 else float(np.mean(reward_log))

    return {
        "final_running_variance": agent.e3._running_variance,
        "warmup_first10_reward":  first10,
        "warmup_last10_reward":   last10,
    }


# ---------------------------------------------------------------------------
# Phase 1: Evaluation with affect recording for the fishtank feed
# ---------------------------------------------------------------------------

def _eval_agent(
    agent: REEAgent,
    env: CausalGridWorldV2,
    num_episodes: int,
    steps_per_episode: int,
    seed: int,
) -> Dict:
    action_dim    = env.action_dim
    device        = agent.device
    episode_rewards: List[float] = []
    episode_harms:   List[float] = []
    n_cands_log:     List[int]   = []
    episodes_log:    List[Dict]  = []

    # Per-channel non-degeneracy accumulators (across all eval steps). Core
    # channels (664-inherited) are load-bearing for PASS; relief/safety channels
    # are reported here for the same non-degeneracy statistics but are NOT
    # load-bearing (see module docstring).
    chan_vals: Dict[str, List[float]] = {
        k: [] for k in ["z_harm_s", "z_harm_un", "z_harm_a", "drive", "z_goal",
                        "vigor", "override", "z_block", "excite", "dread",
                        "safety_cue_signal", "safety_terrain_read",
                        # residue_wanting orphaned-writer fix (916a vs 916) --
                        # reported here, NOT load-bearing (same convention as
                        # excite/dread), so a run of this script directly
                        # evidences the fix.
                        "residue_wanting"]
    }
    freeze_fires = 0
    block_steps  = 0
    relief_fires = 0

    agent.eval()

    for ep_idx in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        # residue_wanting orphaned-writer fix (916a vs 916) -- see the matching
        # comment in _warmup_train above.
        info: Dict = {}

        z_self_prev:  Optional[torch.Tensor] = None
        z_world_prev: Optional[torch.Tensor] = None
        action_prev:  Optional[torch.Tensor] = None
        ep_reward = 0.0
        ep_harm   = 0.0

        ep_steps: List[Dict] = []
        initial_hazards   = [list(h) for h in env.hazards]
        initial_resources = [list(r) for r in env.resources]
        current_hazards   = [list(h) for h in env.hazards]
        current_resources = [list(r) for r in env.resources]

        reef_cells     = _get_reef_cells(env)
        reef_cells_set = getattr(env, "_reef_cells", set())
        prev_in_reef   = False
        pos_prev: Optional[Tuple[int, int]] = None   # V3-EXQ-932: for per-step `moved`

        for step_idx in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            obs_h     = _obs_harm(obs_dict)
            obs_h_a   = _obs_harm_a(obs_dict)
            obs_h_h   = _obs_harm_history(obs_dict)
            with torch.no_grad():
                latent = agent.sense(
                    obs_body, obs_world,
                    obs_harm=obs_h, obs_harm_a=obs_h_a, obs_harm_history=obs_h_h,
                )
                # MECH-302/MECH-304: capture BEFORE select_action(). Both signals
                # are consumed/cleared inside select_action() (relief conditionally
                # on having fired, safety unconditionally every tick) -- reading
                # them after select_action(), like every other channel in
                # _read_affect, would always observe a cleared/zeroed value. See
                # module docstring TIMING NOTE and agent.py select_action()'s
                # MECH-302 / MECH-304 blocks.
                relief_event      = bool(agent._relief_completion_event)
                safety_cue_signal = float(agent._conditioned_safety_signal)
                if z_self_prev is not None and action_prev is not None:
                    agent.record_transition(z_self_prev, action_prev, latent.z_self.detach())
                ticks    = agent.clock.advance()
                e1_prior = (
                    agent._e1_tick(latent) if ticks.get("e1_tick", False)
                    else torch.zeros(1, WORLD_DIM, device=device)
                )
                candidates = agent.generate_trajectories(latent, e1_prior, ticks)

                drive_level      = REEAgent.compute_drive_level(obs_body)
                benefit_exposure = max(0.0, float(info.get("benefit_exposure", 0.0)))
                agent.update_z_goal(benefit_exposure=benefit_exposure, drive_level=drive_level)

                # residue_wanting orphaned-writer fix (916a vs 916) -- see the
                # matching comment in _warmup_train above.
                if benefit_exposure > 0:
                    agent.update_benefit_salience(benefit_exposure)
                if bool(getattr(agent.config.e1, "schema_wanting_enabled", False)):
                    agent.update_schema_wanting(drive_level=drive_level)

                if ticks.get("e3_tick", True):
                    n_cands_log.append(len(candidates))
                action = agent.select_action(candidates, ticks, temperature=1.0)
                if action is None:
                    action = _action_to_onehot(random.randint(0, action_dim - 1), action_dim, device)
                    agent._last_action = action

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)

            # Keep residue (incl. valence) populated during eval too.
            with torch.no_grad():
                agent.update_residue(float(harm_signal))

            if info.get("env_drift_occurred", False):
                current_hazards   = [list(h) for h in env.hazards]
                current_resources = [list(r) for r in env.resources]

            agent_pos = (int(env.agent_x), int(env.agent_y))
            in_reef   = agent_pos in reef_cells_set
            blocked   = bool(info.get("action_blocked_this_step", False))
            if blocked:
                block_steps += 1

            # V3-EXQ-932 coupling instrumentation (behaviour readouts). All read
            # from the POST-step `info` / position -- i.e. the outcome of the
            # action just taken at this step, which is the behaviour the lagged
            # couplings predict FROM the affect signals read at earlier steps.
            #   benefit_exposure_realized: genuine resource contact this step
            #     (real only because use_proxy_fields=True on 916a's env; 0.0
            #     structurally in the pre-916a lineage). The "resource-seeking /
            #     target-acquisition" behavioural readout.
            #   moved: locomotion (position changed since the previous step).
            benefit_exposure_realized = max(0.0, float(info.get("benefit_exposure", 0.0)))
            moved = bool(pos_prev is not None and list(agent_pos) != list(pos_prev))

            # Affect readout (after select_action -> module caches are current,
            # EXCEPT relief_event/safety_cue_signal which were captured above,
            # pre-select_action -- see TIMING NOTE).
            affect = _read_affect(
                agent, latent, obs_body,
                relief_event=relief_event, safety_cue_signal=safety_cue_signal,
            )
            if affect["freeze"]:
                freeze_fires += 1
            if relief_event:
                relief_fires += 1
            for k, lst in chan_vals.items():
                v = affect.get(k)
                if isinstance(v, (int, float)) and v is not None:
                    lst.append(float(v))

            z_harm_s = affect["z_harm_s"] if affect["z_harm_s"] is not None else 0.0
            z_beta_val = (
                float(latent.z_beta.mean().item()) if latent.z_beta is not None else 0.0
            )
            world_change_norm = (
                float((latent.z_world - z_world_prev).norm().item())
                if z_world_prev is not None else 0.0
            )

            mode = _classify_mode(
                z_harm_s, world_change_norm, float(harm_signal),
                in_reef, affect["freeze"], affect["z_block"],
            )

            # Transition label: action_blocked > reef edge > env transition_type.
            if blocked:
                step_transition = "action_blocked"
            elif in_reef and not prev_in_reef:
                step_transition = "reef_entry"
            elif not in_reef and prev_in_reef:
                step_transition = "reef_exit"
            else:
                step_transition = info.get("transition_type", "none")

            step_rec = {
                "t":                 step_idx,
                "pos":               list(agent_pos),
                "action":            int(action.argmax(dim=-1).item()),
                "harm_signal":       float(harm_signal),
                # legacy + cascade
                "z_harm_norm":       z_harm_s,
                "z_harm_s":          affect["z_harm_s"],
                "z_harm_un":         affect["z_harm_un"],
                "z_harm_a":          affect["z_harm_a"],
                "z_world_norm":      float(latent.z_world.norm().item()),
                "z_beta_val":        z_beta_val,
                "world_change_norm": world_change_norm,
                # affect register
                "drive":             affect["drive"],
                "z_goal":            affect["z_goal"],
                "vigor":             affect["vigor"],
                "override":          affect["override"],
                "z_block":           affect["z_block"],
                "freeze":            affect["freeze"],
                "excite":            affect["excite"],
                "dread":             affect["dread"],
                "surprise":          affect["surprise"],
                "residue_wanting":   affect["residue_wanting"],
                "liking":            affect["liking"],
                # relief / safety register (this experiment's addition)
                "relief_event":        affect["relief_event"],
                "safety_cue_signal":   affect["safety_cue_signal"],
                "safety_terrain_read": affect["safety_terrain_read"],
                # behaviour
                "mode":              mode,
                "transition_type":   step_transition,
                "health":            float(info.get("health", 1.0)),
                "energy":            float(info.get("energy", 1.0)),
                "harm_event":        float(harm_signal) < 0,
                "n_cands":           len(candidates),
                "hazards":           [list(h) for h in current_hazards],
                "resources":         [list(r) for r in current_resources],
                "in_reef":           in_reef,
                # V3-EXQ-932 behaviour readouts for the coupling analysis
                "benefit_exposure":  benefit_exposure_realized,
                "moved":             moved,
                # V3-EXQ-932a: the PRECEDENCE-FREE approach event. Same numeric
                # test _classify_mode's approach branch applies, lifted out of the
                # freeze > assert > shelter > avoid chain that masks it, so the
                # decision to DROP the approach DV rests on the event's own base
                # rate rather than on the mode classifier. See docstring item (4b).
                "approach_raw":      bool(float(harm_signal) > APPROACH_RAW_THRESH),
            }
            ep_steps.append(step_rec)

            prev_in_reef = in_reef
            pos_prev     = agent_pos
            z_self_prev  = latent.z_self.detach()
            z_world_prev = latent.z_world.detach()
            action_prev  = action.detach()

            ep_reward += float(harm_signal)
            if float(harm_signal) < 0:
                ep_harm += abs(float(harm_signal))
            if done:
                break

        episode_rewards.append(ep_reward)
        episode_harms.append(ep_harm)
        episodes_log.append({
            "ep":                ep_idx,
            "initial_hazards":   initial_hazards,
            "initial_resources": initial_resources,
            "reef_cells":        reef_cells,
            "steps":             ep_steps,
        })

        print(
            f"  [eval] seed={seed} ep {ep_idx+1}/{num_episodes}"
            f"  reward={ep_reward:.4f}  harm={ep_harm:.4f}  steps={len(ep_steps)}",
            flush=True,
        )

    # Per-channel std (non-degeneracy signal).
    chan_std = {k: (float(np.std(v)) if len(v) >= 2 else 0.0) for k, v in chan_vals.items()}
    chan_mean = {k: (float(np.mean(v)) if v else 0.0) for k, v in chan_vals.items()}

    return {
        "mean_reward":  float(np.mean(episode_rewards)),
        "mean_harm":    float(np.mean(episode_harms)),
        "mean_n_cands": float(np.mean(n_cands_log)) if n_cands_log else 0.0,
        "episodes":     episodes_log,
        "chan_std":     chan_std,
        "chan_mean":    chan_mean,
        "freeze_fires": freeze_fires,
        "block_steps":  block_steps,
        "relief_fires": relief_fires,
        "eval_steps":   int(sum(len(e["steps"]) for e in episodes_log)),
    }


# ===========================================================================
# Per-cell run (one arm x one seed)
# ===========================================================================

CHANNEL_KEYS: List[str] = [
    "z_harm_s", "z_harm_un", "z_harm_a", "drive", "z_goal", "vigor", "override",
    "z_block", "excite", "dread", "safety_cue_signal", "safety_terrain_read",
    "residue_wanting",
]

STD_FLOOR = CHANNEL_STD_FLOOR   # lineage alias (664/916/932 spell it STD_FLOOR)

# z_goal liveness accumulator (Experimental Recording Standard). Agents are built
# fresh per cell; observe() reads live counters off each agent right after it
# finishes stepping rather than keeping every cell's agent alive.
_ZG = ZGoalStreamAccumulator()
# One representative agent per ARM, kept so write_flat_manifest can record
# `enabled_default_off_flags` for BOTH operating points (the arms differ in
# goal.z_goal_seeding_gain, which is exactly a non-default knob worth recording).
_ARM_AGENTS: Dict[str, REEAgent] = {}


def _arm_config_slice(arm_id: str) -> Dict[str, Any]:
    """What the cell's build+collect path READS -- env kwargs, schedule, the
    substrate-operating dims, and this arm's one knob. Declared (not inferred) so
    the fingerprint is anchored on the real inputs."""
    return {
        "env_kwargs": dict(ENV_KWARGS),
        "warmup_episodes": WARMUP_EPISODES,
        "eval_episodes": EVAL_EPISODES,
        "steps_per_episode": STEPS_PER_EPISODE,
        "world_dim": WORLD_DIM, "self_dim": SELF_DIM,
        "harm_dim": HARM_DIM, "harm_a_dim": HARM_A_DIM,
        "harm_history_len": HARM_HISTORY_LEN,
        "pag_max_freeze": PAG_MAX_FREEZE,
        "batch_size": BATCH_SIZE,
        "lr": {"e1": LR_E1, "e2_wf": LR_E2_WF, "e3_harm": LR_E3_HARM, "enc_aux": LR_ENC_AUX},
        "arm_id": arm_id,
        "z_goal_seeding_gain": ARM_GAINS[arm_id],
    }


def _z_goal_active_frac(episodes: Sequence[Dict]) -> float:
    """Fraction of recorded eval steps on which z_goal is non-zero. This is the
    quantity 932's autopsy reports as `active_frac` (0.0087 there), and the one
    the pre-registered dose separation is measured on."""
    tot = 0
    act = 0
    for ep in episodes:
        for s in ep.get("steps", []):
            v = s.get("z_goal")
            if v is None:
                continue
            tot += 1
            if abs(float(v)) > 0.0:
                act += 1
    return (float(act) / tot) if tot else 0.0


def run_cell(arm_id: str, seed: int, dry_run: bool = False) -> Dict:
    warmup_eps = 3 if dry_run else WARMUP_EPISODES
    eval_eps   = 2 if dry_run else EVAL_EPISODES
    steps      = 30 if dry_run else STEPS_PER_EPISODE
    gain       = ARM_GAINS[arm_id]

    # Runner boundary line -- resets episodes_in_run (RE_SEED_CONDITION).
    print(f"\nSeed {seed} Condition {arm_id}", flush=True)
    print(
        f"[EXQ-932a] arm={arm_id} seed={seed}  z_goal_seeding_gain={gain}"
        f"  warmup={warmup_eps}  eval={eval_eps}  steps/ep={steps}  dry_run={dry_run}",
        flush=True,
    )

    with arm_cell(
        seed,
        config_slice=_arm_config_slice(arm_id),
        script_path=Path(__file__),
        config_slice_declared=True,
        # include_driver_script_in_hash left at the default True on purpose: BOTH
        # arms are instrument arms whose readouts are produced by THIS driver's
        # coupling code, which is itself the thing under repair. A cross-driver
        # reuse would silently reuse a cell measured by a DIFFERENT instrument, so
        # binding the driver into the hash is the correctness-preserving choice
        # here. The cells stay reuse-ELIGIBLE (mint-as-you-go) within this driver.
    ) as cell:
        agent, env = _make_agent_and_env(seed, gain)
        print(
            f"[EXQ-932a] arm={arm_id} seed={seed} -- world_obs_dim={env.world_obs_dim}"
            f"  body_obs_dim={env.body_obs_dim}",
            flush=True,
        )
        warmup = _warmup_train(agent, env, warmup_eps, steps, seed)
        ree    = _eval_agent(agent, env, eval_eps, steps, seed)
        _ZG.observe(agent)
        _ARM_AGENTS[arm_id] = agent

        row: Dict[str, Any] = {
            "arm_id":                arm_id,
            "z_goal_seeding_gain":   float(gain),
            "seed":                  seed,
            "warmup_first10_reward": warmup["warmup_first10_reward"],
            "warmup_last10_reward":  warmup["warmup_last10_reward"],
            "warmup_final_rv":       warmup["final_running_variance"],
            "eval_mean_reward":      ree["mean_reward"],
            "eval_mean_harm":        ree["mean_harm"],
            "eval_mean_n_cands":     ree["mean_n_cands"],
            "chan_std":              ree["chan_std"],
            "chan_mean":             ree["chan_mean"],
            "freeze_fires":          ree["freeze_fires"],
            "block_steps":           ree["block_steps"],
            "relief_fires":          ree["relief_fires"],
            "eval_steps":            ree["eval_steps"],
            "z_goal_active_frac":    _z_goal_active_frac(ree["episodes"]),
            "episodes":              ree["episodes"],
        }
        cell.stamp(row)

    print(
        f"[EXQ-932a] arm={arm_id} seed={seed} channel std: "
        + "  ".join(f"{k}={ree['chan_std'].get(k, 0.0):.4f}"
                    for k in ["z_harm_a", "z_harm_un", "drive", "z_goal", "residue_wanting"]),
        flush=True,
    )
    print(
        f"[EXQ-932a] arm={arm_id} seed={seed} z_goal_active_frac="
        f"{row['z_goal_active_frac']:.4f}  eval_steps={ree['eval_steps']}"
        f"  block_steps={ree['block_steps']}",
        flush=True,
    )

    # Per-cell verdict for runner progress: one per arm x seed (= conditions x seeds).
    cell_core_ok = all(ree["chan_std"].get(k, 0.0) > CHANNEL_STD_FLOOR for k in CORE_CHANNELS)
    cell_pass = bool(cell_core_ok and ree["block_steps"] > 0)
    print(f"verdict: {'PASS' if cell_pass else 'FAIL'}", flush=True)
    return row


# ===========================================================================
# Pre-registered preconditions (regime-conditioned; NEVER ANDed whole-run)
# ===========================================================================

PRECONDITION_SPECS: List[PreconditionSpec] = [
    PreconditionSpec(
        name="residue_wanting_nondegenerate_every_seed",
        description=(
            "residue_wanting must vary in EVERY seed (MIN across seeds), not merely "
            "in the best one. V3-EXQ-932 gated on chan_max_std -- a MAX across seeds "
            "-- which is the specific defect that let one seed certify a channel for "
            "all three (autopsy Section 2b)."),
        control=(
            "V3-EXQ-932's landed run: residue_wanting std 0.468 / 0.288 / 0.568 in "
            "seeds 0/1/2 -- non-degenerate in every seed, so the min-across-seeds "
            "form is reachable by construction on this channel."),
        threshold=CHANNEL_STD_FLOOR,
        direction="lower",
        kind="readiness",
    ),
    PreconditionSpec(
        name="core_channels_nondegenerate_every_seed",
        description=(
            "the 664-inherited core channels (z_harm_a / z_harm_un / drive) must vary "
            "in every seed; measured is the WORST (channel, seed) cell."),
        control="V3-EXQ-932 landed run: all three core channels non-degenerate in all seeds.",
        threshold=CHANNEL_STD_FLOOR,
        direction="lower",
        kind="readiness",
    ),
    PreconditionSpec(
        name="z_goal_nondegenerate_every_seed",
        description=(
            "z_goal must vary in EVERY seed for its couplings to be interpretable "
            "within-seed rather than as a between-seed artifact."),
        control=(
            "V3-EXQ-932 landed run: z_goal std 0.000 / 0.000 / 0.078 -- nonzero in "
            "ONE seed only, and chan_max_std certified the channel from it."),
        threshold=CHANNEL_STD_FLOOR,
        direction="lower",
        kind="readiness",
        applies_to=lambda ctx: ctx["arm_id"] == ARM_SEEDED,
        applies_note=(
            "SCOPED OUT of ARM_G1_EMERGENT deliberately. That arm's entire purpose is "
            "to MEASURE whether z_goal forms at V3-EXQ-932's own operating point "
            "(gain 1.0); requiring it to form there would make the arm structurally "
            "un-passable and collapse the two-arm design back to one -- the "
            "V3-EXQ-785 defect. A flat z_goal in the emergent arm is this run's "
            "RESULT, and the per-coupling admissibility gate reports it as "
            "unsettable_x_degenerate rather than as a near-null coupling."),
    ),
    PreconditionSpec(
        name="z_goal_dose_separation",
        description=(
            "the seeded arm's z_goal active fraction must exceed the emergent arm's by "
            "Z_GOAL_DOSE_SEPARATION_FLOOR. Without this the two arms could be "
            "bit-identical (the gain acts only through a strict threshold) and the "
            "contrast vacuous -- see the module docstring's DV-symmetry declaration."),
        control=(
            "Direct probe of this env config (2026-08-19, seed 0): benefit_exposure "
            "max 0.0409, so at gain 4.0 a peak contact reaches 0.0409*4*3 = 0.49 "
            "against a benefit_threshold of 0.1, while V3-EXQ-932 measured "
            "active_frac 0.0087 at gain 1.0."),
        threshold=Z_GOAL_DOSE_SEPARATION_FLOOR,
        direction="lower",
        kind="readiness",
        applies_to=lambda ctx: ctx["arm_id"] == ARM_SEEDED,
        applies_note=(
            "A dose-separation check is a property of the SEEDED arm's manipulation; "
            "asserting it on the emergent (control) arm would be asserting that the "
            "control separates from itself."),
    ),
    PreconditionSpec(
        name="at_least_one_gating_coupling_settled",
        description=(
            "at least one GATING coupling must reach status 'settled' -- adequate n, "
            "adequate affect-channel variance AND adequate DV base rate. This replaces "
            "V3-EXQ-932's `couplings_adequately_powered`, which certified n alone and "
            "so marked a zero-variance DV `powered: true` (autopsy Section 2a). The "
            "two mode-based approach couplings are gating:false and cannot satisfy it."),
        control=(
            "V3-EXQ-932's realised rates: moved 103/1013 (10.2%) and benefit_exposure "
            "220/1013 (21.7%) both clear MIN_DV_POSITIVES=20 and MIN_DV_BASE_RATE=0.05, "
            "so at least two gating couplings are settleable by construction."),
        threshold=0.5,          # a FLOOR on a count: met when >= 1 coupling settled
        direction="lower",
        kind="readiness",
    ),
]


# ===========================================================================
# Aggregate + output
# ===========================================================================

def run(seeds=None, dry_run: bool = False) -> dict:
    if seeds is None:
        seeds = [0, 1, 2]
    seeds = list(seeds)

    print(
        f"[V3-EXQ-932a] z_goal / residue_wanting -> behaviour coupling, RE-INSTRUMENTED\n"
        f"  Seeds: {seeds}   Arms: {ARM_IDS} (z_goal_seeding_gain "
        f"{Z_GOAL_SEEDING_GAIN_EMERGENT} vs {Z_GOAL_SEEDING_GAIN_SEEDED})\n"
        f"  Warmup: {WARMUP_EPISODES} eps  Eval: {EVAL_EPISODES} eps"
        f"  Steps/ep: {STEPS_PER_EPISODE}\n"
        f"  Repairs (failure_autopsy_931-932-wanting-authority-cluster_2026-08-16):\n"
        f"    1 per-seed (MIN across seeds) non-degeneracy, not MAX-across-seeds\n"
        f"    2 admissibility certifies VARIANCE + BASE RATE, not just n\n"
        f"    3 base-rate floors: >= {MIN_DV_POSITIVES} positives and "
        f">= {MIN_DV_BASE_RATE:.0%}\n"
        f"    4 mode-based approach DVs REPORTED-ONLY (never gate) + precedence-free "
        f"approach_raw\n"
        f"    5 pre-registered z_goal seeding dose (MECH-187), not 931's forced write\n"
        f"  A non-settled coupling emits r=null, NEVER 0.0.\n"
        f"  OBSERVATIONAL not causal (causal = V3-EXQ-931 CEM wanting_weight ablation)",
        flush=True,
    )

    # Design-time refusal: no arm may carry a gate it provably cannot satisfy from
    # its pre-registered config, and not every arm may be vacuous (V3-EXQ-785).
    arm_contexts = [{"arm_id": a, "z_goal_seeding_gain": ARM_GAINS[a]} for a in ARM_IDS]
    assert_no_structurally_unsatisfiable_gate(PRECONDITION_SPECS, arm_contexts)

    rows: List[Dict[str, Any]] = []
    for arm_id in ARM_IDS:
        for seed in seeds:
            rows.append(run_cell(arm_id, seed, dry_run=dry_run))

    rows_by_arm: Dict[str, List[Dict[str, Any]]] = {
        a: [r for r in rows if r["arm_id"] == a] for a in ARM_IDS
    }

    # --- per-arm channel non-degeneracy: MIN across seeds is load-bearing ------
    per_arm: Dict[str, Any] = {}
    for arm_id in ARM_IDS:
        arm_rows = rows_by_arm[arm_id]
        chan_std_by_seed = {str(r["seed"]): r["chan_std"] for r in arm_rows}
        chan_min_std = {
            k: min(float(r["chan_std"].get(k, 0.0)) for r in arm_rows) for k in CHANNEL_KEYS
        }
        chan_max_std = {
            k: max(float(r["chan_std"].get(k, 0.0)) for r in arm_rows) for k in CHANNEL_KEYS
        }
        episodes_by_seed = {int(r["seed"]): r.get("episodes", []) for r in arm_rows}
        # {seed: [episode_steps, ...]} -- the coupling instrument pools per seed.
        steps_by_seed = {
            s: [ep["steps"] for ep in eps if ep.get("steps")]
            for s, eps in episodes_by_seed.items()
        }
        couplings = _compute_coupling_block(steps_by_seed)
        active_fracs = {str(r["seed"]): r["z_goal_active_frac"] for r in arm_rows}
        per_arm[arm_id] = {
            "arm_id": arm_id,
            "z_goal_seeding_gain": ARM_GAINS[arm_id],
            "seeds": [r["seed"] for r in arm_rows],
            "chan_std_by_seed": chan_std_by_seed,
            "chan_min_std": chan_min_std,
            "chan_max_std": chan_max_std,   # REPORTED ONLY -- 932/916a comparability
            "z_goal_active_frac_by_seed": active_fracs,
            "z_goal_active_frac_mean": (
                statistics.fmean(active_fracs.values()) if active_fracs else 0.0),
            "couplings": couplings,
            "total_eval_steps": int(sum(int(r["eval_steps"]) for r in arm_rows)),
            "total_block_steps": int(sum(int(r["block_steps"]) for r in arm_rows)),
            "total_freeze_fires": int(sum(int(r["freeze_fires"]) for r in arm_rows)),
        }

    dose_separation = (
        per_arm[ARM_SEEDED]["z_goal_active_frac_mean"]
        - per_arm[ARM_EMERGENT]["z_goal_active_frac_mean"]
    )

    # --- per-arm gate ---------------------------------------------------------
    arm_gates: List[Dict[str, Any]] = []
    for arm_id in ARM_IDS:
        info = per_arm[arm_id]
        n_settled_gating = sum(
            1 for c in GATING_COUPLING_NAMES if info["couplings"][c]["settled"]
        )
        info["n_gating_couplings_settled"] = int(n_settled_gating)
        measured = {
            "residue_wanting_nondegenerate_every_seed":
                float(info["chan_min_std"].get("residue_wanting", 0.0)),
            "core_channels_nondegenerate_every_seed":
                float(min(info["chan_min_std"].get(k, 0.0) for k in CORE_CHANNELS)),
            "z_goal_nondegenerate_every_seed":
                float(info["chan_min_std"].get("z_goal", 0.0)),
            "z_goal_dose_separation": float(dose_separation),
            "at_least_one_gating_coupling_settled": float(n_settled_gating),
        }
        arm_gates.append(evaluate_arm_gate(
            arm_id,
            {"arm_id": arm_id, "z_goal_seeding_gain": ARM_GAINS[arm_id]},
            PRECONDITION_SPECS,
            measured,
        ))

    agg = aggregate_arm_gates(arm_gates)
    measurement_valid = bool(agg["non_degenerate"])
    outcome = "PASS" if measurement_valid else "FAIL"

    # --- interpretation label -------------------------------------------------
    green = set(agg["green_arms"])
    scored_arms = [a for a in ARM_IDS if a in green]
    any_wanting_nontrivial = any(
        per_arm[a]["couplings"][c]["nontrivial"]
        for a in scored_arms for c in COUPLING_NAMES if c.startswith("wanting_t_")
    )
    if not measurement_valid:
        interp_label = "substrate_not_ready_requeue"
    elif any_wanting_nontrivial:
        interp_label = "wanting_behaviour_coupling_detected"
    else:
        interp_label = "wanting_behaviour_coupling_null"

    # --- flat metrics ---------------------------------------------------------
    metrics: Dict[str, Any] = {
        "n_seeds": float(len(seeds)),
        "n_arms": float(len(ARM_IDS)),
        "coupling_floor_r": float(COUPLING_FLOOR_R),
        "min_coupling_n": float(MIN_COUPLING_N),
        "min_dv_positives": float(MIN_DV_POSITIVES),
        "min_dv_base_rate": float(MIN_DV_BASE_RATE),
        "min_coupling_n_per_seed": float(MIN_COUPLING_N_PER_SEED),
        "min_dv_positives_per_seed": float(MIN_DV_POSITIVES_PER_SEED),
        "channel_std_floor": float(CHANNEL_STD_FLOOR),
        "z_goal_dose_separation": float(dose_separation),
        "z_goal_dose_separation_floor": float(Z_GOAL_DOSE_SEPARATION_FLOOR),
    }
    for arm_id in ARM_IDS:
        info = per_arm[arm_id]
        metrics[f"{arm_id}_z_goal_seeding_gain"] = float(ARM_GAINS[arm_id])
        metrics[f"{arm_id}_z_goal_active_frac_mean"] = float(info["z_goal_active_frac_mean"])
        metrics[f"{arm_id}_total_eval_steps"] = float(info["total_eval_steps"])
        metrics[f"{arm_id}_n_gating_couplings_settled"] = float(info["n_gating_couplings_settled"])
        for k in CHANNEL_KEYS:
            metrics[f"{arm_id}_chan_min_std_{k}"] = float(info["chan_min_std"][k])
            metrics[f"{arm_id}_chan_max_std_{k}"] = float(info["chan_max_std"][k])
        for c in COUPLING_NAMES + [CONTEMPORANEOUS_NAME]:
            m = info["couplings"][c]
            metrics[f"{arm_id}_coupling_{c}_r"] = m["r"]        # None when not settled
            metrics[f"{arm_id}_coupling_{c}_rho"] = m["rho"]
            metrics[f"{arm_id}_coupling_{c}_n"] = float(m["n"])
            metrics[f"{arm_id}_coupling_{c}_status"] = m["status"]
            metrics[f"{arm_id}_coupling_{c}_within_seed_pooled_r"] = m["within_seed_pooled_r"]
            metrics[f"{arm_id}_coupling_{c}_n_seeds_defined"] = float(m["n_seeds_defined"])
            metrics[f"{arm_id}_coupling_{c}_nontrivial"] = 1.0 if m["nontrivial"] else 0.0
            metrics[f"{arm_id}_coupling_{c}_pooled_without_within_seed_support"] = (
                1.0 if m["pooled_without_within_seed_support"] else 0.0)
            if m.get("y_base_rate") is not None:
                metrics[f"{arm_id}_coupling_{c}_y_base_rate"] = float(m["y_base_rate"])
                metrics[f"{arm_id}_coupling_{c}_y_positives"] = float(m["y_positives"])
            ctrl = m.get("partial_control")
            if ctrl:
                metrics[f"{arm_id}_coupling_{c}_partial_r_given_{ctrl}"] = m.get(
                    "partial_r_given_" + ctrl)

    # --- criteria (per-arm, keyed to the owning arm's gate) --------------------
    criteria_by_arm = {
        a: [f"{a}::measurement_valid", f"{a}::gating_coupling_settled"] for a in ARM_IDS
    }
    criteria_non_degenerate = arm_criteria_non_degenerate(criteria_by_arm, agg)
    criteria = [
        {"name": f"{a}::measurement_valid", "load_bearing": a in green,
         "passed": a in green, "arm": a}
        for a in ARM_IDS
    ] + [
        {"name": f"{a}::gating_coupling_settled",
         "load_bearing": a in green,
         "passed": bool(per_arm[a]["n_gating_couplings_settled"] > 0), "arm": a}
        for a in ARM_IDS
    ]

    interpretation: Dict[str, Any] = {
        "label": interp_label,
        "preconditions": agg["adjudication_preconditions"],
        "criteria_non_degenerate": criteria_non_degenerate,
        "criteria": criteria,
        "combination_rule": (
            "PASS (measurement_valid) = ANY arm's precondition gate is green -- "
            "aggregate_arm_gates' any-arm-green, NEVER all-arms-green. A red arm does "
            "NOT vacate a green one (failure_autopsy_V3-EXQ-785). Each arm's gate is "
            "the conjunction of the preconditions that APPLY to it; "
            "z_goal_nondegenerate_every_seed and z_goal_dose_separation apply to "
            f"'{ARM_SEEDED}' only and are scoped out of '{ARM_EMERGENT}' (see their "
            "applies_note). As in V3-EXQ-932 this gates MEASUREMENT VALIDITY, not "
            "coupling detection: whether any coupling clears the |r|/|rho| floor is "
            "REPORTED, never gated. UNLIKE V3-EXQ-932, a coupling is only reported at "
            "all once it is 'settled' (n + affect-channel variance + DV base rate); "
            "otherwise r/rho are null, so an emitted 0.0 now means a measured zero."),
        "per_arm_gate": agg["per_arm_gate"],
        "degeneracy_reason": agg["degeneracy_reason"],
        "coupling_summary": {
            "floor_r": float(COUPLING_FLOOR_R),
            "min_n": float(MIN_COUPLING_N),
            "min_dv_positives": int(MIN_DV_POSITIVES),
            "min_dv_base_rate": float(MIN_DV_BASE_RATE),
            "min_coupling_n_per_seed": int(MIN_COUPLING_N_PER_SEED),
            "min_dv_positives_per_seed": int(MIN_DV_POSITIVES_PER_SEED),
            "reference_906c_near_null_r": 0.07,
            "gating_couplings": GATING_COUPLING_NAMES,
            "reported_only_couplings": [
                c for c in COUPLING_NAMES if c not in GATING_COUPLING_NAMES],
            "scored_arms": scored_arms,
            "nontrivial_by_arm": {
                a: [c for c in COUPLING_NAMES + [CONTEMPORANEOUS_NAME]
                    if per_arm[a]["couplings"][c]["nontrivial"]]
                for a in ARM_IDS
            },
            "status_by_arm": {
                a: {c: per_arm[a]["couplings"][c]["status"]
                    for c in COUPLING_NAMES + [CONTEMPORANEOUS_NAME]}
                for a in ARM_IDS
            },
        },
        "approach_dv_disposition": (
            "DROPPED FROM THE GATING SET, with evidence. Both mode-based approach "
            "couplings are gating:false (retained verbatim for 906c/932 "
            "comparability, REPORTED-ONLY). The precedence-free variant "
            "`approach_raw` (harm_signal > "
            f"{APPROACH_RAW_THRESH}) is measured alongside so the drop rests on the "
            "event's own base rate rather than on _classify_mode's affect precedence. "
            "A direct probe of this env config (2026-08-19, seed 0, 66 steps) found "
            "harm_signal > 0.01 on 1/66 steps (1.5%), below MIN_DV_BASE_RATE, so "
            "`approach` is expected unsettable in the reef regime at ANY precedence. "
            "Re-tuning ENV_KWARGS to make it fire was REJECTED: it would change the "
            "substrate under study and break comparability on every other coupling."),
        "note": (
            "OBSERVATIONAL / DIAGNOSTIC -- promotes nothing, weights no governance "
            "(claim_ids=[], experiment_purpose=diagnostic). Lettered re-instrument of "
            "V3-EXQ-932 per failure_autopsy_931-932-wanting-authority-cluster_2026-08-16 "
            "Section 8. 932's MEASUREMENT-VALIDITY PASS stands and is NOT superseded; "
            "what is repaired is its reported, non-gating coupling narrative. A lagged "
            "correlation is consistent with but does NOT establish that the signal "
            "DRIVES behaviour; the partial correlations reduce ONE confound (a shared "
            "z_goal driver), they do not make the design causal. The causal-necessity "
            "test is the separate ablation V3-EXQ-931."),
    }

    summary_markdown = _summary_markdown(
        outcome, interp_label, per_arm, agg, dose_separation, seeds)

    config_snapshot = {
        "env_kwargs": ENV_KWARGS,
        "warmup_episodes": WARMUP_EPISODES,
        "eval_episodes": EVAL_EPISODES,
        "steps_per_episode": STEPS_PER_EPISODE,
        "arms": {a: {"z_goal_seeding_gain": ARM_GAINS[a]} for a in ARM_IDS},
        "coupling_floor_r": COUPLING_FLOOR_R,
        "min_coupling_n": MIN_COUPLING_N,
        "min_dv_positives": MIN_DV_POSITIVES,
        "min_dv_base_rate": MIN_DV_BASE_RATE,
        "min_coupling_n_per_seed": MIN_COUPLING_N_PER_SEED,
        "min_dv_positives_per_seed": MIN_DV_POSITIVES_PER_SEED,
        "channel_std_floor": CHANNEL_STD_FLOOR,
        "z_goal_dose_separation_floor": Z_GOAL_DOSE_SEPARATION_FLOOR,
        "approach_raw_thresh": APPROACH_RAW_THRESH,
        "lag_horizons": {
            "approach": LAG_APPROACH_H, "benefit": LAG_BENEFIT_H,
            "moved": LAG_MOVED_H, "reef_exit": LAG_REEFEXIT_H,
        },
        "core_channels": CORE_CHANNELS,
        "seeds": list(seeds),
    }

    # arm_results carries the per-cell fingerprints; stamp_recording_core HOISTS
    # substrate_hash from them, so this must be in the manifest before the write.
    arm_results = [
        {k: v for k, v in r.items() if k != "episodes"} for r in rows
    ]

    episode_log = {
        "experiment_type": EXPERIMENT_TYPE,
        "env_config":      ENV_KWARGS,
        "phase":           "zgoal_wanting_coupling_reinstrument",
        "toroidal":        ENV_KWARGS.get("toroidal", False),
        # TOP-LEVEL "seeds" is REQUIRED by fishtank_viz.html's loadData(), which
        # hard-fails ("Episode log has no seed data.") on any other key -- so the
        # two arms are FLATTENED into one seed list carrying an `arm` label rather
        # than nested under an "arms" key. Confirmed live twice in this lineage
        # (V3-EXQ-913 wrote "runs"; the V3-EXQ-483 family writes "arms"), and
        # caught here by validate_experiments' fishtank-episode_log-seeds check.
        # The per-arm view is preserved by the `arm` field, which the viewer uses
        # for seed-button labelling, plus the "arms" index below for provenance.
        "seeds": [
            {"seed": r["seed"], "arm": a,
             "z_goal_seeding_gain": ARM_GAINS[a],
             "episodes": r.get("episodes", [])}
            for a in ARM_IDS for r in rows_by_arm[a]
        ],
        "arms": [{"arm_id": a, "z_goal_seeding_gain": ARM_GAINS[a]} for a in ARM_IDS],
    }

    return {
        "status":             outcome,
        "outcome":            outcome,
        "metrics":            metrics,
        "summary_markdown":   summary_markdown,
        "claim_ids":          CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "evidence_direction": "non_contributory",
        "experiment_type":    EXPERIMENT_TYPE,
        "interpretation":     interpretation,
        "arm_results":        arm_results,
        "per_arm":            {a: {k: v for k, v in per_arm[a].items()} for a in ARM_IDS},
        "non_degenerate":     bool(agg["non_degenerate"]),
        "degeneracy_reason":  agg["degeneracy_reason"],
        "per_arm_gate":       agg["per_arm_gate"],
        "config":             config_snapshot,
        "seeds":              list(seeds),
        "episode_log":        episode_log,
    }


def _fmt(v: Optional[float], nd: int = 4) -> str:
    return "null" if v is None else f"{v:+.{nd}f}"


def _summary_markdown(outcome, interp_label, per_arm, agg, dose_separation, seeds) -> str:
    lines: List[str] = []
    lines.append("# V3-EXQ-932a -- z_goal / residue_wanting -> behaviour coupling (RE-INSTRUMENTED)\n")
    lines.append(f"**Status:** {outcome}  (diagnostic / observational -- claim_ids=[], weights no governance)")
    lines.append(f"**Interpretation label:** `{interp_label}`")
    lines.append(f"**Green arms:** {agg['green_arms'] or '(none)'}   **Red arms:** {agg['red_arms'] or '(none)'}")
    lines.append(f"**z_goal dose separation:** {dose_separation:+.4f} "
                 f"(floor {Z_GOAL_DOSE_SEPARATION_FLOOR:+.4f})\n")
    lines.append("Lettered re-instrument of V3-EXQ-932 per "
                 "`failure_autopsy_931-932-wanting-authority-cluster_2026-08-16` Section 8. "
                 "932's MEASUREMENT-VALIDITY PASS stands; its reported coupling narrative is "
                 "what this repairs.\n")
    lines.append("**A non-settled coupling reports `null`, never 0.0.** Statuses: `settled` | "
                 "`underpowered_n` | `unsettable_x_degenerate` (affect channel flat) | "
                 "`unsettable_dv_degenerate` (behaviour DV cannot move).\n")
    if agg["degeneracy_reason"]:
        lines.append(f"> {agg['degeneracy_reason']}\n")
    for arm_id in ARM_IDS:
        info = per_arm[arm_id]
        lines.append(f"## Arm `{arm_id}`  (z_goal_seeding_gain = {info['z_goal_seeding_gain']})\n")
        lines.append(f"- z_goal active_frac (mean over seeds): {info['z_goal_active_frac_mean']:.4f}")
        lines.append(f"- chan_min_std residue_wanting: {info['chan_min_std']['residue_wanting']:.5f}"
                     f"   z_goal: {info['chan_min_std']['z_goal']:.5f}")
        lines.append(f"- chan_max_std (932-comparable, REPORTED ONLY) residue_wanting: "
                     f"{info['chan_max_std']['residue_wanting']:.5f}"
                     f"   z_goal: {info['chan_max_std']['z_goal']:.5f}")
        lines.append(f"- gating couplings settled: {info['n_gating_couplings_settled']}"
                     f" / {len(GATING_COUPLING_NAMES)}\n")
        lines.append("| coupling | status | r | rho | within-seed r | seeds def. | n | DV base rate | partial r |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for c in COUPLING_NAMES + [CONTEMPORANEOUS_NAME]:
            m = info["couplings"][c]
            ctrl = m.get("partial_control")
            pr = m.get("partial_r_given_" + ctrl) if ctrl else None
            br = "n/a" if m.get("y_base_rate") is None else f"{m['y_base_rate']:.3f}"
            gate_tag = "" if c in GATING_COUPLING_NAMES else " *(reported-only)*"
            lines.append(
                f"| `{c}`{gate_tag} | {m['status']} | {_fmt(m['r'])} | {_fmt(m['rho'])} | "
                f"{_fmt(m['within_seed_pooled_r'])} | {m['n_seeds_defined']}/{m['n_seeds_total']} | "
                f"{m['n']} | {br} | {_fmt(pr)} |")
        lines.append("")
    lines.append("`*_to_approach_t1` (mode-based) and `*_to_approachraw_t1` are REPORTED-ONLY and "
                 "can never gate this run -- see `interpretation.approach_dv_disposition`.")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    import argparse
    import json
    import time
    from datetime import datetime

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",   type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    _t0 = time.perf_counter()
    result = run(seeds=args.seeds, dry_run=args.dry_run)

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"]      = ts
    result["timestamp_utc"]      = ts
    result["run_id"]             = f"{EXPERIMENT_TYPE}_{ts}_v3"
    result["architecture_epoch"] = "ree_hybrid_guardrails_v1"
    result["experiment_purpose"] = EXPERIMENT_PURPOSE
    result["claim_ids"]          = CLAIM_IDS

    episode_log = result.pop("episode_log", None)
    # V3-EXQ-932 wrote the episode log UNCONDITIONALLY, so every --dry-run smoke
    # left a toy log under evidence/. Real runs only.
    if episode_log is not None and not args.dry_run:
        out_dir = resolve_evidence_experiments_dir(Path(__file__)) / EXPERIMENT_TYPE
        out_dir.mkdir(parents=True, exist_ok=True)
        episode_log["run_id"] = result["run_id"]
        log_path = out_dir / f"{EXPERIMENT_TYPE}_{ts}_episode_log.json"
        log_path.write_text(json.dumps(episode_log, indent=2) + "\n", encoding="utf-8")
        print(f"Episode log written to: {log_path}", flush=True)
        # Declared companion path is resolved by experiment_runner.py against the
        # MANIFEST's directory (evidence/experiments/), one level above where the
        # log actually lands -- so the prefix is required.
        result["companion_files"] = [f"{EXPERIMENT_TYPE}/{log_path.name}"]

    out_path = write_flat_manifest(
        result,
        dry_run=args.dry_run,
        config=result.get("config"),
        seeds=result.get("seeds"),
        script_path=Path(__file__),
        started_at=_t0,
        agent=list(_ARM_AGENTS.values()) or None,
        z_goal_stream_stats=_ZG.stats(),
    )

    print(f"\nResult written to: {out_path}", flush=True)
    print(f"Status: {result['status']}", flush=True)
    print(f"final_outcome: {result['outcome']}", flush=True)

    _outcome_raw = str(result.get("outcome", "FAIL")).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
        dry_run=bool(args.dry_run),
    )
