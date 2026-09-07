"""V3-EXQ-1007 -- MECH-536 eval-time action-persistence discriminator: does a BG-like latch
on the SAME frozen direction-blind reader abolish the ambitendency two-cycle WITHOUT
restoring foraging competence? (+ fishtank observable with per-episode eval-arm badge)

Claims: MECH-536 (primary), MECH-535 (read-across)

EXPERIMENT_PURPOSE = "evidence"

SLEEP DRIVER: not applicable -- no sleep flag is set by this driver (the x734 all-ON stack
does not enable use_sleep_loop / sws_enabled / rem_enabled / use_sleep_aggregation_cluster
at this rung). Recorded as sleep_driver_pattern="none" rather than omitted.

red-team (fable): CONTESTED -- 4 findings applied (F1 effect guard on the 1.0 floor and the
envelope; F2 constant effect floor; F3 unique-cells gain in C2's non-degeneracy; F4 a MECH-535
weakening branch), 4 informational findings recorded (F5-F8). Full dispositions in the queue
entry note for V3-EXQ-1007.

=== WHY THIS RUN ===

Thought-intake evidence/planning/thought_intake_2026-09-06_direction_blind_reactive_ambitendency.md
registered MECH-535 (direction-blind reactive ambitendency: a memoryless greedy reader of a
representation carrying goal PROXIMITY but not goal DIRECTION produces, from one frozen
policy, a two-cell approach/withdraw limit cycle or a boundary-press fixed point by initial
condition) and MECH-536 (BG-like action persistence is PROTECTIVE against representational
degradation, not NECESSARY for competence). MECH-536's notes name THE DISCRIMINATOR:

    an eval-time action-persistence wrapper on the SAME frozen OFF-arm policy, scored on
    cycle incidence AND resources/episode.
    Cycle gone + competence flat  -> representational deficit (MECH-536 + MECH-535 supported)
    Cycle gone + competence rises -> gating deficit (MECH-536 weakened; routes to ARC-107 root C)

This run is that discriminator. Routing chip: chip-20260906-mech536-eval-persistence-
discriminator. It is a NEW NUMBER, not a 978 letter: it asks what action persistence is FOR
relative to representation quality, which is a different scientific question from 978's (does
directional supervision change z_world?). It is a different axis from V3-EXQ-1002 (readout
adequacy of the same frozen latent under a supervised adapter; landed 2026-09-05, FAIL,
H-C geometry mismatch: a capacity-matched adapter on the 978-OFF latent reproduces the
local_view_greedy oracle at 0.66 held-out agreement against 0.98 from the raw field, and its
cloned policy forages 16.9 res/ep against 51.5) and from GFLAG-0131's stochastic-eval ask
(arm (e) below carries that as a COMPARATOR, not a gate).

=== THE PHENOTYPE, AND ONE FACT THE BRIEF DID NOT STATE ===

V3-EXQ-978's fishtank log (seed 42, 6 field_loss_off + 6 field_loss_on frozen rollouts):
10/12 episodes end in a two-cell limit cycle killed by the causal-footprint contamination
rule at step 11-23 in a HAZARD-FREE env (num_hazards=0; +0.5 contamination per visit, cell
retyped at 2.0, 0.4 health per contaminated step); 2/12 are a boundary-press fixed point
(198/200 stationary, survives, eats nothing).

The scored 978 OFF arm shows the cycle is SEED-SPECIFIC in that run:

    seed 42  competence 0.00  survival  50.75  death_rate 0.80   (cycles, dies)
    seed 43  competence 0.25  survival 200.00  death_rate 0.00   (no cycling deaths at all)
    seed 44  competence 0.55  survival 200.00  death_rate 0.00   (no cycling deaths at all)

A two-cell cycle under contamination cannot survive 200 steps, so seeds 43/44's greedy reader
is fixed-point / wandering, not cycling. Consequence for THIS design: the latch can only reach
the DV through CYCLING episodes (a wall-press repeats one action, so a latch on it changes
nothing). C1 is therefore evaluated PER SEED, conditioned on the greedy anchor actually
cycling on that seed (`cycle_incidence_greedy >= CYCLE_PRESENCE_FLOOR`), and the number of
cycle-present seeds is recorded and carried into the criteria's non-degeneracy flags. If no
seed cycles the manipulation never reaches the DV and the run self-routes
`greedy_cycle_absent_manipulation_unreachable` (non_contributory), not to a claim verdict.
The C1 verdict may rest on ONE seed; this is said here so nobody reads it as three.

=== DESIGN -- the manipulation is EVAL-TIME ONLY; the trained reader is SHARED ===

Training, once per seed (42/43/44), reproduces 978's field_loss_off cell EXACTLY: the x734
all-ON stack on the D3_hazard_free rung, W3_survival_zeroed weighting, its own P0a/P0/P1
warmup via `x734._train_all_on_agent` at `zworld_p0_resource_field_weight=0.0`, then a
`x734.PPOPolicyNet` reader trained by `x808._train_ppo_decomposed` on sense-time z_world
ALONE (32 dims). Every budget, env kwarg, rung, weighting and PPO hyperparameter is IMPORTED
from x734 / x808 / x978 (`_make_agent` is x978's own), never redefined, so this reader cannot
drift from the one 978 scored and 1002 froze. The trained (agent, net) pair is snapshotted
and every eval arm below runs on a fresh deep copy of that snapshot, on a fresh env built
with the SAME seed, so arms are PAIRED on identical initial conditions and identical
post-training agent state (crf_persist_rules_across_episode_reset=True and goal_state make
the agent's internal state carry across episodes; a shared sequential agent would let arm
order leak into the contrast).

Eval arms, all on the same reader:

  (a) greedy_argmax      x737.LatentPPOEvalPolicy's rule (argmax every step, memoryless) --
                         the 978 replication anchor. Expected per seed: 0.00/0.25/0.55 res/ep.
  (b) persist_k2         take the argmax, then EXECUTE it for k=2 consecutive steps before
                         re-deciding. The minimal latch MECH-536 names ("k>=2").
  (c) persist_k4         same, k=4.
  (d) switch_cost        MECH-266-style Schmitt trigger: keep the previously executed action
                         unless the new argmax's logit exceeds the held action's logit by a
                         margin delta. delta is set by a PRE-REGISTERED RULE, not a
                         pre-registered number: the median (SWITCH_COST_MARGIN_QUANTILE) of
                         arm (a)'s own per-step switch margins (logit[argmax] - logit[previous
                         executed action] on steps where they differ), per seed. The realised
                         delta and its n are recorded per seed.
  (e) stochastic_sample  Categorical sampling at temperature 1.0 from the same logits --
                         GFLAG-0131's ask; the non-biological comparator (noise as de facto
                         commitment perturbation). REPORTED, never a gate.

In every arm the reader's forward pass (agent.sense -> z_world -> logits) runs on EVERY step,
so the latent trajectory the agent experiences is computed identically; the arms differ only
in which action is EXECUTED. The arm's `decided` (argmax) and `executed` actions are both
recorded per step.

Anchors (same env, same seed): random_walk; local_view_greedy (the memoryless greedy forager
on the 5x5 field, 45.75 res/ep in 978 -- MECH-536's "commitment is not necessary given a good
representation" fact); and local_view_greedy_persist_k2 -- the SAME k=2 latch on the good
representation. Pre-registered expectation: the latch does not hurt it (C3, reported). If it
does, MECH-536's "protective, not necessary" framing carries a caveat, recorded in
`interpretation.caveats`.

Contamination sub-grid: arms (a) and (b) are re-run on an env with contamination_spread=0.0
(the V3-EXQ-513 idiom) so cycle-TRUNCATION death is separated from the cycle itself: under
contamination-off a greedy two-cycle survives 200 steps eating nothing, and the latch's
competence contrast is survival-matched. Reported as C2_survival_matched, never a gate.

=== WHAT A PURE LATCH DOES BY CONSTRUCTION (read before interpreting C1) ===

A period-2 orbit in position needs the executed action to alternate every step. Under
persist_k>=2 the executed action is constant for k consecutive steps, so a period-2 orbit is
impossible except through blocked moves (a wall), which read as STATIONARY, not as a cycle.
C1 on arms (b)/(c) is therefore expected by construction: it is the check that the
manipulation REACHED the DV (the same role a positive control plays), and its failure would
indicate an instrumentation defect in the wrapper, not a finding. The scientific content is
in C2 and in what REPLACES the cycle: a k-step latch of a two-cycle can produce a period-2k
orbit over 3 cells (A B C B A ...), which is still a bounded orbit and still contamination-
lethal. `bounded_orbit_incidence` (smallest period p in 2..BOUNDED_ORBIT_MAX_PERIOD repeating
>= BOUNDED_ORBIT_MIN_REPEATS times over >= 2 distinct cells) and the per-arm period histogram
are recorded so "the cycle is gone" cannot silently mean "the cycle got longer". Arm (d) is
where C1 is NOT guaranteed (it depends on delta), which is why it is not a verdict arm.

=== DVs ===

  foraging_competence  mean resources consumed per eval episode, from
                       `_lib.capability_eval.evaluate_seed` -- the SAME number 978 reports.
                       NOT reimplemented.
  cycle_incidence      fraction of the SCORED eval episodes whose position trace contains a
                       two-cell period-2 orbit of length >= CYCLE_MIN_LEN (6) steps:
                       positions p[i..i+5] with p[i] != p[i+1] and p[j] == p[j+2]. The trace
                       is recorded by the policy wrapper as a side-channel DURING evaluate_seed
                       (the policy sees env.agent_x/agent_y at act time), so the DV loop is
                       untouched and the detector runs on the scored episodes, not on a
                       separate pass. `--self-test` runs the detector on the 978 episode log
                       (Mac copy) where it must fire on exactly 10/12 episodes and on neither
                       fixed point; synthetic cases run on every box.
  survival_horizon, unique_cells, fixed_point_incidence (stationary fraction >=
  FIXED_POINT_STATIONARY_FRAC), bounded_orbit_incidence, period histogram, per-episode rows,
  per-step logit margins -- secondary, all recorded.

=== PRE-REGISTERED VERDICTS (from MECH-536's notes; operationalised, not softened) ===

Verdict arms: (b) persist_k2 and (c) persist_k4 -- the parameter-free pure latches. (d) is
reported with its own C1/C2 flags (`latch_family` block) and disagreement with (b)/(c) is
flagged, never adjudicated.

  C1  On the cycle-present seeds, a strict majority have cycle_incidence <= 1/20 (0.05) on
      EACH verdict arm. (Load-bearing.)
  C2  On EVERY seed and each verdict arm the per-seed lift over arm (a) is below the effect
      floor C2_EFFECT_FLOOR = 0.5 res/ep (a CONSTANT: half the family's D3 competence floor
      and ~2 x the 978 OFF arm's seed SD of 0.22, which is the brief's "seed SD of (a)" term
      frozen at authoring). Clearing the 1.0 floor counts as a rise ONLY when the lift also
      clears the effect floor (red-team F1: without that guard one resource in 20 episodes on
      a fixed-point seed flipped supports to weakens). An in-run SD term was REMOVED (red-team
      F2): for a null criterion it inflates on heterogeneous lifts and reads a mixed effect as
      flat. The realised lift SD and greedy seed SD are recorded. (Load-bearing.)

  C1 AND C2      -> PASS  latch_abolishes_cycle_competence_flat_representational_deficit
                          MECH-536 supports, MECH-535 supports.
  C1 AND NOT C2  -> FAIL, split by ONE pre-registered refinement made at authoring:
     the latch also removes contamination death, so a latched agent lives ~200 steps instead
     of ~11 and any straight-line runner crosses more food by opportunity alone. A rise that
     stays INSIDE what undirected motion achieves is not a sign of restored direction. So:
       latch competence > random_walk anchor on a strict majority of seeds, OR >= 1.0 on any
       seed -- in both tests only on seeds whose lift is a genuine rise (>= the effect floor)
                                  -> latch_restores_competence_gating_deficit_route_arc107_root_c
                                     MECH-536 weakens, MECH-535 weakens.
       otherwise (rise within the random-walk envelope on every seed)
                                  -> latch_lift_within_undirected_envelope
                                     MECH-536 mixed (a MEASURED, equivocal effect -- exactly
                                     what `mixed` means), MECH-535 unknown.
  NOT C1         -> FAIL  latch_did_not_abolish_cycle_non_contributory; both claims
                          non_contributory, with the arm/seed named.
  Gate red       -> FAIL  substrate_not_ready_requeue (encoder never moved / env not
                          floor-achievable from the local view), or
                          anchor_replication_failed_greedy_supra_floor (the direction-blind
                          premise did not replicate), or
                          greedy_cycle_absent_manipulation_unreachable. MECH-536
                          non_contributory on all three. MECH-535 (red-team F4): on the
                          no-cycle branch, `weakens` if the greedy reader showed NEITHER a cycle
                          NOR a fixed point on a strict majority of seeds (its phenotype did not
                          replicate -- counted against it, not filtered as silence); `unknown`
                          if fixed points replicated without a cycle (half the phenotype);
                          non_contributory on the other two gate labels.

Reported alongside, never gating: C3 (lvg persist_k2 retains >= LVG_LATCH_RETENTION of lvg on
every seed), C2_survival_matched (contamination-off (b) vs (a) lift below the same effect
floor), the switch_cost flags, the stochastic comparator, orbit replacement.

=== NON-DEGENERACY ===

C1 is non-degenerate iff >= 1 cycle-present seed exists AND the latch's cycle incidence
differs from greedy's on at least one of them. C2 is non-degenerate iff the latch CREATED
OPPORTUNITY: on every cycle-present seed the verdict arms' mean survival horizon exceeds
greedy's AND their mean unique cells visited exceeds greedy's by >= UNIQUE_CELLS_MIN_GAIN (3)
(red-team F3: survival alone is satisfied by a boundary press, which never enters a cell and
so never contaminates, and by a k-latched period-4 orbit that dies a few steps later; neither
is foraging exposure). A verdict arm that is a fixed point on >= 50% of episodes of a
cycle-present seed is recorded as the caveat `latch_converted_cycle_to_fixed_point` --
ambitendency became stupor, which is itself the "straight runs into the boundary" reading. Both feed `criteria_non_degenerate`; top-level `non_degenerate` is their AND under
a green gate. Identical competence across greedy and latch on every seed is NOT read as
degenerate: on a fixed-point seed the latch changes nothing by construction and that is the
prediction, not a dead DV -- the anchors (lvg 45.75) certify the DV is live.

=== DV-SYMMETRY INVARIANCE (per arm) ===

DV = consumption count per episode and a predicate on the position trace; symmetry group =
permutations of episodes (and, for competence, of within-episode step order preserving
consumption count). Each arm's manipulation rewrites the EXECUTED action sequence, hence the
trajectory that generates both DVs:
  persist_k2 / persist_k4: replaces argmax(logits_t) by argmax(logits_{t-j}); not a broadcast
      constant on the logits (the argmax is ignored on held steps), not a monotone rescaling,
      not a permutation of interchangeable units.
  switch_cost: adds delta to the HELD action's effective score only -- asymmetric between
      candidates, so the argmax can change; not a broadcast constant.
  stochastic_sample: replaces the argmax by a draw; changes the executed sequence.
  anchors: the k=2 latch on local_view_greedy is the same non-invariant rewrite.
Path open in every arm.

=== STEP 2.5c SUBSTRATE-PATH OVERLAP, dispositioned ===

Open corrupting entries listing ree_core/agent.py (mode-governance-engagement, SD-082),
tonic_vigor.py (MECH-320), e1_deep.py::forward / ::predict_long_horizon
(SD-e1-rollout-consistency-training), ContextMemory.write (contextmemory-write-path),
blocked_agency.py: the eval-path substrate call is `agent.sense()` ONLY (x737._agent_zworld);
use_salience_coordinator, use_tonic_vigor, use_blocked_agency and e1_rollout_consistency are
default-off in this config and sd016_writepath_mode is "off", and lateral_pfc compute_bias
(SD-082) sits on the select_action path the reader never calls. Training runs the imported
x734 recipe unchanged since 978 (which ran under the same open entries), and because the
reader is SHARED, any training-path effect is a constant across every eval arm and cannot
produce an arm contrast. SD-018 (degrading, stack.py / zworld_p0.py) is on the training path
and is recorded, not blocking.

=== RECORDED, NOT GATING (red-team F5-F8) ===

F5: contamination_spread=0 zeroes contamination_view = world_state[175:200], an input the
reader trained with LIVE, so the contamination-off sub-grid is off-distribution for the
encoder and the "greedy two-cycle survives 200 steps eating nothing" reading is asserted, not
guaranteed; `recorded_checks.contamination_off_greedy_cycle_present` says whether it held.
F6: C1 is a reach check for the pure latches (above); the verdict rests on C2. F7: on a seed
where greedy never switches action, delta=0 and arm (d) duplicates arm (a) -- listed in
`metrics.switch_cost_degenerate_seeds`. F8: this driver reseeds torch/numpy before every eval
arm (so arms are RNG-paired) whereas 978 evaluated straight after PPO with no reseed; the env
holds a private RNG so layouts are identical either way, and `agent.sense()` was probed at
authoring (2026-09-07): bit-identical z_world under two different global seeds and no torch
RNG-state change across a sense() call, so the reseed cannot move the greedy replication. A
miss on the per-seed replication would land non_contributory, never a mis-attribution.

=== MECH-535 SUPPORT IS A RE-OBSERVATION PLUS A DISSOCIATION ===

The seeds are 978's, so the greedy reader's cycle on seed 42 re-observes the same exemplar (a
deterministic re-train on the same machine class should reproduce it bit-for-bit). What this
run ADDS for MECH-535 is the latch dissociation (the cycle is representational, not gating)
and the per-seed phenotype distribution (cycle vs fixed point) on the scored episodes of all
three seeds; an independent replication on fresh seeds remains owed and is said so here.

=== FISHTANK ===

A separate observational log pass (NOT the scored data; 978's pattern) re-runs every eval
arm, both contamination-off cells and the three anchors from a fresh copy of the trained
snapshot on another fresh env (seed + 7777), LOG_EPISODES each, first seed only. Per episode
`arm` is the EVAL arm id (REE_assembly 2e10701d76 renders a per-episode arm badge), and per
step the log carries 978's fields (pos / action / health / energy / z_world_norm / the three
resource_field argmax series / resource_field_max / hazards / resources) plus
`decided_action`, `executed_action`, `held`, `top2_margin`, `switch_margin`.

=== COST ===

978 trained 6 cells (2 arms x 3 seeds) in 17.4 h on ree-worker-3. This run trains 3 (one
reader per seed) and adds 10 cheap eval cells plus a 10-arm log pass per seed; estimated
~11 h on the same class.
"""
from __future__ import annotations

import copy
import glob
import json
import sys
import types
from pathlib import Path
from statistics import pstdev
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from experiment_protocol import emit_outcome  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from experiments._metrics import check_degeneracy  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._lib.capability_eval import (  # noqa: E402
    COMPETENCE_RESOURCE_FLOOR,
    LocalViewGreedyPolicy,
    Policy,
    RandomPolicy,
    evaluate_seed,
)
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402
from experiments._lib.zworld_encoder_guard import (  # noqa: E402
    latent_stack_snapshot,
    latent_stack_weight_delta,
)
import experiments.v3_exq_734_env_difficulty_competence_recovery_sweep as x734  # noqa: E402
import experiments.v3_exq_737_ree_latent_policy_head_competence_probe as x737  # noqa: E402
import experiments.v3_exq_808_return_decomposition_objective_misspecification as x808  # noqa: E402
import experiments.v3_exq_978_sd018_directional_field_fishtank as x978  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_1007_mech536_eval_persistence_discriminator"
QUEUE_ID = "V3-EXQ-1007"
EXPERIMENT_PURPOSE = "evidence"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"
CLAIM_IDS = ["MECH-536", "MECH-535"]

DEVICE = torch.device("cpu")

# Same seeds as 978 / 1002 so arm (a) is a per-seed replication of the 978 OFF cell.
# Not a reef config at this rung, so the seed-44 reef-instability rule does not apply.
SEEDS: List[int] = [42, 43, 44]

# ---- IMPORTED, NEVER REDEFINED: this is what makes the trained reader 978's reader ----------
ZWORLD_P0_EPISODES = x734.ZWORLD_P0_EPISODES        # 60
P0_WARMUP_EPISODES = x734.P0_WARMUP_EPISODES        # 200
P1_REINFORCE_EPISODES = x734.P1_REINFORCE_EPISODES  # 90
P1_PPO_EPISODES = x734.P1_PPO_EPISODES              # 1000
EVAL_EPISODES = x734.EVAL_EPISODES                  # 20
STEPS_PER_EPISODE = x734.STEPS_PER_EPISODE          # 200
PPO_ROLLOUT_EPISODES = x734.PPO_ROLLOUT_EPISODES    # 8

RUNG = x734.DIFFICULTY_RUNGS[-1]                    # D3_hazard_free
RUNG_ID = RUNG["rung_id"]
_W3 = next(w for w in x808.WEIGHTINGS if w["id"] == "W3_survival_zeroed")
LEVEL_ID = str(_W3["id"])
W_CONSUME = float(_W3["w_consume"])
W_SURVIVAL = float(_W3["w_survival"])

# 978's OFF arm: the directional head is BUILT (x978._make_agent) but its P0a weight is 0.0.
OFF_ARM_P0A_FIELD_WEIGHT = 0.0
ZWORLD_DELTA_FLOOR = x808.ZWORLD_DELTA_FLOOR        # 1e-6

DRY_RUN_SEEDS = [42]
DRY_RUN_ZWORLD_P0 = x734.DRY_RUN_ZWORLD_P0
DRY_RUN_P0 = 2
DRY_RUN_P1 = 2
DRY_RUN_PPO = 6
DRY_RUN_EVAL = 2
DRY_RUN_STEPS = 20
DRY_RUN_ROLLOUT = 3

# ---- eval arms -------------------------------------------------------------------------
ARM_GREEDY = "greedy_argmax"
ARM_K2 = "persist_k2"
ARM_K4 = "persist_k4"
ARM_SWITCH = "switch_cost"
ARM_SAMPLE = "stochastic_sample"
EVAL_ARM_IDS = [ARM_GREEDY, ARM_K2, ARM_K4, ARM_SWITCH, ARM_SAMPLE]
VERDICT_ARMS = [ARM_K2, ARM_K4]
LATCH_ARMS = [ARM_K2, ARM_K4, ARM_SWITCH]
PERSIST_K = {ARM_K2: 2, ARM_K4: 4}
SAMPLE_TEMPERATURE = 1.0
# delta for arm (d) = this quantile of arm (a)'s per-step switch margins, per seed (a RULE).
SWITCH_COST_MARGIN_QUANTILE = 0.5
SWITCH_MARGIN_MIN_N = 20   # below this the calibration is recorded as thin (informational)

ANCHOR_RANDOM = "random_walk"
ANCHOR_LVG = "local_view_greedy"
ANCHOR_LVG_K2 = "local_view_greedy_persist_k2"
ANCHOR_IDS = [ANCHOR_RANDOM, ANCHOR_LVG, ANCHOR_LVG_K2]
LVG_LATCH_K = 2

CONTAM_OFF_SUFFIX = "@contamination_off"
CONTAM_OFF_ARMS = [ARM_GREEDY, ARM_K2]
CONTAMINATION_OFF_KWARGS = {"contamination_spread": 0.0}   # V3-EXQ-513 idiom, explicit

# --------------------------------------------------------------------------------------
# PRE-REGISTERED THRESHOLDS (constants; never derived from this run's own statistics)
# --------------------------------------------------------------------------------------
CYCLE_MIN_LEN = 6                  # steps of strict two-cell alternation that count as a cycle
CYCLE_INCIDENCE_CEILING = 0.05     # C1: <= 1 of 20 eval episodes
CYCLE_PRESENCE_FLOOR = 0.25        # a seed is cycle-present iff greedy cycles on >= 5/20 eps
FIXED_POINT_STATIONARY_FRAC = 0.9  # stationary transitions fraction that reads as a fixed point
BOUNDED_ORBIT_MAX_PERIOD = 8
BOUNDED_ORBIT_MIN_REPEATS = 3
C2_EFFECT_FLOOR = 0.5              # res/ep; half the family's competence floor, and ~2 x the
                                   # 978 OFF arm's seed SD (0.22). A CONSTANT: red-team F2 --
                                   # an in-run SD term inflates on heterogeneous lifts and reads
                                   # a mixed effect as flat, the wrong direction for a null test
UNIQUE_CELLS_MIN_GAIN = 3          # C2 non-degeneracy: latch visits >= 3 more cells than greedy
PHENOTYPE_INCIDENCE_FLOOR = 0.5    # MECH-535 phenotype (cycle OR fixed point) per seed
LVG_LATCH_RETENTION = 0.5          # C3: lvg persist_k2 keeps >= 50% of lvg, every seed
COMPETENCE_FLOOR = float(COMPETENCE_RESOURCE_FLOOR)   # 1.0

# 978's scored OFF-arm values, for the replication diagnostic (recorded, never gated).
EXPECTED_978_OFF_COMPETENCE = {42: 0.0, 43: 0.25, 44: 0.55}
EXPECTED_978_OFF_SURVIVAL = {42: 50.75, 43: 200.0, 44: 200.0}

# Fishtank episode-log thinning.
LOG_EPISODES = 6
DRY_RUN_LOG_EPISODES = 2

# 978 episode-log companion (Mac copy; origin holds only a placemarker). Used by --self-test.
EXQ978_EPISODE_LOG_GLOB = (
    "/Users/dgolden/REE_Working/REE_assembly/evidence/experiments/"
    "v3_exq_978_sd018_directional_field_fishtank/runs/*/*_episode_log.json"
)
EXQ978_LOG_EXPECTED_CYCLING = 10
EXQ978_LOG_EXPECTED_FIXED_POINTS = 2

_ZG = ZGoalStreamAccumulator()


def _mean(vals: Sequence[float]) -> float:
    return float(sum(vals) / len(vals)) if vals else 0.0


def _strict_majority(n: int) -> int:
    return n // 2 + 1


# --------------------------------------------------------------------------------------
# Trace detectors (pure functions of a position list)
# --------------------------------------------------------------------------------------
def period2_cycle_present(positions: Sequence[Tuple[int, int]],
                          min_len: int = CYCLE_MIN_LEN) -> bool:
    """True iff some window of `min_len` consecutive positions alternates between two DISTINCT
    cells: p[i] != p[i+1] and p[j] == p[j+2] across the window. A stationary run (wall-press)
    has p[i] == p[i+1] and never fires."""
    pos = [tuple(p) for p in positions]
    n = len(pos)
    if n < min_len:
        return False
    for i in range(n - min_len + 1):
        w = pos[i:i + min_len]
        if w[0] == w[1]:
            continue
        if all(w[j] == w[j + 2] for j in range(min_len - 2)):
            return True
    return False


def stationary_fraction(positions: Sequence[Tuple[int, int]]) -> float:
    pos = [tuple(p) for p in positions]
    if len(pos) < 2:
        return 0.0
    same = sum(1 for j in range(len(pos) - 1) if pos[j] == pos[j + 1])
    return float(same) / float(len(pos) - 1)


def bounded_orbit_period(positions: Sequence[Tuple[int, int]],
                         max_period: int = BOUNDED_ORBIT_MAX_PERIOD,
                         min_repeats: int = BOUNDED_ORBIT_MIN_REPEATS) -> Optional[int]:
    """Smallest period p in 2..max_period such that some window of min_repeats*p consecutive
    positions repeats with period p over >= 2 distinct cells. None if no such orbit. A period-2
    two-cell cycle returns 2; a k-latched two-cycle (A B C B A B C B) returns 4."""
    pos = [tuple(p) for p in positions]
    n = len(pos)
    for p in range(2, max_period + 1):
        L = min_repeats * p
        if n < L:
            break
        for i in range(n - L + 1):
            w = pos[i:i + L]
            if len(set(w)) < 2:
                continue
            if all(w[j] == w[j + p] for j in range(L - p)):
                return p
    return None


# --------------------------------------------------------------------------------------
# Trace-recording policy wrappers. The DV loop (evaluate_seed / rollout_episode) is untouched;
# the wrapper records positions at act() time and annotates via post_step().
# --------------------------------------------------------------------------------------
class _TracedPolicy(Policy):
    def __init__(self, name: str) -> None:
        self.name = name
        self.episodes: List[Dict[str, Any]] = []
        self._cur: Optional[Dict[str, Any]] = None

    def _begin_episode(self) -> None:
        self._cur = {"positions": [], "steps": []}
        self.episodes.append(self._cur)

    def _record(self, env: Any, decided: int, executed: int, held: bool,
                top2_margin: Optional[float], switch_margin: Optional[float]) -> None:
        if self._cur is None:
            self._begin_episode()
        pos = (int(env.agent_x), int(env.agent_y))
        self._cur["positions"].append(pos)
        self._cur["steps"].append({
            "pos": [pos[0], pos[1]],
            "decided_action": int(decided),
            "executed_action": int(executed),
            "held": bool(held),
            "top2_margin": top2_margin,
            "switch_margin": switch_margin,
        })

    def post_step(self, env: Any, info: Dict[str, Any], obs_dict: Dict[str, Any]) -> None:
        if self._cur is None or not self._cur["steps"]:
            return
        st = self._cur["steps"][-1]
        st["transition_type"] = str(info.get("transition_type", "none")) if isinstance(info, dict) else "none"
        st["health"] = float(getattr(env, "agent_health", 1.0))

    def last_step_record(self) -> Optional[Dict[str, Any]]:
        if self._cur is None or not self._cur["steps"]:
            return None
        return self._cur["steps"][-1]


class ReaderEvalPolicy(_TracedPolicy):
    """The frozen PPO reader on sense-time z_world under one of five EXECUTION rules.

    mode="greedy" is x737.LatentPPOEvalPolicy's rule exactly (argmax; uniform-random fallback
    on non-finite logits). The forward pass runs on every step in every mode; only the
    executed action differs.
    """

    def __init__(self, net, agent, arm_id: str, mode: str, k: int = 1, delta: float = 0.0,
                 temperature: float = 1.0, seed: int = 0) -> None:
        super().__init__(arm_id)
        assert mode in ("greedy", "persist", "switch_cost", "sample"), mode
        self.net = net
        self.agent = agent
        self.mode = mode
        self.k = int(k)
        self.delta = float(delta)
        self.temperature = float(temperature)
        self._gen = torch.Generator().manual_seed(int(seed))
        self._rng = np.random.RandomState(int(seed))
        self._held: Optional[int] = None
        self._hold_left = 0
        self._prev: Optional[int] = None

    def reset(self, env: Any) -> None:
        self.agent.reset()
        self._begin_episode()
        self._held = None
        self._hold_left = 0
        self._prev = None

    def _logits(self, obs_dict: Dict[str, Any]) -> torch.Tensor:
        state = x737._agent_zworld(self.agent, obs_dict)
        with torch.no_grad():
            logits, _v = self.net(state)
        return logits.reshape(-1)

    def act(self, env: Any, obs_dict: Dict[str, Any]) -> int:
        logits = self._logits(obs_dict)
        n = int(env.action_dim)
        if not torch.isfinite(logits).all():
            a = int(self._rng.randint(0, n))
            self._record(env, a, a, False, None, None)
            self._prev = a
            return a
        order = torch.argsort(logits, descending=True)
        argmax = int(order[0].item())
        top2 = float((logits[order[0]] - logits[order[1]]).item()) if n > 1 else 0.0
        switch_margin: Optional[float] = None
        if self._prev is not None and self._prev != argmax:
            switch_margin = float((logits[argmax] - logits[self._prev]).item())
        held = False
        if self.mode == "greedy":
            a = argmax
        elif self.mode == "persist":
            if self._hold_left > 0 and self._held is not None:
                a = self._held
                self._hold_left -= 1
                held = True
            else:
                a = argmax
                self._held = a
                self._hold_left = self.k - 1
        elif self.mode == "switch_cost":
            if self._prev is None or argmax == self._prev:
                a = argmax
            elif float((logits[argmax] - logits[self._prev]).item()) > self.delta:
                a = argmax
            else:
                a = self._prev
                held = True
        else:  # sample
            probs = torch.softmax(logits / self.temperature, dim=-1)
            a = int(torch.multinomial(probs, 1, generator=self._gen).item())
        self._record(env, argmax, a, held, top2, switch_margin)
        self._prev = a
        return a


class AnchorPolicy(_TracedPolicy):
    """A capability_eval anchor policy, trace-recorded, with an optional k-step latch."""

    def __init__(self, base: Policy, arm_id: str, k: int = 1) -> None:
        super().__init__(arm_id)
        self.base = base
        self.k = int(k)
        self._held: Optional[int] = None
        self._hold_left = 0

    def reset(self, env: Any) -> None:
        self.base.reset(env)
        self._begin_episode()
        self._held = None
        self._hold_left = 0

    def act(self, env: Any, obs_dict: Dict[str, Any]) -> int:
        decided = int(self.base.act(env, obs_dict))   # every step, so base RNG use is identical
        held = False
        if self.k > 1 and self._hold_left > 0 and self._held is not None:
            a = self._held
            self._hold_left -= 1
            held = True
        else:
            a = decided
            self._held = a
            self._hold_left = self.k - 1
        self._record(env, decided, a, held, None, None)
        return a

    def post_step(self, env: Any, info: Dict[str, Any], obs_dict: Dict[str, Any]) -> None:
        self.base.post_step(env, info, obs_dict)
        super().post_step(env, info, obs_dict)


def trace_stats(policy: _TracedPolicy) -> Dict[str, Any]:
    """Per-episode trace readouts + arm-level incidences, from the SCORED episodes."""
    rows = []
    for ep_i, ep in enumerate(policy.episodes):
        pos = ep["positions"]
        steps = ep["steps"]
        period = bounded_orbit_period(pos)
        rows.append({
            "ep": ep_i,
            "n_steps": len(pos),
            "cycle": bool(period2_cycle_present(pos)),
            "stationary_fraction": stationary_fraction(pos),
            "fixed_point": bool(stationary_fraction(pos) >= FIXED_POINT_STATIONARY_FRAC
                                and len(pos) >= CYCLE_MIN_LEN),
            "orbit_period": period,
            "unique_cells": len(set(tuple(p) for p in pos)),
            "n_held_steps": sum(1 for s in steps if s.get("held")),
            "n_consume_events": sum(1 for s in steps if s.get("transition_type") == "resource"),
            "final_health": (steps[-1].get("health") if steps else None),
        })
    n = len(rows)
    hist: Dict[str, int] = {}
    for r in rows:
        key = str(r["orbit_period"]) if r["orbit_period"] is not None else "none"
        hist[key] = hist.get(key, 0) + 1
    margins = [s["top2_margin"] for ep in policy.episodes for s in ep["steps"]
               if s.get("top2_margin") is not None]
    switch = [s["switch_margin"] for ep in policy.episodes for s in ep["steps"]
              if s.get("switch_margin") is not None]
    return {
        "n_episodes": n,
        "cycle_incidence": (float(sum(1 for r in rows if r["cycle"])) / n) if n else 0.0,
        "fixed_point_incidence": (float(sum(1 for r in rows if r["fixed_point"])) / n) if n else 0.0,
        "bounded_orbit_incidence": (float(sum(1 for r in rows if r["orbit_period"] is not None)) / n) if n else 0.0,
        "longer_orbit_incidence": (float(sum(1 for r in rows if (r["orbit_period"] or 0) > 2)) / n) if n else 0.0,
        "orbit_period_histogram": hist,
        "mean_unique_cells": _mean([r["unique_cells"] for r in rows]),
        "mean_held_fraction": _mean([(r["n_held_steps"] / r["n_steps"]) if r["n_steps"] else 0.0
                                     for r in rows]),
        "top2_margin_stats": _quantiles(margins),
        "switch_margin_stats": _quantiles(switch),
        "per_episode": rows,
    }


def _quantiles(vals: Sequence[float]) -> Dict[str, Any]:
    if not vals:
        return {"n": 0}
    arr = np.asarray(vals, dtype=np.float64)
    return {"n": int(arr.size), "mean": float(arr.mean()),
            "q25": float(np.quantile(arr, 0.25)), "q50": float(np.quantile(arr, 0.5)),
            "q75": float(np.quantile(arr, 0.75)), "min": float(arr.min()), "max": float(arr.max())}


# --------------------------------------------------------------------------------------
# Frozen-copy helper. A trained REEAgent holds non-leaf tensors (latents with autograd
# history: _committed_candidates, _last_action, _last_e3_selection_result) in its state, and
# torch.Tensor.__deepcopy__ refuses those; it also holds a module reference (hippocampal._rng). For an EVAL copy the
# graph is dead weight, so within this scope every such tensor is copied as a detached
# leaf. Nothing else about deepcopy changes; the patch is restored on exit.
# --------------------------------------------------------------------------------------
class _DetachingDeepcopy:
    def __init__(self) -> None:
        self.n_detached = 0

    def __enter__(self) -> "_DetachingDeepcopy":
        self._orig = torch.Tensor.__deepcopy__
        # A MODULE reference held as state (hippocampal._rng is the `random` module) is a
        # process singleton: share it, as deepcopy already shares functions and classes.
        self._had_module = types.ModuleType in copy._deepcopy_dispatch
        self._orig_module = copy._deepcopy_dispatch.get(types.ModuleType)
        copy._deepcopy_dispatch[types.ModuleType] = (lambda x, memo: x)
        me = self

        def _dc(t, memo):
            if t.requires_grad and not t.is_leaf:
                me.n_detached += 1
                out = t.detach().clone()
                memo[id(t)] = out
                return out
            return me._orig(t, memo)

        torch.Tensor.__deepcopy__ = _dc
        return self

    def __exit__(self, *exc) -> None:
        torch.Tensor.__deepcopy__ = self._orig
        if self._had_module:
            copy._deepcopy_dispatch[types.ModuleType] = self._orig_module
        else:
            copy._deepcopy_dispatch.pop(types.ModuleType, None)


def _copy_frozen(obj: Any) -> Tuple[Any, int]:
    """Deep copy for evaluation: (copy, number of non-leaf tensors detached)."""
    with _DetachingDeepcopy() as ctx:
        out = copy.deepcopy(obj)
    return out, int(ctx.n_detached)


# --------------------------------------------------------------------------------------
# Config slice for the trained-reader cell (mint-as-you-go; include_driver_script_in_hash=False)
# --------------------------------------------------------------------------------------
def _off_reader_config_slice(dry_run: bool, zworld_p0: int, p0: int, p1: int, ppo_eps: int,
                             steps: int, rollout: int) -> Dict[str, Any]:
    """Everything the TRAINING cell's computation reads (warmup + PPO reader). Conservative
    superset; acceptance thresholds and eval-arm parameters are excluded because they are
    applied AFTER the cell has computed and change no recorded training value."""
    return {
        "env_kwargs": x734._env_kwargs_for_rung(RUNG),
        "rung_id": RUNG_ID,
        "level_id": LEVEL_ID,
        "w_consume": W_CONSUME,
        "w_survival": W_SURVIVAL,
        "zworld_p0_episodes": int(zworld_p0),
        "p0_warmup_episodes": int(p0),
        "p1_reinforce_episodes": int(p1),
        "steps_per_episode": int(steps),
        "p0a_field_weight": float(OFF_ARM_P0A_FIELD_WEIGHT),
        "use_resource_field_head": True,
        "resource_field_dim": int(x978.RESOURCE_FIELD_DIM),
        "online_resource_field_weight_inert": float(x978.P0A_FIELD_WEIGHT),
        "ppo_episodes": int(ppo_eps),
        "ppo_rollout_episodes": int(rollout),
        "ppo_lr": float(x734.PPO_LR),
        "ppo_gamma": float(x734.PPO_GAMMA),
        "ppo_gae_lambda": float(x734.PPO_GAE_LAMBDA),
        "ppo_clip": float(x734.PPO_CLIP),
        "ppo_entropy_beta": float(x734.PPO_ENTROPY_BETA),
        "ppo_value_coef": float(x734.PPO_VALUE_COEF),
        "ppo_grad_clip": float(x734.PPO_GRAD_CLIP),
        "ppo_epochs": int(x734.PPO_EPOCHS),
        "ppo_minibatch_size": int(x734.PPO_MINIBATCH_SIZE),
        "ppo_trunk_hidden": int(x734.PPO_TRUNK_HIDDEN),
        "forage_bonus": float(x734.FORAGE_BONUS),
        "novelty_coef": float(x734.NOVELTY_COEF),
        "dry_run": bool(dry_run),
    }


# --------------------------------------------------------------------------------------
# Fishtank episode log (observational pass; 978's schema + the persistence channels)
# --------------------------------------------------------------------------------------
def _log_pass(agent, policy: _TracedPolicy, env, arm_id: str, seed: int, n_episodes: int,
              steps: int) -> List[Dict[str, Any]]:
    episodes: List[Dict[str, Any]] = []
    head = getattr(getattr(getattr(agent, "latent_stack", None), "split_encoder", None),
                   "resource_field_head", None)
    for ep in range(int(n_episodes)):
        _flat, obs = env.reset()
        policy.reset(env)
        ep_steps: List[Dict[str, Any]] = []
        initial_resources = [list(r) for r in env.resources]
        initial_hazards = [list(h) for h in env.hazards]
        done_cause = "step_limit"
        for t in range(int(steps)):
            field = x978._localfield_vector(obs).reshape(-1)
            true_argmax = int(torch.argmax(field).item())
            pred_argmax: Optional[int] = None
            pred_argmax_enc: Optional[int] = None
            with torch.no_grad():
                z = x737._agent_zworld(agent, obs)
                z_norm = float(z.norm().item())
                if head is not None:
                    pred_argmax = int(torch.argmax(head(z).reshape(-1)).item())
                    z_enc = x978._encoder_path_zworld(agent, obs)
                    if z_enc is not None:
                        pred_argmax_enc = int(torch.argmax(head(z_enc).reshape(-1)).item())
            action = policy.act(env, obs)
            rec = policy.last_step_record() or {}
            _f, harm_signal, done, info, obs = env.step(action)
            if not isinstance(info, dict):
                info = {}
            policy.post_step(env, info, obs)
            ep_steps.append({
                "t": int(t),
                "pos": [int(env.agent_x), int(env.agent_y)],
                "action": int(action),
                "decided_action": rec.get("decided_action"),
                "executed_action": rec.get("executed_action"),
                "held": rec.get("held"),
                "top2_margin": rec.get("top2_margin"),
                "switch_margin": rec.get("switch_margin"),
                "harm_signal": float(harm_signal),
                "transition_type": str(info.get("transition_type", "none")),
                "health": float(getattr(env, "agent_health", 1.0)),
                "energy": float(getattr(env, "agent_energy", 1.0)),
                "z_world_norm": z_norm,
                "resource_field_true_argmax": true_argmax,
                "resource_field_pred_argmax": pred_argmax,
                "resource_field_pred_argmax_encoder_path": pred_argmax_enc,
                "resource_field_max": float(field.max()),
                "hazards": [list(h) for h in env.hazards],
                "resources": [list(r) for r in env.resources],
            })
            if done:
                done_cause = ("health_depleted"
                              if float(getattr(env, "agent_health", 1.0)) <= 0.0
                              else "step_limit")
                break
        positions = [tuple(s["pos"]) for s in ep_steps]
        episodes.append({
            "ep": int(ep),
            "arm": arm_id,
            "seed": int(seed),
            "initial_resources": initial_resources,
            "initial_hazards": initial_hazards,
            "steps": ep_steps,
            "realized_steps": len(ep_steps),
            "done_cause": done_cause,
            "cycle": bool(period2_cycle_present(positions)),
            "orbit_period": bounded_orbit_period(positions),
            "stationary_fraction": stationary_fraction(positions),
        })
    return episodes


# --------------------------------------------------------------------------------------
def _eval_cell(arm_id: str, policy: _TracedPolicy, env, seed: int, eval_eps: int, steps: int,
               contamination_off: bool = False) -> Dict[str, Any]:
    print(f"Seed {seed} Condition {RUNG_ID}:{arm_id}", flush=True)
    eval_row = evaluate_seed(policy, env, eval_eps, steps)
    ts = trace_stats(policy)
    row = {
        "cell_id": f"{arm_id}|seed{seed}",
        "arm_id": arm_id,
        "seed": int(seed),
        "contamination_off": bool(contamination_off),
        "foraging_competence": float(eval_row["foraging_competence"]),
        "competence_supra_floor": bool(eval_row["competence_supra_floor"]),
        "survival_horizon": float(eval_row["survival_horizon"]),
        "death_rate": float(eval_row["death_rate"]),
        "goal_reach_rate": float(eval_row["goal_reach_rate"]),
        "planning_depth": float(eval_row["planning_depth"]),
        "mean_contaminations": float(eval_row["mean_contaminations"]),
        "mean_episode_reward": float(eval_row["mean_episode_reward"]),
        "per_episode_resources": list(eval_row["per_episode_resources"]),
        "resources_per_100_survived_steps": (
            100.0 * float(eval_row["foraging_competence"]) / float(eval_row["survival_horizon"])
            if float(eval_row["survival_horizon"]) > 0 else 0.0),
        "trace": ts,
    }
    print(f"verdict: {'PASS' if eval_row['competence_supra_floor'] else 'FAIL'}", flush=True)
    return row


def _run_seed(seed: int, zworld_p0: int, p0: int, p1: int, ppo_eps: int, eval_eps: int,
              steps: int, rollout: int, log_eps: int, want_log: bool, dry_run: bool
              ) -> Dict[str, Any]:
    env_kwargs = x734._env_kwargs_for_rung(RUNG)
    contam_off_kwargs = dict(env_kwargs)
    contam_off_kwargs.update(CONTAMINATION_OFF_KWARGS)
    total_denom = p0 + p1
    cfg_slice = _off_reader_config_slice(dry_run, zworld_p0, p0, p1, ppo_eps, steps, rollout)

    # ---- the ONE training cell per seed: 978's field_loss_off, verbatim call sequence ------
    print(f"Seed {seed} Condition {RUNG_ID}:reader:warmup", flush=True)
    with arm_cell(seed, config_slice=cfg_slice, script_path=Path(__file__),
                  config_slice_declared=True, include_driver_script_in_hash=False) as cell:
        torch.manual_seed(seed)
        np.random.seed(seed)
        warm_env = x734._make_env(seed, env_kwargs)
        agent = x978._make_agent(warm_env)
        before = latent_stack_snapshot(agent)
        stats = x734._train_all_on_agent(
            agent, warm_env, seed=seed, p0_episodes=p0, p1_episodes=p1,
            steps_per_episode=steps, rung_id=RUNG_ID, total_denominator=total_denom,
            zworld_p0_episodes=zworld_p0,
            zworld_p0_env=(x734._make_env(seed, env_kwargs) if zworld_p0 > 0 else None),
            zworld_p0_dry_run=dry_run,
            zworld_p0_resource_field_weight=OFF_ARM_P0A_FIELD_WEIGHT,
        )
        guard = latent_stack_weight_delta(agent, before)
        p0a = (stats or {}).get("zworld_p0", {}) or {}

        print(f"Seed {seed} Condition {RUNG_ID}:reader:ppo", flush=True)
        torch.manual_seed(seed + 1000)
        np.random.seed(seed + 1000)
        _flat, probe_obs = x734._make_env(seed, env_kwargs).reset()
        z_dim = int(x737._agent_zworld(agent, probe_obs).shape[-1])
        action_dim = int(warm_env.action_dim)
        net = x734.PPOPolicyNet(in_dim=z_dim, action_dim=action_dim).to(DEVICE)
        opt = torch.optim.Adam(net.parameters(), lr=x734.PPO_LR)
        train_decomp = x808._train_ppo_decomposed(
            x734._make_env(seed, env_kwargs), net, opt,
            state_fn=(lambda od: x737._agent_zworld(agent, od)),
            on_reset=agent.reset,
            n_episodes=ppo_eps, rollout_episodes=rollout, steps_per_episode=steps,
            level_id=LEVEL_ID, w_consume=W_CONSUME, w_survival=W_SURVIVAL, seed=seed,
            total_denominator=ppo_eps,
        )
        training_row = {
            "cell_id": f"reader|seed{seed}",
            "arm_id": "reader_training",
            "seed": int(seed),
            "obs_dim": int(z_dim),
            "action_dim": int(action_dim),
            "zworld_weight_delta": guard,
            "p0a": {
                "ran": bool(p0a.get("p0a_ran")),
                "resource_field_weight": p0a.get("p0a_resource_field_weight"),
                "used_resource_field_head": bool(p0a.get("p0a_used_resource_field_head")),
                "used_proximity_head": p0a.get("p0a_used_proximity_head"),
            },
            "ppo_train_decomposition": train_decomp,
        }
        cell.stamp(training_row)
    _ZG.observe(agent)   # AFTER training stepped it -- reads the counters at call time

    # ENCODER AND READER FROZEN from here. Every eval arm starts from a deep copy of this
    # snapshot so arms are paired on identical post-training agent state.
    snapshot, n_detached = _copy_frozen(agent)
    training_row["snapshot_detached_nonleaf_tensors"] = int(n_detached)

    def _fresh_agent():
        return _copy_frozen(snapshot)[0]

    def _arm_seed_reset() -> None:
        torch.manual_seed(seed + 2000)
        np.random.seed(seed + 2000)

    eval_rows: Dict[str, Dict[str, Any]] = {}
    log_episodes: List[Dict[str, Any]] = []
    policies: Dict[str, Any] = {}   # factories for the log pass

    def _reader(arm_id: str, delta: float = 0.0):
        if arm_id == ARM_GREEDY:
            return ReaderEvalPolicy(net, _fresh_agent(), arm_id, "greedy", seed=seed)
        if arm_id in PERSIST_K:
            return ReaderEvalPolicy(net, _fresh_agent(), arm_id, "persist", k=PERSIST_K[arm_id],
                                    seed=seed)
        if arm_id == ARM_SWITCH:
            return ReaderEvalPolicy(net, _fresh_agent(), arm_id, "switch_cost", delta=delta,
                                    seed=seed)
        if arm_id == ARM_SAMPLE:
            return ReaderEvalPolicy(net, _fresh_agent(), arm_id, "sample",
                                    temperature=SAMPLE_TEMPERATURE, seed=seed)
        raise ValueError(arm_id)

    # (a) greedy -- also calibrates arm (d)'s delta from its own switch-margin distribution
    _arm_seed_reset()
    pol = _reader(ARM_GREEDY)
    eval_rows[ARM_GREEDY] = _eval_cell(ARM_GREEDY, pol, x734._make_env(seed, env_kwargs),
                                       seed, eval_eps, steps)
    _ZG.observe(pol.agent)   # an eval-side copy too, so the liveness block sees eval ticks
    switch_margins = [s["switch_margin"] for ep in pol.episodes for s in ep["steps"]
                      if s.get("switch_margin") is not None]
    delta = float(np.quantile(np.asarray(switch_margins), SWITCH_COST_MARGIN_QUANTILE)) \
        if switch_margins else 0.0
    switch_calibration = {
        "rule": f"quantile {SWITCH_COST_MARGIN_QUANTILE} of greedy per-step switch margins",
        "n_switch_steps": len(switch_margins),
        "delta": delta,
        "thin": bool(len(switch_margins) < SWITCH_MARGIN_MIN_N),
        "margin_stats": _quantiles(switch_margins),
    }
    policies[ARM_GREEDY] = lambda: _reader(ARM_GREEDY)

    # (b) (c) (d) (e)
    for arm_id in (ARM_K2, ARM_K4, ARM_SWITCH, ARM_SAMPLE):
        _arm_seed_reset()
        pol = _reader(arm_id, delta=delta)
        eval_rows[arm_id] = _eval_cell(arm_id, pol, x734._make_env(seed, env_kwargs),
                                       seed, eval_eps, steps)
        policies[arm_id] = (lambda aid=arm_id: _reader(aid, delta=delta))

    # anchors
    anchor_rows: Dict[str, Dict[str, Any]] = {}

    def _anchor(aid: str) -> AnchorPolicy:
        if aid == ANCHOR_RANDOM:
            return AnchorPolicy(RandomPolicy(seed), aid, k=1)
        if aid == ANCHOR_LVG:
            return AnchorPolicy(LocalViewGreedyPolicy(seed), aid, k=1)
        if aid == ANCHOR_LVG_K2:
            return AnchorPolicy(LocalViewGreedyPolicy(seed), aid, k=LVG_LATCH_K)
        raise ValueError(aid)

    for aid in ANCHOR_IDS:
        _arm_seed_reset()
        anchor_rows[aid] = _eval_cell(aid, _anchor(aid), x734._make_env(seed, env_kwargs),
                                      seed, eval_eps, steps)
        policies[aid] = (lambda a=aid: _anchor(a))

    # contamination-off sub-grid
    contam_rows: Dict[str, Dict[str, Any]] = {}
    for arm_id in CONTAM_OFF_ARMS:
        cid = arm_id + CONTAM_OFF_SUFFIX
        _arm_seed_reset()
        contam_rows[arm_id] = _eval_cell(cid, _reader(arm_id, delta=delta),
                                         x734._make_env(seed, contam_off_kwargs), seed,
                                         eval_eps, steps, contamination_off=True)
        policies[cid] = (lambda aid=arm_id: _reader(aid, delta=delta))

    # fishtank log pass (observational; fresh env at seed + 7777; NOT the scored data)
    if want_log:
        for cid, factory in policies.items():
            _arm_seed_reset()
            pol = factory()
            kw = contam_off_kwargs if cid.endswith(CONTAM_OFF_SUFFIX) else env_kwargs
            # A reader policy owns its own fresh copy; an anchor has no agent, so the head
            # series are read on ANOTHER fresh copy -- never on `snapshot` itself, which
            # every later _fresh_agent() deep-copies and which sense() would mutate.
            head_agent = getattr(pol, "agent", None)
            if head_agent is None:
                head_agent = _fresh_agent()
            log_episodes.extend(_log_pass(head_agent, pol, x734._make_env(seed + 7777, kw),
                                          cid, seed, log_eps, steps))

    return {
        "seed": int(seed),
        "training": training_row,
        "eval": eval_rows,
        "anchors": anchor_rows,
        "contamination_off": contam_rows,
        "switch_calibration": switch_calibration,
        "log_episodes": log_episodes,
        "env_config": env_kwargs,
        "contamination_off_env_config": contam_off_kwargs,
    }


# --------------------------------------------------------------------------------------
# Pure adjudication (contract-tested by --self-test)
# --------------------------------------------------------------------------------------
LABEL_PASS = "latch_abolishes_cycle_competence_flat_representational_deficit"
LABEL_GATING = "latch_restores_competence_gating_deficit_route_arc107_root_c"
LABEL_ENVELOPE = "latch_lift_within_undirected_envelope"
LABEL_NO_C1 = "latch_did_not_abolish_cycle_non_contributory"
LABEL_NOT_READY = "substrate_not_ready_requeue"
LABEL_ANCHOR = "anchor_replication_failed_greedy_supra_floor"
LABEL_NO_CYCLE = "greedy_cycle_absent_manipulation_unreachable"


def _adjudicate(gate_green: bool, gate_label: Optional[str], c1: bool, c2: bool,
                exceeds_envelope: bool, phenotype_absent: bool = False) -> Dict[str, str]:
    """(outcome, label, blanket direction, per-claim directions) from the pre-registered grid.

    `phenotype_absent` (red-team F4): on the no-cycle gate branch, True means the greedy reader
    showed NEITHER a cycle NOR a fixed point on a strict majority of seeds -- MECH-535's
    phenotype did not replicate, which counts AGAINST it rather than being filtered as silence.
    A stupor-only replication (fixed points, no cycle) reads `unknown` for MECH-535: half the
    phenotype, and the latch has nothing to dissociate.
    """
    if not gate_green:
        lbl = str(gate_label or LABEL_NOT_READY)
        m535 = "non_contributory"
        if lbl == LABEL_NO_CYCLE:
            m535 = "weakens" if phenotype_absent else "unknown"
        return {"outcome": "FAIL", "label": lbl, "direction": "non_contributory",
                "MECH-536": "non_contributory", "MECH-535": m535}
    if not c1:
        return {"outcome": "FAIL", "label": LABEL_NO_C1, "direction": "non_contributory",
                "MECH-536": "non_contributory", "MECH-535": "non_contributory"}
    if c2:
        return {"outcome": "PASS", "label": LABEL_PASS, "direction": "supports",
                "MECH-536": "supports", "MECH-535": "supports"}
    if exceeds_envelope:
        return {"outcome": "FAIL", "label": LABEL_GATING, "direction": "weakens",
                "MECH-536": "weakens", "MECH-535": "weakens"}
    return {"outcome": "FAIL", "label": LABEL_ENVELOPE, "direction": "mixed",
            "MECH-536": "mixed", "MECH-535": "unknown"}


# --------------------------------------------------------------------------------------
def run_experiment(seeds: List[int], dry_run: bool = False) -> Dict[str, Any]:
    zworld_p0 = DRY_RUN_ZWORLD_P0 if dry_run else ZWORLD_P0_EPISODES
    p0 = DRY_RUN_P0 if dry_run else P0_WARMUP_EPISODES
    p1 = DRY_RUN_P1 if dry_run else P1_REINFORCE_EPISODES
    ppo_eps = DRY_RUN_PPO if dry_run else P1_PPO_EPISODES
    eval_eps = DRY_RUN_EVAL if dry_run else EVAL_EPISODES
    steps = DRY_RUN_STEPS if dry_run else STEPS_PER_EPISODE
    rollout = DRY_RUN_ROLLOUT if dry_run else PPO_ROLLOUT_EPISODES
    log_eps = DRY_RUN_LOG_EPISODES if dry_run else LOG_EPISODES

    # The detectors are the instrument C1 routes on: refuse to run if they fail their own
    # contract (synthetic cases always; the 978 log when present on this box).
    st = self_test(verbose=False)
    if not st["ok"]:
        raise RuntimeError("detector/adjudication self-test FAILED: %s" % (st,))

    log_seeds = {seeds[0]} if seeds else set()
    seed_results = [
        _run_seed(s, zworld_p0, p0, p1, ppo_eps, eval_eps, steps, rollout, log_eps,
                  (s in log_seeds), dry_run)
        for s in seeds
    ]
    n_seeds = len(seed_results)

    # ---- per-arm summaries ----------------------------------------------------------------
    def _rows(kind: str, aid: str) -> List[Dict[str, Any]]:
        return [r[kind][aid] for r in seed_results]

    def _arm_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "n_seeds": len(rows),
            "mean_foraging_competence": _mean([x["foraging_competence"] for x in rows]),
            "per_seed_foraging_competence": [x["foraging_competence"] for x in rows],
            "n_seeds_supra_floor": int(sum(1 for x in rows if x["competence_supra_floor"])),
            "mean_survival_horizon": _mean([x["survival_horizon"] for x in rows]),
            "per_seed_survival_horizon": [x["survival_horizon"] for x in rows],
            "per_seed_death_rate": [x["death_rate"] for x in rows],
            "mean_cycle_incidence": _mean([x["trace"]["cycle_incidence"] for x in rows]),
            "per_seed_cycle_incidence": [x["trace"]["cycle_incidence"] for x in rows],
            "per_seed_fixed_point_incidence": [x["trace"]["fixed_point_incidence"] for x in rows],
            "per_seed_bounded_orbit_incidence": [x["trace"]["bounded_orbit_incidence"] for x in rows],
            "per_seed_longer_orbit_incidence": [x["trace"]["longer_orbit_incidence"] for x in rows],
            "per_seed_orbit_period_histogram": [x["trace"]["orbit_period_histogram"] for x in rows],
            "per_seed_mean_unique_cells": [x["trace"]["mean_unique_cells"] for x in rows],
            "per_seed_resources_per_100_survived_steps": [
                x["resources_per_100_survived_steps"] for x in rows],
        }

    per_arm = {aid: _arm_summary(_rows("eval", aid)) for aid in EVAL_ARM_IDS}
    per_anchor = {aid: _arm_summary(_rows("anchors", aid)) for aid in ANCHOR_IDS}
    per_contam = {aid + CONTAM_OFF_SUFFIX: _arm_summary(_rows("contamination_off", aid))
                  for aid in CONTAM_OFF_ARMS}

    greedy = per_arm[ARM_GREEDY]
    rw = per_anchor[ANCHOR_RANDOM]
    lvg = per_anchor[ANCHOR_LVG]
    lvg_k2 = per_anchor[ANCHOR_LVG_K2]

    # ---- readiness gate (evidence run: recorded numerically; the indexer recomputes met) ---
    enc_deltas = [(float(((r["training"].get("zworld_weight_delta") or {}).get(
        "world_encoder_max_abs_delta", 0.0)) or 0.0), r["training"]["cell_id"])
        for r in seed_results]
    enc_measured, enc_cell = min(enc_deltas) if enc_deltas else (0.0, "(none)")
    lvg_measured = min(lvg["per_seed_foraging_competence"]) if n_seeds else 0.0
    lvg_cell = f"{ANCHOR_LVG}|seed{seeds[int(np.argmin(lvg['per_seed_foraging_competence']))]}" \
        if n_seeds else "(none)"
    greedy_max_comp = max(greedy["per_seed_foraging_competence"]) if n_seeds else 0.0
    greedy_max_cyc = max(greedy["per_seed_cycle_incidence"]) if n_seeds else 0.0
    preconditions = [
        {"name": "zworld_encoder_trained_in_p0", "kind": "readiness",
         "description": "the P0a SD-070 recipe must actually move split_encoder.world_encoder "
                        "(worst seed)",
         "control": "latent_stack weight delta over the warmup vs the family's guard floor",
         "measured": enc_measured, "threshold": float(ZWORLD_DELTA_FLOOR), "direction": "lower",
         "offending_cell": enc_cell, "met": bool(enc_measured >= ZWORLD_DELTA_FLOOR)},
        {"name": "d3_local_view_greedy_clears_floor", "kind": "readiness",
         "description": "the env is floor-achievable FROM THE 5x5 LOCAL VIEW (worst seed); "
                        "closes the 732a privileged-oracle confound",
         "control": "local_view_greedy worst-seed res/ep vs the 1.0 competence floor",
         "measured": float(lvg_measured), "threshold": COMPETENCE_FLOOR, "direction": "lower",
         "offending_cell": lvg_cell, "met": bool(lvg_measured >= COMPETENCE_FLOOR)},
        {"name": "greedy_reader_subfloor_replication", "kind": "readiness",
         "description": "arm (a) must NOT clear the 1.0 floor on any seed -- the 978/948/813 "
                        "direction-blind premise. If it does the contrast is uninterpretable",
         "control": "greedy_argmax best-seed res/ep vs the floor",
         "measured": float(greedy_max_comp), "threshold": COMPETENCE_FLOOR, "direction": "upper",
         "comparator": "<", "met": bool(greedy_max_comp < COMPETENCE_FLOOR)},
        {"name": "greedy_cycle_present_on_some_seed", "kind": "readiness",
         "description": "the manipulation reaches the DV only through cycling episodes; at "
                        "least one seed's greedy reader must cycle on >= 25% of eval episodes",
         "control": "greedy_argmax best-seed cycle_incidence vs CYCLE_PRESENCE_FLOOR",
         "measured": float(greedy_max_cyc), "threshold": float(CYCLE_PRESENCE_FLOOR),
         "direction": "lower", "met": bool(greedy_max_cyc >= CYCLE_PRESENCE_FLOOR)},
    ]
    gate_green = all(p["met"] for p in preconditions)
    gate_label: Optional[str] = None
    if not gate_green:
        first = next(p for p in preconditions if not p["met"])
        gate_label = {
            "zworld_encoder_trained_in_p0": LABEL_NOT_READY,
            "d3_local_view_greedy_clears_floor": LABEL_NOT_READY,
            "greedy_reader_subfloor_replication": LABEL_ANCHOR,
            "greedy_cycle_present_on_some_seed": LABEL_NO_CYCLE,
        }[first["name"]]

    # ---- per-seed conditioning ----------------------------------------------------------
    cycle_present = [bool(c >= CYCLE_PRESENCE_FLOOR) for c in greedy["per_seed_cycle_incidence"]]
    cp_idx = [i for i, ok in enumerate(cycle_present) if ok]
    n_cp = len(cp_idx)

    # ---- C1 ---------------------------------------------------------------------------
    def _c1_arm(aid: str) -> Dict[str, Any]:
        cyc = per_arm[aid]["per_seed_cycle_incidence"]
        ok = [bool(cyc[i] <= CYCLE_INCIDENCE_CEILING) for i in cp_idx]
        n_ok = int(sum(ok))
        passed = bool(n_cp >= 1 and n_ok >= _strict_majority(n_cp))
        return {"arm_id": aid, "n_cycle_present_seeds": n_cp, "n_seeds_abolished": n_ok,
                "per_cycle_present_seed_cycle_incidence": [cyc[i] for i in cp_idx],
                "passed": passed}
    c1_by_arm = {aid: _c1_arm(aid) for aid in LATCH_ARMS + [ARM_SAMPLE]}
    c1 = bool(all(c1_by_arm[a]["passed"] for a in VERDICT_ARMS))

    # ---- C2 ---------------------------------------------------------------------------
    greedy_comp = greedy["per_seed_foraging_competence"]
    greedy_seed_sd = float(pstdev(greedy_comp)) if len(greedy_comp) > 1 else 0.0

    def _c2_arm(aid: str, rows_comp: List[float]) -> Dict[str, Any]:
        lifts = [float(rows_comp[i] - greedy_comp[i]) for i in range(n_seeds)]
        lift_sd = float(pstdev(lifts)) if len(lifts) > 1 else 0.0
        floor = float(C2_EFFECT_FLOOR)
        supra = [bool(c >= COMPETENCE_FLOOR) for c in rows_comp]
        rise = [bool(l >= floor) for l in lifts]
        # red-team F1: crossing the 1.0 floor counts as a RISE only when the lift itself clears
        # the effect floor -- otherwise one resource in 20 episodes on a fixed-point seed flips
        # supports to weakens. A seed fails C2 iff its lift is a genuine rise.
        per_seed_pass = [bool(not rise[i]) for i in range(n_seeds)]
        passed = bool(n_seeds >= 1 and all(per_seed_pass))
        return {"arm_id": aid, "per_seed_lift": lifts, "lift_sd": lift_sd,
                "effect_floor": floor, "greedy_seed_sd": greedy_seed_sd,
                "per_seed_supra_floor": supra, "per_seed_rise": rise,
                "per_seed_supra_floor_with_rise": [bool(supra[i] and rise[i]) for i in range(n_seeds)],
                "per_seed_pass": per_seed_pass, "passed": passed}
    c2_by_arm = {aid: _c2_arm(aid, per_arm[aid]["per_seed_foraging_competence"])
                 for aid in LATCH_ARMS + [ARM_SAMPLE]}
    c2 = bool(all(c2_by_arm[a]["passed"] for a in VERDICT_ARMS))

    # random-walk envelope: does a verdict arm beat undirected motion?
    def _exceeds_envelope(aid: str) -> Dict[str, Any]:
        comp = per_arm[aid]["per_seed_foraging_competence"]
        rise = c2_by_arm[aid]["per_seed_rise"]
        # red-team F1: both envelope tests carry the effect guard -- beating random_walk by a
        # hair, or touching 1.0, is not a restored direction unless the lift is a genuine rise.
        above_rw = [bool(rise[i] and comp[i] > rw["per_seed_foraging_competence"][i])
                    for i in range(n_seeds)]
        supra = [bool(rise[i] and comp[i] >= COMPETENCE_FLOOR) for i in range(n_seeds)]
        exceeds = bool(n_seeds >= 1 and (sum(above_rw) >= _strict_majority(n_seeds) or any(supra)))
        return {"arm_id": aid, "per_seed_above_random_walk_with_rise": above_rw,
                "per_seed_supra_floor_with_rise": supra, "exceeds": exceeds}
    env_by_arm = {aid: _exceeds_envelope(aid) for aid in LATCH_ARMS + [ARM_SAMPLE]}
    exceeds_envelope = bool(any(env_by_arm[a]["exceeds"] for a in VERDICT_ARMS))

    # MECH-535 phenotype on the greedy reader: cycle OR fixed point, per seed (red-team F4).
    greedy_phenotype = [min(1.0, greedy["per_seed_cycle_incidence"][i]
                            + greedy["per_seed_fixed_point_incidence"][i]) for i in range(n_seeds)]
    phenotype_absent = bool(n_seeds >= 1 and sum(
        1 for v in greedy_phenotype if v < PHENOTYPE_INCIDENCE_FLOOR) >= _strict_majority(n_seeds))
    verdict = _adjudicate(gate_green, gate_label, c1, c2, exceeds_envelope, phenotype_absent)
    outcome, label = verdict["outcome"], verdict["label"]

    # ---- reported, never gating -----------------------------------------------------------
    c3_retained = [bool(lvg_k2["per_seed_foraging_competence"][i]
                        >= LVG_LATCH_RETENTION * lvg["per_seed_foraging_competence"][i])
                   for i in range(n_seeds)]
    c3 = bool(n_seeds >= 1 and all(c3_retained))
    contam_greedy = per_contam[ARM_GREEDY + CONTAM_OFF_SUFFIX]
    contam_k2 = per_contam[ARM_K2 + CONTAM_OFF_SUFFIX]
    sm_lifts = [float(contam_k2["per_seed_foraging_competence"][i]
                      - contam_greedy["per_seed_foraging_competence"][i]) for i in range(n_seeds)]
    sm_floor = float(C2_EFFECT_FLOOR)
    c2_survival_matched = bool(n_seeds >= 1 and all(l < sm_floor for l in sm_lifts))
    orbit_replacement = {
        aid: {"per_cycle_present_seed_bounded_orbit_incidence":
              [per_arm[aid]["per_seed_bounded_orbit_incidence"][i] for i in cp_idx],
              "per_cycle_present_seed_longer_orbit_incidence":
              [per_arm[aid]["per_seed_longer_orbit_incidence"][i] for i in cp_idx],
              "cycle_replaced_by_longer_orbit": bool(any(
                  per_arm[aid]["per_seed_longer_orbit_incidence"][i] >= CYCLE_PRESENCE_FLOOR
                  for i in cp_idx))}
        for aid in LATCH_ARMS + [ARM_SAMPLE]}
    caveats: List[str] = []
    if not c3:
        caveats.append("latch_costs_good_representation: local_view_greedy_persist_k2 retained "
                       "< %.0f%% of local_view_greedy on some seed -- MECH-536's 'protective, "
                       "not necessary' framing needs a caveat" % (100 * LVG_LATCH_RETENTION))
    if any(per_arm[a]["per_seed_fixed_point_incidence"][i] >= 0.5
           for a in VERDICT_ARMS for i in cp_idx):
        caveats.append("latch_converted_cycle_to_fixed_point: on a cycle-present seed a verdict "
                       "arm is a boundary fixed point on >= 50% of episodes -- ambitendency became "
                       "stupor (the 'straight runs into the boundary' reading); survival is then "
                       "stationary, not exposure")
    if any(orbit_replacement[a]["cycle_replaced_by_longer_orbit"] for a in VERDICT_ARMS):
        caveats.append("cycle_replaced_by_longer_orbit: a verdict arm shows a bounded orbit of "
                       "period > 2 on a cycle-present seed -- the two-cycle was lengthened, "
                       "not dissolved")
    switch_family_agrees = bool(c1_by_arm[ARM_SWITCH]["passed"] == c1
                                and c2_by_arm[ARM_SWITCH]["passed"] == c2)
    replication_978 = [{
        "seed": s, "greedy_competence": greedy_comp[i],
        "expected_978_off": EXPECTED_978_OFF_COMPETENCE.get(s),
        "greedy_survival": greedy["per_seed_survival_horizon"][i],
        "expected_978_off_survival": EXPECTED_978_OFF_SURVIVAL.get(s),
        "competence_abs_diff": (abs(greedy_comp[i] - EXPECTED_978_OFF_COMPETENCE[s])
                                if s in EXPECTED_978_OFF_COMPETENCE else None),
    } for i, s in enumerate(seeds)]

    # ---- non-degeneracy -------------------------------------------------------------------
    latch_cyc_groups = [[greedy["per_seed_cycle_incidence"][i]]
                        + [per_arm[a]["per_seed_cycle_incidence"][i] for a in VERDICT_ARMS]
                        for i in cp_idx]
    deg = check_degeneracy({"cycle_incidence_greedy_vs_latch": {"groups": latch_cyc_groups}}) \
        if latch_cyc_groups else {"non_degenerate": False,
                                  "degeneracy_reason": "no cycle-present seed",
                                  "degenerate_metrics": {"cycle_incidence_greedy_vs_latch":
                                                         "no cycle-present seed"}}
    c1_nd = bool(gate_green and n_cp >= 1 and deg["non_degenerate"])
    # red-team F3: survival alone is satisfied by a wall-press or a lengthened lethal orbit;
    # opportunity means the latched agent also MOVED THROUGH more of the grid than the two-cycle.
    opportunity = [all(per_arm[a]["per_seed_survival_horizon"][i]
                       > greedy["per_seed_survival_horizon"][i]
                       and per_arm[a]["per_seed_mean_unique_cells"][i]
                       >= greedy["per_seed_mean_unique_cells"][i] + UNIQUE_CELLS_MIN_GAIN
                       for a in VERDICT_ARMS)
                   for i in cp_idx]
    c2_nd = bool(gate_green and n_cp >= 1 and all(opportunity))
    non_degenerate = bool(gate_green and c1_nd and c2_nd)
    degeneracy_reason = "" if non_degenerate else (
        "gate red" if not gate_green else
        ("C1: " + deg["degeneracy_reason"] if not c1_nd else
         "C2: on some cycle-present seed a verdict arm did not both outlive greedy AND visit >= %d "
         "more cells -- the latch created no foraging opportunity (wall-press or lengthened orbit), "
         "so 'flat' measured nothing about direction" % UNIQUE_CELLS_MIN_GAIN))

    criteria = [
        {"name": "C1_latch_abolishes_cycle", "load_bearing": True, "passed": c1,
         "description": "on the cycle-present seeds a strict majority have cycle_incidence <= "
                        "%.2f on EACH verdict arm (persist_k2, persist_k4)" % CYCLE_INCIDENCE_CEILING},
        {"name": "C2_competence_flat_under_latch", "load_bearing": True, "passed": c2,
         "description": "on EVERY seed and each verdict arm the per-seed lift over greedy_argmax "
                        "is below the %.2f res/ep effect floor; clearing the %.1f competence "
                        "floor counts as a rise only with such a lift (red-team F1/F2)"
                        % (C2_EFFECT_FLOOR, COMPETENCE_FLOOR)},
        {"name": "C3_latch_harmless_on_good_representation", "load_bearing": False, "passed": c3,
         "description": "local_view_greedy_persist_k2 retains >= %.0f%% of local_view_greedy's "
                        "res/ep on every seed (pre-registered expectation; reported)"
                        % (100 * LVG_LATCH_RETENTION)},
        {"name": "C2_survival_matched_contamination_off", "load_bearing": False,
         "passed": c2_survival_matched,
         "description": "under contamination_spread=0 persist_k2 stays sub-floor and its lift over "
                        "greedy_argmax is below the same effect-floor rule (reported)"},
    ]
    combination_rule = (
        "Verdict = C1 AND C2 on the verdict arms persist_k2 AND persist_k4, under a green gate. "
        "PASS iff both. C1 false -> non_contributory (the manipulation did not reach the DV). "
        "C1 true, C2 false -> weakens (gating deficit; ARC-107 root C) iff a verdict arm's "
        "competence beats random_walk on a strict majority of seeds or clears 1.0 on any seed; "
        "otherwise mixed (a measured rise that stays inside the undirected envelope). switch_cost "
        "and stochastic_sample carry their own C1/C2 flags and are reported, never adjudicated; "
        "C3 and C2_survival_matched are reported."
    )
    non_degenerate_flags = {
        "C1_latch_abolishes_cycle": c1_nd,
        "C2_competence_flat_under_latch": c2_nd,
        "C3_latch_harmless_on_good_representation": bool(gate_green),
        "C2_survival_matched_contamination_off": bool(gate_green and n_cp >= 1),
    }

    metrics = {
        "competence_floor": COMPETENCE_FLOOR,
        "cycle_incidence_ceiling": CYCLE_INCIDENCE_CEILING,
        "cycle_presence_floor": CYCLE_PRESENCE_FLOOR,
        "n_seeds": n_seeds,
        "n_cycle_present_seeds": n_cp,
        "cycle_present_seeds": [seeds[i] for i in cp_idx],
        "greedy_mean_foraging_competence": greedy["mean_foraging_competence"],
        "greedy_per_seed_cycle_incidence": greedy["per_seed_cycle_incidence"],
        "greedy_per_seed_fixed_point_incidence": greedy["per_seed_fixed_point_incidence"],
        "persist_k2_mean_foraging_competence": per_arm[ARM_K2]["mean_foraging_competence"],
        "persist_k4_mean_foraging_competence": per_arm[ARM_K4]["mean_foraging_competence"],
        "switch_cost_mean_foraging_competence": per_arm[ARM_SWITCH]["mean_foraging_competence"],
        "stochastic_sample_mean_foraging_competence": per_arm[ARM_SAMPLE]["mean_foraging_competence"],
        "persist_k2_per_seed_cycle_incidence": per_arm[ARM_K2]["per_seed_cycle_incidence"],
        "persist_k4_per_seed_cycle_incidence": per_arm[ARM_K4]["per_seed_cycle_incidence"],
        "switch_cost_per_seed_cycle_incidence": per_arm[ARM_SWITCH]["per_seed_cycle_incidence"],
        "stochastic_sample_per_seed_cycle_incidence": per_arm[ARM_SAMPLE]["per_seed_cycle_incidence"],
        "persist_k2_effect_floor": c2_by_arm[ARM_K2]["effect_floor"],
        "persist_k4_effect_floor": c2_by_arm[ARM_K4]["effect_floor"],
        "greedy_seed_sd": greedy_seed_sd,
        "random_walk_per_seed_competence": rw["per_seed_foraging_competence"],
        "local_view_greedy_worst_seed_competence": float(lvg_measured),
        "local_view_greedy_persist_k2_per_seed_competence": lvg_k2["per_seed_foraging_competence"],
        "contamination_off_greedy_per_seed_competence": contam_greedy["per_seed_foraging_competence"],
        "contamination_off_persist_k2_per_seed_competence": contam_k2["per_seed_foraging_competence"],
        "contamination_off_greedy_per_seed_cycle_incidence": contam_greedy["per_seed_cycle_incidence"],
        "switch_cost_delta_per_seed": [r["switch_calibration"]["delta"] for r in seed_results],
        "switch_cost_n_switch_steps_per_seed": [r["switch_calibration"]["n_switch_steps"]
                                               for r in seed_results],
        # red-team F7: with no greedy switches delta=0 and arm (d) IS arm (a) on that seed.
        "switch_cost_degenerate_seeds": [seeds[i] for i, r in enumerate(seed_results)
                                         if r["switch_calibration"]["n_switch_steps"] == 0],
        "greedy_per_seed_phenotype_incidence": greedy_phenotype,
        "mech535_phenotype_absent": phenotype_absent,
    }

    interpretation = {
        "label": label,
        "hypothesis_verdict": verdict,
        "combination_rule": combination_rule,
        "preconditions": preconditions,
        "gate_green": gate_green,
        "criteria": criteria,
        "criteria_non_degenerate": non_degenerate_flags,
        "c1_by_arm": c1_by_arm,
        "c2_by_arm": c2_by_arm,
        "random_walk_envelope_by_arm": env_by_arm,
        "exceeds_envelope": exceeds_envelope,
        "cycle_present_per_seed": cycle_present,
        "orbit_replacement": orbit_replacement,
        "latch_family": {"switch_cost_agrees_with_verdict_arms": switch_family_agrees,
                         "switch_calibration_per_seed": [r["switch_calibration"]
                                                         for r in seed_results]},
        "gflag_0131_comparator": {
            "arm": ARM_SAMPLE,
            "c1": c1_by_arm[ARM_SAMPLE]["passed"], "c2": c2_by_arm[ARM_SAMPLE]["passed"],
            "exceeds_envelope": env_by_arm[ARM_SAMPLE]["exceeds"],
            "note": "reported alongside, not a gate: noise as de facto commitment perturbation"},
        "replication_978_off_arm": replication_978,
        "recorded_checks": [
            # red-team F5: contamination_spread=0 zeroes contamination_view = world_state[175:200],
            # an input the reader was trained with live; the sub-grid greedy cycle is asserted,
            # not guaranteed. Recorded so the survival-matched contrast can be read honestly.
            {"name": "contamination_off_greedy_cycle_present",
             "measured": (max(contam_greedy["per_seed_cycle_incidence"]) if n_seeds else 0.0),
             "threshold": float(CYCLE_PRESENCE_FLOOR), "direction": "lower", "gating": False,
             "met": bool(n_seeds and max(contam_greedy["per_seed_cycle_incidence"])
                         >= CYCLE_PRESENCE_FLOOR)},
            {"name": "switch_cost_delta_calibrated_every_seed", "gating": False,
             "measured": (min(r["switch_calibration"]["n_switch_steps"] for r in seed_results)
                          if seed_results else 0),
             "threshold": int(SWITCH_MARGIN_MIN_N), "direction": "lower",
             "met": bool(seed_results and min(r["switch_calibration"]["n_switch_steps"]
                                              for r in seed_results) >= SWITCH_MARGIN_MIN_N)},
        ],
        "caveats": caveats,
        "null_reading": {
            LABEL_NO_C1: "a pure latch that leaves a period-2 orbit standing is an instrumentation "
                         "defect in the wrapper (see WHAT A PURE LATCH DOES BY CONSTRUCTION), not a "
                         "finding about MECH-536",
            LABEL_ENVELOPE: "the latch raised competence, but no further than undirected motion "
                            "(random_walk) achieves; consistent with survival opportunity, silent "
                            "on direction -- governance reads the contamination-off sub-grid",
            LABEL_GATING: "the latch restored competence beyond the undirected envelope: direction "
                          "was recoverable from the representation once the actor stopped "
                          "re-deciding -- a gating deficit; route to ARC-107 root C",
            LABEL_NO_CYCLE: "no seed's greedy reader cycled, so the latch had nothing to abolish; "
                            "the 978 phenotype did not replicate on these seeds at this eval",
        },
    }

    def _fmt(v: Sequence[float]) -> str:
        return "/".join(f"{x:.2f}" for x in v)

    lines = []
    for aid in EVAL_ARM_IDS:
        a = per_arm[aid]
        lines.append(f"| {aid} | {a['mean_foraging_competence']:.4f} | "
                     f"{_fmt(a['per_seed_foraging_competence'])} | "
                     f"{_fmt(a['per_seed_cycle_incidence'])} | "
                     f"{_fmt(a['per_seed_fixed_point_incidence'])} | "
                     f"{_fmt(a['per_seed_survival_horizon'])} |")
    for aid in ANCHOR_IDS:
        a = per_anchor[aid]
        lines.append(f"| {aid} (anchor) | {a['mean_foraging_competence']:.4f} | "
                     f"{_fmt(a['per_seed_foraging_competence'])} | "
                     f"{_fmt(a['per_seed_cycle_incidence'])} | "
                     f"{_fmt(a['per_seed_fixed_point_incidence'])} | "
                     f"{_fmt(a['per_seed_survival_horizon'])} |")
    for cid, a in per_contam.items():
        lines.append(f"| {cid} | {a['mean_foraging_competence']:.4f} | "
                     f"{_fmt(a['per_seed_foraging_competence'])} | "
                     f"{_fmt(a['per_seed_cycle_incidence'])} | "
                     f"{_fmt(a['per_seed_fixed_point_incidence'])} | "
                     f"{_fmt(a['per_seed_survival_horizon'])} |")
    summary_markdown = f"""# {QUEUE_ID} -- MECH-536 eval-time action-persistence discriminator

Outcome: **{outcome}** ({label}) -- MECH-536: {verdict['MECH-536']}, MECH-535: {verdict['MECH-535']}

| arm | mean res/ep | per-seed res/ep | per-seed cycle incidence | per-seed fixed-point incidence | per-seed survival |
|---|---|---|---|---|---|
{chr(10).join(lines)}

Cycle-present seeds (greedy cycle_incidence >= {CYCLE_PRESENCE_FLOOR}): {metrics['cycle_present_seeds']} of {seeds}.
C1 {'PASS' if c1 else 'FAIL'}; C2 {'PASS' if c2 else 'FAIL'} (effect floor {C2_EFFECT_FLOOR:.2f}; k2 lifts {[round(l, 3) for l in c2_by_arm[ARM_K2]['per_seed_lift']]}, k4 lifts {[round(l, 3) for l in c2_by_arm[ARM_K4]['per_seed_lift']]});
exceeds random-walk envelope: {exceeds_envelope}. C3 (latch harmless on lvg) {'PASS' if c3 else 'FAIL'};
C2 survival-matched (contamination off) {'PASS' if c2_survival_matched else 'FAIL'}.
switch_cost delta per seed: {[round(d, 4) for d in metrics['switch_cost_delta_per_seed']]}. Caveats: {caveats or 'none'}.

{combination_rule}

One trained reader per seed (978's field_loss_off cell, imported recipe); every arm is the SAME
reader under a different EXECUTION rule on a deep copy of the post-training snapshot and a fresh
env at the same seed. A fishtank episode-log companion (first seed, every arm, per-episode `arm`
badge) is written alongside; it is an observational pass, not the scored data.
"""

    first_env = seed_results[0]["env_config"] if seed_results else {}
    episode_log = {
        "experiment_type": EXPERIMENT_TYPE,
        "phase": "mech536_eval_persistence_arms",
        "toroidal": bool(first_env.get("toroidal", False)),
        "env_config": first_env,
        "contamination_off_env_config": (seed_results[0]["contamination_off_env_config"]
                                         if seed_results else {}),
        "arm_ids": list(EVAL_ARM_IDS) + [a + CONTAM_OFF_SUFFIX for a in CONTAM_OFF_ARMS]
        + list(ANCHOR_IDS),
        "seeds": [{"seed": r["seed"], "episodes": r["log_episodes"]}
                  for r in seed_results if r["log_episodes"]],
    }

    return {
        "status": outcome,
        "outcome": outcome,
        "overall_pass": bool(outcome == "PASS"),
        "metrics": metrics,
        "interpretation": interpretation,
        "summary_markdown": summary_markdown,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": verdict["direction"],
        "evidence_direction_per_claim": {"MECH-536": verdict["MECH-536"],
                                         "MECH-535": verdict["MECH-535"]},
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "experiment_type": EXPERIMENT_TYPE,
        "sleep_driver_pattern": "none",
        "non_degenerate": non_degenerate,
        "degeneracy_reason": degeneracy_reason,
        "degenerate_metrics": deg.get("degenerate_metrics", {}),
        # arm_results = the fingerprinted TRAINING cell per seed (reuse-eligible mint);
        # eval_results = every eval cell on that reader; anchor_results = the anchors.
        "arm_results": [r["training"] for r in seed_results],
        "eval_results": [r["eval"][aid] for r in seed_results for aid in EVAL_ARM_IDS]
        + [r["contamination_off"][aid] for r in seed_results for aid in CONTAM_OFF_ARMS],
        "anchor_results": [r["anchors"][aid] for r in seed_results for aid in ANCHOR_IDS],
        "per_arm": per_arm,
        "per_anchor": per_anchor,
        "per_contamination_off_arm": per_contam,
        "rung_id": RUNG_ID,
        "level_id": LEVEL_ID,
        "episode_log": episode_log,
        "self_test": st,
        "supersedes": None,
    }


# --------------------------------------------------------------------------------------
# Self-test: detectors (synthetic + the 978 log when present) and the adjudication grid.
# --------------------------------------------------------------------------------------
def self_test(verbose: bool = True) -> Dict[str, Any]:
    out: Dict[str, Any] = {"ok": True, "checks": []}

    def _chk(name: str, cond: bool, detail: Any = None) -> None:
        out["checks"].append({"name": name, "ok": bool(cond), "detail": detail})
        if not cond:
            out["ok"] = False
        if verbose:
            print(f"[self-test] {'OK  ' if cond else 'FAIL'} {name} {detail if detail is not None else ''}",
                  flush=True)

    A, B, C, D = (6, 8), (6, 9), (6, 10), (7, 10)
    two_cycle = [A, B] * 6
    wall = [A] * 12
    straight_then_cycle = [(2, 4), (2, 5), (2, 6), (2, 7)] + [B, A] * 5
    latched_two_cycle = [A, B, C, B] * 3          # k=2 latch of a two-cycle: period 4
    short_alt = [A, B, A, B, A]                   # 5 positions: below CYCLE_MIN_LEN
    wander = [A, B, C, D, (7, 11), (8, 11), (8, 10), (9, 10), (9, 9), (10, 9)]
    _chk("period2 fires on a two-cell cycle", period2_cycle_present(two_cycle))
    _chk("period2 silent on a wall-press", not period2_cycle_present(wall))
    _chk("period2 fires after a straight-run transient", period2_cycle_present(straight_then_cycle))
    _chk("period2 silent on a k=2-latched (period-4) orbit", not period2_cycle_present(latched_two_cycle))
    _chk("period2 silent below CYCLE_MIN_LEN", not period2_cycle_present(short_alt))
    _chk("period2 silent on a wander", not period2_cycle_present(wander))
    # red-team F6 mirror: a run of constant executed action moves the agent monotonically (or
    # holds it), so no length-6 window can alternate two distinct cells.
    held_runs = [A, B, C, (6, 11), (6, 12), (6, 13)] + [(6, 13)] * 6
    _chk("period2 silent on a held-action straight run then wall", not period2_cycle_present(held_runs))
    _chk("orbit period 2 on a two-cycle", bounded_orbit_period(two_cycle) == 2)
    _chk("orbit period 4 on a latched two-cycle", bounded_orbit_period(latched_two_cycle) == 4)
    _chk("orbit None on a wall-press", bounded_orbit_period(wall) is None)
    _chk("orbit None on a wander", bounded_orbit_period(wander) is None)
    _chk("wall-press is a fixed point", stationary_fraction(wall) >= FIXED_POINT_STATIONARY_FRAC)
    _chk("two-cycle is not a fixed point", stationary_fraction(two_cycle) < FIXED_POINT_STATIONARY_FRAC)

    logs = sorted(glob.glob(EXQ978_EPISODE_LOG_GLOB))
    if logs:
        try:
            d = json.load(open(logs[0], encoding="utf-8"))
            eps = [e for s in d.get("seeds", []) for e in s.get("episodes", [])]
            n_cyc = sum(1 for e in eps if period2_cycle_present([tuple(st["pos"]) for st in e["steps"]]))
            n_fp = sum(1 for e in eps if stationary_fraction([tuple(st["pos"]) for st in e["steps"]])
                       >= FIXED_POINT_STATIONARY_FRAC)
            fp_cyc = sum(1 for e in eps
                         if stationary_fraction([tuple(st["pos"]) for st in e["steps"]])
                         >= FIXED_POINT_STATIONARY_FRAC
                         and period2_cycle_present([tuple(st["pos"]) for st in e["steps"]]))
            _chk("978 log: detector fires on exactly %d/%d episodes" % (EXQ978_LOG_EXPECTED_CYCLING, len(eps)),
                 n_cyc == EXQ978_LOG_EXPECTED_CYCLING, {"fired": n_cyc, "n": len(eps)})
            _chk("978 log: exactly %d fixed points" % EXQ978_LOG_EXPECTED_FIXED_POINTS,
                 n_fp == EXQ978_LOG_EXPECTED_FIXED_POINTS, {"fixed_points": n_fp})
            _chk("978 log: no fixed point fires the cycle detector", fp_cyc == 0, {"both": fp_cyc})
            out["exq978_log"] = {"path": logs[0], "n_episodes": len(eps), "cycling": n_cyc,
                                 "fixed_points": n_fp}
        except Exception as exc:  # a malformed local copy must not block a cloud run
            out["exq978_log"] = {"path": logs[0], "error": repr(exc)}
            _chk("978 log readable", False, repr(exc))
    else:
        out["exq978_log"] = {"path": None, "note": "978 episode log not present on this box; "
                                                   "synthetic cases only"}
        if verbose:
            print("[self-test] 978 episode log not present on this box -- synthetic cases only",
                  flush=True)

    grid = [
        ((False, LABEL_NO_CYCLE, True, True, False), ("FAIL", LABEL_NO_CYCLE, "non_contributory", "unknown")),
        ((False, LABEL_NO_CYCLE, True, True, False, True), ("FAIL", LABEL_NO_CYCLE, "non_contributory", "weakens")),
        ((False, LABEL_ANCHOR, True, True, False, True), ("FAIL", LABEL_ANCHOR, "non_contributory", "non_contributory")),
        ((False, LABEL_ANCHOR, True, True, False), ("FAIL", LABEL_ANCHOR, "non_contributory", "non_contributory")),
        ((True, None, False, True, False), ("FAIL", LABEL_NO_C1, "non_contributory", "non_contributory")),
        ((True, None, True, True, False), ("PASS", LABEL_PASS, "supports", "supports")),
        ((True, None, True, True, True), ("PASS", LABEL_PASS, "supports", "supports")),
        ((True, None, True, False, True), ("FAIL", LABEL_GATING, "weakens", "weakens")),
        ((True, None, True, False, False), ("FAIL", LABEL_ENVELOPE, "mixed", "unknown")),
    ]
    for args, exp in grid:
        v = _adjudicate(*args)
        got = (v["outcome"], v["label"], v["MECH-536"], v["MECH-535"])
        _chk("adjudicate%s" % (args,), got == exp, {"got": got, "expected": exp})
    # invariant: a weakens verdict is only ever emitted with C1 true, C2 false, envelope exceeded
    bad = [(g, c1, c2, e) for g in (True, False) for c1 in (True, False) for c2 in (True, False)
           for e in (True, False)
           if _adjudicate(g, None, c1, c2, e)["MECH-536"] == "weakens"
           and not (g and c1 and (not c2) and e)]
    bad535 = [(g, lbl, pa) for g in (True, False) for lbl in (None, LABEL_NO_CYCLE, LABEL_ANCHOR)
              for pa in (True, False)
              if _adjudicate(g, lbl, True, True, False, pa)["MECH-535"] == "weakens"
              and not ((not g) and lbl == LABEL_NO_CYCLE and pa)]
    _chk("cube: MECH-535 weakens on a red gate only via no-cycle AND phenotype absent", not bad535, bad535)
    _chk("cube: weakens only under gate AND C1 AND NOT C2 AND envelope exceeded", not bad, bad)
    bad2 = [(g, c1, c2, e) for g in (True, False) for c1 in (True, False) for c2 in (True, False)
            for e in (True, False)
            if _adjudicate(g, None, c1, c2, e)["outcome"] == "PASS" and not (g and c1 and c2)]
    _chk("cube: PASS only under gate AND C1 AND C2", not bad2, bad2)
    return out


if __name__ == "__main__":
    import argparse
    import time
    from datetime import datetime

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-test", action="store_true",
                        help="run the detector + adjudication contract and exit")
    args = parser.parse_args()

    if args.self_test:
        res = self_test(verbose=True)
        print("self-test: %s (%d checks)" % ("OK" if res["ok"] else "FAILED", len(res["checks"])),
              flush=True)
        sys.exit(0 if res["ok"] else 1)

    t0 = time.perf_counter()
    seeds = args.seeds if args.seeds else (DRY_RUN_SEEDS if args.dry_run else SEEDS)
    result = run_experiment(seeds=seeds, dry_run=args.dry_run)

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"] = ts
    result["timestamp_utc"] = ts
    result["run_id"] = f"{EXPERIMENT_TYPE}_{ts}_v3"
    result["architecture_epoch"] = ARCHITECTURE_EPOCH
    result["queue_id"] = QUEUE_ID

    episode_log = result.pop("episode_log", None)

    full_config = {
        "rung": RUNG,
        "level_id": LEVEL_ID,
        "w_consume": W_CONSUME,
        "w_survival": W_SURVIVAL,
        "off_arm_p0a_field_weight": OFF_ARM_P0A_FIELD_WEIGHT,
        "zworld_p0_episodes": (DRY_RUN_ZWORLD_P0 if args.dry_run else ZWORLD_P0_EPISODES),
        "p0_warmup_episodes": (DRY_RUN_P0 if args.dry_run else P0_WARMUP_EPISODES),
        "p1_reinforce_episodes": (DRY_RUN_P1 if args.dry_run else P1_REINFORCE_EPISODES),
        "p1_ppo_episodes": (DRY_RUN_PPO if args.dry_run else P1_PPO_EPISODES),
        "eval_episodes": (DRY_RUN_EVAL if args.dry_run else EVAL_EPISODES),
        "steps_per_episode": (DRY_RUN_STEPS if args.dry_run else STEPS_PER_EPISODE),
        "log_episodes": (DRY_RUN_LOG_EPISODES if args.dry_run else LOG_EPISODES),
        "eval_arms": EVAL_ARM_IDS,
        "verdict_arms": VERDICT_ARMS,
        "persist_k": PERSIST_K,
        "sample_temperature": SAMPLE_TEMPERATURE,
        "switch_cost_margin_quantile": SWITCH_COST_MARGIN_QUANTILE,
        "anchors": ANCHOR_IDS,
        "lvg_latch_k": LVG_LATCH_K,
        "contamination_off_arms": CONTAM_OFF_ARMS,
        "contamination_off_kwargs": CONTAMINATION_OFF_KWARGS,
        "cycle_min_len": CYCLE_MIN_LEN,
        "cycle_incidence_ceiling": CYCLE_INCIDENCE_CEILING,
        "cycle_presence_floor": CYCLE_PRESENCE_FLOOR,
        "fixed_point_stationary_frac": FIXED_POINT_STATIONARY_FRAC,
        "bounded_orbit_max_period": BOUNDED_ORBIT_MAX_PERIOD,
        "bounded_orbit_min_repeats": BOUNDED_ORBIT_MIN_REPEATS,
        "c2_effect_floor": C2_EFFECT_FLOOR,
        "unique_cells_min_gain": UNIQUE_CELLS_MIN_GAIN,
        "phenotype_incidence_floor": PHENOTYPE_INCIDENCE_FLOOR,
        "lvg_latch_retention": LVG_LATCH_RETENTION,
        "env_kwargs": x734._env_kwargs_for_rung(RUNG),
        "dry_run": bool(args.dry_run),
    }
    out_path = write_flat_manifest(
        result, None, dry_run=args.dry_run,
        config=full_config, seeds=seeds, script_path=Path(__file__),
        started_at=t0, z_goal_stream_stats=_ZG.stats(),
    )

    if episode_log is not None and not args.dry_run:
        episode_log["run_id"] = result["run_id"]
        log_path = Path(out_path).parent / f"{Path(out_path).stem}_episode_log.json"
        log_path.write_text(json.dumps(episode_log, indent=2) + "\n", encoding="utf-8")
        print(f"[fishtank] episode log -> {log_path}", flush=True)
    elif episode_log is not None and args.dry_run:
        n_logged = sum(len(s["episodes"]) for s in episode_log["seeds"])
        print(f"[smoke] fishtank episode log built: {n_logged} episodes across "
              f"{len(episode_log['arm_ids'])} arm ids (not written under --dry-run)", flush=True)

    print(f"outcome: {result['outcome']} ({result['interpretation']['label']})", flush=True)

    _outcome_raw = str(result["outcome"]).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
        dry_run=args.dry_run,
    )
