"""V3-EXQ-840b: MECH-294 behavioural falsifier -- joint vs alternation vs shuffled
theta-burst binding, per-candidate co-binding COHERENCE routed through the
route-range selection authority, read out on the E3 committed-action distribution.

MEASUREMENT-WINDOW FIX (corrects V3-EXQ-840; supersedes it). V3-EXQ-840 FAILed
its readiness gate on all 3 packet-ON arms via `adequate_committed_window_sample`
(worst-cell n_p1_ticks_past_window: JOINT=27, ALT=6, SHUF=92, vs floor=200). A
prior failure_autopsy mischaracterised the cause as a sub-floor
`joint_route_range_supra_floor` (coherence routing) -- WRONG: the run's own
per_arm_gate data shows that precondition MET (measured 0.063841 vs floor 0.01,
~6.4x headroom). Direct per-cell inspection (evidence/experiments/
v3_exq_840_..._20260730T170540Z_v3.json) shows the SAME seeds (45, 49) starved
EVERY arm including ARM_0_OFF (packet off entirely) -- ruling out packet-ON
coherence routing as the cause. Root physical cause: ENV_KWARGS declares a
"HARM-FREE env" (num_hazards=0, hazard_harm=0.0, proximity_harm_scale=0.0) but
left the environment's SELF-CONTAMINATION mechanic live (contaminated_harm=0.4
default, contamination_threshold=2.0) -- contamination accumulates from the
agent's OWN repeated visits to a cell and is NOT gated by num_hazards, so a
seed whose SP-CEM policy revisits a small area drains agent_health via the
"agent_caused_hazard" contact branch (causal_grid_world.py step()), triggering
`done = agent_health<=0.0 or steps>=500` well before the 200-step episode
budget is spent -- starving the post-MEASURE_AFTER_TICK committed-action
sample for exactly those seeds. This is an unintended gap in the original
"HARM-FREE" design intent (every OTHER harm channel was already zeroed), not a
scientific confound introduced by the fix: contaminated_harm is an env-level
parameter applied identically regardless of theta_packet_binding_mode, so
zeroing it changes no arm differentially and does not touch
compose_per_candidate_coherence or the route-range readiness floor (both
confirmed working correctly by the diagnosis; see the "FIX" note on ENV_KWARGS
below). With every health-depletion route closed (hazard cells do not exist at
num_hazards=0; proximity harm and hazard contact harm are already 0.0;
contaminated_harm is now 0.0 too), `done` can never fire on health, and env
steps never approach the internal 500-step cap within a 200-step episode -- so
every cell now deterministically accumulates the full
NOMINAL_WINDOW_TICKS=3600 committed ticks, ~18x the floor, for EVERY seed. This
was preferred over inflating STEPS_PER_EPISODE / P1_MEASUREMENT_EPISODES
(candidate (a)): that approach adds compute without removing the confound
(hard seeds could still terminate early on every extended episode) and remains
probabilistic rather than deterministic. Trade-off, stated honestly: agent_health
(body[2]) was a live, seed-dependent, fluctuating signal in V3-EXQ-840 (not just
for seeds 45/49 -- most seeds showed some contamination-driven attrition, e.g.
ARM_1_JOINT seed44=991/3600 ticks), and is now constant at 1.0 for the whole
run. Since that fluctuation was never an intentional manipulation (it is a
seed-dependent artifact of movement dynamics common to every arm, not
correlated with binding mode), removing it does not bias C1/C2 -- if anything
it removes uncontrolled nuisance variance from the cross-seed noise band both
criteria must exceed. If a future iteration judges that self-contamination
dynamics are themselves worth testing under MECH-294, that is a distinct
question for a new EXQ, not a re-open of this falsifier's measurement-window
defect.

THIS IS AN EVIDENCE RUN (experiment_purpose="evidence", claim_ids=["MECH-294"]).
Unlike the readiness diagnostics that precede it (V3-EXQ-657/657a packet
completeness, V3-EXQ-661 compose-coherence behavioural readiness, and the
route-range readiness V3-EXQ-790/791a), this run WEIGHTS MECH-294's confidence.

WHAT MECH-294 CLAIMS (the load-bearing joint clause). A single ~125ms theta cycle
binds a {goal_latent, action_proposal, risk_estimate, state_summary} tuple
SIMULTANEOUSLY (joint within-cycle co-binding), NOT one content stream alternating
across cycles (the Kay-2020 parsimonious architecture). The substrate-side
falsifier the 2026-04-26 governance hold demanded: a test that discriminates
within-cycle joint binding from cross-cycle alternation on REE's own substrate.

WHAT A BEHAVIOURAL RETEST MUST SHOW (design memo S6/S7.3 interpretation grid,
REE_assembly/docs/architecture/mech_294_multi_content_theta_packet.md). Holding
bandwidth and content identical and varying ONLY the binding structure, the E3
committed-action-class distribution must depend on within-cycle co-binding that
the alternation and shuffled controls cannot provide:
  C1 (joint != alternation): JOINT's committed-class distribution differs from
     ALTERNATION's by more than the within-arm cross-seed noise band. This is the
     operational "within-cycle co-binding carries signal the one-stream-per-cycle
     scheme cannot". LOAD-BEARING.
  C2 (joint != shuffled): JOINT's committed distribution differs from SHUFFLED's
     by the same margin. The stronger leg: co-binding must matter AS co-binding,
     not merely as four-streams-present. LOAD-BEARING.

WHY THIS RUN IS NOW VALID (and was not before). The behavioural falsifier was
substrate-BLOCKED twice and is now unblocked by two landed substrate amends plus a
confirmed route-range readiness:
  (1) compose-coherence amend (2026-06-09): the binding regime gates the E3 bias
      (theta_packet_compose_into_e3_bias) -- but currency_coherence() is a SCALAR,
      so the per-candidate PATTERN was mode-invariant and V3-EXQ-661 measured
      committed-distribution TV ~0 across all modes (max 0.0097). The route-range
      authority's unit-range normalisation erases a uniform scalar scaling
      (JOINT==ALT) and floors SHUFFLED. That was a genuine wiring gap, not a claim
      result -- the exact C1-FAIL "wiring" branch of S6.
  (2) per-candidate co-binding coherence amend (2026-06-19): coherence is rendered
      PER-CANDIDATE with a mode-distinct cross-candidate RANGE (JOINT full range /
      ALTERNATION a distinct non-zero pattern mixing held-prior action refs /
      SHUFFLED ~0), routed via _bdc_coherence through the EXISTING route source
      modulatory_channel_route_source="coherence" into the modulatory accumulator
      the route-range authority rescales + the 569i top-k shortlists. The
      carve-able channel now EXISTS (contract suite is the substrate-readiness gate:
      test_mech294_per_candidate_coherence.py C1..C7).
  (3) route-range authority readiness CONFIRMED: V3-EXQ-790 passed both science
      criteria (routing active 10/10 ARM_1, inactive 10/10 ARM_0; committed-class
      TV supra-floor 8/10; routed range 0.334 supra-floor and 1.644 bounded) and
      its false substrate_not_ready_requeue was WITHDRAWN as a gate defect
      (failure_autopsy_V3-EXQ-790_2026-07-22, user-adjudicated). The corrected-gate
      re-run V3-EXQ-791a PASSed on ree-cloud-2 (label route_range_substrate_ready).
      The general route-range substrate the coherence channel rides on is READY.

THE SAME-STATISTIC READINESS SAFEGUARD (the crux of correctness; skill Step 3.5
643-rule + the DV-symmetry rule). The load-bearing criteria C1/C2 gate on the
committed-action-class distribution, which is a RANK/argmax readout. A rank readout
is INVARIANT under a broadcast additive constant across candidates -- which is
EXACTLY why the 661 scalar-coherence run measured TV ~0. So a committed-TV null is
interpretable as a claim result ONLY IF the manipulation is genuinely NOT a
broadcast scalar this time, i.e. only if the routed per-candidate coherence produces
a supra-floor CROSS-CANDIDATE RANGE. The readiness gate therefore asserts the SAME
statistic the authority carves and the criteria route on -- route_range, a
max-minus-min cross-candidate RANGE -- not a magnitude. If JOINT's routed coherence
range is below floor, the manipulation is (like 661) rescale-invisible and no
committed-TV difference is a claim result: the run self-routes
substrate_not_ready_requeue with non_degenerate=false, NEVER a substrate verdict.

DV-SYMMETRY DECLARATION (skill Step 3, MANDATORY, per arm). The C1/C2 DV is the
committed-action-class histogram, a selected-action (rank/argmax) readout. Its
symmetry group is (i) a uniform additive constant broadcast across candidates and
(ii) any permutation of the candidate set.
  ARM_1_JOINT   -- manipulation: joint co-binding -> compose_per_candidate_coherence
    returns a PER-CANDIDATE bias (each candidate's cosine to the action co-bound
    with its stream this cycle), routed as a supra-floor cross-candidate RANGE. It
    is per-candidate (a function of each candidate's own alignment), NOT a broadcast
    scalar, so NOT invariant under (i); and it is a function of each candidate's own
    coherence rather than of the pooled set, so NOT invariant under (ii). The delta
    is a MEASUREMENT -- CONDITIONAL on route_range supra-floor (readiness R1), which
    is precisely what separates this from the 661 scalar case where the range was ~0
    and the manipulation WAS invariant.
  ARM_2_ALT     -- manipulation: alternation -> a DISTINCT non-zero per-candidate
    pattern (live ref w 1.0 + held-prior action refs at coherence_hold_weight=0.5),
    a non-uniform per-candidate bias -> not a broadcast scalar -> not invariant.
    Its per-candidate PATTERN differs from JOINT's (2026-06-19 smoke: joint-vs-alt
    bias cosine 0.82, a distinct pattern rather than a uniform scaling), which is
    why range-normalisation does not collapse ALT onto JOINT.
  ARM_3_SHUF    -- manipulation: shuffled -> nothing co-bound -> coherence None ->
    route_range ~0 -> the authority applies NO bias. This arm is the INERT control
    (behaviourally == coherence-OFF). It is a STRUCTURAL control, not a measured
    contrast: I do NOT claim a measured effect FROM shuffling. C2 (JOINT vs SHUF)
    measures whether JOINT's supra-floor bias moved committed behaviour AWAY from
    this inert baseline. Accordingly SHUF carries NO route-range-supra-floor
    precondition (asserting one would be structurally unsatisfiable and vacate it,
    the V3-EXQ-785 trap); it is scored only on committed-window sample + diversity.
  ARM_0_OFF     -- packet off, no coherence, no routing: the homogeneous-latent
    null baseline (S7.1 ARM_0). Not part of C1/C2.

INTERPRETATION GRID -> evidence_direction (pre-registered; adapts memo S7.3 to an
evidence run). Applied ONLY when the readiness gate passes (JOINT+ALT+SHUF green):
  C1 pass + C2 pass -> "joint_binding_behaviourally_load_bearing" -> outcome PASS,
    evidence_direction=supports. Within-cycle co-binding is load-bearing vs BOTH
    controls; this is the substrate-side falsification test the 2026-04-26 hold
    demanded, passed.
  C1 pass + C2 FAIL -> "multi_content_present_joint_not_isolated" -> outcome FAIL,
    evidence_direction=mixed. Four streams present carries signal, but co-binding
    AS co-binding is not isolated. Per S7.3: do NOT promote the joint clause;
    refine to a genuinely conjunctive readout.
  C1 FAIL -> "joint_indistinguishable_from_alternation" -> outcome FAIL,
    evidence_direction=weakens. The Kay-2020 parsimonious outcome surviving on REE's
    own substrate: joint binding is not doing work at the committed-action readout.
    Routes to /failure-autopsy to disambiguate residual wiring vs genuine
    over-specification; the joint clause is not promotable and the hold stands.
  readiness gate red (JOINT route_range below floor / candidate diversity
    degenerate / committed-window sample starved) -> "substrate_not_ready_requeue"
    -> outcome FAIL, evidence_direction=non_contributory, non_degenerate=false
    (EXCLUDED from scoring). A starved test is not a claim result.

ARM-SCOPED PRECONDITION GATE (skill Step 3 MULTI-ARM rule + failure_autopsy
V3-EXQ-785/790). Preconditions declare the arms they are MEANINGFUL for via
PreconditionSpec.applies_to, so a structurally-inert arm (SHUFFLED's ~0 range)
never carries a gate it cannot satisfy and never vacates a green arm. The
committed-window sample + candidate-diversity gates apply to every packet-ON arm
(their committed distributions are the C1/C2 DV); the route-range + fresh-select
gates apply to JOINT only (the arm whose routed statistic is the readiness anchor).

SAMPLE-SIZE INTEGRITY (skill Step 3.5 latch rule). E3 populates
last_score_diagnostics ONLY inside select(), and it LATCHES across held ticks
(heartbeat.e3_steps_per_tick / MECH-093-modulated cadence). The route-range readout
is taken on FRESH selections only: last_score_diagnostics is cleared to None
immediately before every select_action, a None reads as a latched tick (recorded in
n_latched_ticks, NO row), and only a populated dict contributes a route row. The
committed-action readout stays on every env step in the window because the HELD
action IS the committed behaviour across latched ticks (denominator
n_p1_ticks_past_window); both denominators are emitted per cell.

ARM REUSE. All arms train SD-056 online (the e2.world_forward divergence the routed
coherence range ultimately depends on), so every cell is stateful-per-cell and
reuse-INELIGIBLE (extra_ineligible_reasons=["online_e2_training_stateful_per_cell"],
same as V3-EXQ-791a). No canonical baseline module / cross-driver mint is factored:
the fingerprint would be marked ineligible regardless, so a mint would be dead
metadata. Fingerprints are still emitted per cell (Phase-0 instrument-only).

SLEEP DRIVER: K=never (no sleep; waking action-selection behavioural readout).

Usage:
  /opt/local/bin/python3 experiments/v3_exq_840b_mech294_theta_packet_binding_committed_action_falsifier.py --dry-run
"""

import argparse
import itertools
import math
import os
import random
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiment_protocol import emit_outcome  # noqa: E402
from experiments._lib.arm_fingerprint import compute_arm_fingerprint, reset_all_rng  # noqa: E402
from experiments._lib.precondition_gate import (  # noqa: E402
    PreconditionSpec,
    aggregate_arm_gates,
    arm_criteria_non_degenerate,
    assert_no_structurally_unsatisfiable_gate,
    evaluate_arm_gate,
)
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_840b_mech294_theta_packet_binding_committed_action_falsifier"
QUEUE_ID = os.environ.get("REE_QUEUE_ID") or "V3-EXQ-840b"
CLAIM_IDS: List[str] = ["MECH-294"]  # behavioural-evidence successor; WEIGHTS confidence
SUPERSEDES = "V3-EXQ-840"  # corrects the committed-window measurement defect; see module docstring
EXPERIMENT_PURPOSE = "evidence"

# --- schedule (mirrors the V3-EXQ-649/791a proven budget) ---
SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
P0_WARMUP_EPISODES = 60           # SD-056 contrastive warmup
P1_MEASUREMENT_EPISODES = 20
STEPS_PER_EPISODE = 200
MEASURE_AFTER_TICK = 20

DRY_RUN_SEEDS = [42]
DRY_RUN_P0 = 2
DRY_RUN_P1 = 2
DRY_RUN_STEPS = 30
DRY_RUN_MEASURE_AFTER_TICK = 2

# --- pre-registered acceptance thresholds ---
C_TV_FLOOR = 0.10                 # C1/C2 committed-class distribution TV floor (memo S7.2)
C0_ROUTE_FLOOR = 0.01             # readiness: JOINT routed per-candidate coherence RANGE floor
C0_MAGNITUDE_CEIL = 1.0e6         # readiness: routed range bounded (643a explosion guard)
DIVERSITY_FLOOR = 1.0             # readiness: candidate first-action classes > 1 (strict)
COMMITTED_WINDOW_FLOOR = 200      # readiness: committed-class DV sample per cell
MIN_SEEDS_FOR_PASS = 7            # of 10

# Fresh-selection floor derived from the MECH-093-modulated cadence WORST case
# (ree_core/heartbeat/clock.py beta_rate_max_steps=20), NOT the nominal
# e3_steps_per_tick=10 (the V3-EXQ-790 mis-derivation). Same derivation as 791a.
BETA_RATE_MAX_STEPS = 20
NOMINAL_WINDOW_TICKS = P1_MEASUREMENT_EPISODES * (STEPS_PER_EPISODE - MEASURE_AFTER_TICK)
FRESH_SELECT_FLOOR = NOMINAL_WINDOW_TICKS // BETA_RATE_MAX_STEPS   # 180

# SD-056 online contrastive training (mirror V3-EXQ-649/791a harness).
SD056_WEIGHT = 0.05
E2_CONTRASTIVE_LR = 1e-3
E2_TRAIN_EVERY_K_TICKS = 1
CONTRASTIVE_BATCH_K = 8
TRANSITION_BUFFER_MAX = 256
MIN_BUFFER_BEFORE_TRAIN = 16
MIN_CLASSES_FOR_TRAIN = 2
MAX_GRAD_NORM = 1.0

# HARM-FREE env (SP-CEM + resources give action-divergent candidates for SD-056 to
# train z_world divergence on and for the packet's action_proposal slot).
ENV_KWARGS: Dict[str, Any] = dict(
    size=12,
    num_hazards=0,
    num_resources=5,
    hazard_harm=0.0,
    env_drift_interval=5,
    env_drift_prob=0.1,
    proximity_harm_scale=0.0,
    proximity_benefit_scale=0.05,
    proximity_approach_threshold=0.2,
    hazard_field_decay=0.5,
    resource_respawn_on_consume=True,
    toroidal=False,
    harm_history_len=10,
    # FIX (V3-EXQ-840b, corrects V3-EXQ-840): the ORIGINAL "HARM-FREE env" intent
    # zeroed hazard_harm and proximity_harm_scale but left contaminated_harm at
    # its 0.4 default. Self-contamination (contamination_grid accumulating from
    # the agent's OWN repeated visits, NOT gated by num_hazards) was therefore
    # the sole live health-depletion channel, and it drained agent_health to 0
    # for specific seeds (45, 49) whose SP-CEM policy revisited a small area,
    # triggering `done` well before the 200-step episode budget and starving
    # the committed-action measurement window for those seeds (see module
    # docstring). Setting it to 0.0 closes the last open health-depletion
    # route (hazard/proximity harm are already 0.0 and num_hazards=0 means no
    # hazard cells exist), completing the harm-free design intent and making
    # every cell's full NOMINAL_WINDOW_TICKS budget deterministic. Applied
    # identically to every arm -- not a binding-mode-differential change.
    contaminated_harm=0.0,
)

# Arms: matched seeds, matched content, ONLY theta_packet_binding_mode varies among
# the three packet-ON arms. ARM_0_OFF is the homogeneous-latent null.
ARMS: List[Dict[str, Any]] = [
    {"arm_id": "ARM_0_OFF", "label": "packet_off_homogeneous_latent_null",
     "packet_on": False, "binding_mode": None},
    {"arm_id": "ARM_1_JOINT", "label": "joint_within_cycle_cobinding",
     "packet_on": True, "binding_mode": "joint"},
    {"arm_id": "ARM_2_ALT", "label": "alternation_one_stream_per_cycle_kay2020",
     "packet_on": True, "binding_mode": "alternation"},
    {"arm_id": "ARM_3_SHUF", "label": "shuffled_independent_content_control",
     "packet_on": True, "binding_mode": "shuffled"},
]

ARM_CONTEXTS: List[Dict[str, Any]] = [
    {"id": a["arm_id"], "arm_id": a["arm_id"],
     "packet_on": bool(a["packet_on"]),
     "is_joint": (a["binding_mode"] == "joint")}
    for a in ARMS
]

_JOINT_SCOPE_NOTE = (
    "ARM_1_JOINT only. This precondition guards the routed per-candidate coherence "
    "RANGE, the statistic the route-range authority carves and the SAME statistic "
    "C1/C2 route on downstream. JOINT is the arm whose supra-floor range is the "
    "precondition for a meaningful committed-action contrast. ALTERNATION carries a "
    "distinct non-zero pattern (recorded, not gated) and SHUFFLED is ~0 by "
    "construction (an inert structural control), so asserting a supra-floor range "
    "on them would be structurally unsatisfiable and vacate them (the V3-EXQ-785 "
    "trap). Their committed distributions ARE gated for sample adequacy + diversity."
)
_PACKET_SCOPE_NOTE = (
    "All packet-ON arms (JOINT/ALTERNATION/SHUFFLED). Each arm's committed-action "
    "distribution is a C1/C2 DV, so each must have an adequately-sampled committed "
    "window and a non-degenerate candidate pool. ARM_0_OFF is the null baseline and "
    "is not part of C1/C2, so it carries neither gate."
)

PRECONDITION_SPECS: List[PreconditionSpec] = [
    PreconditionSpec(
        name="joint_route_range_supra_floor",
        description=(
            "JOINT routed per-candidate coherence cross-candidate RANGE "
            "(modulatory_channel_route_range on the coherence source, pre-rescale) "
            "clears the floor on >= MIN_SEEDS_FOR_PASS seeds. This is the "
            "SAME-STATISTIC safeguard (skill 643 rule): C1/C2 gate on a RANK "
            "readout (committed-class TV), which is invariant under a broadcast "
            "additive constant -- exactly the 661 scalar-coherence null. A "
            "committed-TV difference is a claim result ONLY IF the routed coherence "
            "is a genuine supra-floor cross-candidate RANGE, not a rescale-invisible "
            "scalar. measured is the MIN_SEEDS_FOR_PASS-th largest per-seed "
            "route_range_mean (order statistic matching the count quantifier). Below "
            "floor => the manipulation cannot move committed behaviour => "
            "substrate_not_ready_requeue, NEVER a substrate verdict."
        ),
        control=(
            "JOINT: all four V_s-gated streams co-bound -> compose_per_candidate_"
            "coherence full cross-candidate range; 2026-06-19 smoke measured JOINT "
            "range 0.1 vs a 0.01 floor (~10x headroom)"
        ),
        threshold=C0_ROUTE_FLOOR,
        direction="lower",
        applies_to=lambda ctx: bool(ctx["is_joint"]),
        applies_note=_JOINT_SCOPE_NOTE,
    ),
    PreconditionSpec(
        name="adequate_fresh_selection_sample",
        description=(
            "JOINT recorded >= FRESH_SELECT_FLOOR genuine E3 select() calls on the "
            "CORRECTED denominator (diagnostics cleared before every select_action, "
            "so a latched tick contributes NO row) on >= MIN_SEEDS_FOR_PASS seeds, "
            "so the route_range readiness estimate is real and not ~9x "
            "pseudo-replicated. Floor 3600/beta_rate_max_steps=20 = 180, the "
            "MECH-093-modulated worst-case cadence. measured is the "
            "MIN_SEEDS_FOR_PASS-th largest per-seed n_fresh_select (order statistic "
            "matching the count quantifier)."
        ),
        control=(
            "nominal window P1*(steps-measure_after)=3600 ticks at the "
            "MECH-093 worst-case cadence beta_rate_max_steps=20 -> 180 selections"
        ),
        threshold=float(FRESH_SELECT_FLOOR),
        direction="lower",
        applies_to=lambda ctx: bool(ctx["is_joint"]),
        applies_note=_JOINT_SCOPE_NOTE,
    ),
    PreconditionSpec(
        name="candidate_first_action_diversity_supra_floor",
        description=(
            "Worst packet-ON cell's mean number of distinct candidate first-action "
            "classes > 1.0. The per-candidate coherence gate RANKS candidates by "
            "their co-bound action alignment; if the SP-CEM proposer emits a single "
            "first-action class the gate has nothing to rank and the committed-TV "
            "readout cannot move regardless of binding mode (memo honest-limitation: "
            "'the validation must guard candidate first-action diversity'). This is "
            "the positive control that the routed statistic CAN be non-degenerate. "
            "measured is the WORST cell (all(...) quantifier), offending cell named."
        ),
        control=(
            "SP-CEM with support_preserving_min_first_action_classes=2 stratifies "
            "elites to keep >= 2 first-action classes in the candidate pool"
        ),
        threshold=DIVERSITY_FLOOR,
        direction="lower",
        applies_to=lambda ctx: bool(ctx["packet_on"]),
        applies_note=_PACKET_SCOPE_NOTE,
    ),
    PreconditionSpec(
        name="adequate_committed_window_sample",
        description=(
            "Worst packet-ON cell recorded >= COMMITTED_WINDOW_FLOOR committed "
            "actions in the measurement window (n_p1_ticks_past_window). This is the "
            "C1/C2 DV denominator (the HELD action is the committed behaviour across "
            "latched ticks, so the committed-class histogram is a per-env-step "
            "behavioural readout). A cell with too few committed observations gives "
            "an under-powered TV estimate. measured is the WORST cell (all(...) "
            "quantifier), offending cell named."
        ),
        control=(
            f"nominal window {NOMINAL_WINDOW_TICKS} env steps; early-`done` "
            "truncation reduces it, so a per-cell floor guards the TV estimate"
        ),
        threshold=float(COMMITTED_WINDOW_FLOOR),
        direction="lower",
        applies_to=lambda ctx: bool(ctx["packet_on"]),
        applies_note=_PACKET_SCOPE_NOTE,
    ),
    PreconditionSpec(
        name="routed_range_bounded",
        description=(
            "Routed coherence range stayed finite and below the 643a explosion "
            "ceiling (SD-056 online training numerical stability). Applies to all "
            "arms -- a blown-up range is a stability failure wherever it appears."
        ),
        control="max route_range across this arm's cells",
        threshold=C0_MAGNITUDE_CEIL,
        direction="upper",
    ),
]


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(seed=seed, **ENV_KWARGS)


def _make_agent(env: CausalGridWorldV2, arm: Dict[str, Any]) -> REEAgent:
    """Main-path SP-CEM stack with the shared E3-side bias channels ON, the
    modulatory selection authority ON (gain=0.5), SD-056 contrastive trained online,
    and -- for the packet-ON arms -- the MECH-294 multi-content theta packet with
    per-candidate co-binding coherence composed into the E3 bias and routed through
    modulatory_channel_route_source="coherence". The ONLY swept axis among the
    packet-ON arms is theta_packet_binding_mode."""
    packet_on = bool(arm["packet_on"])
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=32,
        alpha_world=0.9,
        alpha_self=0.3,
        use_harm_stream=True,
        z_harm_dim=32,
        use_affective_harm_stream=True,
        z_harm_a_dim=16,
        harm_history_len=10,
        z_goal_enabled=True,
        goal_weight=0.5,
        drive_weight=2.0,
        e1_goal_conditioned=True,
        use_resource_proximity_head=True,
        resource_proximity_weight=0.5,
        benefit_eval_enabled=True,
        benefit_weight=1.0,
        # ARC-065 SP-CEM (action-divergent candidate pool; feeds the packet's
        # action_proposal slot AND provides candidate first-action diversity).
        use_support_preserving_cem=True,
        support_preserving_stratified_elites=True,
        support_preserving_ao_std_floor=0.2,
        support_preserving_min_first_action_classes=2,
        # Shared E3-side bias channels ON in ALL arms (held identical).
        use_lateral_pfc_analog=True,
        use_mech295_liking_bridge=True,
        # Other policy-layer regulators OFF (binding mode is the swept axis).
        use_structured_curiosity=False,
        use_e3_score_diversity=False,
        use_noise_floor=False,
        use_tonic_vigor=False,
        use_dacc=False,
        use_ofc_analog=False,
        use_gated_policy=False,
        # SD-056 substrate trained online on every arm.
        e2_action_contrastive_enabled=True,
        e2_action_contrastive_weight=SD056_WEIGHT,
        e2_rollout_output_norm_clamp_enabled=True,
        e2_rollout_output_norm_clamp_ratio=2.0,
        candidate_summary_source="e2_world_forward",
        # modulatory-bias-selection-authority (the gate the routed range reaches).
        use_modulatory_selection_authority=True,
        modulatory_authority_gain=0.5,
        # --- MECH-294 multi-content theta packet (per-candidate coherence routed) ---
        # per-stream V_s is REQUIRED by the packet (LOUD ValueError otherwise).
        use_per_stream_vs=packet_on,
        use_multi_content_theta_packet=packet_on,
        theta_packet_binding_mode=(arm["binding_mode"] or "joint"),
        theta_packet_compose_into_e3_bias=packet_on,
        theta_packet_compose_per_candidate_coherence=packet_on,
        theta_packet_coherence_hold_weight=0.5,
        theta_packet_bias_scale=0.1,
        # route the per-candidate coherence RANGE into the authority (packet arms
        # only; OFF arm has no coherence channel so routing stays off).
        use_modulatory_channel_routing=packet_on,
        modulatory_channel_route_source="coherence",
        modulatory_channel_route_min_range_floor=1e-6,
        modulatory_channel_route_weight=1.0,
    )
    return REEAgent(cfg)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _obs(d: Dict[str, Any], key: str) -> Optional[torch.Tensor]:
    h = d.get(key)
    if h is None:
        return None
    return h.float().unsqueeze(0) if h.dim() == 1 else h.float()


def _tv_distance(counts_a: Dict[int, int], counts_b: Dict[int, int]) -> float:
    """Total-variation distance between two committed-class histograms."""
    ta = sum(counts_a.values())
    tb = sum(counts_b.values())
    if ta <= 0 or tb <= 0:
        return 0.0
    classes = set(counts_a) | set(counts_b)
    tv = 0.0
    for cls in classes:
        pa = counts_a.get(cls, 0) / ta
        pb = counts_b.get(cls, 0) / tb
        tv += abs(pa - pb)
    return 0.5 * tv


def _candidate_first_action_classes(candidates: List[Any]) -> int:
    """Number of DISTINCT first-action classes in the candidate pool this tick."""
    classes = set()
    for c in candidates:
        try:
            if c.actions.shape[1] <= 0:
                continue
            fa = c.actions[:, 0, :]
            classes.add(int(fa.argmax(dim=-1).reshape(-1)[0].item()))
        except Exception:
            continue
    return len(classes)


def _sample_class_diverse_batch(
    buffer: Deque[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    k: int,
    rng: random.Random,
) -> Optional[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
    if len(buffer) < MIN_BUFFER_BEFORE_TRAIN:
        return None
    pool = list(buffer)
    rng.shuffle(pool)
    seen_classes: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for tup in pool:
        cls = int(tup[1].argmax().item())
        if cls not in seen_classes:
            seen_classes[cls] = tup
        if len(seen_classes) >= k:
            break
    if len(seen_classes) < MIN_CLASSES_FOR_TRAIN:
        return None
    samples = list(seen_classes.values())
    picked_ids = {id(s) for s in samples}
    for tup in pool:
        if len(samples) >= k:
            break
        if id(tup) in picked_ids:
            continue
        samples.append(tup)
        picked_ids.add(id(tup))
    return samples


def _e2_contrastive_step(
    agent: REEAgent,
    buffer: Deque[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    optimiser: torch.optim.Optimizer,
    rng: random.Random,
) -> Optional[float]:
    batch = _sample_class_diverse_batch(buffer, CONTRASTIVE_BATCH_K, rng)
    if batch is None:
        return None
    z0_K = torch.stack([t[0] for t in batch]).to(agent.device)
    actions_K = torch.stack([t[1] for t in batch]).to(agent.device)
    z1_K = torch.stack([t[2] for t in batch]).to(agent.device)
    optimiser.zero_grad(set_to_none=True)
    loss = agent.e2.world_forward_contrastive_loss(
        z_world_0=z0_K, actions=actions_K, z_world_1_targets=z1_K,
        simulation_mode=False,
    )
    if not torch.is_tensor(loss):
        return None
    loss_val = float(loss.detach().item())
    if not math.isfinite(loss_val):
        return loss_val
    if not loss.requires_grad or loss_val == 0.0:
        return loss_val
    weighted = SD056_WEIGHT * loss
    weighted.backward()
    torch.nn.utils.clip_grad_norm_(agent.e2.parameters(), max_norm=MAX_GRAD_NORM)
    optimiser.step()
    return loss_val


# ---------------------------------------------------------------------------
# Per-(seed, arm) runner
# ---------------------------------------------------------------------------

_ZG = ZGoalStreamAccumulator()


def _run_seed_arm(
    arm: Dict[str, Any],
    seed: int,
    p0_episodes: int,
    p1_episodes: int,
    steps_per_episode: int,
    measure_after_tick: int,
) -> Dict[str, Any]:
    reset_all_rng(seed)

    env = _make_env(seed)
    agent = _make_agent(env, arm)
    e2_opt = torch.optim.Adam(agent.e2.parameters(), lr=E2_CONTRASTIVE_LR)

    transition_buffer: Deque[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ] = deque(maxlen=TRANSITION_BUFFER_MAX)
    sample_rng = random.Random(seed)

    total_train_eps = p0_episodes + p1_episodes

    route_ranges: List[float] = []
    route_range_max = 0.0
    route_active_ticks = 0
    committed_class_counts: Dict[int, int] = {}
    n_p1_ticks_past_window = 0
    n_fresh_select = 0
    n_latched_ticks = 0
    n_contrastive_steps = 0
    diversity_sum = 0
    diversity_ticks = 0
    error_note: Optional[str] = None

    for ep in range(total_train_eps):
        is_p1 = ep >= p0_episodes
        phase_label = "P1" if is_p1 else "P0"

        _, obs_dict = env.reset()
        agent.reset()

        z_self_prev: Optional[torch.Tensor] = None
        action_prev: Optional[torch.Tensor] = None
        pending_capture: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        tick_in_ep = 0

        for _step in range(steps_per_episode):
            body = obs_dict["body_state"].float()
            world = obs_dict["world_state"].float()
            if body.dim() == 1:
                body = body.unsqueeze(0)
            if world.dim() == 1:
                world = world.unsqueeze(0)

            latent = agent.sense(
                obs_body=body, obs_world=world,
                obs_harm=_obs(obs_dict, "harm_obs"),
                obs_harm_a=_obs(obs_dict, "harm_obs_a"),
                obs_harm_history=_obs(obs_dict, "harm_history"),
            )

            if pending_capture is not None:
                z0_prev, a_prev = pending_capture
                z1_obs = latent.z_world.detach().reshape(-1).clone()
                if (
                    torch.isfinite(z0_prev).all()
                    and torch.isfinite(a_prev).all()
                    and torch.isfinite(z1_obs).all()
                ):
                    transition_buffer.append((z0_prev, a_prev, z1_obs))
                pending_capture = None

            if z_self_prev is not None and action_prev is not None:
                agent.record_transition(
                    z_self_prev, action_prev, latent.z_self.detach()
                )

            ticks = agent.clock.advance()
            wdim = latent.z_world.shape[-1]
            # _e1_tick fires the theta packet observe (packet-ON arms); zeros
            # otherwise. Must run when e1_tick so the packet accumulates streams.
            e1_prior = (
                agent._e1_tick(latent) if ticks.get("e1_tick", False)
                else torch.zeros(1, wdim, device=agent.device)
            )
            # generate_trajectories -> _e3_tick internally, which observes the
            # proposer first-action and SEALS the packet -> agent.last_theta_packet.
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)

            if agent.goal_state is not None:
                try:
                    energy = float(body[0, 3].item())
                except Exception:
                    energy = 1.0
                drive_level = max(0.0, 1.0 - energy)
                agent.update_z_goal(benefit_exposure=0.0, drive_level=drive_level)

            # Freshness marker: last_score_diagnostics latches across held ticks, so
            # clear it here -- a populated dict after select_action proves THIS tick
            # ran a genuine selection (skill Step 3.5 latch rule).
            agent.e3.last_score_diagnostics = None

            # select_action composes the per-candidate coherence (packet-ON arms)
            # and routes it via source="coherence".
            action = agent.select_action(candidates, ticks)
            if action is None:
                idx = int(np.random.randint(0, env.action_dim))
                action = torch.zeros(1, env.action_dim, device=agent.device)
                action[0, idx] = 1.0
                agent._last_action = action
            if not torch.isfinite(action).all():
                if error_note is None:
                    error_note = (
                        f"non-finite action at arm={arm['arm_id']} seed={seed} "
                        f"phase={phase_label} ep={ep} step={_step}"
                    )
                break

            # --- readouts (measurement window only) ---
            past_window = is_p1 and tick_in_ep >= measure_after_tick
            if past_window and candidates and len(candidates) >= 2:
                # Candidate first-action diversity (positive control for the routed
                # statistic; measured on the pool, every past-window tick).
                diversity_sum += _candidate_first_action_classes(candidates)
                diversity_ticks += 1

                # Route-range row: FRESH selections only (cleared-then-read latch).
                diag = agent.e3.last_score_diagnostics
                if diag is None:
                    n_latched_ticks += 1
                else:
                    n_fresh_select += 1
                    rr = diag.get("modulatory_channel_route_range")
                    if rr is not None and math.isfinite(float(rr)):
                        route_ranges.append(float(rr))
                        route_range_max = max(route_range_max, float(rr))
                    if bool(diag.get("modulatory_channel_route_active", False)):
                        route_active_ticks += 1

                # Committed-action readout: EVERY env step in the window (the held
                # action is the committed behaviour across latched ticks).
                cls = int(action.argmax(dim=-1).item())
                committed_class_counts[cls] = committed_class_counts.get(cls, 0) + 1
                n_p1_ticks_past_window += 1

            if (
                torch.isfinite(latent.z_world).all()
                and torch.isfinite(action).all()
            ):
                pending_capture = (
                    latent.z_world.detach().reshape(-1).clone(),
                    action.detach().reshape(-1).clone(),
                )

            if tick_in_ep % E2_TRAIN_EVERY_K_TICKS == 0:
                loss_val = _e2_contrastive_step(
                    agent=agent, buffer=transition_buffer,
                    optimiser=e2_opt, rng=sample_rng,
                )
                if loss_val is not None and math.isfinite(loss_val) and is_p1:
                    n_contrastive_steps += 1

            _, harm_signal, done, info, next_obs_dict = env.step(action)
            with torch.no_grad():
                agent.update_residue(
                    harm_signal=float(harm_signal),
                    world_delta=None,
                    hypothesis_tag=False,
                    owned=True,
                )

            z_self_prev = latent.z_self.detach()
            action_prev = action
            obs_dict = next_obs_dict
            tick_in_ep += 1
            if done:
                break

        if (ep + 1) % 10 == 0 or (ep + 1) == total_train_eps:
            print(
                f"  [train] arm={arm['arm_id']} seed={seed} phase={phase_label} "
                f"ep {ep + 1}/{total_train_eps}",
                flush=True,
            )

    def _mean(xs: List[float], default: float = 0.0) -> float:
        return float(sum(xs) / len(xs)) if xs else default

    route_active_frac = (
        float(route_active_ticks) / float(n_fresh_select)
        if n_fresh_select > 0 else 0.0
    )
    mean_first_action_classes = (
        float(diversity_sum) / float(diversity_ticks) if diversity_ticks > 0 else 0.0
    )

    _ZG.observe(agent)
    return {
        "arm_id": arm["arm_id"],
        "label": arm["label"],
        "binding_mode": arm["binding_mode"],
        "packet_on": bool(arm["packet_on"]),
        "seed": int(seed),
        "n_p1_ticks_past_window": int(n_p1_ticks_past_window),
        "n_fresh_select": int(n_fresh_select),
        "n_latched_ticks": int(n_latched_ticks),
        "n_route_rows": int(len(route_ranges)),
        "n_contrastive_steps": int(n_contrastive_steps),
        "error_note": error_note,
        # Routed per-candidate coherence RANGE (the same-statistic readiness anchor).
        "route_range_mean": round(_mean(route_ranges), 6),
        "route_range_max": round(route_range_max, 6),
        "route_active_frac": round(route_active_frac, 4),
        # Candidate first-action diversity (positive control).
        "mean_first_action_classes": round(mean_first_action_classes, 4),
        # Committed-action distribution (the C1/C2 DV).
        "committed_class_counts": {str(k): int(v) for k, v in committed_class_counts.items()},
        "n_committed_classes": int(len(committed_class_counts)),
    }


# ---------------------------------------------------------------------------
# Cross-arm evaluation
# ---------------------------------------------------------------------------

def _arm_rows(rows: List[Dict[str, Any]], arm_id: str) -> List[Dict[str, Any]]:
    return [r for r in rows if r.get("arm_id") == arm_id]


def _mean_key(rows: List[Dict[str, Any]], key: str) -> float:
    vals = [float(r.get(key, 0.0)) for r in rows]
    return float(sum(vals) / len(vals)) if vals else 0.0


def _counts(r: Dict[str, Any]) -> Dict[int, int]:
    return {int(k): int(v) for k, v in (r.get("committed_class_counts") or {}).items()}


def _cell_id(r: Dict[str, Any]) -> str:
    return f"{r.get('arm_id')}::seed{r.get('seed')}"


def _worst_cell(rows: List[Dict[str, Any]], value_fn) -> Tuple[float, Optional[str]]:
    if not rows:
        return 0.0, None
    worst_row = min(rows, key=value_fn)
    return float(value_fn(worst_row)), _cell_id(worst_row)


def _kth_largest(values: List[float], k: int) -> float:
    ordered = sorted(values, reverse=True)
    return float(ordered[k - 1]) if len(ordered) >= k else 0.0


def _matched_seed_tv(
    arm_a: List[Dict[str, Any]], arm_b: List[Dict[str, Any]]
) -> Dict[str, float]:
    """Per-seed TV(committed_a, committed_b) over the shared seeds."""
    by_seed_b = {r["seed"]: r for r in arm_b}
    tv: Dict[str, float] = {}
    for ra in arm_a:
        rb = by_seed_b.get(ra["seed"])
        if rb is not None:
            tv[str(ra["seed"])] = round(_tv_distance(_counts(ra), _counts(rb)), 4)
    return tv


def _cross_seed_baseline_tv(arm: List[Dict[str, Any]]) -> float:
    """Mean pairwise within-arm cross-seed TV (the noise band C1/C2 must exceed)."""
    pairs = list(itertools.combinations(arm, 2))
    if not pairs:
        return 0.0
    vals = [_tv_distance(_counts(a), _counts(b)) for a, b in pairs]
    return float(sum(vals) / len(vals))


def _evaluate(arm_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    off = _arm_rows(arm_results, "ARM_0_OFF")
    joint = _arm_rows(arm_results, "ARM_1_JOINT")
    alt = _arm_rows(arm_results, "ARM_2_ALT")
    shuf = _arm_rows(arm_results, "ARM_3_SHUF")
    by_arm = {"ARM_1_JOINT": joint, "ARM_2_ALT": alt, "ARM_3_SHUF": shuf}

    # ---- arm-scoped precondition gate --------------------------------------
    def _arm_max_route(rows: List[Dict[str, Any]]) -> float:
        return max([float(r.get("route_range_max", 0.0)) for r in rows] or [0.0])

    def _worst_div(rows):
        return _worst_cell(rows, lambda r: float(r.get("mean_first_action_classes", 0.0)))

    def _worst_committed(rows):
        return _worst_cell(rows, lambda r: float(r.get("n_p1_ticks_past_window", 0)))

    joint_kth_route = _kth_largest(
        [float(r.get("route_range_mean", 0.0)) for r in joint], MIN_SEEDS_FOR_PASS
    )
    joint_kth_fresh = _kth_largest(
        [float(r.get("n_fresh_select", 0)) for r in joint], MIN_SEEDS_FOR_PASS
    )

    measured_by_arm: Dict[str, Dict[str, float]] = {}
    for ctx in ARM_CONTEXTS:
        aid = ctx["id"]
        rows = _arm_rows(arm_results, aid)
        m: Dict[str, float] = {"routed_range_bounded": _arm_max_route(rows)}
        if ctx["packet_on"]:
            m["candidate_first_action_diversity_supra_floor"] = _worst_div(rows)[0]
            m["adequate_committed_window_sample"] = _worst_committed(rows)[0]
        if ctx["is_joint"]:
            m["joint_route_range_supra_floor"] = joint_kth_route
            m["adequate_fresh_selection_sample"] = joint_kth_fresh
        measured_by_arm[aid] = m

    arm_gates = [
        evaluate_arm_gate(ctx["id"], ctx, PRECONDITION_SPECS, measured_by_arm[ctx["id"]])
        for ctx in ARM_CONTEXTS
    ]
    gate = aggregate_arm_gates(arm_gates)
    gate_green_by_arm = {g["arm"]: bool(g["gate_green"]) for g in arm_gates}
    joint_green = bool(gate_green_by_arm.get("ARM_1_JOINT", False))
    alt_green = bool(gate_green_by_arm.get("ARM_2_ALT", False))
    shuf_green = bool(gate_green_by_arm.get("ARM_3_SHUF", False))

    # readiness requires ALL THREE packet arms green: the C1/C2 comparison is only
    # estimable if JOINT has a supra-floor routed range AND every packet arm has an
    # adequately-sampled, non-degenerate committed distribution.
    readiness_ok = bool(joint_green and alt_green and shuf_green)

    # ---- criteria ----------------------------------------------------------
    baseline_tv = _cross_seed_baseline_tv(joint)
    c1_tv_per_seed = _matched_seed_tv(joint, alt)
    c2_tv_per_seed = _matched_seed_tv(joint, shuf)
    c1_mean_tv = float(sum(c1_tv_per_seed.values()) / len(c1_tv_per_seed)) if c1_tv_per_seed else 0.0
    c2_mean_tv = float(sum(c2_tv_per_seed.values()) / len(c2_tv_per_seed)) if c2_tv_per_seed else 0.0
    c1_seeds_above_floor = sum(1 for v in c1_tv_per_seed.values() if v >= C_TV_FLOOR)
    c2_seeds_above_floor = sum(1 for v in c2_tv_per_seed.values() if v >= C_TV_FLOOR)
    c1_pass = bool(c1_seeds_above_floor >= MIN_SEEDS_FOR_PASS and c1_mean_tv > baseline_tv)
    c2_pass = bool(c2_seeds_above_floor >= MIN_SEEDS_FOR_PASS and c2_mean_tv > baseline_tv)

    # ---- label + direction (pre-registered grid) ---------------------------
    if not readiness_ok:
        label = "substrate_not_ready_requeue"
        overall_pass = False
        evidence_direction = "non_contributory"
    elif c1_pass and c2_pass:
        label = "joint_binding_behaviourally_load_bearing"
        overall_pass = True
        evidence_direction = "supports"
    elif c1_pass and not c2_pass:
        label = "multi_content_present_joint_not_isolated"
        overall_pass = False
        evidence_direction = "mixed"
    else:  # c1 fail
        label = "joint_indistinguishable_from_alternation"
        overall_pass = False
        evidence_direction = "weakens"

    # Route-range mode-distinctness (recorded diagnostic; NOT a hard gate -- SHUF ~0
    # is a structural control, so this is evidence the manipulation is mode-distinct
    # rather than a rescale-invisible scalar the 661 defect showed).
    route_range_by_arm = {
        "ARM_0_OFF": round(_mean_key(off, "route_range_mean"), 6),
        "ARM_1_JOINT": round(_mean_key(joint, "route_range_mean"), 6),
        "ARM_2_ALT": round(_mean_key(alt, "route_range_mean"), 6),
        "ARM_3_SHUF": round(_mean_key(shuf, "route_range_mean"), 6),
    }
    joint_gt_shuf_range = bool(route_range_by_arm["ARM_1_JOINT"] > route_range_by_arm["ARM_3_SHUF"])

    # non_degenerate: an EVIDENCE-run scoring gate. The committed-TV comparison is
    # not estimable unless the substrate is ready -> exclude from confidence scoring
    # (parallel to superseded / stale_substrate). Only an explicit False excludes.
    non_degenerate = readiness_ok
    if readiness_ok:
        degeneracy_reason = ""
    else:
        reds = [a for a, g in (("ARM_1_JOINT", joint_green),
                               ("ARM_2_ALT", alt_green),
                               ("ARM_3_SHUF", shuf_green)) if not g]
        degeneracy_reason = (
            "substrate not ready: packet arm(s) " + ",".join(reds) + " red on the "
            "arm-scoped readiness gate (JOINT routed coherence RANGE supra-floor / "
            "candidate first-action diversity / committed-window sample). The "
            "committed-action TV contrast is not estimable, so this run does not "
            "weight MECH-294 confidence -- it self-routes substrate_not_ready_requeue."
        )

    return {
        "readiness_ok": readiness_ok,
        "joint_gate_green": joint_green,
        "alt_gate_green": alt_green,
        "shuf_gate_green": shuf_green,
        "readiness": {
            "joint_route_range_mean": round(_mean_key(joint, "route_range_mean"), 6),
            "joint_kth_route_range": round(joint_kth_route, 6),
            "joint_kth_n_fresh_select": int(joint_kth_fresh),
            "route_floor": C0_ROUTE_FLOOR,
            "fresh_select_floor": FRESH_SELECT_FLOOR,
            "fresh_select_floor_derivation": (
                f"{NOMINAL_WINDOW_TICKS} nominal window ticks / "
                f"beta_rate_max_steps={BETA_RATE_MAX_STEPS} (MECH-093 worst-case cadence)"
            ),
            "diversity_floor": DIVERSITY_FLOOR,
            "committed_window_floor": COMMITTED_WINDOW_FLOOR,
            "min_seeds_required": MIN_SEEDS_FOR_PASS,
            "worst_diversity_joint": round(_worst_div(joint)[0], 4),
            "worst_diversity_alt": round(_worst_div(alt)[0], 4),
            "worst_diversity_shuf": round(_worst_div(shuf)[0], 4),
            "worst_committed_joint": int(_worst_committed(joint)[0]),
            "worst_committed_alt": int(_worst_committed(alt)[0]),
            "worst_committed_shuf": int(_worst_committed(shuf)[0]),
        },
        "criteria_pass": {"C1": c1_pass, "C2": c2_pass},
        "c1_joint_vs_alt": {
            "tv_per_seed": c1_tv_per_seed,
            "mean_tv": round(c1_mean_tv, 4),
            "seeds_above_floor": int(c1_seeds_above_floor),
            "tv_floor": C_TV_FLOOR,
            "cross_seed_baseline_tv": round(baseline_tv, 4),
            "exceeds_baseline": bool(c1_mean_tv > baseline_tv),
        },
        "c2_joint_vs_shuf": {
            "tv_per_seed": c2_tv_per_seed,
            "mean_tv": round(c2_mean_tv, 4),
            "seeds_above_floor": int(c2_seeds_above_floor),
            "tv_floor": C_TV_FLOOR,
            "cross_seed_baseline_tv": round(baseline_tv, 4),
            "exceeds_baseline": bool(c2_mean_tv > baseline_tv),
        },
        "route_range_per_arm_mean": route_range_by_arm,
        "route_active_frac_per_arm_mean": {
            "ARM_1_JOINT": round(_mean_key(joint, "route_active_frac"), 4),
            "ARM_2_ALT": round(_mean_key(alt, "route_active_frac"), 4),
            "ARM_3_SHUF": round(_mean_key(shuf, "route_active_frac"), 4),
        },
        "joint_route_range_gt_shuffled": joint_gt_shuf_range,
        "label": label,
        "overall_pass": overall_pass,
        "evidence_direction": evidence_direction,
        # ---- adjudication structures (skill Step 3.5), arm-scoped -----------
        "preconditions": gate["adjudication_preconditions"],
        "preconditions_scope_note": gate["per_arm_gate"]["preconditions_scope_note"],
        "per_arm_gate": gate["per_arm_gate"],
        "non_degenerate": non_degenerate,
        "degeneracy_reason": degeneracy_reason,
        "recorded_preconditions": [
            {
                "name": "per_arm_committed_and_route_sample",
                "kind": "recorded",
                "gating": False,
                "description": (
                    "Per-cell committed-window and fresh-selection sample for every "
                    "arm, recorded in full alongside the gating legs so both the "
                    "count quantifier (order statistics) and the all(...) quantifier "
                    "(worst cell) are independently checkable by a reader."
                ),
                "per_cell": [
                    {"cell": _cell_id(r),
                     "arm_id": r.get("arm_id"),
                     "binding_mode": r.get("binding_mode"),
                     "n_fresh_select": int(r.get("n_fresh_select", 0)),
                     "n_latched_ticks": int(r.get("n_latched_ticks", 0)),
                     "n_p1_ticks_past_window": int(r.get("n_p1_ticks_past_window", 0)),
                     "route_range_mean": float(r.get("route_range_mean", 0.0)),
                     "mean_first_action_classes": float(r.get("mean_first_action_classes", 0.0))}
                    for r in arm_results
                ],
            },
        ],
        "criteria": [
            {"name": "C1_joint_committed_dist_differs_from_alternation",
             "load_bearing": True, "passed": c1_pass},
            {"name": "C2_joint_committed_dist_differs_from_shuffled",
             "load_bearing": True, "passed": c2_pass},
        ],
        "criteria_non_degenerate": arm_criteria_non_degenerate(
            {"ARM_1_JOINT": [
                "C1_joint_committed_dist_differs_from_alternation",
                "C2_joint_committed_dist_differs_from_shuffled",
            ]},
            gate,
            extra={
                # C1 needs JOINT and ALT both green + a real routed range;
                # C2 needs JOINT and SHUF both green + a real routed range.
                "C1_joint_committed_dist_differs_from_alternation":
                    bool(joint_green and alt_green and joint_kth_route > C0_ROUTE_FLOOR),
                "C2_joint_committed_dist_differs_from_shuffled":
                    bool(joint_green and shuf_green and joint_kth_route > C0_ROUTE_FLOOR),
            },
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    t0 = time.perf_counter()

    # Design-time proof: no arm carries a precondition it provably cannot satisfy
    # from its pre-registered config (the V3-EXQ-785 check). SHUFFLED carries no
    # route-range-supra-floor gate, so its structural ~0 range is not a violation.
    assert_no_structurally_unsatisfiable_gate(PRECONDITION_SPECS, ARM_CONTEXTS)

    seeds = DRY_RUN_SEEDS if dry_run else SEEDS
    p0 = DRY_RUN_P0 if dry_run else P0_WARMUP_EPISODES
    p1 = DRY_RUN_P1 if dry_run else P1_MEASUREMENT_EPISODES
    steps = DRY_RUN_STEPS if dry_run else STEPS_PER_EPISODE
    measure_after = DRY_RUN_MEASURE_AFTER_TICK if dry_run else MEASURE_AFTER_TICK

    arm_results: List[Dict[str, Any]] = []
    for arm in ARMS:
        for seed in seeds:
            print(f"Seed {seed} Condition {arm['arm_id']}", flush=True)
            cell = _run_seed_arm(arm, seed, p0, p1, steps, measure_after)
            cell["arm_fingerprint"] = compute_arm_fingerprint(
                config_slice={
                    "arm": {k: arm[k] for k in ("arm_id", "packet_on", "binding_mode")},
                    "env_kwargs": ENV_KWARGS,
                    "sd056_weight": SD056_WEIGHT,
                    "p0_episodes": p0, "p1_episodes": p1, "steps_per_episode": steps,
                },
                seed=seed,
                script_path=Path(__file__),
                rng_fully_reset=True,
                extra_ineligible_reasons=["online_e2_training_stateful_per_cell"],
            )
            arm_results.append(cell)
            passed = cell.get("error_note") is None
            print(f"verdict: {'PASS' if passed else 'FAIL'}", flush=True)

    summary = _evaluate(arm_results)
    outcome = "PASS" if summary["overall_pass"] else "FAIL"

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{timestamp}_v3"

    manifest: Dict[str, Any] = {
        "schema_version": "v1",
        "run_id": run_id,
        "queue_id": QUEUE_ID,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": timestamp,
        "outcome": outcome,
        "result": outcome,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": CLAIM_IDS,
        "supersedes": SUPERSEDES,
        "per_arm_gate": summary["per_arm_gate"],
        "non_degenerate": summary["non_degenerate"],
        "degeneracy_reason": summary["degeneracy_reason"],
        "recorded_preconditions": summary["recorded_preconditions"],
        "evidence_direction": summary["evidence_direction"],
        "evidence_direction_per_claim": {"MECH-294": summary["evidence_direction"]},
        "evidence_direction_note": (
            "Supersedes V3-EXQ-840, which starved its committed-action measurement "
            "window on seeds 45/49 (self-contamination health depletion, an "
            "unintended gap in the HARM-FREE env design) -- fixed here by zeroing "
            "contaminated_harm; see module docstring MEASUREMENT-WINDOW FIX. "
            "MECH-294 behavioural falsifier (EVIDENCE run; weights confidence). Joint "
            "vs alternation vs shuffled theta-burst binding, per-candidate co-binding "
            "COHERENCE routed via modulatory_channel_route_source='coherence' through "
            "the route-range selection authority (landed 2026-06-10; readiness "
            "confirmed V3-EXQ-790/791a), read out on the E3 committed-action "
            "distribution. C1 (JOINT!=ALTERNATION) and C2 (JOINT!=SHUFFLED) are the "
            "load-bearing joint-clause criteria (memo S6/S7.3). "
            "label=joint_binding_behaviourally_load_bearing -> supports (both pass); "
            "multi_content_present_joint_not_isolated -> mixed (C1 pass, C2 fail; do "
            "NOT promote joint clause); joint_indistinguishable_from_alternation -> "
            "weakens (C1 fail -- Kay-2020 parsimonious outcome; route /failure-autopsy "
            "to disambiguate residual wiring vs genuine over-specification; hold "
            "stands); substrate_not_ready_requeue -> non_contributory + "
            "non_degenerate=false (readiness gate red -- JOINT routed coherence RANGE "
            "below floor / candidate diversity degenerate / committed window starved; "
            "EXCLUDED from scoring). The readiness gate asserts the SAME statistic the "
            "authority carves (route_range, a cross-candidate RANGE) as the criteria "
            "route on, so a committed-TV null is a claim result ONLY when the "
            "manipulation is a genuine supra-floor range and NOT the rescale-invisible "
            "scalar the 661 run showed. This run does NOT resolve the primary lit "
            "falsifier (Kay-2020 cross-cycle theta), which is out-of-substrate for V3."
        ),
        "interpretation": {
            "label": summary["label"],
            "preconditions": summary["preconditions"],
            "preconditions_scope_note": summary["preconditions_scope_note"],
            "criteria": summary["criteria"],
            "criteria_non_degenerate": summary["criteria_non_degenerate"],
            "routing": {
                "joint_binding_behaviourally_load_bearing": "PASS/supports -> substrate-side joint-vs-alternation falsification passed; governance may weigh lifting the 2026-04-26 hold on the joint clause",
                "multi_content_present_joint_not_isolated": "FAIL/mixed -> do NOT promote joint clause; refine to a genuinely conjunctive readout (memo S7.3)",
                "joint_indistinguishable_from_alternation": "FAIL/weakens -> /failure-autopsy (wiring vs over-specification); joint clause not promotable, hold stands",
                "substrate_not_ready_requeue": "FAIL/non_contributory + non_degenerate=false -> re-queue at higher P0 budget or fix diversity/route wiring; do NOT weaken",
            },
        },
        "dry_run": bool(dry_run),
        "config": {
            "seeds": seeds,
            "p0_warmup_episodes": p0,
            "p1_measurement_episodes": p1,
            "steps_per_episode": steps,
            "measure_after_tick": measure_after,
            "env_kwargs": ENV_KWARGS,
            "arms": [{k: a[k] for k in ("arm_id", "label", "packet_on", "binding_mode")} for a in ARMS],
            "sd056_weight": SD056_WEIGHT,
            "modulatory_channel_route_source": "coherence",
            "modulatory_authority_gain": 0.5,
            "theta_packet_coherence_hold_weight": 0.5,
            "theta_packet_bias_scale": 0.1,
            "thresholds": {
                "c_tv_floor": C_TV_FLOOR,
                "c0_route_floor": C0_ROUTE_FLOOR,
                "c0_magnitude_ceil": C0_MAGNITUDE_CEIL,
                "diversity_floor": DIVERSITY_FLOOR,
                "committed_window_floor": COMMITTED_WINDOW_FLOOR,
                "fresh_select_floor": FRESH_SELECT_FLOOR,
                "beta_rate_max_steps": BETA_RATE_MAX_STEPS,
                "nominal_window_ticks": NOMINAL_WINDOW_TICKS,
                "min_seeds_for_pass": MIN_SEEDS_FOR_PASS,
            },
        },
        "acceptance_criteria": {
            "readiness_substrate_ready": summary["readiness_ok"],
            "C1_joint_vs_alternation": summary["criteria_pass"]["C1"],
            "C2_joint_vs_shuffled": summary["criteria_pass"]["C2"],
            "overall_pass": summary["overall_pass"],
        },
        "summary": summary,
        "arm_results": arm_results,
    }

    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.json"

    if not dry_run:
        out_path = write_flat_manifest(
            manifest,
            out_dir,
            dry_run=False,
            config=manifest.get("config"),
            seeds=SEEDS,
            script_path=Path(__file__),
            started_at=t0,
            z_goal_stream_stats=_ZG.stats(),
        )
        print(f"Manifest written: {out_path}", flush=True)
    else:
        out_path = Path("/dev/null")
        print("Dry run -- manifest not written.", flush=True)

    print(f"Outcome: {outcome} (label={summary['label']}, direction={summary['evidence_direction']})", flush=True)
    for k, v in manifest["acceptance_criteria"].items():
        print(f"  {k}: {v}", flush=True)
    pag = summary["per_arm_gate"]
    print(f"  per_arm_gate: green={pag['green_arms']} red={pag['red_arms']}", flush=True)
    print(
        f"  C1 joint-vs-alt: mean_tv={summary['c1_joint_vs_alt']['mean_tv']} "
        f"(floor {C_TV_FLOOR}, baseline {summary['c1_joint_vs_alt']['cross_seed_baseline_tv']}, "
        f"seeds>=floor {summary['c1_joint_vs_alt']['seeds_above_floor']})",
        flush=True,
    )
    print(
        f"  C2 joint-vs-shuf: mean_tv={summary['c2_joint_vs_shuf']['mean_tv']} "
        f"(floor {C_TV_FLOOR}, seeds>=floor {summary['c2_joint_vs_shuf']['seeds_above_floor']})",
        flush=True,
    )
    print(f"  route_range_per_arm: {summary['route_range_per_arm_mean']}", flush=True)

    manifest["manifest_path"] = str(out_path)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "V3-EXQ-840b MECH-294 behavioural falsifier (supersedes V3-EXQ-840): "
            "joint/alternation/shuffled theta-burst binding, per-candidate coherence "
            "routed via the route-range authority, committed-action distribution "
            "readout, with contaminated_harm=0.0 fixing the committed-window "
            "measurement defect"
        )
    )
    parser.add_argument("--dry-run", action="store_true", help="Short smoke run.")
    args = parser.parse_args()

    result = run_experiment(dry_run=args.dry_run)

    if args.dry_run:
        sys.exit(0)

    _outcome_raw = str(result.get("outcome", "FAIL")).upper()
    _outcome = _outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL"
    emit_outcome(
        outcome=_outcome,
        manifest_path=str(result.get("manifest_path", Path("/dev/null"))),
    )
