"""
V3-EXQ-1006 -- SD-e1-rollout-consistency-training: the ratified var-bar
PORTFOLIO (GOV-FANOUT-1) for registered question sd_e1_var_bar_readout_crush.
After ITEM 3 (rollout-endpoint InfoNCE, V3-EXQ-1000) cleared the cr_ratio(h=1)
>= 0.1 bar on 6/6 seeds, WHY does per-action endpoint divergence at the E1
output not reach the goal-proximity readout -- e1coe_score_var(h=1) still
36x-1700x short of 0.002?

Claims: [] (diagnostic; the question's claims are INV-088 / MECH-135, which
keep pending_retest_after_substrate regardless of outcome here). DIAGNOSTIC.
unblocks_claims (per the SD entry, NOT claim_ids): MECH-135, INV-088.
hypothesis_space_qid: sd_e1_var_bar_readout_crush (registered 2026-09-04 by
failure_autopsy_V3-EXQ-1000_2026-09-04, user-confirmed, REE_assembly 34fd92e301).

SLEEP DRIVER: none (no sleep flags set).

WHY THIS RUN EXISTS
---------------------------------------------------------------------------
V3-EXQ-1000 (confirmed autopsy 2026-09-04, user decisions Q1-Q4) separated the
two evaluator bars by evidence: the cr_ratio crush WAS the objective class
(H-objective-class-divergence confirmed); the goal-proximity variance bar is a
DIFFERENT statistic and stays open. Its h=1 signature: ARM_RSD's endpoint
centroid is SHRUNK (norm 0.54-0.81 vs 1.41-2.17 same-start incumbents, real
1.39-1.89) and under-dispersed (spread 0.33-0.62x real); at depth (h>=10) one
seed reaches BOTH bars, but only at 3-3.7x real spread. The lineage has NEVER
recorded the goal-proximity variance of its own REAL endpoints (108b: means
only), so whether the 0.002 bar is reachable at real spread is unknown.

The user ratified (Q2, 2026-09-04) ONE new EXQ carrying all three registered
legs as a portfolio -- NOT a lettered 1000a (refused). This is that run.

THE THREE LEGS (verbatim from the registry; declared nulls carried verbatim)
---------------------------------------------------------------------------
H-fidelity-anchor (LEAD, learning-signal axis): the RSD endpoints have a
  shrunk centroid and are under-dispersed at h=1 because InfoNCE carries no
  accuracy term that survives a confident K-way ordering; a fidelity-anchored
  divergence objective (ITEM 3 + an endpoint-accuracy term) restores the
  centroid to the real band with real-scale spread, and the goal-proximity
  variance follows.
  DECLARED NULL: ARM_RSD_ANCHOR's centroid_norm stays outside the real band on
  a majority of seeds at h=1 OR its e1coe_score_var(h=1) is within 2x of
  ARM_RSD's.
H-readout-saturation (MEASURE, measurement axis): the 0.002 var bar, through
  the bounded 1/(1+||z-z_goal||^2) readout, may demand ~3-4x over-dispersion
  relative to real spread. If real endpoints' own goal_proximity variance is
  < 0.002, the bar as registered is an instrument target to re-register, not a
  substrate target.
  DECLARED NULL: real-endpoint goal_proximity variance >= 0.002 on a majority
  of seeds (bar reachable at real spread; hypothesis eliminated).
H-goal-orthogonal-dispersion (REPRESENTATION axis): the dispersion RSD induces
  projects weakly onto the goal axis, so a goal-blind objective moves the
  readout only by brute over-dispersion; a goal-axis-aware objective or a
  full-endpoint readout is needed at real spread.
  DECLARED NULL: ARM_RSD's goal-axis fraction of spread variance is within 2x
  of the real endpoints' (dispersion is not goal-orthogonal; eliminated).

ARMS (all ITEM-1-ON, matched grad steps by construction, identical random
policy / env / seeds -- 976's control, carried through 1000):
  ARM_OFF            depth-0 single-step teacher-forced incumbent (1000's
                     ARM_OFF verbatim; the lineage's OFF baseline).
  ARM_RSD            ITEM 3 rollout_sequence_divergence_loss, defaults
                     (1000's ARM_RSD verbatim; the incumbent under repair).
  ARM_RSD_ANCHOR_EP  LEAD anchor: ITEM 3 InfoNCE + w_t * plain ENDPOINT MSE
                     between the predicted h=RSD_HORIZON endpoint and the
                     observed endpoint, on the SAME K sampled windows. (w_t is
                     the gradient-matched dose -- see ANCHOR_GRAD_AUTHORITY.)
                     Purely endpoint-based, like the InfoNCE it anchors: it adds
                     accuracy pressure at exactly the point the contrastive
                     scores, and carries NO per-step intermediate-state MSE --
                     the term V3-EXQ-976 showed DAMPS per-action divergence.
  ARM_RSD_ANCHOR_RC  the autopsy's OTHER named composition: ITEM 3 InfoNCE +
                     w_t * ITEM 2 rollout_consistency_loss (H=5, decay 0.5) on
                     the same K windows (full per-step targets); same
                     gradient-matched dose rule.
                     At decay 0.5 ~52% of that term's weight sits on step 1,
                     so it is the accuracy-heavy form. Included so a null on
                     ONE anchor form cannot be mis-read as "anchoring fails"
                     when it is "this anchor form fails" (Step 2.5b portfolio
                     verdict-aliasing audit): the two forms differ on exactly
                     the axis 976 found load-bearing (per-step accuracy
                     pressure vs endpoint-only).
The two anchor arms are one registered leg (H-fidelity-anchor) with its
composition choice -- left open by the autopsy ("rollout_consistency_loss at
H=5 decay 0.5, or plain endpoint MSE") -- de-aliased rather than picked. The
LEAD arm is ARM_RSD_ANCHOR_EP; the hypothesis is ELIMINATED only if BOTH
anchor arms land on a declared-null disjunct, SUPPORTED if either restores the
centroid AND lifts the variance (the manifest names which).

Training composition is DRIVER-LEVEL (no new substrate): the InfoNCE term is
the substrate method rollout_sequence_divergence_loss, byte-identical to
ARM_RSD's call; the endpoint MSE is F.mse_loss over a second
predict_long_horizon rollout of the same batch (hidden state saved/reset/
restored exactly as the loss methods do internally); the RC term is the
substrate method rollout_consistency_loss. THE DOSE IS GRADIENT-MATCHED:
the red-team pass (and the authoring probe reproducing it) found that at a
fixed unit weight the anchor's gradient norm on E1's parameters is tens of
times smaller than the InfoNCE's once E1 is trained, i.e. an undosed anchor
whose null would be unattributable. So on every anchored tick w_t =
ANCHOR_GRAD_AUTHORITY * ||grad L_rsd|| / ||grad L_anchor|| (clamped), the
anchor gets EQUAL gradient authority with the InfoNCE, and the realised
authority is a readiness precondition (anchor_engaged) on the anchor arms.
The raw ratio (what a unit weight would have delivered) is recorded per arm
and per episode. Stated limitation: the EP anchor constrains the h=5 endpoint
only, while the label reads h=1 (the h=1 centroid is reached only through the
autoregressive chain); the RC anchor weights step 1 most, so the two forms
bracket that axis. All four arms take exactly one E1 optimiser step per env
step under the uniform TRAIN_WINDOW_H trigger (976/1000 device; on a
degenerate InfoNCE batch the anchor is skipped too, so the arms stay
matched), so e1_grad_steps_matched is true by construction and measured.

READOUTS, at EVERY horizon checkpoint (registered set {1,5,10,20,30}; the
lineage's {2,3} kept so 1000's cells stay comparable), per (arm, seed):
  * e1coe_score_var(h)  variance of goal_proximity over the 40 candidate
                        rollout endpoints (hybrid E1/E2 readout; the
                        C3_VAR_THRESHOLD=0.002 bar) -- and cr_ratio(h) with
                        the CR_ROLLOUT_COLLAPSE_RATIO=0.1 bar; BOTH bars read
                        per horizon (1000 learning #1).
  * real-endpoint goal_proximity VARIANCE and goal_distance VARIANCE over the
    Phase 4b real endpoint samples per horizon -- THE RECORDING GAP, the
    H-readout-saturation leg; per-sample values recorded.
  * centroid-norm band: arm centroid_norm / real centroid_norm (Phase 4b), and
    / same-seed ARM_OFF centroid_norm (the "same-start incumbent" reading);
    band = [CENTROID_BAND_LO, CENTROID_BAND_HI] = [0.75, 4/3], pre-registered
    symmetric in log space: 1000's populations sit at ~0.4x (RSD) and
    ~1.0-1.2x (incumbents), so the band separates them without touching
    either.
  * spread ratio vs real (Phase 4b) and vs the same-start branched real set.
  * endpoint-fidelity MSE for the SAME (start, sequence): Phase 4d branches
    a copy.deepcopy(env) snapshot taken at the warmup state and RESTORES the
    agent's latent state (agent._current_latent, _last_action, _step_count,
    E1 hidden state) per candidate sequence -- verified bit-exact against a
    straight continuation in the authoring probe (agent deepcopy is not
    possible: a stored module reference) -- executes the candidate sequence
    for real, and senses z_world at every checkpoint. EVAL only; training
    is untouched. This yields, per h, 40 REAL endpoints for exactly the
    starts and sequences the rollouts scored, so it also gives a SAME-START
    real centroid / spread / goal_proximity variance beside the lineage's
    from-reset Phase 4b set (both recorded; the registered leg reads 4b).
  * goal-axis projection: fraction of spread variance along the unit axis
    (z_goal - centroid) for real endpoints (4b and 4d) and each arm's
    rollouts; the H-goal-orthogonal-dispersion leg.
  * E1-alone rollout readout beside the hybrid at every h incl. h=30
    (scope clause carried from V3-EXQ-980 -> 1000; NEVER gating).

DECISION RULE (pre-registered; majority = n_seeds//2+1 = 4 of 6; all legs
read at h=1 for the label, and at every h for the record):
  Leg A (per anchor arm X, against ARM_RSD at the same seed):
    centroid_restored[X]: CENTROID_BAND_LO <= centroid_norm(X)/centroid_norm
      (real 4b) <= CENTROID_BAND_HI on >= majority seeds.
    var_lifted[X]: e1coe_score_var_h1(X) >= VAR_LIFT_FACTOR (2.0) x
      e1coe_score_var_h1(ARM_RSD) on >= majority seeds.
    bars_cleared[X]: cr_ratio_h1(X) >= 0.1 AND e1coe_score_var_h1(X) >= 0.002
      on >= majority seeds.
    cr_bar_retained[X]: cr_ratio_h1(X) >= 0.1 on >= majority seeds (the
      anchor must not buy the centroid back by re-crushing divergence).
    arm reading: "supported" iff centroid_restored AND var_lifted AND
      cr_bar_retained (an anchor must not buy the centroid back by
      re-crushing divergence -- red-team #6); "centroid_restored_var_lifted_
      cr_bar_lost" if the first two hold but the cr bar is lost;
      "centroid_restored_var_flat" (null disjunct 2) iff centroid_restored
      and not var_lifted; "centroid_not_restored" (null disjunct 1)
      otherwise (var_lifted without centroid restoration is reported as
      "var_lifted_centroid_not_restored" -- a finding, still the null).
    leg tag: anchor_clears_both_bars_h1 if any green anchor arm has
      bars_cleared; else anchor_restores_centroid_lifts_var if any is
      "supported"; else ..._cr_bar_lost (mixed) if any is that; else, if
      EVERY green anchor arm sits on a declared-null disjunct:
      anchor_restores_centroid_var_flat / anchor_fails_centroid with state
      ELIMINATED only when BOTH anchor arms are green (red-team #4/#5), and
      the same tag + "_partial" with state mixed when one form is unscored;
      else anchor_mixed. anchor_arms_vacuous if no anchor arm passed its own
      gate (leg unresolved).
  Leg B (arm-invariant -- the encoder is identical across arms at a seed,
    Phase 0a is seed-deterministic and E1 training never touches it; read
    off the first green arm): n_ge = #seeds with the SAME-START real
    endpoints' (Phase 4d) goal_proximity variance(h=1) >= 0.002.
    realvar_reaches_bar (H eliminated) if n_ge >= majority; realvar_below_bar
    (H supported: instrument target) if #seeds below >= majority;
    realvar_mixed otherwise. WHY SAME-START (red-team #3, 2026-09-06): the
    registry named Phase 4b's from-reset set, but the bar applies to 40
    rollouts fanning out from ONE start, and at h=1 those reach at most
    n_actions distinct endpoints -- the same-start real set is the matched
    denominator; the from-reset variance adds start dispersion and is an
    UPPER BOUND, recorded beside it (n_reaches_bar_from_reset). The registry
    text is honoured in the record and the stricter comparator routes.
  Leg C: r = goal_axis_fraction(ARM_RSD rollouts, h=1) / goal_axis_fraction
    (real 4b, h=1) per seed. rsd_goal_axis_matches_real (eliminated) if
    0.5 <= r <= 2.0 on >= majority; rsd_goal_orthogonal (supported) if
    r < 0.5 on >= majority; rsd_goal_aligned_excess if r > 2.0 on >=
    majority; rsd_goal_axis_undetermined if fewer than majority seeds have
    a finite r; rsd_goal_axis_mixed otherwise.
  interpretation.label = "<legA>__<legB>__<legC>"; "substrate_not_ready_
  requeue" only when NO arm passes its gate. A leg whose required arm(s)
  are red reads "<leg>_undetermined_arm_red".
  RELATIVE VAR READING (recorded, not in the label): if Leg B says the
  absolute bar exceeds what real endpoints produce, the meaningful anchor
  test is against the real-endpoint variance itself; per anchor arm,
  var_reaches_real_endpoint_var = e1coe_score_var_h1(X) >= real(4b) var(h=1)
  on >= majority seeds, recorded under interpretation.relative_var_reading.

dv_headroom (DECLARED, kind "dv_headroom", RECORDED NOT ADJUDICATING): the
DV whose bar this run reads is e1coe_score_var(h=1) against 0.002. Its
headroom control is the SAME-START real endpoint set's own goal_proximity
variance at h=1 (Phase 4d; the from-reset Phase 4b value recorded beside) --
exactly the H-readout-saturation measurement. achievable = the
majority-order statistic over seeds (the value at least `majority` seeds
reach), required = C3_VAR_THRESHOLD x margin 1.0. It is built with
experiments/_metrics.dv_headroom_check and emitted in
interpretation.RECORDED_preconditions (the indexer's non-adjudicating
sibling channel, surfaced in pending_review under its own non-blocking
heading), NOT in interpretation.preconditions, and NOT passed through
p0_readiness_gate's raising path: here an UNMET headroom IS the registered
leg-B finding (the bar is an instrument target), which the confirmed autopsy
routes to "re-register the bar before licensing any objective", not to
substrate_not_ready_requeue -- and the indexer's flat preconditions[] read
would otherwise adjudicate the WHOLE run precondition_unmet on the very
outcome the portfolio predicts, vacating legs A and C (red-team #2,
2026-09-06, confirmed against build_experiment_indexes._compute_adjudication).
Stated here so nobody "fixes" it into a gate.

PER-ARM GATES (experiments/_lib/precondition_gate.py; no arm's red vacates
another's): 1000's eight readiness preconditions plus branched_same_start_
nondegenerate_h1 and anchor_engaged; rsd_objective_engaged (a property of
the sampled batch, identical across the three RSD-bearing arms by
construction) applies to the RSD-bearing arms; rsd_action_sensitive (1000's
red-team #1/#2 identity-shortcut control) GATES ARM_RSD ONLY -- on an anchor
arm a lost action-sensitivity is a result of the anchor (divergence
re-crushed, read via cr_bar_retained), not a readiness failure, and gating it
there would let the manipulation move its own gate (red-team #4); the anchor
arms' ratios are recorded non-adjudicating. anchor_engaged applies to the
anchor arms. Each arm is measured at its own worst seed. non_degenerate =
any arm green.

GOV-REUSE-1 (Step 2.4): decisive readouts = (a) real-endpoint goal_proximity
variance per horizon -- recorded in NO manifest (reanalysis_query: 0 carry it;
the 1000 manifest stores only cr_real spread/centroid/n, not the endpoint
vectors, so it is not derivable post-hoc; the autopsy names this as the
recording gap); (b) e1coe_score_var(h=1) on an anchored RSD arm -- no such arm
has ever run. Not recoverable -> run.
STEP 2.5 / 2.5a: ITEM 1/2/3 all IMPLEMENTED (ree-v3/CLAUDE.md; SD entry
implementation_log items 1-3); authoring probe against the working tree
confirmed rollout_sequence_divergence_loss / rollout_consistency_loss /
predict_long_horizon accept the [K,...] batch shapes used here, that the
composed loss back-propagates into E1, and the branching restore above.
STEP 2.5b: claim_ids is []. Brake counts for the unblocks claims: MECH-135 1,
INV-088 2 (both MECH-457 fanout autopsies, a different mechanism and a
different substrate) -- this is a NEW EXQ number, a diagnostic portfolio
answering an explicit fanout_recommendation on a substrate enriched since
(ITEM 3 landed 2026-09-03); not the re-derive loop.
STEP 2.5c: SD-e1-rollout-consistency-training (corrupting; e1_deep.py::
forward / ::predict_long_horizon) is this run's OWN target -- the sanctioned
exception to its gate, per its own item-2/3 substrate_gate_notes. Other open
corrupting entries: contextmemory-write-path (ContextMemory.write is called
only from agent.py's select_action/update paths, never from sense() or the
direct E1 calls this driver makes); mode-governance-engagement
(use_salience_coordinator default False; the _et_commit site is in the E3
tick path, never reached -- no select_action call); SD-082
(use_lateral_pfc_analog default False; compute_bias unreachable). None on
this driver's path.
CROSS-MACHINE-CLASS SAFETY: no torch.multinomial anywhere; every DV is a
continuous rollout statistic.
Priority 60 (front-critical: CURRENT_FRONT.md critical-path item 1).
machine_affinity any (cloud-preferred; 1000 ran in 15 min on ree-cloud-2).
EXPERIMENT_PURPOSE = "diagnostic" -- excluded from governance confidence
scoring.
red-team (fable, 2026-09-06): CONTESTED -- 7 findings, all dispositioned;
#1 (anchor undosed at unit weight: gradient-matched dose + anchor_engaged
gate), #2 (dv_headroom in preconditions[] would vacate the run: moved to
recorded_preconditions[]), #3 (leg B denominator: same-start set routes,
from-reset recorded), #4 (eliminated on one green arm: both-green required),
#5 (all null readings count), #6 (cr_bar_retained conditions "supported"),
#7 (starved tick counted: n_e1_grad_steps_real recorded). Full disposition
in the queue entry note.
ASCII-only output (repo rule).
"""

import copy
import itertools
import sys
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.goal import GoalConfig, GoalState
from ree_core.utils.config import REEConfig
from experiment_protocol import emit_outcome
from experiments.pack_writer import write_flat_manifest
from experiments._lib.zworld_p0_warmup import run_zworld_p0
from experiments._lib.capability_eval import RandomPolicy
from experiments._lib.zworld_encoder_guard import (
    latent_stack_snapshot,
    assert_world_encoder_trained,
)
from experiments._lib.arm_fingerprint import arm_cell
from experiments._lib.precondition_gate import (
    PreconditionSpec,
    assert_no_structurally_unsatisfiable_gate,
    evaluate_arm_gate,
    aggregate_arm_gates,
    arm_criteria_non_degenerate,
)
from experiments._metrics import dv_headroom_check


EXPERIMENT_TYPE = "v3_exq_1006_sd_e1_var_bar_portfolio_fidelity_anchor"
CLAIM_IDS: List[str] = []
UNBLOCKS_CLAIMS: List[str] = ["MECH-135", "INV-088"]
EXPERIMENT_PURPOSE = "diagnostic"
SUPERSEDES = None
HYPOTHESIS_SPACE_QID = "sd_e1_var_bar_readout_crush"

ANCHOR_REACHABILITY_EXEMPT = (
    "e1_grad_steps_matched IS the degeneracy definition (cross-arm step-count gap "
    "under the uniform TRAIN_WINDOW_H trigger, constructed to be exactly 0); "
    "rsd_objective_engaged replicates rollout_sequence_divergence_loss's own "
    "argmax+unique(dim=0) non-degeneracy check; rsd_action_sensitive is the "
    "row-permuted action-assignment control from V3-EXQ-1000 (red-team #1/#2). "
    "None is a hand-narrowed predicate on an external signal."
)

AGENT_SEED_ORDER_EXEMPT = (
    "Every within-cell comparison is scored off the literal same agent object; "
    "cross-arm comparisons are the A/B itself and are seed-matched via arm_cell()."
)

# ---------------------------------------------------------------------------
# Pre-registered thresholds
# ---------------------------------------------------------------------------
CR_REAL_FLOOR = 1e-4               # unchanged from V3-EXQ-954/965/968/976/1000
CR_ROLLOUT_COLLAPSE_RATIO = 0.1    # evaluator bar 1 (cr_ratio) -- read per h
C3_VAR_THRESHOLD = 0.002           # evaluator bar 2 (e1coe_score_var) -- read per h
ZWORLD_P0_EPISODES = 60            # SD-070 encoder warmup -- matches lineage
N_REAL_SAMPLES = 40                # Phase 4b per-checkpoint target sample count
MIN_REAL_SAMPLES_PER_HORIZON = 10  # readiness floor: surviving real samples (4b and 4d)
HORIZON_CHECKPOINTS_FULL = [1, 2, 3, 5, 10, 20, 30]
REGISTERED_HORIZONS = [1, 5, 10, 20, 30]   # the registry's "every horizon" set

CENTROID_BAND_LO = 0.75            # arm centroid_norm / real centroid_norm, lower
CENTROID_BAND_HI = 4.0 / 3.0       # ... upper (symmetric in log space)
VAR_LIFT_FACTOR = 2.0              # registry null: anchor var within 2x of RSD's
GOAL_AXIS_WITHIN_FACTOR = 2.0      # registry null: RSD goal-axis frac within 2x real
DV_HEADROOM_MARGIN = 1.0           # bare feasibility of the 0.002 bar at real spread

# ITEM 2 objective parameters (ARM_RSD_ANCHOR_RC's accuracy term).
RC_HORIZON = 5
RC_DECAY = 0.5
ANCHOR_W_RC = 1.0                  # base multiplier on the RC term (dose is w_t, above)

# ITEM 3 objective parameters -- all E1Config defaults, as in V3-EXQ-1000.
RSD_HORIZON = 5
RSD_TEMPERATURE = 0.1
RSD_MIN_BATCH_CLASSES = 2
RSD_WEIGHT = 1.0
RSD_BATCH_K = 8
RSD_BUFFER_MAX = 256

# Anchor DOSE (red-team finding #1, 2026-09-06; confirmed by the authoring
# probe): at a fixed w=1.0 the accuracy term's gradient norm on E1's parameters
# is tens of times SMALLER than the InfoNCE term's once E1 is trained (tau=0.1
# sharpens the InfoNCE logits), so a unit-weight anchor is a few percent of the
# update and a Leg-A null could not be told from "anchor never dosed". The dose
# is therefore GRADIENT-MATCHED per tick:
#     w_t = ANCHOR_GRAD_AUTHORITY * ||grad_theta L_rsd|| / ||grad_theta L_anchor||
# (both norms over E1's parameters via autograd.grad, treated as constants),
# clamped to [ANCHOR_W_MIN, ANCHOR_W_MAX]. ANCHOR_GRAD_AUTHORITY=1.0 gives the
# anchor EQUAL gradient authority with the InfoNCE it anchors -- a canonical,
# pre-registered dose rather than an arbitrary scalar. The realised w_t, the raw
# ratio and the realised authority are recorded per arm and per episode;
# anchor_engaged (a readiness precondition on the anchor arms only) asserts the
# realised authority reached ANCHOR_AUTHORITY_FLOOR on every seed.
ANCHOR_W_EP = 1.0                  # base multiplier on the endpoint-MSE term
ANCHOR_GRAD_AUTHORITY = 1.0
ANCHOR_W_MIN = 1e-3
ANCHOR_W_MAX = 1e4
ANCHOR_AUTHORITY_FLOOR = 0.5

# Action-sensitivity control floor (V3-EXQ-1000 red-team #1/#2), unchanged.
RSD_ACTION_SENSITIVITY_RATIO_FLOOR = 1.05

TRAIN_WINDOW_H = 5                 # == RC_HORIZON == RSD_HORIZON, deliberately

OFF_ARM = "ARM_OFF"
RSD_ARM = "ARM_RSD"
ANCHOR_ARMS = ["ARM_RSD_ANCHOR_EP", "ARM_RSD_ANCHOR_RC"]
LEAD_ANCHOR_ARM = "ARM_RSD_ANCHOR_EP"
ARM_ORDER = [OFF_ARM, RSD_ARM] + ANCHOR_ARMS
RSD_BEARING_ARMS = [RSD_ARM] + ANCHOR_ARMS

ARM_CONFIGS: Dict[str, Dict[str, Any]] = {
    "ARM_OFF": {
        "action_conditioned_transition": True,
        "action_cond_unzero_self_slot": True,
        "output_proj_residual": False,
        "e1_rollout_consistency_enabled": False,
        "e1_rollout_sequence_divergence_enabled": False,
        "e1_loss": "single_step_depth0",
        "anchor": None,
    },
    "ARM_RSD": {
        "action_conditioned_transition": True,
        "action_cond_unzero_self_slot": True,
        "output_proj_residual": False,
        "e1_rollout_consistency_enabled": False,
        "e1_rollout_sequence_divergence_enabled": True,
        "e1_loss": "rollout_sequence_divergence",
        "anchor": None,
    },
    "ARM_RSD_ANCHOR_EP": {
        "action_conditioned_transition": True,
        "action_cond_unzero_self_slot": True,
        "output_proj_residual": False,
        "e1_rollout_consistency_enabled": False,
        "e1_rollout_sequence_divergence_enabled": True,
        "e1_loss": "rollout_sequence_divergence",
        "anchor": "endpoint_mse",
    },
    "ARM_RSD_ANCHOR_RC": {
        "action_conditioned_transition": True,
        "action_cond_unzero_self_slot": True,
        "output_proj_residual": False,
        # the RC flag is enabled so the substrate method is on its documented
        # path; the driver calls it explicitly on the sampled batch.
        "e1_rollout_consistency_enabled": True,
        "e1_rollout_sequence_divergence_enabled": True,
        "e1_loss": "rollout_sequence_divergence",
        "anchor": "rollout_consistency",
    },
}

SEEDS_DEFAULT = [42, 123, 7, 2024, 17, 31]   # the 1000 seed set, verbatim (n=6)


# ---------------------------------------------------------------------------
# Helpers (unchanged from V3-EXQ-1000 unless noted)
# ---------------------------------------------------------------------------

def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


def _env_kwargs() -> Dict[str, Any]:
    """Env config, unchanged from V3-EXQ-954/965/976/1000."""
    return dict(
        size=10, num_hazards=2, num_resources=4,
        hazard_harm=0.02, env_drift_interval=8, env_drift_prob=0.05,
        proximity_harm_scale=0.03, proximity_benefit_scale=0.04,
        proximity_approach_threshold=0.15, hazard_field_decay=0.5,
        resource_respawn_on_consume=True,
    )


def _build_agent(
    seed: int, world_dim: int, self_dim: int, arm: str,
) -> Tuple[REEAgent, CausalGridWorldV2]:
    env = CausalGridWorldV2(seed=seed, **_env_kwargs())
    arm_cfg = ARM_CONFIGS[arm]
    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=self_dim,
        world_dim=world_dim,
        alpha_world=0.9,
        alpha_self=0.3,
        action_conditioned_transition=arm_cfg["action_conditioned_transition"],
        action_cond_unzero_self_slot=arm_cfg["action_cond_unzero_self_slot"],
        output_proj_residual=arm_cfg["output_proj_residual"],
        e1_rollout_consistency_enabled=arm_cfg["e1_rollout_consistency_enabled"],
        e1_rollout_consistency_weight=ANCHOR_W_RC,
        e1_rollout_consistency_horizon=RC_HORIZON,
        e1_rollout_consistency_horizon_weights_decay=RC_DECAY,
        e1_rollout_sequence_divergence_enabled=arm_cfg["e1_rollout_sequence_divergence_enabled"],
        e1_rollout_sequence_divergence_weight=RSD_WEIGHT,
        e1_rollout_sequence_divergence_horizon=RSD_HORIZON,
        e1_rollout_sequence_divergence_temperature=RSD_TEMPERATURE,
        e1_rollout_sequence_divergence_min_batch_classes=RSD_MIN_BATCH_CLASSES,
    )
    config.latent.unified_latent_mode = False
    # from_dims swallows unknown kwargs silently; assert every knob landed.
    assert bool(config.e1.action_conditioned_transition) is True, arm
    assert bool(config.e1.output_proj_residual) == bool(arm_cfg["output_proj_residual"]), arm
    assert bool(config.e1.e1_rollout_consistency_enabled) == bool(arm_cfg["e1_rollout_consistency_enabled"]), arm
    assert bool(config.e1.e1_rollout_sequence_divergence_enabled) == bool(
        arm_cfg["e1_rollout_sequence_divergence_enabled"]
    ), arm
    if arm_cfg["e1_rollout_sequence_divergence_enabled"]:
        assert int(config.e1.e1_rollout_sequence_divergence_horizon) == RSD_HORIZON, arm
        assert int(config.e1.e1_rollout_sequence_divergence_min_batch_classes) == RSD_MIN_BATCH_CLASSES, arm
    if arm_cfg["e1_rollout_consistency_enabled"]:
        assert int(config.e1.e1_rollout_consistency_horizon) == RC_HORIZON, arm
        assert abs(float(config.e1.e1_rollout_consistency_horizon_weights_decay) - RC_DECAY) < 1e-12, arm
    agent = REEAgent(config)
    return agent, env


# ---------------------------------------------------------------------------
# Phase 0a: SD-070 sanctioned z_world encoder warmup (unchanged from lineage)
# ---------------------------------------------------------------------------

def _run_zworld_p0_warmup(
    agent: REEAgent, seed: int, zworld_p0_episodes: int, steps_per_episode: int,
    dry_run: bool = False,
) -> Dict[str, Any]:
    before = latent_stack_snapshot(agent)
    warmup_env = CausalGridWorldV2(seed=seed, **_env_kwargs())
    p0a_report = run_zworld_p0(
        agent, warmup_env, seed, zworld_p0_episodes, steps_per_episode,
        policy=RandomPolicy(seed), label="v3_exq_1006 P0a (SD-070 z_world encoder)",
        dry_run=dry_run,
    )
    encoder_report = assert_world_encoder_trained(
        agent, before, p0=zworld_p0_episodes, strict=False,
        context=EXPERIMENT_TYPE,
        escape_hint="pass zworld_p0_episodes=0 for a deliberate frozen-encoder run",
    )
    return {**p0a_report, **encoder_report}


# ---------------------------------------------------------------------------
# Phase 0b: bespoke E1/E2 training, action-conditioned, uniform trailing
# window (H=TRAIN_WINDOW_H) shared by all four arms.
# ---------------------------------------------------------------------------

def _single_step_loss_state_preserving(
    agent: REEAgent, initial: torch.Tensor, action: torch.Tensor, target: torch.Tensor,
) -> torch.Tensor:
    """Single-step teacher-forced loss from a ZERO hidden state, hidden state
    saved and restored around the call (976's B1 symmetrisation). ARM_OFF."""
    saved = agent.e1._hidden_state
    agent.e1.reset_hidden_state()
    try:
        e1_pred, _ = agent.e1(initial, horizon=1, actions=action)
        return F.mse_loss(e1_pred[:, 0, :], target)
    finally:
        agent.e1._hidden_state = saved


def _endpoint_rollout_state_preserving(
    agent: REEAgent, init: torch.Tensor, acts: torch.Tensor, horizon: int,
) -> torch.Tensor:
    """Batched predict_long_horizon with the SAME hidden-state save/reset/
    restore idiom the substrate's own loss methods use. Returns [K, h, total]."""
    saved = agent.e1._hidden_state
    agent.e1.reset_hidden_state()
    try:
        return agent.e1.predict_long_horizon(init, horizon=horizon, actions=acts[:, :horizon, :])
    finally:
        agent.e1._hidden_state = saved


def _grad_norm(loss: torch.Tensor, params: List[torch.nn.Parameter]) -> float:
    """||grad_params loss||_2 as a float; graph retained for the real backward."""
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    parts = [g.reshape(-1) for g in grads if g is not None]
    if not parts:
        return 0.0
    return float(torch.cat(parts).norm().item())


def _train_agent(
    agent: REEAgent,
    env: CausalGridWorldV2,
    seed: int,
    n_episodes: int,
    steps_per_episode: int,
    e1_call_counter: Dict[str, int],
    arm: str,
) -> Dict[str, Any]:
    """Phase 0b: E1/E2 training on a random policy (agent.sense() exactly ONCE
    per env step, under torch.no_grad(), so Phase 0a's encoder is untouched).

    UNIFORM TRIGGER for all four arms: exactly one E1 optimiser step per env
    step once a trailing window of TRAIN_WINDOW_H observed latents exists.

    single_step_depth0            ARM_OFF (1000 verbatim).
    rollout_sequence_divergence   ARM_RSD (1000 verbatim): buffer the full
                                  H-step window (initial, action sequence,
                                  per-step targets, endpoint); sample K
                                  windows; InfoNCE on endpoints.
      + anchor "endpoint_mse"     ARM_RSD_ANCHOR_EP: + ANCHOR_W_EP *
                                  F.mse_loss(pred endpoint, observed endpoint)
                                  on the SAME K windows.
      + anchor "rollout_consistency"
                                  ARM_RSD_ANCHOR_RC: + ANCHOR_W_RC *
                                  agent.e1.rollout_consistency_loss(init,
                                  per-step targets, actions, H, decay 0.5)
                                  on the SAME K windows.
    The action-sensitivity control is computed on the InfoNCE term ALONE in
    every RSD-bearing arm, so rsd_action_sensitive is one statistic across
    the three arms.
    """
    torch.manual_seed(seed + 2000)
    random.seed(seed + 2000)
    agent.train()

    # Isolated RNG streams (1000): buffer sampling and the control permutation
    # must never touch the shared `random` stream driving the action policy.
    rsd_sample_rng = random.Random(seed + 9000)
    rsd_shuffle_rng = random.Random(seed + 9500)

    arm_cfg = ARM_CONFIGS[arm]
    loss_kind = str(arm_cfg["e1_loss"])
    anchor_kind = arm_cfg["anchor"]
    H = int(TRAIN_WINDOW_H)

    e1_params = [q for q in agent.e1.parameters() if q.requires_grad]
    opt_e1 = optim.Adam(agent.e1.parameters(), lr=1e-3)
    opt_e2 = optim.Adam(agent.e2.parameters(), lr=1e-3)

    rsd_buffer: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    rsd_n_engaged = 0
    rsd_n_degenerate = 0
    rsd_n_buffer_starved = 0
    rsd_real_loss_sum = 0.0
    rsd_shuffled_loss_sum = 0.0
    rsd_n_sensitivity_checks = 0
    rsd_term_sum = 0.0
    anchor_term_sum = 0.0
    n_anchor_ticks = 0
    anchor_w_sum = 0.0
    anchor_raw_ratio_sum = 0.0
    anchor_authority_sum = 0.0
    anchor_n_clamped = 0
    anchor_n_skipped_degenerate = 0

    stats: Dict[str, Any] = {
        "e1_loss_kind": loss_kind,
        "anchor_kind": anchor_kind,
        "train_window_h": H,
        "n_e1_grad_steps": 0,
        "n_e1_grad_steps_real": 0,
        "n_e2_grad_steps": 0,
        "n_windows": 0,
        "trained_loss_mean": 0.0,
        "per_episode_trained_loss": [],
        "per_episode_rsd_term": [],
        "per_episode_anchor_term": [],
        "n_nonfinite_losses": 0,
        "rsd_n_engaged": 0,
        "rsd_n_degenerate": 0,
        "rsd_n_buffer_starved": 0,
        "rsd_engaged_frac": 0.0,
        "rsd_action_sensitivity_ratio": float("nan"),
        "rsd_term_mean": float("nan"),
        "anchor_term_mean": float("nan"),
        "n_anchor_ticks": 0,
        "anchor_w_mean": float("nan"),
        "anchor_raw_grad_ratio_mean": float("nan"),
        "anchor_authority_mean": float("nan"),
        "anchor_n_clamped": 0,
        "anchor_n_skipped_degenerate": 0,
        "per_episode_anchor_w": [],
        "per_episode_anchor_raw_grad_ratio": [],
    }
    trained_sum = 0.0

    for ep in range(n_episodes):
        _, obs_dict = env.reset()
        agent.reset()
        ep_loss_e1 = 0.0
        ep_loss_e2 = 0.0
        ep_rsd = 0.0
        ep_anchor = 0.0
        ep_anchor_w = 0.0
        ep_anchor_raw_ratio = 0.0
        ep_anchor_n = 0
        n_steps = 0

        totals: List[torch.Tensor] = []
        actions: List[torch.Tensor] = []
        latent_prev: Optional[object] = None
        action_prev: Optional[torch.Tensor] = None

        for _ in range(steps_per_episode):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            with torch.no_grad():
                latent_curr = agent.sense(obs_body, obs_world)

            action_idx = random.randint(0, env.action_dim - 1)
            action_curr = _action_to_onehot(action_idx, env.action_dim, agent.device)
            agent.record_executed_action(action_curr)

            total_curr = torch.cat([latent_curr.z_self, latent_curr.z_world], dim=-1).detach()
            totals.append(total_curr)
            actions.append(action_curr)

            # E2 (not under test): identical in every arm.
            if latent_prev is not None:
                opt_e2.zero_grad()
                z_self_pred = agent.e2.predict_next_self(latent_prev.z_self.detach(), action_prev)
                e2_loss = F.mse_loss(z_self_pred, latent_curr.z_self.detach())
                e2_loss.backward()
                opt_e2.step()
                ep_loss_e2 += e2_loss.item()
                stats["n_e2_grad_steps"] += 1

            if len(totals) >= H + 1:
                i = len(totals) - 1 - H
                initial = totals[i]
                window_targets = torch.stack(totals[i + 1:i + 1 + H], dim=1)  # [1,H,total]
                window_acts = torch.stack(actions[i:i + H], dim=1)            # [1,H,action]
                rsd_term_val = float("nan")
                anchor_term_val = float("nan")

                if loss_kind == "single_step_depth0":
                    trained = _single_step_loss_state_preserving(agent, initial, actions[i], totals[i + 1])
                    e1_call_counter["n_e1_calls"] += 1
                    e1_call_counter["n_e1_calls_nonzero_action"] += 1
                elif loss_kind == "rollout_sequence_divergence":
                    rsd_buffer.append((
                        initial.squeeze(0).detach().clone(),          # [total]
                        window_acts.squeeze(0).detach().clone(),      # [H, action]
                        window_targets.squeeze(0).detach().clone(),   # [H, total]
                        totals[i + H].squeeze(0).detach().clone(),    # [total]
                    ))
                    if len(rsd_buffer) > RSD_BUFFER_MAX:
                        rsd_buffer.pop(0)
                    if len(rsd_buffer) < RSD_MIN_BATCH_CLASSES:
                        rsd_n_buffer_starved += 1
                        opt_e1.zero_grad()
                        stats["n_e1_grad_steps"] += 1
                        stats["n_windows"] += 1
                        n_steps += 1
                        _, _, done, _, obs_dict = env.step(action_curr)
                        latent_prev = latent_curr
                        action_prev = action_curr
                        if done:
                            break
                        continue
                    k = min(RSD_BATCH_K, len(rsd_buffer))
                    sampled_idx = rsd_sample_rng.sample(range(len(rsd_buffer)), k)
                    sampled = [rsd_buffer[j] for j in sampled_idx]
                    batch_init = torch.stack([w[0] for w in sampled], dim=0)       # [K,total]
                    batch_acts = torch.stack([w[1] for w in sampled], dim=0)       # [K,H,action]
                    batch_targets = torch.stack([w[2] for w in sampled], dim=0)    # [K,H,total]
                    batch_endpoints = torch.stack([w[3] for w in sampled], dim=0)  # [K,total]
                    seq_classes = batch_acts[:, :RSD_HORIZON, :].argmax(dim=-1)
                    n_distinct = int(torch.unique(seq_classes, dim=0).shape[0])
                    if n_distinct >= RSD_MIN_BATCH_CLASSES:
                        rsd_n_engaged += 1
                    else:
                        rsd_n_degenerate += 1
                    rsd_term = agent.e1.rollout_sequence_divergence_loss(
                        batch_init, batch_acts, batch_endpoints, horizon=RSD_HORIZON,
                    ) * RSD_WEIGHT
                    e1_call_counter["n_e1_calls"] += 1
                    e1_call_counter["n_e1_calls_nonzero_action"] += 1
                    rsd_term_val = float(rsd_term.item())

                    if anchor_kind is None:
                        trained = rsd_term
                    elif n_distinct < RSD_MIN_BATCH_CLASSES:
                        # degenerate batch: the InfoNCE took its zero branch; the
                        # anchor is skipped too (a no-learning tick, exactly as
                        # in ARM_RSD) so the arms stay matched tick for tick.
                        trained = rsd_term
                        anchor_n_skipped_degenerate += 1
                    else:
                        if anchor_kind == "endpoint_mse":
                            preds = _endpoint_rollout_state_preserving(
                                agent, batch_init, batch_acts, RSD_HORIZON,
                            )
                            e1_call_counter["n_e1_calls"] += 1
                            e1_call_counter["n_e1_calls_nonzero_action"] += 1
                            anchor_raw = F.mse_loss(preds[:, -1, :], batch_endpoints) * ANCHOR_W_EP
                        elif anchor_kind == "rollout_consistency":
                            anchor_raw = agent.e1.rollout_consistency_loss(
                                batch_init, batch_targets, actions=batch_acts,
                                horizon=RC_HORIZON, horizon_weights_decay=RC_DECAY,
                            ) * ANCHOR_W_RC
                            e1_call_counter["n_e1_calls"] += 1
                            e1_call_counter["n_e1_calls_nonzero_action"] += 1
                        else:
                            raise ValueError(f"unknown anchor kind: {anchor_kind}")
                        # GRADIENT-MATCHED DOSE (see ANCHOR_GRAD_AUTHORITY): the
                        # anchor's per-tick weight is set so its gradient norm on
                        # E1's parameters equals the InfoNCE term's, times the
                        # pre-registered authority. Both norms are constants here.
                        g_rsd = _grad_norm(rsd_term, e1_params)
                        g_anc = _grad_norm(anchor_raw, e1_params)
                        raw_ratio = g_rsd / max(g_anc, 1e-12)
                        w_target = raw_ratio * ANCHOR_GRAD_AUTHORITY
                        w_t = min(max(w_target, ANCHOR_W_MIN), ANCHOR_W_MAX)
                        if w_t != w_target:
                            anchor_n_clamped += 1
                        realised_authority = (w_t * g_anc) / max(g_rsd, 1e-12)
                        trained = rsd_term + w_t * anchor_raw
                        anchor_term_val = float(anchor_raw.item())
                        anchor_w_sum += w_t
                        anchor_raw_ratio_sum += raw_ratio
                        anchor_authority_sum += realised_authority
                        n_anchor_ticks += 1
                        ep_anchor_w += w_t
                        ep_anchor_raw_ratio += raw_ratio
                        ep_anchor_n += 1

                    # ACTION-SENSITIVITY CONTROL (1000 red-team #1/#2), on the
                    # InfoNCE term alone, identical statistic in all RSD arms.
                    with torch.no_grad():
                        shuffle_perm = torch.tensor(
                            rsd_shuffle_rng.sample(range(k), k), dtype=torch.long,
                        )
                        batch_acts_shuffled = batch_acts[shuffle_perm]
                        shuffled_loss = agent.e1.rollout_sequence_divergence_loss(
                            batch_init, batch_acts_shuffled, batch_endpoints,
                            horizon=RSD_HORIZON,
                        )
                        rsd_real_loss_sum += rsd_term_val / max(RSD_WEIGHT, 1e-12)
                        rsd_shuffled_loss_sum += float(shuffled_loss.item())
                        rsd_n_sensitivity_checks += 1
                else:
                    raise ValueError(f"unknown e1_loss kind: {loss_kind}")

                val = float(trained.item())
                if not (val == val):
                    stats["n_nonfinite_losses"] += 1
                opt_e1.zero_grad()
                trained.backward()
                opt_e1.step()
                stats["n_e1_grad_steps"] += 1
                stats["n_e1_grad_steps_real"] += 1
                stats["n_windows"] += 1
                trained_sum += val
                ep_loss_e1 += val
                if rsd_term_val == rsd_term_val:
                    rsd_term_sum += rsd_term_val
                    ep_rsd += rsd_term_val
                if anchor_term_val == anchor_term_val:
                    anchor_term_sum += anchor_term_val
                    ep_anchor += anchor_term_val
                n_steps += 1

            _, _, done, _, obs_dict = env.step(action_curr)
            latent_prev = latent_curr
            action_prev = action_curr
            if done:
                break

        stats["per_episode_trained_loss"].append(ep_loss_e1 / max(n_steps, 1))
        stats["per_episode_rsd_term"].append(ep_rsd / max(n_steps, 1))
        stats["per_episode_anchor_term"].append(ep_anchor / max(n_steps, 1))
        stats["per_episode_anchor_w"].append(ep_anchor_w / ep_anchor_n if ep_anchor_n else float("nan"))
        stats["per_episode_anchor_raw_grad_ratio"].append(ep_anchor_raw_ratio / ep_anchor_n if ep_anchor_n else float("nan"))
        if (ep + 1) % 20 == 0 or (ep + 1) == n_episodes:
            print(
                f"  [train] label {arm} seed={seed} ep {ep+1}/{n_episodes} "
                f"e1_loss={ep_loss_e1/max(n_steps,1):.5f} "
                f"rsd={ep_rsd/max(n_steps,1):.5f} anchor={ep_anchor/max(n_steps,1):.5f} "
                f"anchor_w={(ep_anchor_w/ep_anchor_n) if ep_anchor_n else float('nan'):.3g} "
                f"e2_loss={ep_loss_e2/max(n_steps,1):.5f}",
                flush=True,
            )

    n_w = max(int(stats["n_windows"]), 1)
    stats["trained_loss_mean"] = trained_sum / n_w
    stats["rsd_n_engaged"] = rsd_n_engaged
    stats["rsd_n_degenerate"] = rsd_n_degenerate
    stats["rsd_n_buffer_starved"] = rsd_n_buffer_starved
    denom = rsd_n_engaged + rsd_n_degenerate
    stats["rsd_engaged_frac"] = (rsd_n_engaged / denom) if denom > 0 else 0.0
    if rsd_n_sensitivity_checks > 0 and rsd_real_loss_sum > 1e-12:
        stats["rsd_action_sensitivity_ratio"] = rsd_shuffled_loss_sum / rsd_real_loss_sum
    stats["rsd_n_sensitivity_checks"] = rsd_n_sensitivity_checks
    if denom > 0:
        stats["rsd_term_mean"] = rsd_term_sum / denom
    if n_anchor_ticks > 0:
        stats["anchor_term_mean"] = anchor_term_sum / n_anchor_ticks
        stats["anchor_w_mean"] = anchor_w_sum / n_anchor_ticks
        stats["anchor_raw_grad_ratio_mean"] = anchor_raw_ratio_sum / n_anchor_ticks
        stats["anchor_authority_mean"] = anchor_authority_sum / n_anchor_ticks
    stats["n_anchor_ticks"] = n_anchor_ticks
    stats["anchor_n_clamped"] = anchor_n_clamped
    stats["anchor_n_skipped_degenerate"] = anchor_n_skipped_degenerate

    agent.eval()
    print(
        f"  [train] Done. {n_episodes} episodes; e1_grad_steps={stats['n_e1_grad_steps']} "
        f"trained_loss_mean={stats['trained_loss_mean']:.6e} "
        f"rsd_engaged={rsd_n_engaged} rsd_degenerate={rsd_n_degenerate} "
        f"rsd_buffer_starved={rsd_n_buffer_starved} anchor_ticks={n_anchor_ticks} "
        f"anchor_w_mean={stats['anchor_w_mean']:.3g} anchor_authority_mean={stats['anchor_authority_mean']:.3g} "
        f"raw_grad_ratio_mean={stats['anchor_raw_grad_ratio_mean']:.3g}",
        flush=True,
    )
    return stats


# ---------------------------------------------------------------------------
# Phase 1: goal template (unchanged from lineage)
# ---------------------------------------------------------------------------

def _collect_goal_template(
    agent: REEAgent, env: CausalGridWorldV2, seed: int, max_steps: int,
) -> Tuple[torch.Tensor, str]:
    torch.manual_seed(seed)
    random.seed(seed)
    _, obs_dict = env.reset()
    agent.reset()

    for _ in range(max_steps):
        obs_body = obs_dict["body_state"]
        obs_world = obs_dict["world_state"]
        with torch.no_grad():
            latent = agent.sense(obs_body, obs_world)
        agent.clock.advance()
        action_idx = random.randint(0, env.action_dim - 1)
        action = _action_to_onehot(action_idx, env.action_dim, agent.device)
        agent.record_executed_action(action)
        _, _, done, info, obs_dict = env.step(action)
        if info.get("transition_type", "none") == "resource":
            print(
                f"  [Phase1] Resource contact, z_world_norm={latent.z_world.norm().item():.3f}",
                flush=True,
            )
            return latent.z_world.detach(), "resource_contact"
        if done:
            _, obs_dict = env.reset()
            agent.reset()

    print("  [Phase1] WARNING: no resource contact -- using fallback unit vector", flush=True)
    z_goal = torch.randn(1, agent.config.latent.world_dim)
    z_goal = F.normalize(z_goal, dim=-1)
    return z_goal, "fallback_unit_vector"


# ---------------------------------------------------------------------------
# Phase 2: warmup state WITH a branchable snapshot (NEW). The lineage's
# _get_warmup_state returned the latent of the observation BEFORE its final
# env.step, so the env it left behind was one step past the state the
# rollouts start from. Here the warmup runs n_warmup_steps (sense, act, step)
# then senses the resulting observation ONCE more without stepping, and the
# snapshot (env deepcopy + agent latent state) is taken at exactly that
# observation -- so the rollouts' start state and the branches' start state
# are the same env state, by construction.
# ---------------------------------------------------------------------------

def _get_warmup_state_with_snapshot(
    agent: REEAgent, env: CausalGridWorldV2, seed: int, n_warmup_steps: int,
) -> Tuple[torch.Tensor, torch.Tensor, List[int], CausalGridWorldV2, Dict[str, Any]]:
    torch.manual_seed(seed + 1000)
    random.seed(seed + 1000)
    _, obs_dict = env.reset()
    agent.reset()
    warmup_actions: List[int] = []

    for _ in range(n_warmup_steps):
        obs_body = obs_dict["body_state"]
        obs_world = obs_dict["world_state"]
        with torch.no_grad():
            agent.sense(obs_body, obs_world)
        agent.clock.advance()
        action_idx = random.randint(0, env.action_dim - 1)
        warmup_actions.append(action_idx)
        action = _action_to_onehot(action_idx, env.action_dim, agent.device)
        agent.record_executed_action(action)
        _, _, done, _, obs_dict = env.step(action)
        if done:
            _, obs_dict = env.reset()
            agent.reset()
            warmup_actions = []

    obs_body = obs_dict["body_state"]
    obs_world = obs_dict["world_state"]
    with torch.no_grad():
        latent = agent.sense(obs_body, obs_world)

    env_snapshot = copy.deepcopy(env)
    latent_snapshot = {
        "current_latent": copy.deepcopy(agent._current_latent),
        "last_action": agent._last_action,
        "step_count": agent._step_count,
        "e1_hidden_state": agent.e1._hidden_state,
    }
    return latent.z_self.detach(), latent.z_world.detach(), warmup_actions, env_snapshot, latent_snapshot


def _restore_latent_snapshot(agent: REEAgent, snap: Dict[str, Any]) -> None:
    agent._current_latent = copy.deepcopy(snap["current_latent"])
    agent._last_action = snap["last_action"]
    agent._step_count = snap["step_count"]
    agent.e1._hidden_state = snap["e1_hidden_state"]


# ---------------------------------------------------------------------------
# Phase 3: candidate sequences (unchanged from lineage)
# ---------------------------------------------------------------------------

def _generate_candidate_sequences(
    n_sequences: int, horizon: int, n_actions: int, seed: int,
) -> List[List[int]]:
    torch.manual_seed(seed + 500)
    random.seed(seed + 500)
    seqs = []
    for _ in range(n_sequences):
        seq = [random.randint(0, n_actions - 1) for _ in range(horizon)]
        seqs.append(seq)
    return seqs


# ---------------------------------------------------------------------------
# Phase 4 (hybrid E1/E2 readout) and 4-e1-alone (980/1000 sibling readout,
# never gating). Unchanged from V3-EXQ-1000.
# ---------------------------------------------------------------------------

def _score_sequence_hybrid_multi_horizon(
    agent: REEAgent,
    z_self_start: torch.Tensor,
    z_world_start: torch.Tensor,
    action_sequence: List[int],
    goal_state: GoalState,
    self_dim: int,
    checkpoints: List[int],
    e1_call_counter: Dict[str, int],
) -> Dict[int, Tuple[float, torch.Tensor]]:
    device = agent.device
    n_actions = agent.config.e2.action_dim
    checkpoint_set = set(checkpoints)

    agent.e1.reset_hidden_state()

    z_self_curr = z_self_start.clone()
    z_world_curr = z_world_start.clone()
    out: Dict[int, Tuple[float, torch.Tensor]] = {}

    for step_idx, a_idx in enumerate(action_sequence, start=1):
        action = _action_to_onehot(a_idx, n_actions, device)
        total_curr = torch.cat([z_self_curr, z_world_curr], dim=-1)
        with torch.no_grad():
            e1_preds, _ = agent.e1(total_curr, horizon=1, actions=action)
        e1_call_counter["n_e1_calls"] += 1
        e1_call_counter["n_e1_calls_nonzero_action"] += (
            1 if float(action.abs().sum()) > 0.0 else 0
        )
        z_world_next = e1_preds[0, 0, self_dim:].unsqueeze(0)
        with torch.no_grad():
            z_self_next = agent.e2.predict_next_self(z_self_curr, action)
        z_self_curr = z_self_next
        z_world_curr = z_world_next

        if step_idx in checkpoint_set:
            score = float(goal_state.goal_proximity(z_world_curr).item())
            out[step_idx] = (score, z_world_curr.detach().clone())

    return out


def _score_sequence_e1_alone_multi_horizon(
    agent: REEAgent,
    z_self_start: torch.Tensor,
    z_world_start: torch.Tensor,
    action_sequence: List[int],
    goal_state: GoalState,
    self_dim: int,
    checkpoints: List[int],
    e1_call_counter: Dict[str, int],
) -> Dict[int, Tuple[float, torch.Tensor]]:
    device = agent.device
    n_actions = agent.config.e2.action_dim
    max_h = max(checkpoints)
    checkpoint_set = set(checkpoints)

    agent.e1.reset_hidden_state()
    total_0 = torch.cat([z_self_start, z_world_start], dim=-1)
    action_tensors = [
        _action_to_onehot(a_idx, n_actions, device) for a_idx in action_sequence[:max_h]
    ]
    action_seq_tensor = torch.stack(action_tensors, dim=1)

    with torch.no_grad():
        preds = agent.e1.predict_long_horizon(total_0, horizon=max_h, actions=action_seq_tensor)
    e1_call_counter["n_e1_calls"] += 1
    e1_call_counter["n_e1_calls_nonzero_action"] += 1

    out: Dict[int, Tuple[float, torch.Tensor]] = {}
    for step_idx in range(1, max_h + 1):
        if step_idx not in checkpoint_set:
            continue
        z_world_h = preds[0, step_idx - 1, self_dim:].unsqueeze(0)
        score = float(goal_state.goal_proximity(z_world_h).item())
        out[step_idx] = (score, z_world_h.detach().clone())

    return out


# ---------------------------------------------------------------------------
# Phase 4b: real z_world sample at every checkpoint, from env.reset (lineage,
# unchanged). Returns z_world vectors so goal statistics can be computed.
# ---------------------------------------------------------------------------

def _collect_real_zworld_sample_multi_horizon(
    agent: REEAgent, env: CausalGridWorldV2, seed: int, n_samples: int,
    checkpoints: List[int],
) -> Dict[int, List[torch.Tensor]]:
    torch.manual_seed(seed + 3000)
    random.seed(seed + 3000)
    max_h = max(checkpoints)
    checkpoint_set = set(checkpoints)
    samples_by_h: Dict[int, List[torch.Tensor]] = {h: [] for h in checkpoints}

    for _ in range(n_samples):
        _, obs_dict = env.reset()
        agent.reset()
        for step_idx in range(1, max_h + 1):
            action_idx = random.randint(0, env.action_dim - 1)
            action = _action_to_onehot(action_idx, env.action_dim, agent.device)
            agent.record_executed_action(action)
            _, _, done, _, obs_dict = env.step(action)
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world)
            if step_idx in checkpoint_set:
                samples_by_h[step_idx].append(latent.z_world.detach())
            if done:
                break

    return samples_by_h


# ---------------------------------------------------------------------------
# Phase 4d (NEW): branched SAME-START real endpoints for the SAME candidate
# sequences the rollouts scored. EVAL ONLY. For each sequence: deepcopy the
# warmup env snapshot, restore the agent latent snapshot, execute the
# sequence for real, sense z_world at every checkpoint. Returns per-h dicts
# keyed by sequence index (a sequence that terminates early is absent from
# the deeper checkpoints; surviving counts are recorded and gated).
# ---------------------------------------------------------------------------

def _collect_branched_real_endpoints(
    agent: REEAgent,
    env_snapshot: CausalGridWorldV2,
    latent_snapshot: Dict[str, Any],
    seqs: List[List[int]],
    checkpoints: List[int],
) -> Tuple[Dict[int, Dict[int, torch.Tensor]], int]:
    max_h = max(checkpoints)
    checkpoint_set = set(checkpoints)
    by_h: Dict[int, Dict[int, torch.Tensor]] = {h: {} for h in checkpoints}
    n_terminated = 0
    n_actions = env_snapshot.action_dim

    for s_idx, seq in enumerate(seqs):
        env_b = copy.deepcopy(env_snapshot)
        _restore_latent_snapshot(agent, latent_snapshot)
        for step_idx, a_idx in enumerate(seq[:max_h], start=1):
            action = _action_to_onehot(a_idx, n_actions, agent.device)
            agent.record_executed_action(action)
            _, _, done, _, obs_dict = env_b.step(action)
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world)
            if step_idx in checkpoint_set:
                by_h[step_idx][s_idx] = latent.z_world.detach().clone()
            if done:
                n_terminated += 1
                break

    # leave the agent at the warmup snapshot afterwards, not mid-branch
    _restore_latent_snapshot(agent, latent_snapshot)
    return by_h, n_terminated


# ---------------------------------------------------------------------------
# Endpoint-set statistics
# ---------------------------------------------------------------------------

def _contrast_ratio(vectors: List[torch.Tensor]) -> Dict[str, float]:
    """CR = spread / ||centroid||. Unchanged from lineage."""
    if len(vectors) < 2:
        return {"spread": 0.0, "centroid_norm": 0.0, "contrast_ratio": float("nan"), "n": len(vectors)}
    stacked = torch.cat(vectors, dim=0)
    centroid = stacked.mean(dim=0, keepdim=True)
    centroid_norm = float(centroid.norm().item())
    deviations = stacked - centroid
    spread = float(torch.sqrt((deviations.pow(2).sum(dim=-1)).mean()).item())
    cr = (spread / centroid_norm) if centroid_norm > 1e-12 else float("nan")
    return {"spread": spread, "centroid_norm": centroid_norm, "contrast_ratio": cr, "n": len(vectors)}


def _goal_stats(vectors: List[torch.Tensor], goal_state: GoalState) -> Dict[str, Any]:
    """goal_proximity / goal_distance VARIANCE (and means, and the per-sample
    values) over a set of z_world vectors -- the H-readout-saturation leg's
    recording-gap statistic. Unbiased variance, as torch.var default (the same
    estimator e1coe_score_var uses)."""
    if len(vectors) < 2:
        return {"n": len(vectors), "prox_var": float("nan"), "dist_var": float("nan"),
                "prox_mean": float("nan"), "dist_mean": float("nan"),
                "prox_values": [], "dist_values": []}
    stacked = torch.cat(vectors, dim=0)
    with torch.no_grad():
        prox = goal_state.goal_proximity(stacked)
        dist = goal_state.goal_distance(stacked)
    return {
        "n": int(stacked.shape[0]),
        "prox_var": float(prox.var().item()),
        "dist_var": float(dist.var().item()),
        "prox_mean": float(prox.mean().item()),
        "dist_mean": float(dist.mean().item()),
        "prox_values": [float(v) for v in prox.tolist()],
        "dist_values": [float(v) for v in dist.tolist()],
    }


def _goal_axis_fraction(vectors: List[torch.Tensor], z_goal: torch.Tensor) -> Dict[str, float]:
    """Fraction of the endpoint set's spread variance that lies along the unit
    goal axis u = (z_goal - centroid)/||.||: sum_i ((e_i - c).u)^2 /
    sum_i ||e_i - c||^2. NaN when the set is degenerate or the centroid sits
    on the goal. Also returns the raw along-axis variance."""
    if len(vectors) < 2:
        return {"fraction": float("nan"), "along_axis_var": float("nan"),
                "total_var": float("nan"), "axis_norm": float("nan")}
    stacked = torch.cat(vectors, dim=0)
    centroid = stacked.mean(dim=0, keepdim=True)
    axis = (z_goal.reshape(1, -1) - centroid)
    axis_norm = float(axis.norm().item())
    dev = stacked - centroid
    total = float(dev.pow(2).sum().item())
    if axis_norm < 1e-9 or total < 1e-18:
        return {"fraction": float("nan"), "along_axis_var": float("nan"),
                "total_var": total, "axis_norm": axis_norm}
    u = axis / axis_norm
    along = float((dev @ u.t()).pow(2).sum().item())
    return {"fraction": along / total, "along_axis_var": along / max(stacked.shape[0] - 1, 1),
            "total_var": total / max(stacked.shape[0] - 1, 1), "axis_norm": axis_norm}


def _fidelity_mse(
    rollout_by_seq: Dict[int, torch.Tensor], real_by_seq: Dict[int, torch.Tensor],
) -> Dict[str, Any]:
    """Per-sequence MSE between the rollout's z_world at h and the branched
    REAL z_world at h for the same (start, sequence); mean over survivors."""
    common = sorted(set(rollout_by_seq.keys()) & set(real_by_seq.keys()))
    if not common:
        return {"n": 0, "mse_mean": float("nan"), "mse_values": []}
    vals = [float(F.mse_loss(rollout_by_seq[i], real_by_seq[i]).item()) for i in common]
    return {"n": len(common), "mse_mean": float(sum(vals) / len(vals)), "mse_values": vals}


# ---------------------------------------------------------------------------
# Phase 4c (positive-control cross-reference): one-step per-action divergence
# probe, DIRECT E1 channel. Unchanged from lineage.
# ---------------------------------------------------------------------------

def _one_step_action_divergence(
    agent: REEAgent,
    z_self_0: torch.Tensor,
    z_world_0: torch.Tensor,
    self_dim: int,
    e1_call_counter: Dict[str, int],
) -> Dict[str, Any]:
    device = agent.device
    n_actions = agent.config.e2.action_dim
    total_curr = torch.cat([z_self_0, z_world_0], dim=-1)

    predictions: List[torch.Tensor] = []
    n_direct_supply_ok = 0
    for a_idx in range(n_actions):
        agent.e1.reset_hidden_state()
        action = _action_to_onehot(a_idx, n_actions, device)
        is_direct_supply_ok = (
            action is not None
            and float(action.abs().sum().item()) == 1.0
            and int((action != 0).sum().item()) == 1
        )
        n_direct_supply_ok += 1 if is_direct_supply_ok else 0
        with torch.no_grad():
            e1_preds, _ = agent.e1(total_curr, horizon=1, actions=action)
        e1_call_counter["n_e1_calls"] += 1
        e1_call_counter["n_e1_calls_nonzero_action"] += 1 if is_direct_supply_ok else 0
        z_world_next = e1_preds[0, 0, self_dim:].unsqueeze(0)
        predictions.append(z_world_next.detach())

    pairwise_dists = [
        float((predictions[i] - predictions[j]).norm().item())
        for i, j in itertools.combinations(range(len(predictions)), 2)
    ]
    cr = _contrast_ratio(predictions)

    return {
        "n_actions": n_actions,
        "pairwise_dists": pairwise_dists,
        "pairwise_dist_mean": float(sum(pairwise_dists) / len(pairwise_dists)) if pairwise_dists else 0.0,
        "pairwise_dist_min": float(min(pairwise_dists)) if pairwise_dists else 0.0,
        "pairwise_dist_max": float(max(pairwise_dists)) if pairwise_dists else 0.0,
        "contrast_ratio": cr,
        "n_direct_supply_ok": n_direct_supply_ok,
        "direct_action_supply_fraction": (n_direct_supply_ok / n_actions) if n_actions else 0.0,
    }


# ---------------------------------------------------------------------------
# Single-cell (seed, arm) runner
# ---------------------------------------------------------------------------

def run_cell(
    seed: int,
    arm: str,
    world_dim: int,
    self_dim: int,
    n_train_episodes: int,
    steps_per_episode: int,
    n_sequences: int,
    rollout_horizon: int,
    n_warmup_steps: int,
    goal_max_steps: int,
    zworld_p0_episodes: int,
    n_real_samples: int,
    checkpoints: List[int],
    dry_run: bool = False,
) -> Tuple[Dict[str, Any], REEAgent]:
    print(f"\n[EXQ-1006] seed={seed} arm={arm}", flush=True)
    print(f"Seed {seed} Condition {arm}", flush=True)

    agent, env = _build_agent(seed, world_dim, self_dim, arm)
    e1_call_counter = {"n_e1_calls": 0, "n_e1_calls_nonzero_action": 0}

    print(f"[EXQ-1006] Phase 0a: SD-070 z_world encoder warmup ({zworld_p0_episodes} eps)...", flush=True)
    readiness_report = _run_zworld_p0_warmup(
        agent, seed, zworld_p0_episodes, steps_per_episode, dry_run=dry_run,
    )
    print(
        f"  encoder_trained={readiness_report.get('zworld_encoder_trained')} "
        f"max_abs_delta={readiness_report.get('world_encoder_max_abs_delta'):.6f}",
        flush=True,
    )

    print(f"[EXQ-1006] Phase 0b: training E1/E2 ({n_train_episodes} eps)...", flush=True)
    train_stats = _train_agent(agent, env, seed, n_train_episodes, steps_per_episode, e1_call_counter, arm)

    print("[EXQ-1006] Phase 1: goal template...", flush=True)
    z_goal_tensor, goal_template_source = _collect_goal_template(agent, env, seed, goal_max_steps)
    goal_config = GoalConfig(goal_dim=world_dim, z_goal_enabled=True, goal_weight=1.0)
    goal_state = GoalState(goal_config, agent.device)
    goal_state._z_goal = z_goal_tensor.to(agent.device)
    print(f"  z_goal_norm={goal_state.goal_norm():.4f} source={goal_template_source}", flush=True)

    print("[EXQ-1006] Phase 2: warmup state + branch snapshot...", flush=True)
    z_self_0, z_world_0, warmup_actions, env_snapshot, latent_snapshot = _get_warmup_state_with_snapshot(
        agent, env, seed, n_warmup_steps,
    )
    base_prox = float(goal_state.goal_proximity(z_world_0).item())
    print(f"  base_prox={base_prox:.4f} z_world_0_norm={float(z_world_0.norm().item()):.4f}", flush=True)

    print(f"[EXQ-1006] Phase 3: generating {n_sequences} candidate sequences...", flush=True)
    seqs = _generate_candidate_sequences(n_sequences, rollout_horizon, env.action_dim, seed)

    print(f"[EXQ-1006] Phase 4: scoring sequences (hybrid + e1-alone) at horizons {checkpoints}...", flush=True)
    scores_by_h: Dict[int, List[float]] = {h: [] for h in checkpoints}
    endpoints_by_h: Dict[int, List[torch.Tensor]] = {h: [] for h in checkpoints}
    endpoints_by_h_seq: Dict[int, Dict[int, torch.Tensor]] = {h: {} for h in checkpoints}
    scores_e1alone_by_h: Dict[int, List[float]] = {h: [] for h in checkpoints}
    endpoints_e1alone_by_h: Dict[int, List[torch.Tensor]] = {h: [] for h in checkpoints}
    endpoints_e1alone_by_h_seq: Dict[int, Dict[int, torch.Tensor]] = {h: {} for h in checkpoints}

    for i, seq in enumerate(seqs):
        per_h = _score_sequence_hybrid_multi_horizon(
            agent, z_self_0, z_world_0, seq, goal_state, self_dim, checkpoints, e1_call_counter,
        )
        for h, (score, endpoint) in per_h.items():
            scores_by_h[h].append(score)
            endpoints_by_h[h].append(endpoint)
            endpoints_by_h_seq[h][i] = endpoint
        per_h_e1alone = _score_sequence_e1_alone_multi_horizon(
            agent, z_self_0, z_world_0, seq, goal_state, self_dim, checkpoints, e1_call_counter,
        )
        for h, (score, endpoint) in per_h_e1alone.items():
            scores_e1alone_by_h[h].append(score)
            endpoints_e1alone_by_h[h].append(endpoint)
            endpoints_e1alone_by_h_seq[h][i] = endpoint
        if (i + 1) % 10 == 0:
            print(f"  scored {i+1}/{n_sequences}", flush=True)

    print(f"[EXQ-1006] Phase 4b: sampling {n_real_samples} real trajectories from reset at horizons {checkpoints}...", flush=True)
    real_samples_by_h = _collect_real_zworld_sample_multi_horizon(
        agent, env, seed, n_real_samples, checkpoints,
    )

    print(f"[EXQ-1006] Phase 4d: branching the {n_sequences} candidate sequences from the warmup snapshot (real same-start endpoints)...", flush=True)
    branched_by_h, n_branch_terminated = _collect_branched_real_endpoints(
        agent, env_snapshot, latent_snapshot, seqs, checkpoints,
    )
    print(f"  branched: n(h=1)={len(branched_by_h.get(1, {}))} terminated_early={n_branch_terminated}", flush=True)

    per_h: Dict[int, Dict[str, Any]] = {}
    for h in checkpoints:
        scores_t = torch.tensor(scores_by_h[h])
        scores_e1_t = torch.tensor(scores_e1alone_by_h[h])
        roll_cr = _contrast_ratio(endpoints_by_h[h])
        roll_e1_cr = _contrast_ratio(endpoints_e1alone_by_h[h])
        real_cr = _contrast_ratio(real_samples_by_h[h])
        branched_list = [branched_by_h[h][k] for k in sorted(branched_by_h[h].keys())]
        branched_cr = _contrast_ratio(branched_list)
        real_goal = _goal_stats(real_samples_by_h[h], goal_state)
        branched_goal = _goal_stats(branched_list, goal_state)
        roll_goal = _goal_stats(endpoints_by_h[h], goal_state)
        roll_e1_goal = _goal_stats(endpoints_e1alone_by_h[h], goal_state)
        real_axis = _goal_axis_fraction(real_samples_by_h[h], goal_state._z_goal)
        branched_axis = _goal_axis_fraction(branched_list, goal_state._z_goal)
        roll_axis = _goal_axis_fraction(endpoints_by_h[h], goal_state._z_goal)
        roll_e1_axis = _goal_axis_fraction(endpoints_e1alone_by_h[h], goal_state._z_goal)
        fid = _fidelity_mse(endpoints_by_h_seq[h], branched_by_h[h])
        fid_e1 = _fidelity_mse(endpoints_e1alone_by_h_seq[h], branched_by_h[h])

        def _ratio(a: float, b: float) -> float:
            return (a / b) if (b == b and b > 1e-12 and a == a) else float("nan")

        cr_real = real_cr["contrast_ratio"]
        cr_ratio = _ratio(roll_cr["contrast_ratio"], cr_real)
        cr_ratio_e1 = _ratio(roll_e1_cr["contrast_ratio"], cr_real)
        var_h = float(scores_t.var().item()) if len(scores_by_h[h]) > 1 else 0.0
        var_e1_h = float(scores_e1_t.var().item()) if len(scores_e1alone_by_h[h]) > 1 else 0.0
        per_h[h] = {
            "e1coe_score_var": var_h,
            "e1coe_score_var_e1alone": var_e1_h,
            "e1coe_score_mean": float(scores_t.mean().item()) if len(scores_by_h[h]) else float("nan"),
            "cr_rollout": roll_cr,
            "cr_rollout_e1alone": roll_e1_cr,
            "cr_real": real_cr,
            "cr_branched_same_start": branched_cr,
            "cr_ratio": cr_ratio,
            "cr_ratio_e1alone": cr_ratio_e1,
            "cr_ratio_vs_branched": _ratio(roll_cr["contrast_ratio"], branched_cr["contrast_ratio"]),
            # centroid-norm band (vs real 4b; vs same-start branched)
            "centroid_norm_ratio_vs_real": _ratio(roll_cr["centroid_norm"], real_cr["centroid_norm"]),
            "centroid_norm_ratio_vs_branched": _ratio(roll_cr["centroid_norm"], branched_cr["centroid_norm"]),
            "centroid_norm_corrected_cr_ratio": _ratio(_ratio(roll_cr["spread"], real_cr["centroid_norm"]), cr_real),
            # spread ratios
            "spread_ratio_vs_real": _ratio(roll_cr["spread"], real_cr["spread"]),
            "spread_ratio_vs_branched": _ratio(roll_cr["spread"], branched_cr["spread"]),
            # real-endpoint goal statistics (THE RECORDING GAP)
            "real_goal_prox_var": real_goal["prox_var"],
            "real_goal_dist_var": real_goal["dist_var"],
            "real_goal_prox_mean": real_goal["prox_mean"],
            "real_goal_dist_mean": real_goal["dist_mean"],
            "real_goal_prox_values": real_goal["prox_values"],
            "real_goal_dist_values": real_goal["dist_values"],
            "branched_goal_prox_var": branched_goal["prox_var"],
            "branched_goal_dist_var": branched_goal["dist_var"],
            "branched_goal_prox_mean": branched_goal["prox_mean"],
            "branched_goal_dist_mean": branched_goal["dist_mean"],
            "branched_goal_prox_values": branched_goal["prox_values"],
            "rollout_goal_dist_var": roll_goal["dist_var"],
            "rollout_goal_dist_mean": roll_goal["dist_mean"],
            "rollout_goal_prox_values": roll_goal["prox_values"],
            "rollout_e1alone_goal_dist_var": roll_e1_goal["dist_var"],
            "var_ratio_vs_real_endpoint": _ratio(var_h, real_goal["prox_var"]),
            "var_ratio_vs_branched_endpoint": _ratio(var_h, branched_goal["prox_var"]),
            # goal-axis projection
            "goal_axis_fraction_real": real_axis["fraction"],
            "goal_axis_fraction_branched": branched_axis["fraction"],
            "goal_axis_fraction_rollout": roll_axis["fraction"],
            "goal_axis_fraction_rollout_e1alone": roll_e1_axis["fraction"],
            "goal_axis_detail": {"real": real_axis, "branched": branched_axis,
                                 "rollout": roll_axis, "rollout_e1alone": roll_e1_axis},
            "goal_axis_fraction_ratio_rollout_vs_real": _ratio(roll_axis["fraction"], real_axis["fraction"]),
            # endpoint fidelity (same start, same sequence)
            "fidelity_mse": fid["mse_mean"],
            "fidelity_n": fid["n"],
            "fidelity_mse_values": fid["mse_values"],
            "fidelity_mse_e1alone": fid_e1["mse_mean"],
            # both bars, per horizon
            "cr_bar_met": bool(cr_ratio == cr_ratio and cr_ratio >= CR_ROLLOUT_COLLAPSE_RATIO),
            "var_bar_met": bool(var_h >= C3_VAR_THRESHOLD),
            "both_bars_met": bool(cr_ratio == cr_ratio and cr_ratio >= CR_ROLLOUT_COLLAPSE_RATIO and var_h >= C3_VAR_THRESHOLD),
            "n_branched": int(branched_cr["n"]),
        }
        print(
            f"  h={h:>2d}: var={var_h:.3e} cr_ratio={cr_ratio:.3e} "
            f"cent_ratio(real)={per_h[h]['centroid_norm_ratio_vs_real']:.3f} "
            f"spread_ratio(real)={per_h[h]['spread_ratio_vs_real']:.3f} "
            f"real_prox_var={real_goal['prox_var']:.3e} branched_prox_var={branched_goal['prox_var']:.3e} "
            f"axis_frac roll/real={roll_axis['fraction']:.3e}/{real_axis['fraction']:.3e} "
            f"fid_mse={fid['mse_mean']:.3e} (n={fid['n']}) "
            f"bars cr={per_h[h]['cr_bar_met']} var={per_h[h]['var_bar_met']}",
            flush=True,
        )

    print("[EXQ-1006] Phase 4c: direct-channel one-step per-action divergence probe...", flush=True)
    action_probe = _one_step_action_divergence(agent, z_self_0, z_world_0, self_dim, e1_call_counter)
    cr_real_h1 = per_h[1]["cr_real"]["contrast_ratio"]
    action_cr = action_probe["contrast_ratio"]["contrast_ratio"]
    ratio_action_vs_real_h1 = (
        (action_cr / cr_real_h1) if (cr_real_h1 == cr_real_h1 and cr_real_h1 > 0) else float("nan")
    )

    missing_action_calls = float(getattr(agent.e1, "_action_cond_missing_calls", 0))
    buffer_stats = agent.e1_action_buffer_stats()
    direct_supply_fraction = (
        (e1_call_counter["n_e1_calls_nonzero_action"] / e1_call_counter["n_e1_calls"])
        if e1_call_counter["n_e1_calls"] else 0.0
    )
    print(
        f"  [vacuity] missing_action_calls={missing_action_calls:.0f} "
        f"direct_action_supply_fraction={direct_supply_fraction:.4f} "
        f"(internal_buffer_nonzero_fraction={buffer_stats.get('nonzero_fraction', 0.0):.4f})",
        flush=True,
    )

    verdict = "PASS" if readiness_report.get("zworld_encoder_trained") else "FAIL"
    print(f"verdict: {verdict}", flush=True)

    row = {
        "seed": seed,
        "arm": arm,
        "readiness": readiness_report,
        "goal_template_source": goal_template_source,
        "z_goal_norm": goal_state.goal_norm(),
        "base_prox": base_prox,
        "z_world_0_norm": float(z_world_0.norm().item()),
        "checkpoints": checkpoints,
        "per_h": {str(h): per_h[h] for h in checkpoints},
        "n_branch_terminated_early": int(n_branch_terminated),
        "action_probe": action_probe,
        "ratio_action_vs_real_h1": ratio_action_vs_real_h1,
        "action_cr": action_cr,
        "cr_real_h1": cr_real_h1,
        "missing_action_calls": missing_action_calls,
        "e1_action_buffer_stats": buffer_stats,
        "direct_action_supply_fraction": direct_supply_fraction,
        "n_e1_calls_total": e1_call_counter["n_e1_calls"],
        "train_stats": train_stats,
        "n_e1_grad_steps": int(train_stats["n_e1_grad_steps"]),
        "rsd_engaged_frac": float(train_stats["rsd_engaged_frac"]),
        "rsd_action_sensitivity_ratio": float(train_stats["rsd_action_sensitivity_ratio"]),
        "rsd_n_sensitivity_checks": int(train_stats["rsd_n_sensitivity_checks"]),
        "anchor_term_mean": float(train_stats["anchor_term_mean"]),
        "rsd_term_mean": float(train_stats["rsd_term_mean"]),
        "anchor_w_mean": float(train_stats["anchor_w_mean"]),
        "anchor_raw_grad_ratio_mean": float(train_stats["anchor_raw_grad_ratio_mean"]),
        "anchor_authority_mean": float(train_stats["anchor_authority_mean"]),
        "anchor_n_clamped": int(train_stats["anchor_n_clamped"]),
        "n_anchor_ticks": int(train_stats["n_anchor_ticks"]),
        "n_e1_grad_steps_real": int(train_stats["n_e1_grad_steps_real"]),
    }
    # flat lineage-comparable aliases at h=1
    row["e1coe_score_var_h1"] = per_h[1]["e1coe_score_var"]
    row["cr_ratio_h1"] = per_h[1]["cr_ratio"]
    row["centroid_norm_ratio_vs_real_h1"] = per_h[1]["centroid_norm_ratio_vs_real"]
    row["real_goal_prox_var_h1"] = per_h[1]["real_goal_prox_var"]
    row["goal_axis_fraction_rollout_h1"] = per_h[1]["goal_axis_fraction_rollout"]
    row["goal_axis_fraction_real_h1"] = per_h[1]["goal_axis_fraction_real"]
    return row, agent


# ---------------------------------------------------------------------------
# Per-arm precondition specs (regime-conditioned)
# ---------------------------------------------------------------------------

def _build_precondition_specs() -> List[PreconditionSpec]:
    is_rsd = lambda ctx: bool(ctx.get("rsd"))
    is_rsd_incumbent = lambda ctx: ctx.get("arm") == RSD_ARM
    is_anchor = lambda ctx: ctx.get("anchor") is not None
    return [
        PreconditionSpec(
            name="encoder_trained", kind="readiness",
            description="At least one split_encoder.world_encoder tensor moved during the Phase 0a SD-070 warmup, every seed (min over seeds of max_abs_delta).",
            control="SD-070 P0 warmup on a random policy", threshold=0.0, direction="lower",
        ),
        PreconditionSpec(
            name="real_zworld_nondegenerate_h1", kind="readiness",
            description=f"Phase 4b CR_real(h=1) finite and > CR_REAL_FLOOR with >= {MIN_REAL_SAMPLES_PER_HORIZON} surviving real samples, every seed (min n over seeds; met also requires the CR floor).",
            control="40 real from-reset trajectories per horizon", threshold=float(MIN_REAL_SAMPLES_PER_HORIZON) - 1e-9, direction="lower",
        ),
        PreconditionSpec(
            name="branched_same_start_nondegenerate_h1", kind="readiness",
            description=f">= {MIN_REAL_SAMPLES_PER_HORIZON} candidate sequences survived the Phase 4d branch to h=1 with a finite same-start CR, every seed (min over seeds).",
            control="env deepcopy + latent-state restore per candidate sequence", threshold=float(MIN_REAL_SAMPLES_PER_HORIZON) - 1e-9, direction="lower",
        ),
        PreconditionSpec(
            name="no_missing_action_calls", kind="readiness",
            description="E1DeepPredictor._action_cond_missing_calls is 0 on every cell (max over seeds).",
            control="every actions= call supplied a real one-hot", threshold=0.5, direction="upper",
        ),
        PreconditionSpec(
            name="direct_action_supply_fraction", kind="readiness",
            description="Fraction of E1 calls that received a genuine one-hot actions= argument, min over seeds (>= 0.999).",
            control="one-hot supply audited at every call site", threshold=0.999 - 1e-9, direction="lower",
        ),
        PreconditionSpec(
            name="cr_ratio_h1_finite", kind="readiness",
            description="cr_ratio(h=1) finite and positive on every seed of this arm (min over seeds).",
            control="the exact statistic bar 1 reads", threshold=0.0, direction="lower",
        ),
        PreconditionSpec(
            name="e1_grad_steps_matched", kind="readiness",
            description="This arm took the same number of E1 optimiser steps as every other arm at each seed (max over seeds of the cross-arm gap). TRUE BY CONSTRUCTION under the uniform trailing-window trigger; measured, not assumed.",
            control="identical random policy, env, seeds and trailing-window schedule in every arm", threshold=0.5, direction="upper",
        ),
        PreconditionSpec(
            name="rsd_objective_engaged", kind="readiness",
            description=f"rollout_sequence_divergence_loss training calls were non-degenerate (n_distinct_full_sequences >= {RSD_MIN_BATCH_CLASSES}) on a nonzero fraction of ticks, every seed (min over seeds).",
            control="K sampled real windows per tick, distinct-full-sequence floor replicated pre-call",
            threshold=0.0, direction="lower",
            applies_to=is_rsd, applies_note="ARM_OFF trains no InfoNCE term; the check is not meaningful there.",
        ),
        PreconditionSpec(
            name="rsd_action_sensitive", kind="readiness",
            description=f"ARM_RSD's InfoNCE term actually DEPENDS on the sampled action sequences: mean loss under a row-permuted action assignment / mean loss under the real assignment >= {RSD_ACTION_SENSITIVITY_RATIO_FLOOR}, every seed (min over seeds). V3-EXQ-1000 red-team #1/#2 identity-shortcut control. GATES ARM_RSD ONLY: on an anchor arm a lost action-sensitivity is a RESULT of the anchor (divergence re-crushed; read via cr_bar_retained), not a readiness failure, and gating it there would let the manipulation move its own gate (red-team #4, 2026-09-06). Anchor arms' ratios are recorded in interpretation.recorded_preconditions.",
            control="row-permuted action-sequence assignment, isolated RNG stream, no_grad, every RSD tick",
            threshold=RSD_ACTION_SENSITIVITY_RATIO_FLOOR - 1e-9, direction="lower",
            applies_to=is_rsd_incumbent, applies_note="gates the incumbent ARM_RSD only; ARM_OFF trains no InfoNCE term, and on the anchor arms the ratio is a recorded result, not a gate.",
        ),
        PreconditionSpec(
            name="anchor_engaged", kind="readiness",
            description=f"The anchor term's realised gradient authority (w_t * ||grad anchor|| / ||grad InfoNCE||, mean over the arm's anchored training ticks) >= {ANCHOR_AUTHORITY_FLOOR} on every seed (min over seeds), with at least one anchored tick. By construction ~ANCHOR_GRAD_AUTHORITY unless the clamp bound; measured, not assumed (red-team #1, 2026-09-06: at a fixed unit weight the anchor was a few percent of the update).",
            control="per-tick autograd.grad norms of both terms on E1's parameters; gradient-matched dose",
            threshold=ANCHOR_AUTHORITY_FLOOR - 1e-9, direction="lower",
            applies_to=is_anchor, applies_note="only the two anchor arms train an anchor term.",
        ),
    ]


def _arm_contexts() -> Dict[str, Dict[str, Any]]:
    return {arm: {"arm": arm, "rsd": arm in RSD_BEARING_ARMS, "anchor": ARM_CONFIGS[arm]["anchor"]}
            for arm in ARM_ORDER}


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def _count(bools: List[bool]) -> int:
    return int(sum(1 for b in bools if b))


def _finite(x: float) -> bool:
    return x == x and x not in (float("inf"), float("-inf"))


def run(
    seeds: List[int],
    world_dim: int,
    self_dim: int,
    n_train_episodes: int,
    steps_per_episode: int,
    n_sequences: int,
    rollout_horizon: int,
    n_warmup_steps: int,
    goal_max_steps: int,
    zworld_p0_episodes: int,
    n_real_samples: int,
    dry_run: bool = False,
) -> Dict[str, Any]:
    checkpoints = sorted(set(
        [h for h in HORIZON_CHECKPOINTS_FULL if h <= rollout_horizon] + [rollout_horizon]
    ))

    specs = _build_precondition_specs()
    arm_ctx = _arm_contexts()
    # design-time proof that no arm's gate is structurally unsatisfiable
    assert_no_structurally_unsatisfiable_gate(specs, list(arm_ctx.values()), arm_id_key="arm")

    cell_config_slice = {
        "world_dim": world_dim, "self_dim": self_dim,
        "n_train_episodes": n_train_episodes, "steps_per_episode": steps_per_episode,
        "n_sequences": n_sequences, "rollout_horizon": rollout_horizon,
        "n_warmup_steps": n_warmup_steps, "goal_max_steps": goal_max_steps,
        "zworld_p0_episodes": zworld_p0_episodes, "n_real_samples": n_real_samples,
        "env_kwargs": _env_kwargs(),
        "rc_horizon": RC_HORIZON, "rc_decay": RC_DECAY, "anchor_w_rc": ANCHOR_W_RC,
        "anchor_w_ep": ANCHOR_W_EP,
        "rsd_horizon": RSD_HORIZON, "rsd_temperature": RSD_TEMPERATURE,
        "rsd_min_batch_classes": RSD_MIN_BATCH_CLASSES, "rsd_weight": RSD_WEIGHT,
        "rsd_batch_k": RSD_BATCH_K, "rsd_buffer_max": RSD_BUFFER_MAX,
        "train_window_h": TRAIN_WINDOW_H,
        "cr_real_floor": CR_REAL_FLOOR, "cr_rollout_collapse_ratio": CR_ROLLOUT_COLLAPSE_RATIO,
        "c3_var_threshold": C3_VAR_THRESHOLD,
        "min_real_samples_per_horizon": MIN_REAL_SAMPLES_PER_HORIZON,
        "horizon_checkpoints_full": list(HORIZON_CHECKPOINTS_FULL),
        "centroid_band": [CENTROID_BAND_LO, CENTROID_BAND_HI],
        "var_lift_factor": VAR_LIFT_FACTOR, "goal_axis_within_factor": GOAL_AXIS_WITHIN_FACTOR,
        "alpha_world": 0.9, "alpha_self": 0.3, "unified_latent_mode": False,
        "train_lr_e1": 1e-3, "train_lr_e2": 1e-3,
    }

    arm_results: List[Dict[str, Any]] = []
    agents_for_manifest = []
    for arm in ARM_ORDER:
        for seed in seeds:
            with arm_cell(
                seed,
                config_slice={**cell_config_slice, "arm": arm, **ARM_CONFIGS[arm]},
                script_path=Path(__file__),
                config_slice_declared=True,
                # MINT AS YOU GO: ARM_OFF is the lineage's OFF baseline; emit it
                # cross-driver reusable (1000 convention). Note this trips
                # substrate_stable_across_run by construction -- read
                # substrate_stability_detail.process_snapshot_drift (1000 #6).
                include_driver_script_in_hash=(arm != OFF_ARM),
            ) as cell:
                row, agent = run_cell(
                    seed=seed, arm=arm,
                    world_dim=world_dim, self_dim=self_dim,
                    n_train_episodes=n_train_episodes, steps_per_episode=steps_per_episode,
                    n_sequences=n_sequences, rollout_horizon=rollout_horizon,
                    n_warmup_steps=n_warmup_steps, goal_max_steps=goal_max_steps,
                    zworld_p0_episodes=zworld_p0_episodes, n_real_samples=n_real_samples,
                    checkpoints=checkpoints, dry_run=dry_run,
                )
                cell.stamp(row)
            arm_results.append(row)
            agents_for_manifest.append(agent)

    by_arm_seed: Dict[Tuple[str, int], Dict[str, Any]] = {(r["arm"], r["seed"]): r for r in arm_results}
    majority_seeds = len(seeds) // 2 + 1

    def ph(arm: str, seed: int, h: int) -> Dict[str, Any]:
        return by_arm_seed[(arm, seed)]["per_h"][str(h)]

    # ---- per-arm gates ----
    grad_step_gap_per_seed: Dict[str, int] = {}
    for seed in seeds:
        counts = [int(by_arm_seed[(arm, seed)]["n_e1_grad_steps"]) for arm in ARM_ORDER]
        grad_step_gap_per_seed[f"seed{seed}"] = int(max(counts) - min(counts))
    max_grad_step_gap = max(grad_step_gap_per_seed.values()) if grad_step_gap_per_seed else 0

    arm_gates = []
    arm_measured_detail: Dict[str, Dict[str, Any]] = {}
    recorded_preconditions: List[Dict[str, Any]] = []
    for arm in ARM_ORDER:
        rows = [by_arm_seed[(arm, s)] for s in seeds]
        enc = [float(r["readiness"].get("world_encoder_max_abs_delta", 0.0)) for r in rows]
        enc_met = all(bool(r["readiness"].get("zworld_encoder_trained")) for r in rows)
        real_n = [int(r["per_h"]["1"]["cr_real"]["n"]) for r in rows]
        real_cr_ok = all(
            _finite(r["per_h"]["1"]["cr_real"]["contrast_ratio"]) and r["per_h"]["1"]["cr_real"]["contrast_ratio"] > CR_REAL_FLOOR
            and r["per_h"]["1"]["cr_real"]["n"] >= MIN_REAL_SAMPLES_PER_HORIZON for r in rows
        )
        br_n = [int(r["per_h"]["1"]["n_branched"]) for r in rows]
        br_ok = all(
            r["per_h"]["1"]["n_branched"] >= MIN_REAL_SAMPLES_PER_HORIZON
            and _finite(r["per_h"]["1"]["cr_branched_same_start"]["contrast_ratio"]) for r in rows
        )
        miss = [float(r["missing_action_calls"]) for r in rows]
        supply = [float(r["direct_action_supply_fraction"]) for r in rows]
        crh1 = [float(r["per_h"]["1"]["cr_ratio"]) for r in rows]
        cr_ok = all(_finite(v) and v > 0 for v in crh1)
        measured = {
            "encoder_trained": min(enc),
            "real_zworld_nondegenerate_h1": float(min(real_n)),
            "branched_same_start_nondegenerate_h1": float(min(br_n)),
            "no_missing_action_calls": max(miss),
            "direct_action_supply_fraction": min(supply),
            "cr_ratio_h1_finite": min([v for v in crh1 if _finite(v)], default=float("nan")),
            "e1_grad_steps_matched": float(max_grad_step_gap),
        }
        overrides = {
            "encoder_trained": enc_met,
            "real_zworld_nondegenerate_h1": real_cr_ok,
            "branched_same_start_nondegenerate_h1": br_ok,
            "cr_ratio_h1_finite": cr_ok,
        }
        detail: Dict[str, Any] = {
            "rsd_engaged_frac_per_seed": {}, "rsd_sensitivity_ratio_per_seed": {},
            "grad_step_gap_per_seed": grad_step_gap_per_seed,
        }
        if arm in RSD_BEARING_ARMS:
            eng = {f"seed{r['seed']}": float(r["rsd_engaged_frac"]) for r in rows}
            sens = {f"seed{r['seed']}": float(r["rsd_action_sensitivity_ratio"]) for r in rows}
            fin_sens = [v for v in sens.values() if _finite(v)]
            measured["rsd_objective_engaged"] = min(eng.values())
            detail["rsd_engaged_frac_per_seed"] = eng
            detail["rsd_sensitivity_ratio_per_seed"] = sens
            if arm == RSD_ARM:
                measured["rsd_action_sensitive"] = min(fin_sens) if fin_sens else float("nan")
                overrides["rsd_action_sensitive"] = bool(
                    len(fin_sens) == len(seeds) and min(fin_sens) >= RSD_ACTION_SENSITIVITY_RATIO_FLOOR
                )
            else:
                # recorded, non-gating, on the anchor arms (red-team #4)
                recorded_preconditions.append({
                    "name": f"{arm}::rsd_action_sensitive_recorded", "kind": "readiness", "arm": arm,
                    "description": "action-sensitivity ratio of this ANCHOR arm's InfoNCE term (recorded, not gating: a lost sensitivity is a result of the anchor, read via cr_bar_retained)",
                    "measured": min(fin_sens) if fin_sens else float("nan"),
                    "threshold": RSD_ACTION_SENSITIVITY_RATIO_FLOOR, "direction": "lower", "comparator": ">=",
                    "met": bool(len(fin_sens) == len(seeds) and min(fin_sens) >= RSD_ACTION_SENSITIVITY_RATIO_FLOOR),
                    "gating": False, "per_seed": sens,
                })
        if ARM_CONFIGS[arm]["anchor"] is not None:
            auth = {f"seed{r['seed']}": float(r["anchor_authority_mean"]) for r in rows}
            fin_auth = [v for v in auth.values() if _finite(v)]
            n_ticks = [int(r["n_anchor_ticks"]) for r in rows]
            measured["anchor_engaged"] = min(fin_auth) if fin_auth else float("nan")
            overrides["anchor_engaged"] = bool(
                len(fin_auth) == len(seeds) and min(fin_auth) >= ANCHOR_AUTHORITY_FLOOR and min(n_ticks) > 0
            )
            detail["anchor_authority_per_seed"] = auth
            detail["anchor_w_mean_per_seed"] = {f"seed{r['seed']}": float(r["anchor_w_mean"]) for r in rows}
            detail["anchor_raw_grad_ratio_per_seed"] = {f"seed{r['seed']}": float(r["anchor_raw_grad_ratio_mean"]) for r in rows}
        gate = evaluate_arm_gate(arm, arm_ctx[arm], specs, measured, met_overrides=overrides)
        # comparator annotations so the indexer's recompute matches the science
        for p in gate["preconditions"]:
            if p["precondition"] in ("encoder_trained", "cr_ratio_h1_finite", "rsd_objective_engaged"):
                p["comparator"] = ">"
            elif p["direction"] == "upper":
                p["comparator"] = "<="
            else:
                p["comparator"] = ">="
            if p["precondition"] == "rsd_objective_engaged":
                p["per_seed"] = detail["rsd_engaged_frac_per_seed"]
            if p["precondition"] == "rsd_action_sensitive":
                p["per_seed"] = detail["rsd_sensitivity_ratio_per_seed"]
            if p["precondition"] == "anchor_engaged":
                p["per_seed"] = detail.get("anchor_authority_per_seed")
                p["anchor_w_mean_per_seed"] = detail.get("anchor_w_mean_per_seed")
                p["raw_grad_ratio_per_seed"] = detail.get("anchor_raw_grad_ratio_per_seed")
            if p["precondition"] == "e1_grad_steps_matched":
                p["per_seed"] = grad_step_gap_per_seed
        arm_gates.append(gate)
        arm_measured_detail[arm] = detail

    aggregate = aggregate_arm_gates(arm_gates)
    green = set(aggregate["green_arms"])
    non_degenerate = bool(aggregate["non_degenerate"])
    degeneracy_reason = aggregate["degeneracy_reason"] or None

    # ---- dv_headroom (recorded, NOT adjudicating): same-start real-endpoint variance at h=1 ----
    real_var_h1_per_seed = {}
    branched_var_h1_per_seed = {}
    ref_arm = OFF_ARM if OFF_ARM in green else (sorted(green)[0] if green else OFF_ARM)
    for seed in seeds:
        real_var_h1_per_seed[f"seed{seed}"] = float(ph(ref_arm, seed, 1)["real_goal_prox_var"])
        branched_var_h1_per_seed[f"seed{seed}"] = float(ph(ref_arm, seed, 1)["branched_goal_prox_var"])

    def _majority_order(vals: Dict[str, float]) -> float:
        v = sorted([x for x in vals.values() if _finite(x)], reverse=True)
        return v[majority_seeds - 1] if len(v) >= majority_seeds else float("nan")

    majority_order_branched_var = _majority_order(branched_var_h1_per_seed)
    majority_order_real_var = _majority_order(real_var_h1_per_seed)
    dv_headroom_entry = dv_headroom_check(
        "dv_headroom_e1coe_score_var_h1",
        dv_name="e1coe_score_var_h1",
        criterion_threshold=C3_VAR_THRESHOLD,
        achievable=majority_order_branched_var,
        margin=DV_HEADROOM_MARGIN,
        control=(
            "SAME-START real endpoint set (Phase 4d: the 40 candidate sequences executed for real from the "
            "warmup snapshot) goal_proximity variance at h=1, per seed, read off arm "
            f"{ref_arm} (arm-invariant: identical encoder per seed); achievable = the majority-order "
            f"statistic (value reached by >= {majority_seeds}/{len(seeds)} seeds). The from-reset Phase 4b "
            "variance is recorded beside it (an upper bound: it adds start dispersion)."
        ),
        gating=False,
        recorded_only=True,
        why_not_adjudicating=(
            "an UNMET headroom here IS the registered H-readout-saturation finding "
            "(the 0.002 bar exceeds what real endpoints produce at real spread) -- "
            "routed by the confirmed V3-EXQ-1000 autopsy to 're-register the bar', "
            "not to substrate_not_ready_requeue; the run still resolves every "
            "relative reading. Lives in interpretation.recorded_preconditions so the "
            "indexer surfaces it without vacating the run (red-team #2, 2026-09-06)."
        ),
        per_seed=branched_var_h1_per_seed,
        per_seed_from_reset=real_var_h1_per_seed,
        majority_order_from_reset=majority_order_real_var,
        reference_arm=ref_arm,
    )
    dv_headroom_entry["met"] = bool(_finite(majority_order_branched_var) and majority_order_branched_var >= dv_headroom_entry["threshold"])
    dv_headroom_entry["comparator"] = ">="
    recorded_preconditions.append(dv_headroom_entry)

    # ---- Legs ----
    NULL_READINGS = ("centroid_not_restored", "var_lifted_centroid_not_restored", "centroid_restored_var_flat")

    def leg_a_arm(arm_x: str, h: int) -> Dict[str, Any]:
        rows = {}
        n_cent = n_var = n_bars = n_cr = n_real_var = n_br_var = 0
        for seed in seeds:
            x = ph(arm_x, seed, h)
            rsd = ph(RSD_ARM, seed, h)
            cent_ratio = x["centroid_norm_ratio_vs_real"]
            cent_ok = _finite(cent_ratio) and CENTROID_BAND_LO <= cent_ratio <= CENTROID_BAND_HI
            var_x = x["e1coe_score_var"]; var_rsd = rsd["e1coe_score_var"]
            var_ok = var_x >= VAR_LIFT_FACTOR * var_rsd and var_x > 0
            cr_ok = _finite(x["cr_ratio"]) and x["cr_ratio"] >= CR_ROLLOUT_COLLAPSE_RATIO
            bars_ok = cr_ok and var_x >= C3_VAR_THRESHOLD
            rv = x["real_goal_prox_var"]; bv = x["branched_goal_prox_var"]
            real_var_ok = _finite(rv) and var_x >= rv
            br_var_ok = _finite(bv) and var_x >= bv
            n_cent += cent_ok; n_var += var_ok; n_bars += bars_ok; n_cr += cr_ok
            n_real_var += real_var_ok; n_br_var += br_var_ok
            rows[f"seed{seed}"] = {
                "centroid_norm_ratio_vs_real": cent_ratio, "centroid_in_band": bool(cent_ok),
                "centroid_norm_ratio_vs_branched": x["centroid_norm_ratio_vs_branched"],
                "centroid_norm_ratio_vs_off": (
                    x["cr_rollout"]["centroid_norm"] / ph(OFF_ARM, seed, h)["cr_rollout"]["centroid_norm"]
                    if ph(OFF_ARM, seed, h)["cr_rollout"]["centroid_norm"] > 1e-12 else float("nan")),
                "spread_ratio_vs_real": x["spread_ratio_vs_real"],
                "spread_ratio_vs_branched": x["spread_ratio_vs_branched"],
                "e1coe_score_var": var_x, "e1coe_score_var_rsd": var_rsd,
                "var_lift_over_rsd": (var_x / var_rsd) if var_rsd > 0 else float("nan"),
                "var_lifted": bool(var_ok),
                "cr_ratio": x["cr_ratio"], "cr_bar_retained": bool(cr_ok), "both_bars": bool(bars_ok),
                "real_goal_prox_var": rv, "branched_goal_prox_var": bv,
                "var_reaches_real_endpoint_var": bool(real_var_ok),
                "var_reaches_branched_endpoint_var": bool(br_var_ok),
                "fidelity_mse": x["fidelity_mse"], "fidelity_mse_rsd": rsd["fidelity_mse"],
                "action_sensitivity_ratio": by_arm_seed[(arm_x, seed)]["rsd_action_sensitivity_ratio"],
                "anchor_w_mean": by_arm_seed[(arm_x, seed)]["anchor_w_mean"],
                "anchor_authority_mean": by_arm_seed[(arm_x, seed)]["anchor_authority_mean"],
            }
        centroid_restored = n_cent >= majority_seeds
        var_lifted = n_var >= majority_seeds
        cr_bar_retained = n_cr >= majority_seeds
        if centroid_restored and var_lifted and cr_bar_retained:
            reading = "supported"
        elif centroid_restored and var_lifted:
            reading = "centroid_restored_var_lifted_cr_bar_lost"
        elif centroid_restored:
            reading = "centroid_restored_var_flat"
        elif var_lifted:
            reading = "var_lifted_centroid_not_restored"
        else:
            reading = "centroid_not_restored"
        return {
            "arm": arm_x, "h": h, "reading": reading,
            "centroid_restored": bool(centroid_restored), "n_centroid_in_band": n_cent,
            "var_lifted": bool(var_lifted), "n_var_lifted": n_var,
            "bars_cleared": bool(n_bars >= majority_seeds), "n_both_bars": n_bars,
            "cr_bar_retained": bool(cr_bar_retained), "n_cr_bar": n_cr,
            "var_reaches_real_endpoint_var": bool(n_real_var >= majority_seeds), "n_var_ge_real": n_real_var,
            "var_reaches_branched_endpoint_var": bool(n_br_var >= majority_seeds), "n_var_ge_branched": n_br_var,
            "per_seed": rows,
        }

    def leg_a(h: int) -> Dict[str, Any]:
        if RSD_ARM not in green:
            return {"tag": "anchor_undetermined_arm_red", "arms": {}, "hypothesis": "H-fidelity-anchor", "state": "undetermined"}
        arms = {x: leg_a_arm(x, h) for x in ANCHOR_ARMS if x in green}
        if not arms:
            return {"tag": "anchor_arms_vacuous", "arms": {}, "hypothesis": "H-fidelity-anchor", "state": "undetermined"}
        readings = {x: a["reading"] for x, a in arms.items()}
        both_green = all(x in green for x in ANCHOR_ARMS)
        hsuf = "h1" if h == 1 else f"h{h}"
        if any(a["bars_cleared"] for a in arms.values()):
            tag, state = f"anchor_clears_both_bars_{hsuf}", "supported"
        elif any(r == "supported" for r in readings.values()):
            tag, state = "anchor_restores_centroid_lifts_var", "supported"
        elif any(r == "centroid_restored_var_lifted_cr_bar_lost" for r in readings.values()):
            tag, state = "anchor_restores_centroid_lifts_var_cr_bar_lost", "mixed"
        elif all(r in NULL_READINGS for r in readings.values()):
            # every green anchor arm landed on a declared-null disjunct
            base = ("anchor_restores_centroid_var_flat"
                    if any(r == "centroid_restored_var_flat" for r in readings.values())
                    else "anchor_fails_centroid")
            if both_green:
                tag, state = base, "eliminated"
            else:
                # one anchor form unscored: the hypothesis is NOT eliminated on
                # the other form alone (red-team #4, 2026-09-06)
                tag, state = base + "_partial", "mixed"
        else:
            tag, state = "anchor_mixed", "mixed"
        supporting = [x for x, r in readings.items() if r == "supported" or arms[x]["bars_cleared"]]
        return {"tag": tag, "state": state, "hypothesis": "H-fidelity-anchor", "arms": arms,
                "readings": readings, "supporting_arms": supporting, "lead_arm": LEAD_ANCHOR_ARM,
                "both_anchor_arms_green": bool(both_green)}

    def leg_b(h: int) -> Dict[str, Any]:
        if not green:
            return {"tag": "realvar_undetermined_arm_red", "hypothesis": "H-readout-saturation", "state": "undetermined"}
        per = {}
        n_ge = n_lt = n_ge_reset = n_lt_reset = 0
        for seed in seeds:
            x = ph(ref_arm, seed, h)
            rv = x["real_goal_prox_var"]; bv = x["branched_goal_prox_var"]
            ge = _finite(bv) and bv >= C3_VAR_THRESHOLD
            lt = _finite(bv) and bv < C3_VAR_THRESHOLD
            ge_r = _finite(rv) and rv >= C3_VAR_THRESHOLD
            lt_r = _finite(rv) and rv < C3_VAR_THRESHOLD
            n_ge += ge; n_lt += lt; n_ge_reset += ge_r; n_lt_reset += lt_r
            per[f"seed{seed}"] = {"branched_goal_prox_var": bv, "branched_goal_dist_var": x["branched_goal_dist_var"],
                                  "branched_goal_prox_mean": x["branched_goal_prox_mean"],
                                  "branched_reaches_bar": bool(ge),
                                  "real_goal_prox_var": rv, "real_goal_dist_var": x["real_goal_dist_var"],
                                  "real_goal_prox_mean": x["real_goal_prox_mean"], "from_reset_reaches_bar": bool(ge_r),
                                  "shortfall_x_branched": (C3_VAR_THRESHOLD / bv) if (_finite(bv) and bv > 0) else float("nan"),
                                  "shortfall_x_from_reset": (C3_VAR_THRESHOLD / rv) if (_finite(rv) and rv > 0) else float("nan"),
                                  "n_branched": x["n_branched"]}
        if n_ge >= majority_seeds:
            tag, state = "realvar_reaches_bar", "eliminated"
        elif n_lt >= majority_seeds:
            tag, state = "realvar_below_bar", "supported"
        else:
            tag, state = "realvar_mixed", "mixed"
        return {"tag": tag, "state": state, "hypothesis": "H-readout-saturation",
                "source": "same-start branched real endpoints (Phase 4d); from-reset Phase 4b recorded beside",
                "n_reaches_bar": n_ge, "n_below_bar": n_lt,
                "n_reaches_bar_from_reset": n_ge_reset, "n_below_bar_from_reset": n_lt_reset,
                "reference_arm": ref_arm, "per_seed": per}

    def leg_c(h: int) -> Dict[str, Any]:
        if RSD_ARM not in green:
            return {"tag": "rsd_goal_axis_undetermined_arm_red", "hypothesis": "H-goal-orthogonal-dispersion", "state": "undetermined"}
        per = {}
        n_within = n_below = n_above = n_valid = 0
        for seed in seeds:
            x = ph(RSD_ARM, seed, h)
            r = x["goal_axis_fraction_ratio_rollout_vs_real"]
            valid = _finite(r)
            within = valid and (1.0 / GOAL_AXIS_WITHIN_FACTOR) <= r <= GOAL_AXIS_WITHIN_FACTOR
            below = valid and r < 1.0 / GOAL_AXIS_WITHIN_FACTOR
            above = valid and r > GOAL_AXIS_WITHIN_FACTOR
            n_valid += valid; n_within += within; n_below += below; n_above += above
            per[f"seed{seed}"] = {"goal_axis_fraction_rollout": x["goal_axis_fraction_rollout"],
                                  "goal_axis_fraction_real": x["goal_axis_fraction_real"],
                                  "goal_axis_fraction_branched": x["goal_axis_fraction_branched"],
                                  "ratio_rollout_vs_real": r, "within_2x": bool(within)}
        if n_valid < majority_seeds:
            tag, state = "rsd_goal_axis_undetermined", "undetermined"
        elif n_within >= majority_seeds:
            tag, state = "rsd_goal_axis_matches_real", "eliminated"
        elif n_below >= majority_seeds:
            tag, state = "rsd_goal_orthogonal", "supported"
        elif n_above >= majority_seeds:
            tag, state = "rsd_goal_aligned_excess", "eliminated"
        else:
            tag, state = "rsd_goal_axis_mixed", "mixed"
        return {"tag": tag, "state": state, "hypothesis": "H-goal-orthogonal-dispersion",
                "n_valid": n_valid, "n_within": n_within, "n_below": n_below, "n_above": n_above, "per_seed": per}

    legs_by_h = {str(h): {"A": leg_a(h), "B": leg_b(h), "C": leg_c(h)} for h in checkpoints}
    legs_h1 = legs_by_h["1"]

    # both-bars cells at any horizon (1000 learning #1)
    evaluator_bar_reached_cells = []
    for arm in ARM_ORDER:
        for seed in seeds:
            for h in checkpoints:
                if ph(arm, seed, h)["both_bars_met"]:
                    evaluator_bar_reached_cells.append(f"{arm}_seed{seed}_h{h}")

    if not non_degenerate:
        label = "substrate_not_ready_requeue"
        status = "FAIL"
    else:
        label = f"{legs_h1['A']['tag']}__{legs_h1['B']['tag']}__{legs_h1['C']['tag']}"
        status = "PASS"
    evidence_direction = "non_contributory"

    # hypothesis-space ledger proposal (drafted, governance applies)
    ledger_proposal = {
        "qid": HYPOTHESIS_SPACE_QID,
        "H-fidelity-anchor": legs_h1["A"].get("state"),
        "H-readout-saturation": legs_h1["B"].get("state"),
        "H-goal-orthogonal-dispersion": legs_h1["C"].get("state"),
        "note": "drafted by the driver from the pre-registered rule at h=1; governance/autopsy adjudicates.",
    }

    criteria = [
        {
            "name": "C0_portfolio_non_degenerate",
            "load_bearing": True,
            "passed": bool(non_degenerate),
            "measured": len(aggregate["green_arms"]),
            "threshold": 1,
            "statement": "At least one arm passed its own regime-conditioned gate (any-arm-green, precondition_gate semantics). This is the ONLY PASS/FAIL criterion; the rest classify.",
        },
        {
            "name": "C1_fidelity_anchor_restores_centroid_and_lifts_var",
            "load_bearing": False,
            "passed": bool(legs_h1["A"].get("state") == "supported"),
            "measured": max([a["n_centroid_in_band"] for a in legs_h1["A"].get("arms", {}).values()], default=0),
            "threshold": majority_seeds,
            "statement": f"Some green anchor arm has centroid_norm/real in [{CENTROID_BAND_LO:.3f},{CENTROID_BAND_HI:.3f}] AND e1coe_score_var(h=1) >= {VAR_LIFT_FACTOR}x ARM_RSD's, each on >= {majority_seeds}/{len(seeds)} seeds (or clears both bars).",
        },
        {
            "name": "C2_fidelity_anchor_clears_both_bars_h1",
            "load_bearing": False,
            "passed": bool(any(a["bars_cleared"] for a in legs_h1["A"].get("arms", {}).values())),
            "measured": max([a["n_both_bars"] for a in legs_h1["A"].get("arms", {}).values()], default=0),
            "threshold": majority_seeds,
            "statement": f"Some green anchor arm has cr_ratio(h=1) >= {CR_ROLLOUT_COLLAPSE_RATIO} AND e1coe_score_var(h=1) >= {C3_VAR_THRESHOLD} on >= {majority_seeds} seeds.",
        },
        {
            "name": "C3_real_endpoint_var_reaches_bar_h1",
            "load_bearing": False,
            "passed": bool(legs_h1["B"].get("tag") == "realvar_reaches_bar"),
            "measured": legs_h1["B"].get("n_reaches_bar", 0),
            "threshold": majority_seeds,
            "statement": f"SAME-START real-endpoint (Phase 4d branched) goal_proximity variance at h=1 >= {C3_VAR_THRESHOLD} on >= {majority_seeds} seeds (H-readout-saturation's declared null; the registry named the Phase 4b set, which is recorded beside it and is an upper bound).",
        },
        {
            "name": "C4_rsd_goal_axis_fraction_within_2x_real",
            "load_bearing": False,
            "passed": bool(legs_h1["C"].get("tag") == "rsd_goal_axis_matches_real"),
            "measured": legs_h1["C"].get("n_within", 0),
            "threshold": majority_seeds,
            "statement": f"ARM_RSD's goal-axis fraction of spread variance within {GOAL_AXIS_WITHIN_FACTOR}x of the real endpoints' on >= {majority_seeds} seeds (H-goal-orthogonal-dispersion's declared null).",
        },
        {
            "name": "C5_both_bars_any_cell_any_h",
            "load_bearing": False,
            "passed": bool(evaluator_bar_reached_cells),
            "measured": len(evaluator_bar_reached_cells),
            "threshold": 1,
            "statement": "Any (arm, seed, h) cell reaches both bars. RECORDED, not routing.",
        },
    ]
    criteria_by_arm: Dict[str, List[str]] = {}
    criteria_by_arm.setdefault(RSD_ARM, []).extend([
        "C1_fidelity_anchor_restores_centroid_and_lifts_var", "C2_fidelity_anchor_clears_both_bars_h1",
        "C4_rsd_goal_axis_fraction_within_2x_real",
    ])
    criteria_by_arm.setdefault(ref_arm, []).append("C3_real_endpoint_var_reaches_bar_h1")
    criteria_non_degenerate = arm_criteria_non_degenerate(criteria_by_arm, aggregate)
    criteria_non_degenerate["C0_portfolio_non_degenerate"] = True
    criteria_non_degenerate["C5_both_bars_any_cell_any_h"] = bool(non_degenerate)
    # anchor criteria also need a green anchor arm
    any_anchor_green = any(x in green for x in ANCHOR_ARMS)
    for k in ("C1_fidelity_anchor_restores_centroid_and_lifts_var", "C2_fidelity_anchor_clears_both_bars_h1"):
        criteria_non_degenerate[k] = bool(criteria_non_degenerate.get(k, False) and any_anchor_green)

    print(f"\n[EXQ-1006] Label: {label}", flush=True)
    print(f"[EXQ-1006] Status: {status}", flush=True)
    print(f"[EXQ-1006] Legs (h=1): A={legs_h1['A'].get('tag')} B={legs_h1['B'].get('tag')} C={legs_h1['C'].get('tag')}", flush=True)

    preconditions = list(aggregate["adjudication_preconditions"])

    result: Dict[str, Any] = {
        "experiment_type": EXPERIMENT_TYPE,
        "claim_ids": CLAIM_IDS,
        "unblocks_claims": UNBLOCKS_CLAIMS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "supersedes": SUPERSEDES,
        "evidence_class": "diagnostic_portfolio",
        "evidence_direction": evidence_direction,
        "hypothesis_space_qid": HYPOTHESIS_SPACE_QID,
        "hypothesis_space_ledger_proposal": ledger_proposal,
        "seeds": seeds,
        "arms": ARM_ORDER,
        "off_arm": OFF_ARM,
        "rsd_arm": RSD_ARM,
        "anchor_arms": ANCHOR_ARMS,
        "lead_anchor_arm": LEAD_ANCHOR_ARM,
        "arm_configs": ARM_CONFIGS,
        "world_dim": world_dim, "self_dim": self_dim,
        "n_train_episodes": n_train_episodes, "steps_per_episode": steps_per_episode,
        "n_sequences": n_sequences, "rollout_horizon": rollout_horizon,
        "horizon_checkpoints": checkpoints, "registered_horizons": REGISTERED_HORIZONS,
        "n_warmup_steps": n_warmup_steps, "zworld_p0_episodes": zworld_p0_episodes,
        "n_real_samples": n_real_samples,
        "rc_horizon": RC_HORIZON, "rc_decay": RC_DECAY, "anchor_w_rc": ANCHOR_W_RC, "anchor_w_ep": ANCHOR_W_EP,
        "anchor_grad_authority": ANCHOR_GRAD_AUTHORITY, "anchor_w_clamp": [ANCHOR_W_MIN, ANCHOR_W_MAX],
        "anchor_authority_floor": ANCHOR_AUTHORITY_FLOOR,
        "rsd_horizon": RSD_HORIZON, "rsd_temperature": RSD_TEMPERATURE,
        "rsd_min_batch_classes": RSD_MIN_BATCH_CLASSES, "rsd_weight": RSD_WEIGHT, "rsd_batch_k": RSD_BATCH_K,
        "registered_cr_real_floor": CR_REAL_FLOOR,
        "registered_majority_seeds": majority_seeds,
        "registered_cr_rollout_collapse_ratio": CR_ROLLOUT_COLLAPSE_RATIO,
        "registered_c3_var_threshold": C3_VAR_THRESHOLD,
        "registered_centroid_band": [CENTROID_BAND_LO, CENTROID_BAND_HI],
        "registered_var_lift_factor": VAR_LIFT_FACTOR,
        "registered_goal_axis_within_factor": GOAL_AXIS_WITHIN_FACTOR,
        "min_real_samples_per_horizon_floor": MIN_REAL_SAMPLES_PER_HORIZON,
        "non_degenerate": non_degenerate,
        "degeneracy_reason": degeneracy_reason,
        "per_arm_gate": aggregate["per_arm_gate"],
        "green_arms": aggregate["green_arms"],
        "red_arms": aggregate["red_arms"],
        "legs_h1": legs_h1,
        "legs_by_h": legs_by_h,
        "evaluator_bar_reached_cells": evaluator_bar_reached_cells,
        "e1_grad_step_gap_per_seed": grad_step_gap_per_seed,
        "real_endpoint_goal_prox_var_h1_per_seed": real_var_h1_per_seed,
        "branched_endpoint_goal_prox_var_h1_per_seed": branched_var_h1_per_seed,
        "dv_headroom_gate": {"available": True, "met": dv_headroom_entry["met"], "gating": False, "adjudicating": False,
                             "achievable": dv_headroom_entry["measured"], "required": dv_headroom_entry["threshold"],
                             "control": dv_headroom_entry["control"]},
        "status": status,
        "outcome": status,
        "verdict": status,
        "interpretation": {
            "label": label,
            "preconditions": preconditions,
            "recorded_preconditions": recorded_preconditions,
            "preconditions_scope_note": (
                "interpretation.preconditions carries ONLY the per-arm readiness gates (green arms' entries "
                "on a partial run, per precondition_gate). interpretation.recorded_preconditions carries "
                "(a) the dv_headroom entry on e1coe_score_var(h=1) against the SAME-START real-endpoint "
                "goal_proximity variance -- an unmet value there IS the registered H-readout-saturation "
                "finding (instrument target), which must surface to governance without vacating the run's "
                "relative readings; and (b) the anchor arms' action-sensitivity ratios, which are results of "
                "the anchor (read via cr_bar_retained), not readiness failures."
            ),
            "criteria": criteria,
            "criteria_aggregation": (
                "PASS iff C0 (any arm green). C1-C5 CLASSIFY the outcome and are load_bearing:false "
                "by design (V3-EXQ-1000 learning #2: a fail-able load-bearing criterion under an "
                "unconditional PASS is a guaranteed vacuous_pass). The label carries the outcome."
            ),
            "criteria_non_degenerate": criteria_non_degenerate,
            "leg_verdicts": {k: {"hypothesis": v.get("hypothesis"), "tag": v.get("tag"), "state": v.get("state")}
                             for k, v in legs_h1.items()},
            "relative_var_reading": {
                x: {"var_reaches_branched_endpoint_var": a["var_reaches_branched_endpoint_var"], "n_branched": a["n_var_ge_branched"],
                    "var_reaches_from_reset_endpoint_var": a["var_reaches_real_endpoint_var"], "n_from_reset": a["n_var_ge_real"]}
                for x, a in legs_h1["A"].get("arms", {}).items()
            },
            "combination_rule": (
                "label = '<legA>__<legB>__<legC>' read at h=1 with majority = n_seeds//2+1. Leg A per green "
                f"anchor arm: centroid_restored (centroid_norm/real in [{CENTROID_BAND_LO:.3f},{CENTROID_BAND_HI:.3f}]) "
                f"and var_lifted (>= {VAR_LIFT_FACTOR}x ARM_RSD) -> supported; either declared-null disjunct "
                "-> that reading; H-fidelity-anchor is ELIMINATED only if no green anchor arm supports it. "
                f"Leg B: real-endpoint goal_proximity variance >= {C3_VAR_THRESHOLD} on majority -> eliminated "
                "(realvar_reaches_bar); < on majority -> supported (realvar_below_bar: instrument target). "
                f"Leg C: ARM_RSD goal-axis fraction / real within [{1/GOAL_AXIS_WITHIN_FACTOR:.2f},{GOAL_AXIS_WITHIN_FACTOR:.2f}] "
                "on majority -> eliminated; < on majority -> supported (rsd_goal_orthogonal). "
                "substrate_not_ready_requeue only when no arm is green; legs whose arms are red read *_undetermined_arm_red."
            ),
            "dv_symmetry_note": (
                "Every arm's DVs are functions of E1's learned weights through the rollout endpoints: "
                "e1coe_score_var (variance of a bounded nonlinear readout), centroid_norm ratio, spread "
                "ratio, goal-axis fraction and fidelity MSE. A changed training objective moves the "
                "endpoints non-uniformly (neither a common rescaling nor a permutation of the candidate "
                "index), so the manipulation reaches every DV. cr_ratio is scale-free and blind to a "
                "shrunk centroid (1000 learning #4) -- which is exactly why the centroid-norm band and the "
                "centroid-corrected cr_ratio are read beside it. Leg B's DV is arm-independent by design "
                "(a measurement of the real endpoints), so no manipulation-invariance question arises."
            ),
            "e1_alone_readout_note": (
                "*_e1alone fields are the V3-EXQ-980/1000 sibling readout at every checkpoint incl. h=30; "
                "RECORDED for the MECH-135 consumer question, never gating."
            ),
            "branched_same_start_note": (
                "Phase 4d fields (*branched*) are REAL endpoints for the SAME (start, sequence) as the "
                "rollouts, obtained by env deepcopy + agent latent-state restore at eval only. They add a "
                "same-start real reference beside the lineage's from-reset Phase 4b set; the registered "
                "legs read Phase 4b so this run stays comparable with 1000."
            ),
        },
        "source_substrate_entry": "SD-e1-rollout-consistency-training (ITEM 3 validated 2026-09-04; var bar open)",
        "source_autopsy": "failure_autopsy_V3-EXQ-1000_2026-09-04 (confirmed; fanout_recommendation; user decisions Q1-Q4)",
        "reference_runs": {
            "v3_exq_976": "v3_exq_976_sd_e1_item2_rollout_consistency_validation_20260902T114700Z_v3",
            "v3_exq_980": "v3_exq_980_sd_e1_h1c_readout_regime_e1_alone_20260902T212300Z_v3",
            "v3_exq_1000": "v3_exq_1000_sd_e1_item3_rollout_endpoint_contrastive_validation_20260903T213659Z_v3",
        },
    }

    for r in arm_results:
        key = f"{r['arm']}_seed{r['seed']}"
        for k in ("e1coe_score_var_h1", "cr_ratio_h1", "centroid_norm_ratio_vs_real_h1",
                  "real_goal_prox_var_h1", "goal_axis_fraction_rollout_h1", "goal_axis_fraction_real_h1",
                  "rsd_engaged_frac", "rsd_action_sensitivity_ratio", "anchor_term_mean", "rsd_term_mean",
                  "anchor_w_mean", "anchor_raw_grad_ratio_mean", "anchor_authority_mean", "anchor_n_clamped",
                  "n_e1_grad_steps", "n_e1_grad_steps_real", "base_prox", "z_goal_norm", "goal_template_source"):
            result[f"cell_{key}_{k}"] = r.get(k)

    result["arm_results"] = arm_results
    result["_agents_for_manifest"] = agents_for_manifest
    return result


if __name__ == "__main__":
    import argparse
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser(
        description=(
            "V3-EXQ-1006: SD-e1 var-bar portfolio -- OFF / RSD / RSD+anchor(EP) / RSD+anchor(RC), "
            "three registered legs read at every horizon (diagnostic)"
        )
    )
    parser.add_argument("--seeds", type=str, default=",".join(str(x) for x in SEEDS_DEFAULT))
    parser.add_argument("--world-dim", type=int, default=32)
    parser.add_argument("--self-dim", type=int, default=32)
    parser.add_argument("--train-episodes", type=int, default=100)
    parser.add_argument("--steps-per-episode", type=int, default=200)
    parser.add_argument("--rollout-horizon", type=int, default=30)
    parser.add_argument("--n-sequences", type=int, default=40)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--goal-max-steps", type=int, default=2000)
    parser.add_argument("--zworld-p0-episodes", type=int, default=ZWORLD_P0_EPISODES)
    parser.add_argument("--n-real-samples", type=int, default=N_REAL_SAMPLES)
    parser.add_argument("--dry-run", "--smoke-test", dest="dry_run", action="store_true")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    if args.dry_run:
        n_train = 2
        steps_ep = 50
        n_sequences = 12
        horizon = 10
        warmup = 5
        goal_max = 300
        zworld_p0 = 3
        n_real = 15
        seeds = seeds[:2]
        print("[V3-EXQ-1006] SMOKE TEST MODE", flush=True)
    else:
        n_train = args.train_episodes
        steps_ep = args.steps_per_episode
        n_sequences = args.n_sequences
        horizon = args.rollout_horizon
        warmup = args.warmup_steps
        goal_max = args.goal_max_steps
        zworld_p0 = args.zworld_p0_episodes
        n_real = args.n_real_samples

    t0 = time.perf_counter()
    result = run(
        seeds=seeds,
        world_dim=args.world_dim,
        self_dim=args.self_dim,
        n_train_episodes=n_train,
        steps_per_episode=steps_ep,
        n_sequences=n_sequences,
        rollout_horizon=horizon,
        n_warmup_steps=warmup,
        goal_max_steps=goal_max,
        zworld_p0_episodes=zworld_p0,
        n_real_samples=n_real,
        dry_run=args.dry_run,
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["timestamp_utc"] = ts
    result["run_timestamp"] = ts
    result["run_id"] = f"{EXPERIMENT_TYPE}_{ts}_v3"
    result["architecture_epoch"] = "ree_hybrid_guardrails_v1"

    agents_for_manifest = result.pop("_agents_for_manifest", [])

    full_config = {
        "seeds": seeds,
        "arms": ARM_ORDER,
        "arm_configs": ARM_CONFIGS,
        "world_dim": args.world_dim, "self_dim": args.self_dim,
        "n_train_episodes": n_train, "steps_per_episode": steps_ep,
        "n_sequences": n_sequences, "rollout_horizon": horizon,
        "n_warmup_steps": warmup, "goal_max_steps": goal_max,
        "zworld_p0_episodes": zworld_p0, "n_real_samples": n_real,
        "rc_horizon": RC_HORIZON, "rc_decay": RC_DECAY, "anchor_w_rc": ANCHOR_W_RC, "anchor_w_ep": ANCHOR_W_EP,
        "anchor_grad_authority": ANCHOR_GRAD_AUTHORITY, "anchor_w_clamp": [ANCHOR_W_MIN, ANCHOR_W_MAX],
        "anchor_authority_floor": ANCHOR_AUTHORITY_FLOOR,
        "rsd_horizon": RSD_HORIZON, "rsd_temperature": RSD_TEMPERATURE,
        "rsd_min_batch_classes": RSD_MIN_BATCH_CLASSES, "rsd_weight": RSD_WEIGHT,
        "rsd_batch_k": RSD_BATCH_K, "rsd_buffer_max": RSD_BUFFER_MAX,
        "train_window_h": TRAIN_WINDOW_H,
        "cr_real_floor": CR_REAL_FLOOR,
        "cr_rollout_collapse_ratio": CR_ROLLOUT_COLLAPSE_RATIO,
        "c3_var_threshold": C3_VAR_THRESHOLD,
        "centroid_band": [CENTROID_BAND_LO, CENTROID_BAND_HI],
        "var_lift_factor": VAR_LIFT_FACTOR,
        "goal_axis_within_factor": GOAL_AXIS_WITHIN_FACTOR,
        "min_real_samples_per_horizon_floor": MIN_REAL_SAMPLES_PER_HORIZON,
        "dv_headroom": {"dv_name": "e1coe_score_var_h1", "control_arm": "SAME-START real endpoints (Phase 4d); from-reset Phase 4b recorded",
                        "statistic": "explicit (majority-order over seeds)", "margin": DV_HEADROOM_MARGIN,
                        "criterion_threshold": C3_VAR_THRESHOLD, "gating": False, "adjudicating": False},
        "alpha_world": 0.9, "alpha_self": 0.3, "unified_latent_mode": False,
    }

    out_path = write_flat_manifest(
        result,
        dry_run=args.dry_run,
        config=full_config,
        seeds=seeds,
        script_path=Path(__file__),
        started_at=t0,
        agent=agents_for_manifest,
    )

    print(f"\nResult written to: {out_path}", flush=True)
    print(f"Status: {result['status']}", flush=True)
    print(f"Label: {result['interpretation']['label']}", flush=True)

    if args.dry_run:
        print("[V3-EXQ-1006] SMOKE TEST COMPLETE", flush=True)
        for k in ["status", "non_degenerate", "degeneracy_reason", "green_arms", "red_arms"]:
            print(f"  {k}: {result.get(k, 'N/A')}", flush=True)
        print(f"  label: {result['interpretation']['label']}", flush=True)
        print(f"  leg_verdicts: {result['interpretation']['leg_verdicts']}", flush=True)
        print(f"  dv_headroom_gate: {result.get('dv_headroom_gate')}", flush=True)
        print(f"  n_preconditions(adjudicating)={len(result['interpretation']['preconditions'])} n_recorded={len(result['interpretation']['recorded_preconditions'])}", flush=True)
        print(f"  e1_grad_step_gap_per_seed: {result.get('e1_grad_step_gap_per_seed')}", flush=True)
        for arm in ARM_ORDER:
            for seed in seeds:
                r = next(x for x in result["arm_results"] if x["arm"] == arm and x["seed"] == seed)
                p1 = r["per_h"]["1"]
                print(
                    f"  [smoke] {arm}/seed{seed}: var_h1={p1['e1coe_score_var']:.3e} cr_ratio_h1={p1['cr_ratio']:.3e} "
                    f"cent_ratio={p1['centroid_norm_ratio_vs_real']:.3f} spread_ratio={p1['spread_ratio_vs_real']:.3f} "
                    f"real_prox_var={p1['real_goal_prox_var']:.3e} axis roll/real={p1['goal_axis_fraction_rollout']:.3e}/{p1['goal_axis_fraction_real']:.3e} "
                    f"fid={p1['fidelity_mse']:.3e} rsd_sens={r['rsd_action_sensitivity_ratio']:.3f} anchor={r['anchor_term_mean']:.3e} "
                    f"anchor_w={r['anchor_w_mean']:.3g} authority={r['anchor_authority_mean']:.3g} raw_ratio={r['anchor_raw_grad_ratio_mean']:.3g}",
                    flush=True,
                )

    _outcome_raw = str(result.get("status", "FAIL")).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
        dry_run=args.dry_run,
    )
