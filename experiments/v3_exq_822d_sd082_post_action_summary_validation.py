"""V3-EXQ-822d -- SD-082 VALIDATION RETEST after the per-candidate-summary substrate fix.

WHAT THIS RUN IS. The designated `validation_experiment` for the SD-082
substrate_queue entry, whose per-candidate-summary AMEND landed
2026-08-30T11:32:08Z (ree-v3 merge ef88faa, from integration/sd082-percandidate-
summary 3aa45de). Both the substrate_queue entry and ree-v3/CLAUDE.md's
"SD-082 AMEND: per-candidate summary was a shared constant" section name
V3-EXQ-822d explicitly and record it as "PLANNED but NOT YET QUEUED ... a
/queue-experiment follow-on session is needed". This is that session's output.

THE DEFECT THIS RUN TESTS THE FIX FOR. Confirmed by
failure_autopsy_V3-EXQ-822c_2026-08-29.md (user-adjudicated). Under the default
candidate_summary_source='proposer', every caller's manual fallback read
trajectory.world_states[:, 0, :] -- but E2FastPredictor.rollout_with_world seeds
world_states=[initial_z_world], so index 0 is the rollout's SHARED initial world
state. Candidates differ only in the actions applied from t>=1, so that read is
bit-identical across all K candidates BY CONSTRUCTION and carries zero
candidate-discriminating information. SD-082's own centering step
(summaries - summaries.mean(dim=0)) then annihilates that constant to float32
cancellation noise. Measured consequence in V3-EXQ-822c:
rule_summary_magnitude_ratio 2.8e6-4.5e6 in all six cells (~4000x its 1e3
ceiling), while prop_delta nonetheless cleared the 1e-3 non-vacuity floor --
an authentic-looking, meaningless number. Severity: corrupting.

THE FIX, AND THE ONE LINE THAT CHANGES HERE. candidate_summary_source gains a
third value 'proposer_post_action' (default stays 'proposer', bit-identical);
agent._proposer_post_action_summaries() reads world_states[:, 1:, :].mean(0) --
the POST-ACTION states, reflecting each candidate's own action sequence -- at
zero extra model calls. compute_bias additionally gained a diagnostic-only
centering-degeneracy guard (LateralPFCConfig.candidate_summary_degeneracy_floor,
default 1e-4) that records candidate_summary_norm_pre/post_centering and flags
candidate_summary_degenerate, never raising and never changing the bias.

STEP 2.5a EMPIRICAL PROBE (this session, before authoring; 40 real ticks on the
822c config at seed 101). Plumbing reaches: REEConfig.from_dims ->
agent.config.candidate_summary_source == 'proposer_post_action'.
_candidate_world_summaries() returned NON-None on 40/40 ticks (the dispatch
bypasses every manual fallback, so the 822/822a/822b starvation route is closed
at source). And, directly measuring the defect and its repair on the same
candidates:

    post-centering cross-candidate summary norm
      LEGACY  ws[0, 0, :]           = 5.66e-08      <- float32 cancellation noise
      FIXED   proposer_post_action  = 4.01e-01      <- 7.1e6 x more signal

with candidate_summary_degenerate = False under the fix (pre 25.23 -> post 0.44).
That 7.1-million-fold separation is why the summary-spread readiness gate below
is set at a ratio floor of 1e-3: four orders of magnitude above the legacy
reading and ~17x below the observed fixed reading.

EXPERIMENT_PURPOSE = evidence (CHANGED from 822c's diagnostic, deliberately).
822b/822c were diagnostics asking WHY propagation read zero. That question is
answered and the substrate is fixed. This run asks SD-082's own stated
acceptance question, which the autopsy is explicit was never actually put:
"SD-082's centering is NOT falsified -- it is UNTESTED, because its input never
carried the cross-candidate variance it was designed to preserve." Both claims
are pending_retest_after_substrate=true. So this run is scored.

PRE-REGISTERED ACCEPTANCE (SD-082's own, per the autopsy Section 3 and the
amended failure_record target -- constants below, not derived from this run):
  C1 propagation non-vacuous : on_prop_delta_mean >= 1e-3
  C2 ON>OFF contrast         : (on - off) > 1e-3 on a MAJORITY of seeds
Both load-bearing; overall PASS = readiness AND C1 AND C2.

SEEDS RAISED 3 -> 5, ON THE AUTOPSY'S OWN FINDING. Section 4 grades Scale
"adequate for the primary finding; INSUFFICIENT for the ON-vs-OFF read (3 seeds,
direction inverts)" -- 822c's per-seed contrast was +0.001771 / +0.000033 /
-0.002036, i.e. carried entirely by one seed. C2 is exactly that read, so
running it at n=3 would reproduce the same under-powering the autopsy flagged.
Five seeds make "majority" 3/5 rather than 2/3. Seeds 101/202/303 are retained
verbatim from 822a/822c for continuity; 404/505 are the new draw.

READINESS (gating; unmet -> outcome FAIL, non_degenerate=false,
interpretation.label = substrate_not_ready_requeue, never a substrate verdict):
  (a) z_world common-mode cone present               [== 822a/822c, floor 0.90]
  (b) ON pool differentiated / OFF pool pinned       [== 822a/822c]
  (c) ON rule active on >= frac floor of P2 ticks    [== 822a/822c]
  (d) NEW -- prop-sample sufficiency: EVERY cell's n_prop_samples clears a floor
      of 20. This is the gate the substrate_queue entry names in so many words
      ("asserts n_prop_samples > 0 as a readiness gate BEFORE trusting any
      prop_delta aggregate -- the exact measurement-starvation gap 822/822a/822b
      fell into"), set at 20 rather than the literal 1 because 822c observed
      145-200 per cell, so 20 is comfortably non-binding while still strictly
      implying > 0. The lesson it encodes is autopsy Section 5's: a no-data
      default (`fmean(xs) if xs else 0.0`) that shares the numeric value of a
      meaningful result is the most expensive shape in this corpus.
  (e) NEW -- fix engaged: cross-candidate summary spread survives centering.
      Worst-cell median of (candidate_summary_norm_post_centering /
      _pre_centering) clears 1e-3, AND no cell ever set
      candidate_summary_degenerate, AND the manual ws[0,0,:] fallback was taken
      ZERO times (a nonzero count means the dispatch did not engage).
  (f) capture positive control: the head-diagnostics path returns finite,
      in-range values on a synthetic pre-training forward pass, every cell
      [== 822c].

SAME-STATISTIC DISCIPLINE for gate (e) (the V3-EXQ-643 rule). C1/C2 route on
prop_delta, a rule-state-ablation delta whose MEANINGFULNESS depends entirely on
the candidate summaries carrying cross-candidate variance. Gate (e) therefore
asserts a cross-candidate SPREAD (the post-centering norm IS the deviation from
the cross-candidate mean, since centering removes that mean), scale-normalised
by the pre-centering norm -- NOT a magnitude, mean_abs, or max_abs, any of which
can be large while the spread is ~0. That is precisely the failure mode here:
the legacy read had pre-centering norm 25.23 (large) and post-centering spread
5.66e-08 (nil). A magnitude-shaped gate would have passed it.

DV-SYMMETRY INVARIANCE (Step 3 mandatory declaration, per arm). The manipulation
is crf_cue_centering (OFF -> collapsed rule pool, ON -> differentiated).

  * prop_delta_mean -- mean over K candidates of |bias(rule_state) - bias(0)|.
    Symmetry group: permutation of candidates (it is a symmetric aggregate), and
    addition of a candidate-independent constant to the per-candidate DIFFERENCE.
    The manipulation is not a candidate permutation. It is NOT invariant under
    the second either -- but ONLY because of the fix, and this is the whole
    point. rule_state is broadcast identically across candidates
    (rule_repeated = rule_state.expand(k, -1)); the head is nonlinear
    (Linear->ReLU->Linear), so changing rule_state moves each candidate's bias by
    a DIFFERENT amount only when `summaries` actually differ across candidates.
    Under the pre-fix 'proposer' path summaries were constant, so the per-
    candidate difference was one shared constant -- the manipulation WAS
    invariant in the relevant sense, and 822c's prop_delta was an arithmetic
    artifact rather than a measurement (autopsy disposition (b)). Gate (e) is
    what certifies the invariance is broken before C1/C2 are read at all.
  * rule_flip_frac -- fraction of ticks where the argmax over the bias vector
    moves under rule-state ablation. Invariant under adding a broadcast constant
    to the bias vector (a constant cannot move an argmax). Under the pre-fix path
    the ablation delta WAS exactly such a constant, so flip was structurally
    pinned at zero -- and V3-EXQ-822c measured rule_flip_frac = 0.0000 in all six
    cells, which is an independent arithmetic confirmation of the autopsy's
    structural finding. Under proposer_post_action the delta is per-candidate and
    a flip is no longer forbidden. Recorded as a DIAGNOSTIC, NOT gating and NOT
    load-bearing: flip > 0 is confirmatory, but flip == 0 is not decisive, since
    a genuinely near-invariant head can produce no flip without any symmetry
    pinning it. Reading a zero here as a verdict would repeat this lineage's
    founding error.
  * rule_state_diff -- mean pairwise (1 - cosine) over sampled rule_states. The
    manipulation acts directly on rule-pool differentiation, which is what this
    measures; no invariance. [== 822a/822c]
  * candidate_summary_norm_post/pre_centering -- crf_cue_centering does not touch
    candidate world summaries at all, so this readout is EXPECTED to be
    arm-independent. That is by design, not a null finding: it is a readiness
    measure of the fix, not an effect measure of the manipulation.

GOV-REUSE-1 (Step 2.4). Decisive readout: on_prop_delta_mean and the per-seed
ON-OFF contrast, measured under candidate_summary_source='proposer_post_action'.
That config value did not exist before ree-v3 3aa45de (2026-08-30); a grep of
REE_assembly/evidence/experiments/ finds 0 manifests mentioning
'proposer_post_action' and 0 mentioning 'candidate_summary_degenerate'. The
readout is not recorded and not derivable post-hoc from any prior manifest --
822/822a/822b recorded n_prop_samples = 0 in all 18 cells (no data at all), and
822c measured the DEFECTIVE path. Not recoverable -> run.

RE-DERIVE BRAKE (Step 2.5b) -- FIRES, AND IS RELEASED. Counted under the skill's
own predicate (substrate_ceiling category OR non_contributory direction, keyed by
run at most-recent adjudication): SD-078 = 4 hits, SD-082 = 2 hits, both at or
over the threshold of 2. Braking autopsies: failure_autopsy_816c-822_2026-07-26,
failure_autopsy_batch-822a-826-817a-827_2026-07-26,
failure_autopsy_2026-07-28-sweep, failure_autopsy_V3-EXQ-822c_2026-08-29.
RELEASED because the named upstream substrate is now BUILT: the most recent
counted autopsy's recommended_substrate_queue_entry is
{action: amend, target_sd_id: SD-082}, and that amend is IMPLEMENTED
(ree-v3 ef88faa, 2026-08-30; ree-v3/CLAUDE.md "SD-082 AMEND: per-candidate
summary ..."). Per the skill, a released brake makes the re-test meaningful --
this is not a fourth blind lettered iteration around the same ceiling, it is the
first test of the claim on substrate that can carry the signal.
(Note the 822c autopsy recorded "brake does not fire" using the narrower R1-R3
predicate, which counts only literal substrate_ceiling readings. Both readings
are stated so a later reader is not left reconciling them.)

STEP 2.5c substrate-path overlap gate. Resolved by call-trace over 30 real ticks
of this exact driver path rather than by module-name matching. Of the OPEN
corrupting substrate_queue entries: salience_coordinator.py
(mode-governance-engagement) NOT executed; blocked_agency.py
(sd_blocked_agency_mismatch_floor_calibration) NOT executed;
ContextMemory.write / compute_write_addressing_loss / agent.compute_prediction_loss
(contextmemory-write-path-addressing-degeneracy) NOT executed -- this driver is a
pure forward/select path and never trains E1, so the degenerate WRITE path is
never taken (only e1_deep read/forward/generate_prior/predict_long_horizon/
get_schema_salience/reset_hidden_state are reached); probe_warmup.py
(SD-PROBE-WARMUP) not imported. The ONLY overlap is SD-082 itself, whose entry
this run exists to validate and whose corrupting failure_record was marked
resolved by the fix above.

ETHICS PREFLIGHT (Step 2.6). All involvement flags false, decision allow. No
negative-valence drive, no suffering-like accumulator, no self-model, no
inescapability manipulation, no offline replay over harm, no social/language, no
human or clinical data. Pre-ethical V3 instrumentation only (SENT-0).

SUPERSEDES: nothing. 822c is NOT superseded -- it is the diagnostic that measured
the defect, its failure_record is marked `resolved` (not `superseded`) by the fix,
and its finding stands. Stated explicitly so a later reader does not infer a
supersession from the shared lineage.

ARM REUSE: in-line emit only (arm_cell defaults, driver folded into the hash). No
canonical baseline module and no separate mint job -- this lineage re-tunes its
OFF path at every letter (822 -> 822a consumer flags -> 822b capture flag ->
822c measurement fix -> 822d summary source), which is exactly the
substrate-in-flux calibration family the skill's carve-out names: the fingerprint
correctly refuses a drifted mint, so there is nothing to skip.

See SD-078, SD-082, SD-033a, ARC-063, SD-008;
REE_assembly/docs/architecture/sd_082_rule_selection_action_consumer.md;
REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-822c_2026-08-29.md;
ree-v3/CLAUDE.md "SD-082 AMEND: per-candidate summary was a shared constant";
experiments/v3_exq_822c_sd082_candidate_summary_fallback_fix.py (parent design).
"""
from __future__ import annotations

import argparse
import math
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ree_core.utils.config import REEConfig  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from experiment_protocol import emit_outcome  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402

_ZG = ZGoalStreamAccumulator()
# One representative agent per arm, kept alive purely so stamp_recording_core can
# read enabled_default_off_flags off a live .config. Bounded at 2 (not 10) -- the
# accumulator above, not this dict, is what measures z_goal liveness across all
# cells, and z_goal_stream_stats takes precedence over agent= for that block.
_ARM_AGENTS: Dict[str, Any] = {}

EXPERIMENT_TYPE = "v3_exq_822d_sd082_post_action_summary_validation"
EXPERIMENT_PURPOSE = "evidence"
CLAIM_IDS = ["SD-078", "SD-082"]

# `head_diagnostics_capture_active` and `summary_dispatch_engaged` are config-echo
# plus finiteness/count checks on deterministic forward passes, not hand-tuned
# numeric scoring predicates: capture_head_diagnostics=True and
# candidate_summary_source="proposer_post_action" are passed explicitly in
# _make_agent (both arms), so both are true by construction unless the three-site
# plumbing itself is broken -- which the SD-082 amend's own 22 contracts, not a
# precondition here, are the right place to catch.
ANCHOR_REACHABILITY_EXEMPT = (
    "head_diagnostics_capture_active / summary_dispatch_engaged are config echoes "
    "(capture_head_diagnostics and candidate_summary_source are set explicitly, not "
    "learned or tuned) plus finiteness/count checks on deterministic forward passes; "
    "reachable by construction, no scoring predicate involved."
)

SEEDS = [101, 202, 303, 404, 505]  # 101/202/303 == 822a/822c; 404/505 the new draw.
ARMS = ["ARM_OFF", "ARM_ON"]
P0_WARMUP_EPISODES = 60
P1_BIAS_TRAIN_EPISODES = 70
P2_MEASUREMENT_EPISODES = 40
STEPS_PER_EPISODE = 48

# --- Pre-registered readiness thresholds (a)-(c) identical to 822a/822c. ---
CONE_MIN_COSINE_FLOOR = 0.90
DIST_FLOOR = 0.05
MIN_LIVE_RULES = 2
OFF_MAX_LIVE_CEIL = 1
FRAC_ACTIVE_FLOOR = 0.10
SEED_PASS_FRACTION = 3.0 / 5.0  # majority of 5 (822a/822c used 2/3 of 3).

# --- Pre-registered acceptance (SD-082's own, per the autopsy). ---
PROP_NONVAC_FLOOR = 1e-3   # C1: on_prop_delta_mean must clear this.
CONTRAST_MARGIN = 1e-3     # C2: per-seed (on - off) must EXCEED this, signed.
# CONTRAST_MARGIN is set equal to PROP_NONVAC_FLOOR deliberately and named
# separately so the choice is explicit rather than an accidental reuse: it demands
# the ON-over-OFF effect be at least as large as the smallest propagation the
# claim calls meaningful at all.

# --- Pre-registered readiness thresholds (d)-(f), new in this letter. ---
PROP_SAMPLE_FLOOR = 20              # (d) worst-cell n_prop_samples; 822c saw 145-200.
SUMMARY_SPREAD_RATIO_FLOOR = 1e-3   # (e) post/pre centering norm; legacy ~2.2e-9, fixed ~1.8e-2.
HEAD_DIAG_SAMPLE_FLOOR = 5          # (f) == 822c.

# --- Diagnostic (non-gating) interpretation bands, carried from 822c. ---
MAGNITUDE_RATIO_LOW = 1e-3
MAGNITUDE_RATIO_HIGH = 1e3
DEAD_RELU_HIGH_FLOOR = 0.90
DEAD_RELU_MED_FLOOR = 0.50
LAST_LAYER_WEIGHT_DELTA_FLOOR = 1e-3

# P1 REINFORCE (mirror V3-EXQ-598b / 654f / 822a / 822c).
LR_LPFC_BIAS = 5e-4
REINFORCE_BATCH_SIZE = 32
OUTCOME_BUF_MAX = 512
POLICY_TEMPERATURE = 1.0
ADV_MIN_THRESHOLD = 0.005
EMA_DECAY = 0.9

ENV_KWARGS = dict(size=8, num_hazards=2, num_resources=6, use_proxy_fields=True)


def _build_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(seed=seed, **ENV_KWARGS)


def _make_agent(env: CausalGridWorldV2, centering: bool) -> REEAgent:
    """Identical to 822c's _make_agent plus candidate_summary_source=
    "proposer_post_action" on BOTH arms -- the one substrate change this letter
    validates. Everything else (schedule, env, consumer flags, capture flag) is
    held fixed against 822c so the comparison is controlled."""
    return REEAgent(REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=4,
        alpha_world=0.9,
        use_lateral_pfc_analog=True,
        lateral_pfc_train_rule_bias_head=True,
        lateral_pfc_rule_readout_consumer=True,
        lateral_pfc_capture_head_diagnostics=True,
        candidate_summary_source="proposer_post_action",  # <-- the only new flag vs 822c.
        use_gated_policy=True,
        use_candidate_rule_field=True,
        crf_persist_rules_across_episode_reset=True,
        crf_mature_pool_dynamics=True,
        crf_availability_maintenance=True,
        crf_maintenance_floor=0.45,
        crf_maintenance_couple_to_theta=True,
        crf_tolerance_conflict_cap=3,
        crf_cue_centering=centering,
        crf_cue_baseline_alpha=0.02,
    ))


def _config_slice(centering: bool) -> Dict[str, Any]:
    return {
        "env": dict(ENV_KWARGS),
        "schedule": {"p0": P0_WARMUP_EPISODES, "p1": P1_BIAS_TRAIN_EPISODES,
                     "p2": P2_MEASUREMENT_EPISODES, "steps": STEPS_PER_EPISODE},
        "alpha_world": 0.9,
        "use_lateral_pfc_analog": True,
        "lateral_pfc_train_rule_bias_head": True,
        "lateral_pfc_rule_readout_consumer": True,
        "lateral_pfc_capture_head_diagnostics": True,
        "candidate_summary_source": "proposer_post_action",
        "use_gated_policy": True,
        "use_candidate_rule_field": True,
        "crf_persist_rules_across_episode_reset": True,
        "crf_mature_pool_dynamics": True,
        "crf_availability_maintenance": True,
        "crf_maintenance_floor": 0.45,
        "crf_maintenance_couple_to_theta": True,
        "crf_tolerance_conflict_cap": 3,
        "crf_cue_centering": centering,
        "crf_cue_baseline_alpha": 0.02,
    }


def _cone_min_cosine(vecs: List[torch.Tensor]) -> float:
    if len(vecs) < 2:
        return 1.0
    m = torch.nn.functional.normalize(torch.stack(vecs).float(), dim=-1)
    c = m @ m.t()
    n = c.shape[0]
    iu = torch.triu_indices(n, n, offset=1)
    return float(c[iu[0], iu[1]].min())


def _mean_pairwise_one_minus_cosine(vecs: List[torch.Tensor]) -> float:
    vecs = [v for v in vecs if v is not None and float(v.norm()) > 1e-9]
    if len(vecs) < 2:
        return 0.0
    m = torch.nn.functional.normalize(torch.stack(vecs).float(), dim=-1)
    c = m @ m.t()
    n = c.shape[0]
    iu = torch.triu_indices(n, n, offset=1)
    return float((1.0 - c[iu[0], iu[1]]).mean())


def _candidate_summaries(
    agent: REEAgent, candidates, counters: Dict[str, int]
) -> Optional[torch.Tensor]:
    """With candidate_summary_source="proposer_post_action" the dispatch in
    agent._candidate_world_summaries() returns a real [K, world_dim] tensor, so
    the manual ws[0, 0, :] fallback below should NEVER be taken. It is retained
    only so a plumbing regression degrades to 822c's behaviour instead of
    crashing -- and every fallback is COUNTED, because a nonzero count means
    'proposer_post_action' did not engage and readiness gate (e) must fail rather
    than the run silently measuring the defective path again."""
    summ = agent._candidate_world_summaries(candidates)
    if summ is not None:
        counters["dispatch"] += 1
        return summ.detach()
    if not candidates:
        return None
    counters["fallback"] += 1
    cand_world_list: List[torch.Tensor] = []
    for c in candidates:
        if c.world_states is not None:
            ws = c.get_world_state_sequence()  # [batch, horizon+1, world_dim]
            cand_world_list.append(ws[0, 0, :])
        elif agent._current_latent is not None:
            cand_world_list.append(agent._current_latent.z_world[0].detach())
        else:
            return None
    return torch.stack(cand_world_list, dim=0).detach()


def _head_diag_snapshot(agent: REEAgent) -> Dict[str, float]:
    """Unconditional (always-fresh, never-latched) weight-norm read."""
    st = agent.lateral_pfc.get_state()
    return {
        "capture_head_diagnostics": bool(st.get("capture_head_diagnostics", False)),
        "first_linear_weight_norm": float(st["rule_bias_head_first_linear_weight_norm"]),
        "first_linear_bias_norm": float(st["rule_bias_head_first_linear_bias_norm"]),
        "last_linear_weight_norm": float(st["rule_bias_head_last_linear_weight_norm"]),
        "last_linear_bias_norm": float(st["rule_bias_head_last_linear_bias_norm"]),
    }


def _init_positive_control(agent: REEAgent) -> Dict[str, Any]:
    """Readiness (f) positive control: force one real compute_bias() call with
    capture_head_diagnostics=True on a synthetic 3-candidate input BEFORE any
    training, confirming the capture path returns finite, in-range values
    independently of whether training later moves anything. The synthetic input
    is torch.randn, i.e. genuinely differentiated across candidates -- so this
    also confirms the degeneracy guard does NOT false-positive on a
    non-degenerate summary."""
    lpfc = agent.lateral_pfc
    wd = agent.config.latent.world_dim
    synth = torch.randn(3, wd, device=agent.device)
    with torch.no_grad():
        lpfc.compute_bias(synth)
        st = lpfc.get_state()
    snap = _head_diag_snapshot(agent)
    snap["dead_relu_frac"] = float(st["hidden_dead_relu_frac"])
    snap["magnitude_ratio"] = float(st["rule_summary_magnitude_ratio"])
    snap["summary_norm_pre_centering"] = float(st["candidate_summary_norm_pre_centering"])
    snap["summary_norm_post_centering"] = float(st["candidate_summary_norm_post_centering"])
    snap["summary_degenerate"] = bool(st["candidate_summary_degenerate"])
    snap["dead_relu_frac_finite_in_range"] = (
        math.isfinite(snap["dead_relu_frac"]) and 0.0 <= snap["dead_relu_frac"] <= 1.0)
    snap["magnitude_ratio_finite"] = math.isfinite(snap["magnitude_ratio"])
    return snap


def _prop_delta_and_flip_with_diag(
    agent: REEAgent, summaries: torch.Tensor
) -> Optional[Tuple[float, bool, Dict[str, Any]]]:
    """822c's latch-safe measurement, extended with the SD-082 amend's three new
    degeneracy fields plus rule_state_norm (the autopsy Section 3 "cheap
    confirmer": without it the magnitude ratio cannot be decomposed into its
    numerator and denominator, which is the one premise that autopsy could not
    verify from its own artifact).

    LATCH ORDERING (unchanged from 822c and load-bearing). compute_bias() is
    called TWICE per active tick -- once on the real rule_state, once on a zeroed
    one -- and hidden_dead_relu_frac / rule_summary_magnitude_ratio are LATCHED
    (get_state() returns whatever the most recent call wrote). get_state() is
    therefore read immediately after the FIRST call and BEFORE rule_state is
    zeroed, so the diagnostics and rule_state_norm both belong to the real call.
    The two centering-degeneracy norms depend only on candidate_world_summaries
    and so are call-invariant, but they are read at the same point anyway."""
    lpfc = getattr(agent, "lateral_pfc", None)
    if lpfc is None or summaries is None:
        return None
    try:
        with torch.no_grad():
            bias_field = lpfc.compute_bias(summaries).detach().clone()
            diag_state = lpfc.get_state()
            head_diag = {
                "dead_relu_frac": float(diag_state["hidden_dead_relu_frac"]),
                "magnitude_ratio": float(diag_state["rule_summary_magnitude_ratio"]),
                "rule_state_norm": float(diag_state["rule_state_norm"]),
                "summary_norm_pre_centering": float(
                    diag_state["candidate_summary_norm_pre_centering"]),
                "summary_norm_post_centering": float(
                    diag_state["candidate_summary_norm_post_centering"]),
                "summary_degenerate": bool(diag_state["candidate_summary_degenerate"]),
            }
            saved = lpfc.rule_state.detach().clone()
            lpfc.rule_state.zero_()
            bias_zero = lpfc.compute_bias(summaries).detach().clone()
            lpfc.rule_state.copy_(saved)
        delta = float((bias_field - bias_zero).abs().mean().item())
        flip = int(bias_field.argmax().item()) != int(bias_zero.argmax().item())
        if not math.isfinite(delta):
            return None
        return delta, bool(flip), head_diag
    except Exception:
        return None


def _lpfc_reinforce_loss(agent: REEAgent,
                         outcome_buf: List[Tuple[torch.Tensor, int, float]],
                         baseline: float, device) -> torch.Tensor:
    if agent.lateral_pfc is None or len(outcome_buf) < 2:
        return torch.zeros(1, device=device)
    n = len(outcome_buf)
    idxs = np.random.choice(n, size=min(REINFORCE_BATCH_SIZE, n), replace=False)
    terms: List[torch.Tensor] = []
    for i in idxs:
        cand_features, sel_idx, ep_return = outcome_buf[int(i)]
        adv = ep_return - baseline
        if abs(adv) < ADV_MIN_THRESHOLD:
            continue
        bias = agent.lateral_pfc.compute_bias(cand_features.to(device))
        log_p = F.log_softmax(-bias / POLICY_TEMPERATURE, dim=0)
        terms.append(-adv * log_p[min(sel_idx, bias.shape[0] - 1)])
    if not terms:
        return torch.zeros(1, device=device)
    return torch.stack(terms).mean()


def _run_cell(arm: str, seed: int) -> Dict[str, Any]:
    centering = (arm == "ARM_ON")
    print(f"Seed {seed} Condition {arm}", flush=True)
    with arm_cell(seed, config_slice=_config_slice(centering),
                  script_path=Path(__file__)) as cell:
        env = _build_env(seed)
        agent = _make_agent(env, centering)
        crf = agent.candidate_rule_field
        wd = agent.config.latent.world_dim
        bias_opt = torch.optim.Adam(
            list(agent.lateral_pfc.bias_head_parameters()), lr=LR_LPFC_BIAS)

        head_diag_by_phase: Dict[str, Any] = {"init": _init_positive_control(agent)}
        summary_counters = {"dispatch": 0, "fallback": 0}

        total_eps = P0_WARMUP_EPISODES + P1_BIAS_TRAIN_EPISODES + P2_MEASUREMENT_EPISODES
        p2_start = P0_WARMUP_EPISODES + P1_BIAS_TRAIN_EPISODES

        reinforce_baseline = 0.0
        outcome_buf: List[Tuple[torch.Tensor, int, float]] = []

        zworlds: List[torch.Tensor] = []
        rule_state_samples: List[torch.Tensor] = []
        prop_deltas: List[float] = []
        flips: List[int] = []
        p2_dead_relu_samples: List[float] = []
        p2_magnitude_ratio_samples: List[float] = []
        p2_rule_state_norms: List[float] = []
        p2_spread_ratios: List[float] = []
        n_summary_degenerate_ticks = 0
        n_magnitude_ratio_inf = 0
        p2_ticks = 0
        p2_active_ticks = 0
        max_live = 0

        for ep in range(total_eps):
            is_p1 = (P0_WARMUP_EPISODES <= ep < p2_start)
            is_p2 = (ep >= p2_start)
            phase = "P2" if is_p2 else ("P1" if is_p1 else "P0")
            _, obs = env.reset()
            agent.reset()
            ep_reward = 0.0
            ep_buf: List[Tuple[torch.Tensor, int]] = []

            for _step in range(STEPS_PER_EPISODE):
                latent = agent.sense(obs["body_state"], obs["world_state"])
                ticks = agent.clock.advance()
                e1 = (agent._e1_tick(latent) if ticks.get("e1_tick")
                      else torch.zeros(1, wd, device=agent.device))
                candidates = agent.generate_trajectories(latent, e1, ticks)

                p1_snap: Optional[torch.Tensor] = None
                if is_p1 and candidates and len(candidates) >= 2:
                    cs = _candidate_summaries(agent, candidates, summary_counters)
                    if cs is not None and torch.isfinite(cs).all():
                        p1_snap = cs.clone()

                action = agent.select_action(candidates, ticks)
                if action is None:
                    idx = int(np.random.randint(0, 4))
                    action = torch.zeros(1, 4, device=agent.device)
                    action[0, idx] = 1.0
                    agent._last_action = action
                committed_class = int(action[0].argmax().item())

                if is_p1 and p1_snap is not None:
                    sel = 0
                    for ci, c in enumerate(candidates):
                        if (getattr(c, "actions", None) is not None
                                and c.actions.shape[1] >= 1
                                and int(c.actions[:, 0, :].argmax(-1).reshape(-1)[0].item())
                                == committed_class):
                            sel = min(ci, p1_snap.shape[0] - 1)
                            break
                    ep_buf.append((p1_snap, sel))

                if is_p2 and ticks.get("e3_tick", False):
                    p2_ticks += 1
                    if len(zworlds) < 200:
                        zworlds.append(latent.z_world.detach().reshape(-1).clone())
                    st = crf.get_state()
                    n_active = int(st.get("crf_n_active_last", 0))
                    max_live = max(max_live, int(st.get("crf_n_live", len(crf._rules))))
                    if n_active >= 1:
                        p2_active_ticks += 1
                        rs = agent.lateral_pfc.rule_state.detach().reshape(-1).clone()
                        if float(rs.norm()) > 1e-9:
                            rule_state_samples.append(rs)
                        if candidates and len(candidates) >= 2:
                            summ = _candidate_summaries(agent, candidates, summary_counters)
                            if summ is not None and torch.isfinite(summ).all():
                                pf = _prop_delta_and_flip_with_diag(agent, summ)
                                if pf is not None:
                                    delta, flip, hd = pf
                                    prop_deltas.append(delta)
                                    flips.append(1 if flip else 0)
                                    p2_dead_relu_samples.append(hd["dead_relu_frac"])
                                    p2_rule_state_norms.append(hd["rule_state_norm"])
                                    if hd["summary_degenerate"]:
                                        n_summary_degenerate_ticks += 1
                                    pre = hd["summary_norm_pre_centering"]
                                    if pre > 0.0:
                                        p2_spread_ratios.append(
                                            hd["summary_norm_post_centering"] / pre)
                                    # The magnitude ratio is rule_norm/summary_norm and is
                                    # 0.0 by construction on a tick where rule_state is
                                    # still zero -- recording those would drag the median
                                    # to 0 and manufacture a spurious LOW-band flag. Only
                                    # ticks with a genuinely live rule_state are recorded.
                                    mr = hd["magnitude_ratio"]
                                    if hd["rule_state_norm"] > 1e-9:
                                        if math.isfinite(mr):
                                            p2_magnitude_ratio_samples.append(mr)
                                        else:
                                            n_magnitude_ratio_inf += 1

                _, _h, done, _info, obs = env.step(int(action.argmax(dim=-1).item()))
                if is_p1:
                    ep_reward += float(_h)
                if done:
                    break

            if is_p1:
                reinforce_baseline = (EMA_DECAY * reinforce_baseline
                                      + (1.0 - EMA_DECAY) * ep_reward)
                for cand_features, sel in ep_buf:
                    outcome_buf.append((cand_features, sel, ep_reward))
                if len(outcome_buf) > OUTCOME_BUF_MAX:
                    outcome_buf = outcome_buf[-OUTCOME_BUF_MAX:]
                l_loss = _lpfc_reinforce_loss(
                    agent, outcome_buf, reinforce_baseline, agent.device)
                if l_loss.requires_grad:
                    bias_opt.zero_grad()
                    l_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        agent.lateral_pfc.bias_head_parameters(), 1.0)
                    bias_opt.step()

            if (ep + 1) == P0_WARMUP_EPISODES:
                head_diag_by_phase["post_p0"] = _head_diag_snapshot(agent)
            elif (ep + 1) == p2_start:
                head_diag_by_phase["post_p1"] = _head_diag_snapshot(agent)
            elif (ep + 1) == total_eps:
                head_diag_by_phase["post_p2"] = _head_diag_snapshot(agent)

            if (ep + 1) % 20 == 0 or (ep + 1) == total_eps:
                print(f"  [train] crf seed={seed} arm={arm} phase={phase} "
                      f"ep {ep+1}/{total_eps} live={len(crf._rules)} "
                      f"minted={crf._n_minted}", flush=True)

        rule_state_diff = _mean_pairwise_one_minus_cosine(rule_state_samples)
        prop_delta_mean = statistics.fmean(prop_deltas) if prop_deltas else 0.0
        rule_flip_frac = (sum(flips) / len(flips)) if flips else 0.0
        frac_active = (p2_active_ticks / p2_ticks) if p2_ticks else 0.0

        dead_relu_frac_p2_mean = (statistics.fmean(p2_dead_relu_samples)
                                  if p2_dead_relu_samples else 0.0)
        dead_relu_frac_p2_worst = max(p2_dead_relu_samples) if p2_dead_relu_samples else 0.0
        magnitude_ratio_p2_median = (statistics.median(p2_magnitude_ratio_samples)
                                     if p2_magnitude_ratio_samples else 0.0)
        # Worst (MINIMUM) spread ratio, not the mean -- the readiness gate quantifies
        # over ticks ("centering never annihilated the summary"), so the reported
        # statistic must be the extremum the gate tests, else an in-band mean can
        # mask an out-of-band tick and the indexer's recompute silently passes it.
        spread_ratio_p2_min = min(p2_spread_ratios) if p2_spread_ratios else 0.0
        spread_ratio_p2_median = (statistics.median(p2_spread_ratios)
                                  if p2_spread_ratios else 0.0)
        rule_state_norm_p2_median = (statistics.median(p2_rule_state_norms)
                                     if p2_rule_state_norms else 0.0)
        last_layer_weight_delta_init_to_p1 = float(
            head_diag_by_phase.get("post_p1", {}).get("last_linear_weight_norm", 0.0)
            - head_diag_by_phase["init"]["last_linear_weight_norm"])

        row = {
            "arm": arm,
            "seed": seed,
            "rule_state_diff": float(rule_state_diff),
            "prop_delta_mean": float(prop_delta_mean),
            "rule_flip_frac": float(rule_flip_frac),
            "crf_max_pairwise_rule_dist": float(crf.max_pairwise_rule_distance()),
            "crf_max_live_rules": int(max_live),
            "crf_live_rules_final": int(len(crf._rules)),
            "crf_n_minted": int(crf._n_minted),
            "crf_frac_active_p2": float(frac_active),
            "crf_baseline_allocated": crf._baseline is not None,
            "n_rule_state_samples": len(rule_state_samples),
            "n_prop_samples": len(prop_deltas),
            "zworld_cone_min_cosine": _cone_min_cosine(zworlds),
            "n_zworld_sampled": len(zworlds),
            # --- SD-082 amend: fix-engagement readouts (new in 822d) ---
            "n_summary_dispatch": int(summary_counters["dispatch"]),
            "n_summary_fallback": int(summary_counters["fallback"]),
            "summary_spread_ratio_p2_min": float(spread_ratio_p2_min),
            "summary_spread_ratio_p2_median": float(spread_ratio_p2_median),
            "n_summary_degenerate_ticks": int(n_summary_degenerate_ticks),
            "n_spread_ratio_samples": len(p2_spread_ratios),
            "rule_state_norm_p2_median": float(rule_state_norm_p2_median),
            "n_rule_state_norm_samples": len(p2_rule_state_norms),
            # --- head-internals diagnostics, carried from 822c ---
            "n_head_diag_samples": len(p2_dead_relu_samples),
            "hidden_dead_relu_frac_p2_mean": float(dead_relu_frac_p2_mean),
            "hidden_dead_relu_frac_p2_worst": float(dead_relu_frac_p2_worst),
            "rule_summary_magnitude_ratio_p2_median": float(magnitude_ratio_p2_median),
            "n_magnitude_ratio_live_samples": len(p2_magnitude_ratio_samples),
            "n_magnitude_ratio_inf": int(n_magnitude_ratio_inf),
            "head_diag_by_phase": head_diag_by_phase,
            "last_layer_weight_delta_init_to_p1": last_layer_weight_delta_init_to_p1,
        }
        cell.stamp(row)
    _ZG.observe(agent)
    _ARM_AGENTS[arm] = agent
    passed = (row["rule_state_diff"] > 0.05) if centering else \
             (row["crf_max_live_rules"] <= OFF_MAX_LIVE_CEIL)
    print(f"verdict: {'PASS' if passed else 'FAIL'}", flush=True)
    return row


def _worst_cell(rows: List[Dict[str, Any]], key: str, mode: str = "min"):
    fn = min if mode == "min" else max
    r = fn(rows, key=lambda x: x[key])
    return float(r[key]), f"{r['arm']}/seed{r['seed']}"


def run_experiment() -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for seed in SEEDS:
        for arm in ARMS:
            rows.append(_run_cell(arm, seed))

    off = [r for r in rows if r["arm"] == "ARM_OFF"]
    on = [r for r in rows if r["arm"] == "ARM_ON"]

    # --- READINESS (a)-(c): identical in form to 822a/822c. ---
    cone_worst, cone_cell = _worst_cell(rows, "zworld_cone_min_cosine", mode="min")
    cone_present = cone_worst >= CONE_MIN_COSINE_FLOOR
    n_on_diff = sum(1 for r in on if r["crf_max_pairwise_rule_dist"] > DIST_FLOOR
                    and r["crf_max_live_rules"] >= MIN_LIVE_RULES)
    on_diff_pool = (n_on_diff / len(on)) >= SEED_PASS_FRACTION
    n_off_pinned = sum(1 for r in off if r["crf_max_live_rules"] <= OFF_MAX_LIVE_CEIL)
    off_pinned_pool = (n_off_pinned / len(off)) >= SEED_PASS_FRACTION
    n_on_active = sum(1 for r in on if r["crf_frac_active_p2"] >= FRAC_ACTIVE_FLOOR)
    on_active = (n_on_active / len(on)) >= SEED_PASS_FRACTION

    # --- READINESS (d): prop-sample sufficiency, EVERY cell (worst-cell). ---
    n_prop_worst, n_prop_cell = _worst_cell(rows, "n_prop_samples", mode="min")
    prop_samples_sufficient = n_prop_worst >= PROP_SAMPLE_FLOOR

    # --- READINESS (e): the fix engaged -- cross-candidate spread survives centering. ---
    spread_worst, spread_cell = _worst_cell(rows, "summary_spread_ratio_p2_min", mode="min")
    spread_ok = spread_worst >= SUMMARY_SPREAD_RATIO_FLOOR
    n_degen_worst, degen_cell = _worst_cell(rows, "n_summary_degenerate_ticks", mode="max")
    no_degenerate_tick = n_degen_worst <= 0.0
    n_fallback_worst, fallback_cell = _worst_cell(rows, "n_summary_fallback", mode="max")
    dispatch_engaged = n_fallback_worst <= 0.0
    fix_engaged = spread_ok and no_degenerate_tick and dispatch_engaged

    # --- READINESS (f): capture positive control (worst cell). ---
    init_flags = [r["head_diag_by_phase"]["init"] for r in rows]
    def _ctrl_ok(f):
        return bool(f["capture_head_diagnostics"] and f["dead_relu_frac_finite_in_range"]
                    and f["magnitude_ratio_finite"])
    capture_worst = min(1.0 if _ctrl_ok(f) else 0.0 for f in init_flags)
    capture_active = capture_worst >= 1.0
    capture_offending = next(
        (f"{r['arm']}/seed{r['seed']}" for r, f in zip(rows, init_flags) if not _ctrl_ok(f)),
        "none")
    n_diag_worst, n_diag_cell = _worst_cell(rows, "n_head_diag_samples", mode="min")
    head_diag_sufficient = n_diag_worst >= HEAD_DIAG_SAMPLE_FLOOR

    ready = (cone_present and on_diff_pool and off_pinned_pool and on_active
             and prop_samples_sufficient and fix_engaged and capture_active
             and head_diag_sufficient)

    # --- PRE-REGISTERED ACCEPTANCE (SD-082's own). ---
    on_prop = statistics.fmean(r["prop_delta_mean"] for r in on)
    off_prop = statistics.fmean(r["prop_delta_mean"] for r in off)
    c1 = on_prop >= PROP_NONVAC_FLOOR

    off_by_seed = {r["seed"]: r for r in off}
    per_seed_contrast: List[Dict[str, Any]] = []
    for r in sorted(on, key=lambda x: x["seed"]):
        o = off_by_seed.get(r["seed"])
        if o is None:
            continue
        d = r["prop_delta_mean"] - o["prop_delta_mean"]
        per_seed_contrast.append({
            "seed": r["seed"], "on_prop_delta_mean": r["prop_delta_mean"],
            "off_prop_delta_mean": o["prop_delta_mean"], "on_minus_off": d,
            "passes_margin": bool(d > CONTRAST_MARGIN),
        })
    n_contrast_seeds = sum(1 for c in per_seed_contrast if c["passes_margin"])
    c2 = (n_contrast_seeds / len(per_seed_contrast)) >= SEED_PASS_FRACTION \
        if per_seed_contrast else False

    # Supplementary standardized effect size -- DIAGNOSTIC, never gating. Recorded
    # so a later autopsy has the effect magnitude without this run inventing a
    # second acceptance gate on top of the claim's own pre-registered one.
    deltas = [c["on_minus_off"] for c in per_seed_contrast]
    contrast_mean = statistics.fmean(deltas) if deltas else 0.0
    contrast_sd = statistics.stdev(deltas) if len(deltas) >= 2 else 0.0
    contrast_t_like = (contrast_mean / (contrast_sd / math.sqrt(len(deltas)))
                       if contrast_sd > 0.0 and deltas else 0.0)

    overall = bool(ready and c1 and c2)

    # --- Non-degeneracy (applies to evidence runs -- the V3-EXQ-514m net). ---
    c1_non_degenerate = bool(fix_engaged and prop_samples_sufficient)
    c2_non_degenerate = bool(
        c1_non_degenerate and len(per_seed_contrast) == len(SEEDS)
        and len(set(round(d, 12) for d in deltas)) > 1)

    # --- Diagnostic flags (informational; not gating, all components recorded). ---
    on_dead_relu = statistics.fmean(r["hidden_dead_relu_frac_p2_mean"] for r in on)
    off_dead_relu = statistics.fmean(r["hidden_dead_relu_frac_p2_mean"] for r in off)
    on_last_layer_delta = statistics.fmean(
        r["last_layer_weight_delta_init_to_p1"] for r in on)
    live_ratios = [r["rule_summary_magnitude_ratio_p2_median"] for r in on
                   if r["n_magnitude_ratio_live_samples"] > 0]
    magnitude_in_band = all(MAGNITUDE_RATIO_LOW <= x <= MAGNITUDE_RATIO_HIGH
                            for x in live_ratios) if live_ratios else False
    on_flip = statistics.fmean(r["rule_flip_frac"] for r in on)
    off_flip = statistics.fmean(r["rule_flip_frac"] for r in off)

    diagnostic_flags = {
        "candidate_summary_fix_engaged": fix_engaged,
        "magnitude_ratio_in_band": magnitude_in_band,
        "rule_flip_observed": bool(on_flip > 0.0 or off_flip > 0.0),
        "dead_relu_confirmed": bool(on_dead_relu >= DEAD_RELU_HIGH_FLOOR),
        "dead_relu_partial_contributor": bool(
            DEAD_RELU_MED_FLOOR <= on_dead_relu < DEAD_RELU_HIGH_FLOOR),
        "head_untrained_last_layer_static": bool(
            abs(on_last_layer_delta) < LAST_LAYER_WEIGHT_DELTA_FLOOR),
    }

    # Label ladder ordered by CAUSAL DEPTH, not by the order the hypotheses were
    # written -- autopsy Section 5's third lesson (822c's own ladder suppressed its
    # best finding by testing a proxy measure before a direct one).
    if not ready:
        label = "substrate_not_ready_requeue"
    elif not c1:
        label = "propagation_still_vacuous_after_summary_fix"
    elif not c2:
        label = "propagation_non_vacuous_without_rule_state_contrast"
    else:
        label = "sd082_propagation_with_rule_contrast_confirmed"

    # Per-claim directions. SD-082 owns the "does the consumer couple rule_state to
    # a candidate-discriminating action bias at all" half (C1); SD-078 owns the
    # "does DIFFERENTIATION produce a contrast" half (C2).
    if not ready:
        dir_082 = dir_078 = "unknown"
        overall_dir = "unknown"
    else:
        dir_082 = "supports" if c1 else "weakens"
        dir_078 = "supports" if c2 else "weakens"
        overall_dir = ("supports" if (c1 and c2)
                       else ("weakens" if not (c1 or c2) else "mixed"))

    metrics = {
        "on_prop_delta_mean": on_prop,
        "off_prop_delta_mean": off_prop,
        "on_rule_state_diff_mean": statistics.fmean(r["rule_state_diff"] for r in on),
        "off_rule_state_diff_mean": statistics.fmean(r["rule_state_diff"] for r in off),
        "n_contrast_seeds_passing": n_contrast_seeds,
        "n_contrast_seeds_total": len(per_seed_contrast),
        "contrast_mean_on_minus_off": contrast_mean,
        "contrast_sd_on_minus_off": contrast_sd,
        "contrast_t_like_diagnostic_only": contrast_t_like,
        "per_seed_contrast": per_seed_contrast,
        "readiness_cone_present": cone_present,
        "readiness_on_diff_pool": on_diff_pool,
        "readiness_off_pinned_pool": off_pinned_pool,
        "readiness_on_active": on_active,
        "readiness_prop_samples_sufficient": prop_samples_sufficient,
        "readiness_fix_engaged": fix_engaged,
        "readiness_spread_ok": spread_ok,
        "readiness_no_degenerate_tick": no_degenerate_tick,
        "readiness_dispatch_engaged": dispatch_engaged,
        "readiness_capture_active": capture_active,
        "readiness_head_diag_sufficient": head_diag_sufficient,
        "zworld_cone_min_cosine_worst": cone_worst,
        "zworld_cone_worst_cell": cone_cell,
        "n_prop_samples_worst": n_prop_worst,
        "n_prop_samples_worst_cell": n_prop_cell,
        "summary_spread_ratio_worst": spread_worst,
        "summary_spread_ratio_worst_cell": spread_cell,
        "n_summary_fallback_worst": n_fallback_worst,
        "n_summary_degenerate_ticks_worst": n_degen_worst,
        "on_rule_state_norm_p2_median_values": [
            r["rule_state_norm_p2_median"] for r in on],
        "on_rule_summary_magnitude_ratio_p2_median_values": live_ratios,
        "on_hidden_dead_relu_frac_p2_mean": on_dead_relu,
        "off_hidden_dead_relu_frac_p2_mean": off_dead_relu,
        "on_last_layer_weight_delta_init_to_p1_mean": on_last_layer_delta,
        "on_rule_flip_frac_mean": on_flip,
        "off_rule_flip_frac_mean": off_flip,
        "c1_pass": c1,
        "c2_pass": c2,
        "diagnostic_flags": diagnostic_flags,
    }

    result: Dict[str, Any] = {
        "outcome": "PASS" if overall else "FAIL",
        "evidence_direction": overall_dir,
        "evidence_direction_per_claim": {"SD-078": dir_078, "SD-082": dir_082},
        "metrics": metrics,
        "per_seed_rows": rows,
        "arm_results": rows,
        "interpretation": {
            "label": label,
            "preconditions": [
                {"name": "zworld_common_mode_cone_present",
                 "description": "min pairwise z_world cosine clears the cone floor (== 822a/822c).",
                 "measured": cone_worst, "threshold": CONE_MIN_COSINE_FLOOR,
                 "direction": "lower", "offending_cell": cone_cell, "met": cone_present},
                {"name": "on_pool_differentiated",
                 "description": "ARM_ON differentiated on >= majority of seeds (== 822a/822c).",
                 "measured": float(n_on_diff),
                 "threshold": float(math.ceil(SEED_PASS_FRACTION * len(on))),
                 "direction": "lower", "met": on_diff_pool},
                {"name": "off_pool_pinned",
                 "description": "ARM_OFF pool pinned on >= majority of seeds (== 822a/822c).",
                 "measured": float(n_off_pinned),
                 "threshold": float(math.ceil(SEED_PASS_FRACTION * len(off))),
                 "direction": "lower", "met": off_pinned_pool},
                {"name": "on_rule_active_p2",
                 "description": ("count of ARM_ON seeds whose P2 rule-active fraction "
                                 "clears FRAC_ACTIVE_FLOOR, over a majority of seeds "
                                 "(== 822a/822c in substance). `measured` is the SEED "
                                 "COUNT, not the mean fraction: `met` quantifies over "
                                 "seeds, and a mean clearing the floor can coexist with "
                                 "a minority of seeds clearing it, which the indexer's "
                                 "recompute would silently pass. Per-seed fractions are "
                                 "in per_seed_rows[].crf_frac_active_p2."),
                 "measured": float(n_on_active),
                 "threshold": float(math.ceil(SEED_PASS_FRACTION * len(on))),
                 "direction": "lower", "met": on_active},
                {"name": "prop_samples_sufficient",
                 "description": ("worst-cell n_prop_samples clears the floor -- the gate "
                                 "822/822a/822b lacked, where n_prop_samples was 0 in all "
                                 "18 cells and fmean's empty-list default of 0.0 was read "
                                 "as a measured structural zero for a month."),
                 "measured": n_prop_worst, "threshold": float(PROP_SAMPLE_FLOOR),
                 "direction": "lower", "offending_cell": n_prop_cell,
                 "met": prop_samples_sufficient},
                {"name": "candidate_summary_spread_survives_centering",
                 "description": ("worst-cell MINIMUM over P2 ticks of "
                                 "post_centering_norm / pre_centering_norm -- the "
                                 "cross-candidate SPREAD the load-bearing prop_delta "
                                 "criteria depend on, not a magnitude. Legacy ws[0,0,:] "
                                 "measures ~2.2e-9 here; proposer_post_action ~1.8e-2."),
                 "measured": spread_worst, "threshold": SUMMARY_SPREAD_RATIO_FLOOR,
                 "direction": "lower", "offending_cell": spread_cell,
                 "control": ("Step 2.5a probe, 40 real ticks at seed 101: legacy "
                             "post-centering norm 5.66e-08 vs proposer_post_action "
                             "4.01e-01 on the same candidates"),
                 "met": spread_ok},
                {"name": "no_degenerate_centering_tick",
                 "description": ("the substrate's own candidate_summary_degenerate guard "
                                 "never fired on any P2 tick in any cell."),
                 "measured": n_degen_worst, "threshold": 0.0, "direction": "upper",
                 "offending_cell": degen_cell, "met": no_degenerate_tick},
                {"name": "summary_dispatch_engaged",
                 "description": ("_candidate_world_summaries() returned non-None on every "
                                 "read; the manual ws[0,0,:] fallback (the 822c path) was "
                                 "taken zero times."),
                 "measured": n_fallback_worst, "threshold": 0.0, "direction": "upper",
                 "offending_cell": fallback_cell, "met": dispatch_engaged},
                {"name": "head_diagnostics_capture_active",
                 "description": ("capture_head_diagnostics echoes True and the synthetic "
                                 "init positive control returns finite, in-range values "
                                 "in every cell (== 822c)."),
                 "measured": capture_worst, "threshold": 1.0, "direction": "lower",
                 "control": ("synthetic differentiated 3-candidate forward pass "
                             "immediately post-construction, pre-training"),
                 "offending_cell": capture_offending, "met": capture_active},
                {"name": "head_diag_samples_sufficient",
                 "description": "worst-cell count of P2 head-diagnostic reads clears the floor (== 822c).",
                 "measured": n_diag_worst, "threshold": float(HEAD_DIAG_SAMPLE_FLOOR),
                 "direction": "lower", "offending_cell": n_diag_cell,
                 "met": head_diag_sufficient},
            ],
            "criteria": [
                {"name": "C1_propagation_non_vacuous", "load_bearing": True, "passed": c1},
                {"name": "C2_on_over_off_contrast_majority_seeds",
                 "load_bearing": True, "passed": c2},
            ],
            "combination_rule": (
                "overall PASS = readiness AND C1 AND C2 (plain conjunction of both "
                "load-bearing criteria; C1 is SD-082's absolute propagation floor, C2 is "
                "SD-078's ON-over-OFF contrast on a majority of seeds). Not an OR and "
                "not a weighted rule."),
            "criteria_non_degenerate": {
                "C1_propagation_non_vacuous": c1_non_degenerate,
                "C2_on_over_off_contrast_majority_seeds": c2_non_degenerate,
            },
            "claim_criterion_map": {
                "SD-082": {
                    "criteria": ["C1_propagation_non_vacuous",
                                 "C2_on_over_off_contrast_majority_seeds"],
                    "role": "primary",
                    "note": ("SD-082's own pre-registered acceptance is BOTH halves: "
                             "on_prop_delta_mean >= 1e-3 WITH an ON>OFF contrast. The "
                             "run's evidence_direction_per_claim entry for SD-082 is "
                             "driven by C1 (does the consumer couple rule_state to the "
                             "bias at all); C2 is reported against it as well."),
                },
                "SD-078": {
                    "criteria": ["C2_on_over_off_contrast_majority_seeds"],
                    "role": "secondary_mediated",
                    "note": ("SD-078's own differentiation is verified here as a "
                             "READINESS PRECONDITION (on_pool_differentiated / "
                             "off_pool_pinned / rule_state_diff), not as a DV. Its "
                             "evidence_direction is driven only by C2 -- whether that "
                             "differentiation makes a downstream difference, mediated "
                             "entirely by SD-082's consumer. A later autopsy must NOT "
                             "read a blanket verdict on this run as a direct test of "
                             "SD-078: if SD-082's consumer is the thing that fails, "
                             "SD-078 is a co-tag, not the subject."),
                },
            },
            "diagnostic_flags": diagnostic_flags,
        },
    }

    reasons: List[str] = []
    if not cone_present:
        reasons.append("z_world common-mode cone absent")
    if not on_diff_pool:
        reasons.append("ON pool not differentiated")
    if not off_pinned_pool:
        reasons.append("OFF pool not pinned")
    if not on_active:
        reasons.append("ON rule not active enough in P2")
    if not prop_samples_sufficient:
        reasons.append(f"prop samples starved (worst cell {n_prop_cell}: {n_prop_worst:g})")
    if not spread_ok:
        reasons.append(f"candidate-summary spread annihilated by centering "
                       f"(worst cell {spread_cell}: {spread_worst:g})")
    if not no_degenerate_tick:
        reasons.append(f"substrate degeneracy guard fired ({degen_cell})")
    if not dispatch_engaged:
        reasons.append(f"proposer_post_action dispatch did not engage; manual "
                       f"ws[0,0,:] fallback taken ({fallback_cell})")
    if not capture_active:
        reasons.append("head-diagnostics capture positive control failed")
    if not head_diag_sufficient:
        reasons.append("P2 head-diagnostic samples starved")
    if reasons or not (c1_non_degenerate and c2_non_degenerate):
        if not c1_non_degenerate and "prop samples starved" not in " ".join(reasons):
            reasons.append("C1 degenerate: prop_delta did not measure a "
                           "candidate-discriminating quantity")
        if not c2_non_degenerate:
            reasons.append("C2 degenerate: per-seed ON-OFF contrast has no variation "
                           "across seeds, or a seed is missing an arm")
        result["non_degenerate"] = False
        result["degeneracy_reason"] = "; ".join(reasons)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    t0 = time.perf_counter()
    global SEEDS, P0_WARMUP_EPISODES, P1_BIAS_TRAIN_EPISODES, P2_MEASUREMENT_EPISODES
    if args.dry_run:
        SEEDS = [101]
        P0_WARMUP_EPISODES = 3
        P1_BIAS_TRAIN_EPISODES = 3
        P2_MEASUREMENT_EPISODES = 3

    result = run_experiment()
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    full_config = {
        "seeds": SEEDS, "arms": ARMS,
        "p0": P0_WARMUP_EPISODES, "p1": P1_BIAS_TRAIN_EPISODES,
        "p2": P2_MEASUREMENT_EPISODES, "steps_per_episode": STEPS_PER_EPISODE,
        "candidate_summary_source": "proposer_post_action",
        "cone_min_cosine_floor": CONE_MIN_COSINE_FLOOR,
        "dist_floor": DIST_FLOOR, "min_live_rules": MIN_LIVE_RULES,
        "off_max_live_ceil": OFF_MAX_LIVE_CEIL, "frac_active_floor": FRAC_ACTIVE_FLOOR,
        "seed_pass_fraction": SEED_PASS_FRACTION,
        "prop_nonvac_floor": PROP_NONVAC_FLOOR,
        "contrast_margin": CONTRAST_MARGIN,
        "prop_sample_floor": PROP_SAMPLE_FLOOR,
        "summary_spread_ratio_floor": SUMMARY_SPREAD_RATIO_FLOOR,
        "head_diag_sample_floor": HEAD_DIAG_SAMPLE_FLOOR,
        "magnitude_ratio_low": MAGNITUDE_RATIO_LOW,
        "magnitude_ratio_high": MAGNITUDE_RATIO_HIGH,
        "dead_relu_high_floor": DEAD_RELU_HIGH_FLOOR,
        "dead_relu_med_floor": DEAD_RELU_MED_FLOOR,
        "last_layer_weight_delta_floor": LAST_LAYER_WEIGHT_DELTA_FLOOR,
        "lr_lpfc_bias": LR_LPFC_BIAS,
        "reinforce_batch_size": REINFORCE_BATCH_SIZE,
        "adv_min_threshold": ADV_MIN_THRESHOLD,
        "arm_config_slice_off": _config_slice(False),
        "arm_config_slice_on": _config_slice(True),
        "validates_substrate": ("SD-082 per-candidate-summary amend, ree-v3 ef88faa "
                                "(2026-08-30); routed by "
                                "failure_autopsy_V3-EXQ-822c_2026-08-29 Section 6"),
        "supersedes_note": ("does NOT supersede v3_exq_822c -- 822c is the diagnostic "
                            "that measured the defect and its failure_record is marked "
                            "resolved (not superseded) by the substrate fix this run "
                            "validates"),
    }
    manifest = {
        "run_id": f"{EXPERIMENT_TYPE}_{ts}_v3",
        "experiment_type": EXPERIMENT_TYPE,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "evidence_direction": result["evidence_direction"],
        "evidence_direction_per_claim": result["evidence_direction_per_claim"],
        "outcome": result["outcome"],
        "timestamp_utc": ts,
        "metrics": result["metrics"],
        "per_seed_rows": result["per_seed_rows"],
        "arm_results": result["arm_results"],
        "interpretation": result["interpretation"],
    }
    if "non_degenerate" in result:
        manifest["non_degenerate"] = result["non_degenerate"]
        manifest["degeneracy_reason"] = result["degeneracy_reason"]

    out_path = write_flat_manifest(
        manifest,
        dry_run=args.dry_run,
        config=full_config,
        seeds=SEEDS,
        script_path=Path(__file__),
        started_at=t0,
        agent=list(_ARM_AGENTS.values()) or None,
        z_goal_stream_stats=_ZG.stats(),
    )
    m = result["metrics"]
    print(f"outcome: {result['outcome']}", flush=True)
    print(f"label: {result['interpretation']['label']}", flush=True)
    print(f"ON prop={m['on_prop_delta_mean']:.6f} OFF prop={m['off_prop_delta_mean']:.6f} "
          f"C1={m['c1_pass']} C2={m['c2_pass']} "
          f"({m['n_contrast_seeds_passing']}/{m['n_contrast_seeds_total']} seeds)",
          flush=True)
    print(f"fix_engaged={m['readiness_fix_engaged']} "
          f"spread_worst={m['summary_spread_ratio_worst']:.3g} "
          f"fallbacks={m['n_summary_fallback_worst']:g} "
          f"degen_ticks={m['n_summary_degenerate_ticks_worst']:g} "
          f"n_prop_worst={m['n_prop_samples_worst']:g}", flush=True)
    print(f"ready(cone={m['readiness_cone_present']} diffpool={m['readiness_on_diff_pool']} "
          f"pinned={m['readiness_off_pinned_pool']} active={m['readiness_on_active']} "
          f"propsamp={m['readiness_prop_samples_sufficient']} "
          f"capture={m['readiness_capture_active']} "
          f"headdiag={m['readiness_head_diag_sufficient']})", flush=True)
    print(f"flip: ON={m['on_rule_flip_frac_mean']:.4f} OFF={m['off_rule_flip_frac_mean']:.4f} "
          f"(822c measured 0.0000 in all 6 cells -- broadcast-constant symmetry)", flush=True)
    print(f"diagnostic_flags: {m['diagnostic_flags']}", flush=True)
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
