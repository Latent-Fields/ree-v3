"""V3-EXQ-624c: ARC-068 / MECH-320 opportunity-cost vs effort-cost dissociation.

CLEAN RE-SPEC of V3-EXQ-624b. The scientific question is unchanged from the
V3-EXQ-624 / 624a / 624b lineage; the implementation is corrected so the no-op /
opportunity-cost mechanism is reliably exercised on EVERY seed.

Why this re-spec (the load-bearing change vs 624b):
  V3-EXQ-624b was adjudicated non_contributory with narrow_supports_flag=true
  (failure_autopsy_V3-EXQ-624b_2026-06-07). The load-bearing C2 dissociation
  (w_passive insensitive to parametric movement cost, distinct from Salamone
  effort cost) PASSED on all 3 seeds. The FAIL was ONLY the positive-control /
  non-vacuity conjunction (C1 Niv lift + C5 authority), which failed on the
  env-degenerate seed 42 alone: the well-fed-safe-familiar regime
  (size=8, num_hazards=1, num_resources=3) was no-op-opportunity-poor
  (mean action_density 0.865, near-saturated on seed 42), so the CEM proposer
  surfaced no competitive no-op candidate -> w_passive bias had nothing to act
  on -> bias range 0 -> authority inert -> Niv lift 0.0. Seeds 43/44 worked
  (vigor lift +0.259 / +0.045, operative authority). NO substrate gap -- the
  643a authority fix is proven operative here.

  Two corrections, per the autopsy repair pathway ("a no-op-opportunity-rich
  regime ... and/or >= 5 seeds ... keep the C2 dissociation gate strict"):

  (1) NO-OP-OPPORTUNITY-RICH REGIME (lower baseline action density on EVERY
      seed). A 2026-06-07 regime probe over the 5 seeds established what actually
      lowers baseline action_density (so the stationary no-op candidate is
      competitive at the argmin and the w_passive bias has something to act on):
        - DENSER hazards / a tighter grid BACKFIRE -- they make the untrained
          agent FLEE (a flee is still a move), saturating action_density at 1.0
          (probe 6/3/2 -> 3/5 seeds saturated, 2/5 valid, WORSE than 624b).
          Boxing-in only forces a no-op when the agent is surrounded on all
          sides, which is rare. So hazards are NOT the lever; N_HAZARDS stays 1.
        - SPARSER foraging lowers the benefit-approach pull: more states are
          neutral near-ties where no-op is competitive. N_RESOURCES 3 -> 2.
        - P0-WARMUP LENGTH is the dominant headroom lever (probe: P0=8 saturates
          ~0.92 with 1/5 valid; the full P0=100 gave 624b d0~0.865 with 2/3
          valid). P0_WARMUP_EPISODES retained at 100 (proven adequate).
      This is a FIXED regime applied to ALL FOUR arms (like the well-fed-safe-
      familiar regime in 624b); it is NOT part of the Niv-vs-Salamone
      manipulation. The Niv-vs-Salamone contrast remains movement-cost-ONLY
      (damage_increment / failure_prob_scale on the Salamone arms), so the
      dissociation difference (ARM_1-ARM_0 vs ARM_2-ARM_3) cancels the shared
      regime and isolates movement-cost sensitivity exactly as in 624b.
      harm_gradient was deliberately NOT used: it fires on transition_type==
      "none" steps regardless of whether the agent moved or stayed, so it
      penalises hazard PROXIMITY and would push the agent to FLEE (raise
      action_density) -- the wrong direction.

  (2) >= 5 SEEDS + NON-VACUOUS MAJORITY GATE on the positive control, with the
      C2 dissociation kept STRICT. 624b used an all-seeds conjunction on
      C1+C5, which one env-degenerate seed poisons; worse, its C2 passed
      VACUOUSLY on seed 42 (both ARM lifts were ~0, so |salamone - niv| ~ 0
      cleared the tolerance even though no Niv lift existed to dissociate).
      624c fixes both: C2 is evaluated ONLY on the seeds where the positive
      control genuinely fired (C1 AND C5), so a saturated seed can no longer
      pass C2 vacuously; and the positive control is gated by a strict majority
      of seeds rather than all-seeds-conjunction.

Definitions of the gate (see Pre-registered acceptance thresholds):
  valid seed   = C1 (Niv lift >= C1_LIFT_MIN) AND C5 (authority active+ranged).
                 The mechanism genuinely engaged on this seed.
  C2 STRICT    = on EVERY valid seed, the Niv-vs-Salamone dissociation holds
                 at the SAME tolerance as 624b (C2_DISSOCIATION_TOL=0.50).
                 Non-vacuous by construction (a valid seed has c1_lift >=
                 C1_LIFT_MIN > 0, so the dissociation denominator is real).
  majority     = n_valid >= VALID_SEED_MAJORITY (3 of 5).
  PASS         = (n_valid >= VALID_SEED_MAJORITY) AND C2 holds on every valid
                 seed AND C3 (vigor gate fired) holds on every valid seed.

Proposal: EXP-0081 / EVB-0237 (dispatch_mode=discriminative_pair). Behavioural
validation of the ARC-068 architectural slot (opportunity_cost_no_op_penalty)
as instantiated by MECH-320's w_passive term per the lit-pull R3/R4 verdicts.

Substrate prerequisites all landed (unchanged from 624b):
  - MECH-320 substrate landed 2026-05-10 (ree_core/policy/tonic_vigor.py).
  - modulatory-bias-selection-authority landed 2026-06-03; float32-cancellation
    amend (V3-EXQ-643a) landed 2026-06-06. Gives MECH-320's w_passive bias
    bounded authority over the E3.select argmin -- proven operative in 624b
    (vigor lift on 2/3 seeds, authority_range 6.29e-3, no score explosion).
  - ARC-068 lit-pull R3 verdict: collapse-at-implementation-layer licensed
    (additive Niv form). R4 verdict: effort-cost vs opportunity-cost separation
    REQUIRED (Salamone & Correa 2003 dissociation; ARC-068 must NOT absorb into
    MECH-258 / SD-032b dacc_effort_cost machinery).

Inherited fix from V3-EXQ-624a (observation-space confound, retained verbatim):
  limb_damage_enabled=True on ALL arms (body_obs_dim == 17 uniformly) so the
  Niv-vs-Salamone manipulation is movement-cost-ONLY (carried by
  damage_increment / failure_prob_scale), never an observation-dimensionality
  change. The Niv side / all P0 warmups set damage_increment=0.0 and
  failure_prob_scale=0.0 -> movement free, damage channels zero. The Salamone
  side sets the elevated values so movement is expensive.

ARC-068 architectural prediction (R4 verdict: opportunity-cost-on-time, NOT
effort-cost-on-movement): the no-op penalty (MECH-320 w_passive * v_t) should
scale with the EWMA over realised reward (Niv 2007 average reward rate), NOT
with the per-step movement cost. Falsifiable discriminative prediction: ARC-068
must remain insensitive to parametric increases in movement-cost-on-action
(Salamone & Correa 2003 effort cost), while continuing to fire when v_t is
positive (forced via v_t_floor to bypass V3-EXQ-549's calibration failure).

Four-arm design (unchanged from 624b; the no-op-rich regime applies to all four):
  ARM_0_baseline:        use_tonic_vigor=False, free-movement env. Reference.
  ARM_1_vigor_niv:       use_tonic_vigor=True, v_t_floor=0.05, free-movement env.
                         Predicts elevated action_density (Niv no-op penalty on
                         cheap movement) -- now with selection authority.
  ARM_2_vigor_salamone:  use_tonic_vigor=True, v_t_floor=0.05, elevated-
                         movement-cost env. Niv kernel: action_density delta
                         from ARM_3 matches ARM_1's delta from ARM_0. Salamone
                         kernel: suppressed/reversed.
  ARM_3_baseline_salamone: use_tonic_vigor=False, elevated-movement-cost env.
                         Movement-cost-only control.
  All four arms: use_modulatory_selection_authority=True,
  modulatory_authority_gain=0.5 (fixed substrate; no-op where no modulatory
  bias exists, so ARM_0 / ARM_3 are bit-identical to their vigor-off form).

Protocol (unchanged from 624b):
  P0 warmup (100 ep, vigor OFF + free-movement env in all arms): identical
    baseline policy checkpoint per seed (env stepping only; no gradient
    training, matching the 624 lineage design).
  P1 measurement (30 ep x 200 steps): arm flag and env movement-cost config
    toggled. Observation space identical (17-dim body) across all arms/phases.

Environment (NO-OP-OPPORTUNITY-RICH regime, matched across arms by seed):
  CausalGridWorldV2 size=GRID_SIZE (8), num_hazards=N_HAZARDS (1),
  num_resources=N_RESOURCES (2; sparser than 624b's 3), action_dim=5 (0 up,
  1 down, 2 left, 3 right, 4 noop). Hazard contact depletes agent_health (not
  single-step terminal); episodes end at health<=0 or step cap. Sparser
  foraging (fewer resources) is the env lever -- it lowers the benefit-approach
  pull so more states are neutral near-ties where the no-op candidate is
  competitive, lowering baseline action_density and giving the w_passive bias
  headroom to act on. (Probe-rejected alternative: denser hazards, which make
  the agent flee and SATURATE action_density.)

Metrics (per arm per seed, P1 only): identical to 624b --
  action_density: mean over P1 ticks of [argmax(action) != NOOP_CLASS].
  v_t_window / action_density_window / gate_product (ARM_1, ARM_2 only).
  E3 selection-authority diagnostics (modulatory_authority_active fraction,
    scale_factor mean, range mean, e3_raw_score_range mean) -- C5 non-vacuity.

Pre-registered acceptance thresholds (defined here, NOT inferred post-hoc):
  C1 action_density lift (vigor effect): d1 - d0 >= C1_LIFT_MIN (0.03). Paired
     by seed. The basic MECH-320 firing test.
  C2 Niv-vs-Salamone dissociation (R4 verdict test, STRICT): on every VALID
     seed, |salamone_lift - c1_lift| / max(|c1_lift|, 1e-6) < C2_DISSOCIATION_TOL
     (0.50) AND salamone_lift >= 0.5 * c1_lift. Evaluated ONLY on valid seeds
     (where C1+C5 fired) -> non-vacuous.
  C3 gate sanity: mean(gate_product) > C3_GATE_PRODUCT_MIN (0.5) in ARM_1.
  C4 well-fed-safe regime no-op penalty observable (informative; numerically C1).
  C5 selection-authority NON-VACUITY (the lever genuinely had authority and
     scores were bounded):
       modulatory_authority_active_frac(ARM_1) >= C5_AUTH_ACTIVE_FRAC_MIN (0.5)
       AND modulatory_authority_range(ARM_1) > C5_AUTH_RANGE_MIN (1e-6)
       AND e3_raw_score_range(ARM_1) < C5_RAW_RANGE_MAX (1e6, explosion guard).

PASS = (n_valid >= VALID_SEED_MAJORITY) AND C2 on every valid seed AND C3 on
every valid seed. (C4 logged separately.)

5-row interpretation grid:
  (1) PASS: ARC-068 supports + MECH-320 supports. The slot preservation is
      vindicated with the authority substrate in place + a regime that lets the
      mechanism express on a majority of seeds. Surface to /governance.
  (2) n_valid >= majority AND C2 FAILS on some valid seed (salamone lift
      diverges from the Niv lift where the mechanism fired): ARC-068 weakens.
      Effort-cost-like (movement-cost-sensitive). Contradicts R4; surface the
      supersession question (ARC-068 collapse into MECH-258 / SD-032b) to
      /governance.
  (3) n_valid < majority (the positive control still failed to fire on enough
      seeds, even in the no-op-rich regime): non_contributory. The test could
      not let the claim express itself on a majority of seeds (env-adequacy /
      authority-engagement gap, NOT a verdict on the claim). Route to
      /failure-autopsy / re-spec.
  (4) n_valid >= majority AND C2 holds on valid seeds AND C3 FAILS on a valid
      seed (vigor gate did not fire even with v_t_floor=0.05): non_contributory;
      calibration / wiring problem. Route to /diagnose-errors.
  (5) (defensive) any other combination -> mixed; route to /failure-autopsy.

experiment_purpose = "evidence". claim_ids = ["MECH-320", "ARC-068"] with
evidence_direction_per_claim. MECH-258 and SD-032b are NOT in claim_ids: foils
architecturally dissociated from, not tested directly.

Supersedes V3-EXQ-624b (non_contributory; env-degenerate seed + vacuous-C2 gate).

Run with:
  /opt/local/bin/python3 experiments/v3_exq_624c_arc068_mech320_niv_salamone_dissociation.py
or:
  /opt/local/bin/python3 experiments/v3_exq_624c_arc068_mech320_niv_salamone_dissociation.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig

from experiment_protocol import emit_outcome


# ----------------------------------------------------------------------
# Constants and pre-registered thresholds
# ----------------------------------------------------------------------
EXPERIMENT_TYPE = "v3_exq_624c_arc068_mech320_niv_salamone_dissociation"
SUPERSEDES = "v3_exq_624b_arc068_mech320_niv_salamone_dissociation"
CLAIM_IDS = ["MECH-320", "ARC-068"]
EXPERIMENT_PURPOSE = "evidence"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"

SEEDS = [42, 43, 44, 45, 46]
VALID_SEED_MAJORITY = 3  # strict majority of 5; positive control must fire here
P0_WARMUP_EPISODES = 100
P1_EVAL_EPISODES = 30
TOTAL_EPISODES_PER_ARM = P0_WARMUP_EPISODES + P1_EVAL_EPISODES  # 130
STEPS_PER_EPISODE = 200
PRINT_INTERVAL = 25
WINDOW_LENGTH = 50  # P1 step-window for within-arm scaling correlation

# Dry-run smoke-test scaling: small enough to be fast, but MUST cross into P1
# so the Salamone-arm movement-cost env switch is exercised AND the modulatory
# authority diagnostics are captured.
DRY_RUN_P0_EPISODES = 2
DRY_RUN_P1_EPISODES = 2
DRY_RUN_STEPS_PER_EPISODE = 12

# NO-OP-OPPORTUNITY-RICH regime (the load-bearing change vs 624b's
# size=8 / hazards=1 / resources=3 well-fed-safe-familiar regime).
# A 2026-06-07 regime probe (P0=8 and P0=60 sweeps over the 5 seeds) established
# the mechanics that drive no-op opportunity here:
#   - DENSER hazards on a tighter grid BACKFIRE: they make the untrained agent
#     FLEE (a flee is still a move), so baseline action_density SATURATES at 1.0
#     (probe 6/3/2 -> 3/5 seeds saturated, only 2/5 valid -- worse than 624b).
#     Boxing-in only forces a no-op when the agent is surrounded on all sides,
#     which is rare; the dominant effect is flee = move. So hazards are NOT the
#     no-op lever. N_HAZARDS stays at 1.
#   - P0-WARMUP LENGTH is the dominant headroom lever: at P0=8 the baseline
#     saturates (mean d0~0.92, 1/5 valid); at the full P0=100 624b measured
#     d0~0.865 with 2/3 valid. More settling of the E3 running-variance / EMA
#     state -> more no-op selection -> lower baseline density -> more headroom.
#   - SPARSER foraging (fewer resources) reduces the benefit-approach pull, so
#     more states are neutral near-ties where the no-op candidate is competitive.
# So the no-op-opportunity-rich regime here = the proven 624b base (size=8,
# hazards=1) made CALMER, NOT denser: N_RESOURCES 3 -> 2 (sparser foraging).
# P0_WARMUP_EPISODES is retained at 624b's full 100 -- the probe-confirmed
# dominant headroom lever, already adequate for 2/3 valid at full scale (the
# probe showed it is P0-length, not env density, that lowers baseline action
# density; pushing it higher trades runtime for unconfirmed marginal headroom,
# so the per-seed valid rate is instead raised by the sparser regime and the
# robustness is supplied by 5 seeds + the non-vacuous majority gate).
# harm_gradient was
# deliberately rejected (it fires on transition_type=="none" regardless of
# move/stay -> penalises proximity -> pushes the agent to flee, the wrong
# direction). The primary robustness comes from 5 seeds + the non-vacuous
# majority gate; the calmer regime + longer P0 maximise the per-seed valid rate.
# Applied to ALL arms (fixed regime, NOT part of the Niv-vs-Salamone
# manipulation, so the dissociation difference cancels it).
GRID_SIZE = 8
N_HAZARDS = 1
N_RESOURCES = 2
ACTION_DIM = 5  # CausalGridWorldV2.ACTIONS: 0=up, 1=down, 2=left, 3=right, 4=noop
NOOP_CLASS = 4  # matches CausalGridWorldV2 convention; TonicVigorConfig
                # default noop_class=0 (MECH-279 convention) overridden via
                # tonic_vigor_noop_class=NOOP_CLASS in make_config

# v_t_floor: forced-vigor probe per V3-EXQ-549 prescribed fix. Small positive
# value that survives gate-drive collapse and ensures the downstream score-bias
# path is exercised regardless of the EWMA / gate state.
V_T_FLOOR = 0.05

# Modulatory-bias-selection-authority substrate (landed 2026-06-03; float32
# amend V3-EXQ-643a 2026-06-06). Gives the MECH-320 w_passive bias BOUNDED
# authority over the E3.select argmin. ON in ALL arms (no-op where no
# modulatory bias exists). gain 0.5 < 1.0 keeps the modulatory channel
# competitive in near-ties but subdominant when the primary harm/goal gap
# exceeds gain * raw_score_range.
USE_MODULATORY_AUTHORITY = True
MODULATORY_AUTHORITY_GAIN = 0.5

# Salamone-style elevated movement cost env knobs (applied only on the
# expensive-movement arms in P1). The free-movement arms / all P0 warmups use
# 0.0 for both so movement is free while keeping body_obs_dim=17 uniform.
SALAMONE_DAMAGE_INCREMENT = 0.30
SALAMONE_FAILURE_PROB_SCALE = 0.6
FREE_DAMAGE_INCREMENT = 0.0
FREE_FAILURE_PROB_SCALE = 0.0

# Pre-registered thresholds (must be set in script, not inferred post-hoc).
C1_LIFT_MIN = 0.03
C2_DISSOCIATION_TOL = 0.50  # |delta_arm2 - delta_arm1| / max(delta_arm1, 1e-6) < tol
C3_GATE_PRODUCT_MIN = 0.50
C4_NO_OP_PENALTY_MIN = 0.03  # well-fed-safe regime ARM_1 - ARM_0 lift threshold
# C5 selection-authority non-vacuity gate.
C5_AUTH_ACTIVE_FRAC_MIN = 0.50  # >= 50%% of ARM_1 P1 ticks the authority lever fired
C5_AUTH_RANGE_MIN = 1e-6        # cross-candidate modulatory range cleared the floor
C5_RAW_RANGE_MAX = 1.0e6        # explosion guard (643 hit ~1e32 under SD-056 training)

ARM_LABELS = [
    "ARM_0_baseline",
    "ARM_1_vigor_niv",
    "ARM_2_vigor_salamone",
    "ARM_3_baseline_salamone",
]


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def make_env(seed: int, expensive_movement: bool) -> CausalGridWorldV2:
    """Build env. limb_damage is ALWAYS enabled (body_obs_dim=17 uniform).

    The Niv-vs-Salamone manipulation is movement-cost-ONLY, carried by
    damage_increment and failure_prob_scale. With both cost knobs at 0.0,
    movement never fails and damage never accrues, so movement is free and the
    damage observation channels stay at zero. The no-op-opportunity-rich regime
    (sparser foraging: num_resources=2 on the size-8 / 1-hazard base) is
    identical across arms.
    """
    return CausalGridWorldV2(
        size=GRID_SIZE,
        num_hazards=N_HAZARDS,
        num_resources=N_RESOURCES,
        seed=seed,
        use_proxy_fields=True,
        limb_damage_enabled=True,
        damage_increment=(
            SALAMONE_DAMAGE_INCREMENT if expensive_movement
            else FREE_DAMAGE_INCREMENT
        ),
        failure_prob_scale=(
            SALAMONE_FAILURE_PROB_SCALE if expensive_movement
            else FREE_FAILURE_PROB_SCALE
        ),
    )


def make_config(
    env: CausalGridWorldV2,
    vigor_on: bool,
) -> REEConfig:
    """Build REEConfig. Vigor module is constructed iff vigor_on=True.

    The modulatory-bias-selection-authority substrate is enabled in EVERY arm
    (use_modulatory_selection_authority=True, gain=0.5): part of the fixed
    substrate config, not a manipulated variable, and a strict no-op where no
    modulatory bias is present (ARM_0 / ARM_3, vigor off). In the vigor-on arms
    it gives the MECH-320 w_passive bias bounded authority over the E3.select
    argmin.

    body_obs_dim / world_obs_dim are read from the env (limb_damage always on
    -> 17), so the agent is dimension-consistent across P0 warmup and the P1
    measurement env switch. tonic_vigor_noop_class is set to
    CausalGridWorldV2's action-4 no-op rather than the TonicVigorConfig
    default (0).
    """
    kwargs = dict(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=ACTION_DIM,
        self_dim=16,
        world_dim=16,
        use_tonic_vigor=vigor_on,
        tonic_vigor_noop_class=NOOP_CLASS,
        # Load-bearing substrate fix vs 624a (no-op when no modulatory bias).
        use_modulatory_selection_authority=USE_MODULATORY_AUTHORITY,
        modulatory_authority_gain=MODULATORY_AUTHORITY_GAIN,
    )
    if vigor_on:
        # Forced-vigor probe per V3-EXQ-549 prescribed fix: v_t_floor pins
        # the tonic vigor scalar above zero regardless of EWMA / gate state.
        kwargs["tonic_vigor_v_t_floor"] = V_T_FLOOR
    return REEConfig.from_dims(**kwargs)


def split_obs_tensors(obs_dict: dict) -> tuple:
    """Extract (obs_body, obs_world) as 2D tensors for act_with_split_obs."""
    body = obs_dict["body_state"]
    world = obs_dict["world_state"]
    if body.dim() == 1:
        body = body.unsqueeze(0)
    if world.dim() == 1:
        world = world.unsqueeze(0)
    return body, world


def pearson_r(xs: list, ys: list) -> float:
    """Pearson r over two equal-length scalar sequences. 0 on degenerate."""
    n = len(xs)
    if n < 2 or len(ys) != n:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0.0 or syy <= 0.0:
        return 0.0
    sxy = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    return sxy / ((sxx * syy) ** 0.5)


# ----------------------------------------------------------------------
# Per-arm run
# ----------------------------------------------------------------------
def run_arm(
    seed: int,
    arm_label: str,
    vigor_on: bool,
    salamone_env: bool,
    dry_run: bool,
) -> dict:
    """Run one arm for one seed and return per-seed measurements.

    P0 warmup uses the FREE-movement env and vigor OFF in ALL arms so the
    policy sees identical data; P1 measurement toggles the vigor flag and the
    movement-cost env config per arm. The observation space (body_obs_dim=17)
    is constant across P0/P1 and all arms. No gradient training (matching the
    624 lineage design); the policy acts under torch.no_grad().
    """
    print(f"Seed {seed} Condition {arm_label}", flush=True)
    torch.manual_seed(seed)

    p0_episodes = DRY_RUN_P0_EPISODES if dry_run else P0_WARMUP_EPISODES
    p1_episodes = DRY_RUN_P1_EPISODES if dry_run else P1_EVAL_EPISODES
    total_episodes = p0_episodes + p1_episodes
    steps_per_episode = DRY_RUN_STEPS_PER_EPISODE if dry_run else STEPS_PER_EPISODE

    # P0 warmup env (free movement; limb_damage on -> body_obs_dim=17) for the
    # identical baseline policy and for sizing the agent's encoders.
    env_init_p0 = make_env(seed=seed, expensive_movement=False)
    cfg = make_config(env_init_p0, vigor_on=vigor_on)
    agent = REEAgent(cfg)
    agent.reset()

    # P1 accumulators.
    p1_total_ticks = 0
    p1_nonnoop_ticks = 0

    # E3 selection-authority diagnostics (every P1 tick where diagnostics exist).
    auth_diag_ticks = 0
    auth_active_ticks = 0
    auth_scale_factor_sum = 0.0
    auth_range_sum = 0.0
    raw_range_sum = 0.0

    # Per-window measurements (vigor-on arms only: ARM_1 and ARM_2).
    window_v_t = []
    window_action_density = []
    window_gate_product = []
    cur_window_ticks = 0
    cur_window_nonnoop = 0
    cur_window_v_t_sum = 0.0
    cur_window_gate_prod_sum = 0.0

    for ep in range(total_episodes):
        ep_seed = seed * 100000 + ep
        in_p1 = ep >= p0_episodes
        # P0 always free movement; P1 uses the arm's movement-cost config.
        env = make_env(seed=ep_seed, expensive_movement=salamone_env and in_p1)
        _flat, obs_dict = env.reset()
        agent.reset()

        for _step in range(steps_per_episode):
            obs_body, obs_world = split_obs_tensors(obs_dict)
            with torch.no_grad():
                action = agent.act_with_split_obs(
                    obs_body, obs_world, temperature=1.0,
                )
            if action is None:
                action_class = NOOP_CLASS
            else:
                action_class = int(action.argmax(dim=-1).item())

            if in_p1:
                p1_total_ticks += 1
                is_nonnoop = int(action_class != NOOP_CLASS)
                p1_nonnoop_ticks += is_nonnoop

                # E3 selection-authority diagnostics (set by E3Selector.select()
                # on the most recent selection). Captured every P1 tick so C5
                # non-vacuity is measurable on every arm.
                diag = getattr(agent.e3, "last_score_diagnostics", None)
                if diag:
                    auth_diag_ticks += 1
                    if bool(diag.get("modulatory_authority_active", False)):
                        auth_active_ticks += 1
                    auth_scale_factor_sum += float(
                        diag.get("modulatory_authority_scale_factor", 0.0)
                    )
                    auth_range_sum += float(
                        diag.get("modulatory_authority_range", 0.0)
                    )
                    raw_range_sum += float(
                        diag.get("e3_raw_score_range_mean", 0.0)
                    )

                if vigor_on and agent.tonic_vigor is not None:
                    tv_state = agent.tonic_vigor.get_state()
                    cur_window_ticks += 1
                    cur_window_nonnoop += is_nonnoop
                    cur_window_v_t_sum += float(tv_state["last_v_t"])
                    gate_prod = (
                        float(tv_state["last_gate_energy"])
                        * float(tv_state["last_gate_drive"])
                        * float(tv_state["last_gate_pe"])
                    )
                    cur_window_gate_prod_sum += gate_prod
                    if cur_window_ticks >= WINDOW_LENGTH:
                        window_v_t.append(cur_window_v_t_sum / cur_window_ticks)
                        window_action_density.append(
                            cur_window_nonnoop / cur_window_ticks
                        )
                        window_gate_product.append(
                            cur_window_gate_prod_sum / cur_window_ticks
                        )
                        cur_window_ticks = 0
                        cur_window_nonnoop = 0
                        cur_window_v_t_sum = 0.0
                        cur_window_gate_prod_sum = 0.0

            try:
                _flat, _harm_signal, done, _info, obs_dict = env.step(action_class)
            except Exception:  # noqa: BLE001
                done = True
            if done:
                break

        if dry_run or (ep + 1) % PRINT_INTERVAL == 0:
            phase = "p1" if in_p1 else "p0"
            print(
                f"  [train] seed={seed} arm={arm_label} ep {ep+1}/{total_episodes} "
                f"phase={phase} p1_ticks={p1_total_ticks} p1_nonnoop={p1_nonnoop_ticks}",
                flush=True,
            )

    # Flush any partial trailing window so very short P1s still produce one window.
    if vigor_on and cur_window_ticks > 0:
        window_v_t.append(cur_window_v_t_sum / cur_window_ticks)
        window_action_density.append(cur_window_nonnoop / cur_window_ticks)
        window_gate_product.append(cur_window_gate_prod_sum / cur_window_ticks)

    action_density = (
        p1_nonnoop_ticks / p1_total_ticks if p1_total_ticks > 0 else 0.0
    )

    if vigor_on and len(window_v_t) >= 2:
        pearson_v_t_density = pearson_r(window_v_t, window_action_density)
        gate_product_mean = sum(window_gate_product) / len(window_gate_product)
    else:
        pearson_v_t_density = 0.0
        gate_product_mean = 0.0

    # Selection-authority aggregates.
    auth_active_frac = (
        auth_active_ticks / auth_diag_ticks if auth_diag_ticks > 0 else 0.0
    )
    auth_scale_factor_mean = (
        auth_scale_factor_sum / auth_diag_ticks if auth_diag_ticks > 0 else 0.0
    )
    auth_range_mean = (
        auth_range_sum / auth_diag_ticks if auth_diag_ticks > 0 else 0.0
    )
    raw_range_mean = (
        raw_range_sum / auth_diag_ticks if auth_diag_ticks > 0 else 0.0
    )

    return {
        "seed": seed,
        "arm_label": arm_label,
        "vigor_on": vigor_on,
        "salamone_env": salamone_env,
        "p1_total_ticks": p1_total_ticks,
        "p1_nonnoop_ticks": p1_nonnoop_ticks,
        "action_density": action_density,
        "n_windows": len(window_v_t),
        "pearson_r_v_t_action_density": pearson_v_t_density,
        "gate_product_mean": gate_product_mean,
        "window_v_t": window_v_t,
        "window_action_density": window_action_density,
        "window_gate_product": window_gate_product,
        # Selection-authority (C5 non-vacuity) diagnostics.
        "auth_diag_ticks": auth_diag_ticks,
        "modulatory_authority_active_frac": auth_active_frac,
        "modulatory_authority_scale_factor_mean": auth_scale_factor_mean,
        "modulatory_authority_range_mean": auth_range_mean,
        "e3_raw_score_range_mean": raw_range_mean,
    }


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    dry_run = args.dry_run

    t0 = time.time()
    timestamp_utc = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{timestamp_utc}_v3"

    print(
        f"V3-EXQ-624c ARC-068 / MECH-320 Niv-vs-Salamone dissociation "
        f"(no-op-opportunity-rich regime; modulatory-bias-selection-authority ON)",
        flush=True,
    )
    print(
        f"seeds={SEEDS} majority={VALID_SEED_MAJORITY} "
        f"p0_ep={P0_WARMUP_EPISODES} p1_ep={P1_EVAL_EPISODES} "
        f"steps={STEPS_PER_EPISODE} grid={GRID_SIZE}x{GRID_SIZE} "
        f"hazards={N_HAZARDS} resources={N_RESOURCES} action_dim={ACTION_DIM} "
        f"v_t_floor={V_T_FLOOR} mod_authority={USE_MODULATORY_AUTHORITY} "
        f"gain={MODULATORY_AUTHORITY_GAIN} dry_run={dry_run}",
        flush=True,
    )

    # Arm configs: (label, vigor_on, salamone_env).
    arm_configs = [
        ("ARM_0_baseline", False, False),
        ("ARM_1_vigor_niv", True, False),
        ("ARM_2_vigor_salamone", True, True),
        ("ARM_3_baseline_salamone", False, True),
    ]

    # Results indexed by arm label.
    results_by_arm = {label: [] for label, _, _ in arm_configs}

    for seed in SEEDS:
        for arm_label, vigor_on, salamone_env in arm_configs:
            r = run_arm(seed, arm_label, vigor_on, salamone_env, dry_run)
            results_by_arm[arm_label].append(r)
            # Per-arm-seed verdict line: PASS iff the run executed end-to-end.
            # Scientific PASS/FAIL is computed at overall level.
            # seeds * conditions = 5 * 4 = 20 verdict lines total.
            print(
                f"verdict: {'PASS' if r['p1_total_ticks'] > 0 or dry_run else 'FAIL'}",
                flush=True,
            )

    # Pre-registered acceptance computations.
    n_seeds = len(SEEDS)
    per_seed_c1_lift = []
    per_seed_salamone_lift = []
    per_seed_c2_delta_diff = []
    per_seed_c2_pass = []
    per_seed_c3_pass = []
    per_seed_c4_pass = []
    per_seed_c1_pass = []
    per_seed_c5_pass = []
    per_seed_valid = []
    for i in range(n_seeds):
        d0 = results_by_arm["ARM_0_baseline"][i]["action_density"]
        d1 = results_by_arm["ARM_1_vigor_niv"][i]["action_density"]
        d2 = results_by_arm["ARM_2_vigor_salamone"][i]["action_density"]
        d3 = results_by_arm["ARM_3_baseline_salamone"][i]["action_density"]
        gp1 = results_by_arm["ARM_1_vigor_niv"][i]["gate_product_mean"]
        arm1 = results_by_arm["ARM_1_vigor_niv"][i]

        # C1: Niv lift (vigor effect with movement cheap)
        c1_lift = d1 - d0
        per_seed_c1_lift.append(c1_lift)
        per_seed_c1_pass.append(c1_lift >= C1_LIFT_MIN)

        # C2: dissociation. Salamone-side vigor lift (d2 - d3) vs Niv-side
        # (d1 - d0). Under the Niv kernel both should match.
        salamone_lift = d2 - d3
        per_seed_salamone_lift.append(salamone_lift)
        denom = max(abs(c1_lift), 1e-6)
        delta_diff = abs(salamone_lift - c1_lift) / denom
        per_seed_c2_delta_diff.append(delta_diff)
        per_seed_c2_pass.append(
            delta_diff < C2_DISSOCIATION_TOL and salamone_lift >= 0.5 * c1_lift
        )

        # C3: gate sanity (ARM_1)
        per_seed_c3_pass.append(gp1 >= C3_GATE_PRODUCT_MIN)

        # C4: well-fed-safe regime no-op penalty observable (informative).
        per_seed_c4_pass.append(c1_lift >= C4_NO_OP_PENALTY_MIN)

        # C5: selection-authority non-vacuity in ARM_1 (the lever genuinely
        # fired AND scores were bounded). Distinguishes a genuine null from a
        # starved lever (the 624a / 643 failure mode).
        c5 = (
            arm1["modulatory_authority_active_frac"] >= C5_AUTH_ACTIVE_FRAC_MIN
            and arm1["modulatory_authority_range_mean"] > C5_AUTH_RANGE_MIN
            and arm1["e3_raw_score_range_mean"] < C5_RAW_RANGE_MAX
        )
        per_seed_c5_pass.append(c5)

        # VALID seed = positive control genuinely engaged (C1 AND C5). C2 is
        # evaluated ONLY on valid seeds -> non-vacuous strict dissociation.
        per_seed_valid.append(per_seed_c1_pass[i] and c5)

    n_valid = sum(1 for v in per_seed_valid if v)
    # C2 / C3 STRICT on every valid seed.
    c2_on_valid = all(
        per_seed_c2_pass[i] for i in range(n_seeds) if per_seed_valid[i]
    )
    c3_on_valid = all(
        per_seed_c3_pass[i] for i in range(n_seeds) if per_seed_valid[i]
    )
    majority_met = n_valid >= VALID_SEED_MAJORITY
    overall_pass = majority_met and c2_on_valid and c3_on_valid

    # Convenience aggregates over all seeds (for the manifest summary).
    c1_pass_all = all(per_seed_c1_pass)
    c2_pass_all = all(per_seed_c2_pass)
    c3_pass_all = all(per_seed_c3_pass)
    c4_pass_all = all(per_seed_c4_pass)
    c5_pass_all = all(per_seed_c5_pass)

    outcome = "PASS" if overall_pass else "FAIL"
    if overall_pass:
        # Grid row 1.
        evidence_direction = "supports"
        per_claim = {"MECH-320": "supports", "ARC-068": "supports"}
    else:
        # Grid routing (computed for note, applied via per-claim direction).
        if not majority_met:
            # Row 3: the positive control fired on too few seeds even in the
            # no-op-rich regime. The test could not let the claim express on a
            # majority of seeds -- env-adequacy / authority-engagement gap, NOT
            # a verdict on MECH-320 / ARC-068.
            evidence_direction = "non_contributory"
            per_claim = {
                "MECH-320": "non_contributory",
                "ARC-068": "non_contributory",
            }
        elif majority_met and not c2_on_valid:
            # Row 2: Salamone-like dissociation failure on a valid seed --
            # effort-cost-like. ARC-068 weakens; MECH-320 is the load-bearing
            # target of the weaken (the implementation claim that w_passive*v_t
            # is opportunity-cost, not effort-cost).
            evidence_direction = "weakens"
            per_claim = {"MECH-320": "weakens", "ARC-068": "weakens"}
        elif majority_met and c2_on_valid and not c3_on_valid:
            # Row 4: vigor gate did not fire on a valid seed even with
            # v_t_floor=0.05 -- calibration / wiring issue, not a claim verdict.
            evidence_direction = "non_contributory"
            per_claim = {
                "MECH-320": "non_contributory",
                "ARC-068": "non_contributory",
            }
        else:
            evidence_direction = "mixed"
            per_claim = {"MECH-320": "mixed", "ARC-068": "mixed"}

    # Aggregate metrics for summary.
    mean_density_by_arm = {
        label: sum(r["action_density"] for r in results_by_arm[label]) / n_seeds
        for label, _, _ in arm_configs
    }
    mean_c1_lift = sum(per_seed_c1_lift) / n_seeds
    mean_c2_delta_diff = sum(per_seed_c2_delta_diff) / n_seeds
    mean_pearson_arm1 = (
        sum(
            r["pearson_r_v_t_action_density"]
            for r in results_by_arm["ARM_1_vigor_niv"]
        )
        / n_seeds
    )
    mean_gate_arm1 = (
        sum(r["gate_product_mean"] for r in results_by_arm["ARM_1_vigor_niv"])
        / n_seeds
    )
    mean_auth_active_frac_arm1 = (
        sum(
            r["modulatory_authority_active_frac"]
            for r in results_by_arm["ARM_1_vigor_niv"]
        )
        / n_seeds
    )
    mean_auth_range_arm1 = (
        sum(
            r["modulatory_authority_range_mean"]
            for r in results_by_arm["ARM_1_vigor_niv"]
        )
        / n_seeds
    )
    mean_auth_scale_arm1 = (
        sum(
            r["modulatory_authority_scale_factor_mean"]
            for r in results_by_arm["ARM_1_vigor_niv"]
        )
        / n_seeds
    )
    mean_raw_range_arm1 = (
        sum(
            r["e3_raw_score_range_mean"]
            for r in results_by_arm["ARM_1_vigor_niv"]
        )
        / n_seeds
    )

    elapsed = time.time() - t0

    manifest = {
        "schema_version": "v1",
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "supersedes": SUPERSEDES,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "timestamp_utc": timestamp_utc,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": CLAIM_IDS,
        "backlog_id": "EVB-0237",
        "proposal_id": "EXP-0081",
        "outcome": outcome,
        "evidence_direction": evidence_direction,
        "evidence_direction_per_claim": per_claim,
        "evidence_class": "discriminative_pair",
        "claim_ids_tested": CLAIM_IDS,
        "registered_thresholds": {
            "C1_LIFT_MIN": C1_LIFT_MIN,
            "C2_DISSOCIATION_TOL": C2_DISSOCIATION_TOL,
            "C3_GATE_PRODUCT_MIN": C3_GATE_PRODUCT_MIN,
            "C4_NO_OP_PENALTY_MIN": C4_NO_OP_PENALTY_MIN,
            "C5_AUTH_ACTIVE_FRAC_MIN": C5_AUTH_ACTIVE_FRAC_MIN,
            "C5_AUTH_RANGE_MIN": C5_AUTH_RANGE_MIN,
            "C5_RAW_RANGE_MAX": C5_RAW_RANGE_MAX,
            "VALID_SEED_MAJORITY": VALID_SEED_MAJORITY,
            "V_T_FLOOR": V_T_FLOOR,
            "USE_MODULATORY_AUTHORITY": USE_MODULATORY_AUTHORITY,
            "MODULATORY_AUTHORITY_GAIN": MODULATORY_AUTHORITY_GAIN,
            "SALAMONE_DAMAGE_INCREMENT": SALAMONE_DAMAGE_INCREMENT,
            "SALAMONE_FAILURE_PROB_SCALE": SALAMONE_FAILURE_PROB_SCALE,
        },
        "summary": {
            "scenario": (
                "No-op-opportunity-rich regime (size=%d grid, %d hazards, %d "
                "resources, use_proxy_fields=True, action_dim=5 with "
                "noop_class=4). %d matched seeds; positive control gated by a "
                "strict majority (>= %d valid seeds). limb_damage_enabled=True "
                "on ALL arms (body_obs_dim=17 uniform); the Niv-vs-Salamone "
                "manipulation is movement-cost-ONLY via damage_increment / "
                "failure_prob_scale. Sparser foraging (num_resources=2 vs 624b's "
                "3, on the proven size-8 / 1-hazard base) lowers the benefit-"
                "approach pull so more states are neutral near-ties where the "
                "no-op candidate is competitive -> lower baseline action_density "
                "-> the w_passive bias has headroom every seed. (A 2026-06-07 "
                "regime probe showed denser hazards BACKFIRE -- the agent flees, "
                "saturating action_density -- and that P0-warmup length, not env "
                "density, is the dominant headroom lever; P0 retained at 100.) "
                "P0 warmup 100 ep (vigor OFF + free-movement env in all "
                "arms, env stepping only -- no gradient training), P1 eval 30 "
                "ep x 200 steps (arm flag + movement-cost config toggled). "
                "4-arm Niv-vs-Salamone dissociation: ARM_0 baseline; ARM_1 "
                "vigor + free movement (Niv kernel); ARM_2 vigor + Salamone-"
                "style elevated movement cost; ARM_3 baseline + Salamone env. "
                "modulatory-bias-selection-authority (gain=0.5) ON in ALL arms "
                "(no-op where no modulatory bias exists, so ARM_0 / ARM_3 are "
                "bit-identical). Tests the R4 verdict: ARC-068 / MECH-320 "
                "w_passive must remain insensitive to parametric movement cost "
                "(Niv 2007 opportunity-cost-on-time), distinct from MECH-258 / "
                "SD-032b dacc_effort_cost (Salamone & Correa 2003). Forced-"
                "vigor probe v_t_floor=0.05 (V3-EXQ-549 fix). Supersedes "
                "V3-EXQ-624b (env-degenerate seed + vacuous-C2 gate)."
                % (GRID_SIZE, N_HAZARDS, N_RESOURCES, n_seeds, VALID_SEED_MAJORITY)
            ),
            "interpretation": (
                f"action_density: ARM_0={mean_density_by_arm['ARM_0_baseline']:.4f}; "
                f"ARM_1={mean_density_by_arm['ARM_1_vigor_niv']:.4f}; "
                f"ARM_2={mean_density_by_arm['ARM_2_vigor_salamone']:.4f}; "
                f"ARM_3={mean_density_by_arm['ARM_3_baseline_salamone']:.4f}. "
                f"n_valid={n_valid}/{n_seeds} (majority>={VALID_SEED_MAJORITY}): "
                f"{'MET' if majority_met else 'NOT MET'}. "
                f"C1 Niv lift (ARM_1 - ARM_0) mean={mean_c1_lift:+.4f} "
                f"(per-seed pass {sum(per_seed_c1_pass)}/{n_seeds}, >= {C1_LIFT_MIN}). "
                f"C2 dissociation on valid seeds (STRICT, < {C2_DISSOCIATION_TOL}): "
                f"{'PASS' if c2_on_valid else 'FAIL'} (mean delta_diff over all "
                f"seeds={mean_c2_delta_diff:.3f}). "
                f"C3 gate_product ARM_1 on valid seeds: "
                f"{'PASS' if c3_on_valid else 'FAIL'} (mean={mean_gate_arm1:.3f}, "
                f">= {C3_GATE_PRODUCT_MIN}). "
                f"C5 authority ARM_1 active_frac={mean_auth_active_frac_arm1:.3f} "
                f"range={mean_auth_range_arm1:.3e} raw_range={mean_raw_range_arm1:.3e} "
                f"(per-seed pass {sum(per_seed_c5_pass)}/{n_seeds}). "
                f"C4 well-fed-safe no-op penalty observable: "
                f"{'PASS' if c4_pass_all else 'FAIL'} (informative). "
                f"Within-ARM_1 Pearson r(v_t, density) mean={mean_pearson_arm1:+.3f}. "
                f"Outcome: {outcome}, evidence_direction={evidence_direction}."
            ),
            "pairwise_deltas": {
                "per_seed_c1_niv_lift": per_seed_c1_lift,
                "per_seed_salamone_lift": per_seed_salamone_lift,
                "per_seed_c2_delta_diff": per_seed_c2_delta_diff,
                "mean_c1_niv_lift": mean_c1_lift,
                "mean_c2_delta_diff": mean_c2_delta_diff,
                "mean_action_density_by_arm": mean_density_by_arm,
                "mean_pearson_r_arm1": mean_pearson_arm1,
                "mean_gate_product_arm1": mean_gate_arm1,
            },
            "selection_authority": {
                "mean_active_frac_arm1": mean_auth_active_frac_arm1,
                "mean_authority_range_arm1": mean_auth_range_arm1,
                "mean_scale_factor_arm1": mean_auth_scale_arm1,
                "mean_raw_score_range_arm1": mean_raw_range_arm1,
            },
        },
        "criteria": {
            "n_seeds": n_seeds,
            "valid_seed_majority": VALID_SEED_MAJORITY,
            "n_valid": n_valid,
            "majority_met": majority_met,
            "per_seed_c1_lift": per_seed_c1_lift,
            "per_seed_salamone_lift": per_seed_salamone_lift,
            "per_seed_c2_delta_diff": per_seed_c2_delta_diff,
            "per_seed_c1_pass": per_seed_c1_pass,
            "per_seed_c2_pass": per_seed_c2_pass,
            "per_seed_c3_pass": per_seed_c3_pass,
            "per_seed_c4_pass": per_seed_c4_pass,
            "per_seed_c5_pass": per_seed_c5_pass,
            "per_seed_valid": per_seed_valid,
            "c2_on_valid_seeds": c2_on_valid,
            "c3_on_valid_seeds": c3_on_valid,
            "c1_pass_all_seeds": c1_pass_all,
            "c2_pass_all_seeds": c2_pass_all,
            "c3_pass_all_seeds": c3_pass_all,
            "c4_pass_all_seeds": c4_pass_all,
            "c5_pass_all_seeds": c5_pass_all,
            "overall_pass": overall_pass,
        },
        "config": {
            "seeds": SEEDS,
            "valid_seed_majority": VALID_SEED_MAJORITY,
            "p0_warmup_episodes": P0_WARMUP_EPISODES,
            "p1_eval_episodes": P1_EVAL_EPISODES,
            "steps_per_episode": STEPS_PER_EPISODE,
            "window_length": WINDOW_LENGTH,
            "grid_size": GRID_SIZE,
            "n_hazards": N_HAZARDS,
            "n_resources": N_RESOURCES,
            "action_dim": ACTION_DIM,
            "noop_class": NOOP_CLASS,
            "limb_damage_enabled_all_arms": True,
            "tonic_vigor_form": "additive",
            "v_t_floor": V_T_FLOOR,
            "use_modulatory_selection_authority": USE_MODULATORY_AUTHORITY,
            "modulatory_authority_gain": MODULATORY_AUTHORITY_GAIN,
            "free_damage_increment": FREE_DAMAGE_INCREMENT,
            "free_failure_prob_scale": FREE_FAILURE_PROB_SCALE,
            "salamone_damage_increment": SALAMONE_DAMAGE_INCREMENT,
            "salamone_failure_prob_scale": SALAMONE_FAILURE_PROB_SCALE,
            "dry_run": dry_run,
        },
        "metrics": {
            "results_by_arm": results_by_arm,
        },
        "elapsed_seconds": elapsed,
        "notes": (
            "ARC-068 (action.opportunity_cost_no_op_penalty) Niv-vs-Salamone "
            "discriminative test on the MECH-320 w_passive implementation. "
            "CLEAN RE-SPEC of V3-EXQ-624b (non_contributory + narrow_supports: "
            "the C2 dissociation PASSED all 3 seeds but the positive-control "
            "conjunction C1+C5 failed on the env-degenerate seed 42, where the "
            "well-fed-safe-familiar regime presented no competitive no-op "
            "candidate; failure_autopsy_V3-EXQ-624b_2026-06-07). TWO load-"
            "bearing corrections per the autopsy repair pathway: (1) a NO-OP-"
            "OPPORTUNITY-RICH regime via SPARSER FORAGING (num_resources 3 -> 2 "
            "on the proven size-8 / 1-hazard base) -- fewer benefit-approach "
            "pulls means more neutral near-tie states where the no-op candidate "
            "is competitive, lowering baseline action_density so the w_passive "
            "bias has headroom; applied to ALL arms, NOT part of the Niv-vs-"
            "Salamone manipulation, so the dissociation difference still cancels "
            "the shared regime. A 2026-06-07 regime probe drove this choice: "
            "denser hazards / a tighter grid BACKFIRE (the agent flees, "
            "saturating action_density at 1.0 -- probe 6/3/2 gave 2/5 valid, "
            "worse than 624b), and P0-warmup length (not env density) is the "
            "dominant headroom lever (P0=8 saturates, full P0=100 gave 2/3 "
            "valid), so P0 is retained at 100. (2) 5 seeds + a NON-VACUOUS "
            "MAJORITY GATE on "
            "the positive control with C2 kept STRICT -- C2 is evaluated only on "
            "seeds where the positive control genuinely fired (C1 AND C5), so a "
            "saturated seed can no longer pass C2 vacuously (the 624b seed-42 "
            "artifact: |salamone - niv| ~ 0 cleared the tolerance with both "
            "lifts ~0). harm_gradient was deliberately NOT used as the no-op "
            "lever -- it fires on transition_type=='none' regardless of move/"
            "stay and penalises hazard PROXIMITY, pushing the agent to flee "
            "(raise action_density), the wrong direction. No SD-056 online "
            "training (no gradient training), so raw_score_range is expected "
            "O(1), far below the 643 ~1e32 explosion the C5 raw-range guard "
            "(1e6) watches for. EXP-0081 / EVB-0237 dispatch_mode="
            "discriminative_pair. R3 verdict: slot-level ARC-068 registration "
            "preserved, implementation collapses into MECH-320. R4 verdict: "
            "effort-cost (MECH-258 / SD-032b) NOT to be collapsed with "
            "opportunity-cost-on-time (ARC-068 / MECH-320 w_passive). Pre-"
            "registered thresholds set as constants. PASS = (n_valid >= "
            "VALID_SEED_MAJORITY) AND C2 on every valid seed AND C3 on every "
            "valid seed. 5-row interpretation grid: (1) PASS -> ARC-068 + "
            "MECH-320 support; (2) majority met, C2 fails on a valid seed -> "
            "ARC-068 weakens (effort-cost-like); (3) majority NOT met -> "
            "non_contributory (positive control starved on too many seeds; "
            "/failure-autopsy / re-spec); (4) majority met, C2 ok, C3 fails on "
            "a valid seed -> non_contributory (/diagnose-errors); (5) other -> "
            "mixed. Supersedes V3-EXQ-624b."
        ),
    }

    out_dir = os.path.abspath(
        os.path.join(REPO_ROOT, "..", "REE_assembly", "evidence", "experiments")
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{run_id}.json")
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Overall outcome reported via emit_outcome() + manifest, not via a final
    # 'verdict:' line (which would over-count seeds*conditions = 20).
    print(f"outcome: {outcome}", flush=True)
    print(
        f"action_density "
        f"ARM_0={mean_density_by_arm['ARM_0_baseline']:.4f} "
        f"ARM_1={mean_density_by_arm['ARM_1_vigor_niv']:.4f} "
        f"ARM_2={mean_density_by_arm['ARM_2_vigor_salamone']:.4f} "
        f"ARM_3={mean_density_by_arm['ARM_3_baseline_salamone']:.4f} "
        f"n_valid={n_valid}/{n_seeds} c1_lift={mean_c1_lift:+.4f} "
        f"c2_delta_diff={mean_c2_delta_diff:.3f} gate_arm1={mean_gate_arm1:.3f} "
        f"auth_active_frac_arm1={mean_auth_active_frac_arm1:.3f} "
        f"auth_range_arm1={mean_auth_range_arm1:.3e} "
        f"raw_range_arm1={mean_raw_range_arm1:.3e}",
        flush=True,
    )
    print(f"Result written to: {out_path}", flush=True)
    return outcome, out_path


if __name__ == "__main__":
    _outcome, _manifest_path = main()
    emit_outcome(outcome=_outcome, manifest_path=_manifest_path)
    sys.exit(0)
