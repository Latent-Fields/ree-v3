"""V3-EXQ-624b: ARC-068 / MECH-320 opportunity-cost vs effort-cost dissociation.

CLEAN RETEST of V3-EXQ-624a now that the modulatory-bias-selection-authority
substrate is validated (V3-EXQ-643a PASS, 2026-06-06). The scientific question
is unchanged from V3-EXQ-624 / V3-EXQ-624a.

Why this retest (the load-bearing change vs 624a):
  V3-EXQ-624a FAILed NOT as a scientific null but because the MECH-320 vigor
  score-bias had ZERO authority over the committed argmin at E3.select. Its
  failure_record signature was "action_density lift ARM_1 - ARM_0 = 0.0 (both
  0.865) with gate_product 1.0 / v_t 0.05" -- the vigor gate fired and v_t was
  positive, but a fixed small modulatory magnitude (~0.05-0.1) added to primary
  E3 scores whose raw_score_range is much larger never changes the argmin (the
  2026-06-03 cluster autopsy failure_autopsy_604a-624a-630, shared with
  V3-EXQ-604a curiosity_bias=0.0 and V3-EXQ-614d within-class temperature
  byte-identical). The fix is the modulatory-bias-selection-authority substrate
  (landed 2026-06-03; gap-relative scaling so the combined modulatory
  contribution is rescaled to modulatory_authority_gain * raw_score_range, giving
  the bias BOUNDED authority over the argmin without touching primary scores).
  That substrate's own readiness validation (V3-EXQ-643) hit a SECOND, distinct
  bug -- float32 catastrophic cancellation when the combined modulatory
  contribution was reconstructed as (scores - scores_raw) at large primary-score
  magnitude -- fixed in V3-EXQ-643a (the accumulator is now tracked explicitly).
  Both blockers are now resolved. 624b therefore differs from 624a by exactly
  ONE substrate flag: use_modulatory_selection_authority=True (gain=0.5).

  use_modulatory_selection_authority is enabled in ALL FOUR arms (it is part of
  the fixed substrate config, NOT a manipulated variable). It is a strict no-op
  in the vigor-OFF arms (ARM_0 / ARM_3): with no tonic-vigor / dACC / curiosity
  bias active, the combined modulatory contribution is None and the authority
  block does not run, so ARM_0 / ARM_3 are bit-identical to their 624a form.
  The authority block only does anything in the vigor-ON arms (ARM_1 / ARM_2),
  where the MECH-320 w_passive bias now reaches the argmin. Tonic vigor remains
  the single manipulated variable across the ARM_0 vs ARM_1 contrast.

Proposal: EXP-0081 / EVB-0237 (dispatch_mode=discriminative_pair). Behavioural
validation of the ARC-068 architectural slot (opportunity_cost_no_op_penalty)
as instantiated by MECH-320's w_passive term per the lit-pull R3 verdict
2026-05-16 (slot-level ARC-068 registration preserved; implementation collapses
into MECH-320 w_passive*v_t at e3.select).

Substrate prerequisites all landed:
  - MECH-320 substrate landed 2026-05-10 (ree_core/policy/tonic_vigor.py).
  - modulatory-bias-selection-authority landed 2026-06-03; float32-cancellation
    amend (V3-EXQ-643a) landed 2026-06-06. Gives MECH-320's w_passive bias
    bounded authority over the E3.select argmin.
  - V3-EXQ-547 substrate-readiness diagnostic PASS 2026-05-10 (6/6 sub-tests).
  - ARC-068 lit-pull synthesis lit_conf ~0.78 supports-direction (Niv 2007 +
    Kurzban 2013 + Kolling 2016 + Shenhav 2016 + Salamone & Correa 2003 +
    Constantino & Daw 2015). R3 verdict: collapse-at-implementation-layer
    licensed (additive Niv form). R4 verdict: effort-cost vs opportunity-cost
    separation REQUIRED (Salamone & Correa 2003 dissociation; ARC-068 must NOT
    absorb into MECH-258 / SD-032b dacc_effort_cost machinery).

Inherited fix from V3-EXQ-624a (observation-space confound, retained verbatim):
  limb_damage_enabled=True on ALL arms (body_obs_dim == 17 uniformly) so the
  Niv-vs-Salamone manipulation is movement-cost-ONLY (carried by
  damage_increment / failure_prob_scale), never an observation-dimensionality
  change. The Niv side / all P0 warmups set damage_increment=0.0 and
  failure_prob_scale=0.0 -> movement free, damage channels zero. The Salamone
  side sets the elevated values so movement is expensive.

ARC-068 architectural prediction (R4 verdict: opportunity-cost-on-time, NOT
effort-cost-on-movement):
  In the well-fed-safe-familiar regime where movement is cheap, the no-op
  penalty (MECH-320 w_passive * v_t) should still scale with the EWMA over
  realised reward (Niv 2007 average reward rate), NOT with the per-step
  movement cost. This is the falsifiable discriminative prediction: ARC-068
  must remain insensitive to parametric increases in movement-cost-on-action
  (Salamone & Correa 2003 effort cost), while continuing to fire when v_t is
  positive (forced via v_t_floor to bypass V3-EXQ-549's calibration failure).

Four-arm design (unchanged from 624a):
  ARM_0_baseline:
    use_tonic_vigor=False, free-movement env (limb_damage_enabled=True with
    damage_increment=0.0, failure_prob_scale=0.0). Reference action_density.
  ARM_1_vigor_niv:
    use_tonic_vigor=True, additive form, v_t_floor=0.05 (forced-vigor probe
    per V3-EXQ-549's prescribed fix; small positive floor that survives
    gate-drive collapse), free-movement env.
    Predicts elevated action_density (Niv-style no-op penalty on cheap
    movement) -- NOW that the bias has selection authority.
  ARM_2_vigor_salamone:
    use_tonic_vigor=True, additive form, v_t_floor=0.05, env with elevated
    movement cost (damage_increment=0.30, failure_prob_scale=0.6).
    Niv kernel predicts: action_density delta from ARM_3 matches ARM_1's delta
    from ARM_0 (no-op penalty independent of movement cost). Salamone kernel
    predicts: suppressed or reversed (no-op penalty conflated with effort cost).
  ARM_3_baseline_salamone:
    use_tonic_vigor=False, same elevated-movement-cost env as ARM_2.
    Movement-cost-only control.

  All four arms: use_modulatory_selection_authority=True,
  modulatory_authority_gain=0.5 (fixed substrate; no-op where no modulatory
  bias exists).

Protocol (unchanged from 624a):
  P0 warmup (100 ep, vigor OFF + free-movement env in all arms): identical
    baseline policy checkpoint per seed (env stepping only; no gradient
    training, matching the 624 lineage design).
  P1 measurement (30 ep x 200 steps): arm flag and env movement-cost config
    toggled. Observation space identical (17-dim body) across all arms/phases.

Environment (well-fed-safe-familiar regime, matched across arms by seed):
  CausalGridWorldV2 size=8, num_hazards=1, num_resources=3, action_dim=5
  (action 0 = up, 1 = down, 2 = left, 3 = right, 4 = noop).

Metrics (per arm per seed, P1 only):
  action_density: mean over P1 ticks of [argmax(action) != NOOP_CLASS].
    The behavioural signature: opportunity-cost no-op penalty lifts this.
  v_t_window / action_density_window / gate_product (ARM_1, ARM_2 only).
  E3 selection-authority diagnostics (from agent.e3.last_score_diagnostics,
    captured every P1 tick): modulatory_authority_active fraction,
    modulatory_authority_scale_factor mean, modulatory_authority_range mean,
    e3_raw_score_range mean. These provide the NON-VACUITY evidence: the C1
    lift is only interpretable as a scientific result if the authority lever
    genuinely fired (active_frac > 0, range > floor) AND primary scores were
    bounded (raw_score_range did not explode the way SD-056 online training
    drove V3-EXQ-643 to ~1e32). 624b does NOT train SD-056 online (no gradient
    training at all), so raw_score_range is expected to be O(1); the readout
    confirms it.

Pre-registered acceptance thresholds (defined here, NOT inferred post-hoc):
  C1 action_density lift (vigor effect): mean(action_density_ARM_1) -
      mean(action_density_ARM_0) >= C1_LIFT_MIN (default 0.03; 3pp).
      Paired by seed. The basic MECH-320 firing test, per the substrate target.
  C2 Niv-vs-Salamone dissociation (R4 verdict test): the ARM_2 - ARM_3 lift
      should match the ARM_1 - ARM_0 lift within C2_DISSOCIATION_TOL (0.50 of
      the ARM_1 - ARM_0 lift), and salamone_lift >= 0.5 * c1_lift. If the
      implementation is secretly effort-cost-like, ARM_2 - ARM_3 is suppressed
      or reversed relative to ARM_1 - ARM_0.
  C3 gate sanity: mean(gate_product) > C3_GATE_PRODUCT_MIN (0.5) in ARM_1.
  C4 well-fed-safe regime no-op penalty observable (Salamone discrimination,
      informative-only; numerically C1).
  C5 selection-authority NON-VACUITY (NEW for 624b -- the lever genuinely had
      authority and scores were bounded):
        modulatory_authority_active_frac(ARM_1) >= C5_AUTH_ACTIVE_FRAC_MIN (0.5)
        AND modulatory_authority_range(ARM_1) > C5_AUTH_RANGE_MIN (1e-6)
        AND e3_raw_score_range(ARM_1) < C5_RAW_RANGE_MAX (1e6, explosion guard).
      C5 is what distinguishes a genuine scientific null (lever fired, no lift)
      from a starved lever (the 624a / 643 failure mode). When C5 fails the
      run is NOT a valid test of the claim -> non_contributory, NOT a weaken.

PASS = C1 AND C2 AND C3 AND C5 across all seeds (C4 logged separately).

5-row interpretation grid:
  (1) C5 PASS AND C1+C2+C3 PASS:
      ARC-068 supports + MECH-320 supports. The slot preservation is
      vindicated with the authority substrate in place. Surface to /governance
      for next-cycle promotion-candidate consideration.
  (2) C5 PASS AND C1+C3 PASS AND C2 FAIL (ARM_2 - ARM_3 lift << ARM_1 - ARM_0):
      ARC-068 weakens. Effort-cost-like (movement-cost-sensitive). Contradicts
      R4; surface the supersession question (ARC-068 collapse into MECH-258 /
      SD-032b) to /governance.
  (3) C5 PASS AND C1 FAIL with C3 PASS (authority fired, v_t positive, no lift):
      MECH-320 / ARC-068 WEAKEN. The bias now has genuine authority over the
      argmin and v_t is positive, yet action_density does not lift -> the
      no-op penalty does not produce the predicted behaviour. Route to
      /failure-autopsy / /governance.
  (4) C5 FAIL (authority never fired OR raw_score_range exploded):
      non_contributory. The test could not let the claim express itself
      (substrate / harness issue, not a verdict on MECH-320 / ARC-068). Route
      to /diagnose-errors (scores exploded) or /failure-autopsy (authority
      inert despite a non-zero bias).
  (5) C3 FAIL (vigor gate does not fire even with v_t_floor=0.05):
      non_contributory; calibration / wiring problem. Route to /diagnose-errors.

experiment_purpose = "evidence" (tests the load-bearing claim of MECH-320
under the ARC-068 architectural framing, now that the modulatory-bias-selection-
authority substrate lets the bias reach the argmin).

claim_ids = ["MECH-320", "ARC-068"] with evidence_direction_per_claim.
MECH-258 and SD-032b are NOT in claim_ids: they are foils architecturally
dissociated from, not tested directly.

Supersedes V3-EXQ-624a (zeroed-lever FAIL).

Run with:
  /opt/local/bin/python3 experiments/v3_exq_624b_arc068_mech320_niv_salamone_dissociation.py
or:
  /opt/local/bin/python3 experiments/v3_exq_624b_arc068_mech320_niv_salamone_dissociation.py --dry-run
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
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from pathlib import Path  # noqa: E402


# ----------------------------------------------------------------------
# Constants and pre-registered thresholds
# ----------------------------------------------------------------------
EXPERIMENT_TYPE = "v3_exq_624b_arc068_mech320_niv_salamone_dissociation"
SUPERSEDES = "v3_exq_624a_arc068_mech320_niv_salamone_dissociation"
CLAIM_IDS = ["MECH-320", "ARC-068"]
EXPERIMENT_PURPOSE = "evidence"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"

SEEDS = [42, 43, 44]
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

GRID_SIZE = 8
N_HAZARDS = 1
N_RESOURCES = 3
ACTION_DIM = 5  # CausalGridWorldV2.ACTIONS: 0=up, 1=down, 2=left, 3=right, 4=noop
NOOP_CLASS = 4  # matches CausalGridWorldV2 convention; TonicVigorConfig
                # default noop_class=0 (MECH-279 convention) overridden via
                # tonic_vigor_noop_class=NOOP_CLASS in make_config

# v_t_floor: forced-vigor probe per V3-EXQ-549 prescribed fix. Small positive
# value that survives gate-drive collapse and ensures the downstream score-bias
# path is exercised regardless of the EWMA / gate state.
V_T_FLOOR = 0.05

# Modulatory-bias-selection-authority substrate (landed 2026-06-03; float32
# amend V3-EXQ-643a 2026-06-06). The load-bearing change vs 624a: gives the
# MECH-320 w_passive bias BOUNDED authority over the E3.select argmin. ON in
# ALL arms (no-op where no modulatory bias exists). gain 0.5 < 1.0 keeps the
# modulatory channel competitive in near-ties but subdominant when the primary
# harm/goal gap exceeds gain * raw_score_range.
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
# C5 selection-authority non-vacuity gate (new for 624b).
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
    damage observation channels stay at zero -- behaviourally the
    well-fed-safe-familiar regime, but with the 17-dim observation space the
    Salamone arms also use.
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
    (use_modulatory_selection_authority=True, gain=0.5): it is part of the
    fixed substrate config, not a manipulated variable, and is a strict no-op
    where no modulatory bias is present (ARM_0 / ARM_3, vigor off -> no
    tonic-vigor / dACC / curiosity bias -> the authority block does not run).
    In the vigor-on arms it gives the MECH-320 w_passive bias bounded authority
    over the E3.select argmin (the load-bearing fix vs 624a).

    body_obs_dim / world_obs_dim are read from the env (limb_damage always on
    -> 17), so the agent is dimension-consistent across P0 warmup and the P1
    measurement env switch.

    tonic_vigor_noop_class is set to CausalGridWorldV2's action-4 no-op rather
    than the TonicVigorConfig default (0).
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
        f"V3-EXQ-624b ARC-068 / MECH-320 Niv-vs-Salamone dissociation "
        f"(modulatory-bias-selection-authority ON)",
        flush=True,
    )
    print(
        f"seeds={SEEDS} p0_ep={P0_WARMUP_EPISODES} p1_ep={P1_EVAL_EPISODES} "
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
            # seeds * conditions = 3 * 4 = 12 verdict lines total.
            print(
                f"verdict: {'PASS' if r['p1_total_ticks'] > 0 or dry_run else 'FAIL'}",
                flush=True,
            )

    # Pre-registered acceptance computations.
    n_seeds = len(SEEDS)
    per_seed_c1_lift = []
    per_seed_c2_delta_diff = []
    per_seed_c2_pass = []
    per_seed_c3_pass = []
    per_seed_c4_pass = []
    per_seed_c1_pass = []
    per_seed_c5_pass = []
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

    c1_pass_all = all(per_seed_c1_pass)
    c2_pass_all = all(per_seed_c2_pass)
    c3_pass_all = all(per_seed_c3_pass)
    c4_pass_all = all(per_seed_c4_pass)
    c5_pass_all = all(per_seed_c5_pass)
    overall_pass = c1_pass_all and c2_pass_all and c3_pass_all and c5_pass_all

    outcome = "PASS" if overall_pass else "FAIL"
    if overall_pass:
        # Grid row 1.
        evidence_direction = "supports"
        per_claim = {"MECH-320": "supports", "ARC-068": "supports"}
    else:
        # Grid routing (computed for note, applied via per-claim direction).
        if not c5_pass_all:
            # Row 4: the authority lever never fired (or scores exploded). The
            # test could not let the claim express itself -- substrate / harness
            # issue, NOT a verdict on MECH-320 / ARC-068.
            evidence_direction = "non_contributory"
            per_claim = {
                "MECH-320": "non_contributory",
                "ARC-068": "non_contributory",
            }
        elif not c3_pass_all:
            # Row 5: vigor gate did not fire even with v_t_floor=0.05.
            evidence_direction = "non_contributory"
            per_claim = {
                "MECH-320": "non_contributory",
                "ARC-068": "non_contributory",
            }
        elif c1_pass_all and not c2_pass_all:
            # Row 2: Salamone-like dissociation failure -- effort-cost-like.
            # ARC-068 weakens; MECH-320 is the load-bearing target of the weaken
            # (the implementation claim that w_passive*v_t is opportunity-cost).
            evidence_direction = "weakens"
            per_claim = {"MECH-320": "weakens", "ARC-068": "weakens"}
        elif not c1_pass_all and c3_pass_all and c5_pass_all:
            # Row 3: authority fired, v_t positive, but NO behavioural lift.
            # With the bias now genuinely reaching the argmin, this is a real
            # weaken of the behavioural claim (distinct from the 624a starved
            # lever, which C5 would have caught).
            evidence_direction = "weakens"
            per_claim = {"MECH-320": "weakens", "ARC-068": "weakens"}
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
            "V_T_FLOOR": V_T_FLOOR,
            "USE_MODULATORY_AUTHORITY": USE_MODULATORY_AUTHORITY,
            "MODULATORY_AUTHORITY_GAIN": MODULATORY_AUTHORITY_GAIN,
            "SALAMONE_DAMAGE_INCREMENT": SALAMONE_DAMAGE_INCREMENT,
            "SALAMONE_FAILURE_PROB_SCALE": SALAMONE_FAILURE_PROB_SCALE,
        },
        "summary": {
            "scenario": (
                "Well-fed-safe-familiar regime (8x8 grid, 1 hazard, 3 resources, "
                "use_proxy_fields=True, action_dim=5 with noop_class=4). 3 matched "
                "seeds. limb_damage_enabled=True on ALL arms (body_obs_dim=17 "
                "uniform observation space); the Niv-vs-Salamone manipulation is "
                "movement-cost-ONLY via damage_increment / failure_prob_scale. "
                "P0 warmup 100 ep (vigor OFF + free-movement env in all arms, env "
                "stepping only -- no gradient training), P1 eval 30 ep x 200 steps "
                "(arm flag + movement-cost config toggled). 4-arm Niv-vs-Salamone "
                "dissociation: ARM_0 baseline; ARM_1 vigor + free movement (Niv "
                "kernel); ARM_2 vigor + Salamone-style elevated movement cost; "
                "ARM_3 baseline + Salamone env. The modulatory-bias-selection-"
                "authority substrate (use_modulatory_selection_authority=True, "
                "gain=0.5) is ON in ALL arms -- the load-bearing change vs 624a, "
                "giving the MECH-320 w_passive bias bounded authority over the "
                "E3.select argmin (no-op where no modulatory bias exists, so "
                "ARM_0 / ARM_3 are bit-identical to 624a). Tests the R4 verdict: "
                "ARC-068 / MECH-320 w_passive must remain insensitive to "
                "parametric movement cost (Niv 2007 opportunity-cost-on-time), "
                "distinct from MECH-258 / SD-032b dacc_effort_cost (Salamone & "
                "Correa 2003). Forced-vigor probe v_t_floor=0.05 (V3-EXQ-549 fix). "
                "Supersedes V3-EXQ-624a (zeroed-lever FAIL)."
            ),
            "interpretation": (
                f"action_density: ARM_0={mean_density_by_arm['ARM_0_baseline']:.4f}; "
                f"ARM_1={mean_density_by_arm['ARM_1_vigor_niv']:.4f}; "
                f"ARM_2={mean_density_by_arm['ARM_2_vigor_salamone']:.4f}; "
                f"ARM_3={mean_density_by_arm['ARM_3_baseline_salamone']:.4f}. "
                f"C1 Niv lift (ARM_1 - ARM_0) mean={mean_c1_lift:+.4f}: "
                f"{'PASS' if c1_pass_all else 'FAIL'} (>= {C1_LIFT_MIN}). "
                f"C2 dissociation |delta_arm2 - delta_arm1|/max(...) mean={mean_c2_delta_diff:.3f}: "
                f"{'PASS' if c2_pass_all else 'FAIL'} (< {C2_DISSOCIATION_TOL}). "
                f"C3 gate_product ARM_1 mean={mean_gate_arm1:.3f}: "
                f"{'PASS' if c3_pass_all else 'FAIL'} (>= {C3_GATE_PRODUCT_MIN}). "
                f"C5 authority ARM_1 active_frac={mean_auth_active_frac_arm1:.3f} "
                f"range={mean_auth_range_arm1:.3e} raw_range={mean_raw_range_arm1:.3e}: "
                f"{'PASS' if c5_pass_all else 'FAIL'}. "
                f"C4 well-fed-safe no-op penalty observable: "
                f"{'PASS' if c4_pass_all else 'FAIL'} (informative). "
                f"Within-ARM_1 Pearson r(v_t, density) mean={mean_pearson_arm1:+.3f}. "
                f"Outcome: {outcome}, evidence_direction={evidence_direction}."
            ),
            "pairwise_deltas": {
                "per_seed_c1_niv_lift": per_seed_c1_lift,
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
            "per_seed_c1_lift": per_seed_c1_lift,
            "per_seed_c2_delta_diff": per_seed_c2_delta_diff,
            "per_seed_c1_pass": per_seed_c1_pass,
            "per_seed_c2_pass": per_seed_c2_pass,
            "per_seed_c3_pass": per_seed_c3_pass,
            "per_seed_c4_pass": per_seed_c4_pass,
            "per_seed_c5_pass": per_seed_c5_pass,
            "c1_pass_all_seeds": c1_pass_all,
            "c2_pass_all_seeds": c2_pass_all,
            "c3_pass_all_seeds": c3_pass_all,
            "c4_pass_all_seeds": c4_pass_all,
            "c5_pass_all_seeds": c5_pass_all,
            "overall_pass": overall_pass,
        },
        "config": {
            "seeds": SEEDS,
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
            "discriminative test on the MECH-320 w_passive implementation. CLEAN "
            "RETEST of V3-EXQ-624a, which FAILed because the vigor score-bias had "
            "ZERO authority over the E3.select argmin (failure_record signature: "
            "action_density lift ARM_1 - ARM_0 = 0.0, both 0.865, gate_product "
            "1.0, v_t 0.05 -- the zeroed lever, not a scientific null; 2026-06-03 "
            "cluster autopsy failure_autopsy_604a-624a-630). The load-bearing "
            "change is use_modulatory_selection_authority=True (gain=0.5): the "
            "modulatory-bias-selection-authority substrate (landed 2026-06-03; "
            "float32-cancellation amend V3-EXQ-643a 2026-06-06) rescales the "
            "combined modulatory contribution to gain * raw_score_range so the "
            "MECH-320 w_passive bias reaches the argmin without touching primary "
            "scores. ON in all four arms (no-op where no modulatory bias exists, "
            "so ARM_0 / ARM_3 are bit-identical to 624a; vigor stays the single "
            "manipulated variable). C5 records the E3 selection-authority "
            "diagnostics (modulatory_authority_active_frac / range / "
            "scale_factor + e3_raw_score_range) so a C1 lift of 0 can be "
            "adjudicated as a genuine scientific null (lever fired, bounded "
            "scores) vs a starved lever / exploded scores (non_contributory). "
            "624b does NOT train SD-056 online (no gradient training), so "
            "raw_score_range is expected O(1) and far below the 643 ~1e32 "
            "explosion that the C5 raw-range guard (1e6) watches for. "
            "EXP-0081 / EVB-0237 dispatch_mode=discriminative_pair. R3 verdict "
            "2026-05-16: slot-level ARC-068 registration preserved, implementation "
            "collapses into MECH-320. R4 verdict: effort-cost (MECH-258 / SD-032b) "
            "NOT to be collapsed with opportunity-cost-on-time (ARC-068 / MECH-320 "
            "w_passive). Pre-registered thresholds C1/C2/C3/C4/C5 set as constants "
            "before execution. PASS = C1 AND C2 AND C3 AND C5 across all seeds. "
            "5-row interpretation grid: (1) C5+C1+C2+C3 PASS -> ARC-068 + MECH-320 "
            "support; (2) C5+C1+C3 PASS, C2 FAIL -> ARC-068 weakens (effort-cost-"
            "like); (3) C5+C3 PASS, C1 FAIL -> MECH-320/ARC-068 weaken (authority "
            "reached argmin, v_t positive, no behavioural lift); (4) C5 FAIL -> "
            "non_contributory (lever starved / scores exploded; /diagnose-errors "
            "or /failure-autopsy); (5) C3 FAIL -> non_contributory "
            "(/diagnose-errors). Supersedes V3-EXQ-624a."
        ),
    }

    out_dir = os.path.abspath(
        os.path.join(REPO_ROOT, "..", "REE_assembly", "evidence", "experiments")
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=False,
        config=manifest.get("config"),
        seeds=SEEDS,
        script_path=Path(__file__),
    )

    # Overall outcome reported via emit_outcome() + manifest, not via a final
    # 'verdict:' line (which would over-count seeds*conditions = 12).
    print(f"outcome: {outcome}", flush=True)
    print(
        f"action_density "
        f"ARM_0={mean_density_by_arm['ARM_0_baseline']:.4f} "
        f"ARM_1={mean_density_by_arm['ARM_1_vigor_niv']:.4f} "
        f"ARM_2={mean_density_by_arm['ARM_2_vigor_salamone']:.4f} "
        f"ARM_3={mean_density_by_arm['ARM_3_baseline_salamone']:.4f} "
        f"c1_lift={mean_c1_lift:+.4f} c2_delta_diff={mean_c2_delta_diff:.3f} "
        f"gate_arm1={mean_gate_arm1:.3f} "
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
