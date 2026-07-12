"""V3-EXQ-624a: ARC-068 / MECH-320 opportunity-cost vs effort-cost dissociation.

Supersedes V3-EXQ-624 (crashed deterministically with a body-observation
dimension mismatch -- see "Fix (V3-EXQ-624a)" below). The scientific question
is unchanged from V3-EXQ-624.

Proposal: EXP-0081 / EVB-0237 (dispatch_mode=discriminative_pair). Behavioural
validation of the ARC-068 architectural slot (opportunity_cost_no_op_penalty)
as instantiated by MECH-320's w_passive term per the lit-pull R3 verdict
2026-05-16 (slot-level ARC-068 registration preserved; implementation collapses
into MECH-320 w_passive*v_t at e3.select).

Substrate prerequisites all landed:
  - MECH-320 substrate landed 2026-05-10 (ree_core/policy/tonic_vigor.py).
  - V3-EXQ-547 substrate-readiness diagnostic PASS 2026-05-10 (6/6 sub-tests).
  - ARC-068 lit-pull synthesis lit_conf ~0.78 supports-direction (Niv 2007 +
    Kurzban 2013 + Kolling 2016 + Shenhav 2016 + Salamone & Correa 2003 +
    Constantino & Daw 2015). R3 verdict: collapse-at-implementation-layer
    licensed (additive Niv form). R4 verdict: effort-cost vs opportunity-cost
    separation REQUIRED (Salamone & Correa 2003 dissociation; ARC-068 must NOT
    absorb into MECH-258 / SD-032b dacc_effort_cost machinery).

Fix (V3-EXQ-624a) -- root cause and the redesign that resolves it:
  V3-EXQ-624 built the agent ONCE per arm from the P0 warmup env, which had
  limb_damage_enabled=False (CausalGridWorldV2.body_obs_dim == 12). The
  Salamone arms (ARM_2 / ARM_3) then switched their P1 measurement env to
  limb_damage_enabled=True, which makes body_obs_dim == 17 (SD-022: +damage[4]
  +residual_pain). The 17-dim body observation hit the agent's 12-input body
  encoder on the first P1 tick -> RuntimeError: mat1 and mat2 shapes cannot be
  multiplied (1x17 and 12x12). The agent's encoder dimensions are fixed at
  construction; you cannot change body_obs_dim mid-agent-life.

  This was not only a crash: limb_damage_enabled toggled BOTH the movement-cost
  mechanism AND the observation dimensionality (12 vs 17). That conflated the
  Niv-vs-Salamone manipulation (which must be movement-cost-ONLY, per the
  Salamone & Correa 2003 effort-cost dissociation) with an observation-space
  change. A C2 dissociation failure under V3-EXQ-624's design could have been
  attributed to the 5 extra observation dimensions rather than to movement cost.

  624a holds the observation space CONSTANT across all arms and both phases:
  limb_damage_enabled=True everywhere (body_obs_dim == 17 uniformly), and the
  movement cost is the ONLY manipulated variable, carried by damage_increment
  and failure_prob_scale. The Niv side (and the P0 warmup in every arm) sets
  damage_increment=0.0 and failure_prob_scale=0.0 -> damage never accrues
  (causal_grid_world.py: damage += damage_increment*harm_mag) and movement
  never fails (P(fail) = damage * failure_prob_scale = 0), so movement is FREE
  and the damage observation channels stay at zero. The Salamone side sets the
  elevated values so movement is expensive. Net effect: the agent is built from
  a 17-dim env and sees 17-dim body observations throughout (no crash), and the
  Niv-vs-Salamone contrast is a clean single-variable movement-cost manipulation.

Why a new experiment instead of citing V3-EXQ-549:
  V3-EXQ-549 (ARC-066 / MECH-320 discriminative pair, 2026-05-11) measured
  ARM_OFF vs ARM_ON (vigor disabled vs additive form) under default
  TonicVigorConfig. Outcome FAIL -> reclassified non_contributory: v_t stayed
  at exactly 0.0 across all 120 measurement windows because the gate_drive
  product collapsed (drive_level reached gate_drive_max=0.7 under default env
  config). Substrate gates were correct (C3 PASS); the EWMA path simply never
  produced positive v_t under the validation env. V3-EXQ-549's recommendation
  was: "forced-vigor probe (set tonic_vigor_floor > 0 or force v_raw=1.0) to
  confirm the downstream score-bias -> action-selection path before re-running
  the discriminative pair." ARC-068 itself has 0 experimental entries to date;
  EXP-0081 fires on active_conflict, low_exp_conf, missing_experimental_evidence,
  lit_only_above_cap, synthetic_signals_only, insufficient_experimental_replication,
  directional_conflict_alert.

ARC-068 architectural prediction (R4 verdict: opportunity-cost-on-time, NOT
effort-cost-on-movement):
  In the well-fed-safe-familiar regime where movement is cheap, the no-op
  penalty (MECH-320 w_passive * v_t) should still scale with the EWMA over
  realised reward (Niv 2007 average reward rate), NOT with the per-step
  movement cost. This is the falsifiable discriminative prediction: ARC-068
  must remain insensitive to parametric increases in movement-cost-on-action
  (Salamone & Correa 2003 effort cost), while continuing to fire when v_t is
  positive (forced via v_t_floor to bypass V3-EXQ-549's calibration failure).

Four-arm design (per proposal acceptance checks adjusted for the dissociation:
exactly one discriminative pair plus two foil arms isolating the Niv vs
Salamone axis; no broad profile sweep):
  ARM_0_baseline:
    use_tonic_vigor=False, free-movement env (limb_damage_enabled=True with
    damage_increment=0.0, failure_prob_scale=0.0). Reference action_density.
  ARM_1_vigor_niv:
    use_tonic_vigor=True, additive form, v_t_floor=0.05 (forced-vigor probe
    per V3-EXQ-549's prescribed fix; small positive floor that survives
    gate-drive collapse), free-movement env.
    Predicts elevated action_density (Niv-style no-op penalty on cheap
    movement).
  ARM_2_vigor_salamone:
    use_tonic_vigor=True, additive form, v_t_floor=0.05, env with elevated
    movement cost (damage_increment=0.30, failure_prob_scale=0.6).
    Movement is now expensive. Niv kernel predicts: action_density delta
    from ARM_0 matches ARM_1's delta (no-op penalty independent of movement
    cost). Salamone kernel predicts: action_density delta from ARM_0 is
    suppressed or reversed (no-op penalty conflated with effort cost).
  ARM_3_baseline_salamone:
    use_tonic_vigor=False, same elevated-movement-cost env as ARM_2.
    Movement-cost-only control: confirms the baseline effort-cost response
    without ARC-068 firing.

Protocol:
  P0 warmup (100 ep, vigor OFF + free-movement env in all arms): identical
    baseline policy checkpoint per seed. Vigor is target-free and its EWMA
    starts at zero; warmup with vigor OFF + the SAME free-movement env (and
    the SAME 17-dim observation space) in all arms ensures the policy learns
    on identical data so any P1 difference is attributable to the arm
    intervention not training divergence.
  P1 measurement (30 ep x 200 steps): arm flag and env movement-cost config
    toggled. ARM_0 and ARM_3 keep vigor=False; ARM_1 and ARM_2 instantiate
    TonicVigor with v_t_floor=0.05. ARM_0 and ARM_1 keep free movement; ARM_2
    and ARM_3 use the Salamone-style elevated-movement-cost env. The
    observation space is identical (17-dim body) across all arms and phases.

Environment (well-fed-safe-familiar regime, matched across arms by seed):
  CausalGridWorldV2 size=8, num_hazards=1, num_resources=3, action_dim=5
  (action 0 = up, 1 = down, 2 = left, 3 = right, 4 = noop).
  ALL arms: limb_damage_enabled=True (body_obs_dim=17, uniform observation
    space). Free-movement arms / P0: damage_increment=0.0,
    failure_prob_scale=0.0 (movement never fails, damage channels stay zero).
  ARM_2 / ARM_3 P1: damage_increment=0.30, failure_prob_scale=0.6 (movement
    fails with probability damage * 0.6 per limb; cumulative damage accrues on
    hazard contact). Heal_rate unchanged from default.
  use_proxy_fields=True throughout (small encoder, fast Mac runtime).

Metrics (per arm per seed, P1 only):
  action_density: mean over P1 ticks of [argmax(action) != NOOP_CLASS].
    The behavioural signature: opportunity-cost no-op penalty lifts this.
    The Niv-vs-Salamone dissociation is computed on the cross-arm deltas.
  v_t_window / action_density_window: per-50-step windows in ARM_1 and
    ARM_2 only. Pearson r between window-mean v_t and window-mean
    action_density measures the within-arm Niv reward-rate scaling
    signature.
  gate_product_mean: mean of gate_energy * gate_drive * gate_pe in
    ARM_1 / ARM_2. Confirms gates open enough for downstream firing.

Pre-registered acceptance thresholds (defined here, NOT inferred post-hoc):
  C1 action_density lift (vigor effect): mean(action_density_ARM_1) -
      mean(action_density_ARM_0) >= C1_LIFT_MIN (default 0.03; 3pp).
      Paired by seed. The basic MECH-320 firing test.
  C2 Niv-vs-Salamone dissociation (R4 verdict test): under the Niv kernel,
      the ARM_1 - ARM_0 lift should match the ARM_2 - ARM_3 lift within
      C2_DISSOCIATION_TOL (default 0.50 of the ARM_1 - ARM_0 lift). i.e.
      adding movement cost (ARM_3 vs ARM_0) and adding vigor under
      movement cost (ARM_2 vs ARM_3) should produce a vigor-lift that
      mirrors the no-movement-cost case. If the implementation is secretly
      effort-cost-like, the ARM_2 - ARM_3 lift should be suppressed or
      reversed relative to ARM_1 - ARM_0.
  C3 gate sanity: mean(gate_product) > C3_GATE_PRODUCT_MIN (default 0.5)
      in ARM_1. With v_t_floor=0.05 the substrate fires regardless of
      gate state, but C3 confirms the regular gate path is also clean
      under the validation env.
  C4 well-fed-safe regime no-op penalty observable (Salamone discrimination,
      informative-only): action_density in ARM_1 must exceed action_density
      in ARM_0 even with movement cost near zero. Salamone's framework
      predicts no penalty here; ARC-068 predicts a positive penalty.
      Equivalent to C1 in this design but stated explicitly as the
      Salamone-falsifier criterion. Recorded for transparency; not a
      hard PASS gate (C1 covers the same signal numerically).

PASS = C1 AND C2 AND C3 across all seeds (informative-only C4 logged
separately).

4-row interpretation grid:
  (1) C1 PASS AND C2 PASS AND C3 PASS:
      ARC-068 supports + MECH-320 supports. conflict_ratio drops; the slot
      preservation is vindicated. Surface to /governance for next-cycle
      consideration of promotion candidate -> provisional.
  (2) C1 PASS AND C3 PASS AND C2 FAIL with ARM_2 - ARM_3 lift << ARM_1 - ARM_0 lift:
      ARC-068 weakens. The implementation behaves as effort-cost-like
      (movement-cost-sensitive). Contradicts R4 verdict; surface to
      /governance for the supersession question (ARC-068 collapse into
      MECH-258 / SD-032b effort-cost machinery).
  (3) C1 FAIL with C3 PASS (v_t fires, no behavioural lift):
      Implementation does not match the architectural claim. The gates open
      and v_t is positive but the score-bias does not propagate to action
      selection. Route to /failure-autopsy; substrate-side wiring
      investigation.
  (4) C3 FAIL (substrate does not fire even with v_t_floor=0.05):
      Calibration / wiring problem deeper than the V3-EXQ-549 finding.
      Route to /diagnose-errors.

experiment_purpose = "evidence" (tests the load-bearing claim of MECH-320
under the ARC-068 architectural framing).

claim_ids = ["MECH-320", "ARC-068"] with evidence_direction_per_claim.
MECH-258 and SD-032b are NOT in claim_ids: they are foils architecturally
dissociated from, not tested directly. We do not measure dACC bundle
output or precision-weighted PE; we measure whether MECH-320's no-op
penalty is movement-cost-sensitive (Salamone) or movement-cost-insensitive
(Niv).

Run with:
  /opt/local/bin/python3 experiments/v3_exq_624a_arc068_mech320_niv_salamone_dissociation.py
or:
  /opt/local/bin/python3 experiments/v3_exq_624a_arc068_mech320_niv_salamone_dissociation.py --dry-run
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
EXPERIMENT_TYPE = "v3_exq_624a_arc068_mech320_niv_salamone_dissociation"
SUPERSEDES = "v3_exq_624_arc068_mech320_niv_salamone_dissociation"
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
# so the Salamone-arm movement-cost env switch (the V3-EXQ-624 crash site) is
# actually exercised. V3-EXQ-624's dry-run broke at ep>=2, which never reached
# P1 -- that is why the body_obs_dim mismatch slipped past the smoke test.
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
    damage_increment and failure_prob_scale -- NOT by toggling limb_damage on
    and off (which would also change the observation dimensionality and was the
    V3-EXQ-624 crash + obs-space confound). With both cost knobs at 0.0,
    movement never fails (P(fail) = damage * failure_prob_scale = 0) and damage
    never accrues (damage += damage_increment * harm_mag = 0), so movement is
    free and the damage observation channels stay at zero -- behaviourally the
    well-fed-safe-familiar regime, but with the 17-dim observation space the
    Salamone arms also use.

    expensive_movement=False (free): the Niv regime and ALL P0 warmups.
    expensive_movement=True: the Salamone elevated-movement-cost P1 regime.
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

    body_obs_dim / world_obs_dim are read from the env, so the agent's encoders
    are sized to match the env the agent will actually receive observations
    from. Because make_env always enables limb_damage, env.body_obs_dim == 17
    for every arm and every phase -- the agent is dimension-consistent across
    the P0 warmup and the P1 measurement env switch.

    Uses v_t_floor=V_T_FLOOR when vigor is on (forced-vigor probe per the
    V3-EXQ-549 prescribed fix).

    tonic_vigor_noop_class is set to CausalGridWorldV2's action-4 no-op rather
    than the TonicVigorConfig default (0), which would otherwise bias against
    "up" instead of against passivity.
    """
    kwargs = dict(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=ACTION_DIM,
        self_dim=16,
        world_dim=16,
        use_tonic_vigor=vigor_on,
        tonic_vigor_noop_class=NOOP_CLASS,
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

    P0 warmup uses the FREE-movement env (no Salamone movement cost) and vigor
    OFF in ALL arms so the policy learns on identical data. P1 measurement
    toggles the vigor flag and the movement-cost env config per arm. The
    observation space (body_obs_dim=17) is constant across P0/P1 and all arms.
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
        f"V3-EXQ-624a ARC-068 / MECH-320 Niv-vs-Salamone dissociation",
        flush=True,
    )
    print(
        f"seeds={SEEDS} p0_ep={P0_WARMUP_EPISODES} p1_ep={P1_EVAL_EPISODES} "
        f"steps={STEPS_PER_EPISODE} grid={GRID_SIZE}x{GRID_SIZE} "
        f"hazards={N_HAZARDS} resources={N_RESOURCES} action_dim={ACTION_DIM} "
        f"v_t_floor={V_T_FLOOR} dry_run={dry_run}",
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
            # Per-arm-seed verdict line: PASS iff the run executed end-to-end
            # (P1 ticks accumulated, or dry-run). Scientific PASS/FAIL is
            # computed at overall level. seeds * conditions = 3 * 4 = 12
            # verdict lines total.
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
    for i in range(n_seeds):
        d0 = results_by_arm["ARM_0_baseline"][i]["action_density"]
        d1 = results_by_arm["ARM_1_vigor_niv"][i]["action_density"]
        d2 = results_by_arm["ARM_2_vigor_salamone"][i]["action_density"]
        d3 = results_by_arm["ARM_3_baseline_salamone"][i]["action_density"]
        gp1 = results_by_arm["ARM_1_vigor_niv"][i]["gate_product_mean"]

        # C1: Niv lift (vigor effect with movement cheap)
        c1_lift = d1 - d0
        per_seed_c1_lift.append(c1_lift)
        per_seed_c1_pass.append(c1_lift >= C1_LIFT_MIN)

        # C2: dissociation. The Salamone-side vigor lift is (d2 - d3); the
        # Niv-side vigor lift is (d1 - d0). Under the Niv kernel both should
        # match; the dissociation metric measures their relative discrepancy.
        salamone_lift = d2 - d3
        denom = max(abs(c1_lift), 1e-6)
        delta_diff = abs(salamone_lift - c1_lift) / denom
        per_seed_c2_delta_diff.append(delta_diff)
        # C2 PASS: dissociation within tolerance AND salamone_lift positive
        # (i.e. vigor still produces a lift in the high-movement-cost env;
        # if it produced zero or negative lift even though Niv-side is
        # positive, that signals effort-cost-like behaviour).
        per_seed_c2_pass.append(
            delta_diff < C2_DISSOCIATION_TOL and salamone_lift >= 0.5 * c1_lift
        )

        # C3: gate sanity (ARM_1)
        per_seed_c3_pass.append(gp1 >= C3_GATE_PRODUCT_MIN)

        # C4: well-fed-safe regime no-op penalty observable (Salamone
        # discrimination; informative). Equivalent to C1 numerically here.
        per_seed_c4_pass.append(c1_lift >= C4_NO_OP_PENALTY_MIN)

    c1_pass_all = all(per_seed_c1_pass)
    c2_pass_all = all(per_seed_c2_pass)
    c3_pass_all = all(per_seed_c3_pass)
    c4_pass_all = all(per_seed_c4_pass)
    overall_pass = c1_pass_all and c2_pass_all and c3_pass_all

    outcome = "PASS" if overall_pass else "FAIL"
    if overall_pass:
        evidence_direction = "supports"
        per_claim = {"MECH-320": "supports", "ARC-068": "supports"}
    else:
        # Interpretation-grid routing (computed for note, applied via
        # per-claim direction below).
        if not c3_pass_all:
            # Row 4: substrate did not fire (even with v_t_floor=0.05).
            evidence_direction = "non_contributory"
            per_claim = {
                "MECH-320": "non_contributory",
                "ARC-068": "non_contributory",
            }
        elif c1_pass_all and not c2_pass_all:
            # Row 2: Salamone-like dissociation failure. The implementation
            # behaves as movement-cost-sensitive. ARC-068 weakens; MECH-320
            # is the load-bearing target of the weaken (the implementation
            # claim that w_passive*v_t is opportunity-cost-on-time).
            evidence_direction = "weakens"
            per_claim = {"MECH-320": "weakens", "ARC-068": "weakens"}
        elif not c1_pass_all and c3_pass_all:
            # Row 3: v_t fires but no behavioural lift; implementation does
            # not match claim. Not directly weakens (script-side problem).
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
            "V_T_FLOOR": V_T_FLOOR,
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
                "P0 warmup 100 ep (vigor OFF + free-movement env in all arms), P1 "
                "eval 30 ep x 200 steps (arm flag + movement-cost config toggled). "
                "4-arm Niv-vs-Salamone dissociation: ARM_0 baseline; ARM_1 vigor + "
                "free movement (Niv kernel); ARM_2 vigor + Salamone-style elevated "
                "movement cost; ARM_3 baseline + Salamone env (movement-cost-only "
                "control). Tests the R4 verdict: ARC-068 / MECH-320 w_passive must "
                "remain insensitive to parametric movement cost (Niv 2007 "
                "opportunity-cost-on-time), distinct from MECH-258 / SD-032b "
                "dacc_effort_cost (Salamone & Correa 2003). Forced-vigor probe "
                "v_t_floor=0.05 implements V3-EXQ-549 prescribed fix bypassing "
                "gate-drive collapse. Supersedes V3-EXQ-624 (body_obs_dim 12/17 "
                "crash + obs-space confound)."
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
        },
        "criteria": {
            "n_seeds": n_seeds,
            "per_seed_c1_lift": per_seed_c1_lift,
            "per_seed_c2_delta_diff": per_seed_c2_delta_diff,
            "per_seed_c1_pass": per_seed_c1_pass,
            "per_seed_c2_pass": per_seed_c2_pass,
            "per_seed_c3_pass": per_seed_c3_pass,
            "per_seed_c4_pass": per_seed_c4_pass,
            "c1_pass_all_seeds": c1_pass_all,
            "c2_pass_all_seeds": c2_pass_all,
            "c3_pass_all_seeds": c3_pass_all,
            "c4_pass_all_seeds": c4_pass_all,
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
            "Supersedes V3-EXQ-624: that run crashed (RuntimeError 1x17 vs "
            "12x12) because the agent was built from a limb_damage-OFF P0 env "
            "(body_obs_dim=12) while the Salamone arms switched their P1 env to "
            "limb_damage-ON (body_obs_dim=17). 624a enables limb_damage on ALL "
            "arms (uniform 17-dim observation space) and carries the "
            "Niv-vs-Salamone manipulation through damage_increment / "
            "failure_prob_scale only -- fixing the crash AND removing the "
            "observation-dimensionality confound from the dissociation test. "
            "EXP-0081 / EVB-0237 dispatch_mode=discriminative_pair. ARC-068 "
            "lit-pull R3 verdict 2026-05-16: slot-level registration preserved, "
            "implementation collapses into MECH-320. R4 verdict: effort-cost "
            "(MECH-258 / SD-032b) NOT to be collapsed with opportunity-cost-on-"
            "time (ARC-068 / MECH-320 w_passive). Substrate landed 2026-05-10; "
            "V3-EXQ-547 substrate-readiness PASS; V3-EXQ-549 discriminative-pair "
            "non_contributory (v_t=0.0 calibration failure). This experiment uses "
            "v_t_floor=0.05 forced-vigor probe to bypass gate-drive collapse and "
            "adds the Niv-vs-Salamone dissociation arms (ARM_2 + ARM_3) to "
            "specifically test the R4 verdict. Pre-registered thresholds C1/C2/"
            "C3/C4 set as constants in this script before execution. 4-row "
            "interpretation grid: (1) C1+C2+C3 PASS -> ARC-068 supports + "
            "MECH-320 supports (slot vindicated); (2) C1+C3 PASS + C2 FAIL "
            "-> ARC-068 weakens (effort-cost-like; surface supersession question "
            "for /governance); (3) C1 FAIL + C3 PASS -> implementation does not "
            "match claim, route to /failure-autopsy; (4) C3 FAIL -> substrate "
            "does not fire, route to /diagnose-errors."
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
        f"gate_arm1={mean_gate_arm1:.3f}",
        flush=True,
    )
    print(f"Result written to: {out_path}", flush=True)
    return outcome, out_path


if __name__ == "__main__":
    _outcome, _manifest_path = main()
    emit_outcome(outcome=_outcome, manifest_path=_manifest_path)
    sys.exit(0)
