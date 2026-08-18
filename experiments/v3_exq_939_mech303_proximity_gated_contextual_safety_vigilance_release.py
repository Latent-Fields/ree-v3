"""V3-EXQ-939: MECH-303 behavioural retest -- proximity-gated LIVE contextual-safety
accumulation and its effect on BACKGROUND AVOIDANCE-COMMITMENT (vigilance) release.

PURPOSE. MECH-303's primary pre-registered output (claims.yaml) is: "lowers the background
avoidance commitment threshold in familiar safe environments -- a diffuse reduction in
vigilance tone driven by context, not by a specific cue signal." This is the owed
behavioural retest (`pending_retest_after_substrate`), unblocked by
SD-MECH303-THRESHOLD-SOURCING landing 2026-08-14 (ree-v3 b257e7ad14).

WHY THIS IS A DIFFERENT EXPERIMENT FROM ITS TWO PREDECESSORS. Both prior runs left the
behavioural question untouched, for different reasons:

  V3-EXQ-930 (confirmed failure_autopsy_V3-EXQ-930_2026-08-16, governance-applied
  2026-08-16, evidence_direction non_contributory) validated the DEDICATED SIGNAL at the
  SIGNAL LAYER only. It NEVER ENABLED THE MECH-303 GATE: use_contextual_safety_terrain
  stayed at its False default, contextual_safety_gate_source stayed 'z_harm_a', the
  accumulate_safety block (agent.py:5166-5212) did not execute, the agent was UNTRAINED
  (fresh REEAgent per cell, no training), and no vigilance/avoidance readout was taken.
  930 measured two INPUTS to a switched-off gate. It is NOT cited here as support.

  V3-EXQ-764 is the promote-to-active falsifier: BUILT + validated, HELD (never queued)
  since 2026-07-15. Its blocker is the z_world SAFE-vs-UNSAFE separability ceiling (SD-008
  under-differentiation, rank AUC ~0.83): its context-specificity DV compares a terrain
  TAUGHT IN SAFE and READ IN UNSAFE, so ALL of its discrimination has to come out of
  z_world KEYING, and it capped at 0.292 against a 0.34 margin. That blocker is UNCHANGED
  -- SD-MECH303-THRESHOLD-SOURCING fixed the WRITE gate ("is harm absent right now?"), not
  the READ keying. This experiment is therefore NOT 764 re-queued, and must not be read as
  clearing 764: it is a design that routes its load-bearing contrasts through ACCUMULATION
  HISTORY (a write-side variable the dedicated signal makes context-differential for the
  first time) instead of through cross-context z_world keying. 764 stays HELD.

WHAT THE NEW SIGNAL MAKES POSSIBLE. Under the old z_harm_a gate the accumulate decision was
blind to spatial context (V3-EXQ-917: AUC <= 0.52 at all 18 swept thresholds; SD-011: the
affective-harm encoder reads ~0.547 safe vs ~0.542 unsafe), so a safe and a hazardous
context accumulated IDENTICALLY and no accumulation-side contrast existed at all. 764 had to
FREEZE accumulation for its test window for exactly this reason. With
contextual_safety_gate_source="proximity_signal" the gate reads a dedicated anticipatory
hazard-proximity EMA that is ~0.0 in hazard-free contexts and ~0.68-0.87 in dense-hazard
contexts, so accumulation now runs LIVE and is context-differential.

DESIGN -- 2x2, EVERY ARM TESTED IN ITS OWN EXPOSURE CONTEXT.
Crossing {exposure context: SAFE (num_hazards=0) / HAZARD (num_hazards=8)} x {accumulation
gate: NATURAL (shipped default threshold 0.25) / FORCED (threshold overridden)}:

  A_safe_gate_natural         SAFE   context, prox_thresh=0.25 -> gate OPEN  -> ACCUMULATES
  B_safe_gate_forced_closed   SAFE   context, prox_thresh=0.0  -> gate CLOSED-> no accumulation
  C_hazard_gate_natural       HAZARD context, prox_thresh=0.25 -> gate CLOSED-> no accumulation
  D_hazard_gate_forced_open   HAZARD context, prox_thresh=2.0  -> gate OPEN  -> ACCUMULATES

A/B and C/D are WITHIN-CONTEXT contrasts, so neither depends on cross-context z_world
keying. D is the yoked-accumulation control and is what makes the causal reading clean: it
asks whether the terrain CAN read above the release threshold IN THE HAZARDOUS CONTEXT when
accumulation is permitted there. If it can, then C's non-release is caused by the gate
WITHHOLDING accumulation, not by the hazardous context suppressing the read -- which is
precisely the confound that caps 764.

NOTE ON THE ABSENT FLAG-OFF ARM. use_contextual_safety_terrain=False is deliberately NOT an
arm. That ablation is near-analytic (the release block is flag-gated, so it returns 0 by
construction) and 764 already measured it at a 0.748 gap. B -- terrain flag ON, read path
live, accumulation gate held shut -- is the informative ablation of the same thing.

DVs (pre-registered constants below; PASS = DV1 AND DV2 AND DV3, plain conjunction, also
recorded as `combination_rule`):
  DV1 accumulation-necessity  (context held SAFE):   release_A - release_B >= DV_MARGIN
  DV2 accumulation-is-the-cause (context held HAZARD): release_D - release_C >= DV_MARGIN
  DV3 gate-natural context appropriateness:            release_A - release_C >= DV_MARGIN

LOAD-BEARING: DV2 and DV3. DV1 is reported and gates the PASS but is marked NOT
load-bearing: forcing the gate shut is a config override, so a large A-B gap is close to
structural. DV3 is the genuinely non-analytic one -- whether the SHIPPED default threshold
(0.25) naturally withholds accumulation in a hazardous context by enough to change the
behavioural readout is an empirical question about the proximity EMA's realized
distribution under a moving agent, the RBF read at the test z_world, and the release
threshold. It could have come out either way.

DV-SYMMETRY DECLARATION (one line per arm; the V3-EXQ-604c class). The DV, release_rate, is
a count of threshold crossings of `residue_field.evaluate_safety(z_world).mean()` against a
FIXED pre-registered contextual_safety_release_threshold. Its symmetry group is (i) any
transform of the safety scalar that preserves each trial's position relative to that fixed
threshold, and (ii) permutation of test trials. Every arm's manipulation changes the
accumulated terrain's MAGNITUDE (by ~accum_weight x num_safety_steps at the test z_world),
which moves it ACROSS the fixed threshold rather than rescaling it -- so no arm's
manipulation is invariant under that group:
  A: accumulates ~EXPOSURE_STEPS increments -> terrain magnitude rises above threshold.
  B: accumulates zero -> terrain stays at its zero-init, structurally below threshold.
  C: accumulates only on the few ticks the agent strays far from hazards -> below threshold.
  D: accumulates ~EXPOSURE_STEPS increments IN THE HAZARD CONTEXT -> above threshold.
The manipulation is an additive change to the very quantity the threshold is applied to; it
is not a broadcast constant across candidates, not a monotone rescaling of a rank-based DV,
and not a permutation of interchangeable units.

READINESS / NON-VACUITY (self-routes substrate_not_ready_requeue -> non_contributory, NEVER
a false weakens; the 759/688 lesson, and the 916 num_safety_steps=0 lesson). Four
readiness-kind preconditions, all measured, all reported with numeric measured+threshold so
the indexer recomputes `met` itself:
  P1 arm_A_release_rate_positive_control -- arm A's OWN release_rate, the best case and the
     SAME STATISTIC (release_rate) every DV routes on. Below floor -> the substrate cannot
     express the mechanism even at its best, so nothing was measured.
  P2 arm_A_accumulation_occurred -- arm A's num_safety_steps. V3-EXQ-916 ran its entire
     experiment at num_safety_steps=0 and nobody noticed; this is that catch.
  P3 proximity_safe_below_gate / P4 proximity_hazard_above_gate -- the dedicated channel's
     realized means in the two contexts, against the SAME threshold the gate applies. If the
     channel does not straddle the gate, the manipulation never happened.
  P5 world_encoder_trained -- SD-070 weight-delta on split_encoder.world_encoder.

MANDATORY RECORDING (carried forward from failure_autopsy_V3-EXQ-930_2026-08-16's explicit
recording-gap finding: "the per-tick z_harm_a series and a damage-exposure readout existed
at run time and were discarded, so a per-seed-standardised discrimination cannot be computed
post hoc -- fold both into the recording spec of the owed retest ... do not re-run blind"):
  * per-tick z_harm_a norm SERIES per (arm, seed), retained in full, plus per-seed mean/std
    so the PER-SEED STANDARDISATION the autopsy requires is computable post hoc. The z_harm_a
    encoder is untrained and its between-seed init offset (sd ~6.4e-02) exceeds within-run
    modulation (~1e-03) by ~60x, so an ABSOLUTE-threshold read on the raw norm cannot
    discriminate; the standardised series is recorded so that question stays open rather
    than being re-run blind.
  * damage-exposure readout -- per-tick sum(env.limb_damage) plus a damaged-tick count, the
    manipulation check that the SD-022 damage pathway was actually exercised under
    limb_damage_enabled=True. 930 recorded none, so its z_harm_a null was unfalsifiable.
  * per-tick proximity-signal series, num_safety_steps / total_safety, and the terrain read
    at test, per cell.

SUBSTRATE. Run under the PRODUCTION scenario limb_damage_enabled=True (SD-022 damage-sourced
z_harm_a) -- the exact scenario in which z_harm_a cannot serve MECH-303's gate and the
dedicated signal must. alpha_world=0.9 for z_world fidelity (SD-008; V3-EXQ-760 precedent).
safety_terrain_bandwidth=0.03 (SD-067): the shared kernel_bandwidth 1.0 is ~15x too wide for
the z_world residual scale and saturates evaluate_safety. Confounding release paths held off
at test: urgency_interrupt_threshold=1e9 (MECH-091 abort), no goal, MECH-304 conditioned
store OFF -- so the ONLY release path in every arm is the MECH-303 contextual gate.

PHASED TRAINING. P0a SD-070 z_world encoder warmup (run_zworld_p0, 60 episodes -- the
validated operating point) on a DEDICATED warmup env; then the encoder is not stepped again
-- P1 exposure writes the terrain under torch.no_grad() (accumulate_safety), and P2 reads it
read-only. So no head is trained on a moving latent target.

GAP-I / COMMITMENT WALL. beta_gate.elevate() fires only from the F-selection/commitment path
this substrate cannot sustain (603h), so the commitment is HARNESS-INDUCED each test trial
(764 option i) and the DV is whether the MECH-303 gate RELEASES it. The DV therefore never
requires the substrate to SUSTAIN multi-step commitment, which is what
`feedback_dont_queue_commitment_dependent_behavioural` warns against.

GOV-REUSE-1 (Step 2.4). Decisive readout: `release_rate` under
use_contextual_safety_terrain=True + contextual_safety_gate_source="proximity_signal" with
LIVE accumulation. Checked V3-EXQ-930 (gate never enabled; no release readout at all),
V3-EXQ-917 (threshold sweep on the z_harm_a-sourced gate; no release readout), V3-EXQ-760
(read-only evaluate_safety discrimination; no release readout, no live proximity gate),
V3-EXQ-764 (never ran -- HELD, no manifest), V3-EXQ-916/916a (num_safety_steps=0 for the
entire run). The gate source knob did not exist before ree-v3 b257e7ad14 (2026-08-14), so no
manifest on any prior substrate_hash can carry this readout. Not recoverable -> ran fresh.
"""
import argparse
import random
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402

from _lib.arm_fingerprint import arm_cell  # noqa: E402
from _lib.capability_eval import RandomPolicy  # noqa: E402
from _lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402
from _lib.zworld_p0_warmup import run_zworld_p0  # noqa: E402
from _lib.zworld_encoder_guard import (  # noqa: E402
    ZWorldEncoderUntrainedError,
    assert_world_encoder_trained,
    latent_stack_snapshot,
)
from pack_writer import write_flat_manifest  # noqa: E402
from experiment_protocol import emit_outcome  # noqa: E402

_ZG = ZGoalStreamAccumulator()

EXPERIMENT_PURPOSE = "evidence"
EXPERIMENT_TYPE = "v3_exq_939_mech303_proximity_gated_contextual_safety_vigilance_release"
ARCH_EPOCH = "ree_hybrid_guardrails_v1"
CLAIM_IDS = ["MECH-303"]

DEVICE = torch.device("cpu")

# --- pre-registered constants -------------------------------------------------------------
N_SEEDS = 6
SEED_BASE = 1000
TEST_SEED_OFFSET = 500        # held-out test-env layout, distinct from the exposure layout
P0_EPISODES = 60              # SD-070 validated operating point
P0_STEPS_PER_EPISODE = 40
EXPOSURE_STEPS = 240          # P1 accumulation walk, per (arm, seed)
TOTAL_DENOM = P0_EPISODES + EXPOSURE_STEPS   # progress denominator M == queue episodes_per_run
N_TEST_TRIALS = 30
SETTLE_STEPS = 8              # settle the recurrent z_world into the test context

DV_MARGIN = 0.34              # every DV gap must clear this (764's registered margin)
READINESS_FLOOR = 0.34        # arm A positive-control release_rate floor (SAME statistic)
ACCUM_STEPS_FLOOR = 20        # arm A num_safety_steps floor (the V3-EXQ-916 zero-accumulation catch)
MIN_VALID_SEEDS = 4           # seeds clearing readiness needed for a non-vacuous verdict

# Gate-source knobs (SD-MECH303-THRESHOLD-SOURCING, ree-v3 b257e7ad14).
GATE_SOURCE = "proximity_signal"
PROX_THRESH_NATURAL = 0.25    # the SHIPPED default -- deliberately not retuned here
PROX_THRESH_FORCED_CLOSED = 0.0   # signal is clipped to [0,1] and >= 0 -> gate never opens
PROX_THRESH_FORCED_OPEN = 2.0     # signal is clipped to [0,1] -> gate always open

CONTEXTUAL_ACCUM_WEIGHT = 0.05
CONTEXTUAL_RELEASE_THRESHOLD = 0.5   # 764's calibrated value, between the safe and unsafe reads
SAFETY_TERRAIN_BW = 0.03             # SD-067 dedicated (tighter) RBF bandwidth

GRID_SIZE = 10
NUM_RESOURCES = 3
NUM_HAZARDS_SAFE = 0
NUM_HAZARDS_HAZARD = 8
HARM_OBS_A_DIM = 7            # harm_obs_a is 7-dim under limb_damage_enabled=True (SD-022)

_SUBSTRATE_COMMON = dict(
    alpha_world=0.9,                      # SD-008 z_world fidelity (760 precedent)
    use_harm_stream=True, harm_obs_dim=51,
    use_affective_harm_stream=True, harm_obs_a_dim=HARM_OBS_A_DIM,
)

ARMS = [
    "A_safe_gate_natural",
    "B_safe_gate_forced_closed",
    "C_hazard_gate_natural",
    "D_hazard_gate_forced_open",
]
_ARM_SPEC = {
    # arm -> (num_hazards, proximity_threshold)
    "A_safe_gate_natural":       (NUM_HAZARDS_SAFE,   PROX_THRESH_NATURAL),
    "B_safe_gate_forced_closed": (NUM_HAZARDS_SAFE,   PROX_THRESH_FORCED_CLOSED),
    "C_hazard_gate_natural":     (NUM_HAZARDS_HAZARD, PROX_THRESH_NATURAL),
    "D_hazard_gate_forced_open": (NUM_HAZARDS_HAZARD, PROX_THRESH_FORCED_OPEN),
}
SAFE_ARMS = ("A_safe_gate_natural", "B_safe_gate_forced_closed")
HAZARD_ARMS = ("C_hazard_gate_natural", "D_hazard_gate_forced_open")


def _env_kwargs(num_hazards):
    return dict(
        size=GRID_SIZE, num_hazards=num_hazards, num_resources=NUM_RESOURCES,
        use_proxy_fields=True,
        limb_damage_enabled=True, heal_rate=0.4,     # production scenario (SD-022 damage-sourced z_harm_a)
        safety_proximity_signal_enabled=True,        # emit the dedicated MECH-303 channel
    )


def _sense(agent, obs):
    """Forward the dedicated proximity channel into sense(). Without obs_safety_proximity the
    proximity gate simply never fires (agent.py has NO silent fallback to z_harm_a, by
    design), so this argument is load-bearing, not cosmetic."""
    sp = obs.get("safety_proximity_harm")
    return agent.sense(
        obs["body_state"], obs["world_state"],
        obs_harm=obs.get("harm_obs"), obs_harm_a=obs.get("harm_obs_a"),
        obs_safety_proximity=float(sp.item()) if sp is not None else None,
    )


def _act(agent, latent):
    ticks = agent.clock.advance()
    world_dim = agent.config.latent.world_dim
    e1_prior = agent._e1_tick(latent) if ticks.get("e1_tick") else torch.zeros(1, world_dim, device=DEVICE)
    candidates = agent.generate_trajectories(latent, e1_prior, ticks)
    action = agent.select_action(candidates, ticks)
    return int(action.argmax(dim=-1).item())


def _prox(obs):
    sp = obs.get("safety_proximity_harm")
    return float(sp.item()) if sp is not None else float("nan")


def _damage(env):
    return float(np.sum(env.limb_damage)) if getattr(env, "limb_damage_enabled", False) else 0.0


def _config_slice(arm):
    num_hazards, prox_thresh = _ARM_SPEC[arm]
    return {
        "arm": arm,
        "env_kwargs": _env_kwargs(num_hazards),
        # Declared explicitly even though env_kwargs already binds size/num_resources: this
        # fingerprint is CROSS-DRIVER reusable (include_driver_script_in_hash=False), so an
        # under-declared slice is a false-cache-HIT bug, not just a false MISS
        # (arm_reuse_fingerprint_plan.md 7b). SEED_BASE/TEST_SEED_OFFSET pick the exposure and
        # held-out test layouts and are otherwise invisible to the slice.
        "grid_size": GRID_SIZE,
        "num_resources": NUM_RESOURCES,
        "seed_base": SEED_BASE,
        "test_seed_offset": TEST_SEED_OFFSET,
        "substrate_common": dict(_SUBSTRATE_COMMON),
        "use_contextual_safety_terrain": True,
        "contextual_safety_gate_source": GATE_SOURCE,
        "contextual_safety_proximity_threshold": prox_thresh,
        "contextual_safety_accum_weight": CONTEXTUAL_ACCUM_WEIGHT,
        "contextual_safety_release_threshold": CONTEXTUAL_RELEASE_THRESHOLD,
        "safety_terrain_bandwidth": SAFETY_TERRAIN_BW,
        "urgency_interrupt_threshold": 1e9,
        "beta_gate_bistable": True,
        "p0_episodes": P0_EPISODES,
        "p0_steps_per_episode": P0_STEPS_PER_EPISODE,
        "exposure_steps": EXPOSURE_STEPS,
        "n_test_trials": N_TEST_TRIALS,
        "settle_steps": SETTLE_STEPS,
    }


def _build(arm, seed):
    num_hazards, prox_thresh = _ARM_SPEC[arm]
    env = CausalGridWorldV2(seed=SEED_BASE + seed, **_env_kwargs(num_hazards))
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim, world_obs_dim=env.world_obs_dim, action_dim=5,
        use_contextual_safety_terrain=True,
        contextual_safety_gate_source=GATE_SOURCE,
        contextual_safety_proximity_threshold=prox_thresh,
        contextual_safety_accum_weight=CONTEXTUAL_ACCUM_WEIGHT,
        contextual_safety_release_threshold=CONTEXTUAL_RELEASE_THRESHOLD,
        safety_terrain_bandwidth=SAFETY_TERRAIN_BW,
        **_SUBSTRATE_COMMON,
    )
    # from_dims round-trip is verified at runtime below (from_dims silently swallows unknown
    # kwargs, so a renamed knob would otherwise leave the gate at its default -- the
    # reference-reeconfig-from-dims-silent-kwargs hazard).
    for _k, _want in (
        ("use_contextual_safety_terrain", True),
        ("contextual_safety_gate_source", GATE_SOURCE),
        ("contextual_safety_proximity_threshold", prox_thresh),
        ("contextual_safety_accum_weight", CONTEXTUAL_ACCUM_WEIGHT),
        ("contextual_safety_release_threshold", CONTEXTUAL_RELEASE_THRESHOLD),
    ):
        _got = getattr(cfg, _k, None)
        if _got != _want:
            raise RuntimeError(
                "REEConfig.from_dims did not take %s (got %r, want %r) -- the MECH-303 gate "
                "would run at its default. Do not proceed." % (_k, _got, _want)
            )
    cfg.heartbeat.beta_gate_bistable = True
    cfg.e3.urgency_interrupt_threshold = 1e9   # disable MECH-091 abort -> isolate the safety gate
    agent = REEAgent(cfg)
    return env, agent


def _p0_warmup(agent, arm, seed):
    """P0a: SD-070 z_world encoder warmup on a DEDICATED env (the rollout consumes env RNG,
    so reusing the exposure env would shift the layout sequence P1 then sees)."""
    num_hazards, _ = _ARM_SPEC[arm]
    warm_env = CausalGridWorldV2(seed=SEED_BASE + seed, **_env_kwargs(num_hazards))
    before = latent_stack_snapshot(agent)
    stats = run_zworld_p0(
        agent, warm_env, seed, P0_EPISODES, P0_STEPS_PER_EPISODE,
        policy=RandomPolicy(seed), label=f"exq939 arm={arm}", dry_run=_DRY_RUN,
    )
    guard = {"guard_checked": False}
    encoder_trained = False
    try:
        guard = assert_world_encoder_trained(
            agent, before, p0=P0_EPISODES, strict=True, context=f"exq939 arm={arm} seed={seed}",
        )
        encoder_trained = True
    except ZWorldEncoderUntrainedError as exc:
        # Do NOT crash: an untrained encoder is a substrate-readiness state, and the correct
        # route for it is substrate_not_ready_requeue (a recorded non_contributory), never an
        # ERROR that looks like a code bug or a FAIL that looks like a claim verdict.
        guard = {"guard_checked": True, "error": str(exc)[:400]}
        print(f"  [P0a-UNTRAINED] arm={arm} seed={seed}: {exc}", flush=True)
    stats = dict(stats)
    stats["encoder_trained"] = bool(encoder_trained)
    stats["weight_delta_guard"] = guard
    return stats


def _expose(env, agent, arm, seed):
    """P1: random-walk the exposure context. accumulate_safety runs inside sense(), gated on
    the DEDICATED proximity channel. Records the per-tick series the 930 autopsy found
    missing (z_harm_a norm, proximity, limb damage)."""
    agent.reset()
    _, obs = env.reset()
    zharm_a, prox, damage = [], [], []
    for step in range(EXPOSURE_STEPS):
        prox.append(_prox(obs))
        damage.append(_damage(env))
        latent = _sense(agent, obs)
        zharm_a.append(
            float(latent.z_harm_a.detach().norm().item()) if latent.z_harm_a is not None else float("nan")
        )
        _ = _act(agent, latent)
        _flat, _harm, done, _info, obs = env.step(random.randint(0, 4))
        if done:
            _, obs = env.reset()
        if (step + 1) % 50 == 0 or (step + 1) == EXPOSURE_STEPS:
            print(f"  [train] arm={arm} seed={seed} phase=P1 "
                  f"ep {P0_EPISODES + step + 1}/{TOTAL_DENOM}", flush=True)
    return {"zharm_a": zharm_a, "prox": prox, "damage": damage}


def _test_release(agent, test_env):
    """P2: induced-commitment release readout in the arm's OWN context. agent.reset() clears
    the recurrent latent so the test-context z_world is clean; the safety terrain lives in
    the residue field and SURVIVES the reset (documented invariant). The only release path
    live here is the MECH-303 contextual gate (urgency interrupt off, no goal, store off)."""
    agent.reset()
    _, obs = test_env.reset()
    for _ in range(SETTLE_STEPS):
        _sense(agent, obs)
        _flat, _harm, done, _info, obs = test_env.step(random.randint(0, 4))
        if done:
            _, obs = test_env.reset()
    released = 0
    preds, zharm_a, prox, damage = [], [], [], []
    for _t in range(N_TEST_TRIALS):
        prox.append(_prox(obs))
        damage.append(_damage(test_env))
        latent = _sense(agent, obs)
        zharm_a.append(
            float(latent.z_harm_a.detach().norm().item()) if latent.z_harm_a is not None else float("nan")
        )
        if latent.z_world is not None and hasattr(agent.residue_field, "evaluate_safety"):
            preds.append(float(agent.residue_field.evaluate_safety(latent.z_world.detach()).mean().detach()))
        agent.beta_gate.elevate()
        was = agent.beta_gate.is_elevated
        _ = _act(agent, latent)
        if was and not agent.beta_gate.is_elevated:
            released += 1
        _flat, _harm, done, _info, obs = test_env.step(random.randint(0, 4))
        if done:
            _, obs = test_env.reset()
    return {
        "release_rate": released / max(N_TEST_TRIALS, 1),
        "n_released": released,
        "n_test_trials": N_TEST_TRIALS,
        "mean_contextual_safety_pred": float(np.mean(preds)) if preds else 0.0,
        "series": {"zharm_a": zharm_a, "prox": prox, "damage": damage},
    }


def _series_stats(vals):
    """Per-seed summary of a per-tick series. `std` is what makes the PER-SEED
    STANDARDISATION the 930 autopsy requires computable post hoc from the recorded series."""
    clean = [v for v in vals if v is not None and np.isfinite(v)]
    if not clean:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "n": len(clean),
        "mean": float(np.mean(clean)),
        "std": float(np.std(clean)),
        "min": float(np.min(clean)),
        "max": float(np.max(clean)),
    }


def _run_cell(arm, seed):
    with arm_cell(seed, config_slice=_config_slice(arm), script_path=Path(__file__),
                  config_slice_declared=True,
                  include_driver_script_in_hash=False) as cell:
        env, agent = _build(arm, seed)
        p0 = _p0_warmup(agent, arm, seed)
        expo = _expose(env, agent, arm, seed)
        rf = agent.residue_field
        num_safety_steps = int(getattr(rf, "num_safety_steps", 0))
        total_safety = float(getattr(rf, "total_safety", 0.0))
        num_hazards, prox_thresh = _ARM_SPEC[arm]
        test_env = CausalGridWorldV2(seed=SEED_BASE + TEST_SEED_OFFSET + seed,
                                     **_env_kwargs(num_hazards))
        test = _test_release(agent, test_env)
        _ZG.observe(agent)

        all_zharm = list(expo["zharm_a"]) + list(test["series"]["zharm_a"])
        all_prox = list(expo["prox"]) + list(test["series"]["prox"])
        all_damage = list(expo["damage"]) + list(test["series"]["damage"])
        damaged_ticks = sum(1 for d in all_damage if d > 1e-9)
        row = {
            "arm": arm,
            "seed": seed,
            "num_hazards": num_hazards,
            "proximity_threshold": prox_thresh,
            "release_rate": test["release_rate"],
            "n_released": test["n_released"],
            "mean_contextual_safety_pred": test["mean_contextual_safety_pred"],
            "num_safety_steps": num_safety_steps,
            "total_safety": total_safety,
            "accumulation_frac": num_safety_steps / max(EXPOSURE_STEPS, 1),
            # --- 930 autopsy recording gap, closed -------------------------------------
            "zharm_a_stats": _series_stats(all_zharm),
            "proximity_stats": _series_stats(all_prox),
            "damage_stats": _series_stats(all_damage),
            "damaged_tick_count": damaged_ticks,
            "damaged_tick_frac": damaged_ticks / max(len(all_damage), 1),
            "series": {                      # retained in full -- see module docstring
                "zharm_a_norm": [float(v) for v in all_zharm],
                "safety_proximity_harm": [float(v) for v in all_prox],
                "limb_damage_sum": [float(v) for v in all_damage],
            },
            "p0a": p0,
            "encoder_trained": bool(p0.get("encoder_trained")),
        }
        cell.stamp(row)
    return row


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return float(statistics.fmean(vals)) if vals else 0.0


def _worst_cell(rows, key):
    """Extremum + the offending cell id, so a `measured` recomputes exactly against an
    all(...) style `met` instead of being masked by an in-band mean."""
    cand = [(r.get(key), r) for r in rows if r.get(key) is not None]
    if not cand:
        return None, None
    val, r = min(cand, key=lambda t: t[0])
    return float(val), f"{r['arm']}::seed{r['seed']}"


def run_experiment(seeds):
    t_start = datetime.utcnow()
    per_cell = []
    for seed in seeds:
        for arm in ARMS:
            print(f"Seed {seed} Condition {arm}", flush=True)
            row = _run_cell(arm, seed)
            print(f"  [{arm} seed={seed}] release_rate={row['release_rate']:.2f} "
                  f"num_safety_steps={row['num_safety_steps']} "
                  f"pred={row['mean_contextual_safety_pred']:.3f} "
                  f"prox_mean={(row['proximity_stats']['mean'] or 0.0):.3f}", flush=True)
            print("verdict: PASS", flush=True)   # per-cell run-completion marker
            per_cell.append(row)

    def rows(arm):
        return [r for r in per_cell if r["arm"] == arm]

    rel = {a: _mean([r["release_rate"] for r in rows(a)]) for a in ARMS}
    dv1 = rel["A_safe_gate_natural"] - rel["B_safe_gate_forced_closed"]
    dv2 = rel["D_hazard_gate_forced_open"] - rel["C_hazard_gate_natural"]
    dv3 = rel["A_safe_gate_natural"] - rel["C_hazard_gate_natural"]

    # ---- readiness preconditions (all numeric -> the indexer recomputes `met`) ----------
    a_rows = rows("A_safe_gate_natural")
    safe_rows = [r for r in per_cell if r["arm"] in SAFE_ARMS]
    hazard_rows = [r for r in per_cell if r["arm"] in HAZARD_ARMS]
    a_worst_release, a_worst_release_cell = _worst_cell(a_rows, "release_rate")
    a_worst_accum, a_worst_accum_cell = _worst_cell(a_rows, "num_safety_steps")
    safe_prox_max = max([(r["proximity_stats"]["mean"] or 0.0) for r in safe_rows] or [0.0])
    hazard_prox_min = min([(r["proximity_stats"]["mean"] or 0.0) for r in hazard_rows] or [0.0])
    n_encoder_trained = sum(1 for r in per_cell if r.get("encoder_trained"))

    preconditions = [
        {"name": "arm_A_release_rate_positive_control", "kind": "readiness",
         "description": "arm A's OWN release_rate -- best case, and the SAME statistic every DV routes on",
         "control": "arm A (safe context, gate at the shipped default) is the best case by construction",
         "measured": a_worst_release if a_worst_release is not None else 0.0,
         "threshold": READINESS_FLOOR, "direction": "lower",
         "offending_cell": a_worst_release_cell,
         "met": all(r["release_rate"] >= READINESS_FLOOR for r in a_rows) if a_rows else False},
        {"name": "arm_A_accumulation_occurred", "kind": "readiness",
         "description": "arm A num_safety_steps -- the V3-EXQ-916 zero-accumulation catch",
         "control": "arm A's proximity gate is open by construction in a hazard-free context",
         "measured": float(a_worst_accum) if a_worst_accum is not None else 0.0,
         "threshold": float(ACCUM_STEPS_FLOOR), "direction": "lower",
         "offending_cell": a_worst_accum_cell,
         "met": all(r["num_safety_steps"] >= ACCUM_STEPS_FLOOR for r in a_rows) if a_rows else False},
        {"name": "proximity_safe_below_gate", "kind": "readiness",
         "description": "realized proximity mean in SAFE contexts must sit below the gate threshold",
         "control": "num_hazards=0 emits a hazard-proximity EMA of ~0 by construction",
         "measured": float(safe_prox_max), "threshold": PROX_THRESH_NATURAL, "direction": "upper",
         "met": bool(safe_prox_max < PROX_THRESH_NATURAL)},
        {"name": "proximity_hazard_above_gate", "kind": "readiness",
         "description": "realized proximity mean in HAZARD contexts must sit above the gate threshold",
         "control": "num_hazards=8 emits an elevated hazard-proximity EMA by construction",
         "measured": float(hazard_prox_min), "threshold": PROX_THRESH_NATURAL, "direction": "lower",
         "met": bool(hazard_prox_min > PROX_THRESH_NATURAL)},
        {"name": "world_encoder_trained", "kind": "readiness",
         "description": "SD-070 P0a warmup moved split_encoder.world_encoder in every cell",
         "control": "run_zworld_p0 weight-delta guard on a matched-distribution warmup env",
         "measured": float(n_encoder_trained), "threshold": float(len(per_cell)), "direction": "lower",
         "met": n_encoder_trained == len(per_cell)},
    ]
    readiness_ok = all(bool(p["met"]) for p in preconditions)
    n_valid_seeds = len({r["seed"] for r in a_rows if r["release_rate"] >= READINESS_FLOOR})

    dv1_pass = dv1 >= DV_MARGIN
    dv2_pass = dv2 >= DV_MARGIN
    dv3_pass = dv3 >= DV_MARGIN
    overall_pass = bool(readiness_ok and n_valid_seeds >= MIN_VALID_SEEDS
                        and dv1_pass and dv2_pass and dv3_pass)

    # Non-degeneracy: a DV whose two arms are bit-identical across every seed measured nothing.
    def spread(arm_a, arm_b):
        pair = [r["release_rate"] for r in rows(arm_a)] + [r["release_rate"] for r in rows(arm_b)]
        return float(max(pair) - min(pair)) if pair else 0.0

    criteria_non_degenerate = {
        "DV1_accumulation_necessity": spread("A_safe_gate_natural", "B_safe_gate_forced_closed") > 0.0,
        "DV2_accumulation_is_the_cause": spread("D_hazard_gate_forced_open", "C_hazard_gate_natural") > 0.0,
        "DV3_gate_natural_context_appropriate": spread("A_safe_gate_natural", "C_hazard_gate_natural") > 0.0,
    }
    non_degenerate = readiness_ok and any(criteria_non_degenerate.values())
    degeneracy_reason = None
    if not readiness_ok:
        unmet = [p["name"] for p in preconditions if not p["met"]]
        degeneracy_reason = ("substrate not ready: unmet readiness precondition(s) %s -- the "
                             "mechanism could not express itself, so nothing was measured"
                             % (", ".join(unmet),))
    elif not any(criteria_non_degenerate.values()):
        degeneracy_reason = "every DV's arm pair is bit-identical across all seeds -- no contrast measured"

    if not readiness_ok or n_valid_seeds < MIN_VALID_SEEDS:
        label = "substrate_not_ready_requeue"
        direction = "non_contributory"
    elif overall_pass:
        label = "mech303_proximity_gated_accumulation_lowers_background_vigilance"
        direction = "supports"
    elif dv3_pass and not dv2_pass:
        label = "mech303_context_difference_not_attributable_to_accumulation"
        direction = "mixed"
    else:
        label = "mech303_accumulation_does_not_lower_background_vigilance"
        direction = "weakens"

    manifest = {
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": ARCH_EPOCH,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "timestamp_utc": t_start.strftime("%Y%m%dT%H%M%SZ"),
        "outcome": "PASS" if overall_pass else "FAIL",
        "evidence_direction": direction,
        "release_rate_by_arm": rel,
        "dv1_accumulation_necessity_gap": dv1,
        "dv2_accumulation_is_the_cause_gap": dv2,
        "dv3_gate_natural_context_gap": dv3,
        "dv_margin": DV_MARGIN,
        "n_valid_seeds": n_valid_seeds,
        "combination_rule": (
            "PASS = readiness_ok AND n_valid_seeds >= %d AND (DV1 >= %.2f) AND (DV2 >= %.2f) "
            "AND (DV3 >= %.2f) -- a plain conjunction of three gaps. DV2 and DV3 are the "
            "load-bearing criteria; DV1 is a config-forced ablation and is reported but is "
            "NOT load-bearing." % (MIN_VALID_SEEDS, DV_MARGIN, DV_MARGIN, DV_MARGIN)
        ),
        "criteria": [
            {"name": "DV1_accumulation_necessity", "load_bearing": False, "passed": bool(dv1_pass),
             "gap": dv1, "margin": DV_MARGIN},
            {"name": "DV2_accumulation_is_the_cause", "load_bearing": True, "passed": bool(dv2_pass),
             "gap": dv2, "margin": DV_MARGIN},
            {"name": "DV3_gate_natural_context_appropriate", "load_bearing": True, "passed": bool(dv3_pass),
             "gap": dv3, "margin": DV_MARGIN},
        ],
        "criteria_non_degenerate": criteria_non_degenerate,
        "non_degenerate": bool(non_degenerate),
        "interpretation": {
            "label": label,
            "preconditions": preconditions,
            "criteria_non_degenerate": criteria_non_degenerate,
        },
        "arm_results": per_cell,
        "per_seed_release_rate": {a: [r["release_rate"] for r in rows(a)] for a in ARMS},
        "per_seed_num_safety_steps": {a: [r["num_safety_steps"] for r in rows(a)] for a in ARMS},
        "custom_information": {
            "gate_source": GATE_SOURCE,
            "proximity_threshold_by_arm": {a: _ARM_SPEC[a][1] for a in ARMS},
            "zharm_a_recording_note": (
                "Per-tick z_harm_a norm series retained per cell under arm_results[].series. "
                "The encoder is UNTRAINED and its between-seed init offset (sd ~6.4e-02) "
                "exceeds within-run modulation (~1e-03) by ~60x, so an ABSOLUTE-threshold read "
                "on the raw norm cannot discriminate -- standardise PER SEED using "
                "arm_results[].zharm_a_stats.{mean,std} before any discrimination analysis. "
                "Recorded per the explicit recording-gap finding in "
                "failure_autopsy_V3-EXQ-930_2026-08-16."
            ),
            "damage_recording_note": (
                "Per-tick sum(limb_damage) retained per cell; damaged_tick_frac is the "
                "manipulation check that the SD-022 damage pathway was exercised under "
                "limb_damage_enabled=True. V3-EXQ-930 recorded none, which is why its "
                "damage-sourced z_harm_a null was unfalsifiable."
            ),
            "seed_44_avoided": True,
        },
    }
    if degeneracy_reason:
        manifest["degeneracy_reason"] = degeneracy_reason
    return manifest, overall_pass


_DRY_RUN = False


def build_and_run():
    global _DRY_RUN, EXPOSURE_STEPS, N_TEST_TRIALS, P0_EPISODES, TOTAL_DENOM, MIN_VALID_SEEDS
    t0 = time.perf_counter()
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    args = ap.parse_args()
    _DRY_RUN = bool(args.dry_run)

    if args.dry_run:
        EXPOSURE_STEPS = 30
        N_TEST_TRIALS = 6
        P0_EPISODES = 3
        TOTAL_DENOM = P0_EPISODES + EXPOSURE_STEPS
        MIN_VALID_SEEDS = 1

    seeds = list(range(2 if args.dry_run else args.seeds))
    random.seed(SEED_BASE)
    torch.manual_seed(SEED_BASE)
    np.random.seed(SEED_BASE)

    manifest, _overall = run_experiment(seeds)
    ts = manifest["timestamp_utc"]
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"
    manifest["run_id"] = run_id

    full_config = {
        "arms": ARMS,
        "arm_spec": {a: {"num_hazards": _ARM_SPEC[a][0], "proximity_threshold": _ARM_SPEC[a][1]}
                     for a in ARMS},
        "env_kwargs_safe": _env_kwargs(NUM_HAZARDS_SAFE),
        "env_kwargs_hazard": _env_kwargs(NUM_HAZARDS_HAZARD),
        "substrate_common": dict(_SUBSTRATE_COMMON),
        "gate_source": GATE_SOURCE,
        "proximity_threshold_natural": PROX_THRESH_NATURAL,
        "proximity_threshold_forced_closed": PROX_THRESH_FORCED_CLOSED,
        "proximity_threshold_forced_open": PROX_THRESH_FORCED_OPEN,
        "contextual_safety_accum_weight": CONTEXTUAL_ACCUM_WEIGHT,
        "contextual_safety_release_threshold": CONTEXTUAL_RELEASE_THRESHOLD,
        "safety_terrain_bandwidth": SAFETY_TERRAIN_BW,
        "p0_episodes": P0_EPISODES,
        "p0_steps_per_episode": P0_STEPS_PER_EPISODE,
        "exposure_steps": EXPOSURE_STEPS,
        "n_test_trials": N_TEST_TRIALS,
        "settle_steps": SETTLE_STEPS,
        "dv_margin": DV_MARGIN,
        "readiness_floor": READINESS_FLOOR,
        "accum_steps_floor": ACCUM_STEPS_FLOOR,
        "min_valid_seeds": MIN_VALID_SEEDS,
        "seed_base": SEED_BASE,
        "test_seed_offset": TEST_SEED_OFFSET,
    }

    out_path = write_flat_manifest(
        manifest, None, dry_run=args.dry_run,
        config=full_config, seeds=seeds, script_path=Path(__file__), started_at=t0,
        z_goal_stream_stats=_ZG.stats(),
    )
    print(f"outcome={manifest['outcome']} evidence_direction={manifest['evidence_direction']} "
          f"label={manifest['interpretation']['label']} "
          f"dv1={manifest['dv1_accumulation_necessity_gap']:.3f} "
          f"dv2={manifest['dv2_accumulation_is_the_cause_gap']:.3f} "
          f"dv3={manifest['dv3_gate_natural_context_gap']:.3f}", flush=True)
    print(f"manifest={out_path}", flush=True)
    return manifest, out_path, run_id, args.dry_run


if __name__ == "__main__":
    _manifest, _out_path, _run_id, _dry = build_and_run()
    emit_outcome(
        outcome=_manifest["outcome"] if _manifest["outcome"] in ("PASS", "FAIL") else "FAIL",
        manifest_path=_out_path,
        run_id=_run_id,
        dry_run=_dry,
    )
