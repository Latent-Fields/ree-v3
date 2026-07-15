"""V3-EXQ-764: MECH-303 promote-to-active BEHAVIOURAL contextual-safety falsifier.

STATUS: BUILT + VALIDATED, HELD (NOT QUEUED) 2026-07-15 (user decision). The SD-067
read-resolution fix works (terrain-necessity gap 0.748 PASS, store-dissociation 0.833
PASS on the 6-seed inline) but the context-specificity DV (rel_A - rel_C = 0.292) sits
under the 0.34 margin because the safe-vs-unsafe z_world separability is only rank
AUC ~0.83 (SD-008 under-differentiation): 5/6 seeds discriminate but 1 inverts, and a
variance-aware paired test does not clear it. No read-side fix breaks that input ceiling.
This script is complete + runner-conformant; QUEUE it via /queue-experiment only once
z_world can separate safe/unsafe contexts (SD-008 differentiation work). See
REE_assembly/docs/architecture/sd_067_safety_terrain_bandwidth.md "Disposition".

Tests the pre-registered MECH-303 promote-to-active signature end-to-end: does a
passively-safe context (sustained no-harm accumulation via the contextual safety
terrain) LOWER the agent's background avoidance-commitment / vigilance level
relative to an unsafe context? MECH-303's primary output (claims.yaml) is: "lowers
the background avoidance commitment threshold in familiar safe environments -- a
diffuse reduction in vigilance tone driven by context, not by a specific cue
signal." The release mechanism is fully wired in agent.py select_action(): when
use_contextual_safety_terrain is set and beta_gate is elevated, the agent reads
residue_field.evaluate_safety(z_world) and RELEASES the avoidance commitment if the
accumulated terrain at the current context exceeds contextual_safety_release_threshold.

V3-EXQ-760 gave the necessary-not-sufficient REPRESENTATION-level evidence (read-only
evaluate_safety DISCRIMINATES safe vs unsafe contexts, gated AUC 0.884 vs ungated
0.488; promoted candidate->provisional 2026-07-15). This is the SUFFICIENT behavioural
gate -- the mirror of how MECH-304 went 759 (representation) -> 763 (behavioural) ->
active. This script is the structural mirror of V3-EXQ-763 with terrain <-> store
swapped: MECH-303 contextual terrain is the mechanism under test; MECH-304 cue-store
is the dissociation control.

SUBSTRATE DEPENDENCY (SD-067, 2026-07-15): the contextual safety terrain is a
Euclidean RBF over accumulated z_world centers; its read gate saturates on the
z_world common-mode (SD-008) -- safe (num_hazards=0) and unsafe (num_hazards=8)
contexts sit only ~0.065 apart in z_world, and the shared kernel_bandwidth (1.0) is
~15x too wide, so evaluate_safety reads ~identically in every context (safe 99.7 vs
unsafe 99.7) and the fixed release threshold fires INDISCRIMINATELY (this test
self-routes a false weakens without the fix). This run sets safety_terrain_bandwidth
=0.03 (SD-067): a dedicated tighter bandwidth resolves the small residual into an
absolute gap the release gate can use (safe pred ~1.12 vs unsafe ~0.38). Centering
(the SD-066 lever for MECH-304's cosine store) is a no-op here -- the RBF is Euclidean
and Euclidean distance is translation-invariant -- so bandwidth is the correct lever.
Additionally FREEZES accumulation for the test window (harm gate dropped below the
z_harm_a floor): the affective-harm encoder z_harm_a does NOT distinguish hazard
density (SD-011: ~0.547 safe vs ~0.542 unsafe), so live accumulation would pollute
the unsafe read -- freezing isolates the MECH-303 EXPRESSION pathway (the
promote-to-active claim) from the accumulation gate's SD-011 dependency.

DESIGN (mechanistic, "optogenetic" induction -- mirrors 763 option i). The competence
wall (GAP-I: beta_gate.elevate() fires only from the F-selection/commitment path the
substrate cannot sustain -- 603h) is bypassed by harness-inducing the commitment via
agent.beta_gate.elevate() each test trial, then measuring whether the MECH-303
contextual gate RELEASES it. This isolates the MECH-303 release mechanism as the
background-vigilance readout. Confounding release paths are held constant/off at test:
urgency-interrupt disabled (urgency_interrupt_threshold=1e9), no goal, and (arm D) the
MECH-302 comparator reset -- so the ONLY release path in arms A/B/C is the contextual
safety gate, and in arm D the conditioned-store gate.

ARMS (crossing use_contextual_safety_terrain {ON, OFF} x context {passively-safe, unsafe}):
  A  terrain_on_safe  : terrain ON,  safe-context taught + SAFE at test    -> contextual release
  B  terrain_off_safe : terrain OFF, safe-context taught + SAFE at test     -> ABLATION (terrain necessary)
  C  terrain_on_unsafe: terrain ON,  safe-context taught + UNSAFE at test   -> context-specificity
  D  store_spare      : terrain OFF, use_conditioned_safety_store ON, cue taught + present
                        -> DISSOCIATION: MECH-304 cue-store release still fires

TEACHING (A/B/C): roam a safe (low-harm) interior so z_harm_a stays below the harm
  threshold every tick and MECH-303 accumulate_safety writes the passive terrain broadly
  over the region containing the test cell (terrain-OFF arm B still roams + senses; with
  the terrain disabled no store forms -> the clean ablation).
TEACHING (D): SD-022 scheduled_limb_damage -> damage->heal->relief (MECH-302) with SD-065
  safety_cue_on_relief so the cue co-occurs with the relief window; the SD-066 centered
  ConditionedSafetyStore writes its prototype ONLY on the real event_fired tick.
TEST (A/B): agent in a SAFE context (hazard far -> matches the accumulated terrain),
  beta_gate.elevate() induced, one sense+select_action, record contextual release.
TEST (C): agent in an UNSAFE context (hazards surrounding -> z_world shifts off the
  accumulated safe terrain), beta_gate.elevate() induced, record release (should NOT fire).
TEST (D): cue present, no threat, beta_gate.elevate() induced, record store release.

DVs (pre-registered):
  DV1 terrain-necessity + context-specificity (terrain-ON arm A is the anchor):
      release_A - release_B >= DV1_MARGIN  AND  release_A - release_C >= DV1_MARGIN
  DV2 dissociation / MECH-304 sparing under MECH-303 ablation:
      release_D (cue-store) >= DV2_MARGIN
  PASS = DV1 AND DV2.

READINESS / NON-VACUITY (self-route substrate_not_ready_requeue -> non_contributory, NEVER
  a false weakens; 759/688 lesson): the positive control is arm A's OWN release_rate in
  the taught SAFE context -- the best case and the SAME statistic (release_rate) DV1
  routes on. If the terrain/encoder cannot represent the safe context well enough to
  release even in the best case, the substrate is not ready and the run self-routes
  non_contributory (not a MECH-303 refutation). n_valid_seeds requires the terrain to have
  formed (evaluate_safety in the safe context clears the release threshold) for arm A AND
  the store prototype to have formed for arm D.

claim_ids=[MECH-303]; experiment_purpose=evidence. A behavioural PASS with the ablation +
context-specificity + MECH-304 dissociation is what gates MECH-303 provisional->active.
"""
import argparse
import random
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
from pack_writer import write_flat_manifest  # noqa: E402
from experiment_protocol import emit_outcome  # noqa: E402

EXPERIMENT_PURPOSE = "evidence"
EXPERIMENT_TYPE = "v3_exq_764_mech303_contextual_safety_avoidance_release_behavioural_falsifier"
ARCH_EPOCH = "ree_hybrid_guardrails_v1"
CLAIM_IDS = ["MECH-303"]

DEVICE = torch.device("cpu")

# --- pre-registered constants -------------------------------------------------
N_SEEDS = 6
SEED_BASE = 1000
TEACH_STEPS = 240            # progress denominator M (per seed x arm)
N_TEST_TRIALS = 30
STORE_THRESHOLD = 0.6        # SD-066 centered-readout gate (arm D store dissociation)
PROTO_MIN_NORM = 0.05        # centered store prototype (residual) norm floor for "formed" (arm D)
READINESS_FLOOR = 0.34       # positive-control (arm A safe context, best case) release_rate floor
DV1_MARGIN = 0.34            # release_A - release_{B,C} >= this
DV2_MARGIN = 0.34            # cue-store release_D >= this
MIN_VALID_SEEDS = 4          # seeds passing readiness needed for a non-vacuous verdict
# SD-067: dedicated (tighter) safety-terrain RBF bandwidth. The shared kernel_bandwidth
# (1.0) is ~15x too wide for the z_world residual scale (~0.065 between safe/unsafe), so
# evaluate_safety saturates and the release fires indiscriminately -- WITHOUT this the
# context-specificity DV fails for a substrate reason (false weakens). Calibrated to 0.03
# on the real field: safe pred ~1.12 vs unsafe ~0.38 -> release gap 0.43 > margin. See
# sd_067_safety_terrain_bandwidth.md.
SAFETY_TERRAIN_BW = 0.03
# Eval control: freeze contextual accumulation during the test window by dropping the
# harm gate below the z_harm_a floor. The affective-harm encoder z_harm_a does NOT
# distinguish hazard density (SD-011: ~0.547 safe vs ~0.542 unsafe, both below the 0.55
# accumulate gate), so live test-time accumulation would pollute the unsafe read. Freezing
# isolates the MECH-303 EXPRESSION pathway (the promote-to-active claim) from the
# accumulation gate's SD-011 dependency (mirrors 763 resetting the comparator at test).
TEST_FREEZE_HARM_THRESHOLD = -1.0

# Substrate operating config shared by every arm (the "substrate is ON" baseline).
# safety_store_centered=True (SD-066) is inert unless use_conditioned_safety_store is
# also set (arm D only); alpha_world=0.9 gives z_world fidelity for the terrain read
# (SD-008; 760 precedent).
_COMMON_SUBSTRATE = dict(
    alpha_world=0.9,                       # SD-008 z_world fidelity (760 precedent)
    use_harm_stream=True, harm_obs_dim=51,
    use_affective_harm_stream=True, harm_obs_a_dim=7,
    use_suffering_derivative_comparator=True,   # MECH-302 relief teaching signal (arm D)
    safety_store_threshold=STORE_THRESHOLD,
    safety_store_centered=True,                 # SD-066 common-mode-invariant gate (arm D)
)
# Terrain-arm contexts mirror the VALIDATED V3-EXQ-760 discrimination construction
# (gated AUC 0.884): SAFE = num_hazards=0 (hazard_field_view exactly 0 everywhere) vs
# UNSAFE = num_hazards=8 (uniformly elevated) -- whole-env layouts with globally distinct
# z_world geometry. NOT the same env with hazards nudged near the agent (that does not
# shift z_world off the accumulated safe-terrain basin -> would fail context-specificity
# for a design reason, not a real null). use_proxy_fields=True matches 760.
GRID_SIZE = 10
NUM_RESOURCES = 3
NUM_HAZARDS_UNSAFE = 8
# Env for the terrain arms' SAFE teaching + SAFE test (A/B); also arm C's SAFE teaching.
_ENV_SAFE = dict(
    size=GRID_SIZE, num_hazards=0, num_resources=NUM_RESOURCES,
    limb_damage_enabled=True, heal_rate=0.4, use_proxy_fields=True,
)
# Env for arm C's UNSAFE test context (hazard-dense -> z_world off the safe terrain).
_ENV_UNSAFE = dict(
    size=GRID_SIZE, num_hazards=NUM_HAZARDS_UNSAFE, num_resources=NUM_RESOURCES,
    limb_damage_enabled=True, heal_rate=0.4, use_proxy_fields=True,
)
# Env teaching curriculum for the MECH-304 store arm (D): damage->heal->relief + cue pairing.
_ENV_CUE_TEACH = dict(
    size=10, num_hazards=3, num_resources=2,
    limb_damage_enabled=True, heal_rate=0.4,
    scheduled_limb_damage_enabled=True, scheduled_limb_damage_interval=15,
    scheduled_limb_damage_prob=1.0, scheduled_limb_damage_magnitude=0.5,
    scheduled_limb_damage_limb_selection="all",
    safety_cue_enabled=True, safety_cue_on_relief=True, safety_cue_heal_floor=0.02,
)
CONTEXTUAL_HARM_THRESHOLD = 0.55   # above the per-seed z_harm_a safe baseline -> safe ticks accumulate
CONTEXTUAL_ACCUM_WEIGHT = 0.05
CONTEXTUAL_RELEASE_THRESHOLD = 0.5  # calibrated in-smoke between safe-context and unsafe-context evaluate_safety
TEST_SEED_OFFSET = 500              # held-out test-env seed (distinct layout from teaching)
SETTLE_STEPS = 8                    # settle recurrent z_world into the test context before measuring

ARMS = ["A_terrain_on_safe", "B_terrain_off_safe", "C_terrain_on_unsafe", "D_store_spare"]


def _sense(agent, obs):
    return agent.sense(
        obs["body_state"], obs["world_state"],
        obs_harm=obs.get("harm_obs"), obs_harm_a=obs.get("harm_obs_a"),
    )


def _act(agent, latent):
    ticks = agent.clock.advance()
    world_dim = agent.config.latent.world_dim
    e1_prior = agent._e1_tick(latent) if ticks.get("e1_tick") else torch.zeros(1, world_dim, device=DEVICE)
    candidates = agent.generate_trajectories(latent, e1_prior, ticks)
    action = agent.select_action(candidates, ticks)
    return int(action.argmax(dim=-1).item())


def _build(arm, seed):
    terrain_on = arm in ("A_terrain_on_safe", "C_terrain_on_unsafe")
    store_on = arm == "D_store_spare"
    if arm == "D_store_spare":
        env = CausalGridWorldV2(seed=SEED_BASE + seed, **_ENV_CUE_TEACH)
    else:
        env = CausalGridWorldV2(seed=SEED_BASE + seed, **_ENV_SAFE)   # SAFE (num_hazards=0) teaching
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim, world_obs_dim=env.world_obs_dim, action_dim=5,
        use_conditioned_safety_store=store_on,
        use_contextual_safety_terrain=terrain_on,
        contextual_safety_release_threshold=CONTEXTUAL_RELEASE_THRESHOLD,
        safety_terrain_bandwidth=SAFETY_TERRAIN_BW,   # SD-067 (threaded only when terrain_on)
        **_COMMON_SUBSTRATE,
    )
    cfg.heartbeat.beta_gate_bistable = True
    cfg.e3.urgency_interrupt_threshold = 1e9    # disable MECH-091 abort -> isolate safety gate
    if terrain_on:
        cfg.contextual_safety_harm_threshold = CONTEXTUAL_HARM_THRESHOLD
        cfg.contextual_safety_accum_weight = CONTEXTUAL_ACCUM_WEIGHT
    agent = REEAgent(cfg)
    return env, agent


def _teach_safe_context(env, agent, arm, seed):
    """Arms A/B/C teaching: random-walk roam a SAFE env (num_hazards=0) so z_harm_a stays
    below the harm threshold every tick and MECH-303 accumulate_safety writes the passive
    terrain over the safe z_world geometry (mirrors V3-EXQ-760's num_hazards=0 SAFE-context
    accumulation walk). Terrain-OFF arm B still roams + senses (no terrain forms -> the
    clean ablation)."""
    agent.reset()
    _, obs = env.reset()
    for step in range(TEACH_STEPS):
        _ = _act(agent, _sense(agent, obs))    # accumulate_safety runs inside sense()
        _flat, _harm, done, _info, obs = env.step(random.randint(0, 4))
        if done:
            _, obs = env.reset()
        if (step + 1) % 50 == 0:
            print(f"  [train] arm={arm} seed={seed} ep {step + 1}/{TEACH_STEPS}", flush=True)
    return 0.0   # no store prototype for the terrain arms (store OFF)


def _teach_cue_store(env, agent, arm, seed):
    """Arm D teaching (mirror of 763 _teach): random-roaming damage->heal->relief with the
    SD-065 cue paired to the relief window; the SD-066 centered store writes its prototype
    ONLY on the real event_fired tick. Random roam diversifies relief context so the CUE
    is the feature consistently paired with relief -> cue-dominant centered prototype."""
    _, obs = env.reset()
    agent.reset()
    for step in range(TEACH_STEPS):
        latent = _sense(agent, obs)
        _ = _act(agent, latent)          # advances the store update (in sense) + clock
        action = random.randint(0, 4)    # roam (deterministic: RNG reset at cell entry)
        _flat, _harm, done, _info, obs = env.step(action)
        if done:
            _, obs = env.reset()   # keep the agent (store persists); reset env only
        if (step + 1) % 50 == 0:
            print(f"  [train] arm={arm} seed={seed} ep {step + 1}/{TEACH_STEPS}", flush=True)
    proto_norm = 0.0
    if agent.conditioned_safety_store is not None:
        p = agent.conditioned_safety_store._prototype
        proto_norm = float(np.sqrt(sum(x * x for x in p)))
    return proto_norm


def _test_contextual_release(agent, test_env, n=N_TEST_TRIALS):
    """A/B/C test: induced commitment across a release-walk in the given test env
    (SAFE num_hazards=0 for A/B; UNSAFE num_hazards=8 for C). agent.reset() clears the
    recurrent latent so the test-context z_world is clean (terrain in residue SURVIVES the
    reset per the documented invariant; terrain arms have NO store to wipe). Release fires
    only when the MECH-303 terrain gate reads the current context as safe (evaluate_safety
    at the current z_world clears contextual_safety_release_threshold)."""
    agent.reset()
    _, obs = test_env.reset()
    for _ in range(SETTLE_STEPS):   # settle recurrent z_world into the test context
        _sense(agent, obs)
        _flat, _harm, done, _info, obs = test_env.step(random.randint(0, 4))
        if done:
            _, obs = test_env.reset()
    released = 0
    preds = []
    zharms = []
    for _t in range(n):
        latent = _sense(agent, obs)
        if latent.z_world is not None and hasattr(agent.residue_field, "evaluate_safety"):
            preds.append(float(agent.residue_field.evaluate_safety(latent.z_world.detach()).mean().detach()))
        zharms.append(float(latent.z_harm.detach().norm()) if latent.z_harm is not None else 0.0)
        agent.beta_gate.elevate()
        was = agent.beta_gate.is_elevated
        _ = _act(agent, latent)
        if was and not agent.beta_gate.is_elevated:
            released += 1
        _flat, _harm, done, _info, obs = test_env.step(random.randint(0, 4))   # advance the walk
        if done:
            _, obs = test_env.reset()
    return {
        "release_rate": released / max(n, 1),
        "mean_contextual_safety_pred": float(np.mean(preds)) if preds else 0.0,
        "mean_z_harm_norm": float(np.mean(zharms)) if zharms else 0.0,
    }


def _test_cue_store(env, agent, n=N_TEST_TRIALS):
    """D test (mirror of 763 _test_safety_gate, cue present, no threat): induced commitment
    + cue; MECH-304 conditioned-store release. Comparator reset kills the MECH-302 confound
    (no relief at test) so the ONLY release path is the conditioned-safety store gate."""
    if agent.suffering_comparator is not None:
        agent.suffering_comparator.reset()   # kill MECH-302 confound (no relief at test)
    agent_cell = (5, 5)
    haz = [(1, 1)]   # no threat -> clean store release (dissociation demonstration)
    released = 0
    sigs = []
    for _t in range(n):
        _, obs = env.reset_to(agent_cell, haz)
        env.set_safety_cue(True)
        _flat, _harm, _done, _info, obs = env.step(4)   # materialize cue in obs
        latent = _sense(agent, obs)
        sigs.append(float(agent._conditioned_safety_signal))
        agent.beta_gate.elevate()
        was = agent.beta_gate.is_elevated
        _ = _act(agent, latent)
        if was and not agent.beta_gate.is_elevated:
            released += 1
    return {
        "release_rate": released / max(n, 1),
        "mean_safety_signal": float(np.mean(sigs)) if sigs else 0.0,
    }


def _config_slice(arm):
    """Declared config slice for the per-cell arm fingerprint. Declares ONLY what the
    arm's build+collect path reads."""
    terrain_on = arm in ("A_terrain_on_safe", "C_terrain_on_unsafe")
    store_on = arm == "D_store_spare"
    env_kw = dict(_ENV_CUE_TEACH) if store_on else dict(_ENV_SAFE)
    test_env_kw = dict(_ENV_UNSAFE) if arm == "C_terrain_on_unsafe" else (
        dict(_ENV_CUE_TEACH) if store_on else dict(_ENV_SAFE))
    slice_ = {
        "arm": arm,
        "env_kwargs": env_kw,
        "test_env_kwargs": test_env_kw,
        "substrate": dict(_COMMON_SUBSTRATE),
        "use_conditioned_safety_store": store_on,
        "use_contextual_safety_terrain": terrain_on,
        "teach_steps": TEACH_STEPS,
        "n_test_trials": N_TEST_TRIALS,
        "settle_steps": SETTLE_STEPS,
        "test_seed_offset": TEST_SEED_OFFSET,
        "urgency_interrupt_threshold": 1e9,
        "test_context": "unsafe" if arm == "C_terrain_on_unsafe" else "safe",
    }
    if terrain_on:
        slice_.update({
            "contextual_safety_release_threshold": CONTEXTUAL_RELEASE_THRESHOLD,
            "contextual_safety_harm_threshold": CONTEXTUAL_HARM_THRESHOLD,
            "contextual_safety_accum_weight": CONTEXTUAL_ACCUM_WEIGHT,
            "safety_terrain_bandwidth": SAFETY_TERRAIN_BW,   # SD-067 (read only by terrain arms)
            "test_freeze_harm_threshold": TEST_FREEZE_HARM_THRESHOLD,
        })
    return slice_


def _run_cell(arm, seed):
    # Arm B is the terrain-OFF baseline: mint it cross-driver reuse-eligible
    # (include_driver_script_in_hash=False) so a future MECH-303-lineage driver can
    # reuse it; the config_slice + substrate_hash guards prevent a false hit if the
    # contextual_safety_* calibration iterates. The ON arms (A/C/D) are driver-bound
    # in-line emits (they always run fresh -- the mechanism / context is the treatment).
    off_baseline = arm == "B_terrain_off_safe"
    with arm_cell(seed, config_slice=_config_slice(arm), script_path=Path(__file__),
                  config_slice_declared=True,
                  include_driver_script_in_hash=not off_baseline) as cell:
        env, agent = _build(arm, seed)
        if arm == "D_store_spare":
            proto_norm = _teach_cue_store(env, agent, arm, seed)
        else:
            proto_norm = _teach_safe_context(env, agent, arm, seed)
        row = {"arm": arm, "seed": seed, "proto_norm": proto_norm}
        if arm == "D_store_spare":
            main = _test_cue_store(env, agent)
            row.update({
                "release_rate": main["release_rate"],
                "mean_safety_signal": main["mean_safety_signal"],
                "proto_formed": proto_norm >= PROTO_MIN_NORM,
            })
        else:
            # Eval control: freeze contextual accumulation for the test window so
            # live accumulation (the SD-011-insensitive harm gate cannot block it) does
            # not pollute the unsafe read. Isolates the MECH-303 expression pathway.
            agent.config.contextual_safety_harm_threshold = TEST_FREEZE_HARM_THRESHOLD
            # Held-out test env of the right context class (terrain lives in the agent's
            # residue and persists across the env swap).
            if arm == "C_terrain_on_unsafe":
                test_env = CausalGridWorldV2(seed=SEED_BASE + TEST_SEED_OFFSET + seed, **_ENV_UNSAFE)
            else:
                test_env = CausalGridWorldV2(seed=SEED_BASE + TEST_SEED_OFFSET + seed, **_ENV_SAFE)
            main = _test_contextual_release(agent, test_env)
            row.update({
                "release_rate": main["release_rate"],
                "mean_contextual_safety_pred": main["mean_contextual_safety_pred"],
                "mean_z_harm_norm": main["mean_z_harm_norm"],
            })
            if arm == "A_terrain_on_safe":
                # readiness anchor: arm A safe-context release IS the best-case positive
                # control -- SAME statistic (release_rate) DV1 routes on -- and terrain_ready
                # confirms the passive terrain formed above the release gate.
                row["terrain_ready"] = main["mean_contextual_safety_pred"] >= CONTEXTUAL_RELEASE_THRESHOLD
        cell.stamp(row)
    return row


def run_experiment(seeds):
    per_cell = []
    for seed in seeds:
        for arm in ARMS:
            print(f"Seed {seed} Condition {arm}", flush=True)
            row = _run_cell(arm, seed)
            print(f"  [{arm} seed={seed}] release_rate={row['release_rate']:.2f} "
                  f"pred={row.get('mean_contextual_safety_pred', 0.0):.3f} "
                  f"proto={row.get('proto_norm', 0.0):.3f}", flush=True)
            print("verdict: PASS", flush=True)   # per-cell run completion marker
            per_cell.append(row)

    def _rows(arm):
        return [r for r in per_cell if r["arm"] == arm]

    A = _rows("A_terrain_on_safe"); B = _rows("B_terrain_off_safe")
    C = _rows("C_terrain_on_unsafe"); D = _rows("D_store_spare")

    # Readiness: positive control = arm A safe-context release (best case), the DV-routed
    # statistic. Substrate-formed check: arm A terrain accumulated above the gate AND arm D
    # store prototype formed.
    pc_releases = [r["release_rate"] for r in A]
    terrain_ready = [r for r in A if r.get("terrain_ready")]
    proto_formed = [r for r in D if r.get("proto_formed")]
    mean_pc_release = float(np.mean(pc_releases)) if pc_releases else 0.0
    n_valid_seeds = min(len(terrain_ready), len(proto_formed))

    readiness_met = (mean_pc_release >= READINESS_FLOOR) and (n_valid_seeds >= MIN_VALID_SEEDS)

    rel_A = float(np.mean([r["release_rate"] for r in A])) if A else 0.0
    rel_B = float(np.mean([r["release_rate"] for r in B])) if B else 0.0
    rel_C = float(np.mean([r["release_rate"] for r in C])) if C else 0.0
    rel_D = float(np.mean([r["release_rate"] for r in D])) if D else 0.0

    dv1_terrain = rel_A - rel_B
    dv1_context = rel_A - rel_C
    dv1_pass = (dv1_terrain >= DV1_MARGIN) and (dv1_context >= DV1_MARGIN)
    dv2_pass = rel_D >= DV2_MARGIN

    # Non-degeneracy: the load-bearing separations must have real cross-arm spread.
    load_bearing_values = [rel_A, rel_B, rel_C, rel_D]
    non_degenerate = readiness_met and (float(np.std(load_bearing_values)) > 1e-3)

    if not readiness_met:
        outcome = "FAIL"
        evidence_direction = "non_contributory"
        label = "substrate_not_ready_requeue"
        overall_pass = False
    else:
        overall_pass = bool(dv1_pass and dv2_pass)
        outcome = "PASS" if overall_pass else "FAIL"
        evidence_direction = "supports" if overall_pass else "weakens"
        label = "mech303_contextual_vigilance_release_confirmed" if overall_pass else "mech303_behavioural_null"

    manifest = {
        "run_id": None,   # set below
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": ARCH_EPOCH,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "outcome": outcome,
        "evidence_direction": evidence_direction,
        "non_degenerate": non_degenerate,
        "degeneracy_reason": None if non_degenerate else (
            "substrate_not_ready" if not readiness_met else "zero_cross_arm_spread"),
        "timestamp_utc": datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
        "arm_results": per_cell,
        "dv1_terrain_necessity_and_context_specificity": {
            "release_A_terrain_on_safe": rel_A,
            "release_B_terrain_off_safe": rel_B,
            "release_C_terrain_on_unsafe": rel_C,
            "dv1_terrain_necessity_gap": dv1_terrain,
            "dv1_context_specificity_gap": dv1_context,
            "margin": DV1_MARGIN,
            "load_bearing": True,
            "passed": bool(dv1_pass),
        },
        "dv2_dissociation_sparing": {
            "release_D_store_cue": rel_D,
            "margin": DV2_MARGIN,
            "load_bearing": True,
            "passed": bool(dv2_pass),
        },
        "readiness": {
            "positive_control_release_safe_context": mean_pc_release,
            "readiness_floor": READINESS_FLOOR,
            "n_valid_seeds": n_valid_seeds,
            "min_valid_seeds": MIN_VALID_SEEDS,
            "readiness_met": bool(readiness_met),
        },
        "interpretation": {
            "label": label,
            "preconditions": [
                {"name": "arm_a_positive_control_release_safe_context",
                 "description": "arm A (terrain taught, SAFE context) release_rate -- SAME statistic DV1 routes on",
                 "measured": mean_pc_release, "threshold": READINESS_FLOOR,
                 "control": "arm A safe-context best-case (the passive-safe condition itself)",
                 "met": bool(mean_pc_release >= READINESS_FLOOR)},
                {"name": "n_valid_seeds_terrain_and_prototype",
                 "description": "seeds where arm A terrain formed above the release gate AND arm D store prototype formed",
                 "measured": n_valid_seeds, "threshold": MIN_VALID_SEEDS,
                 "control": "terrain pred >= release threshold and store prototype norm >= PROTO_MIN_NORM",
                 "met": bool(n_valid_seeds >= MIN_VALID_SEEDS)},
            ],
            "criteria_non_degenerate": {
                "DV1_terrain_necessity": bool(abs(dv1_terrain) > 1e-3),
                "DV1_context_specificity": bool(abs(dv1_context) > 1e-3),
                "DV2_store_sparing": bool(rel_D > 1e-3 or not readiness_met),
            },
        },
        "criteria": [
            {"name": "DV1_terrain_necessity_and_context_specificity", "load_bearing": True, "passed": bool(dv1_pass)},
            {"name": "DV2_dissociation_sparing", "load_bearing": True, "passed": bool(dv2_pass)},
        ],
        "ethics_preflight": {
            "involves_negative_valence": False,
            "involves_suffering_like_state": False,
            "involves_self_model": False,
            "involves_inescapability_or_helplessness": False,
            "involves_offline_replay_over_harm": False,
            "involves_social_mind_or_language": False,
            "involves_human_data_or_clinical_context": False,
            "decision": "allow",
        },
        "notes": (
            "MECH-303 promote-to-active behavioural contextual-safety falsifier: does a "
            "passively-safe context LOWER background avoidance-commitment/vigilance? Structural "
            "mirror of V3-EXQ-763 (MECH-304) with terrain <-> store swapped. Requires SD-067 "
            "(safety_terrain_bandwidth=0.03): the shared kernel_bandwidth saturates the Euclidean "
            "RBF terrain read on the z_world common-mode (SD-008) so the release fires "
            "indiscriminately -- WITHOUT SD-067 the context-specificity DV self-routes a false "
            "weakens; centering (the SD-066 lever) is a no-op for a translation-invariant Euclidean "
            "RBF. Accumulation FROZEN for the test window (harm gate below the z_harm_a floor) since "
            "z_harm_a does not distinguish hazard density (SD-011) -- isolates the MECH-303 "
            "expression pathway. Mechanistic beta_gate.elevate() induction (option i); "
            "urgency-interrupt disabled + no goal isolate the contextual safety gate. DV1 "
            "terrain-necessity (A vs B ablation) AND context-specificity (A safe vs C unsafe); DV2 "
            "MECH-304 cue-store dissociation (release_D with terrain OFF). Readiness anchors on arm "
            "A's own safe-context release_rate (the DV-routed statistic) -> substrate_not_ready_requeue "
            "below floor. OFF (terrain-off) arm B minted cross-driver reuse-eligible "
            "(include_driver_script_in_hash=False); a canonical baseline module is deferred while the "
            "contextual_safety_* calibration may still iterate (mirrors 763's in-flux note)."
        ),
    }
    return manifest, overall_pass


def build_and_run():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    args = ap.parse_args()

    t0 = time.perf_counter()
    seeds = list(range(2 if args.dry_run else args.seeds))
    global TEACH_STEPS, N_TEST_TRIALS, MIN_VALID_SEEDS
    if args.dry_run:
        TEACH_STEPS = 30
        N_TEST_TRIALS = 4
        MIN_VALID_SEEDS = 1

    manifest, _overall_pass = run_experiment(seeds)

    ts = manifest["timestamp_utc"]
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"
    manifest["run_id"] = run_id

    out_dir = _HERE.parent.parent / "REE_assembly" / "evidence" / "experiments"
    full_config = {
        "common_substrate": _COMMON_SUBSTRATE,
        "env_safe": _ENV_SAFE,
        "env_unsafe": _ENV_UNSAFE,
        "env_cue_teach": _ENV_CUE_TEACH,
        "num_hazards_unsafe": NUM_HAZARDS_UNSAFE,
        "settle_steps": SETTLE_STEPS,
        "test_seed_offset": TEST_SEED_OFFSET,
        "store_threshold": STORE_THRESHOLD,
        "proto_min_norm": PROTO_MIN_NORM,
        "readiness_floor": READINESS_FLOOR,
        "dv1_margin": DV1_MARGIN, "dv2_margin": DV2_MARGIN,
        "teach_steps": TEACH_STEPS, "n_test_trials": N_TEST_TRIALS,
        "contextual_harm_threshold": CONTEXTUAL_HARM_THRESHOLD,
        "contextual_accum_weight": CONTEXTUAL_ACCUM_WEIGHT,
        "contextual_release_threshold": CONTEXTUAL_RELEASE_THRESHOLD,
        "safety_terrain_bandwidth": SAFETY_TERRAIN_BW,
        "test_freeze_harm_threshold": TEST_FREEZE_HARM_THRESHOLD,
    }

    out_path = write_flat_manifest(
        manifest, out_dir, dry_run=args.dry_run,
        config=full_config, seeds=seeds, script_path=Path(__file__), started_at=t0,
    )
    print(f"outcome={manifest['outcome']} evidence_direction={manifest['evidence_direction']} "
          f"label={manifest['interpretation']['label']}", flush=True)
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
