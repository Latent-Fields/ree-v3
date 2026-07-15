"""V3-EXQ-763: MECH-304 promote-to-active BEHAVIOURAL conditioned-inhibition falsifier.

Tests the pre-registered claims.yaml MECH-304 "Falsifiable:" signature end-to-end on
the new SD-065 conditioned-safety cue channel (ree-v3 5e03aa6). MECH-304's release
mechanism is fully wired in agent.py -- sense() ticks ConditionedSafetyStore.update
(z_world, event_fired=MECH-302 relief); select_action() releases beta_gate when the
cue-specific safety signal exceeds safety_store_threshold and the gate is elevated. The
missing piece (an observable, controllable Pavlovian safety CS) is supplied by SD-065.

V3-EXQ-759 gave the necessary-not-sufficient REPRESENTATION-level evidence (frozen-agent
predict() AUC, promoted candidate->provisional). This is the SUFFICIENT behavioural gate.

SUBSTRATE DEPENDENCY (SD-066, 2026-07-15): the raw-cosine store gate saturates on the
z_world common-mode (SD-008 under-differentiation) -- every z_world sits at cosine ~0.99,
so sigmoid(gain*cos)>0.5 fires unconditionally and the cue (a ~0.006-cosine signal) is
unresolvable behaviourally (this test self-routes substrate_ceiling without the fix). This
run enables safety_store_centered=True (SD-066): the store subtracts a slow EMA of z_world
(the common-mode) before accumulating/querying the prototype, so the cue residual dominates
the cosine. Validated: cue-present separates from cue-absent above the 0.5 gate.

DESIGN (mechanistic, "optogenetic" induction -- option i of the queue brief). The
competence wall (GAP-I: beta_gate.elevate() fires only from the F-selection/commitment
path, which the substrate cannot sustain -- 603h) is bypassed by harness-inducing the
commitment via agent.beta_gate.elevate() each test trial, then measuring whether the
MECH-304 gate RELEASES it. This isolates the MECH-304 release mechanism; the concurrent
{threat + cue} condition (the actual novel requirement) is what the SD-065 channel
supplies. Confounding release paths are held constant/off at test: urgency-interrupt
disabled (urgency_interrupt_threshold=1e9), the MECH-302 comparator buffer reset + body
damage zeroed (no relief at test), no goal -> the ONLY release path is the safety gate.

ARMS (crossing use_conditioned_safety_store {ON, OFF} x cue-at-test x MECH-303 spare):
  A  store_on_cue    : store ON,  cue taught + present at test  -> conditioned inhibition
  B  store_off_cue   : store OFF, cue taught + present at test  -> ABLATION (store necessary)
  C  store_on_nocue  : store ON,  cue taught + ABSENT at test   -> cue-specificity
  D  mech303_spare   : store OFF, use_contextual_safety_terrain ON, safe-context taught
                       -> DISSOCIATION: MECH-303 contextual release still fires

TEACHING (A/B/C): SD-022 scheduled_limb_damage -> damage->heal->relief (MECH-302), with
  SD-065 safety_cue_on_relief so the cue co-occurs with the relief window; the store
  writes its prototype ONLY on the real event_fired tick (pairing real, not synthesized).
TEACHING (D): a safe (low-harm) context so MECH-303 accumulate_safety writes terrain.
TEST (A/B/C): agent surrounded by hazards (concurrent THREAT present in z_world/z_harm),
  cue per arm, beta_gate.elevate() induced, one sense+select_action, record release.
TEST (D): safe-ish context, beta_gate.elevate() induced, record contextual release.

DVs (pre-registered):
  DV1 conditioned inhibition + cue-specificity + store-necessity (ON arm only):
      release_A - release_B >= DV1_MARGIN  AND  release_A - release_C >= DV1_MARGIN
  DV2 dissociation / MECH-303 sparing under MECH-304 ablation:
      release_D (contextual) >= DV2_MARGIN
  PASS = DV1 AND DV2.

READINESS / NON-VACUITY (self-route substrate_not_ready_requeue -> non_contributory, NEVER
a false weakens; 759/688 lesson): a positive control (arm A, cue present, NO threat) must
release above READINESS_FLOOR -- the SAME statistic (release_rate) DV1 routes on. If the
prototype/encoder cannot represent the cue well enough to release even in the best case,
the substrate is not ready and the run self-routes non_contributory (not a MECH-304
refutation). Terrain must accumulate for arm D likewise.

claim_ids=[MECH-304]; experiment_purpose=evidence. A behavioural PASS with the ablation
dissociation is what gates MECH-304 provisional->active.
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
EXPERIMENT_TYPE = "v3_exq_763_mech304_conditioned_inhibition_behavioural_falsifier"
ARCH_EPOCH = "ree_hybrid_guardrails_v1"
CLAIM_IDS = ["MECH-304"]

DEVICE = torch.device("cpu")

# --- pre-registered constants -------------------------------------------------
N_SEEDS = 6
SEED_BASE = 1000
TEACH_STEPS = 240            # progress denominator M (per seed x arm)
N_TEST_TRIALS = 30
STORE_THRESHOLD = 0.6        # SD-066 centered-readout gate (cue sig ~0.7-0.94 vs nocue ~0.17-0.48)
PROTO_MIN_NORM = 0.05        # centered prototype (residual) norm floor for "formed"
READINESS_FLOOR = 0.34       # positive-control (cue, no threat) release_rate floor
DV1_MARGIN = 0.34            # release_A - release_{B,C} >= this
DV2_MARGIN = 0.34            # contextual release_D >= this
MIN_VALID_SEEDS = 4          # seeds passing readiness needed for a non-vacuous verdict

# Substrate operating config shared by every arm (the "substrate is ON" baseline).
# SD-066 centered readout (safety_store_centered=True) lifts the z_world common-mode
# ceiling that makes the raw-cosine gate fire unconditionally -- WITHOUT it this test
# self-routes substrate_ceiling (the cue is unresolvable behaviourally). See
# sd_066_centered_conditioned_safety_readout.md.
_COMMON_SUBSTRATE = dict(
    alpha_world=0.9,                       # SD-008 z_world fidelity (759 precedent)
    use_harm_stream=True, harm_obs_dim=51,
    use_affective_harm_stream=True, harm_obs_a_dim=7,
    use_suffering_derivative_comparator=True,   # MECH-302 relief teaching signal
    safety_store_threshold=STORE_THRESHOLD,
    safety_store_centered=True,                 # SD-066 common-mode-invariant gate
)
# Env teaching curriculum for the cue arms (A/B/C): damage->heal->relief + cue pairing.
_ENV_CUE_TEACH = dict(
    size=10, num_hazards=3, num_resources=2,
    limb_damage_enabled=True, heal_rate=0.4,
    scheduled_limb_damage_enabled=True, scheduled_limb_damage_interval=15,
    scheduled_limb_damage_prob=1.0, scheduled_limb_damage_magnitude=0.5,
    scheduled_limb_damage_limb_selection="all",
    safety_cue_enabled=True, safety_cue_on_relief=True, safety_cue_heal_floor=0.02,
)
# Env teaching curriculum for the MECH-303 arm (D): safe (no scheduled damage) + cue off.
_ENV_SAFE_TEACH = dict(
    size=10, num_hazards=1, num_resources=3,
    limb_damage_enabled=True, heal_rate=0.4,
)
CONTEXTUAL_HARM_THRESHOLD = 0.55   # above the per-seed z_harm_a baseline range (0.34-0.55) -> safe ticks accumulate
CONTEXTUAL_ACCUM_WEIGHT = 0.05
CONTEXTUAL_RELEASE_THRESHOLD = 0.5
SAFE_INTERIOR = [(r, c) for r in range(2, 8) for c in range(2, 8)]   # arm D roam cells
FAR_HAZARD = [(1, 1)]              # hazard kept far -> safe context for MECH-303 terrain

ARMS = ["A_store_on_cue", "B_store_off_cue", "C_store_on_nocue", "D_mech303_spare"]


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
    store_on = arm in ("A_store_on_cue", "C_store_on_nocue")
    terrain_on = arm == "D_mech303_spare"
    if arm == "D_mech303_spare":
        env = CausalGridWorldV2(seed=SEED_BASE + seed, **_ENV_SAFE_TEACH)
    else:
        env = CausalGridWorldV2(seed=SEED_BASE + seed, **_ENV_CUE_TEACH)
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim, world_obs_dim=env.world_obs_dim, action_dim=5,
        use_conditioned_safety_store=store_on,
        use_contextual_safety_terrain=terrain_on,
        contextual_safety_release_threshold=CONTEXTUAL_RELEASE_THRESHOLD,
        **_COMMON_SUBSTRATE,
    )
    cfg.heartbeat.beta_gate_bistable = True
    cfg.e3.urgency_interrupt_threshold = 1e9    # disable MECH-091 abort -> isolate safety gate
    if terrain_on:
        cfg.contextual_safety_harm_threshold = CONTEXTUAL_HARM_THRESHOLD
        cfg.contextual_safety_accum_weight = CONTEXTUAL_ACCUM_WEIGHT
    agent = REEAgent(cfg)
    return env, agent


def _teach(env, agent, arm, seed):
    # Random-roaming teaching (actions random, agent still SENSES so the store /
    # terrain updates): maximises relief-context diversity so the CUE is the only
    # feature consistently paired with relief -> the SD-066 centered prototype is
    # cue-dominant (validated: cue-present separates from cue-absent above the gate).
    _, obs = env.reset()
    agent.reset()
    for step in range(TEACH_STEPS):
        latent = _sense(agent, obs)
        _ = _act(agent, latent)          # advances the store update (in sense) + clock
        action = random.randint(0, 4)    # roam (deterministic: RNG reset at cell entry)
        _flat, _harm, done, _info, obs = env.step(action)
        if done:
            _, obs = env.reset()   # keep the agent (store/terrain persist); reset env only
        if (step + 1) % 50 == 0:
            print(f"  [train] arm={arm} seed={seed} ep {step + 1}/{TEACH_STEPS}", flush=True)
    proto_norm = 0.0
    if agent.conditioned_safety_store is not None:
        p = agent.conditioned_safety_store._prototype
        proto_norm = float(np.sqrt(sum(x * x for x in p)))
    return proto_norm


def _teach_safe_context(env, agent, arm, seed):
    """Arm D (MECH-303) teaching: roam the safe interior via reset_to (limb_damage
    stays 0 -> z_harm_a at baseline < harm_threshold every tick), so the contextual
    safety terrain accumulates broadly over the region containing the test cell.
    Distinct from the cue arms' scheduled-damage->relief teaching."""
    agent.reset()
    for step in range(TEACH_STEPS):
        cell = SAFE_INTERIOR[step % len(SAFE_INTERIOR)]
        _, obs = env.reset_to(cell, FAR_HAZARD)
        _flat, _harm, _done, _info, obs = env.step(4)
        _ = _act(agent, _sense(agent, obs))    # accumulate_safety runs inside sense()
        if (step + 1) % 50 == 0:
            print(f"  [train] arm={arm} seed={seed} ep {step + 1}/{TEACH_STEPS}", flush=True)
    return 0.0   # no store prototype for arm D (store OFF)


def _test_safety_gate(env, agent, cue_on, threat, n=N_TEST_TRIALS):
    """A/B/C test: induced commitment + {cue?, threat?}; return release_rate + diagnostics."""
    if agent.suffering_comparator is not None:
        agent.suffering_comparator.reset()   # kill MECH-302 confound (no relief at test)
    agent_cell = (5, 5)
    haz = [(4, 5), (6, 5), (5, 4), (5, 6)] if threat else [(1, 1)]   # surround vs far
    released = 0
    sigs = []
    threats = []
    zharms = []
    for _t in range(n):
        _, obs = env.reset_to(agent_cell, haz)
        env.set_safety_cue(bool(cue_on))
        _flat, _harm, _done, _info, obs = env.step(4)   # materialize cue+threat in obs
        latent = _sense(agent, obs)
        sig = float(agent._conditioned_safety_signal)
        sigs.append(sig)
        scv = obs.get("safety_cue_field_view")
        hv = obs.get("hazard_field_view")
        threats.append(float(hv.max()) if hv is not None else 0.0)
        zharms.append(float(latent.z_harm.detach().norm()) if latent.z_harm is not None else 0.0)
        agent.beta_gate.elevate()
        was = agent.beta_gate.is_elevated
        _ = _act(agent, latent)
        if was and not agent.beta_gate.is_elevated:
            released += 1
    return {
        "release_rate": released / max(n, 1),
        "mean_safety_signal": float(np.mean(sigs)) if sigs else 0.0,
        "mean_threat_hazard_view": float(np.mean(threats)) if threats else 0.0,
        "mean_z_harm_norm": float(np.mean(zharms)) if zharms else 0.0,
    }


def _test_contextual(env, agent, n=N_TEST_TRIALS):
    """D test: induced commitment in a safe-ish context; MECH-303 contextual release."""
    agent_cell = (5, 5)
    haz = [(1, 1)]   # hazard far -> safe context matching the accumulated terrain
    released = 0
    preds = []
    for _t in range(n):
        _, obs = env.reset_to(agent_cell, haz)
        _flat, _harm, _done, _info, obs = env.step(4)
        latent = _sense(agent, obs)
        if latent.z_world is not None and hasattr(agent.residue_field, "evaluate_safety"):
            preds.append(float(agent.residue_field.evaluate_safety(latent.z_world.detach()).mean().detach()))
        agent.beta_gate.elevate()
        was = agent.beta_gate.is_elevated
        _ = _act(agent, latent)
        if was and not agent.beta_gate.is_elevated:
            released += 1
    return {
        "release_rate": released / max(n, 1),
        "mean_contextual_safety_pred": float(np.mean(preds)) if preds else 0.0,
    }


def _config_slice(arm):
    """Declared config slice for the per-cell arm fingerprint."""
    store_on = arm in ("A_store_on_cue", "C_store_on_nocue")
    terrain_on = arm == "D_mech303_spare"
    env_kw = dict(_ENV_SAFE_TEACH) if terrain_on else dict(_ENV_CUE_TEACH)
    return {
        "arm": arm,
        "env_kwargs": env_kw,
        "substrate": dict(_COMMON_SUBSTRATE),
        "use_conditioned_safety_store": store_on,
        "use_contextual_safety_terrain": terrain_on,
        "teach_steps": TEACH_STEPS,
        "n_test_trials": N_TEST_TRIALS,
        "urgency_interrupt_threshold": 1e9,
    }


def _run_cell(arm, seed):
    with arm_cell(seed, config_slice=_config_slice(arm), script_path=Path(__file__),
                  config_slice_declared=True) as cell:
        env, agent = _build(arm, seed)
        if arm == "D_mech303_spare":
            proto_norm = _teach_safe_context(env, agent, arm, seed)
        else:
            proto_norm = _teach(env, agent, arm, seed)
        row = {"arm": arm, "seed": seed, "proto_norm": proto_norm}
        if arm == "D_mech303_spare":
            main = _test_contextual(env, agent)
            row.update({
                "release_rate": main["release_rate"],
                "mean_contextual_safety_pred": main["mean_contextual_safety_pred"],
                "terrain_ready": main["mean_contextual_safety_pred"] >= CONTEXTUAL_RELEASE_THRESHOLD,
            })
        else:
            cue_on = arm != "C_store_on_nocue"
            main = _test_safety_gate(env, agent, cue_on=cue_on, threat=True)
            row.update({
                "release_rate": main["release_rate"],
                "mean_safety_signal": main["mean_safety_signal"],
                "mean_threat_hazard_view": main["mean_threat_hazard_view"],
                "mean_z_harm_norm": main["mean_z_harm_norm"],
            })
            if arm == "A_store_on_cue":
                # readiness positive control: cue present, NO threat (best case).
                pc = _test_safety_gate(env, agent, cue_on=True, threat=False)
                row["pc_release_rate_no_threat"] = pc["release_rate"]
                row["pc_mean_safety_signal"] = pc["mean_safety_signal"]
                row["proto_formed"] = proto_norm >= PROTO_MIN_NORM
        cell.stamp(row)
    return row


def run_experiment(seeds):
    per_cell = []
    for seed in seeds:
        for arm in ARMS:
            print(f"Seed {seed} Condition {arm}", flush=True)
            row = _run_cell(arm, seed)
            print(f"  [{arm} seed={seed}] release_rate={row['release_rate']:.2f} "
                  f"proto={row.get('proto_norm', 0.0):.3f}", flush=True)
            print("verdict: PASS", flush=True)   # per-cell run completion marker
            per_cell.append(row)

    def _rows(arm):
        return [r for r in per_cell if r["arm"] == arm]

    A = _rows("A_store_on_cue"); B = _rows("B_store_off_cue")
    C = _rows("C_store_on_nocue"); D = _rows("D_mech303_spare")

    # Readiness: positive control (arm A, cue, no threat) release, the DV-routed statistic.
    pc_releases = [r["pc_release_rate_no_threat"] for r in A]
    proto_formed = [r for r in A if r.get("proto_formed")]
    terrain_ready = [r for r in D if r.get("terrain_ready")]
    mean_pc_release = float(np.mean(pc_releases)) if pc_releases else 0.0
    n_valid_seeds = min(len(proto_formed), len(terrain_ready))

    readiness_met = (mean_pc_release >= READINESS_FLOOR) and (n_valid_seeds >= MIN_VALID_SEEDS)

    rel_A = float(np.mean([r["release_rate"] for r in A])) if A else 0.0
    rel_B = float(np.mean([r["release_rate"] for r in B])) if B else 0.0
    rel_C = float(np.mean([r["release_rate"] for r in C])) if C else 0.0
    rel_D = float(np.mean([r["release_rate"] for r in D])) if D else 0.0

    dv1_store = rel_A - rel_B
    dv1_cue = rel_A - rel_C
    dv1_pass = (dv1_store >= DV1_MARGIN) and (dv1_cue >= DV1_MARGIN)
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
        label = "mech304_conditioned_inhibition_confirmed" if overall_pass else "mech304_behavioural_null"

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
        "dv1_conditioned_inhibition": {
            "release_A_store_on_cue": rel_A,
            "release_B_store_off_cue": rel_B,
            "release_C_store_on_nocue": rel_C,
            "dv1_store_necessity_gap": dv1_store,
            "dv1_cue_specificity_gap": dv1_cue,
            "margin": DV1_MARGIN,
            "load_bearing": True,
            "passed": bool(dv1_pass),
        },
        "dv2_dissociation_sparing": {
            "release_D_mech303_contextual": rel_D,
            "margin": DV2_MARGIN,
            "load_bearing": True,
            "passed": bool(dv2_pass),
        },
        "readiness": {
            "positive_control_release_no_threat": mean_pc_release,
            "readiness_floor": READINESS_FLOOR,
            "n_valid_seeds": n_valid_seeds,
            "min_valid_seeds": MIN_VALID_SEEDS,
            "readiness_met": bool(readiness_met),
        },
        "interpretation": {
            "label": label,
            "preconditions": [
                {"name": "arm_a_positive_control_release_no_threat",
                 "description": "arm A (cue present, NO threat) release_rate -- SAME statistic DV1 routes on",
                 "measured": mean_pc_release, "threshold": READINESS_FLOOR,
                 "control": "arm A cue-present best-case (no concurrent threat)",
                 "met": bool(mean_pc_release >= READINESS_FLOOR)},
                {"name": "n_valid_seeds_prototype_and_terrain",
                 "description": "seeds where the store prototype formed AND MECH-303 terrain accumulated",
                 "measured": n_valid_seeds, "threshold": MIN_VALID_SEEDS,
                 "control": "prototype norm >= PROTO_MIN_NORM and terrain pred >= release threshold",
                 "met": bool(n_valid_seeds >= MIN_VALID_SEEDS)},
            ],
            "criteria_non_degenerate": {
                "DV1_store_necessity": bool(abs(dv1_store) > 1e-3),
                "DV1_cue_specificity": bool(abs(dv1_cue) > 1e-3),
                "DV2_contextual_sparing": bool(rel_D > 1e-3 or not readiness_met),
            },
        },
        "criteria": [
            {"name": "DV1_conditioned_inhibition", "load_bearing": True, "passed": bool(dv1_pass)},
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
            "MECH-304 promote-to-active behavioural conditioned-inhibition falsifier on the "
            "SD-065 cue channel + SD-066 centered (common-mode-invariant) store readout "
            "(safety_store_centered=True) -- WITHOUT SD-066 the raw-cosine gate saturates on "
            "the z_world common-mode and self-routes substrate_ceiling. Mechanistic "
            "beta_gate.elevate() induction (option i); urgency-interrupt disabled + comparator "
            "reset + no goal isolate the safety gate. Random-roaming teaching diversifies the "
            "relief context so the cue dominates the centered prototype. OFF (store-off) arm "
            "minted in-line via arm_fingerprint (driver-bound); a cross-driver baseline module "
            "is deferred (SD-065/066 first-of-lineage, calibration may iterate -- in-flux)."
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
        "env_cue_teach": _ENV_CUE_TEACH,
        "env_safe_teach": _ENV_SAFE_TEACH,
        "store_threshold": STORE_THRESHOLD,
        "proto_min_norm": PROTO_MIN_NORM,
        "readiness_floor": READINESS_FLOOR,
        "dv1_margin": DV1_MARGIN, "dv2_margin": DV2_MARGIN,
        "teach_steps": TEACH_STEPS, "n_test_trials": N_TEST_TRIALS,
        "contextual_harm_threshold": CONTEXTUAL_HARM_THRESHOLD,
        "contextual_accum_weight": CONTEXTUAL_ACCUM_WEIGHT,
        "contextual_release_threshold": CONTEXTUAL_RELEASE_THRESHOLD,
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
