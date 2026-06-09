#!/opt/local/bin/python3
"""
V3-EXQ-659 -- MECH-074a BLA encoding-gain -> arousal-preferential replay probe.

Targeted probe (proposal EXP-0119 / backlog EVB-0074) for the amygdala read/write
head claim MECH-074, narrowed to the single most-eval-ready sub-claim MECH-074a
(BLAAnalog.encoding_gain). dispatch_mode=targeted_probe, seed_policy=distinct_seeds.

CLAIM UNDER TEST (MECH-074a): "BLA analogue applies an arousal-dependent
multiplicative gain to HippocampalModule write strength." Falsifiable form:
"threat-context recall improves under BLA-modulated gain relative to gain=1; neutral
recall is not harmed by gain>1." Otherwise the encoding-gain arithmetic is
mis-specified.

SUBSTRATE EVAL-PATH WIRING (verified at /queue-experiment Step 2.5, 2026-06-09):
  sense() -> BLAAnalog.tick(z_harm_a) -> encoding_gain (inverted-U over arousal)
  -> agent._episode_bla_peak_encoding_gain -> Trajectory.memory_strength on the
  exploration trace flushed at agent.reset() (agent.py ~2223)
  -> HippocampalModule._sample_exploration_trajectory() multinomial REPLAY-SAMPLING
     weight (module.py ~1584): weights = [memory_strength ...]; idx = multinomial(weights).
  So encoding_gain is genuinely CONSUMED: a higher-arousal episode produces a higher
  encoding_gain -> higher memory_strength -> higher probability that trace is replayed.
  The probe is therefore discriminative, not vacuous. (arousal_tag, the SD-011
  z_harm_a-derived per-trace tag, is written IDENTICALLY in both arms -- it does NOT
  depend on encoding_gain -- so the cross-arm delta isolates the encoding-gain channel.)

DESIGN -- primary-vs-ablation, distinct seeds [42, 43, 44]:
  Both arms identical EXCEPT the encoding-gain channel:
    ARM_PRIMARY  : bla_encoding_gain_max=2.5, floor=1.0 -> natural inverted-U.
    ARM_ABLATION : bla_encoding_gain_max=1.0, floor=1.0 -> encoding_gain == 1.0 always
                   (verified: every inverted-U branch + window_tail collapses to 1.0).
  BLA otherwise fully ON in both arms (retrieval_bias / remap unchanged); only the
  074a encoding-gain -> memory_strength -> replay-sampling-weight pathway differs.
  Per cell: (1) WARMUP trains the SD-011 AffectiveHarmEncoder (compute_harm_accum_loss)
  so ||z_harm_a|| tracks accumulated hazard exposure and crosses the BLA arousal
  threshold under contact; (2) clear the exploration buffer; (3) COLLECT episodes
  (frozen encoder) recording one exploration trace per episode with
  memory_strength=peak encoding_gain and arousal_tag=peak arousal; (4) READOUT: sample
  the buffer K times via the memory_strength-weighted multinomial and measure the
  AROUSAL OVER-REPRESENTATION index AOR = mean(arousal_tag | sampled) - mean(arousal_tag | buffer).
  retrieval_bias is deliberately NOT supplied at readout so the only weight driving the
  sample is memory_strength (= encoding_gain) -- the clean 074a isolation.

PREDICTION:
  PRIMARY  -> AOR > 0 (high-arousal traces preferentially replayed: encoding gain
              correlates with arousal -> higher memory_strength -> over-sampled).
  ABLATION -> AOR ~ 0 (uniform memory_strength -> sampling independent of arousal).

ACCEPTANCE (pre-registered):
  PASS iff (load-bearing) per-seed contrast (AOR_primary - AOR_ablation) > AOR_CONTRAST_MARGIN
  AND AOR_primary > 0 on >= 2/3 seeds, AND (sanity) ablation AOR stays near 0 on the
  majority of seeds, AND (NON-VACUITY) the PRIMARY cells genuinely exercised the
  mechanism: buffer has >= MIN_TRACES traces with arousal_tag std > AROUSAL_STD_FLOOR
  AND memory_strength range > MS_RANGE_FLOOR (encoding_gain actually elevated). If
  non-vacuity fails the run self-routes substrate_not_ready_requeue (NOT a claim verdict).

experiment_purpose: evidence (tests the MECH-074a mechanism directly; tags MECH-074a only
per the claim_ids-accuracy rule -- the parent MECH-074 routes child evidence at governance).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from experiment_protocol import emit_outcome  # noqa: E402
from _harness import StepHarness  # noqa: E402
from _lib.arm_fingerprint import arm_cell  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_659_mech074a_bla_encoding_gain_replay_bias"
QUEUE_ID = "V3-EXQ-659"
CLAIM_IDS: List[str] = ["MECH-074a"]
EXPERIMENT_PURPOSE = "evidence"

SEEDS = [42, 43, 44]

# Encoder dims.
WORLD_DIM = 32
SELF_DIM = 32
HARM_DIM = 32       # z_harm (SD-010 sensory)
HARM_A_DIM = 16     # z_harm_a (SD-011 affective)
HARM_HISTORY_LEN = 10

# Budgets. No warmup phase (see the NO WARMUP note above): the encoder is left
# untrained on purpose, so episodes_per_run == COLLECT_EPISODES.
COLLECT_EPISODES = 40   # alternating threat/neutral; one exploration trace per episode
STEPS_PER_EPISODE = 60
READOUT_SAMPLES = 2000  # multinomial draws over the exploration buffer
TOTAL_EPISODES = COLLECT_EPISODES  # = episodes_per_run

# BLA calibration (identical across arms; only encoding_gain_max distinguishes them).
BLA_AROUSAL_THRESHOLD_ON = 0.4
BLA_AROUSAL_PEAK = 0.7
BLA_WINDOW_STEPS = 5     # short post-event window -> per-episode encoding_gain reflects
                         # THAT episode's arousal (bla.reset() also called per collect ep).

# Pre-registered acceptance thresholds.
MIN_TRACES = 8           # non-vacuity: buffer must have enough traces
AROUSAL_STD_FLOOR = 0.02  # non-vacuity: arousal_tag must vary across traces
MS_RANGE_FLOOR = 0.05    # non-vacuity (PRIMARY): encoding_gain actually elevated
AOR_CONTRAST_MARGIN = 0.02  # per-seed primary-minus-ablation contrast floor
ABLATION_AOR_FLAT_MAX = 0.02  # |AOR_ablation| sanity ceiling
SEEDS_PASS_MIN = 2       # >= 2/3 seeds

# Threat context: SD-022 scheduled limb-damage injection drives RELIABLE,
# policy-independent body damage -> harm_obs_a rises -> ||z_harm_a|| crosses the BLA
# arousal threshold -> encoding_gain elevates (PRIMARY) and arousal_tag > 0. Scheduled
# injection (not just hazard contact) is used because the agent here is UNTRAINED (see
# the no-warmup note below) and a frozen policy does not reliably accumulate enough
# hazard-contact damage to clear the threshold. Same lever SD-050/MECH-302 relief
# experiments use to guarantee harm trajectories regardless of policy.
THREAT_ENV_KWARGS: Dict[str, Any] = dict(
    size=10,
    num_hazards=4,
    num_resources=4,
    hazard_harm=0.05,
    env_drift_interval=5,
    env_drift_prob=0.1,
    proximity_harm_scale=0.1,
    proximity_benefit_scale=0.05,
    proximity_approach_threshold=0.2,
    hazard_field_decay=0.5,
    resource_respawn_on_consume=True,
    use_proxy_fields=True,
    toroidal=False,
    harm_history_len=HARM_HISTORY_LEN,
    limb_damage_enabled=True,
    damage_increment=0.15,
    failure_prob_scale=0.3,
    heal_rate=0.002,
    scheduled_limb_damage_enabled=True,
    scheduled_limb_damage_interval=8,
    scheduled_limb_damage_prob=1.0,
    scheduled_limb_damage_magnitude=0.25,
    scheduled_limb_damage_limb_selection="all",
)
# Neutral context: NO hazards AND no scheduled injection -> no damage -> low
# ||z_harm_a|| (~0.3, below the 0.4 BLA threshold) -> encoding_gain stays at floor (1.0)
# and arousal_tag == 0. Same obs dims as the threat env (hazard field is always
# emitted) so one agent runs in both. The neutral/threat MIX is what produces the
# cross-trace arousal + memory_strength variance the discriminative DV needs (the
# McGaugh/Roozendaal emotional-memory-consolidation paradigm).
#
# NO WARMUP / UNTRAINED ENCODER (deliberate): the MECH-074a mechanism under test is the
# pure BLA arithmetic encoding_gain(||z_harm_a||) -> memory_strength -> replay-sampling
# weight. The untrained AffectiveHarmEncoder already maps harm_obs_a magnitude
# monotonically to ||z_harm_a|| (0 harm -> 0.31; 1.0 harm -> 0.60; 5.0 harm -> 2.36),
# with the RESTING norm (0.31) below the BLA threshold (0.4). Training the encoder on
# the harm-accum aux loss INFLATES the resting norm to ~1.2, which saturates the BLA
# threshold in EVERY context and destroys the threat/neutral arousal contrast. So the
# encoder is left untrained on purpose; the probe tests the BLA gain arithmetic over a
# controlled arousal contrast, not the encoder's learned semantics.
NEUTRAL_ENV_KWARGS: Dict[str, Any] = dict(
    THREAT_ENV_KWARGS, num_hazards=0, scheduled_limb_damage_enabled=False,
)

ARMS: List[Dict[str, Any]] = [
    {"arm_id": "PRIMARY", "bla_encoding_gain_max": 2.5},   # natural inverted-U
    {"arm_id": "ABLATION", "bla_encoding_gain_max": 1.0},  # encoding_gain == 1.0
]


def _make_env(seed: int, kwargs: Dict[str, Any]) -> CausalGridWorldV2:
    return CausalGridWorldV2(seed=seed, **kwargs)


def _build_config(env: CausalGridWorldV2, bla_encoding_gain_max: float) -> REEConfig:
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        alpha_world=0.9,
        world_dim=WORLD_DIM,
        self_dim=SELF_DIM,
        use_harm_stream=True,
        z_harm_dim=HARM_DIM,
        use_affective_harm_stream=True,
        z_harm_a_dim=HARM_A_DIM,
        harm_history_len=HARM_HISTORY_LEN,
        limb_damage_enabled=True,
        damage_increment=float(THREAT_ENV_KWARGS["damage_increment"]),
        failure_prob_scale=float(THREAT_ENV_KWARGS["failure_prob_scale"]),
        heal_rate=float(THREAT_ENV_KWARGS["heal_rate"]),
        # SD-035 amygdala analogue -- BLA on; CeA off (not under test).
        use_amygdala_analog=True,
        use_bla_analog=True,
        use_cea_analog=False,
        bla_encoding_gain_max=float(bla_encoding_gain_max),
        bla_encoding_gain_floor=1.0,
        bla_arousal_threshold_on=BLA_AROUSAL_THRESHOLD_ON,
        bla_arousal_peak=BLA_AROUSAL_PEAK,
        bla_window_steps=BLA_WINDOW_STEPS,
        # MECH-165 / SD-035 read/write head: exploration-trace recording on so
        # encoding_gain lands as memory_strength on each flushed trajectory.
        replay_diversity_enabled=True,
    )
    cfg.residue.valence_enabled = True
    return cfg


def _collect_traces(
    agent: REEAgent,
    threat_env: CausalGridWorldV2,
    neutral_env: CausalGridWorldV2,
    *,
    num_episodes: int,
    steps_per_episode: int,
    seed: int,
    label: str,
) -> Dict[str, float]:
    """Run frozen-encoder episodes ALTERNATING threat/neutral context, recording one
    exploration trace per episode. Each flushed trace carries memory_strength=peak
    encoding_gain and arousal_tag=peak z_harm_a-derived tag for that episode -- so the
    threat (high-arousal) traces and neutral (low-arousal) traces produce the
    cross-trace variance the discriminative DV needs."""
    # Clear any warmup residue so the measured buffer is the COLLECT regime only.
    agent.hippocampal._exploration_buffer.clear()
    agent.eval()
    max_threat_zharm_a = 0.0
    max_neutral_zharm_a = 0.0
    n_threat = 0
    n_neutral = 0
    for ep in range(num_episodes):
        is_threat = (ep % 2 == 0)
        env = threat_env if is_threat else neutral_env
        harness = StepHarness(agent, env, train_mode=False, seed=seed + ep)
        _, obs_dict = env.reset()
        agent.reset()
        if agent.bla is not None:
            agent.bla.reset()
        harness.reset()
        for _ in range(steps_per_episode):
            result = harness.step(obs_dict)
            lat = result.latent
            if lat is not None and getattr(lat, "z_harm_a", None) is not None:
                zn = float(torch.linalg.norm(lat.z_harm_a.detach().flatten()).item())
                if is_threat:
                    max_threat_zharm_a = max(max_threat_zharm_a, zn)
                else:
                    max_neutral_zharm_a = max(max_neutral_zharm_a, zn)
            obs_dict = result.next_obs_dict
            if result.done:
                break
        if is_threat:
            n_threat += 1
        else:
            n_neutral += 1
        if (ep + 1) % 10 == 0 or (ep + 1) == num_episodes:
            print(f"  [train] {label} ep {ep + 1}/{TOTAL_EPISODES}", flush=True)
    # Final flush of the last episode's trajectory.
    agent.reset()
    n_gain_elev = int(getattr(agent.bla, "_n_gain_elevations", 0)) if agent.bla else 0
    return {
        "max_threat_zharm_a_norm": max_threat_zharm_a,
        "max_neutral_zharm_a_norm": max_neutral_zharm_a,
        "n_threat_episodes": n_threat,
        "n_neutral_episodes": n_neutral,
        "n_gain_elevations": n_gain_elev,
    }


def _readout(agent: REEAgent, *, seed: int, n_samples: int) -> Dict[str, Any]:
    """Sample the exploration buffer by the memory_strength-weighted multinomial
    (the wired MECH-074a replay-sampling path) and measure arousal over-representation."""
    buf = agent.hippocampal._exploration_buffer
    n_traces = len(buf)
    buffer_arousal = [float(getattr(t, "arousal_tag", 0.0)) for t in buf]
    buffer_ms = [float(getattr(t, "memory_strength", 1.0)) for t in buf]

    result: Dict[str, Any] = {
        "n_traces": n_traces,
        "buffer_mean_arousal": float(np.mean(buffer_arousal)) if n_traces else 0.0,
        "buffer_std_arousal": float(np.std(buffer_arousal)) if n_traces else 0.0,
        "memory_strength_min": float(min(buffer_ms)) if n_traces else 1.0,
        "memory_strength_max": float(max(buffer_ms)) if n_traces else 1.0,
        "memory_strength_range": (float(max(buffer_ms) - min(buffer_ms)) if n_traces else 0.0),
        "sampled_mean_arousal": 0.0,
        "arousal_over_representation": 0.0,
    }
    if n_traces < 2:
        return result

    # Deterministic readout RNG (seed-derived) for reproducibility.
    torch.manual_seed(seed + 100000)
    sampled_arousal: List[float] = []
    for _ in range(n_samples):
        # retrieval_bias deliberately omitted -> only memory_strength drives the
        # weight, isolating the MECH-074a encoding-gain channel.
        traj = agent.hippocampal._sample_exploration_trajectory()
        if traj is not None:
            sampled_arousal.append(float(getattr(traj, "arousal_tag", 0.0)))
    if sampled_arousal:
        s_mean = float(np.mean(sampled_arousal))
        result["sampled_mean_arousal"] = s_mean
        result["arousal_over_representation"] = s_mean - result["buffer_mean_arousal"]
    return result


def _run_cell(seed: int, arm: Dict[str, Any], *, dry: bool) -> Dict[str, Any]:
    arm_id = arm["arm_id"]
    print(f"Seed {seed} Condition {arm_id}", flush=True)

    collect_eps = 8 if dry else COLLECT_EPISODES
    steps = 20 if dry else STEPS_PER_EPISODE
    n_samples = 200 if dry else READOUT_SAMPLES

    config_slice = {
        "arm_id": arm_id,
        "bla_encoding_gain_max": arm["bla_encoding_gain_max"],
        "bla_encoding_gain_floor": 1.0,
        "bla_arousal_threshold_on": BLA_AROUSAL_THRESHOLD_ON,
        "world_dim": WORLD_DIM,
        "harm_a_dim": HARM_A_DIM,
        "collect_eps": collect_eps,
        "steps_per_episode": steps,
    }
    label = f"mech074a seed={seed} arm={arm_id}"
    with arm_cell(
        seed,
        config_slice=config_slice,
        script_path=Path(__file__),
        extra_ineligible_reasons=["stateful_replay_buffer_readout"],
    ) as cell:
        threat_env = _make_env(seed, THREAT_ENV_KWARGS)
        neutral_env = _make_env(seed + 1, NEUTRAL_ENV_KWARGS)
        cfg = _build_config(threat_env, arm["bla_encoding_gain_max"])
        agent = REEAgent(cfg)

        coll = _collect_traces(
            agent, threat_env, neutral_env,
            num_episodes=collect_eps, steps_per_episode=steps, seed=seed, label=label,
        )
        read = _readout(agent, seed=seed, n_samples=n_samples)

        row: Dict[str, Any] = {
            "seed": int(seed),
            "arm": arm_id,
            **coll,
            **read,
        }
        cell.stamp(row)

    # Per-cell verdict (progress only; cohort eval is authoritative).
    aor = float(row["arousal_over_representation"])
    if arm_id == "PRIMARY":
        cell_pass = aor > 0.0 and row["n_traces"] >= MIN_TRACES
    else:
        cell_pass = abs(aor) <= ABLATION_AOR_FLAT_MAX
    print(f"verdict: {'PASS' if cell_pass else 'FAIL'}", flush=True)
    row["cell_pass"] = bool(cell_pass)
    return row


def _evaluate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_seed: Dict[int, Dict[str, Dict[str, Any]]] = {}
    for r in rows:
        by_seed.setdefault(int(r["seed"]), {})[r["arm"]] = r

    per_seed: List[Dict[str, Any]] = []
    primary_vacuity_ok = 0
    ablation_flat_ok = 0
    seeds_pass = 0
    for seed, arms in sorted(by_seed.items()):
        p = arms.get("PRIMARY")
        a = arms.get("ABLATION")
        if p is None or a is None:
            continue
        aor_p = float(p["arousal_over_representation"])
        aor_a = float(a["arousal_over_representation"])
        contrast = aor_p - aor_a
        vac_ok = (
            int(p["n_traces"]) >= MIN_TRACES
            and float(p["buffer_std_arousal"]) > AROUSAL_STD_FLOOR
            and float(p["memory_strength_range"]) > MS_RANGE_FLOOR
        )
        flat_ok = abs(aor_a) <= ABLATION_AOR_FLAT_MAX
        seed_pass = bool(contrast > AOR_CONTRAST_MARGIN and aor_p > 0.0 and vac_ok)
        if vac_ok:
            primary_vacuity_ok += 1
        if flat_ok:
            ablation_flat_ok += 1
        if seed_pass:
            seeds_pass += 1
        per_seed.append({
            "seed": seed,
            "aor_primary": aor_p,
            "aor_ablation": aor_a,
            "contrast": contrast,
            "primary_n_traces": int(p["n_traces"]),
            "primary_buffer_std_arousal": float(p["buffer_std_arousal"]),
            "primary_memory_strength_range": float(p["memory_strength_range"]),
            "primary_n_gain_elevations": int(p.get("n_gain_elevations", 0)),
            "non_vacuity_ok": vac_ok,
            "ablation_flat_ok": flat_ok,
            "seed_pass": seed_pass,
        })

    n_seeds = len(per_seed)
    # Non-vacuity is a precondition on the PRIMARY cells: the mechanism must have
    # been exercised on a majority of seeds, else the test is starved (not falsified).
    non_vacuity_met = primary_vacuity_ok >= SEEDS_PASS_MIN
    contrast_met = seeds_pass >= SEEDS_PASS_MIN
    ablation_flat_met = ablation_flat_ok >= SEEDS_PASS_MIN

    if not non_vacuity_met:
        outcome = "FAIL"
        label = "substrate_not_ready_requeue"
        evidence_direction = "non_contributory"
    elif contrast_met and ablation_flat_met:
        outcome = "PASS"
        label = "mech074a_encoding_gain_biases_replay_supports"
        evidence_direction = "supports"
    else:
        outcome = "FAIL"
        label = "mech074a_encoding_gain_no_replay_bias_weakens"
        evidence_direction = "weakens"

    return {
        "outcome": outcome,
        "interpretation_label": label,
        "evidence_direction": evidence_direction,
        "n_seeds": n_seeds,
        "seeds_pass": seeds_pass,
        "primary_vacuity_ok": primary_vacuity_ok,
        "ablation_flat_ok": ablation_flat_ok,
        "non_vacuity_met": non_vacuity_met,
        "contrast_met": contrast_met,
        "ablation_flat_met": ablation_flat_met,
        "per_seed": per_seed,
    }


def _utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def main(dry_run: bool = False) -> "Tuple[str, Path] | int":
    seeds = [SEEDS[0]] if dry_run else SEEDS
    rows: List[Dict[str, Any]] = []
    t0 = time.time()
    for seed in seeds:
        for arm in ARMS:
            rows.append(_run_cell(seed, arm, dry=dry_run))

    acceptance = _evaluate(rows)
    outcome = acceptance["outcome"]
    elapsed = time.time() - t0

    # Non-vacuity precondition + load-bearing criterion for the adjudication gate.
    primary_rows = [r for r in rows if r["arm"] == "PRIMARY"]
    measured_ms_range = (
        max((float(r["memory_strength_range"]) for r in primary_rows), default=0.0)
    )
    measured_arousal_std = (
        max((float(r["buffer_std_arousal"]) for r in primary_rows), default=0.0)
    )
    interpretation = {
        "label": acceptance["interpretation_label"],
        "preconditions": [
            {
                "name": "primary_encoding_gain_elevated",
                "description": (
                    "PRIMARY-arm exploration buffer memory_strength range "
                    "(= encoding_gain spread) clears the floor on a known-positive "
                    "control (the natural inverted-U arm under hazard arousal)."
                ),
                "measured": measured_ms_range,
                "threshold": MS_RANGE_FLOOR,
                "control": "PRIMARY arm (bla_encoding_gain_max=2.5) under hazard contact",
                "met": bool(acceptance["non_vacuity_met"]),
            },
            {
                "name": "primary_arousal_variance_present",
                "description": (
                    "PRIMARY-arm buffer arousal_tag std clears the floor (a range of "
                    "arousal to be over-represented)."
                ),
                "measured": measured_arousal_std,
                "threshold": AROUSAL_STD_FLOOR,
                "control": "PRIMARY arm buffer across collect episodes",
                "met": bool(measured_arousal_std > AROUSAL_STD_FLOOR),
            },
        ],
        "criteria": [
            {
                "name": "primary_minus_ablation_contrast",
                "load_bearing": True,
                "passed": bool(acceptance["contrast_met"]),
            },
            {
                "name": "ablation_aor_flat",
                "load_bearing": False,
                "passed": bool(acceptance["ablation_flat_met"]),
            },
        ],
        "criteria_non_degenerate": {
            "primary_minus_ablation_contrast": bool(acceptance["non_vacuity_met"]),
            "ablation_aor_flat": bool(acceptance["primary_vacuity_ok"] >= 1),
        },
    }

    if dry_run:
        print(f"[{EXPERIMENT_TYPE}] dry-run outcome={outcome} "
              f"label={acceptance['interpretation_label']}", flush=True)
        for ps in acceptance["per_seed"]:
            print(f"  seed={ps['seed']} aor_p={ps['aor_primary']:.4f} "
                  f"aor_a={ps['aor_ablation']:.4f} contrast={ps['contrast']:.4f} "
                  f"vac_ok={ps['non_vacuity_ok']} ms_range="
                  f"{ps['primary_memory_strength_range']:.4f} "
                  f"gain_elev={ps['primary_n_gain_elevations']}", flush=True)
        return 0

    run_id = f"{EXPERIMENT_TYPE}_{_utc_compact()}_v3"
    out_dir = (
        REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.json"
    manifest = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "backlog_id": "EVB-0074",
        "proposal_id": "EXP-0119",
        "claim_ids": CLAIM_IDS,
        "claim_ids_tested": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "evidence_class": "substrate_mechanism",
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": _utc_compact(),
        "outcome": outcome,
        "evidence_direction": acceptance["evidence_direction"],
        "dispatch_mode": "targeted_probe",
        "seed_policy": "distinct_seeds",
        "seeds": SEEDS,
        "interpretation": interpretation,
        "acceptance": acceptance,
        "thresholds": {
            "MIN_TRACES": MIN_TRACES,
            "AROUSAL_STD_FLOOR": AROUSAL_STD_FLOOR,
            "MS_RANGE_FLOOR": MS_RANGE_FLOOR,
            "AOR_CONTRAST_MARGIN": AOR_CONTRAST_MARGIN,
            "ABLATION_AOR_FLAT_MAX": ABLATION_AOR_FLAT_MAX,
            "SEEDS_PASS_MIN": SEEDS_PASS_MIN,
            "BLA_AROUSAL_THRESHOLD_ON": BLA_AROUSAL_THRESHOLD_ON,
            "BLA_AROUSAL_PEAK": BLA_AROUSAL_PEAK,
            "BLA_WINDOW_STEPS": BLA_WINDOW_STEPS,
        },
        "summary": (
            f"MECH-074a BLA encoding-gain replay-bias probe. outcome={outcome}; "
            f"label={acceptance['interpretation_label']}; seeds_pass="
            f"{acceptance['seeds_pass']}/{acceptance['n_seeds']}; "
            f"primary_vacuity_ok={acceptance['primary_vacuity_ok']}."
        ),
        "arm_results": rows,
        "elapsed_seconds": elapsed,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    return outcome, out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    res = main(dry_run=args.dry_run)
    if res == 0:
        sys.exit(0)
    _outcome, _out_path = res
    emit_outcome(
        outcome=_outcome if _outcome in ("PASS", "FAIL") else "FAIL",
        manifest_path=_out_path,
    )
    sys.exit(0)
