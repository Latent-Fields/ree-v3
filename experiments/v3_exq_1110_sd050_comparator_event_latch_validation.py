"""V3-EXQ-1110: SD-050 suffering-comparator event latch -- substrate validation.

Red-team (fable, 2026-09-26): CONTESTED -> F1 fixed (C5 counts relief-block CALLS;
VALENCE_LIKING writes are no-ops here, now recorded separately), F2 fixed (PASS reads
"one event per shadow burst"; bursts are delimited by norm rises, see WHAT C1 TESTS),
F3/F4 dismissed as tripwires (legacy bit-identity is carried by contract R2's verbatim
oracle), F5 fixed (C5 split per arm).

PURPOSE (diagnostic, substrate readiness -- not claim evidence).
Validates the suffering-derivative-comparator-refractory build (2026-09-26,
IGW-20260924-224; ree-v3/docs/substrate/SD-050-suffering-comparator-event-latch.md).
The unlatched SufferingDerivativeComparator fires on EVERY tick whose rolling-window
drop clears the threshold, so one damage->heal descent emits a TRAIN of relief events
(~9-17 per scheduled injection in V3-EXQ-517d; failure_autopsy_gflag0452-D1-cluster).
With suffering_event_latch_enabled=True it should emit ONE event per descent.

DESIGN. Two arms on the real agent loop, same seeds, same env:
  ARM_LATCH_OFF: comparator ON, latch OFF (legacy per-tick firing)
  ARM_LATCH_ON : comparator ON, latch ON  (re-arm-on-rise, rearm_rise=drop_threshold)
Arm trajectories MAY diverge after the first event (an event releases beta when it is
elevated), so cross-arm event counts are NOT treated as a paired comparison. The decisive
DV is therefore WITHIN-ARM and PAIRED: in every arm a SHADOW unlatched comparator
(instrument only; identical parameters, never read by the agent) is ticked on the
exact z_harm_a norm the agent's comparator sees. Shadow fires are grouped into
BURSTS (consecutive shadow fires less than COMP_WINDOW_LENGTH ticks apart = one
descent's train).

WHAT C1 TESTS (red-team F2). With limb_damage_enabled, harm_obs_a IS the limb-damage
vector (causal_grid_world.py harm_obs_a re-source, SD-022), healed monotonically by
heal_rate; the norm rises only on damage (scheduled injection or hazard contact). So a
shadow burst boundary and the latch's re-arm condition are the SAME event (a rise), and
C1/C2 test that the latch (a) engages at all (else fires/burst == the train ratio) and
(b) does not re-arm on sub-threshold jitter (else fires/burst > 1) or over-suppress a
genuine new onset (else fires/burst < 1). They do not test a descent segmentation richer
than "delimited by a norm rise". Reconciliation diagnostics per cell: rearm_count,
suppressed_count, injections_with_norm_rise (injections whose next-tick norm rose by at
least COMP_DROP_THRESHOLD).

VALENCE WRITES (red-team F1). ResidueField.update_valence is a silent no-op while no RBF
center is active (residue/field.py _nearest_active_center), and this loop never calls
agent.update_residue, so the relief block's VALENCE_LIKING call writes nothing. C5
therefore checks one relief-block CALL per event (the pipeline is reached), and the
manifest records inner_valence_writes + residue_active_any separately. This is also true
of V3-EXQ-517c/517d's "writes" metric (same loop shape).

ENVIRONMENT. 517d's CausalGridWorldV2 + SD-022 scheduled-limb-damage curriculum,
but num_hazards=1 and contamination_spread=0.0. Measured 2026-09-26 on the current
substrate: at 517d's num_hazards=3 (contamination on) every episode ends in death
(health 0) 12-75 steps in, BEFORE any heal descent, and the 517d driver itself
records 0 events in both comparator arms -- the DV cannot move there (INERT). At
num_hazards=1 / contamination 0.0 episodes last 140-300 steps and real descents
clear the 0.005 threshold (max 30-tick drop 0.0065-0.0138 in 3/12 probe episodes).

CRITERIA (pre-registered; thresholds calibrated on probe seeds 42/43 and run on
held-out seeds 45/46/47):
  P0 readiness (ON arm, worst seed): shadow train ratio = shadow_fires /
     shadow_bursts >= P0_MIN_TRAIN_RATIO and shadow_bursts >= P0_MIN_BURSTS -- the
     unlatched train the latch must compress actually exists on this stream.
     Below-floor -> substrate_not_ready_requeue.
  C1 (load-bearing) ON arm, worst seed: latched_fires / shadow_bursts <=
     C1_MAX_FIRES_PER_BURST (one event per descent, allowing genuine re-onsets
     inside a burst).
  C2 ON arm, worst seed: latched_fires / shadow_bursts >= C2_MIN_FIRES_PER_BURST
     (the latch does not over-suppress: most descents still produce an event).
  C3 ON arm, all seeds: every latched fire tick is also a shadow fire tick
     (strict-subset invariant, same stream).
  C4 OFF arm, all seeds: agent fire ticks == shadow fire ticks, tick for tick
     (latch-OFF is bit-identical to legacy inside the agent loop).
  C5 each arm, all seeds: relief-block VALENCE_LIKING CALLS == comparator fires (the
     relief pipeline is reached exactly once per event; see VALENCE WRITES).
  C3 / C4 are tripwires: C3 holds by the latch's construction and C4 compares two
     instances of the same class fed the same scalar; legacy bit-identity is carried by
     contract R2 (verbatim pre-latch oracle). A C3/C4 failure means an in-loop wiring
     break (a different norm or reset reaching the agent's comparator).
PASS = P0 met AND C1..C5 all true.

claim_ids = ["SD-050"]: the latch amends SD-050's comparator. Diagnostic purpose ->
excluded from governance scoring. Does NOT test MECH-302's action-contingency (D2):
relief here is passive heal_rate decay of limb damage, not agent-caused
(registered as substrate_queue action-contingent-relief-provider).

SEED RISK (red-team F7): P0 is worst-seed; a seed whose 40 episodes yield < 5 bursts
routes the whole run to substrate_not_ready_requeue even if the other seeds pass (the
per-seed rows keep that attributable). Calibration: 8 and 20 bursts in 20 episodes.
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ree_core.agent import REEAgent  # noqa: E402
from ree_core.comparator.suffering_derivative_comparator import (  # noqa: E402
    SufferingDerivativeComparator,
)
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from experiment_protocol import emit_outcome  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_1110_sd050_comparator_event_latch_validation"
QUEUE_ID = "V3-EXQ-1110"
CLAIM_IDS = ["SD-050"]
EXPERIMENT_PURPOSE = "diagnostic"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"

# Pre-registered thresholds (calibrated on probe seeds 42/43; see docstring).
P0_MIN_TRAIN_RATIO = 2.0
P0_MIN_BURSTS = 5
C1_MAX_FIRES_PER_BURST = 1.5
C2_MIN_FIRES_PER_BURST = 0.5

SEEDS = [45, 46, 47]
EPISODES = 40
STEPS_PER_EPISODE = 300
PRINT_INTERVAL = 5

GRID_SIZE = 12
N_HAZARDS = 1
N_RESOURCES = 2
CONTAMINATION_SPREAD = 0.0
BODY_OBS_DIM = 17
WORLD_OBS_DIM = 250
HARM_OBS_A_DIM = 7
ACTION_DIM = 5

COMP_WINDOW_LENGTH = 30
COMP_DROP_THRESHOLD = 0.005
COMP_MIN_INITIAL_NORM = 0.01

SCHED_INTERVAL = 50
SCHED_PROB = 0.5
SCHED_MAGNITUDE = 0.4
SCHED_LIMB_SELECTION = "random"

ARMS = [("ARM_LATCH_OFF", False), ("ARM_LATCH_ON", True)]

# The two readiness preconditions ARE the degeneracy definition of C1/C2 (the
# unlatched train the latch must compress exists on this stream), not a narrower
# hand-written signature of some other state. Reached on calibration seeds 42/43
# (20 episodes): shadow train ratio 7.1 / 29.5 vs floor 2.0; shadow bursts 8 / 20
# vs floor 5.
ANCHOR_REACHABILITY_EXEMPT = (
    "preconditions are the C1/C2 degeneracy definition; reached on calibration "
    "seeds 42/43 (ratio 7.1/29.5 >= 2.0, bursts 8/20 >= 5)"
)

_ZG = ZGoalStreamAccumulator()


def make_config(latch_on: bool) -> REEConfig:
    return REEConfig.from_dims(
        body_obs_dim=BODY_OBS_DIM,
        world_obs_dim=WORLD_OBS_DIM,
        action_dim=ACTION_DIM,
        alpha_world=0.3,  # 517d parity; the DV reads z_harm_a, not z_world
        use_affective_harm_stream=True,
        harm_obs_a_dim=HARM_OBS_A_DIM,
        use_suffering_derivative_comparator=True,
        suffering_window_length=COMP_WINDOW_LENGTH,
        suffering_drop_threshold=COMP_DROP_THRESHOLD,
        suffering_min_initial_norm=COMP_MIN_INITIAL_NORM,
        valence_liking_enabled=True,
        relief_completion_weight=1.0,
        suffering_event_latch_enabled=latch_on,
        suffering_rearm_rise=None,
    )


def make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        size=GRID_SIZE,
        num_hazards=N_HAZARDS,
        num_resources=N_RESOURCES,
        seed=seed,
        contamination_spread=CONTAMINATION_SPREAD,
        limb_damage_enabled=True,
        damage_increment=0.15,
        failure_prob_scale=0.3,
        heal_rate=0.002,
        scheduled_limb_damage_enabled=True,
        scheduled_limb_damage_interval=SCHED_INTERVAL,
        scheduled_limb_damage_prob=SCHED_PROB,
        scheduled_limb_damage_magnitude=SCHED_MAGNITUDE,
        scheduled_limb_damage_limb_selection=SCHED_LIMB_SELECTION,
    )


def count_bursts(ticks, gap):
    n, last = 0, None
    for t in ticks:
        if last is None or t - last >= gap:
            n += 1
        last = t
    return n


def step_agent(agent, env, obs_dict, shadow):
    """sense -> (shadow tick) -> clock -> e1 -> generate -> select_action -> env.step."""
    body = obs_dict["body_state"]
    world = obs_dict["world_state"]
    if body.dim() == 1:
        body = body.unsqueeze(0)
    if world.dim() == 1:
        world = world.unsqueeze(0)
    sense_kwargs = {"obs_body": body, "obs_world": world}
    obs_harm = obs_dict.get("harm_obs")
    if obs_harm is not None:
        sense_kwargs["obs_harm"] = obs_harm.unsqueeze(0) if obs_harm.dim() == 1 else obs_harm
    obs_harm_a = obs_dict.get("harm_obs_a")
    if obs_harm_a is not None:
        sense_kwargs["obs_harm_a"] = obs_harm_a.unsqueeze(0) if obs_harm_a.dim() == 1 else obs_harm_a

    writes = [0]
    inner = [0]
    orig_uv = agent.residue_field.update_valence
    orig_inner = agent.residue_field.rbf_field.update_valence

    def _patched_uv(*args, **kwargs):
        writes[0] += 1
        return orig_uv(*args, **kwargs)

    def _patched_inner(*args, **kwargs):
        inner[0] += 1
        return orig_inner(*args, **kwargs)

    agent.residue_field.update_valence = _patched_uv
    agent.residue_field.rbf_field.update_valence = _patched_inner
    try:
        with torch.no_grad():
            latent = agent.sense(**sense_kwargs)
            event_fired = bool(getattr(agent, "_relief_completion_event", False))
            norm = float(latent.z_harm_a.norm().item())
            sim_mode = bool(getattr(latent, "hypothesis_tag", False))
            shadow_fired = shadow.tick(norm, sim_mode)
            beta_elevated_at_event = bool(event_fired and agent.beta_gate.is_elevated)
            ticks = agent.clock.advance()
            world_dim = agent.config.latent.world_dim
            e1_prior = (
                agent._e1_tick(latent) if ticks.get("e1_tick", True)
                else torch.zeros(1, world_dim)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            action = agent.select_action(candidates, ticks)
    finally:
        agent.residue_field.update_valence = orig_uv
        agent.residue_field.rbf_field.update_valence = orig_inner

    action_idx = int(action.argmax(dim=-1).item()) if action is not None else 0
    _flat, _harm, done, info, next_obs = env.step(action_idx)
    return {
        "event_fired": event_fired,
        "shadow_fired": bool(shadow_fired),
        "norm": norm,
        "valence_writes": writes[0],
        "inner_valence_writes": inner[0],
        "latched_after": bool(getattr(agent.suffering_comparator, "latched", False)),
        "beta_release_at_event": beta_elevated_at_event,
        "injected": bool(info.get("scheduled_limb_damage_injected_this_step", False)),
        "done": bool(done),
        "next_obs": next_obs,
    }


def run_cell(seed, arm_label, latch_on, n_episodes, full_config):
    with arm_cell(seed, config_slice={**full_config, "arm": arm_label},
                  script_path=Path(__file__)) as cell:
        agent = REEAgent(make_config(latch_on))
        agent.reset()
        fires = shadow_fires = bursts = writes = releases = injections = steps = 0
        subset_violations = off_identity_violations = 0
        inner_writes = rearms = suppressed = inj_with_rise = 0
        per_episode = []
        norm_min, norm_max, max_drop_w = float("inf"), float("-inf"), float("-inf")
        for ep in range(n_episodes):
            env = make_env(seed * 10000 + ep)
            _flat, obs = env.reset()
            agent.reset()
            shadow = SufferingDerivativeComparator(
                COMP_WINDOW_LENGTH, COMP_DROP_THRESHOLD, COMP_MIN_INITIAL_NORM,
                latch_enabled=False,
            )
            f_ticks, s_ticks, norms = [], [], []
            ep_writes = ep_injections = 0
            prev_latched = False
            pending_inj = False
            for t in range(STEPS_PER_EPISODE):
                d = step_agent(agent, env, obs, shadow)
                obs = d["next_obs"]
                steps += 1
                if pending_inj and len(norms) > 0 and d["norm"] - norms[-1] >= COMP_DROP_THRESHOLD:
                    inj_with_rise += 1
                pending_inj = d["injected"]
                norms.append(d["norm"])
                inner_writes += d["inner_valence_writes"]
                if prev_latched and not d["latched_after"]:
                    rearms += 1
                prev_latched = d["latched_after"]
                if d["event_fired"]:
                    f_ticks.append(t)
                    releases += int(d["beta_release_at_event"])
                    if not d["shadow_fired"]:
                        subset_violations += 1
                if d["shadow_fired"]:
                    s_ticks.append(t)
                if (not latch_on) and d["event_fired"] != d["shadow_fired"]:
                    off_identity_violations += 1
                ep_writes += d["valence_writes"]
                ep_injections += int(d["injected"])
                if d["done"]:
                    break
            suppressed += agent.suffering_comparator.suppressed_count
            ep_bursts = count_bursts(s_ticks, COMP_WINDOW_LENGTH)
            fires += len(f_ticks)
            shadow_fires += len(s_ticks)
            bursts += ep_bursts
            writes += ep_writes
            injections += ep_injections
            norm_min = min(norm_min, min(norms))
            norm_max = max(norm_max, max(norms))
            w = COMP_WINDOW_LENGTH
            if len(norms) >= w:
                max_drop_w = max(max_drop_w, max(norms[i] - norms[i + w - 1]
                                                 for i in range(len(norms) - w + 1)))
            per_episode.append({
                "ep": ep, "len": len(norms), "fires": len(f_ticks),
                "shadow_fires": len(s_ticks), "shadow_bursts": ep_bursts,
                "writes": ep_writes, "injections": ep_injections,
            })
            if (ep + 1) % PRINT_INTERVAL == 0:
                print(f"  [train] seed={seed} arm={arm_label} ep {ep+1}/{n_episodes} "
                      f"fires={fires} shadow_fires={shadow_fires} bursts={bursts}",
                      flush=True)
        _ZG.observe(agent)
        row = {
            "seed": seed, "arm": arm_label, "latch_on": latch_on,
            "episodes": n_episodes, "steps": steps,
            "fires": fires, "shadow_fires": shadow_fires, "shadow_bursts": bursts,
            "writes": writes, "beta_releases_at_event": releases,
            "inner_valence_writes": inner_writes,
            "residue_active_any": bool(agent.residue_field.rbf_field.active_mask.any()),
            "rearm_count": rearms, "suppressed_count": suppressed,
            "injections_with_norm_rise": inj_with_rise,
            "curriculum_injections": injections,
            "subset_violations": subset_violations,
            "off_identity_violations": off_identity_violations,
            "shadow_train_ratio": (shadow_fires / bursts) if bursts else None,
            "fires_per_burst": (fires / bursts) if bursts else None,
            "norm_min": norm_min, "norm_max": norm_max,
            "max_window_drop": max_drop_w if max_drop_w > float("-inf") else None,
            "per_episode": per_episode,
        }
        cell.stamp(row)
    return row


def _worst(rows, key, fn):
    vals = [(r[key], r["seed"]) for r in rows if r[key] is not None]
    if len(vals) < len(rows):
        return None, [r["seed"] for r in rows if r[key] is None][0]
    v = fn(vals, key=lambda x: x[0])
    return v[0], v[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    # Calibration-only overrides (the queued run uses the pre-registered defaults).
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--episodes", type=int, default=None)
    args = parser.parse_args()
    dry_run = args.dry_run
    t0 = time.perf_counter()
    timestamp_utc = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{timestamp_utc}_v3"
    seeds = args.seeds or (SEEDS[:1] if dry_run else SEEDS)
    n_episodes = args.episodes or (2 if dry_run else EPISODES)

    full_config = {
        "episodes": n_episodes, "steps_per_episode": STEPS_PER_EPISODE,
        "grid_size": GRID_SIZE, "num_hazards": N_HAZARDS, "num_resources": N_RESOURCES,
        "contamination_spread": CONTAMINATION_SPREAD,
        "body_obs_dim": BODY_OBS_DIM, "world_obs_dim": WORLD_OBS_DIM,
        "harm_obs_a_dim": HARM_OBS_A_DIM, "action_dim": ACTION_DIM,
        "comp_window_length": COMP_WINDOW_LENGTH,
        "comp_drop_threshold": COMP_DROP_THRESHOLD,
        "comp_min_initial_norm": COMP_MIN_INITIAL_NORM,
        "suffering_rearm_rise": None,
        "scheduled_limb_damage_interval": SCHED_INTERVAL,
        "scheduled_limb_damage_prob": SCHED_PROB,
        "scheduled_limb_damage_magnitude": SCHED_MAGNITUDE,
        "scheduled_limb_damage_limb_selection": SCHED_LIMB_SELECTION,
        "heal_rate": 0.002, "damage_increment": 0.15, "failure_prob_scale": 0.3,
        "alpha_world": 0.3,
    }

    print(f"{QUEUE_ID}: SD-050 comparator event latch validation", flush=True)
    rows = []
    for seed in seeds:
        for arm_label, latch_on in ARMS:
            print(f"Seed {seed} Condition {arm_label}", flush=True)
            row = run_cell(seed, arm_label, latch_on, n_episodes, full_config)
            rows.append(row)
            cell_ok = (row["subset_violations"] == 0
                       and row["off_identity_violations"] == 0
                       and row["writes"] == row["fires"])
            print(f"  fires={row['fires']} shadow_fires={row['shadow_fires']} "
                  f"bursts={row['shadow_bursts']} writes={row['writes']} "
                  f"steps={row['steps']} injections={row['curriculum_injections']}",
                  flush=True)
            print(f"verdict: {'PASS' if cell_ok else 'FAIL'}", flush=True)

    on_rows = [r for r in rows if r["latch_on"]]
    off_rows = [r for r in rows if not r["latch_on"]]

    p0_ratio, p0_ratio_seed = _worst(on_rows, "shadow_train_ratio", min)
    p0_bursts, p0_bursts_seed = _worst(on_rows, "shadow_bursts", min)
    p0_met = (p0_ratio is not None and p0_ratio >= P0_MIN_TRAIN_RATIO
              and p0_bursts is not None and p0_bursts >= P0_MIN_BURSTS)
    c1_val, c1_seed = _worst(on_rows, "fires_per_burst", max)
    c2_val, c2_seed = _worst(on_rows, "fires_per_burst", min)
    c1 = c1_val is not None and c1_val <= C1_MAX_FIRES_PER_BURST
    c2 = c2_val is not None and c2_val >= C2_MIN_FIRES_PER_BURST
    c3_viol = sum(r["subset_violations"] for r in on_rows)
    c4_viol = sum(r["off_identity_violations"] for r in off_rows)
    c5_on = sum(abs(r["writes"] - r["fires"]) for r in on_rows)
    c5_off = sum(abs(r["writes"] - r["fires"]) for r in off_rows)
    c5_mismatch = c5_on + c5_off
    c3, c4, c5 = c3_viol == 0, c4_viol == 0, c5_mismatch == 0
    overall = bool(p0_met and c1 and c2 and c3 and c4 and c5)

    if not p0_met:
        label = "substrate_not_ready_requeue"
    elif overall:
        label = "latch_one_event_per_shadow_burst_validated"
    elif c5_off > 0:
        label = "relief_block_call_defect_latch_independent"
    elif not (c3 and c4 and c5):
        label = "latch_wiring_defect"
    elif not c1:
        label = "latch_under_compresses_train"
    else:
        label = "latch_over_suppresses_descents"

    total_on_fires = sum(r["fires"] for r in on_rows)
    total_on_shadow = sum(r["shadow_fires"] for r in on_rows)
    criteria = [
        {"name": "P0_shadow_train_exists", "load_bearing": False, "passed": p0_met,
         "measured": p0_ratio, "threshold": P0_MIN_TRAIN_RATIO,
         "measured_bursts": p0_bursts, "threshold_bursts": P0_MIN_BURSTS,
         "offending_cell": {"ratio_seed": p0_ratio_seed, "bursts_seed": p0_bursts_seed}},
        {"name": "C1_one_event_per_descent", "load_bearing": True, "passed": c1,
         "measured": c1_val, "threshold": C1_MAX_FIRES_PER_BURST, "comparator": "<=",
         "offending_cell": c1_seed},
        {"name": "C2_no_over_suppression", "load_bearing": False, "passed": c2,
         "measured": c2_val, "threshold": C2_MIN_FIRES_PER_BURST, "comparator": ">=",
         "offending_cell": c2_seed},
        {"name": "C3_latched_subset_of_shadow", "load_bearing": False, "passed": c3,
         "measured": c3_viol, "threshold": 0, "comparator": "=="},
        {"name": "C4_off_arm_identical_to_legacy", "load_bearing": False, "passed": c4,
         "measured": c4_viol, "threshold": 0, "comparator": "=="},
        {"name": "C5_one_relief_block_call_per_event", "load_bearing": False,
         "passed": c5, "measured": c5_mismatch, "threshold": 0, "comparator": "==",
         "measured_on_arm": c5_on, "measured_off_arm": c5_off},
    ]

    manifest = {
        "schema_version": "v1",
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "timestamp_utc": timestamp_utc,
        "queue_id": QUEUE_ID,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": CLAIM_IDS,
        "outcome": "PASS" if overall else "FAIL",
        "evidence_direction": "non_contributory",
        "combination_rule": "PASS = P0 AND C1 AND C2 AND C3 AND C4 AND C5",
        "criteria": criteria,
        "interpretation": {
            "label": label,
            "preconditions": [
                {"name": "on_arm_shadow_train_ratio", "measured": p0_ratio,
                 "threshold": P0_MIN_TRAIN_RATIO, "direction": "lower", "met": bool(
                     p0_ratio is not None and p0_ratio >= P0_MIN_TRAIN_RATIO),
                 "control": "unlatched shadow comparator on the ON arm's own norm "
                            "stream; worst seed", "offending_cell": p0_ratio_seed},
                {"name": "on_arm_shadow_bursts", "measured": p0_bursts,
                 "threshold": P0_MIN_BURSTS, "direction": "lower", "met": bool(
                     p0_bursts is not None and p0_bursts >= P0_MIN_BURSTS),
                 "control": "descents available to latch; worst seed",
                 "offending_cell": p0_bursts_seed},
            ],
            "criteria_non_degenerate": {
                "C1": bool(p0_met and total_on_shadow > total_on_fires),
                "C2": bool(p0_met and total_on_fires > 0),
                "C3": bool(total_on_fires > 0),
                "C4": bool(sum(r["fires"] for r in off_rows) > 0),
                "C5": bool(sum(r["fires"] for r in rows) > 0),
            },
        },
        "readout": {
            k: v for k, v in {
                "p0_min_shadow_train_ratio": p0_ratio,
                "p0_min_shadow_bursts": p0_bursts,
                "c1_max_fires_per_burst": c1_val,
                "c2_min_fires_per_burst": c2_val,
                "c3_subset_violations": c3_viol,
                "c4_off_identity_violations": c4_viol,
                "c5_call_event_mismatch": c5_mismatch,
                "c5_call_event_mismatch_on": c5_on,
                "c5_call_event_mismatch_off": c5_off,
                "inner_valence_writes_total": sum(r["inner_valence_writes"] for r in rows),
                "residue_active_any_cells": sum(int(r["residue_active_any"]) for r in rows),
                "on_rearm_count": sum(r["rearm_count"] for r in on_rows),
                "on_suppressed_count": sum(r["suppressed_count"] for r in on_rows),
                "injections_with_norm_rise_total": sum(r["injections_with_norm_rise"] for r in rows),
                "curriculum_injections_total": sum(r["curriculum_injections"] for r in rows),
                "on_total_fires": total_on_fires,
                "on_total_shadow_fires": total_on_shadow,
                "on_total_shadow_bursts": sum(r["shadow_bursts"] for r in on_rows),
                "off_total_fires": sum(r["fires"] for r in off_rows),
                "off_total_shadow_bursts": sum(r["shadow_bursts"] for r in off_rows),
                "overall_pass": int(overall),
            }.items() if v is not None
        },
        "arm_results": rows,
        "registered_thresholds": {
            "P0_MIN_TRAIN_RATIO": P0_MIN_TRAIN_RATIO, "P0_MIN_BURSTS": P0_MIN_BURSTS,
            "C1_MAX_FIRES_PER_BURST": C1_MAX_FIRES_PER_BURST,
            "C2_MIN_FIRES_PER_BURST": C2_MIN_FIRES_PER_BURST,
        },
        "ethics_preflight": {
            "involves_negative_valence": False, "involves_suffering_like_state": False,
            "involves_self_model": False, "involves_inescapability_or_helplessness": False,
            "involves_offline_replay_over_harm": False,
            "involves_social_mind_or_language": False,
            "involves_human_data_or_clinical_context": False, "decision": "allow",
        },
    }
    out_path = write_flat_manifest(
        manifest, dry_run=dry_run, config=full_config, seeds=seeds,
        script_path=Path(__file__), started_at=t0,
        z_goal_stream_stats=_ZG.stats(),
    )
    print(f"outcome: {manifest['outcome']} label={label}", flush=True)
    print(f"Result written to: {out_path}", flush=True)
    return manifest["outcome"], out_path, dry_run


if __name__ == "__main__":
    _outcome, _out_path, _dry_run = main()
    emit_outcome(
        outcome=_outcome if _outcome in ("PASS", "FAIL") else "FAIL",
        manifest_path=_out_path,
        dry_run=_dry_run,
    )
