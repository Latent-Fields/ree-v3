#!/opt/local/bin/python3
"""V3-EXQ-540b -- MECH-307 consumer-conjunction threshold sweep.

Claims: MECH-307, MECH-295  (consumer-side calibration; not single-claim evidence)
Anchor: REE_assembly/docs/architecture/anticipatory_affect_conjunction_vs_dual_channel.md
Supersedes: V3-EXQ-540a (2026-05-11T20:17:50Z FAIL on C2/C3).

V3-EXQ-540a FAIL analysis:
  C1 substrate-dissociation: PASS across all three arms. ARM_2_full produced
    liking_writes_mean=2711, gap4_e1_prior_writes_mean=2711, z_beta_excursion=0.15,
    split_surprise_centers_total=82 (positive=59, negative=82). Substrate Option-b
    wiring (commit fcfcf10) is correctly populating all four gap markers.
  C2 conjunction-read: FAIL. conj_fire_ticks_total=0 in ALL arms including ARM_2_full.
    The MECH-295 bridge's compute_conjunction_score_bias() predicate
      v[VALENCE_WANTING] > 0.6 AND v[VALENCE_LIKING] > 0.3 AND
      v[positive_surprise] > 0 AND z_beta_arousal > 0.6
    never fires, even when all four substrate counters are populated.
  C3 approach_commit_rate: saturated at 1.000 in every arm (independent issue:
    the _approach_commit_at_step predicate triggers trivially under default
    substrate). Separable from C2.

Diagnosis (per 540a's own pre-registered outcome branches): C1 PASS + C2 FAIL
routes to "threshold sweep needed before any architectural conclusion". The
defaults (0.6 / 0.3 / 0.6) appear miscalibrated relative to the per-tick
amplitudes the substrate accumulates (each schema_salience x gain x drive
write adds <= 0.5 to one slot; the consumer predicate reads the RBF-weighted
sum at the current z_world location, which decays with kernel distance).

This experiment is a focused diagnostic threshold sweep with substrate fully
ON in every arm; only the three consumer-side thresholds vary across arms.

Conditions (4 arms x 3 seeds = 12 cells)
----------------------------------------
All arms share: use_mech307_conjunction=True (master flag flips all three
substrate-side sub-flags via __post_init__) + use_mech307_consumer_conjunction_read=True
(bridge consumer ON) + mech307_conjunction_gain=1.0.

ARM_default (replicates 540a ARM_2_full):
  wanting_threshold=0.6, liking_threshold=0.3, z_beta_threshold=0.6
  Predicted: conj_fire_rate ~ 0 (reproduces 540a observation).

ARM_half:
  wanting_threshold=0.3, liking_threshold=0.15, z_beta_threshold=0.3

ARM_low:
  wanting_threshold=0.1, liking_threshold=0.05, z_beta_threshold=0.1

ARM_floor:
  wanting_threshold=0.01, liking_threshold=0.005, z_beta_threshold=0.01
  Predicted: conj_fire_rate close to ceiling (sanity arm; if THIS doesn't
  fire something is structurally wrong with the consumer-read path).

Pre-registered acceptance criteria
----------------------------------
C1 (monotone-in-threshold): conjunction_fire_rate_mean is non-increasing
  across the sorted arm sequence (ARM_floor >= ARM_low >= ARM_half >= ARM_default).
  Allows ties (zero ties at the upper end are expected). Sanity check that
  the consumer-read path is responsive to threshold magnitude.

C2 (at least one non-default arm fires): at least one of {ARM_floor, ARM_low,
  ARM_half} produces conjunction_fire_rate >= 0.10 in 2/3 seeds. This is the
  load-bearing diagnostic: if NO non-default threshold setting can produce
  conjunction fires from the populated substrate, the issue is not threshold
  miscalibration but a deeper structural problem (kernel-decay, normalization,
  or write/read site mismatch).

PASS = C1 AND C2 -> calibration issue confirmed; route to a 540c behavioural
  retest with the lowest-threshold arm that fires reliably.

PARTIAL (C1 PASS, C2 FAIL) -> even floor thresholds don't fire; structural
  audit required (suspect: kernel decay reading nearby zeros, or RBF center
  drift, or write/read site z_world mismatch).

PARTIAL (C1 FAIL, C2 PASS) -> conjunction fires but non-monotone (likely
  noise at very-low-threshold ceiling); informative but unexpected.

FAIL on both -> substrate audit required; the 540a substrate-dissociation
  PASS may be misleading and a deeper substrate bug is present.

Out of scope (separable):
  C3 approach_commit_rate saturation. The 540a metric saturates at 1.0
  because beta_gate is always elevated in this env config and wanting > 0.05
  is too lenient. This is a metric-design bug independent of the conjunction
  threshold question and is left for a separate redesign session.

Smoke
-----
  /opt/local/bin/python3 experiments/v3_exq_540b_mech307_conjunction_threshold_sweep.py --dry-run

Estimated runtime: ~120 min on Mac (4 arms x 3 seeds x ~10 min/cell).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorld  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from ree_core.residue.field import (  # noqa: E402
    VALENCE_WANTING, VALENCE_LIKING, VALENCE_SURPRISE,
    VALENCE_POSITIVE_SURPRISE, VALENCE_NEGATIVE_SURPRISE,
)
from experiment_protocol import emit_outcome  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402


EXPERIMENT_TYPE = "v3_exq_540b_mech307_conjunction_threshold_sweep"
QUEUE_ID = "V3-EXQ-540b"
SUPERSEDES = "V3-EXQ-540a"
CLAIM_IDS = ["MECH-307", "MECH-295"]
EXPERIMENT_PURPOSE = "diagnostic"   # consumer-side threshold calibration

SEEDS = [42, 43, 44]
P0_EPISODES = 50
EVAL_EPISODES = 20
STEPS_PER_EPISODE = 200

# Episodes per (seed x arm) cell -- must equal the M denominator in the
# [train] ep N/M progress prints, AND the episodes_per_run queue field.
EPISODES_PER_RUN = P0_EPISODES + EVAL_EPISODES   # 70

# Acceptance thresholds (pre-registered).
C1_MONOTONE_TOLERANCE = 1e-4       # allow zero ties + small noise floor
C2_FIRE_RATE_FLOOR = 0.10
C2_MIN_SEEDS_FIRING = 2            # of 3 seeds


# All arms share substrate-fully-on + consumer-conjunction-read-on. Only the
# three consumer-side thresholds vary. Each arm's `flags` dict carries the
# substrate sub-flag + threshold key/value pairs to be applied via setattr
# on the REEConfig (see _make_config).
ARMS = [
    {
        "arm": "ARM_default",
        "flags": {
            "use_mech307_split_surprise": True,
            "use_mech307_schema_multichannel": True,
            "use_mech307_predicted_location_write": True,
            "mech307_conjunction_wanting_threshold": 0.6,
            "mech307_conjunction_liking_threshold": 0.3,
            "mech307_conjunction_z_beta_threshold": 0.6,
        },
    },
    {
        "arm": "ARM_half",
        "flags": {
            "use_mech307_split_surprise": True,
            "use_mech307_schema_multichannel": True,
            "use_mech307_predicted_location_write": True,
            "mech307_conjunction_wanting_threshold": 0.3,
            "mech307_conjunction_liking_threshold": 0.15,
            "mech307_conjunction_z_beta_threshold": 0.3,
        },
    },
    {
        "arm": "ARM_low",
        "flags": {
            "use_mech307_split_surprise": True,
            "use_mech307_schema_multichannel": True,
            "use_mech307_predicted_location_write": True,
            "mech307_conjunction_wanting_threshold": 0.1,
            "mech307_conjunction_liking_threshold": 0.05,
            "mech307_conjunction_z_beta_threshold": 0.1,
        },
    },
    {
        "arm": "ARM_floor",
        "flags": {
            "use_mech307_split_surprise": True,
            "use_mech307_schema_multichannel": True,
            "use_mech307_predicted_location_write": True,
            "mech307_conjunction_wanting_threshold": 0.01,
            "mech307_conjunction_liking_threshold": 0.005,
            "mech307_conjunction_z_beta_threshold": 0.01,
        },
    },
]
# Ordering used in monotone-in-threshold check (descending threshold first).
ARM_ORDER_DESCENDING = ("ARM_default", "ARM_half", "ARM_low", "ARM_floor")


def _make_env(seed: int) -> CausalGridWorld:
    """Foraging-class env with sparse contacts -- same regime as EXQ-539."""
    return CausalGridWorld(
        size=10,
        num_hazards=3,
        num_resources=8,
        hazard_harm=0.01,
        resource_benefit=0.18,
        use_proxy_fields=True,
        seed=seed,
        proximity_benefit_scale=0.18,
        resource_respawn_on_consume=True,
    )


def _make_config(env: CausalGridWorld, mech307_flags: Dict) -> REEConfig:
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        alpha_world=0.9,
        world_dim=32,
        drive_weight=2.0,
        z_goal_enabled=True,
    )
    cfg.surprise_gated_replay = True
    cfg.e1.schema_wanting_enabled = True
    cfg.schema_wanting_threshold = 0.1
    cfg.schema_wanting_gain = 0.5
    cfg.residue.valence_enabled = True
    # MECH-295 bridge ON in ALL arms (downstream consumer must exist).
    cfg.use_mech295_liking_bridge = True
    cfg.mech295_drive_to_liking_gain = 1.0
    cfg.mech295_liking_to_approach_cue_gain = 0.5
    cfg.mech295_min_drive_to_fire = 0.1
    cfg.mech295_min_z_goal_norm_to_fire = 0.05
    # MECH-307 Path B consumer conjunction read ON in ALL arms. The bridge
    # short-circuits to zero bias when conjunction predicates fail (which
    # is what happens in ARM_0 and ARM_1), so this is a clean per-arm
    # discriminator: ARM_2 is the only arm where the predicate can hold.
    cfg.use_mech307_consumer_conjunction_read = True
    cfg.mech307_conjunction_gain = 1.0
    # Apply per-arm substrate flags.
    for flag, val in mech307_flags.items():
        setattr(cfg, flag, val)
    return cfg


def _measure_arm(
    seed: int,
    arm_label: str,
    mech307_flags: Dict,
    n_warmup: int,
    n_eval: int,
    steps_per_episode: int,
) -> Dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = _make_env(seed)
    cfg = _make_config(env, mech307_flags)
    agent = REEAgent(cfg)

    liking_writes = 0
    schema_wanting_writes = 0   # standard MECH-216 fire (any arm with schema_wanting_enabled)
    gap4_e1_prior_writes = 0    # ONLY when use_mech307_predicted_location_write=True AND _cached_e1_prior set
    z_beta_dim0_excursions: List[float] = []
    harm_paired_surprise_writes = 0
    nonharm_surprise_writes = 0
    approach_commit_steps = 0
    total_eval_steps = 0
    contact_events = 0
    n_conjunction_read_opportunities = 0
    n_conjunction_fires_pretick = 0   # bridge counter snapshot before each tick

    def _approach_commit_at_step(_agent: REEAgent) -> bool:
        beta_elevated = bool(getattr(_agent.beta_gate, "is_elevated", False))
        if not beta_elevated:
            return False
        if _agent._current_latent is None:
            return False
        z = _agent._current_latent.z_world
        with torch.no_grad():
            v = _agent.residue_field.evaluate_valence(z)
        wanting_amp = float(v[0, VALENCE_WANTING].item())
        return wanting_amp > 0.05

    total_episodes = n_warmup + n_eval

    # -------- P0 warmup --------
    for ep in range(n_warmup):
        _, obs_dict = env.reset()
        agent.reset()
        for _ in range(steps_per_episode):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            latent = agent.sense(obs_body, obs_world)
            ticks = agent.clock.advance()
            e1_prior = (
                agent._e1_tick(latent) if ticks.get("e1_tick")
                else torch.zeros(1, agent.config.latent.world_dim)
            )
            drive = float(REEAgent.compute_drive_level(obs_body))
            agent.update_schema_wanting(drive_level=drive)
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            action = agent.select_action(candidates, ticks)
            action_idx = int(action.argmax(dim=-1).item())
            _, harm_signal, done, info, obs_dict = env.step(action_idx)
            ttype = info.get("transition_type", "none")
            if ttype == "resource":
                agent.update_z_goal(float(harm_signal), drive_level=drive)
            agent.update_residue(float(harm_signal))
            if done:
                break
        if (ep + 1) % 10 == 0 or (ep + 1) == n_warmup:
            print(
                f"  [train] {arm_label} seed={seed} ep {ep+1}/{total_episodes} (warmup)",
                flush=True,
            )

    # -------- Eval --------
    for ep in range(n_eval):
        _, obs_dict = env.reset()
        agent.reset()
        for _ in range(steps_per_episode):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world)
                ticks = agent.clock.advance()
                e1_prior = (
                    agent._e1_tick(latent) if ticks.get("e1_tick")
                    else torch.zeros(1, agent.config.latent.world_dim)
                )
                drive = float(REEAgent.compute_drive_level(obs_body))

                z_world_pre = latent.z_world.clone()
                z_beta_pre_dim0 = (
                    float(latent.z_beta[..., 0].abs().mean().item())
                    if latent.z_beta is not None and latent.z_beta.numel() > 0
                    else 0.0
                )
                v_pre = agent.residue_field.evaluate_valence(z_world_pre)
                liking_pre = float(v_pre[0, VALENCE_LIKING].item())
                wanting_pre = float(v_pre[0, VALENCE_WANTING].item())

                agent.update_schema_wanting(drive_level=drive)

                v_post = agent.residue_field.evaluate_valence(z_world_pre)
                liking_delta = float(v_post[0, VALENCE_LIKING].item()) - liking_pre
                wanting_delta = float(v_post[0, VALENCE_WANTING].item()) - wanting_pre
                z_beta_post_dim0 = (
                    float(latent.z_beta[..., 0].abs().mean().item())
                    if latent.z_beta is not None and latent.z_beta.numel() > 0
                    else 0.0
                )
                if liking_delta > 1e-9:
                    liking_writes += 1
                if wanting_delta > 1e-9:
                    schema_wanting_writes += 1
                    # Gap 4 distinguisher: the schema-wanting write target is
                    # the cached e1_prior ONLY when both the flag is on AND
                    # the agent's _cached_e1_prior is populated (set in
                    # _e1_tick when the e1 tick fires this step).
                    if (
                        getattr(cfg, "use_mech307_predicted_location_write", False)
                        and getattr(agent, "_cached_e1_prior", None) is not None
                    ):
                        gap4_e1_prior_writes += 1
                z_beta_dim0_excursions.append(z_beta_post_dim0 - z_beta_pre_dim0)

                # Snapshot bridge conjunction-fire counter before select_action so
                # we can detect a conjunction fire at this tick.
                n_fires_pre_tick = (
                    int(agent.mech295_bridge._n_conjunction_fires)
                    if agent.mech295_bridge is not None else 0
                )

                candidates = agent.generate_trajectories(latent, e1_prior, ticks)
                action = agent.select_action(candidates, ticks)

                n_fires_post_tick = (
                    int(agent.mech295_bridge._n_conjunction_fires)
                    if agent.mech295_bridge is not None else 0
                )
                if n_fires_post_tick > n_fires_pre_tick:
                    n_conjunction_fires_pretick += 1
                # The bridge enters the read path on every select_action where
                # goal is active and current_latent / z_beta are present;
                # count opportunities by checking whether the bridge's read
                # path ran (n_conjunction_reads counter advances even on
                # zero-fire ticks under our wiring? -- NO: bridge increments
                # n_conjunction_reads only when n_fires > 0 this call. We
                # track opportunities via a separate condition: bridge active
                # AND goal active AND latent present).
                if (
                    agent.mech295_bridge is not None
                    and agent.goal_state is not None
                    and agent.goal_state.is_active()
                    and agent._current_latent is not None
                ):
                    n_conjunction_read_opportunities += 1

            action_idx = int(action.argmax(dim=-1).item())
            if _approach_commit_at_step(agent):
                approach_commit_steps += 1
            total_eval_steps += 1

            _, harm_signal, done, info, obs_dict = env.step(action_idx)
            ttype = info.get("transition_type", "none")
            if ttype == "resource":
                agent.update_z_goal(float(harm_signal), drive_level=drive)
                contact_events += 1

            surprise_count_pre = agent._surprise_write_count
            agent.update_residue(float(harm_signal))
            surprise_count_post = agent._surprise_write_count
            if surprise_count_post > surprise_count_pre and harm_signal < 0:
                harm_paired_surprise_writes += 1
            elif surprise_count_post > surprise_count_pre:
                nonharm_surprise_writes += 1

            if done:
                break
        global_ep = n_warmup + ep + 1
        if (ep + 1) % 5 == 0 or (ep + 1) == n_eval:
            print(
                f"  [train] {arm_label} seed={seed} ep {global_ep}/{total_episodes} (eval)",
                flush=True,
            )

    approach_commit_rate = approach_commit_steps / max(1, total_eval_steps)
    z_beta_excursion_mean = (
        float(np.mean([abs(x) for x in z_beta_dim0_excursions]))
        if z_beta_dim0_excursions else 0.0
    )

    valence_vecs = agent.residue_field.rbf_field.valence_vecs
    active_mask = agent.residue_field.rbf_field.active_mask
    if active_mask.any():
        active_pos = valence_vecs[active_mask][:, VALENCE_POSITIVE_SURPRISE]
        active_neg = valence_vecs[active_mask][:, VALENCE_NEGATIVE_SURPRISE]
        # Gap 1 Option-b marker: count active centers with a non-zero write
        # to either of the new split-surprise channels. Under Option-b the
        # legacy VALENCE_SURPRISE slot still receives the unsigned magnitude
        # via the backward-compat write at the same site, so it cannot
        # distinguish Option-b-fired centers from true-legacy fired centers.
        # The split channels are the discriminative observable.
        n_positive_surprise_centers = int((active_pos.abs() > 1e-9).sum().item())
        n_negative_surprise_centers = int((active_neg.abs() > 1e-9).sum().item())
        n_split_surprise_centers = int(
            ((active_pos.abs() > 1e-9) | (active_neg.abs() > 1e-9)).sum().item()
        )
    else:
        n_positive_surprise_centers = 0
        n_negative_surprise_centers = 0
        n_split_surprise_centers = 0

    bridge_diag = (
        agent.mech295_bridge.get_diagnostics()
        if agent.mech295_bridge is not None else {}
    )

    return {
        "seed": seed,
        "arm": arm_label,
        "mech307_flags": dict(mech307_flags),
        "liking_writes": liking_writes,
        "schema_wanting_writes": schema_wanting_writes,
        "gap4_e1_prior_writes": gap4_e1_prior_writes,
        "z_beta_excursion_mean": z_beta_excursion_mean,
        "harm_paired_surprise_writes": harm_paired_surprise_writes,
        "nonharm_surprise_writes": nonharm_surprise_writes,
        "n_positive_surprise_centers": n_positive_surprise_centers,
        "n_negative_surprise_centers": n_negative_surprise_centers,
        "n_split_surprise_centers": n_split_surprise_centers,
        "approach_commit_rate": approach_commit_rate,
        "approach_commit_steps": approach_commit_steps,
        "total_eval_steps": total_eval_steps,
        "contact_events": contact_events,
        # MECH-307 Path B (consumer conjunction read) diagnostics.
        "n_conjunction_read_opportunities": n_conjunction_read_opportunities,
        "n_conjunction_fire_ticks": n_conjunction_fires_pretick,
        "bridge_n_conjunction_fires": int(bridge_diag.get("n_conjunction_fires", 0)),
        "bridge_n_conjunction_reads": int(bridge_diag.get("n_conjunction_reads", 0)),
        "conjunction_fire_rate": (
            n_conjunction_fires_pretick / max(1, n_conjunction_read_opportunities)
        ),
    }


def _aggregate(per_cell: List[Dict]) -> Dict[str, Dict]:
    bucket: Dict[str, Dict] = {}
    for r in per_cell:
        arm = r["arm"]
        if arm not in bucket:
            bucket[arm] = {
                "arm": arm,
                "n_seeds": 0,
                "liking_writes_mean": 0.0,
                "schema_wanting_writes_mean": 0.0,
                "gap4_e1_prior_writes_mean": 0.0,
                "z_beta_excursion_mean": 0.0,
                "harm_paired_surprise_writes_total": 0,
                "nonharm_surprise_writes_total": 0,
                "n_positive_surprise_centers_total": 0,
                "n_negative_surprise_centers_total": 0,
                "n_split_surprise_centers_total": 0,
                "approach_commit_rate_mean": 0.0,
                "contact_events_total": 0,
                "n_conjunction_fire_ticks_total": 0,
                "n_conjunction_read_opportunities_total": 0,
                "conjunction_fire_rate_mean": 0.0,
                "per_seed_approach": [],
                "per_seed_split_surprise_centers": [],
                "per_seed_neg_surprise_centers": [],
                "per_seed_conjunction_fire_rate": [],
            }
        b = bucket[arm]
        b["n_seeds"] += 1
        b["liking_writes_mean"] += r["liking_writes"]
        b["schema_wanting_writes_mean"] += r["schema_wanting_writes"]
        b["gap4_e1_prior_writes_mean"] += r["gap4_e1_prior_writes"]
        b["z_beta_excursion_mean"] += r["z_beta_excursion_mean"]
        b["harm_paired_surprise_writes_total"] += r["harm_paired_surprise_writes"]
        b["nonharm_surprise_writes_total"] += r["nonharm_surprise_writes"]
        b["n_positive_surprise_centers_total"] += r["n_positive_surprise_centers"]
        b["n_negative_surprise_centers_total"] += r["n_negative_surprise_centers"]
        b["n_split_surprise_centers_total"] += r["n_split_surprise_centers"]
        b["approach_commit_rate_mean"] += r["approach_commit_rate"]
        b["contact_events_total"] += r["contact_events"]
        b["n_conjunction_fire_ticks_total"] += r["n_conjunction_fire_ticks"]
        b["n_conjunction_read_opportunities_total"] += r["n_conjunction_read_opportunities"]
        b["conjunction_fire_rate_mean"] += r["conjunction_fire_rate"]
        b["per_seed_approach"].append(r["approach_commit_rate"])
        b["per_seed_neg_surprise_centers"].append(r["n_negative_surprise_centers"])
        b["per_seed_split_surprise_centers"].append(r["n_split_surprise_centers"])
        b["per_seed_conjunction_fire_rate"].append(r["conjunction_fire_rate"])
    for arm, b in bucket.items():
        n = max(1, b["n_seeds"])
        b["liking_writes_mean"] /= n
        b["schema_wanting_writes_mean"] /= n
        b["gap4_e1_prior_writes_mean"] /= n
        b["z_beta_excursion_mean"] /= n
        b["approach_commit_rate_mean"] /= n
        b["conjunction_fire_rate_mean"] /= n
    return bucket


def _evaluate_acceptance(agg: Dict[str, Dict]) -> Dict:
    """Threshold-sweep acceptance.

    C1 (monotone-in-threshold): conjunction_fire_rate_mean is non-increasing
        across (ARM_floor, ARM_low, ARM_half, ARM_default). Allows ties +
        small tolerance to absorb noise floor.
    C2 (at-least-one-non-default-fires): at least one of the non-default
        arms produces conjunction_fire_rate >= C2_FIRE_RATE_FLOOR (0.10) in
        at least C2_MIN_SEEDS_FIRING (2) of 3 seeds.
    """
    fire_rates = {
        arm: agg[arm]["conjunction_fire_rate_mean"]
        for arm in ARM_ORDER_DESCENDING
    }

    # C1 monotone: fire_rates should be non-decreasing as thresholds drop.
    # ARM_ORDER_DESCENDING is sorted highest threshold -> lowest threshold.
    # Expected: fire_rates[default] <= fire_rates[half] <= fire_rates[low] <= fire_rates[floor].
    monotone_ok = True
    monotone_pairs = []
    prev = None
    prev_arm = None
    for arm in ARM_ORDER_DESCENDING:
        cur = fire_rates[arm]
        if prev is not None:
            pair_ok = cur >= prev - C1_MONOTONE_TOLERANCE
            monotone_pairs.append({
                "lower_threshold_arm": arm,
                "higher_threshold_arm": prev_arm,
                "fire_rate_lower": cur,
                "fire_rate_higher": prev,
                "pair_ok": bool(pair_ok),
            })
            if not pair_ok:
                monotone_ok = False
        prev = cur
        prev_arm = arm
    c1 = bool(monotone_ok)

    # C2 at least one non-default arm fires the conjunction at floor rate
    # in at least 2/3 seeds.
    non_default = [a for a in ARM_ORDER_DESCENDING if a != "ARM_default"]
    arm_firing_results = []
    any_arm_fires = False
    for arm in non_default:
        per_seed = agg[arm]["per_seed_conjunction_fire_rate"]
        n_seeds_firing = sum(
            1 for r in per_seed if r >= C2_FIRE_RATE_FLOOR
        )
        fires = n_seeds_firing >= C2_MIN_SEEDS_FIRING
        arm_firing_results.append({
            "arm": arm,
            "n_seeds_firing": int(n_seeds_firing),
            "per_seed_fire_rates": [round(r, 4) for r in per_seed],
            "fires_at_floor": bool(fires),
        })
        if fires:
            any_arm_fires = True
    c2 = bool(any_arm_fires)

    overall = c1 and c2
    return {
        "C1_monotone_in_threshold": bool(c1),
        "C1_monotone_pairs": monotone_pairs,
        "C2_at_least_one_arm_fires": bool(c2),
        "C2_per_arm_firing": arm_firing_results,
        "fire_rates_by_arm": {a: round(r, 4) for a, r in fire_rates.items()},
        "all_pass": bool(overall),
    }


def main(dry_run: bool = False):
    print(f"[{EXPERIMENT_TYPE}] starting (dry_run={dry_run})")
    seeds = (SEEDS[0],) if dry_run else SEEDS
    n_warmup = 4 if dry_run else P0_EPISODES
    n_eval = 2 if dry_run else EVAL_EPISODES

    per_cell: List[Dict] = []
    t0 = time.time()
    for seed in seeds:
        for arm_cfg in ARMS:
            arm_label = arm_cfg["arm"]
            print(f"Seed {seed} Condition {arm_label}", flush=True)
            arm_t0 = time.time()
            r = _measure_arm(
                seed=seed,
                arm_label=arm_label,
                mech307_flags=arm_cfg["flags"],
                n_warmup=n_warmup,
                n_eval=n_eval,
                steps_per_episode=STEPS_PER_EPISODE,
            )
            per_cell.append(r)
            # cell_pass is a runner-visible "made it through eval without
            # crashing" marker; the load-bearing acceptance is computed at
            # aggregate time over conjunction_fire_rate (C1/C2 below).
            cell_pass = r["total_eval_steps"] > 0
            print(
                f"  seed={seed} arm={arm_label:<18} "
                f"liking={r['liking_writes']:>3} "
                f"sched_wanting={r['schema_wanting_writes']:>3} "
                f"gap4={r['gap4_e1_prior_writes']:>3} "
                f"z_beta_exc={r['z_beta_excursion_mean']:.4f} "
                f"split_surp_centers={r['n_split_surprise_centers']} "
                f"pos={r['n_positive_surprise_centers']} "
                f"neg={r['n_negative_surprise_centers']} "
                f"conj_fire_ticks={r['n_conjunction_fire_ticks']} "
                f"conj_fire_rate={r['conjunction_fire_rate']:.3f} "
                f"approach={r['approach_commit_rate']:.3f} "
                f"contacts={r['contact_events']} "
                f"elapsed={time.time()-arm_t0:.1f}s",
                flush=True,
            )
            print(f"verdict: {'PASS' if cell_pass else 'FAIL'}", flush=True)

    agg = _aggregate(per_cell)
    acceptance = _evaluate_acceptance(agg)
    elapsed = time.time() - t0
    outcome = "PASS" if acceptance["all_pass"] else "FAIL"

    print(f"[{EXPERIMENT_TYPE}] aggregates:")
    for arm in ARM_ORDER_DESCENDING:
        a = agg[arm]
        print(
            f"  {arm:<18} liking={a['liking_writes_mean']:.1f} "
            f"sched_wanting={a['schema_wanting_writes_mean']:.1f} "
            f"gap4={a['gap4_e1_prior_writes_mean']:.1f} "
            f"z_beta_exc={a['z_beta_excursion_mean']:.4f} "
            f"split_surp_centers={a['n_split_surprise_centers_total']} "
            f"pos={a['n_positive_surprise_centers_total']} "
            f"neg={a['n_negative_surprise_centers_total']} "
            f"conj_fire_ticks={a['n_conjunction_fire_ticks_total']} "
            f"conj_fire_rate={a['conjunction_fire_rate_mean']:.3f} "
            f"approach={a['approach_commit_rate_mean']:.3f} "
            f"per_seed_approach={[round(x, 3) for x in a['per_seed_approach']]}",
            flush=True,
        )
    print(f"[{EXPERIMENT_TYPE}] acceptance:")
    for k, v in acceptance.items():
        print(f"  {k}: {v}")
    print(f"[{EXPERIMENT_TYPE}] outcome={outcome} elapsed={elapsed:.1f}s")
    print(f"Done. Outcome: {outcome}")

    if dry_run:
        print(f"[{EXPERIMENT_TYPE}] dry-run complete; not writing manifest.")
        return 0

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{timestamp}_v3"
    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-claim direction classification.
    # PASS = thresholds are the calibration knob; the conjunction fires when
    #   set low enough. MECH-307 read-side architecture validated as a
    #   calibration question; MECH-295 bridge consumer is operating correctly.
    # PARTIAL (monotone but no arm fires) = even at floor thresholds the
    #   predicate doesn't fire; the bridge consumer-read path has a deeper
    #   structural issue (kernel decay / write-read site mismatch). Mark
    #   MECH-307 mixed and MECH-295 weakens.
    # PARTIAL (non-monotone but some arm fires) = unexpected noise; mixed.
    # FAIL on both = consumer-read path is broken structurally.
    if outcome == "PASS":
        direction = "supports"
        per_claim = {
            "MECH-307": "supports",
            "MECH-295": "supports",
        }
    elif acceptance["C1_monotone_in_threshold"]:
        # Monotone but no arm reaches floor: structural read-side issue.
        direction = "mixed"
        per_claim = {
            "MECH-307": "mixed",
            "MECH-295": "weakens",
        }
    elif acceptance["C2_at_least_one_arm_fires"]:
        # Fires but non-monotone: unexpected.
        direction = "mixed"
        per_claim = {
            "MECH-307": "mixed",
            "MECH-295": "mixed",
        }
    else:
        direction = "weakens"
        per_claim = {cid: "weakens" for cid in CLAIM_IDS}

    manifest = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "supersedes": SUPERSEDES,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "started_utc": datetime.utcnow().isoformat() + "Z",
        "outcome": outcome,
        "evidence_direction": direction,
        "evidence_direction_per_claim": per_claim,
        "elapsed_seconds": elapsed,
        "n_seeds": len(seeds),
        "p0_episodes": P0_EPISODES,
        "eval_episodes": EVAL_EPISODES,
        "steps_per_episode": STEPS_PER_EPISODE,
        "arms": list(agg.values()),
        "per_seed_per_arm": per_cell,
        "acceptance": acceptance,
        "thresholds": {
            "C1_monotone_tolerance": C1_MONOTONE_TOLERANCE,
            "C2_fire_rate_floor": C2_FIRE_RATE_FLOOR,
            "C2_min_seeds_firing": C2_MIN_SEEDS_FIRING,
        },
        "note": (
            "V3-EXQ-540b -- MECH-307 consumer-conjunction threshold sweep. "
            "Supersedes V3-EXQ-540a (2026-05-11T20:17:50Z FAIL). 540a confirmed "
            "C1 substrate dissociation PASS across all three arms (ARM_2_full "
            "produced liking=2711, gap4=2711, z_beta=0.15, split_centers=82) but "
            "FAILed C2 (conj_fire_rate=0 in ALL arms including ARM_2_full). The "
            "MECH-295 bridge's compute_conjunction_score_bias predicate "
            "(v_wanting > 0.6 AND v_liking > 0.3 AND v_positive_surprise > 0 AND "
            "z_beta_arousal > 0.6) never fires even when all four substrate "
            "counters are populated. 540b varies only the three consumer-side "
            "thresholds across 4 arms (ARM_default 0.6/0.3/0.6, ARM_half "
            "0.3/0.15/0.3, ARM_low 0.1/0.05/0.1, ARM_floor 0.01/0.005/0.01) with "
            "substrate fully ON in every arm via use_mech307_conjunction=True. "
            "Acceptance: C1 monotone fire-rate decrease across descending-threshold "
            "ordering; C2 at least one non-default arm fires conjunction in >= 2/3 "
            "seeds at floor rate 0.10. PASS -> calibration issue confirmed; queue "
            "540c behavioural retest with lowest-firing threshold arm. PARTIAL "
            "(monotone, no fires) -> structural read-side audit. PARTIAL "
            "(non-monotone, fires) -> unexpected noise; investigate. FAIL on both -> "
            "deeper substrate bug suspected. C3 (approach_commit_rate lift) is "
            "OUT OF SCOPE here: 540a observed approach=1.0 saturation across all "
            "arms (the _approach_commit_at_step predicate triggers trivially); "
            "separate metric-redesign session required and is independent of the "
            "threshold question."
        ),
    }
    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=False,
        config=manifest.get("config"),
        seeds=SEEDS,
        script_path=Path(__file__),
    )
    print(f"Result written to: {out_path}", flush=True)
    return outcome, out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Smoke run, no manifest.")
    args = parser.parse_args()
    result = main(dry_run=args.dry_run)
    if args.dry_run or result == 0:
        sys.exit(0)
    _outcome, _out_path = result
    emit_outcome(
        outcome=_outcome if _outcome in ("PASS", "FAIL") else "FAIL",
        manifest_path=_out_path,
    )
    sys.exit(0)
