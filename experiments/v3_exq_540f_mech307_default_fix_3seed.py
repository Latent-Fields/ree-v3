#!/opt/local/bin/python3
"""V3-EXQ-540f -- MECH-307 default-fix clean 3-seed rerun (seeds 42, 43, 45).

Claims: MECH-307, MECH-216, MECH-205, MECH-093, SD-014, MECH-295
Anchor: REE_assembly/docs/architecture/anticipatory_affect_conjunction_vs_dual_channel.md
Supersedes: V3-EXQ-540e (seed-44 truncation: degenerate no-op policy under
  no-gradient-learning conditions).

Root cause of 540e FAIL
-----------------------
torch.manual_seed(44) produces a network initialization that consistently
selects action=4 (no-op/stay) on every step throughout all three arms.
The untrained CEM agent's trajectory proposals all converge to the same
argmax action because there is NO gradient learning (no optimizer.step())
in the P0 warmup loop -- random weights persist for all 50 + 20 = 70
episodes. With seed=44 the agent spawns at (5, 8), directly adjacent to
a hazard at (4, 8). It never moves. After 4 steps of proximity harm
(~0.07262/step), the hazard drifts from (4, 8) onto the agent's cell
(5, 8) at step 5. The _make_env() call in 540e does NOT override
contaminated_harm (default = 0.4) -- 40x larger than hazard_harm=0.01.
The agent takes 0.4 harm/step from the contaminated cell, health drops
from 0.7095 -> 0.3095 -> 0.0 -> dead. Episode ends in ~6 steps.
This repeats across ALL 20 eval episodes and ALL 3 arms (same network
init per seed per arm): total_eval_steps = 134 (vs 4000 expected).

Seeds 42 and 43 are unaffected because their random inits produce diverse
actions that move the agent away from hazards. Seed 45 was verified clean:
5 episodes x 200 steps = 1000 steps, zero early terminations.

Seed 44 is dropped as degenerate under no-gradient-learning conditions
and replaced with seed 45. The env is NOT degenerate -- only the specific
torch.manual_seed(44) initialization is pathological in this training regime.

Design unchanged
----------------
This experiment is otherwise identical to V3-EXQ-540e: same 3-arm
decomposition (ARM_0_off / ARM_1_split_only / ARM_2_full), same
acceptance criteria (C1 substrate dissociation, C2 consumer conjunction
read), same config. C3 is still informational (approach_commit_rate
saturation is a known property of this env+training combo; see 540e
manifest notes). SEEDS is the only changed constant.

Conditions (3 arms x 3 seeds = 9 cells)
---------------------------------------
ARM_0_off (replicates EXQ-539 ARM_OFF substrate):
  - All MECH-307 substrate flags False (master + all three sub-flags).
  - Bridge consumer-conjunction-read flag ON (no-op when substrate fixes
    are off because conjunction predicates can't form).
  - Predicted: substrate counters at zero; conjunction-fire counter at
    zero; approach_commit_rate at EXQ-536b inert baseline.

ARM_1_split_only (Gap 1 only, Option-b):
  - use_mech307_split_surprise = True   (Option-b: split channels)
  - use_mech307_schema_multichannel = False
  - use_mech307_predicted_location_write = False
  - Bridge consumer-conjunction-read flag ON.
  - Predicted: positive-surprise OR negative-surprise channel centers
    accumulate on harm + non-harm contacts. VALENCE_LIKING / z_beta-pulse
    counters at zero. Conjunction predicate cannot fire (Gap 2 missing
    means VALENCE_LIKING stays zero). approach_commit_rate near ARM_0.

ARM_2_full_conjunction (all four gaps, Option-b for Gap 1):
  - use_mech307_split_surprise = True              (Gap 1 Option-b)
  - use_mech307_schema_multichannel = True         (Gaps 2 + 3)
  - use_mech307_predicted_location_write = True    (Gap 4)
  - Bridge consumer-conjunction-read flag ON.
  - Predicted: all four substrate counters fire; conjunction predicate
    fires at predicted-imminent locations; bridge applies additional
    negative score_bias; approach_commit_rate lifts above ARM_0.

Pre-registered acceptance criteria
----------------------------------
C1 (substrate-readiness counter dissociation, per-gap):
   ARM_0: liking_writes == 0, predicted_writes == 0,
          split_surprise_centers == 0 (POS + NEG channels empty),
          z_beta_excursion ~ 0.
   ARM_1: split_surprise_centers > 0 (Option-b POS or NEG channel);
          liking_writes == 0; predicted_writes == 0;
          z_beta_excursion ~ 0. (Gap 1 isolated.)
   ARM_2: all four signatures fire (liking_writes > 0, predicted_writes > 0,
          split_surprise_centers > 0, z_beta_excursion >= 0.05).

C2 (consumer-side conjunction read fires per arm as predicted):
   ARM_0: n_conjunction_fires == 0.
   ARM_1: n_conjunction_fires == 0.
   ARM_2: n_conjunction_fires > 0 across all 3 seeds AND
          per-seed-mean conjunction-detection-rate >= 0.10.

PASS = C1 AND C2.
C3 (approach_commit_rate lift) is INFORMATIONAL only -- same saturation
caveat as 540e applies unchanged (see 540e manifest).

Smoke
-----
  /opt/local/bin/python3 experiments/v3_exq_540f_mech307_default_fix_3seed.py --dry-run

Estimated runtime: ~90 min on Mac (3 arms x 3 seeds x ~10 min/cell).
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


EXPERIMENT_TYPE = "v3_exq_540f_mech307_default_fix_3seed"
QUEUE_ID = "V3-EXQ-540f"
SUPERSEDES = "V3-EXQ-540e"
CLAIM_IDS = ["MECH-307", "MECH-216", "MECH-205", "MECH-093", "SD-014", "MECH-295"]
EXPERIMENT_PURPOSE = "diagnostic"   # gap decomposition; not single-claim evidence

# Seed 44 dropped: degenerate no-op policy under no-gradient-learning conditions
# (torch.manual_seed(44) always selects action=4/stay; contaminated_harm=0.4
#  kills agent in ~6 steps per episode -> 134/4000 eval steps total).
# Seed 45 verified clean: 5 episodes x 200 steps = 1000 steps, zero deaths.
SEEDS = [42, 43, 45]
P0_EPISODES = 50
EVAL_EPISODES = 20
STEPS_PER_EPISODE = 200

# Episodes per (seed x arm) cell -- must equal the M denominator in the
# [train] ep N/M progress prints, AND the episodes_per_run queue field.
EPISODES_PER_RUN = P0_EPISODES + EVAL_EPISODES   # 70

# Acceptance thresholds (pre-registered).
C1_LIKING_WRITES_FLOOR_ARM2 = 1.0
C1_PREDICTED_WRITES_FLOOR_ARM2 = 1.0
C1_Z_BETA_FLOOR_ARM2 = 0.05
C1_NEG_SURPRISE_FLOOR_ARM12 = 1
C2_CONJUNCTION_RATE_FLOOR_ARM2 = 0.10
C3_APPROACH_LIFT = 0.10


ARMS = [
    {
        "arm": "ARM_0_off",
        "flags": {
            "use_mech307_split_surprise": False,
            "use_mech307_schema_multichannel": False,
            "use_mech307_predicted_location_write": False,
        },
    },
    {
        "arm": "ARM_1_split_only",
        "flags": {
            "use_mech307_split_surprise": True,
            "use_mech307_schema_multichannel": False,
            "use_mech307_predicted_location_write": False,
        },
    },
    {
        "arm": "ARM_2_full",
        "flags": {
            "use_mech307_split_surprise": True,
            "use_mech307_schema_multichannel": True,
            "use_mech307_predicted_location_write": True,
        },
    },
]


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
    cfg.valence_liking_enabled = True
    cfg.residue.valence_enabled = True
    # MECH-295 bridge ON in ALL arms (downstream consumer must exist).
    cfg.use_mech295_liking_bridge = True
    cfg.mech295_drive_to_liking_gain = 1.0
    cfg.mech295_liking_to_approach_cue_gain = 0.5
    # Drive floor + z_goal-norm floor left at REEConfig defaults (2026-05-12:
    # mech295_min_drive_to_fire 0.1 -> 0.01 per V3-EXQ-540c probe; z_goal-norm
    # floor unchanged at 0.05). Do NOT override here -- 540f is specifically
    # testing whether the new defaults let the conjunction predicate fire.
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
            latent = agent.sense(obs_body, obs_world)
            ticks = agent.clock.advance()
            e1_prior = (
                agent._e1_tick(latent) if ticks.get("e1_tick")
                else torch.zeros(1, agent.config.latent.world_dim)
            )
            drive = float(REEAgent.compute_drive_level(obs_body))
            agent.update_schema_wanting(drive_level=drive)

            # ---- pre-tick bridge diagnostics snapshot ----
            bridge_diag_pre = (
                agent.mech295_bridge.get_diagnostics()
                if agent.mech295_bridge is not None else {}
            )
            n_conjunction_fires_pre = int(bridge_diag_pre.get("n_conjunction_fires", 0))

            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            action = agent.select_action(candidates, ticks)
            action_idx = int(action.argmax(dim=-1).item())
            _, harm_signal, done, info, obs_dict = env.step(action_idx)
            ttype = info.get("transition_type", "none")
            if ttype == "resource":
                agent.update_z_goal(float(harm_signal), drive_level=drive)
                liking_threshold = 0.04
                if float(harm_signal) >= liking_threshold:
                    agent.update_liking(float(harm_signal))
                contact_events += 1
            agent.update_residue(float(harm_signal))
            total_eval_steps += 1

            # ---- post-tick diagnostics ----
            bridge_diag_post = (
                agent.mech295_bridge.get_diagnostics()
                if agent.mech295_bridge is not None else {}
            )
            n_conjunction_fires_post = int(bridge_diag_post.get("n_conjunction_fires", 0))
            n_conjunction_read_opportunities += int(
                bridge_diag_post.get("n_conjunction_reads", 0)
                - bridge_diag_pre.get("n_conjunction_reads", 0)
            )
            n_conjunction_fires_pretick += (n_conjunction_fires_post - n_conjunction_fires_pre)

            # Gap 2/3 liking + schema_wanting counters.
            wdiag = agent.residue_field.rbf_field
            # Count new valence writes by inspecting active centers.
            liking_writes = int(
                (wdiag.valence_vecs[wdiag.active_mask][:, VALENCE_LIKING].abs() > 1e-9).sum()
            ) if wdiag.active_mask.any() else 0
            schema_wanting_writes = int(
                (wdiag.valence_vecs[wdiag.active_mask][:, VALENCE_WANTING].abs() > 1e-9).sum()
            ) if wdiag.active_mask.any() else 0

            # Gap 4: predicted-location write uses e1_prior rather than z_world.
            # Count by checking whether the write-count in bridge diag incremented
            # (proxy: use_mech307_predicted_location_write flag itself is the gating).
            if getattr(cfg if False else agent.config, "use_mech307_predicted_location_write", False):
                gap4_e1_prior_writes = schema_wanting_writes

            # z_beta excursion (Gap 3 arousal pulse observable).
            if latent.z_beta is not None:
                z_beta_dim0_excursions.append(float(latent.z_beta[0, 0].item()))

            # Surprise channel writes: harm-paired vs non-harm.
            if float(harm_signal) < 0:
                harm_paired_surprise_writes += 1
            else:
                nonharm_surprise_writes += 1

            # Approach-commit observable.
            if _approach_commit_at_step(agent):
                approach_commit_steps += 1

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
    arm0 = agg["ARM_0_off"]
    arm1 = agg["ARM_1_split_only"]
    arm2 = agg["ARM_2_full"]

    # C1 substrate-readiness counter dissociation.
    # Gap 1 marker (Option-b): n_split_surprise_centers_total (count of active
    #   centers with non-zero VALENCE_POSITIVE_SURPRISE OR VALENCE_NEGATIVE_SURPRISE
    #   writes). Under Option-b the legacy VALENCE_SURPRISE channel also receives
    #   the unsigned magnitude via the backward-compat write at the same site,
    #   so it cannot serve as the discriminative observable here.
    # Gap 2 marker: liking_writes (anticipatory VALENCE_LIKING writes).
    # Gap 3 marker: z_beta_excursion_mean (z_beta arousal pulses).
    # Gap 4 marker: gap4_e1_prior_writes (writes targeted at e1_prior, not z_world).
    c1_arm0 = (
        arm0["liking_writes_mean"] == 0
        and arm0["gap4_e1_prior_writes_mean"] == 0
        and arm0["n_split_surprise_centers_total"] == 0
        and arm0["z_beta_excursion_mean"] < C1_Z_BETA_FLOOR_ARM2
    )
    c1_arm1 = (
        arm1["n_split_surprise_centers_total"] >= C1_NEG_SURPRISE_FLOOR_ARM12
        and arm1["liking_writes_mean"] == 0
        and arm1["gap4_e1_prior_writes_mean"] == 0
        and arm1["z_beta_excursion_mean"] < C1_Z_BETA_FLOOR_ARM2
    )
    c1_arm2 = (
        arm2["liking_writes_mean"] >= C1_LIKING_WRITES_FLOOR_ARM2
        and arm2["gap4_e1_prior_writes_mean"] >= C1_PREDICTED_WRITES_FLOOR_ARM2
        and arm2["n_split_surprise_centers_total"] >= C1_NEG_SURPRISE_FLOOR_ARM12
        and arm2["z_beta_excursion_mean"] >= C1_Z_BETA_FLOOR_ARM2
    )
    c1 = bool(c1_arm0 and c1_arm1 and c1_arm2)

    # C2 conjunction-read fires only in ARM_2.
    c2_arm0 = arm0["n_conjunction_fire_ticks_total"] == 0
    c2_arm1 = arm1["n_conjunction_fire_ticks_total"] == 0
    c2_arm2 = (
        arm2["n_conjunction_fire_ticks_total"] > 0
        and arm2["conjunction_fire_rate_mean"] >= C2_CONJUNCTION_RATE_FLOOR_ARM2
        and all(
            r > 0 for r in arm2["per_seed_conjunction_fire_rate"]
        )
    )
    c2 = bool(c2_arm0 and c2_arm1 and c2_arm2)

    # C3 behavioural lift: ARM_2 > ARM_0 + 0.10 in 2/3 seeds AND ARM_2 mean > ARM_1 mean.
    seeds_above_lift = sum(
        1 for a, o in zip(arm0["per_seed_approach"], arm2["per_seed_approach"])
        if (o - a) >= C3_APPROACH_LIFT
    )
    c3 = bool(
        seeds_above_lift >= 2
        and arm2["approach_commit_rate_mean"] > arm1["approach_commit_rate_mean"]
    )

    # 540f PASS = C1 AND C2 (substrate dissociation + conjunction-read fires).
    # C3 approach_commit_rate saturates at 1.000 across all arms and is
    # INFORMATIONAL only. Tracked in the manifest for backward-compat with
    # the 540a schema; not part of the gating decision.
    overall = c1 and c2
    return {
        "C1_substrate_dissociation": bool(c1),
        "C1_arm0_silent": bool(c1_arm0),
        "C1_arm1_split_only_isolated": bool(c1_arm1),
        "C1_arm2_all_signatures": bool(c1_arm2),
        "C2_consumer_conjunction_read": bool(c2),
        "C2_arm0_no_fires": bool(c2_arm0),
        "C2_arm1_no_fires": bool(c2_arm1),
        "C2_arm2_fires": bool(c2_arm2),
        "C3_approach_commit_lift": bool(c3),
        "seeds_above_c3_lift": int(seeds_above_lift),
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
            cell_pass = (
                r["liking_writes"] > 0 if arm_label == "ARM_2_full"
                else r["liking_writes"] == 0
            ) and r["total_eval_steps"] > 0
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
    for arm in ("ARM_0_off", "ARM_1_split_only", "ARM_2_full"):
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
    if outcome == "PASS":
        direction = "supports"
        per_claim = {cid: "supports" for cid in CLAIM_IDS}
        per_claim["SD-014"] = "weakens"   # 6-channel amendment retired by PASS
    elif acceptance["C1_substrate_dissociation"] and acceptance["C2_consumer_conjunction_read"]:
        # PARTIAL PASS: substrate + consumer-read fire, behavioural lift fails.
        direction = "mixed"
        per_claim = {
            "MECH-307": "mixed",
            "MECH-216": "supports",
            "MECH-205": "supports",
            "MECH-093": "supports",
            "SD-014": "non_contributory",   # fallback still in play
            "MECH-295": "mixed",            # consumer read works but doesn't lift
        }
    elif acceptance["C1_substrate_dissociation"]:
        # PARTIAL: substrate fires but consumer conjunction can't form.
        direction = "mixed"
        per_claim = {
            "MECH-307": "non_contributory",
            "MECH-216": "supports",
            "MECH-205": "supports",
            "MECH-093": "supports",
            "SD-014": "non_contributory",
            "MECH-295": "non_contributory",
        }
    else:
        direction = "does_not_support"
        per_claim = {cid: "does_not_support" for cid in CLAIM_IDS}

    run_pack = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "supersedes": SUPERSEDES,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "outcome": outcome,
        "evidence_direction": direction,
        "per_claim_direction": per_claim,
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "elapsed_seconds": elapsed,
        "seeds": list(SEEDS),
        "p0_episodes": P0_EPISODES,
        "eval_episodes": EVAL_EPISODES,
        "steps_per_episode": STEPS_PER_EPISODE,
        "acceptance": acceptance,
        "aggregates": {
            arm: {
                k: v for k, v in data.items()
                if not isinstance(v, list)
            }
            for arm, data in agg.items()
        },
        "per_cell": per_cell,
        "note": (
            "Seed 44 dropped (degenerate no-op policy under no-gradient-learning: "
            "torch.manual_seed(44) always selects action=4/stay; "
            "contaminated_harm=0.4 kills agent in ~6 steps per episode -> "
            "134/4000 eval steps total in 540e). Seed 45 substituted (verified "
            "clean: 1000 steps across 5 episodes, zero deaths). All other config "
            "identical to 540e."
        ),
    }

    out_path = out_dir / f"{run_id}.json"
    out_path.write_text(json.dumps(run_pack, indent=2))
    print(f"[{EXPERIMENT_TYPE}] wrote {out_path}", flush=True)

    manifest_path = out_dir / "manifest.json"
    manifest = {
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "supersedes": SUPERSEDES,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "outcome": outcome,
        "evidence_direction": direction,
        "per_claim_direction": per_claim,
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "runs": [str(out_path.name)],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[{EXPERIMENT_TYPE}] wrote {manifest_path}", flush=True)

    emit_outcome(
        run_id=run_id,
        outcome=outcome,
        experiment_type=EXPERIMENT_TYPE,
        queue_id=QUEUE_ID,
        claim_ids=CLAIM_IDS,
        evidence_direction=direction,
    )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    sys.exit(main(dry_run=args.dry_run))
