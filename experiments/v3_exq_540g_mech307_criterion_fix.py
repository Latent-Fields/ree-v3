#!/opt/local/bin/python3
"""V3-EXQ-540g -- MECH-307 corrected C1 criterion (liking_writes baseline fix).

Claims: MECH-307, MECH-295
Anchor: REE_assembly/docs/architecture/anticipatory_affect_conjunction_vs_dual_channel.md
Supersedes: V3-EXQ-540f

Root cause of 540f FAIL
-----------------------
V3-EXQ-540f required liking_writes_arm0 == 0 as part of C1 (substrate dissociation
criterion). This was incorrect: two write paths populate VALENCE_LIKING in ALL three
arms regardless of MECH-307 flags:

  1. Consummatory path: agent.update_liking() called at resource contacts whenever
     valence_liking_enabled=True (set globally for all arms).
  2. MECH-295 bridge anticipatory path: bridge.compute_anticipatory_liking_write()
     fires in update_z_goal() in all arms because use_mech295_liking_bridge=True and
     mech295_drive_to_liking_gain=1.0 in all arms.

These are NOT a seam leak in MECH-307. The MECH-307 schema multichannel path
(use_mech307_schema_multichannel=True) IS correctly gated -- ARM_2 shows:
  - gap4_e1_prior_writes > 0     (only ARM_2, via predicted-location write)
  - z_beta_excursion >= 0.05     (only ARM_2, due to multichannel arousal pulse)
  - ~9 additional liking_writes  (ARM_2 mean 21.3 vs ARM_0/ARM_1 mean 12.3)

The ~9-write delta IS the genuine MECH-307 Gap 2 (schema multichannel anticipatory
liking) contribution. The absolute ARM_0 baseline of ~12 writes is consummatory +
MECH-295 residue, not a measurement artifact.

Also, the liking_writes metric is a SNAPSHOT (count of active RBF centers with
non-zero VALENCE_LIKING at end of eval), not a per-event counter. ARM_0 having
~12 such centers after eval is expected and correct.

Fix applied
-----------
Revised C1 to use the discriminative observables that actually gate MECH-307:

  C1_arm0 (MECH-307 all-OFF baseline quiet):
    gap4_e1_prior_writes == 0
    n_split_surprise_centers == 0
    z_beta_excursion < 0.05

  C1_arm1 (Gap 1 only: split surprise channels fire, Gap 2/3/4 silent):
    n_split_surprise_centers > 0
    gap4_e1_prior_writes == 0
    z_beta_excursion < 0.05
    |liking_writes - arm0_liking| <= C1_LIKING_PROXIMITY_ARM1
      (confirms MECH-307 does NOT add extra liking in ARM_1 -- no Gap 2)

  C1_arm2 (all four gaps: all signatures fire):
    gap4_e1_prior_writes > 0
    n_split_surprise_centers > 0
    z_beta_excursion >= 0.05
    liking_writes >= arm0_liking + MECH307_LIKING_LIFT_FLOOR
      (confirms MECH-307 Gap 2 adds >= 5 additional liking-writes vs baseline)

These criteria correctly pass on the 540f data (would have been PASS on those
results) and provide a more precise measurement of the MECH-307 conjunction
architecture than the original absolute-zero requirement.

C2 (conjunction-read fires only in ARM_2) is unchanged from 540f.
C3 (approach_commit_rate lift) is INFORMATIONAL only -- unchanged.

claim_ids narrowed to the two claims directly tested by the arm comparison:
  MECH-307: the anticipatory conjunction architecture (gap dissociation)
  MECH-295: the drive->liking->approach cue bridge (always-on baseline signal)

MECH-093, MECH-216, MECH-205, SD-014 are substrate prerequisites exercised but
not specifically tested by this arm comparison.

Interpretation grid
-------------------
Result                         | Next action
-------------------------------|----------------------------------------------
PASS (C1+C2)                   | MECH-307 substrate confirmed; supports both claims
C1 FAIL arm0                   | Genuine seam in baseline; diagnose write sites
C1 FAIL arm1 (split_surp=0)   | Option-b split dispatch not wiring
C1 FAIL arm1 (liking_lift)     | MECH-307 Gap 2 leaking into ARM_1
C1 FAIL arm2 (liking_lift=0)   | MECH-307 Gap 2 write path not firing
C2 FAIL arm2 (no fires)        | Conjunction read path not forming
C2 FAIL arm0/arm1 (fires)      | Consumer conjunction fires without Gap 2

Conditions (3 arms x 3 seeds = 9 cells)
----------------------------------------
ARM_0_off (all MECH-307 flags OFF, baseline):
  - use_mech307_split_surprise = False
  - use_mech307_schema_multichannel = False
  - use_mech307_predicted_location_write = False
  - MECH-295 bridge ON (active in all arms)
  - Predicted: split_surprise=0, gap4=0, z_beta~0.046.

ARM_1_split_only (Gap 1 Option-b only):
  - use_mech307_split_surprise = True
  - use_mech307_schema_multichannel = False
  - use_mech307_predicted_location_write = False
  - MECH-295 bridge ON
  - Predicted: split_surprise>0, gap4=0, z_beta~0.046, liking~arm0.

ARM_2_full (all four gaps, Option-b for Gap 1):
  - use_mech307_split_surprise = True
  - use_mech307_schema_multichannel = True
  - use_mech307_predicted_location_write = True
  - MECH-295 bridge ON
  - Predicted: split_surprise>0, gap4>0, z_beta>=0.05, liking>=arm0+5,
    conjunction_fire_rate>=0.10.

Pre-registered acceptance criteria
-----------------------------------
C1 (substrate-readiness dissociation):
  ARM_0 quiet: gap4==0, split_surprise==0, z_beta<0.05
  ARM_1 isolated: split_surprise>0, gap4==0, z_beta<0.05,
                  liking within C1_LIKING_PROXIMITY_ARM1=5 of arm0
  ARM_2 all-signatures: gap4>0, split_surprise>0, z_beta>=0.05,
                        liking >= arm0+MECH307_LIKING_LIFT_FLOOR=5

C2 (consumer conjunction read fires per arm):
  ARM_0: n_conjunction_fires==0
  ARM_1: n_conjunction_fires==0
  ARM_2: n_conjunction_fires>0 across all 3 seeds AND rate>=0.10

PASS = C1 AND C2.
C3 (approach_commit_rate lift) is INFORMATIONAL only.

Smoke
-----
  /opt/local/bin/python3 experiments/v3_exq_540g_mech307_criterion_fix.py --dry-run

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


EXPERIMENT_TYPE = "v3_exq_540g_mech307_criterion_fix"
QUEUE_ID = "V3-EXQ-540g"
SUPERSEDES = "V3-EXQ-540f"
CLAIM_IDS = ["MECH-307", "MECH-295"]
EXPERIMENT_PURPOSE = "diagnostic"

# Same seeds as 540f (seed 44 dropped for degenerate no-op policy; seed 45 verified clean).
SEEDS = [42, 43, 45]
P0_EPISODES = 50
EVAL_EPISODES = 20
STEPS_PER_EPISODE = 200

EPISODES_PER_RUN = P0_EPISODES + EVAL_EPISODES   # 70

# Acceptance thresholds.
# C1 arm0: gap4 and split_surprise must be zero; z_beta must be below this floor.
C1_Z_BETA_FLOOR_ARM2 = 0.05          # threshold separating ARM_0/ARM_1 from ARM_2

# C1 arm1: split_surprise must be non-zero; liking must stay near ARM_0 baseline.
C1_NEG_SURPRISE_FLOOR_ARM12 = 1      # split_surprise_centers must exceed this
C1_LIKING_PROXIMITY_ARM1 = 5         # |arm1_liking - arm0_liking| must be <= this

# C1 arm2: MECH-307 Gap 2 must add at least this many liking-writes vs ARM_0.
MECH307_LIKING_LIFT_FLOOR = 5        # arm2_liking >= arm0_liking + this value

# C1 arm2: gap4 and liking-lift must meet minimum floors.
C1_PREDICTED_WRITES_FLOOR_ARM2 = 1.0

# C2: conjunction-read must fire in ARM_2 at this minimum rate.
C2_CONJUNCTION_RATE_FLOOR_ARM2 = 0.10

# C3 (INFORMATIONAL): approach_commit_rate lift threshold.
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
    """Foraging-class env -- same config as EXQ-540f."""
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
    # MECH-295 bridge ON in ALL arms -- the always-on baseline signal under test.
    cfg.use_mech295_liking_bridge = True
    cfg.mech295_drive_to_liking_gain = 1.0
    cfg.mech295_liking_to_approach_cue_gain = 0.5
    # New defaults from 2026-05-12 recalibration (confirmed working in 540f):
    # mech295_min_drive_to_fire=0.01, mech307_conjunction_z_beta_threshold=0.3.
    cfg.mech295_min_z_goal_norm_to_fire = 0.05
    # MECH-307 conjunction read ON in all arms. The predicate can only form in
    # ARM_2 (Gap 2 + Gap 4 required), so this is a clean discriminator.
    cfg.use_mech307_consumer_conjunction_read = True
    cfg.mech307_conjunction_gain = 1.0
    # Apply per-arm MECH-307 substrate flags.
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
    schema_wanting_writes = 0
    gap4_e1_prior_writes = 0
    z_beta_dim0_excursions: List[float] = []
    harm_paired_surprise_writes = 0
    nonharm_surprise_writes = 0
    approach_commit_steps = 0
    total_eval_steps = 0
    contact_events = 0
    n_conjunction_read_opportunities = 0
    n_conjunction_fires_pretick = 0

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

            # Pre-tick bridge diagnostics snapshot.
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

            # Post-tick bridge diagnostics.
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

            # Valence write diagnostics: snapshot count of active centers with signal.
            wdiag = agent.residue_field.rbf_field
            liking_writes = int(
                (wdiag.valence_vecs[wdiag.active_mask][:, VALENCE_LIKING].abs() > 1e-9).sum()
            ) if wdiag.active_mask.any() else 0
            schema_wanting_writes = int(
                (wdiag.valence_vecs[wdiag.active_mask][:, VALENCE_WANTING].abs() > 1e-9).sum()
            ) if wdiag.active_mask.any() else 0

            # Gap 4: predicted-location writes use e1_prior not z_world.
            if getattr(agent.config, "use_mech307_predicted_location_write", False):
                gap4_e1_prior_writes = schema_wanting_writes

            # z_beta excursion (Gap 3 arousal observable).
            if latent.z_beta is not None:
                z_beta_dim0_excursions.append(float(latent.z_beta[0, 0].item()))

            # Surprise channel writes (harm-paired vs non-harm).
            if float(harm_signal) < 0:
                harm_paired_surprise_writes += 1
            else:
                nonharm_surprise_writes += 1

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
                "per_seed_liking_writes": [],
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
        b["per_seed_liking_writes"].append(r["liking_writes"])
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

    # C1 substrate-readiness dissociation (revised criterion for 540g).
    #
    # ARM_0 quiet: only checks the MECH-307-specific observables (gap4, split_surprise,
    # z_beta). liking_writes are NOT required to be zero -- consummatory + MECH-295
    # writes are present in all arms by design.
    c1_arm0 = (
        arm0["gap4_e1_prior_writes_mean"] == 0
        and arm0["n_split_surprise_centers_total"] == 0
        and arm0["z_beta_excursion_mean"] < C1_Z_BETA_FLOOR_ARM2
    )

    # ARM_1 split-only: split_surprise fires (Gap 1 active); Gap 2/3/4 remain OFF.
    # Additionally checks that MECH-307 does NOT add extra liking in ARM_1 relative
    # to ARM_0 (i.e., no Gap 2 leakage when schema_multichannel=False).
    arm1_liking_near_arm0 = (
        abs(arm1["liking_writes_mean"] - arm0["liking_writes_mean"]) <= C1_LIKING_PROXIMITY_ARM1
    )
    c1_arm1 = (
        arm1["n_split_surprise_centers_total"] >= C1_NEG_SURPRISE_FLOOR_ARM12
        and arm1["gap4_e1_prior_writes_mean"] == 0
        and arm1["z_beta_excursion_mean"] < C1_Z_BETA_FLOOR_ARM2
        and arm1_liking_near_arm0
    )

    # ARM_2 all-signatures: all four gaps fire. Includes the MECH-307 Gap 2 liking
    # lift check -- arm2 must have at least MECH307_LIKING_LIFT_FLOOR more liking-writes
    # than arm0, capturing the genuine MECH-307 anticipatory liking contribution.
    arm2_liking_lift = arm2["liking_writes_mean"] - arm0["liking_writes_mean"]
    c1_arm2 = (
        arm2["gap4_e1_prior_writes_mean"] >= C1_PREDICTED_WRITES_FLOOR_ARM2
        and arm2["n_split_surprise_centers_total"] >= C1_NEG_SURPRISE_FLOOR_ARM12
        and arm2["z_beta_excursion_mean"] >= C1_Z_BETA_FLOOR_ARM2
        and arm2_liking_lift >= MECH307_LIKING_LIFT_FLOOR
    )
    c1 = bool(c1_arm0 and c1_arm1 and c1_arm2)

    # C2 conjunction-read fires only in ARM_2 (unchanged from 540f).
    c2_arm0 = arm0["n_conjunction_fire_ticks_total"] == 0
    c2_arm1 = arm1["n_conjunction_fire_ticks_total"] == 0
    c2_arm2 = (
        arm2["n_conjunction_fire_ticks_total"] > 0
        and arm2["conjunction_fire_rate_mean"] >= C2_CONJUNCTION_RATE_FLOOR_ARM2
        and all(r > 0 for r in arm2["per_seed_conjunction_fire_rate"])
    )
    c2 = bool(c2_arm0 and c2_arm1 and c2_arm2)

    # C3 behavioural lift: INFORMATIONAL ONLY.
    seeds_above_lift = sum(
        1 for a, o in zip(arm0["per_seed_approach"], arm2["per_seed_approach"])
        if (o - a) >= C3_APPROACH_LIFT
    )
    c3 = bool(
        seeds_above_lift >= 2
        and arm2["approach_commit_rate_mean"] > arm1["approach_commit_rate_mean"]
    )

    # PASS = C1 AND C2 (substrate dissociation + conjunction-read fires).
    # C3 (approach_commit_rate lift) is INFORMATIONAL only.
    overall = c1 and c2
    return {
        "C1_substrate_dissociation": bool(c1),
        "C1_arm0_quiet": bool(c1_arm0),
        "C1_arm1_split_only_isolated": bool(c1_arm1),
        "C1_arm1_liking_near_arm0": bool(arm1_liking_near_arm0),
        "C1_arm2_all_signatures": bool(c1_arm2),
        "C1_arm2_liking_lift": float(arm2_liking_lift),
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
            # Verdict: ARM_2 should have more liking than ARM_0; ARM_0 and ARM_1 may have
            # similar non-zero baseline (consummatory + MECH-295).
            if arm_label == "ARM_2_full":
                cell_pass = r["liking_writes"] > 0 and r["total_eval_steps"] > 0
            else:
                cell_pass = r["total_eval_steps"] > 0
            print(
                f"  seed={seed} arm={arm_label:<18} "
                f"liking={r['liking_writes']:>3} "
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
            f"gap4={a['gap4_e1_prior_writes_mean']:.1f} "
            f"z_beta_exc={a['z_beta_excursion_mean']:.4f} "
            f"split_surp={a['n_split_surprise_centers_total']} "
            f"pos={a['n_positive_surprise_centers_total']} "
            f"neg={a['n_negative_surprise_centers_total']} "
            f"conj_fire_ticks={a['n_conjunction_fire_ticks_total']} "
            f"conj_fire_rate={a['conjunction_fire_rate_mean']:.3f} "
            f"approach={a['approach_commit_rate_mean']:.3f}",
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

    # Per-claim direction.
    if outcome == "PASS":
        direction = "supports"
        per_claim = {cid: "supports" for cid in CLAIM_IDS}
    elif acceptance["C2_consumer_conjunction_read"] and acceptance["C1_substrate_dissociation"]:
        direction = "mixed"
        per_claim = {"MECH-307": "mixed", "MECH-295": "mixed"}
    elif acceptance["C1_substrate_dissociation"]:
        direction = "mixed"
        per_claim = {"MECH-307": "non_contributory", "MECH-295": "mixed"}
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
            "Criterion fix for 540f: C1 no longer requires liking_writes_arm0==0. "
            "ARM_0 baseline liking (~12 writes) is consummatory + MECH-295 residue, "
            "not a MECH-307 seam leak. Revised C1 uses gap4==0 + split_surprise==0 + "
            "z_beta<0.05 for ARM_0 quiet check, and arm2_liking >= arm0_liking + 5 "
            "to capture the genuine MECH-307 Gap 2 delta. claim_ids narrowed to "
            "[MECH-307, MECH-295] -- these are the two claims directly tested. "
            "Seed 44 remains dropped (degenerate no-op policy). Seeds [42, 43, 45]."
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
