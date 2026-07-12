#!/opt/local/bin/python3
"""V3-EXQ-540e -- MECH-307 default-fix behavioural validation (3-arm under new defaults).

Claims: MECH-307, MECH-216, MECH-205, MECH-093, SD-014, MECH-295
Anchor: REE_assembly/docs/architecture/anticipatory_affect_conjunction_vs_dual_channel.md
Supersedes: V3-EXQ-540d (requeued V3-EXQ-540c read-site probe).

Validation run after the 2026-05-12 two-default fix:
  mech295_min_drive_to_fire        0.1 -> 0.01
  mech307_conjunction_z_beta_threshold  0.6 -> 0.3

V3-EXQ-540c (read-site probe) confirmed at 10x scale across 1087 bridge
calls / 34784 candidate-reads that conj_fire_rate=0 across the 540a/540b/540c
chain was due to two default-value miscalibrations, not a code bug. At the
half tier (0.3/0.15/0.3) the four-way predicate clears 94.66% of candidates
WHEN the drive gate doesn't short-circuit first; the legacy drive floor 0.1
short-circuited 100% of calls under standard env config (drive_level
max=0.030, never reached 0.1). Lowering both defaults clears the gates that
were structurally unreachable in the env. 540e replays the 540a 3-arm
decomposition under these defaults.

This is a 3-arm structural-AND-behavioural test under the now-canonical
config. The same arm structure as 540a; what changes is the SHARED defaults
applied to all three arms via REEConfig.from_dims. The discriminative power
is unchanged (ARM_0 master OFF -> substrate silent; ARM_1 only Gap 1 ->
split_surp_centers > 0; ARM_2 all gaps -> all four fire), but ARM_2 is now
expected to ALSO produce non-zero conjunction-fire-rate and non-zero
dacc_score_bias contribution.

Phase 1 follow-on after EXQ-539. EXQ-539 (2-arm binary, 2026-05-08) FAILed
C5 (approach_commit_rate lift): substrate counters C1-C4 PASS but the legacy
MECH-295 cue path (drive * goal_proximity only) does not read the conjunction
signal. This experiment runs the 3-arm gap-decomposition described in the
anchor doc lines 224-243, with one architectural change relative to EXQ-539:
the MECH-295 bridge's new compute_conjunction_score_bias() path is ON in all
arms so a downstream consumer can actually read the conjunction signal.

ARM_3 (SD-014 6-channel fallback) is DEFERRED -- requires substrate work
(VALENCE_EXCITEMENT + VALENCE_DREAD discrete channels) that is not yet built.
If ARM_2 fails C3 here, ARM_3 becomes the natural follow-on.

Conditions (3 arms x 3 seeds = 9 cells)
---------------------------------------
ARM_0_off (replicates EXQ-539 ARM_OFF substrate):
  - All MECH-307 substrate flags False (master + all three sub-flags).
  - Bridge consumer-conjunction-read flag ON (no-op when substrate fixes are
    off because conjunction predicates can't form).
  - Predicted: substrate counters at zero; conjunction-fire counter at zero;
    approach_commit_rate at the EXQ-536b inert baseline.

ARM_1_split_only (Gap 1 only, Option-b):
  - use_mech307_split_surprise = True   (Option-b: split channels)
  - use_mech307_schema_multichannel = False
  - use_mech307_predicted_location_write = False
  - Bridge consumer-conjunction-read flag ON.
  - Predicted: positive-surprise OR negative-surprise channel centers
    accumulate on harm + non-harm contacts (signed channels populated;
    legacy VALENCE_SURPRISE channel also receives magnitude via the
    backward-compat write at the same site). VALENCE_LIKING / z_beta-pulse
    counters at zero. Conjunction predicate cannot fire (Gap 2 missing
    means VALENCE_LIKING stays zero, so liking > threshold never holds).
    approach_commit_rate at-or-near ARM_0 baseline.

ARM_2_full_conjunction (all four gaps, Option-b for Gap 1):
  - use_mech307_split_surprise = True              (Gap 1 Option-b)
  - use_mech307_schema_multichannel = True         (Gaps 2 + 3)
  - use_mech307_predicted_location_write = True    (Gap 4)
    Equivalent setting: use_mech307_conjunction = True (master flag whose
    __post_init__ resolver flips all three sub-flags). This script sets
    sub-flags directly for per-arm transparency.
  - Bridge consumer-conjunction-read flag ON.
  - Predicted: all four substrate counters fire; conjunction predicate
    fires at predicted-imminent locations; bridge applies an additional
    negative score_bias on conjunction-state candidates; approach_commit_rate
    lifts above ARM_0.

Pre-registered acceptance criteria
----------------------------------
C1 (substrate-readiness counter dissociation, per-gap):
   ARM_0: liking_writes == 0, predicted_writes == 0,
          split_surprise_centers == 0 (POS + NEG channels empty),
          z_beta_excursion ~ 0.
   ARM_1: split_surprise_centers > 0 (Option-b POS or NEG channel populated);
          liking_writes == 0; predicted_writes == 0;
          z_beta_excursion ~ 0. (Gap 1 isolated.)
   ARM_2: all four signatures fire (liking_writes > 0, predicted_writes > 0,
          split_surprise_centers > 0, z_beta_excursion >= 0.05).

C2 (consumer-side conjunction read fires per arm as predicted):
   ARM_0: n_conjunction_fires == 0 (no conjunction can form).
   ARM_1: n_conjunction_fires == 0 (Gap 1 alone insufficient -- liking
          stays zero so v_l > liking_threshold never holds).
   ARM_2: n_conjunction_fires > 0 across all 3 seeds AND
          per-seed-mean conjunction-detection-rate >= 0.10 (10% of read
          opportunities show at least one candidate firing the predicate).

C3 (load-bearing behavioural -- approach_commit_rate lift):
   ARM_2 approach_commit_rate >= ARM_0 + 0.10 across at least 2 of 3 seeds
   AND ARM_2 mean > ARM_1 mean (full conjunction beats Gap-1-alone).

PASS = C1 AND C2 AND C3 -> MECH-307 conjunction architecture is the
  correct read-side wiring; SD-014 6-channel amendment can be retired.

PARTIAL PASS (C1 + C2 PASS, C3 FAIL) -> the conjunction signal forms AND
  the consumer reads it, but the additional approach bias is not enough
  to lift commit rate. Routes to SD-014 6-channel ARM_3 follow-on (the
  reading question becomes whether ONE channel is easier to consume than
  FOUR), or to a parameter sweep on mech307_conjunction_gain.

PARTIAL PASS (C1 PASS, C2 FAIL) -> substrate fires but the conjunction
  predicate doesn't compose at predicted locations -- threshold sweep
  needed before any architectural conclusion.

FAIL on C1 -> substrate fix incorrect for one or more gaps; route to
  /diagnose-errors.

C3 NOTE (saturated metric, kept observable but not load-bearing):
The 540a/540b runs both observed approach_commit_rate=1.000 in EVERY arm
including ARM_0_off. The `_approach_commit_at_step` predicate (beta_elevated
AND wanting > 0.05) triggers trivially under this env config and provides
no discriminative signal. 540e retains the predicate for backward-compat
with 540a's manifest schema but C3 is informational, not gating. The
load-bearing signal is C2 conjunction_fire_rate (now expected to lift
from 0 in 540a/b to >= 0.50 in ARM_2 under the new defaults).

Smoke
-----
  /opt/local/bin/python3 experiments/v3_exq_540e_mech307_default_fix_validation.py --dry-run

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
from experiments.pack_writer import write_flat_manifest  # noqa: E402


EXPERIMENT_TYPE = "v3_exq_540e_mech307_default_fix_validation"
QUEUE_ID = "V3-EXQ-540e"
SUPERSEDES = "V3-EXQ-540d"
CLAIM_IDS = ["MECH-307", "MECH-216", "MECH-205", "MECH-093", "SD-014", "MECH-295"]
EXPERIMENT_PURPOSE = "diagnostic"   # gap decomposition; not single-claim evidence

SEEDS = [42, 43, 44]
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
    cfg.residue.valence_enabled = True
    # MECH-295 bridge ON in ALL arms (downstream consumer must exist).
    cfg.use_mech295_liking_bridge = True
    cfg.mech295_drive_to_liking_gain = 1.0
    cfg.mech295_liking_to_approach_cue_gain = 0.5
    # Drive floor + z_goal-norm floor left at REEConfig defaults (2026-05-12:
    # mech295_min_drive_to_fire 0.1 -> 0.01 per V3-EXQ-540c probe; z_goal-norm
    # floor unchanged at 0.05). Do NOT override here -- 540e is specifically
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

    # 540e PASS = C1 AND C2 (substrate dissociation + conjunction-read fires).
    # C3 approach_commit_rate saturates at 1.000 across all arms in 540a/540b/540e
    # and is INFORMATIONAL only. The separable metric-redesign session that
    # would fix the saturation is out of scope here. Tracked in the manifest
    # for backward-compat with the 540a schema; not part of the gating decision.
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
            "MECH-307": "mixed",
            "MECH-216": "supports",
            "MECH-205": "supports",
            "MECH-093": "non_contributory",
            "SD-014": "non_contributory",
            "MECH-295": "weakens",
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
            "C1_liking_writes_floor_arm2": C1_LIKING_WRITES_FLOOR_ARM2,
            "C1_predicted_writes_floor_arm2": C1_PREDICTED_WRITES_FLOOR_ARM2,
            "C1_z_beta_floor_arm2": C1_Z_BETA_FLOOR_ARM2,
            "C1_neg_surprise_floor_arm12": C1_NEG_SURPRISE_FLOOR_ARM12,
            "C2_conjunction_rate_floor_arm2": C2_CONJUNCTION_RATE_FLOOR_ARM2,
            "C3_approach_lift": C3_APPROACH_LIFT,
        },
        "note": (
            "V3-EXQ-540e -- MECH-307 default-fix behavioural validation. "
            "Supersedes V3-EXQ-540d (requeued V3-EXQ-540c read-site probe). "
            "Runs the 540a 3-arm decomposition under the 2026-05-12 default "
            "fix: mech295_min_drive_to_fire 0.1 -> 0.01 and "
            "mech307_conjunction_z_beta_threshold 0.6 -> 0.3. V3-EXQ-540c "
            "(read-site probe, 10x scale) confirmed the conjunction predicate "
            "components clear 94.66% of candidates at the half tier "
            "(0.3/0.15/0.3) IF the drive gate doesn't block; the legacy 0.1 "
            "drive floor was never crossed under the standard env config "
            "(drive_level max=0.030 across 1087 bridge calls). With both "
            "defaults lowered, 540e tests whether the conjunction predicate "
            "now fires in ARM_2 and whether the bridge's score-bias "
            "contribution discriminates ARM_2 from ARM_0. Same arms as 540a "
            "(ARM_0_off, ARM_1_split_only, ARM_2_full). C1 substrate "
            "dissociation expected PASS (no change in substrate writes). C2 "
            "conjunction-read expected to FIRE in ARM_2 (probe predicts "
            "rate >= 0.50). C3 approach_commit_rate retained for backward-"
            "compat manifest schema but informational only -- the 540a/540b "
            "metric saturates at 1.0 across all arms and provides no "
            "discriminative signal. PASS = C1 AND C2; PARTIAL (C1+C2 PASS, "
            "C3 FAIL) acceptable. FAIL on C1 -> /diagnose-errors. FAIL on C2 "
            "despite default fix -> the probe hypothesis was wrong; re-audit "
            "the bridge consumer-read path (possible candidates: v_s reads "
            "the legacy VALENCE_SURPRISE channel rather than "
            "VALENCE_POSITIVE_SURPRISE under Option-b -- semantic mismatch "
            "deferred to a separate session). ARM_3 (SD-014 6-channel "
            "fallback) remains deferred -- only triggered if 540e FAILs C2."
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
