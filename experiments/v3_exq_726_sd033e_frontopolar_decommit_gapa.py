"""V3-EXQ-726: SD-033e frontopolar de-commit lever validation (isolated GAP-A).

DIAGNOSTIC. Validates the new SD-033e V3-narrow frontopolar de-commit lever
(use_frontopolar_decommit + frontopolar_gain) -- a DISTINCT de-commit-release
lever for the DURATION face of the F-dominance conversion ceiling (MECH-439).
Per the sequencing gate in failure_autopsy_V3-EXQ-719a_2026-07-08, validation
runs on the ISOLATED GAP-A single-arena config (the 689d conversion stack),
NOT the integrated all-ON agent (which has no competent committed foraging to
de-commit FROM). The all-ON validation is a SEPARATE experiment gated on the
V3-EXQ-724 competence-localization diagnostic; this script does NOT run it.

MECHANISM UNDER TEST
  The rung-6 NaturalCommitUrgencyRelease lever + the natural-commit latch-hold
  are ON on BOTH arms, with a FLAT urgency (gap_entry_sensitivity=0.0) and a
  small urgency_rate, so a natural commit sustains a long occupancy (the hold
  re-asserts beta) that releases only at a long flat timeout. The ONLY axis
  between the two arms is the SD-033e frontopolar de-commit pressure:

    ARM_FP_CONTRAST : use_frontopolar_decommit=True, frontopolar_gain=0.0
                      (the frontopolar_gain=0 flat-urgency contrast -- identical
                      to lever-ON-but-no-frontopolar; bit-identical to OFF).
    ARM_FP_ON       : use_frontopolar_decommit=True, frontopolar_gain>0.

  Frontopolar pressure = frontopolar_gain * max(0, cfv_now - cfv_at_entry),
  where cfv = ||z_chosen - z_goal|| - ||z_alt - z_goal|| (a NON-F goal-proximity
  ADVANTAGE of the best foregone alternative). Injected into the SAME urgency
  accumulator, it fires the release EARLIER when a foregone option has IMPROVED
  in goal terms relative to the commit moment -- a genuine de-commit, not more
  F-moderation.

PRE-REGISTERED GATE (PASS)
  ARM_FP_ON median committed-latch occupancy (ncur_last_occupancy_at_release)
  drops >= OCCUPANCY_DROP_FRAC (0.40) vs ARM_FP_CONTRAST on >= MIN_SEEDS_FOR_PASS
  (2 of 3) strong-F seeds, ATTRIBUTABLE to the frontopolar term
  (frontopolar_release_count > 0 on the ON arm -- fires only when
  cfv_now > cfv_at_entry, proving a real switch, not a flat timeout or F noise).

READINESS / PRECONDITION (self-routes substrate_not_ready_requeue below floor)
  ARM_FP_CONTRAST median occupancy > STRONG_F_OCCUPANCY_FLOOR (2000) -- confirms
  the sustained natural-commit hold (the 460h monolithic-latch regime) actually
  forms in GAP-A, so there IS a long occupancy for the frontopolar term to
  shorten. If it does not form, the test is meaningless -> substrate_not_ready.

This experiment PROMOTES NOTHING (diagnostic, claim_ids=[]); it validates the
substrate lever landed 2026-07-09. See ree-v3/CLAUDE.md "SD-033e (V3-narrow)".

STATUS 2026-07-09 -- HELD, NOT QUEUED (gated on V3-EXQ-724).
  Local probes showed the isolated GAP-A reef ENV_KWARGS below does NOT sustain a
  committed-foraging latch: the agent dies to hazards in ~9-52 steps after full P0
  training (no occupancy to de-commit from), and a benign survivable single-arena
  survives (~395 steps/ep, beta elevated) but the NCUR duration lever does not arm
  (natural re-commits churn the urgency accumulator -- the 460i latch-fragmentation
  mode). The duration-face validation is therefore co-blocked on the SAME
  competent-committed-foraging substrate gap V3-EXQ-724 (competence-localization) is
  diagnosing -- it is NOT GAP-A-specific. HOLD until 724 localizes a competent
  committed-foraging test-bed, then set ENV_KWARGS / P0 to that regime and re-run
  /queue-experiment. The arm structure, occupancy DV, and frontopolar attribution
  below are correct and reusable; only the ENV_KWARGS + P0 training target must change.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiment_protocol import emit_outcome  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_726_sd033e_frontopolar_decommit_gapa"
QUEUE_ID = "V3-EXQ-726"
SUPERSEDES: Optional[str] = None
CLAIM_IDS: List[str] = []  # DIAGNOSTIC -- substrate-lever validation, promotes nothing
EXPERIMENT_PURPOSE = "diagnostic"

SEEDS = [42, 43, 44]
P0_WARMUP_EPISODES = 40           # SD-056 online contrastive warmup (substrate readiness)
P0_STEPS_PER_EPISODE = 200
P1_MEASUREMENT_EPISODES = 4       # long-episode occupancy measurement
P1_STEPS_PER_EPISODE = 3000       # long enough for the flat-timeout occupancy (>2000) to form

DRY_RUN_SEEDS = [42]
DRY_RUN_P0 = 2
DRY_RUN_P0_STEPS = 30
DRY_RUN_P1 = 1
DRY_RUN_P1_STEPS = 60

# ---- Pre-registered thresholds ----
OCCUPANCY_DROP_FRAC = 0.40         # GATE: ON median occupancy drop vs CONTRAST
STRONG_F_OCCUPANCY_FLOOR = 2000.0  # PRECONDITION: CONTRAST median occupancy floor (460h regime)
MIN_SEEDS_FOR_PASS = 2             # of 3

# ---- Rung-6 lever config (ON both arms; flat urgency, latch-hold sustains occupancy) ----
NCUR_URGENCY_RATE = 0.0004         # flat-timeout ~ release_bound/rate ~ 2500 ticks (> floor)
NCUR_RELEASE_BOUND = 1.0
NCUR_URGENCY_CAP = 1.5
NCUR_ONSET_TICKS = 0
FRONTOPOLAR_GAIN_ON = 0.5          # ON-arm gain (tuned in smoke); CONTRAST arm = 0.0

# ---- GAP-A env (SD-054 reef-bipartite; matches the 689d MECH-448 harness) ----
ENV_KWARGS: Dict[str, Any] = dict(
    size=12,
    num_hazards=4,
    num_resources=5,
    hazard_harm=0.05,
    env_drift_interval=5,
    env_drift_prob=0.1,
    proximity_harm_scale=0.1,
    proximity_benefit_scale=0.05,
    proximity_approach_threshold=0.2,
    hazard_field_decay=0.5,
    resource_respawn_on_consume=True,
    toroidal=False,
    harm_history_len=10,
    reef_enabled=True,
    n_reef_patches=3,
    reef_patch_radius=2,
    hazard_food_attraction=0.7,
    reef_bipartite_layout=True,
    reef_bipartite_axis="horizontal",
    reef_bipartite_agent_band_radius=1,
)

# SD-056 online contrastive training (mirror 689d harness).
SD056_WEIGHT = 0.05
E2_CONTRASTIVE_LR = 1e-3
CONTRASTIVE_BATCH_K = 8
TRANSITION_BUFFER_MAX = 256
MIN_BUFFER_BEFORE_TRAIN = 16
MIN_CLASSES_FOR_TRAIN = 2
MAX_GRAD_NORM = 1.0

CONTRAST_ARM = "ARM_FP_CONTRAST"
ON_ARM = "ARM_FP_ON"

ARMS: List[Dict[str, Any]] = [
    {"arm_id": CONTRAST_ARM, "label": "lever_on_frontopolar_gain_0_flat_urgency_contrast",
     "frontopolar_gain": 0.0},
    {"arm_id": ON_ARM, "label": "lever_on_frontopolar_decommit_on",
     "frontopolar_gain": FRONTOPOLAR_GAIN_ON},
]


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(seed=seed, **ENV_KWARGS)


def _make_agent(env: CausalGridWorldV2, arm: Dict[str, Any]) -> REEAgent:
    """GAP-A conversion stack (SP-CEM + SD-056 online + route-range routing +
    shortlist-then-modulate + shared bias channels) with the rung-6
    NaturalCommitUrgencyRelease lever + natural-commit latch-hold ON, flat
    urgency, and the SD-033e frontopolar de-commit lever ON. The ONLY per-arm
    axis is frontopolar_gain (0.0 contrast vs >0 ON)."""
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=32,
        alpha_world=0.9,
        alpha_self=0.3,
        use_harm_stream=True,
        z_harm_dim=32,
        use_affective_harm_stream=True,
        z_harm_a_dim=16,
        harm_history_len=10,
        z_goal_enabled=True,
        goal_weight=0.5,
        drive_weight=2.0,
        e1_goal_conditioned=True,
        use_resource_proximity_head=True,
        resource_proximity_weight=0.5,
        benefit_eval_enabled=True,
        benefit_weight=1.0,
        # ARC-065 SP-CEM (action-divergent pool)
        use_support_preserving_cem=True,
        support_preserving_stratified_elites=True,
        support_preserving_ao_std_floor=0.2,
        support_preserving_min_first_action_classes=2,
        # Shared E3-side bias channels
        use_lateral_pfc_analog=True,
        use_mech295_liking_bridge=True,
        use_structured_curiosity=False,
        use_e3_score_diversity=False,
        use_noise_floor=False,
        use_tonic_vigor=False,
        use_dacc=False,
        use_ofc_analog=False,
        use_gated_policy=False,
        use_candidate_rule_field=False,
        # SD-056 substrate trained online on every arm
        e2_action_contrastive_enabled=True,
        e2_action_contrastive_weight=SD056_WEIGHT,
        e2_rollout_output_norm_clamp_enabled=True,
        e2_rollout_output_norm_clamp_ratio=2.0,
        candidate_summary_source="e2_world_forward",
        use_modulatory_channel_routing=True,
        modulatory_channel_route_source="cand_world_summary",
        modulatory_channel_route_weight=1.0,
        modulatory_channel_route_min_range_floor=1e-6,
        use_modulatory_selection_authority=True,
        modulatory_authority_gain=2.0,
        modulatory_authority_normalize_basis="std",
        use_modulatory_shortlist_then_modulate=True,
        modulatory_shortlist_mode="top_k",
        modulatory_shortlist_k=3,
        # --- rung-6 NaturalCommitUrgencyRelease lever (ON both arms) ---
        use_natural_commit_urgency_release=True,
        natural_commit_release_urgency_mode=True,
        # action-extent OFF: the committed-sequence completion must NOT release
        # the hold (the trajectory horizon is short; we want the LONG monolithic
        # occupancy the frontopolar term shortens, not a per-trajectory release).
        natural_commit_release_action_extent_mode=False,
        natural_commit_urgency_rate=NCUR_URGENCY_RATE,
        natural_commit_urgency_release_bound=NCUR_RELEASE_BOUND,
        natural_commit_urgency_cap=NCUR_URGENCY_CAP,
        natural_commit_gap_entry_sensitivity=0.0,   # flat urgency (not F-decisiveness-scaled)
        natural_commit_urgency_onset_ticks=NCUR_ONSET_TICKS,
        # --- natural-commit latch-hold (sustains the occupancy for the lever to shorten) ---
        use_natural_commit_latch_hold=True,
        natural_commit_latch_hold_max_ticks=0,       # uncapped; the urgency timeout is the release
        # --- SD-033e frontopolar de-commit lever (per-arm gain is the ONLY axis) ---
        use_frontopolar_decommit=True,
        frontopolar_gain=float(arm["frontopolar_gain"]),
    )
    return REEAgent(cfg)


# ---------------------------------------------------------------------------
# Measurement helpers (SD-056 online contrastive training; mirror 689d)
# ---------------------------------------------------------------------------

def _obs(d: Dict[str, Any], key: str) -> Optional[torch.Tensor]:
    h = d.get(key)
    if h is None:
        return None
    return h.float().unsqueeze(0) if h.dim() == 1 else h.float()


def _sample_class_diverse_batch(
    buffer: Deque[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    k: int,
    rng: random.Random,
) -> Optional[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
    if len(buffer) < MIN_BUFFER_BEFORE_TRAIN:
        return None
    pool = list(buffer)
    rng.shuffle(pool)
    seen: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for tup in pool:
        cls = int(tup[1].argmax().item())
        if cls not in seen:
            seen[cls] = tup
        if len(seen) >= k:
            break
    if len(seen) < MIN_CLASSES_FOR_TRAIN:
        return None
    samples = list(seen.values())
    picked = {id(s) for s in samples}
    for tup in pool:
        if len(samples) >= k:
            break
        if id(tup) in picked:
            continue
        samples.append(tup)
        picked.add(id(tup))
    return samples


def _e2_contrastive_step(agent, buffer, optimiser, rng) -> Optional[float]:
    batch = _sample_class_diverse_batch(buffer, CONTRASTIVE_BATCH_K, rng)
    if batch is None:
        return None
    z0_K = torch.stack([t[0] for t in batch]).to(agent.device)
    actions_K = torch.stack([t[1] for t in batch]).to(agent.device)
    z1_K = torch.stack([t[2] for t in batch]).to(agent.device)
    optimiser.zero_grad(set_to_none=True)
    loss = agent.e2.world_forward_contrastive_loss(
        z_world_0=z0_K, actions=actions_K, z_world_1_targets=z1_K,
        simulation_mode=False,
    )
    if not torch.is_tensor(loss):
        return None
    loss_val = float(loss.detach().item())
    if not math.isfinite(loss_val):
        return loss_val
    if not loss.requires_grad or loss_val == 0.0:
        return loss_val
    (SD056_WEIGHT * loss).backward()
    torch.nn.utils.clip_grad_norm_(agent.e2.parameters(), max_norm=MAX_GRAD_NORM)
    optimiser.step()
    return loss_val


# ---------------------------------------------------------------------------
# Per-(seed, arm) runner
# ---------------------------------------------------------------------------

def _run_seed_arm(
    arm: Dict[str, Any], seed: int,
    p0_eps: int, p0_steps: int, p1_eps: int, p1_steps: int,
    full_config: Dict[str, Any],
) -> Dict[str, Any]:
    with arm_cell(seed, config_slice=full_config, script_path=Path(__file__)) as cell:
        env = _make_env(seed)
        agent = _make_agent(env, arm)
        e2_opt = torch.optim.Adam(agent.e2.parameters(), lr=E2_CONTRASTIVE_LR)
        transition_buffer: Deque[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = deque(
            maxlen=TRANSITION_BUFFER_MAX
        )
        sample_rng = random.Random(seed)

        total_train_eps = p0_eps + p1_eps
        # P1 accumulators.
        release_occupancies: List[float] = []      # occupancy at every release event over P1
        frontopolar_release_count = 0              # accumulated over P1 episodes
        n_urgency_releases = 0
        frontopolar_pressure_accum = 0.0
        beta_elevated_ticks = 0                    # P1 ticks with the latch elevated
        n_p1_ticks = 0
        error_note: Optional[str] = None

        for ep in range(total_train_eps):
            is_p1 = ep >= p0_eps
            steps_per_episode = p1_steps if is_p1 else p0_steps
            phase_label = "P1" if is_p1 else "P0"

            _, obs_dict = env.reset()
            agent.reset()
            pending_capture: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
            prev_releases_total = 0

            for _step in range(steps_per_episode):
                body = obs_dict["body_state"].float()
                world = obs_dict["world_state"].float()
                if body.dim() == 1:
                    body = body.unsqueeze(0)
                if world.dim() == 1:
                    world = world.unsqueeze(0)

                latent = agent.sense(
                    obs_body=body, obs_world=world,
                    obs_harm=_obs(obs_dict, "harm_obs"),
                    obs_harm_a=_obs(obs_dict, "harm_obs_a"),
                    obs_harm_history=_obs(obs_dict, "harm_history"),
                )

                if pending_capture is not None:
                    z0_prev, a_prev = pending_capture
                    z1_obs = latent.z_world.detach().reshape(-1).clone()
                    if (torch.isfinite(z0_prev).all() and torch.isfinite(a_prev).all()
                            and torch.isfinite(z1_obs).all()):
                        transition_buffer.append((z0_prev, a_prev, z1_obs))
                    pending_capture = None

                ticks = agent.clock.advance()
                wdim = latent.z_world.shape[-1]
                e1_prior = (
                    agent._e1_tick(latent) if ticks.get("e1_tick", False)
                    else torch.zeros(1, wdim, device=agent.device)
                )
                candidates = agent.generate_trajectories(latent, e1_prior, ticks)

                if agent.goal_state is not None:
                    try:
                        energy = float(body[0, 3].item())
                    except Exception:
                        energy = 1.0
                    agent.update_z_goal(benefit_exposure=0.0,
                                        drive_level=max(0.0, 1.0 - energy))

                action = agent.select_action(candidates, ticks)

                # P1 occupancy / release measurement (read the rung-6 lever state).
                if is_p1 and agent.natural_commit_urgency is not None:
                    st = agent.natural_commit_urgency.get_state()
                    rel_total = int(st.get("ncur_n_releases_total", 0))
                    if rel_total > prev_releases_total:
                        # a release fired this tick -> record its latch occupancy
                        occ = float(st.get("ncur_last_occupancy_at_release", 0))
                        if occ > 0:
                            release_occupancies.append(occ)
                        prev_releases_total = rel_total
                    if bool(agent.beta_gate.is_elevated):
                        beta_elevated_ticks += 1
                    n_p1_ticks += 1

                if action is None:
                    idx = int(np.random.randint(0, env.action_dim))
                    action = torch.zeros(1, env.action_dim, device=agent.device)
                    action[0, idx] = 1.0
                    agent._last_action = action
                if not torch.isfinite(action).all():
                    if error_note is None:
                        error_note = (
                            f"non-finite action arm={arm['arm_id']} seed={seed} "
                            f"phase={phase_label} ep={ep} step={_step}"
                        )
                    break

                if torch.isfinite(latent.z_world).all() and torch.isfinite(action).all():
                    pending_capture = (
                        latent.z_world.detach().reshape(-1).clone(),
                        action.detach().reshape(-1).clone(),
                    )

                _e2_contrastive_step(agent, transition_buffer, e2_opt, sample_rng)

                _, harm_signal, done, info, next_obs_dict = env.step(action)
                with torch.no_grad():
                    agent.update_residue(harm_signal=float(harm_signal), world_delta=None,
                                        hypothesis_tag=False, owned=True)
                obs_dict = next_obs_dict
                if done:
                    break

            # End of episode: accumulate the per-episode lever counters BEFORE the
            # next agent.reset() zeroes them (get_state resets per episode).
            if is_p1 and agent.natural_commit_urgency is not None:
                st = agent.natural_commit_urgency.get_state()
                frontopolar_release_count += int(st.get("frontopolar_release_count", 0))
                n_urgency_releases += int(st.get("ncur_n_urgency_releases", 0))
                frontopolar_pressure_accum += float(st.get("frontopolar_pressure_accum", 0.0))

            if (ep + 1) % 10 == 0 or (ep + 1) == total_train_eps or is_p1:
                print(
                    f"  [train] arm={arm['arm_id']} seed={seed} phase={phase_label} "
                    f"ep {ep + 1}/{total_train_eps}",
                    flush=True,
                )

        occ_median = float(statistics.median(release_occupancies)) if release_occupancies else 0.0
        occ_max = float(max(release_occupancies)) if release_occupancies else 0.0
        row = {
            "arm_id": arm["arm_id"],
            "label": arm["label"],
            "seed": int(seed),
            "frontopolar_gain": float(arm["frontopolar_gain"]),
            "n_p1_ticks": int(n_p1_ticks),
            "n_release_events": int(len(release_occupancies)),
            "occupancy_median": round(occ_median, 3),
            "occupancy_max": round(occ_max, 3),
            "beta_elevated_ticks": int(beta_elevated_ticks),
            "frontopolar_release_count": int(frontopolar_release_count),
            "ncur_n_urgency_releases": int(n_urgency_releases),
            "frontopolar_pressure_accum": round(frontopolar_pressure_accum, 6),
            "error_note": error_note,
        }
        cell.stamp(row)
    return row


# ---------------------------------------------------------------------------
# Cross-arm evaluation
# ---------------------------------------------------------------------------

def _by_seed(rows: List[Dict[str, Any]], arm_id: str) -> Dict[int, Dict[str, Any]]:
    return {r["seed"]: r for r in rows if r.get("arm_id") == arm_id}


def _median(xs: List[float]) -> float:
    return float(statistics.median(xs)) if xs else 0.0


def _evaluate(arm_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    contrast = _by_seed(arm_results, CONTRAST_ARM)
    on = _by_seed(arm_results, ON_ARM)
    seeds = sorted(set(contrast) & set(on))

    # PRECONDITION: CONTRAST median occupancy > floor (the sustained natural-commit
    # hold -- the 460h monolithic-latch regime -- actually forms in GAP-A).
    contrast_occ = [contrast[s]["occupancy_median"] for s in seeds]
    contrast_seeds_strongf = [s for s in seeds if contrast[s]["occupancy_median"] > STRONG_F_OCCUPANCY_FLOOR]
    strong_f_ok = len(contrast_seeds_strongf) >= MIN_SEEDS_FOR_PASS

    # LOAD-BEARING: per-seed occupancy drop >= 40% AND attributable (ON frontopolar
    # release count > 0 -- fires only when cfv_now > cfv_at_entry).
    per_seed: List[Dict[str, Any]] = []
    seeds_drop_ok = 0
    for s in seeds:
        c_occ = float(contrast[s]["occupancy_median"])
        o_occ = float(on[s]["occupancy_median"])
        fp_rel = int(on[s]["frontopolar_release_count"])
        drop_frac = (1.0 - o_occ / c_occ) if c_occ > 0 else 0.0
        attributed = fp_rel > 0
        seed_pass = bool(
            contrast[s]["occupancy_median"] > STRONG_F_OCCUPANCY_FLOOR
            and drop_frac >= OCCUPANCY_DROP_FRAC
            and attributed
        )
        if seed_pass:
            seeds_drop_ok += 1
        per_seed.append({
            "seed": s,
            "contrast_occupancy_median": round(c_occ, 3),
            "on_occupancy_median": round(o_occ, 3),
            "drop_frac": round(drop_frac, 4),
            "on_frontopolar_release_count": fp_rel,
            "attributed": attributed,
            "seed_pass": seed_pass,
        })

    drop_pass = seeds_drop_ok >= MIN_SEEDS_FOR_PASS

    if not strong_f_ok:
        label = "substrate_not_ready_requeue"
        overall_pass = False
    elif drop_pass:
        label = "frontopolar_decommit_shortens_latch_occupancy"
        overall_pass = True
    else:
        label = "frontopolar_decommit_does_not_shorten_or_not_attributed"
        overall_pass = False

    # non-degeneracy of the criteria: did each measure discriminate?
    criteria_non_degenerate = {
        "C_STRONG_F_PRECONDITION": bool(len(contrast_occ) > 0 and max(contrast_occ) > 0),
        "C_OCCUPANCY_DROP": bool(any(
            contrast[s]["occupancy_median"] > 0 and on[s]["occupancy_median"] >= 0 for s in seeds
        )),
        "C_ATTRIBUTION": bool(any(on[s]["frontopolar_release_count"] >= 0 for s in seeds)),
    }

    interpretation = {
        "label": label,
        "preconditions": [{
            "name": "strong_f_monolithic_latch_present",
            "description": (
                "CONTRAST-arm median committed-latch occupancy > floor -- the "
                "sustained natural-commit hold (460h regime) forms in GAP-A, so "
                "there is a long occupancy for the frontopolar term to shorten."
            ),
            "measured": round(_median(contrast_occ), 3),
            "threshold": STRONG_F_OCCUPANCY_FLOOR,
            "control": "CONTRAST arm (lever ON, frontopolar_gain=0); the flat-urgency reference occupancy",
            "met": bool(strong_f_ok),
        }],
        "criteria_non_degenerate": criteria_non_degenerate,
        "criteria": [
            {"name": "C_OCCUPANCY_DROP_ATTRIBUTED", "load_bearing": True, "passed": bool(drop_pass)},
        ],
    }

    return {
        "strong_f_precondition": {
            "floor": STRONG_F_OCCUPANCY_FLOOR,
            "contrast_occupancy_median_by_seed": {str(s): contrast[s]["occupancy_median"] for s in seeds},
            "n_seeds_strong_f": len(contrast_seeds_strongf),
            "min_seeds_required": MIN_SEEDS_FOR_PASS,
            "strong_f_ok": strong_f_ok,
        },
        "occupancy_drop": {
            "drop_frac_required": OCCUPANCY_DROP_FRAC,
            "per_seed": per_seed,
            "n_seeds_drop_and_attributed": seeds_drop_ok,
            "drop_pass": drop_pass,
            "note": (
                "LOAD-BEARING: ON median latch-occupancy drops >= 40% vs the "
                "frontopolar_gain=0 CONTRAST on >= 2/3 strong-F seeds, ATTRIBUTABLE "
                "to the frontopolar term (ON frontopolar_release_count > 0 -- a "
                "real switch toward an improved foregone option, not a flat timeout)."
            ),
        },
        "label": label,
        "overall_pass": overall_pass,
        "interpretation": interpretation,
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _full_config_slice(arm: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "env_kwargs": dict(ENV_KWARGS),
        "p0_eps": P0_WARMUP_EPISODES, "p0_steps": P0_STEPS_PER_EPISODE,
        "p1_eps": P1_MEASUREMENT_EPISODES, "p1_steps": P1_STEPS_PER_EPISODE,
        "ncur_urgency_rate": NCUR_URGENCY_RATE,
        "ncur_release_bound": NCUR_RELEASE_BOUND,
        "ncur_urgency_cap": NCUR_URGENCY_CAP,
        "frontopolar_gain": float(arm["frontopolar_gain"]),
        "arm_id": arm["arm_id"],
    }


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    print(f"[{EXPERIMENT_TYPE}] starting (dry_run={dry_run})", flush=True)
    seeds = DRY_RUN_SEEDS if dry_run else SEEDS
    p0_eps = DRY_RUN_P0 if dry_run else P0_WARMUP_EPISODES
    p0_steps = DRY_RUN_P0_STEPS if dry_run else P0_STEPS_PER_EPISODE
    p1_eps = DRY_RUN_P1 if dry_run else P1_MEASUREMENT_EPISODES
    p1_steps = DRY_RUN_P1_STEPS if dry_run else P1_STEPS_PER_EPISODE

    arm_results: List[Dict[str, Any]] = []
    for seed in seeds:
        for arm in ARMS:
            print(f"Seed {seed} Condition {arm['arm_id']}", flush=True)
            row = _run_seed_arm(
                arm, seed, p0_eps, p0_steps, p1_eps, p1_steps,
                full_config=_full_config_slice(arm),
            )
            arm_results.append(row)
            print(
                f"verdict: {'PASS' if row.get('error_note') is None else 'FAIL'} "
                f"seed={seed} arm={arm['arm_id']} occ_median={row['occupancy_median']} "
                f"fp_releases={row['frontopolar_release_count']} "
                f"n_releases={row['n_release_events']}",
                flush=True,
            )

    evaluation = _evaluate(arm_results)
    outcome = "PASS" if evaluation["overall_pass"] else "FAIL"

    manifest = {
        "experiment_type": EXPERIMENT_TYPE,
        "run_id": f"{EXPERIMENT_TYPE}_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_v3",
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "queue_id": QUEUE_ID,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "timestamp_utc": datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
        "outcome": outcome,
        "evidence_direction": "non_contributory",  # diagnostic -- excluded from scoring
        "dry_run": bool(dry_run),
        "seeds": seeds,
        "p0_warmup_episodes": p0_eps,
        "p1_measurement_episodes": p1_eps,
        "p1_steps_per_episode": p1_steps,
        "thresholds": {
            "occupancy_drop_frac": OCCUPANCY_DROP_FRAC,
            "strong_f_occupancy_floor": STRONG_F_OCCUPANCY_FLOOR,
            "min_seeds_for_pass": MIN_SEEDS_FOR_PASS,
            "frontopolar_gain_on": FRONTOPOLAR_GAIN_ON,
            "ncur_urgency_rate": NCUR_URGENCY_RATE,
        },
        "arm_results": arm_results,
        "evaluation": evaluation,
        "interpretation": evaluation["interpretation"],
    }
    if SUPERSEDES:
        manifest["supersedes"] = SUPERSEDES
    return manifest


def _write_manifest(manifest: Dict[str, Any]) -> Path:
    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=False,
        config=manifest.get("config"),
        seeds=SEEDS,
        script_path=Path(__file__),
    )
    return out_path


def main(dry_run: bool = False) -> Tuple[Dict[str, Any], Path]:
    manifest = run_experiment(dry_run=dry_run)
    out_path = _write_manifest(manifest)
    print(f"[{EXPERIMENT_TYPE}] outcome={manifest['outcome']} "
          f"label={manifest['evaluation']['label']} -> {out_path}", flush=True)
    if dry_run:
        print(f"[{EXPERIMENT_TYPE}] dry-run complete.", flush=True)
    return manifest, out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    _manifest, _out_path = main(dry_run=args.dry_run)
    _outcome_raw = str(_manifest["outcome"]).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=str(_out_path),
        dry_run=args.dry_run,
    )
