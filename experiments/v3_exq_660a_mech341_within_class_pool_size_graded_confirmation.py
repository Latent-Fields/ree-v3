#!/opt/local/bin/python3
"""
V3-EXQ-660a -- MECH-341 within-class-representative-diversity GRADED CONFIRMATION
via a CEM candidate-POOL-SIZE dose-response (successor to V3-EXQ-660).

WHY THIS RUN (the design judgment)
----------------------------------
V3-EXQ-660 LANDED PASS/supports: the within-class-representative lever is
load-bearing (legacy argmin within_class_rep_cond_entropy = 4.781414 vs any
positive within-class temperature = 4.861916, a ~0.08-nat lift). BUT the 660
temperature sweep proved INERT: arms T=0.5 / 1.0 / 2.0 produced BYTE-IDENTICAL
values on BOTH the primary readout (4.861916) AND the secondary committed-class
control (1.033608) despite different sample counts. Temperature has literally
zero leverage beyond the legacy-vs-any-sampling switch.

MECHANISM of the temperature insensitivity (the reason this run does NOT sweep
temperature again): within each committed first-action class the per-member
scores are near-DEGENERATE, so softmax(-class_scores / T) collapses to ~uniform
at EVERY positive T. Under uniform within-class sampling the selected
representative is driven entirely by the (fixed-per-seed) RNG draw, not by T --
hence byte-identical across T. Under uniform sampling,
    within_class_rep_cond_entropy ~= log(number of DISTINCT within-class
                                         representatives available),
so the diversity is bounded by the COUNT of distinct within-class
representatives in the candidate pool, NOT by the temperature knob. The viable
GRADED axis is therefore the CEM candidate-POOL SIZE (HippocampalConfig.
num_candidates), which controls how many within-class members exist for the
lever to spread across. A temperature re-sweep would wash out again; pool size
will not (entropy of a uniform distribution over N is log N, monotone in N).

This run converts 660's BINARY legacy-vs-any lift into a DOSE-RESPONSE and tests
whether the ~0.08-nat lift is LOAD-BEARING (grows as more within-class
representatives become available) versus a fixed noise-floor artifact (flat
regardless of available diversity).

GRADED AXIS
-----------
cfg.hippocampal.num_candidates = K, swept over K in {16, 32, 64, 128}
(K=32 reproduces the 660 configuration). The proposer caps its returned
candidate pool at n = num_candidates (ree_core/hippocampal/module.py:1190/1216),
and REEConfig.from_dims does NOT assign hippocampal.num_candidates, so setting it
AFTER from_dims and BEFORE REEAgent(cfg) is honored. elite_fraction stays at its
default 0.2.

EVERYTHING ELSE IS IDENTICAL TO V3-EXQ-660's _make_agent: GAP-A
candidate_summary_source = "e2_world_forward" (V3-EXQ-649 PASS); modulatory-bias
selection authority ON (gain 0.5; V3-EXQ-643a fix); SP-CEM Layer A; MECH-341 both
sub-flavours (entropy bonus + stratified select), bias_scale 2.0; MECH-313 noise
floor; V_s minimal stack; SD-056 amend levers all ON. The within-class temperature
is FIXED (not swept): None for the LEGACY control, 1.0 for the SAMPLED lever
(T-value is immaterial -- 660 proved every positive T is identical).

ARMS (8 = 2 lever states x 4 pool sizes), 3 seeds [42,43,44]
-----------------------------------------------------------
  For each K in {16, 32, 64, 128}:
    ARM_LEGACY_K{K}:  num_candidates=K, within_class_temperature=None   (argmin)
    ARM_SAMPLED_K{K}: num_candidates=K, within_class_temperature=1.0    (sampled)
P0 (30 ep warmup, instrumentation OFF) + P1 (60 ep, instrumentation ON), 200
steps/ep -- IDENTICAL phase budget + ENV_KWARGS to V3-EXQ-660 for comparability.

PRIMARY READOUT (per seed, per arm): within_class_rep_cond_entropy_nats =
H(rep_signature | committed_class) (identical to 660). Secondary / control:
committed_class_entropy_nats. AVAILABILITY statistic (the non-vacuity gate):
mean distinct within-class rep signatures available per committed class
(mean over P1 ticks of |{rep_signature(t) : t in pool, first_action_class(t) ==
committed_class}|) -- the SAME within-class-availability the C_GRADED lift
converts into selected diversity. Realized candidate-pool size is recorded per
arm to confirm K is honored.

PRE-REGISTERED ACCEPTANCE (paired by seed index)
------------------------------------------------
  Delta(K)[seed] = within_class_rep_cond_entropy[SAMPLED,K,seed]
                 - within_class_rep_cond_entropy[LEGACY,K,seed]

  NON-VACUITY PRECONDITION (load-bearing readiness, SAME statistic as the
    criterion routes on): mean available distinct within-class representatives
    per committed class RISES across K in the SAMPLED arms -- nondecreasing
    across consecutive K AND avail(K_max) - avail(K_min) >= AVAIL_RISE_FLOOR.
    UNMET -> the graded axis did not move the substrate's within-class-diversity
    availability -> self-route substrate_not_ready_requeue -> FAIL/non_contributory
    (NOT a weakens; protects MECH-341 from a false negative at a scale where pool
    size has no leverage).

  C_GRADED (PRIMARY / load-bearing): for >= MIN_SEEDS_FOR_GRADED of 3 seeds, the
    per-seed Delta(K) is monotone-nondecreasing across K (within MONO_TOL) AND
    Delta(K_max) - Delta(K_min) >= C2_LIFT_MARGIN_NATS (0.05 nats, matching the
    660 lift-margin scale). I.e. a positive dose-response: the lift GROWS with
    available within-class diversity.

  C_ABS (supporting, reported, NOT gating): absolute SAMPLED
    within_class_rep_cond_entropy mean is monotone-nondecreasing across K.

  OUTCOME MAP:
    non-vacuity UNMET                        -> FAIL / non_contributory
                                                (substrate_not_ready_requeue)
    non-vacuity MET and C_GRADED holds       -> PASS / supports
                                                (within-class sub-axis confirmed
                                                load-bearing + GRADED; v3_pending
                                                clearance candidate pending
                                                governance ratification; do NOT
                                                auto-flip the within-class default)
    non-vacuity MET but C_GRADED fails       -> FAIL / weakens (the lift is a
                                                fixed structural artifact
                                                independent of available
                                                within-class diversity -- not
                                                load-bearing / not graded)

Claims: [MECH-341]. experiment_purpose = evidence. NO supersedes -- V3-EXQ-660
stays STANDING evidence (this asks the distinct graded-axis question, not a
corrected re-ask; 660's temperature design was not buggy -- temperature is
genuinely inert).

See REE_assembly/docs/architecture/mech_341_e3_score_diversity_preservation.md,
REE_assembly/docs/architecture/modulatory_bias_selection_authority.md,
REE_assembly/evidence/planning/behavioral_diversity_isolation_plan.md GAP-B,
ree-v3/experiments/v3_exq_660_mech341_within_class_representative_diversity.py.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from experiment_protocol import emit_outcome
from experiments._lib.arm_fingerprint import arm_cell
from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig
from experiments.pack_writer import write_flat_manifest  # noqa: E402


EXPERIMENT_TYPE = "v3_exq_660a_mech341_within_class_pool_size_graded_confirmation"
QUEUE_ID = "V3-EXQ-660a"
SUPERSEDES = None  # V3-EXQ-660 stays standing evidence (distinct graded-axis question)
CLAIM_IDS: List[str] = ["MECH-341"]
EXPERIMENT_PURPOSE = "evidence"

# V3-EXQ-660a graded axis: CEM candidate-pool size. K=32 reproduces 660.
POOL_SIZES: List[int] = [16, 32, 64, 128]

# Fixed within-class temperature for the SAMPLED lever state (None = legacy
# argmin control). 660 proved every positive T is byte-identical, so the value
# is immaterial; 1.0 is the mid-T the 660 lineage used.
SAMPLED_WITHIN_CLASS_T: float = 1.0

# GAP-A shared-channel re-sourcing (V3-EXQ-649 PASS). ON for ALL arms.
CANDIDATE_SUMMARY_SOURCE = "e2_world_forward"

# modulatory-bias-selection-authority (V3-EXQ-643a-fixed substrate). ON for ALL
# arms. gain=0.5 keeps modulatory signals competitive in near-ties but
# subdominant when the primary harm/goal gap exceeds gain * raw_score_range.
USE_MODULATORY_SELECTION_AUTHORITY = True
MODULATORY_AUTHORITY_GAIN = 0.5

# C_GRADED PRIMARY threshold: per-seed paired-lift growth across the pool-size
# sweep. Margin matches the 660 lift-margin scale (0.05 nats).
C2_LIFT_MARGIN_NATS = 0.05
MIN_SEEDS_FOR_GRADED = 2   # >= 2/3 seeds must show the graded dose-response
MONO_TOL = 1e-6            # tolerance for "monotone-nondecreasing"

# Non-vacuity: mean distinct within-class representatives available per committed
# class must rise by at least this many (distinct rep signatures) from K_min to
# K_max in the SAMPLED arms. Sub-floor -> substrate_not_ready_requeue.
AVAIL_RISE_FLOOR = 0.25

# C1-style within-class branch firing floor (per-seed) and multi-rep availability
# floor (per-seed) -- retained from 660 as basic operativeness diagnostics.
WITHIN_CLASS_FIRE_FLOOR = 10
MULTI_REP_TICK_FLOOR = 5

# MECH-341 sub-flavour scale used in the entropy-ON arms (matches 660 / 614e).
MECH341_ENTROPY_BIAS_SCALE = 2.0

# SD-056 amend lever defaults applied uniformly across all arms (match 660).
SD056_MULTISTEP_CONTRASTIVE = True
SD056_CONTRASTIVE_HORIZON = 5
SD056_OUTPUT_NORM_CLAMP = True
SD056_OUTPUT_NORM_CLAMP_RATIO = 2.0
SD056_T1_CONTRASTIVE_ENABLED = True
SD056_T1_CONTRASTIVE_WEIGHT = 0.01

SEEDS = [42, 43, 44]
P0_WARMUP_EPISODES = 30
P1_MEASUREMENT_EPISODES = 60
STEPS_PER_EPISODE = 200

DRY_RUN_SEEDS = [42]
DRY_RUN_P0 = 2
DRY_RUN_P1 = 2
DRY_RUN_STEPS = 30

MIN_SEEDS_PER_ARM_FOR_PASS = 2  # of 3 (used for C3 / operativeness majority checks)

# V_s (D) thresholds (minimal stack) -- match 660.
VS_SNAPSHOT_REFRESH_THRESHOLD = 0.5
VS_E1_THRESHOLD = 0.4


# IDENTICAL to V3-EXQ-660 / 614c / 614d / 614e for direct manifest comparability.
ENV_KWARGS = dict(
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


def _build_arms() -> List[Dict[str, Any]]:
    arms: List[Dict[str, Any]] = []
    for k in POOL_SIZES:
        arms.append({
            "arm_id": f"ARM_LEGACY_K{k}",
            "label": f"legacy_argmin_pool{k}",
            "num_candidates": int(k),
            "within_class_temperature": None,
            "lever": "legacy",
        })
        arms.append({
            "arm_id": f"ARM_SAMPLED_K{k}",
            "label": f"within_class_sampled_pool{k}",
            "num_candidates": int(k),
            "within_class_temperature": SAMPLED_WITHIN_CLASS_T,
            "lever": "sampled",
        })
    return arms


ARMS: List[Dict[str, Any]] = _build_arms()


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(seed=seed, **ENV_KWARGS)


def _make_agent(
    within_class_temperature: Optional[float],
    num_candidates: int,
    env: CausalGridWorldV2,
) -> REEAgent:
    """Build the SD-056-amended 4-substrate baseline + modulatory authority +
    GAP-A shared-channel candidate_summary_source -- IDENTICAL to V3-EXQ-660
    _make_agent EXCEPT the CEM candidate-pool size is set per-arm.

    cfg.hippocampal.num_candidates is set AFTER REEConfig.from_dims (which does
    NOT assign it) and BEFORE REEAgent(cfg), so the per-arm K is honored.
    """
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
        # A (SP-CEM)
        use_support_preserving_cem=True,
        support_preserving_stratified_elites=True,
        support_preserving_ao_std_floor=0.2,
        support_preserving_min_first_action_classes=2,
        # B (MECH-341)
        use_e3_score_diversity=True,
        use_e3_diversity_entropy_bonus=True,
        use_e3_diversity_stratified_select=True,
        e3_diversity_entropy_bias_scale=MECH341_ENTROPY_BIAS_SCALE,
        e3_diversity_stratified_within_class_temperature=within_class_temperature,
        # GAP-A shared-channel candidate-summary re-sourcing (V3-EXQ-649 PASS).
        candidate_summary_source=CANDIDATE_SUMMARY_SOURCE,
        # modulatory-bias-selection-authority (V3-EXQ-643a-fixed). ON for all arms.
        use_modulatory_selection_authority=USE_MODULATORY_SELECTION_AUTHORITY,
        modulatory_authority_gain=MODULATORY_AUTHORITY_GAIN,
        # C (MECH-313)
        use_noise_floor=True,
        noise_floor_alpha=0.1,
        # D (V_s minimal)
        use_per_stream_vs=True,
        use_vs_rollout_gating=True,
        vs_gate_snapshot_refresh_threshold=VS_SNAPSHOT_REFRESH_THRESHOLD,
        vs_gate_e1_threshold=VS_E1_THRESHOLD,
        # SD-056 amend (uniform across ALL arms; not an isolation axis).
        e2_action_contrastive_enabled=SD056_T1_CONTRASTIVE_ENABLED,
        e2_action_contrastive_weight=SD056_T1_CONTRASTIVE_WEIGHT,
        e2_action_contrastive_multistep_enabled=SD056_MULTISTEP_CONTRASTIVE,
        e2_action_contrastive_horizon=SD056_CONTRASTIVE_HORIZON,
        e2_rollout_output_norm_clamp_enabled=SD056_OUTPUT_NORM_CLAMP,
        e2_rollout_output_norm_clamp_ratio=SD056_OUTPUT_NORM_CLAMP_RATIO,
    )
    # GRADED AXIS: CEM candidate-pool size (HippocampalConfig.num_candidates).
    # from_dims does not assign this field, so the post-build set is honored.
    cfg.hippocampal.num_candidates = int(num_candidates)
    return REEAgent(cfg)


# ---------------------------------------------------------------------------
# Per-tick measurement helpers (identical to V3-EXQ-660)
# ---------------------------------------------------------------------------


def _first_action_class(traj) -> int:
    return int(
        traj.actions[:, 0, :].argmax(dim=-1).detach().reshape(-1)[0].item()
    )


def _rep_signature(traj) -> Tuple[int, ...]:
    """Within-class representative discriminator: argmax of each action step
    AFTER the first (members of one first-action class share step 0)."""
    actions = traj.actions
    if actions.dim() < 3 or actions.shape[1] < 2:
        return ()
    horizon = int(actions.shape[1])
    sig: List[int] = []
    for t in range(1, horizon):
        sig.append(int(actions[:, t, :].argmax(dim=-1).detach().reshape(-1)[0].item()))
    return tuple(sig)


def _obs_harm(obs_dict):
    h = obs_dict.get("harm_obs")
    return h.float().unsqueeze(0) if h is not None else None


def _obs_harm_a(obs_dict):
    h = obs_dict.get("harm_obs_a")
    return h.float().unsqueeze(0) if h is not None else None


def _obs_harm_history(obs_dict):
    h = obs_dict.get("harm_history")
    return h.float().unsqueeze(0) if h is not None else None


def _entropy_from_counts(counts: Dict[Any, int]) -> float:
    n = sum(counts.values())
    if n <= 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        if c <= 0:
            continue
        p = c / n
        h -= p * math.log(p)
    return float(h)


def _conditional_entropy(pairs: List[Tuple[int, Tuple[int, ...]]]) -> float:
    """H(rep_signature | committed_class), weighted by committed-class freq."""
    if not pairs:
        return 0.0
    by_class: Dict[int, Dict[Tuple[int, ...], int]] = {}
    for cls, sig in pairs:
        by_class.setdefault(cls, {})
        by_class[cls][sig] = by_class[cls].get(sig, 0) + 1
    total = len(pairs)
    h_cond = 0.0
    for cls, sig_counts in by_class.items():
        n_c = sum(sig_counts.values())
        h_cond += (n_c / total) * _entropy_from_counts(sig_counts)
    return float(h_cond)


def _run_seed_arm(
    arm: Dict[str, Any],
    seed: int,
    p0_episodes: int,
    p1_episodes: int,
    steps_per_episode: int,
) -> Dict[str, Any]:
    within_class_temperature = arm.get("within_class_temperature")
    num_candidates = int(arm["num_candidates"])

    config_slice = {
        "arm_id": arm["arm_id"],
        "within_class_temperature": within_class_temperature,
        "num_candidates": num_candidates,
        "candidate_summary_source": CANDIDATE_SUMMARY_SOURCE,
        "use_modulatory_selection_authority": USE_MODULATORY_SELECTION_AUTHORITY,
        "modulatory_authority_gain": MODULATORY_AUTHORITY_GAIN,
        "mech341_entropy_bias_scale": MECH341_ENTROPY_BIAS_SCALE,
        "p0_episodes": int(p0_episodes),
        "p1_episodes": int(p1_episodes),
        "steps_per_episode": int(steps_per_episode),
        "env_kwargs": dict(ENV_KWARGS),
    }

    with arm_cell(
        seed,
        config_slice=config_slice,
        script_path=Path(__file__),
    ) as cell:
        env = _make_env(seed)
        agent = _make_agent(within_class_temperature, num_candidates, env)
        agent.eval()

        total_train_eps = p0_episodes + p1_episodes
        error_note: Optional[str] = None
        n_p0_ticks = 0
        n_p1_ticks = 0
        n_p1_pre_ge2 = 0
        n_p1_pre_eq1 = 0
        committed_classes_p1: Dict[int, int] = {}

        selected_pairs: List[Tuple[int, Tuple[int, ...]]] = []
        n_multi_rep_ticks = 0

        # Availability accumulators: sum of distinct within-class rep signatures
        # in the committed class per P1 tick, and the realized candidate-pool
        # size per P1 tick (to confirm K is honored).
        sum_distinct_within_class_reps = 0
        n_avail_ticks = 0
        sum_pool_size = 0
        max_pool_size = 0
        n_pool_obs = 0

        n_within_class_sampled_total = 0
        n_stratified_fired_total = 0
        n_authority_normalized_total = 0

        for ep in range(total_train_eps):
            is_p1 = ep >= p0_episodes
            phase_label = "P1" if is_p1 else "P0"

            _, obs_dict = env.reset()
            agent.reset()

            z_self_prev: Optional[torch.Tensor] = None
            action_prev: Optional[torch.Tensor] = None

            for _step in range(steps_per_episode):
                body = obs_dict["body_state"].float()
                world = obs_dict["world_state"].float()
                if body.dim() == 1:
                    body = body.unsqueeze(0)
                if world.dim() == 1:
                    world = world.unsqueeze(0)

                latent = agent.sense(
                    obs_body=body, obs_world=world,
                    obs_harm=_obs_harm(obs_dict),
                    obs_harm_a=_obs_harm_a(obs_dict),
                    obs_harm_history=_obs_harm_history(obs_dict),
                )

                if z_self_prev is not None and action_prev is not None:
                    agent.record_transition(
                        z_self_prev, action_prev, latent.z_self.detach()
                    )

                ticks = agent.clock.advance()
                wdim = latent.z_world.shape[-1]
                e1_prior = (
                    agent._e1_tick(latent) if ticks.get("e1_tick", False)
                    else torch.zeros(1, wdim, device=agent.device)
                )
                candidates = agent.generate_trajectories(latent, e1_prior, ticks)

                pre_e3_classes: List[int] = []
                if is_p1 and candidates:
                    pre_e3_classes = sorted({
                        _first_action_class(t) for t in candidates
                    })
                    # realized candidate-pool size (confirms K is honored).
                    sum_pool_size += len(candidates)
                    max_pool_size = max(max_pool_size, len(candidates))
                    n_pool_obs += 1

                action = agent.select_action(candidates, ticks)
                if not torch.isfinite(action).all():
                    if error_note is None:
                        error_note = (
                            f"non-finite action at arm={arm['arm_id']} seed={seed} "
                            f"phase={phase_label} ep={ep} step={_step}"
                        )
                    break

                if is_p1:
                    committed_class = int(action[0].argmax().item())
                    committed_classes_p1[committed_class] = (
                        committed_classes_p1.get(committed_class, 0) + 1
                    )

                    sel_traj = getattr(agent.e3, "_last_selected_trajectory", None)
                    if sel_traj is not None:
                        sel_cls = _first_action_class(sel_traj)
                        sel_sig = _rep_signature(sel_traj)
                        selected_pairs.append((sel_cls, sel_sig))

                    if candidates:
                        pool_sigs_in_class = {
                            _rep_signature(t)
                            for t in candidates
                            if _first_action_class(t) == committed_class
                        }
                        # AVAILABILITY statistic: distinct within-class reps
                        # available in the committed class this tick.
                        sum_distinct_within_class_reps += len(pool_sigs_in_class)
                        n_avail_ticks += 1
                        if len(pool_sigs_in_class) >= 2:
                            n_multi_rep_ticks += 1

                    pre_count = len(pre_e3_classes)
                    if pre_count >= 2:
                        n_p1_pre_ge2 += 1
                    elif pre_count == 1:
                        n_p1_pre_eq1 += 1

                    n_p1_ticks += 1
                else:
                    n_p0_ticks += 1

                _, _harm_signal, done, info, obs_dict = env.step(action)

                if agent.goal_state is not None:
                    benefit_exposure = float(info.get("benefit_exposure", 0.0))
                    energy = float(body[0, 3].item())
                    drive_level = max(0.0, 1.0 - energy)
                    agent.update_z_goal(
                        benefit_exposure=benefit_exposure,
                        drive_level=drive_level,
                    )

                z_self_prev = latent.z_self.detach()
                action_prev = action.detach()

                if done:
                    break

            if is_p1 and getattr(agent, "score_diversity", None) is not None:
                sd_state = agent.score_diversity.get_state()
                n_within_class_sampled_total += int(
                    sd_state.get("mech341_n_within_class_sampled", 0)
                )
                n_stratified_fired_total += int(
                    sd_state.get("mech341_n_stratified_fired", 0)
                )
                n_authority_normalized_total += int(
                    sd_state.get("mech341_n_authority_normalized", 0)
                )

            if (ep + 1) % 10 == 0 or (ep + 1) == total_train_eps:
                print(
                    f"  [train] arm={arm['arm_id']} seed={seed} phase={phase_label} "
                    f"ep {ep + 1}/{total_train_eps}",
                    flush=True,
                )

            if error_note is not None:
                break

        if n_p1_ticks > 0:
            frac_pre_ge2 = float(n_p1_pre_ge2 / n_p1_ticks)
        else:
            frac_pre_ge2 = 0.0

        committed_class_entropy = _entropy_from_counts(committed_classes_p1)
        within_class_rep_cond_entropy = _conditional_entropy(selected_pairs)
        mean_distinct_within_class_reps = (
            float(sum_distinct_within_class_reps / n_avail_ticks)
            if n_avail_ticks > 0 else 0.0
        )
        realized_pool_size_mean = (
            float(sum_pool_size / n_pool_obs) if n_pool_obs > 0 else 0.0
        )

        seed_substrate_ready = bool(frac_pre_ge2 > 0.3)
        within_class_branch_active = bool(
            n_within_class_sampled_total >= WITHIN_CLASS_FIRE_FLOOR
        )
        multi_rep_available = bool(n_multi_rep_ticks >= MULTI_REP_TICK_FLOOR)

        row = {
            "arm_id": arm["arm_id"],
            "lever": arm["lever"],
            "num_candidates": int(num_candidates),
            "seed": int(seed),
            "within_class_temperature": within_class_temperature,
            "p0_episodes_run": int(min(p0_episodes, total_train_eps)),
            "p1_episodes_run": int(max(0, total_train_eps - p0_episodes)),
            "n_p0_ticks": int(n_p0_ticks),
            "n_p1_ticks": int(n_p1_ticks),
            "n_p1_pre_ge2": int(n_p1_pre_ge2),
            "n_p1_pre_eq1": int(n_p1_pre_eq1),
            "frac_pre_ge2": round(frac_pre_ge2, 6),
            "committed_classes_p1_counts": {
                str(k): int(v) for k, v in sorted(committed_classes_p1.items())
            },
            "n_unique_committed_classes": int(len(committed_classes_p1)),
            # PRIMARY readout:
            "within_class_rep_cond_entropy_nats": round(
                within_class_rep_cond_entropy, 6
            ),
            "n_selected_pairs": int(len(selected_pairs)),
            "n_multi_rep_ticks": int(n_multi_rep_ticks),
            # AVAILABILITY statistic (non-vacuity gate):
            "mean_distinct_within_class_reps": round(
                mean_distinct_within_class_reps, 6
            ),
            "n_avail_ticks": int(n_avail_ticks),
            # realized pool size (confirms K is honored):
            "realized_pool_size_mean": round(realized_pool_size_mean, 4),
            "realized_pool_size_max": int(max_pool_size),
            # SECONDARY / negative-control readout:
            "committed_class_entropy_nats": round(committed_class_entropy, 6),
            # MECH-341 within-class firing diagnostics:
            "mech341_n_within_class_sampled": int(n_within_class_sampled_total),
            "mech341_n_stratified_fired": int(n_stratified_fired_total),
            "mech341_n_authority_normalized": int(n_authority_normalized_total),
            # Per-seed flags:
            "within_class_branch_active": within_class_branch_active,
            "multi_rep_available": multi_rep_available,
            "seed_substrate_ready": seed_substrate_ready,
            "error_note": error_note,
        }
        cell.stamp(row)

    return row


def _interpret_arm(seed_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    completed = [r for r in seed_rows if r["error_note"] is None]
    n_seeds_completed = len(completed)

    rep_entropies = [r["within_class_rep_cond_entropy_nats"] for r in completed]
    mean_rep_entropy = (
        sum(rep_entropies) / len(rep_entropies) if rep_entropies else 0.0
    )
    avail = [r["mean_distinct_within_class_reps"] for r in completed]
    mean_avail = sum(avail) / len(avail) if avail else 0.0
    pool_means = [r["realized_pool_size_mean"] for r in completed]
    mean_realized_pool = sum(pool_means) / len(pool_means) if pool_means else 0.0

    n_substrate_ready = sum(1 for r in completed if r["seed_substrate_ready"])
    n_within_class_active = sum(
        1 for r in completed if r["within_class_branch_active"]
    )
    n_multi_rep = sum(1 for r in completed if r["multi_rep_available"])

    return {
        "n_seeds_completed": int(n_seeds_completed),
        "mean_within_class_rep_cond_entropy_nats": round(mean_rep_entropy, 6),
        "mean_distinct_within_class_reps": round(mean_avail, 6),
        "mean_realized_pool_size": round(mean_realized_pool, 4),
        "n_seeds_substrate_ready": int(n_substrate_ready),
        "majority_substrate_ready": bool(
            n_substrate_ready >= MIN_SEEDS_PER_ARM_FOR_PASS
        ),
        "n_seeds_within_class_active": int(n_within_class_active),
        "n_seeds_multi_rep_available": int(n_multi_rep),
    }


def _is_monotone_nondecreasing(values: List[float], tol: float = MONO_TOL) -> bool:
    return all(values[i + 1] >= values[i] - tol for i in range(len(values) - 1))


def run_experiment(
    seeds: List[int],
    p0_episodes: int,
    p1_episodes: int,
    steps_per_episode: int,
    dry_run: bool,
) -> Dict[str, Any]:
    arms_out: List[Dict[str, Any]] = []
    for arm in ARMS:
        print(
            f"Arm {arm['arm_id']} ({arm['label']}) "
            f"(K={arm['num_candidates']}, within_T={arm['within_class_temperature']}, "
            f"P0={p0_episodes} ep, P1={p1_episodes} ep, "
            f"steps={steps_per_episode}, "
            f"candidate_summary_source={CANDIDATE_SUMMARY_SOURCE}, "
            f"authority=ON gain={MODULATORY_AUTHORITY_GAIN}, dry_run={dry_run})",
            flush=True,
        )
        seed_rows: List[Dict[str, Any]] = []
        for s in seeds:
            print(f"Seed {s} Condition {arm['label']}", flush=True)
            row = _run_seed_arm(arm, s, p0_episodes, p1_episodes, steps_per_episode)
            seed_rows.append(row)
            verdict = "PASS" if row["error_note"] is None else "FAIL"
            print(
                f"  arm={arm['arm_id']} seed={s} K={arm['num_candidates']} "
                f"realized_pool_mean={row['realized_pool_size_mean']} "
                f"realized_pool_max={row['realized_pool_size_max']} "
                f"mean_distinct_reps={row['mean_distinct_within_class_reps']} "
                f"within_class_rep_cond_entropy={row['within_class_rep_cond_entropy_nats']}",
                flush=True,
            )
            print(f"verdict: {verdict}", flush=True)
        cross = _interpret_arm(seed_rows)
        arms_out.append({
            "arm_id": arm["arm_id"],
            "label": arm["label"],
            "lever": arm["lever"],
            "num_candidates": int(arm["num_candidates"]),
            "within_class_temperature": arm.get("within_class_temperature"),
            "per_seed_results": seed_rows,
            "cross_seed_interpretation": cross,
        })

    by_id = {a["arm_id"]: a for a in arms_out}

    def _entropy_by_seed(arm) -> Dict[int, float]:
        return {
            int(r["seed"]): r["within_class_rep_cond_entropy_nats"]
            for r in arm["per_seed_results"]
            if r["error_note"] is None
        }

    # ---- Realized-pool-size honored check (K is actually applied) ----
    realized_pool_by_K = {
        k: by_id[f"ARM_SAMPLED_K{k}"]["cross_seed_interpretation"][
            "mean_realized_pool_size"
        ]
        for k in POOL_SIZES
    }
    pool_size_honored = _is_monotone_nondecreasing(
        [realized_pool_by_K[k] for k in POOL_SIZES]
    ) and (
        realized_pool_by_K[POOL_SIZES[-1]] > realized_pool_by_K[POOL_SIZES[0]]
    )

    # ---- NON-VACUITY: available distinct within-class reps rise across K ----
    avail_by_K_sampled = {
        k: by_id[f"ARM_SAMPLED_K{k}"]["cross_seed_interpretation"][
            "mean_distinct_within_class_reps"
        ]
        for k in POOL_SIZES
    }
    avail_series = [avail_by_K_sampled[k] for k in POOL_SIZES]
    avail_rise = float(avail_series[-1] - avail_series[0])
    non_vacuity_met = bool(
        _is_monotone_nondecreasing(avail_series)
        and avail_rise >= AVAIL_RISE_FLOOR
    )

    # ---- C_GRADED: per-seed paired-lift dose-response across K ----
    legacy_ent_by_K = {k: _entropy_by_seed(by_id[f"ARM_LEGACY_K{k}"]) for k in POOL_SIZES}
    sampled_ent_by_K = {k: _entropy_by_seed(by_id[f"ARM_SAMPLED_K{k}"]) for k in POOL_SIZES}

    delta_by_K_by_seed: Dict[int, Dict[int, float]] = {}  # seed -> {K: delta}
    for k in POOL_SIZES:
        leg = legacy_ent_by_K[k]
        sam = sampled_ent_by_K[k]
        for s in seeds:
            if s in leg and s in sam:
                delta_by_K_by_seed.setdefault(s, {})[k] = float(sam[s] - leg[s])

    per_seed_graded: Dict[int, Dict[str, Any]] = {}
    n_seeds_graded = 0
    for s in seeds:
        dmap = delta_by_K_by_seed.get(s, {})
        if len(dmap) != len(POOL_SIZES):
            per_seed_graded[s] = {
                "complete": False, "monotone": False, "margin": None, "graded": False,
            }
            continue
        deltas = [dmap[k] for k in POOL_SIZES]
        monotone = _is_monotone_nondecreasing(deltas)
        margin = float(deltas[-1] - deltas[0])
        graded = bool(monotone and margin >= C2_LIFT_MARGIN_NATS)
        per_seed_graded[s] = {
            "complete": True,
            "deltas_by_K": {str(k): round(dmap[k], 6) for k in POOL_SIZES},
            "monotone": bool(monotone),
            "margin": round(margin, 6),
            "graded": graded,
        }
        if graded:
            n_seeds_graded += 1

    c_graded = bool(n_seeds_graded >= MIN_SEEDS_FOR_GRADED)

    # ---- C_ABS (supporting): absolute SAMPLED entropy rises across K ----
    abs_sampled_by_K = {
        k: by_id[f"ARM_SAMPLED_K{k}"]["cross_seed_interpretation"][
            "mean_within_class_rep_cond_entropy_nats"
        ]
        for k in POOL_SIZES
    }
    c_abs = _is_monotone_nondecreasing([abs_sampled_by_K[k] for k in POOL_SIZES])

    # ---- Outcome map ----
    if not non_vacuity_met:
        outcome_label = "FAIL"
        mech341_direction = "non_contributory"
        interpretation_label = (
            "FAIL_substrate_not_ready_requeue_within_class_availability_flat_across_pool_size"
        )
    elif c_graded:
        outcome_label = "PASS"
        mech341_direction = "supports"
        interpretation_label = (
            "PASS_graded_within_class_representative_diversity_dose_response_on_pool_size"
        )
    else:
        outcome_label = "FAIL"
        mech341_direction = "weakens"
        interpretation_label = (
            "FAIL_lift_not_graded_fixed_structural_artifact_independent_of_pool_size"
        )

    overall_direction = mech341_direction

    total_seeds = len(ARMS) * len(seeds)
    total_completed = sum(
        a["cross_seed_interpretation"]["n_seeds_completed"] for a in arms_out
    )

    return {
        "outcome": outcome_label,
        "overall_direction": overall_direction,
        "evidence_direction_per_claim": {"MECH-341": mech341_direction},
        "interpretation_label": interpretation_label,
        "seeds": seeds,
        "pool_sizes": list(POOL_SIZES),
        "n_arms": len(arms_out),
        "total_seeds_attempted": int(total_seeds),
        "total_seeds_completed": int(total_completed),
        "p0_episodes": int(p0_episodes),
        "p1_episodes": int(p1_episodes),
        "steps_per_episode": int(steps_per_episode),
        "decision_rule_thresholds": {
            "c2_lift_margin_nats": float(C2_LIFT_MARGIN_NATS),
            "min_seeds_for_graded": int(MIN_SEEDS_FOR_GRADED),
            "avail_rise_floor": float(AVAIL_RISE_FLOOR),
            "mono_tol": float(MONO_TOL),
            "mech341_entropy_bias_scale": float(MECH341_ENTROPY_BIAS_SCALE),
            "candidate_summary_source": CANDIDATE_SUMMARY_SOURCE,
            "use_modulatory_selection_authority": bool(
                USE_MODULATORY_SELECTION_AUTHORITY
            ),
            "modulatory_authority_gain": float(MODULATORY_AUTHORITY_GAIN),
            "sampled_within_class_temperature": float(SAMPLED_WITHIN_CLASS_T),
        },
        "acceptance_criteria": {
            "non_vacuity_within_class_availability_rises": bool(non_vacuity_met),
            "non_vacuity_avail_by_K_sampled": {
                str(k): round(float(v), 6) for k, v in avail_by_K_sampled.items()
            },
            "non_vacuity_avail_rise": round(avail_rise, 6),
            "C_GRADED_paired_lift_dose_response": bool(c_graded),
            "C_GRADED_n_seeds": int(n_seeds_graded),
            "C_GRADED_per_seed": {str(s): per_seed_graded[s] for s in seeds},
            "C_ABS_absolute_sampled_entropy_rises": bool(c_abs),
            "C_ABS_abs_sampled_by_K": {
                str(k): round(float(v), 6) for k, v in abs_sampled_by_K.items()
            },
            "realized_pool_size_honored": bool(pool_size_honored),
            "realized_pool_by_K_sampled": {
                str(k): round(float(v), 4) for k, v in realized_pool_by_K.items()
            },
        },
        "interpretation": {
            "label": interpretation_label,
            "preconditions": [
                {
                    "name": "within_class_representative_availability_rises_with_pool_size",
                    "description": (
                        "mean distinct within-class rep signatures available per "
                        "committed class (the SAME availability the C_GRADED lift "
                        "converts into selected diversity -- a count/range "
                        "statistic, NOT a magnitude proxy) rises monotonically "
                        "across the swept CEM pool sizes K. Below-floor -> the "
                        "graded axis has no leverage at this scale -> "
                        "substrate_not_ready_requeue, NOT a weakens."
                    ),
                    "measured": round(avail_rise, 6),
                    "threshold": float(AVAIL_RISE_FLOOR),
                    "direction": "lower",
                    "control": "SAMPLED arms across K (pool sizes 16->128)",
                    "met": bool(non_vacuity_met),
                },
                {
                    "name": "cem_pool_size_honored",
                    "description": (
                        "realized candidate-pool size rises with the configured "
                        "num_candidates K, confirming cfg.hippocampal.num_candidates "
                        "is honored (not silently overridden by from_dims/__init__)."
                    ),
                    "measured": round(
                        float(realized_pool_by_K[POOL_SIZES[-1]]
                              - realized_pool_by_K[POOL_SIZES[0]]),
                        4,
                    ),
                    "threshold": 1.0,
                    "direction": "lower",
                    "control": "SAMPLED arms K=16 vs K=128 realized pool size",
                    "met": bool(pool_size_honored),
                },
            ],
            "criteria": [
                {
                    "name": "C_GRADED_paired_lift_monotone_in_pool_size",
                    "load_bearing": True,
                    "passed": bool(c_graded),
                },
                {
                    "name": "C_ABS_absolute_sampled_entropy_rises",
                    "load_bearing": False,
                    "passed": bool(c_abs),
                },
            ],
            "criteria_non_degenerate": {
                "non_vacuity": bool(non_vacuity_met),
                "pool_size_honored": bool(pool_size_honored),
                "C_GRADED": bool(non_vacuity_met),
            },
        },
        "interpretation_grid": {
            "FAIL_substrate_not_ready_requeue": (
                "Available distinct within-class representatives did NOT rise "
                "across the CEM pool-size sweep (or the pool size was not "
                "honored). The graded axis has no leverage at this scale -- the "
                "lever could not be exercised. NOT an MECH-341 falsification; "
                "re-queue at a scale / substrate where pool size moves within-"
                "class availability (substrate_not_ready_requeue)."
            ),
            "PASS_graded_dose_response": (
                "The within-class-representative-diversity lift GROWS "
                "monotonically with available within-class diversity (CEM pool "
                "size): a positive dose-response. The MECH-341 within-class "
                "sub-axis is load-bearing AND graded -- the 660 ~0.08-nat lift is "
                "not a fixed noise floor. Route to /governance: MECH-341 "
                "v3_pending clearance candidate (supports). Do NOT auto-flip the "
                "within-class default (governance ratification gate)."
            ),
            "FAIL_lift_not_graded": (
                "Available within-class diversity rose with pool size, yet the "
                "paired lift (sampled - legacy) did NOT grow / was non-monotone. "
                "The ~0.08-nat lift is a FIXED structural artifact independent of "
                "available diversity -- not a load-bearing, graded contribution. "
                "A genuine negative for the MECH-341 within-class sub-axis as a "
                "graded lever. Route to /governance (weakens) + propagate to "
                "Q-054 + the arc_062 GAP-B wiring decision."
            ),
        },
        "arms": arms_out,
    }


def _build_manifest(
    result: Dict[str, Any],
    timestamp_utc: str,
    dry_run: bool,
) -> Dict[str, Any]:
    run_id = f"{EXPERIMENT_TYPE}_{timestamp_utc}_v3"
    manifest = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": timestamp_utc,
        "outcome": result["outcome"],
        "evidence_direction": result["overall_direction"],
        "evidence_direction_per_claim": result["evidence_direction_per_claim"],
        "evidence_direction_note": (
            f"V3-EXQ-660a MECH-341 within-class-representative-diversity GRADED "
            f"CONFIRMATION via a CEM candidate-pool-size dose-response (successor "
            f"to V3-EXQ-660; V3-EXQ-660 stays standing evidence -- NO supersede). "
            f"660 PASSED (lever load-bearing: legacy 4.781 vs sampled 4.862) but "
            f"its temperature sweep was byte-identical across T=0.5/1.0/2.0 "
            f"(near-degenerate within-class scores -> softmax ~uniform at every "
            f"T). This run holds the within-class temperature FIXED (None legacy "
            f"/ {SAMPLED_WITHIN_CLASS_T} sampled) and sweeps the CEM pool size K "
            f"in {list(POOL_SIZES)} (cfg.hippocampal.num_candidates), the count "
            f"of available within-class representatives -- the actual graded "
            f"axis. NON-VACUITY (load-bearing readiness, same statistic as the "
            f"criterion): available distinct within-class reps must rise across K "
            f"(else substrate_not_ready_requeue -> non_contributory, NOT a "
            f"weakens). C_GRADED (PRIMARY): per-seed paired lift "
            f"Delta(K)=sampled-legacy is monotone-nondecreasing across K AND "
            f"Delta(K_max)-Delta(K_min) >= {C2_LIFT_MARGIN_NATS} nats on "
            f">= {MIN_SEEDS_FOR_GRADED}/3 seeds. PASS=non-vacuity AND C_GRADED -> "
            f"supports (graded + load-bearing; v3_pending clearance candidate "
            f"pending governance ratification). non-vacuity met but C_GRADED "
            f"fails -> weakens (fixed structural artifact, not graded). "
            f"interpretation_label={result['interpretation_label']}. "
            f"non_vacuity="
            f"{result['acceptance_criteria']['non_vacuity_within_class_availability_rises']}, "
            f"C_GRADED={result['acceptance_criteria']['C_GRADED_paired_lift_dose_response']}, "
            f"C_ABS={result['acceptance_criteria']['C_ABS_absolute_sampled_entropy_rises']}, "
            f"pool_honored="
            f"{result['acceptance_criteria']['realized_pool_size_honored']}. "
            f"experiment_purpose=evidence; MECH-341 is a v3_pending candidate."
        ),
        "dry_run": bool(dry_run),
        "env_kwargs": dict(ENV_KWARGS),
        "config_summary": {
            "graded_axis": "cem_pool_size (cfg.hippocampal.num_candidates)",
            "pool_sizes": list(POOL_SIZES),
            "within_class_temperature_fixed": {
                "legacy": None, "sampled": SAMPLED_WITHIN_CLASS_T,
            },
            "vs_stack": "minimal (use_per_stream_vs + use_vs_rollout_gating)",
            "mech341_sub_flavours": "both (entropy_bonus + stratified_select)",
            "mech341_entropy_bias_scale": MECH341_ENTROPY_BIAS_SCALE,
            "candidate_summary_source": CANDIDATE_SUMMARY_SOURCE,
            "use_modulatory_selection_authority": USE_MODULATORY_SELECTION_AUTHORITY,
            "modulatory_authority_gain": MODULATORY_AUTHORITY_GAIN,
            "primary_readout": "within_class_rep_cond_entropy_nats (H(rep_signature | committed_class))",
            "non_vacuity_statistic": "mean_distinct_within_class_reps (availability)",
            "z_goal_enabled": True,
            "drive_weight": 2.0,
            "alpha_world": 0.9,
            "reef_enabled": True,
            "reef_bipartite_layout": True,
            "sd056_amend_active": True,
            "elite_fraction": "default 0.2 (untouched)",
            "use_differentiable_cem": "NOT FLIPPED (default False; SD-055 safety note)",
        },
        "result": result,
    }
    if SUPERSEDES is not None:
        manifest["supersedes"] = SUPERSEDES
    return manifest


def main() -> Tuple[Optional[str], Optional[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    if args.dry_run:
        seeds = list(DRY_RUN_SEEDS)
        p0 = DRY_RUN_P0
        p1 = DRY_RUN_P1
        steps = DRY_RUN_STEPS
    else:
        seeds = list(SEEDS)
        p0 = P0_WARMUP_EPISODES
        p1 = P1_MEASUREMENT_EPISODES
        steps = STEPS_PER_EPISODE

    result = run_experiment(
        seeds=seeds,
        p0_episodes=p0,
        p1_episodes=p1,
        steps_per_episode=steps,
        dry_run=bool(args.dry_run),
    )

    timestamp_utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest = _build_manifest(result, timestamp_utc, dry_run=bool(args.dry_run))

    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
    else:
        out_dir = (
            Path(__file__).resolve().parents[2]
            / "REE_assembly" / "evidence" / "experiments"
        )
    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=args.dry_run,
        config=manifest.get("config") or manifest.get("config_summary"),
        seeds=SEEDS,
        script_path=Path(__file__),
    )

    print(f"manifest: {out_path}", flush=True)
    if not args.dry_run:
        print(f"Result written to: {out_path}", flush=True)
    print(
        f"outcome: {result['outcome']} "
        f"completed={result['total_seeds_completed']}/{result['total_seeds_attempted']} "
        f"non_vacuity={result['acceptance_criteria']['non_vacuity_within_class_availability_rises']} "
        f"C_GRADED={result['acceptance_criteria']['C_GRADED_paired_lift_dose_response']} "
        f"C_ABS={result['acceptance_criteria']['C_ABS_absolute_sampled_entropy_rises']} "
        f"pool_honored={result['acceptance_criteria']['realized_pool_size_honored']} "
        f"label={result['interpretation_label']}",
        flush=True,
    )

    if args.dry_run:
        try:
            out_path.unlink()
        except FileNotFoundError:
            pass

    outcome_norm = result["outcome"].upper()
    outcome_emit = outcome_norm if outcome_norm in ("PASS", "FAIL") else "FAIL"
    manifest_for_sentinel = str(out_path) if not args.dry_run else None
    return outcome_emit, manifest_for_sentinel


if __name__ == "__main__":
    _outcome, _manifest_path = main()
    if _outcome is not None:
        emit_outcome(outcome=_outcome, manifest_path=_manifest_path)
    sys.exit(0)
