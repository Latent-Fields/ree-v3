#!/opt/local/bin/python3
"""
V3-EXQ-660 -- MECH-341 committed-class diversity RETEST on the GAP-A-ready /
authority-ready stack, using a WITHIN-CLASS-REPRESENTATIVE-DIVERSITY readout.

Supersedes the within-class lineage's committed-class-entropy readout
(V3-EXQ-614e, autopsied non_contributory / substrate_ceiling 2026-06-07). The
614e autopsy (failure_autopsy_V3-EXQ-614e_2026-06-07) established TWO things:
  Learning #2: committed-class (across-class / first-action) entropy is the
    WRONG matched readout for a WITHIN-class lever -- it cannot move when the
    within-class representative shifts, so a flat committed-class entropy is
    expected, not a falsification.
  The bottleneck relocated from the authority gate (GAP-B, resolved by the
    V3-EXQ-643a float32-cancellation fix) to the UPSTREAM candidate pool
    (GAP-A: all K candidates collapsed to identical z_world after one E2 step,
    cand_world_pairwise_dist=0.0000), so within-class temperature had nothing
    to diversify.

GAP-A is now substrate-ready: V3-EXQ-649 PASS (2026-06-07) validated the
shared-channel re-sourcing of cand_world_summaries from the SD-056-trained
e2.world_forward (REEConfig.candidate_summary_source="e2_world_forward"),
delivering per-candidate spread >= the 0.05 readiness floor. So this retest:
  (1) arms the GAP-A fix on every arm
     (candidate_summary_source="e2_world_forward"), so per-candidate scores
     are action-divergent and the within-class score ordering is meaningful;
  (2) arms the V3-EXQ-643a authority fix on every arm
     (use_modulatory_selection_authority=True, gain=0.5);
  (3) sweeps ONLY the MECH-341 within-class lever
     (e3_diversity_stratified_within_class_temperature in {None, 0.5, 1.0, 2.0});
  (4) measures the CORRECT matched readout -- WITHIN-CLASS-REPRESENTATIVE
     DIVERSITY -- as the PRIMARY DV, and keeps committed-class (first-action)
     entropy as a SECONDARY / negative-control readout (expected flat).

WITHIN-CLASS-REPRESENTATIVE-DIVERSITY readout (the design change vs 614e)
-----------------------------------------------------------------------
The within-class temperature lever samples WHICH representative wins inside a
committed first-action class via softmax(-class_scores / T) (see
ree_core/predictors/e3_score_diversity.py stratified_select). All members of a
first-action class share the SAME first action (that is what defines the class);
they DIFFER in their subsequent actions (second action onward). So the
within-class representative is identified by its POST-FIRST-ACTION signature.

The selected representative's full trajectory is exposed every tick on
agent.e3._last_selected_trajectory (e3_selector.py:978, set unconditionally).
Per P1 tick we read:
  committed_class   = argmax(selected_traj.actions[:, 0, :])   (across-class axis)
  rep_signature     = tuple(argmax(selected_traj.actions[:, t, :]) for t>=1)
                      (the within-class representative discriminator)

PRIMARY DV (C2): within_class_rep_cond_entropy = H(rep_signature | committed_class)
  -- the conditional entropy of which within-class representative is selected,
  given the committed class. As within-class temperature rises, the selected
  representative inside a committed class should vary more across ticks, so this
  conditional entropy RISES. Measured as a per-seed PAIRED lift over the
  ARM_0_LEGACY (argmin within-class) baseline, mirroring the 614e C2 structure
  but on the corrected readout.

SECONDARY / negative control (reported, NOT a pass/fail gate):
  committed_class_entropy (first-action across-class). 614e showed this is
  byte-identical to within-class temperature; that insensitivity is EXPECTED
  here (the within-class lever does not act on the across-class axis) and is
  NOT a falsification of MECH-341.

Substrate-readiness / non-vacuity precondition (protects MECH-341 from a false
weakens at degenerate scale)
-----------------------------------------------------------------------------
The within-class lever can only diversify a representative when the committed
class actually offers >= 2 DISTINCT representatives in the candidate pool. If
the GAP-A pool is degenerate at the run scale (within-class members collapse to
a single rep signature, or the within-class branch never fires), the test is
vacuous. C1 below gates exactly this and self-routes the run to
substrate_not_ready_requeue -> non_contributory (NOT a weakens). The readiness
statistic (multi-rep availability) is the SAME within-class-representative axis
the C2 criterion routes on -- not a magnitude proxy.

Arms (4, on the SD-056-amended + GAP-A-ready + authority-ready stack)
--------------------------------------------------------------------
  ARM_0_LEGACY:   stratified_within_class_temperature = None   (legacy argmin within-class)
  ARM_1_T_0_5:    stratified_within_class_temperature = 0.5     (sharpened)
  ARM_2_T_1_0:    stratified_within_class_temperature = 1.0     (mid-T)
  ARM_3_T_2_0:    stratified_within_class_temperature = 2.0     (flatter)

All four arms run with:
  candidate_summary_source = "e2_world_forward"  (GAP-A; V3-EXQ-649 PASS)
  use_modulatory_selection_authority = True, gain = 0.5  (V3-EXQ-643a)
  Layer A: SP-CEM ON; Layer B: MECH-341 ON (both sub-flavours; bias_scale=2.0)
  Layer C: MECH-313 noise floor ON; Layer D: V_s minimal stack ON
  SD-056 amend: all levers ON (multi-step contrastive h=5 + per-step output norm
                clamp ratio=2.0 + t=1 contrastive). Eval-time agent (no online
                SD-056 training), so the 643 instability is not a concern; the
                clamp is retained for lineage parity.

Pre-registered acceptance criteria
----------------------------------
  C1 (substrate-operative non-vacuity -- the load-bearing GAP-A readiness gate):
     across the positive-temperature arms, on a majority (>= 2/3) of seeds the
     within-class branch fires (mech341_n_within_class_sampled >=
     WITHIN_CLASS_FIRE_FLOOR) AND the committed classes offer >= 2 distinct
     within-class representatives on at least MULTI_REP_TICK_FLOOR ticks
     (n_multi_rep_ticks; the GAP-A within-class-diversity-delivered statistic).
     C1 FALSE -> the test is vacuous -> substrate_not_ready_requeue.
  C2 (PRIMARY -- the substrate target): within_class_rep_cond_entropy RISES with
     within-class temperature. Each positive-temperature arm produces at least
     C2_MIN_LIFT_SEEDS_PER_ARM per-seed PAIRED lift over ARM_0_LEGACY of at
     least C2_LIFT_MARGIN_NATS. Paired by seed index (same seed -> same env).
  C3 (GAP-A first-action class readiness, context): all 4 arms produce
     frac_pre_ge2 > 0.3 on a majority of seeds.

Overall outcome
---------------
  PASS = C1 (substrate operative + within-class diversity available) AND C2
         (within-class temperature lifts within-class-representative diversity).
         MECH-341 within-class sub-axis supports.
  FAIL/weakens = C1 holds but C2 fails -- the within-class lever is operative and
         the pool offers >= 2 representatives, yet within-class temperature adds
         no within-class-representative diversity. A genuine negative for the
         MECH-341 within-class sub-axis.
  FAIL/non_contributory = C1 fails -- the GAP-A within-class pool is degenerate
         or the within-class branch did not fire; the lever could not express
         itself. NOT an MECH-341 falsification. Route substrate_not_ready_requeue
         (re-queue on a more-trained / larger-pool substrate).

Claims: [MECH-341] (single claim; the within-class Layer-B sub-axis is the only
varied lever). experiment_purpose=evidence.

Phases
------
P0 (30 ep, instrumentation OFF): encoder + E2 (SD-056) warmup so e2.world_forward
   is trained enough to deliver action-divergent candidate summaries (GAP-A).
P1 (60 ep, instrumentation ON): behavioural measurement window. Matches the
   614b / 614c / 614d / 614e P1 budget for direct manifest comparability.

Budget: 4 arms x 3 seeds x 90 ep x 200 steps = 216k steps total. ~3-4 h.
The C2 paired-lift comparison is within-run (same seed, same machine), so no
cross-machine reference is required -> machine_affinity "any".

See REE_assembly/docs/architecture/mech_341_e3_score_diversity_preservation.md,
REE_assembly/docs/architecture/mech_314a_phase2_novelty_source_design.md
(GAP-A shared-channel section),
REE_assembly/docs/architecture/modulatory_bias_selection_authority.md
(V3-EXQ-643a fix),
REE_assembly/evidence/planning/behavioral_diversity_isolation_plan.md GAP-B,
REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-614e_2026-06-07.{md,json}.
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


EXPERIMENT_TYPE = "v3_exq_660_mech341_within_class_representative_diversity"
QUEUE_ID = "V3-EXQ-660"
SUPERSEDES = "V3-EXQ-614e"
CLAIM_IDS: List[str] = ["MECH-341"]
EXPERIMENT_PURPOSE = "evidence"

# V3-EXQ-660 sweep axis: within-class temperature in {None, 0.5, 1.0, 2.0}.
# None = legacy argmin within-class (the paired control arm).
WITHIN_CLASS_T_BY_ARM: Dict[str, Optional[float]] = {
    "ARM_0_LEGACY": None,
    "ARM_1_T_0_5": 0.5,
    "ARM_2_T_1_0": 1.0,
    "ARM_3_T_2_0": 2.0,
}

# GAP-A shared-channel re-sourcing (V3-EXQ-649 PASS). ON for ALL arms.
CANDIDATE_SUMMARY_SOURCE = "e2_world_forward"

# modulatory-bias-selection-authority (V3-EXQ-643a-fixed substrate). ON for ALL
# arms. gain=0.5 keeps modulatory signals competitive in near-ties but
# subdominant when the primary harm/goal gap exceeds gain * raw_score_range.
USE_MODULATORY_SELECTION_AUTHORITY = True
MODULATORY_AUTHORITY_GAIN = 0.5

# C2 PRIMARY: per-arm PAIRED lift in within_class_rep_cond_entropy_nats over
# ARM_0_LEGACY, >= C2_MIN_LIFT_SEEDS_PER_ARM seeds for EACH positive-T arm.
# Margin small relative to conditional-entropy magnitudes but above per-seed
# measurement noise (matches the 614e committed-class margin).
C2_LIFT_MARGIN_NATS = 0.05
C2_MIN_LIFT_SEEDS_PER_ARM = 1   # >= 1 paired-lift seed PER positive-T arm

# C1 within-class branch firing floor (per-seed). A positive-temperature arm
# whose within-class branch accumulated at least this many samples over P1 is
# branch-active.
WITHIN_CLASS_FIRE_FLOOR = 10

# C1 GAP-A within-class-diversity-availability floor (per-seed). At least this
# many P1 ticks where the committed class offered >= 2 DISTINCT within-class
# representative signatures in the candidate pool. This is the
# same-statistic-as-the-criterion readiness check: C2 routes on
# within-class-representative diversity, so the readiness asserts within-class-
# representative AVAILABILITY (not a magnitude proxy).
MULTI_REP_TICK_FLOOR = 5

# SD-056 amend lever defaults applied uniformly across all 4 arms.
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

MIN_SEEDS_PER_ARM_FOR_PASS = 2  # of 3 (used for C1 / C3 majority checks)

# MECH-341 sub-flavour scale used in the entropy-ON arms (matches 614e).
MECH341_ENTROPY_BIAS_SCALE = 2.0

# V_s (D) thresholds (minimal stack).
VS_SNAPSHOT_REFRESH_THRESHOLD = 0.5
VS_E1_THRESHOLD = 0.4


# IDENTICAL to V3-EXQ-611 / 614c / 614d / 614e for direct manifest comparability.
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


ARMS: List[Dict[str, Any]] = [
    {
        "arm_id": "ARM_0_LEGACY",
        "label": "within_class_legacy_argmin",
        "within_class_temperature": None,
    },
    {
        "arm_id": "ARM_1_T_0_5",
        "label": "within_class_T_0_5_sharpened",
        "within_class_temperature": 0.5,
    },
    {
        "arm_id": "ARM_2_T_1_0",
        "label": "within_class_T_1_0_mid",
        "within_class_temperature": 1.0,
    },
    {
        "arm_id": "ARM_3_T_2_0",
        "label": "within_class_T_2_0_flatter",
        "within_class_temperature": 2.0,
    },
]


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(seed=seed, **ENV_KWARGS)


def _make_agent(
    within_class_temperature: Optional[float],
    env: CausalGridWorldV2,
) -> REEAgent:
    """Build a REEAgent with the SD-056-amended 4-substrate baseline + the
    modulatory-bias-selection-authority substrate ON + the GAP-A shared-channel
    candidate_summary_source re-sourcing ON.

    Config is bit-identical to V3-EXQ-614e _make_agent EXCEPT
    candidate_summary_source="e2_world_forward" (the GAP-A fix, V3-EXQ-649 PASS)
    -- so the per-candidate scores consumed by the within-class stratified
    selection are action-divergent and the within-class score ordering is
    meaningful (614e collapsed because the shared cand_world_summaries were
    sourced from the monostrategic proposer first-step z_world).
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
        # MECH-341 within-class proportional sampling sweep axis. None = legacy
        # argmin within-class (the paired control arm).
        e3_diversity_stratified_within_class_temperature=within_class_temperature,
        # GAP-A shared-channel candidate-summary re-sourcing (V3-EXQ-649 PASS).
        candidate_summary_source=CANDIDATE_SUMMARY_SOURCE,
        # modulatory-bias-selection-authority (V3-EXQ-643a-fixed). ON for all
        # arms -- arms Site 1 (additive authority) AND Site 2 (stratified
        # across-class unit-range normalization).
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
    return REEAgent(cfg)


# ---------------------------------------------------------------------------
# Per-tick measurement helpers
# ---------------------------------------------------------------------------


def _first_action_class(traj) -> int:
    return int(
        traj.actions[:, 0, :].argmax(dim=-1).detach().reshape(-1)[0].item()
    )


def _rep_signature(traj) -> Tuple[int, ...]:
    """Within-class representative discriminator: argmax of each action step
    AFTER the first. Members of the same first-action class share step 0 and
    differ from step 1 onward, so the post-first-action signature identifies
    WHICH within-class representative was selected.
    """
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


def _conditional_entropy(
    pairs: List[Tuple[int, Tuple[int, ...]]]
) -> float:
    """H(rep_signature | committed_class), weighted by committed-class frequency.

    pairs: list of (committed_first_action_class, rep_signature) over P1 ticks.
    For each committed class c, compute the entropy of the rep-signature
    distribution within c, then weight by n_c / N. Classes that only ever
    presented one rep signature contribute 0 (no within-class choice was made).
    """
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

    # arm_fingerprint per-cell obligation: complete RNG reset on enter + stamp.
    # config_slice captures the cell's distinguishing config so distinct arms /
    # seeds produce distinct content-addressed fingerprints.
    config_slice = {
        "arm_id": arm["arm_id"],
        "within_class_temperature": within_class_temperature,
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
        # Each cell builds a fresh agent + env; no shared mutable state across
        # cells (separate optimizer/buffers per cell), so the cell is a pure
        # function of (substrate, config, seed).
    ) as cell:
        env = _make_env(seed)
        agent = _make_agent(within_class_temperature, env)
        agent.eval()

        total_train_eps = p0_episodes + p1_episodes
        error_note: Optional[str] = None
        n_p0_ticks = 0
        n_p1_ticks = 0
        n_p1_pre_ge2 = 0
        n_p1_pre_eq1 = 0
        committed_classes_p1: Dict[int, int] = {}

        # PRIMARY readout accumulators: (committed_class, rep_signature) pairs +
        # the GAP-A within-class-availability statistic.
        selected_pairs: List[Tuple[int, Tuple[int, ...]]] = []
        n_multi_rep_ticks = 0

        # MECH-341 within-class firing diagnostics, accumulated over the P1 window.
        n_within_class_sampled_total = 0
        n_stratified_fired_total = 0
        n_authority_normalized_total = 0
        last_within_class_temperature = 0.0
        last_rep_score_range = 0.0

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

                    # PRIMARY readout: the selected within-class representative's
                    # full action signature. _last_selected_trajectory is set
                    # unconditionally every tick (e3_selector.py:978).
                    sel_traj = getattr(agent.e3, "_last_selected_trajectory", None)
                    if sel_traj is not None:
                        sel_cls = _first_action_class(sel_traj)
                        sel_sig = _rep_signature(sel_traj)
                        selected_pairs.append((sel_cls, sel_sig))

                    # GAP-A within-class-availability: did the committed class
                    # offer >= 2 DISTINCT representative signatures in the pool?
                    if candidates:
                        pool_sigs_in_class = {
                            _rep_signature(t)
                            for t in candidates
                            if _first_action_class(t) == committed_class
                        }
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

            # MECH-341 within-class + authority diagnostics: read AFTER the inner
            # step loop and BEFORE the next agent.reset() clears them. Accumulate
            # only over the P1 measurement window.
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
                ep_within_temp = float(
                    sd_state.get("mech341_last_within_class_temperature", 0.0)
                )
                if ep_within_temp != 0.0:
                    last_within_class_temperature = ep_within_temp
                ep_rep_range = float(sd_state.get("mech341_last_rep_score_range", 0.0))
                if ep_rep_range != 0.0:
                    last_rep_score_range = ep_rep_range

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

        seed_substrate_ready = bool(frac_pre_ge2 > 0.3)
        within_class_branch_active = bool(
            n_within_class_sampled_total >= WITHIN_CLASS_FIRE_FLOOR
        )
        multi_rep_available = bool(n_multi_rep_ticks >= MULTI_REP_TICK_FLOOR)

        row = {
            "arm_id": arm["arm_id"],
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
            # PRIMARY readout (within-class representative diversity):
            "within_class_rep_cond_entropy_nats": round(
                within_class_rep_cond_entropy, 6
            ),
            "n_selected_pairs": int(len(selected_pairs)),
            "n_multi_rep_ticks": int(n_multi_rep_ticks),
            # SECONDARY / negative-control readout (committed-class across-class):
            "committed_class_entropy_nats": round(committed_class_entropy, 6),
            # MECH-341 within-class firing diagnostics (per-seed accumulated):
            "mech341_n_within_class_sampled": int(n_within_class_sampled_total),
            "mech341_n_stratified_fired": int(n_stratified_fired_total),
            "mech341_n_authority_normalized": int(n_authority_normalized_total),
            "mech341_last_within_class_temperature": round(
                last_within_class_temperature, 6
            ),
            "mech341_last_rep_score_range": round(last_rep_score_range, 6),
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
    com_entropies = [r["committed_class_entropy_nats"] for r in completed]
    mean_rep_entropy = (
        sum(rep_entropies) / len(rep_entropies) if rep_entropies else 0.0
    )
    mean_committed_entropy = (
        sum(com_entropies) / len(com_entropies) if com_entropies else 0.0
    )

    n_substrate_ready = sum(1 for r in completed if r["seed_substrate_ready"])
    n_within_class_active = sum(
        1 for r in completed if r["within_class_branch_active"]
    )
    n_multi_rep = sum(1 for r in completed if r["multi_rep_available"])

    return {
        "n_seeds_completed": int(n_seeds_completed),
        "mean_within_class_rep_cond_entropy_nats": round(mean_rep_entropy, 6),
        "mean_committed_class_entropy_nats": round(mean_committed_entropy, 6),
        "n_seeds_substrate_ready": int(n_substrate_ready),
        "majority_substrate_ready": bool(
            n_substrate_ready >= MIN_SEEDS_PER_ARM_FOR_PASS
        ),
        "n_seeds_within_class_active": int(n_within_class_active),
        "majority_within_class_active": bool(
            n_within_class_active >= MIN_SEEDS_PER_ARM_FOR_PASS
        ),
        "n_seeds_multi_rep_available": int(n_multi_rep),
        "majority_multi_rep_available": bool(
            n_multi_rep >= MIN_SEEDS_PER_ARM_FOR_PASS
        ),
        "within_class_samples_per_seed": [
            int(r["mech341_n_within_class_sampled"]) for r in completed
        ],
    }


def _classify_outcome(c1: bool, c2: bool, c3: bool) -> Tuple[str, str, str]:
    """V3-EXQ-660 outcome map.

    C1: substrate-operative non-vacuity (within-class branch fires AND committed
        classes offer >= 2 representatives on a majority of seeds in the positive
        arms -- the GAP-A within-class-diversity-delivered readiness gate).
    C2 (PRIMARY): per-arm within-class-representative diversity lift (each
        positive-T arm has >= C2_MIN_LIFT_SEEDS_PER_ARM paired-lift seeds over
        ARM_0_LEGACY on within_class_rep_cond_entropy).
    C3: GAP-A first-action class readiness (frac_pre_ge2 > 0.3 majority).

    PASS = C1 AND C2.
    """
    if c1 and c2:
        return (
            "PASS", "supports",
            "PASS_C1_C2_within_class_temperature_lifts_within_class_representative_diversity",
        )
    if c1 and (not c2):
        return (
            "FAIL", "weakens",
            "FAIL_C1_holds_C2_fails_within_class_lever_operative_pool_diverse_but_no_representative_diversity_lift",
        )
    # not c1 -- the GAP-A within-class pool is degenerate or the branch did not
    # fire; the lever could not express itself. NOT an MECH-341 falsification.
    return (
        "FAIL", "non_contributory",
        "FAIL_C1_substrate_not_ready_requeue_within_class_pool_degenerate",
    )


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
            f"(P0={p0_episodes} ep, P1={p1_episodes} ep, "
            f"steps_per_episode={steps_per_episode}, "
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
            print(f"verdict: {verdict}", flush=True)
        cross = _interpret_arm(seed_rows)
        arms_out.append({
            "arm_id": arm["arm_id"],
            "label": arm["label"],
            "within_class_temperature": arm.get("within_class_temperature"),
            "per_seed_results": seed_rows,
            "cross_seed_interpretation": cross,
        })

    # ----- Cross-arm acceptance criteria (V3-EXQ-660) -----
    by_id = {a["arm_id"]: a for a in arms_out}
    arm_legacy = by_id["ARM_0_LEGACY"]
    swept_arms = [by_id["ARM_1_T_0_5"], by_id["ARM_2_T_1_0"], by_id["ARM_3_T_2_0"]]

    # --- C2 (PRIMARY): per-seed PAIRED within_class_rep_cond_entropy lift over
    #     ARM_0_LEGACY, >= C2_MIN_LIFT_SEEDS_PER_ARM seeds for EACH positive arm.
    legacy_rep_by_seed: Dict[int, float] = {
        int(r["seed"]): r["within_class_rep_cond_entropy_nats"]
        for r in arm_legacy["per_seed_results"]
        if r["error_note"] is None
    }

    def _arm_paired_lift_count(arm) -> int:
        n = 0
        for r in arm["per_seed_results"]:
            if r["error_note"] is not None:
                continue
            seed = int(r["seed"])
            if seed not in legacy_rep_by_seed:
                continue
            lift = (
                r["within_class_rep_cond_entropy_nats"]
                - legacy_rep_by_seed[seed]
            )
            if lift >= C2_LIFT_MARGIN_NATS:
                n += 1
        return n

    c2_per_arm_lift = {a["arm_id"]: _arm_paired_lift_count(a) for a in swept_arms}
    c2_holds = all(
        n >= C2_MIN_LIFT_SEEDS_PER_ARM for n in c2_per_arm_lift.values()
    )

    c2_per_arm_rep_mean = {
        a["arm_id"]: a["cross_seed_interpretation"][
            "mean_within_class_rep_cond_entropy_nats"
        ]
        for a in swept_arms
    }
    legacy_rep_mean = arm_legacy["cross_seed_interpretation"][
        "mean_within_class_rep_cond_entropy_nats"
    ]

    # --- C1 (substrate-operative non-vacuity): within-class branch fires AND
    #     multi-rep available on a majority of seeds in the positive arms.
    swept_branch_active_majority = all(
        a["cross_seed_interpretation"]["majority_within_class_active"]
        for a in swept_arms
    )
    swept_multi_rep_majority = all(
        a["cross_seed_interpretation"]["majority_multi_rep_available"]
        for a in swept_arms
    )
    c1_holds = bool(swept_branch_active_majority and swept_multi_rep_majority)

    # --- C3 (GAP-A first-action class readiness): all arms frac_pre_ge2 > 0.3.
    c3_per_arm_ready = {
        a["arm_id"]: a["cross_seed_interpretation"]["n_seeds_substrate_ready"]
        for a in arms_out
    }
    c3_holds = all(
        n >= MIN_SEEDS_PER_ARM_FOR_PASS for n in c3_per_arm_ready.values()
    )

    multi_rep_per_arm = {
        a["arm_id"]: a["cross_seed_interpretation"]["n_seeds_multi_rep_available"]
        for a in arms_out
    }
    within_class_active_per_arm = {
        a["arm_id"]: a["cross_seed_interpretation"]["n_seeds_within_class_active"]
        for a in swept_arms
    }

    # SECONDARY / negative-control readout (reported, NOT gated).
    committed_class_mean_by_arm = {
        a["arm_id"]: a["cross_seed_interpretation"][
            "mean_committed_class_entropy_nats"
        ]
        for a in arms_out
    }

    (
        outcome_label, mech341_direction, interpretation_label,
    ) = _classify_outcome(c1_holds, c2_holds, c3_holds)
    overall_direction = mech341_direction

    total_seeds = len(ARMS) * len(seeds)
    total_completed = sum(
        a["cross_seed_interpretation"]["n_seeds_completed"] for a in arms_out
    )

    return {
        "outcome": outcome_label,
        "overall_direction": overall_direction,
        "evidence_direction_per_claim": {
            "MECH-341": mech341_direction,
        },
        "interpretation_label": interpretation_label,
        "seeds": seeds,
        "n_arms": len(arms_out),
        "total_seeds_attempted": int(total_seeds),
        "total_seeds_completed": int(total_completed),
        "p0_episodes": int(p0_episodes),
        "p1_episodes": int(p1_episodes),
        "steps_per_episode": int(steps_per_episode),
        "decision_rule_thresholds": {
            "min_seeds_per_arm_for_pass": int(MIN_SEEDS_PER_ARM_FOR_PASS),
            "mech341_entropy_bias_scale": float(MECH341_ENTROPY_BIAS_SCALE),
            "vs_snapshot_refresh_threshold": float(VS_SNAPSHOT_REFRESH_THRESHOLD),
            "vs_e1_threshold": float(VS_E1_THRESHOLD),
            "c2_lift_margin_nats": float(C2_LIFT_MARGIN_NATS),
            "c2_min_lift_seeds_per_arm": int(C2_MIN_LIFT_SEEDS_PER_ARM),
            "within_class_fire_floor": int(WITHIN_CLASS_FIRE_FLOOR),
            "multi_rep_tick_floor": int(MULTI_REP_TICK_FLOOR),
            "candidate_summary_source": CANDIDATE_SUMMARY_SOURCE,
            "use_modulatory_selection_authority": bool(
                USE_MODULATORY_SELECTION_AUTHORITY
            ),
            "modulatory_authority_gain": float(MODULATORY_AUTHORITY_GAIN),
        },
        "acceptance_criteria": {
            "C1_substrate_operative_non_vacuity": c1_holds,
            "C1_swept_within_class_branch_active_majority": swept_branch_active_majority,
            "C1_swept_multi_rep_available_majority": swept_multi_rep_majority,
            "C2_within_class_representative_diversity_lift_per_arm": c2_holds,
            "C2_per_arm_paired_lift_seed_counts": {
                k: int(v) for k, v in c2_per_arm_lift.items()
            },
            "C2_legacy_within_class_rep_cond_entropy_mean": round(
                legacy_rep_mean, 6
            ),
            "C2_per_arm_within_class_rep_cond_entropy_mean": {
                k: round(float(v), 6) for k, v in c2_per_arm_rep_mean.items()
            },
            "C3_gap_a_first_action_class_readiness": c3_holds,
            "C3_per_arm_ready_seed_counts": {
                k: int(v) for k, v in c3_per_arm_ready.items()
            },
        },
        "secondary_negative_control": {
            "committed_class_entropy_mean_by_arm": {
                k: round(float(v), 6) for k, v in committed_class_mean_by_arm.items()
            },
            "note": (
                "Committed-class (first-action / across-class) entropy is the "
                "SECONDARY / negative-control readout. The within-class lever "
                "does NOT act on the across-class axis, so a flat committed-class "
                "entropy across within-class temperature is EXPECTED (614e showed "
                "byte-identical) and is NOT a falsification of MECH-341. Reported "
                "for context only; not a pass/fail gate."
            ),
        },
        "non_vacuity_diagnostics": {
            "within_class_active_seed_counts_per_arm": {
                k: int(v) for k, v in within_class_active_per_arm.items()
            },
            "multi_rep_available_seed_counts_by_arm": {
                k: int(v) for k, v in multi_rep_per_arm.items()
            },
            "note": (
                "C1 is the GAP-A within-class-diversity-delivered readiness gate. "
                "It asserts the SAME within-class-representative axis the C2 "
                "criterion routes on (representative AVAILABILITY: >= 2 distinct "
                "rep signatures in the committed class), not a magnitude proxy. "
                "C1 FALSE -> substrate_not_ready_requeue -> non_contributory "
                "(protects MECH-341 from a false weakens at degenerate scale)."
            ),
        },
        "interpretation_grid": {
            "PASS_C1_C2": (
                "Within-class proportional sampling lever VALIDATED on the "
                "GAP-A-ready / authority-ready substrate using the matched "
                "within-class-representative-diversity readout. MECH-341 "
                "within-class sub-axis is load-bearing: within-class temperature "
                "lifts which representative is selected inside the committed "
                "class. Route to /governance: MECH-341 v3_pending clearance "
                "candidate (supports). Do NOT auto-flip the within-class default "
                "(governance ratification gate)."
            ),
            "FAIL_C1_holds_C2_fails": (
                "Lever operative (within-class branch fires) AND the GAP-A pool "
                "offers >= 2 within-class representatives, yet within-class "
                "temperature adds no within-class-representative diversity over "
                "legacy argmin. A GENUINE negative for the MECH-341 within-class "
                "sub-axis. Route to /governance (within-class sub-axis not "
                "load-bearing); propagate to Q-054 + the arc_062 GAP-B wiring "
                "decision."
            ),
            "FAIL_C1_substrate_not_ready_requeue": (
                "The within-class branch did not fire and/or the committed "
                "classes did not offer >= 2 distinct within-class representatives "
                "(GAP-A within-class pool degenerate at this scale, or the e2 "
                "world-forward under-trained). The lever could not express "
                "itself. NOT an MECH-341 falsification. Re-queue on a more-trained "
                "/ larger-pool substrate (substrate_not_ready_requeue)."
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
    return {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "supersedes": SUPERSEDES,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": timestamp_utc,
        "outcome": result["outcome"],
        "evidence_direction": result["overall_direction"],
        "evidence_direction_per_claim": result["evidence_direction_per_claim"],
        "evidence_direction_note": (
            f"V3-EXQ-660 MECH-341 committed-class diversity RETEST on the "
            f"GAP-A-ready (candidate_summary_source=e2_world_forward; V3-EXQ-649 "
            f"PASS) + authority-ready (V3-EXQ-643a) stack, using the CORRECTED "
            f"within-class-representative-diversity readout (supersedes "
            f"V3-EXQ-614e, whose committed-class-entropy readout was the wrong "
            f"matched axis for a within-class lever -- 614e autopsy Learning #2). "
            f"C2 (PRIMARY) tests whether within_class_rep_cond_entropy "
            f"(H(rep_signature | committed_class)) RISES with within-class "
            f"temperature (per-arm >= {C2_MIN_LIFT_SEEDS_PER_ARM} paired-lift "
            f"seed). C1 is the GAP-A within-class-diversity-delivered non-vacuity "
            f"gate (within-class branch fires + committed class offers >= 2 "
            f"distinct representatives); C1 FALSE self-routes "
            f"substrate_not_ready_requeue -> non_contributory (NOT a weakens). "
            f"Committed-class (across-class) entropy is a SECONDARY / "
            f"negative-control readout, expected flat. interpretation_label="
            f"{result['interpretation_label']}. "
            f"C1={result['acceptance_criteria']['C1_substrate_operative_non_vacuity']}, "
            f"C2={result['acceptance_criteria']['C2_within_class_representative_diversity_lift_per_arm']}, "
            f"C3={result['acceptance_criteria']['C3_gap_a_first_action_class_readiness']}. "
            f"experiment_purpose=evidence; MECH-341 is a v3_pending candidate "
            f"(e3_scoring_preserves_trajectory_class_diversity)."
        ),
        "dry_run": bool(dry_run),
        "env_kwargs": dict(ENV_KWARGS),
        "config_summary": {
            "vs_stack": "minimal (use_per_stream_vs + use_vs_rollout_gating)",
            "mech341_sub_flavours": "both (entropy_bonus + stratified_select)",
            "mech341_entropy_bias_scale": MECH341_ENTROPY_BIAS_SCALE,
            "within_class_temperature_by_arm": {
                k: v for k, v in WITHIN_CLASS_T_BY_ARM.items()
            },
            "candidate_summary_source": CANDIDATE_SUMMARY_SOURCE,
            "use_modulatory_selection_authority": USE_MODULATORY_SELECTION_AUTHORITY,
            "modulatory_authority_gain": MODULATORY_AUTHORITY_GAIN,
            "primary_readout": "within_class_rep_cond_entropy_nats (H(rep_signature | committed_class))",
            "secondary_negative_control_readout": "committed_class_entropy_nats (first-action / across-class)",
            "z_goal_enabled": True,
            "drive_weight": 2.0,
            "alpha_world": 0.9,
            "reef_enabled": True,
            "reef_bipartite_layout": True,
            "sd056_amend_active": True,
            "sd056_t1_contrastive_enabled": SD056_T1_CONTRASTIVE_ENABLED,
            "sd056_t1_contrastive_weight": SD056_T1_CONTRASTIVE_WEIGHT,
            "sd056_multistep_contrastive": SD056_MULTISTEP_CONTRASTIVE,
            "sd056_contrastive_horizon": SD056_CONTRASTIVE_HORIZON,
            "sd056_output_norm_clamp": SD056_OUTPUT_NORM_CLAMP,
            "sd056_output_norm_clamp_ratio": SD056_OUTPUT_NORM_CLAMP_RATIO,
            "use_differentiable_cem": "NOT FLIPPED (default False; SD-055 safety note)",
        },
        "result": result,
    }


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
        f"C1={result['acceptance_criteria']['C1_substrate_operative_non_vacuity']} "
        f"C2={result['acceptance_criteria']['C2_within_class_representative_diversity_lift_per_arm']} "
        f"C3={result['acceptance_criteria']['C3_gap_a_first_action_class_readiness']} "
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
