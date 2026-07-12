#!/opt/local/bin/python3
"""V3-EXQ-604c -- Q-044 / MECH-314 sub-flavour ablation on the now-VALIDATED
GAP-A behavioral-diversity substrate (governance-weighted evidence).

SUPERSEDES V3-EXQ-604b (run v3_exq_604b_..._20260607T110114Z, adjudicated
NON_CONTRIBUTORY / substrate_ceiling at /governance cycle
governance-cycle-20260607T1426Z, 2026-06-07; failure_autopsy
gapA-cluster-604b-648a-649 in REE_assembly/evidence/planning/). Same scientific
question (does the MECH-314 structured-curiosity bias change E3 selection, and
which sub-flavours are load-bearing?), corrected substrate + readiness gate ->
alphabetic suffix.

=== WHY 604b WAS NON_CONTRIBUTORY (and what 604c fixes) ===

604b ran 2026-06-07T11:01Z -- BEFORE the behavioral_diversity_isolation:GAP-A
substrate (ARC-065 shared candidate-summary channel; action-conditional candidate
diversity) was validated ready by V3-EXQ-649 (PASS 13:14Z, consumed cand pairwise
spread 0.090 >= 0.05). With the default candidate-source ("proposer"), the K CEM
candidates collapse to a class-uniform pool (cand_world_pairwise_dist ~0 under
monostrategy; V3-EXQ-614e). So the structured-curiosity bias had MAGNITUDE
(last_bias_max_abs=0.0177) but ~ZERO cross-candidate RANGE: a near-uniform per-tick
offset that cannot move the E3 argmin even with selection authority ON. 604b's
readiness gate keyed on bias MAGNITUDE (last_bias_max_abs > floor), the SAME
same-statistic confound that produced four prior nulls (604a / 624a / 614d / 614e /
604b): a uniform offset clears a magnitude floor while its range stays ~0, so the
gate read "ready" and C1 self-routed a (false) null. It was NOT a weakening of
MECH-314/314a/314b/314c/Q-044 (those stay candidate / v3_pending with
pending_retest_after_substrate=true).

604c applies two now-landed corrections (both no-op-default REEConfig flags):
  (1) GAP-A substrate ON. curiosity_candidate_source="e2_world_forward" (the
      MECH-314a Phase-2 amend, 648a-validated) so the per-candidate curiosity
      NOVELTY is sourced from the SD-056-trained action-conditional
      e2.world_forward(z0, a_i) predictions -- the candidate pool the curiosity
      bias consumes now carries action-conditional cross-candidate spread, so the
      bias has genuine RANGE across candidates. candidate_summary_source=
      "e2_world_forward" (the ARC-065 GAP-A shared-channel re-source, 649-validated,
      ree-v3 71dfb2b) is set in lockstep for fidelity to the validated GAP-A fix;
      it feeds the SHARED E3-side bias channels (lateral_pfc / ofc / mech295 /
      gated_policy / tonic_vigor) which are all OFF in this design (structured
      curiosity is the single modulatory channel), so it is inert-but-consistent
      here -- the LOAD-BEARING flag for this experiment is curiosity_candidate_source.
  (2) modulatory-bias-selection-authority ON (use_modulatory_selection_authority,
      643a-validated after the float32 catastrophic-cancellation fix). With the
      now-divergent curiosity bias, the gap-relative authority rescale lets it
      reach the committed argmin.

=== THE LOAD-BEARING READINESS-GATE FIX (the autopsy's named correction) ===

The readiness precondition must assert the cross-candidate RANGE statistic that C1
actually routes on -- NOT the bias magnitude (the magnitude-only gate is the
same-statistic confound behind the four prior nulls). 604c gates on TWO range legs
measured on the ARM_ALL_ON positive control (the arm where curiosity is loudest):
  R1 (consumed-spread): mean cand_world_pairwise_dist >= CANDIDATE_SPREAD_FLOOR
     (0.05; the SAME statistic and floor 649 cleared at 0.090). Post-amend this IS
     the representation the curiosity novelty consumes (curiosity_candidate_source=
     e2_world_forward), so it is a cross-candidate RANGE, causally upstream of the
     bias range.
  R2 (bias-range; the same-statistic gate C1 routes on): mean per-candidate
     curiosity_bias_range_mean >= CURIOSITY_BIAS_RANGE_FLOOR (1e-4; from
     agent._last_score_bias_decomp["curiosity_bias_range_mean"], the
     max-minus-min of the curiosity bias across the K candidates). This is the
     EXACT statistic the authority rescales and the argmin reads -- the leg 604b
     lacked. Below floor -> the bias still collapsed (substrate not ready / source
     not wired) -> substrate_not_ready_requeue, NEVER a MECH-314 falsification.
  R3 (non-vacuity): mean raw_bounded_frac across arms >= BOUNDED_FRAC (fraction of
     P1 ticks with primary e3_raw_score_range_mean < RAW_SCORE_RANGE_BOUND).
     Below -> SD-056 online training numerically unstable (643a explosion lesson)
     and an authority fire on exploded scores would be a degenerate artifact;
     non_contributory.

NOTE the directionality discipline (648a/649 false-flag lesson): R1/R2 are FLOOR
gates (measured >= threshold; default lower direction). R3 is a floor on a FRACTION
(bounded-frac >= 0.9; default lower) -- it is NOT phrased as a "max-magnitude <
ceiling" check, so it carries no upper-bound directionality trap. No precondition
here is named with abs/mean_abs/max_abs/norm/magnitude (the magnitude-vs-range
validator WARN) -- every readiness leg asserts a range/spread/fraction.

=== ARMS (5; identical varying axis to 604a/604b; GAP-A source + authority + clamp ON uniformly) ===

All arms run SD-056 ON (action-divergent candidates, online-trained), the GAP-A
e2.world_forward candidate source ON, the rollout-norm clamp ON (643a stability),
and modulatory authority ON. The single varying axis is structured curiosity:
  ARM_OFF              -- use_structured_curiosity=False (MECH-314 parent anchor)
  ARM_ALL_ON           -- novelty + uncertainty + LP all ON
  ARM_NOVELTY_OFF      -- MECH-314a OFF (314b + 314c ON)
  ARM_UNCERTAINTY_OFF  -- MECH-314b OFF
  ARM_LP_OFF           -- MECH-314c OFF

=== ACCEPTANCE (pre-registered) ===

  C0 (substrate-readiness GUARD): mean cand_world_pairwise_dist > C0_PAIRWISE_DIST_
     FLOOR (0.03) in >= MIN_SEEDS / SEEDS in EACH arm. Confirms SD-056 + GAP-A made
     candidates action-divergent in every arm. C0 FAIL -> non_contributory.
  C1 (MECH-314 PARENT effect; LOAD-BEARING): ARM_ALL_ON selected_action_class_
     entropy differs from ARM_OFF by > DISTINCT_MARGIN (0.03). With the now-divergent
     curiosity bias + authority ON, the curiosity bonus can change selection ->
     supports MECH-314 parent.
  C2 (Q-044 sub-flavour discriminability; informative): at least one sub-flavour
     ablation arm differs from ARM_ALL_ON by > DISTINCT_MARGIN. Identifies which
     sub-flavours are behaviourally load-bearing at the selection level. By the
     MECH-314 Phase-1 honest-scoping caveat, MECH-314a (novelty, per-candidate)
     is the expected discriminator; 314b/314c are broadcast scalars by design.

Overall PASS (evidence supports MECH-314) = readiness_met AND C0 AND C1.
C2 is reported and feeds evidence_direction_per_claim for the sub-flavours; it is
NOT required for overall PASS (a parent-effect with only 314a load-bearing is a
valid, expected outcome that tells governance 314b/314c are not selection-level
load-bearing at Phase 1).

=== EVIDENCE-DIRECTION ROUTING (governance-weighted) ===

  readiness fails OR C0 fails  -> ALL claims non_contributory (test could not let
                                  the claim express; substrate/measurement not ready).
  MECH-314 (parent)            -> supports if C1 else weakens.
  Q-044                        -> supports if C2 (>=1 sub-flavour discriminates) else mixed.
  MECH-314a/b/c                -> supports if that arm's ablation discriminates (in
                                  c2_arms_passed) else mixed.

architecture_epoch: "ree_hybrid_guardrails_v1"

Smoke:
  /opt/local/bin/python3 experiments/v3_exq_604c_q044_mech314_subflavour_ablation_gapa_ready.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiment_protocol import emit_outcome  # noqa: E402
from experiments._lib.arm_fingerprint import (  # noqa: E402
    compute_arm_fingerprint,
    reset_all_rng,
)
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402


EXPERIMENT_TYPE = "v3_exq_604c_q044_mech314_subflavour_ablation_gapa_ready"
QUEUE_ID = "V3-EXQ-604c"
CLAIM_IDS: List[str] = ["MECH-314", "MECH-314a", "MECH-314b", "MECH-314c", "Q-044"]
EXPERIMENT_PURPOSE = "evidence"
SUPERSEDES = "V3-EXQ-604b"

SEEDS = [42, 43, 44]
P0_WARMUP_EPISODES = 60          # mirror 569d/604a/604b/643a/648a extended warmup (E2 contrastive train)
P1_MEASUREMENT_EPISODES = 20
STEPS_PER_EPISODE = 200

DRY_RUN_SEEDS = [42]
DRY_RUN_P0 = 2
DRY_RUN_P1 = 2
DRY_RUN_STEPS = 30

# Acceptance thresholds (pre-registered).
C0_PAIRWISE_DIST_FLOOR = 0.03            # substrate-operative floor (per 569d calibration), EACH arm
DISTINCT_MARGIN = 0.03                   # selected-entropy delta for C1 / C2 (604's margin)
MIN_SEEDS_PER_ARM_FOR_PASS = 2           # of 3

# Readiness preconditions -- the 604c LOAD-BEARING FIX: cross-candidate RANGE
# statistics C1 routes on (NOT the bias-magnitude gate that produced the prior
# nulls). Both R1/R2 are FLOOR gates on the ARM_ALL_ON positive control.
CANDIDATE_SPREAD_FLOOR = 0.05            # R1: consumed cand_world_pairwise_dist (the 649 floor)
CURIOSITY_BIAS_RANGE_FLOOR = 1.0e-4     # R2: per-candidate curiosity_bias_range (same statistic C1 routes on; matches 648a)
RAW_SCORE_RANGE_BOUND = 1e3             # R3: per-tick primary raw_score_range upper bound (counted into a fraction)
BOUNDED_FRAC = 0.9                      # R3: fraction of P1 ticks that must be bounded (a FLOOR on a fraction)

# SD-056 online contrastive training (569d/604a/604b harness) + rollout-norm clamp.
SD056_WEIGHT = 0.05                      # 569d ARM_2 operative weight
E2_CONTRASTIVE_LR = 1e-3
E2_TRAIN_EVERY_K_TICKS = 1
CONTRASTIVE_BATCH_K = 8
TRANSITION_BUFFER_MAX = 256
MIN_BUFFER_BEFORE_TRAIN = 16
MIN_CLASSES_FOR_TRAIN = 2
MAX_GRAD_NORM = 1.0
ROLLOUT_CLAMP_RATIO = 2.0               # 604b/643a: bound rollout z_world norm vs initial (SD-056 amend)

# modulatory-bias-selection-authority (VALIDATED 643a) -- ON uniformly across arms.
MODULATORY_AUTHORITY_GAIN = 0.5
MODULATORY_AUTHORITY_MIN_RANGE_FLOOR = 1e-6

# Curiosity magnitudes (604 calibration; EXQ-573 elevated regime).
CURIOSITY_BIAS_SCALE = 0.5
CURIOSITY_WEIGHT = 0.25

# ENV identical to V3-EXQ-604a / 604b / 569d / 643a so manifest-comparability holds.
# Hazards present so the residue field is non-empty -> MECH-314a residue-source
# novelty has an active comparison set (the GAP-A e2.world_forward candidate source
# then makes the per-candidate distances to those centres genuinely differ).
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


# ---------------------------------------------------------------------------
# Arm definitions (identical varying axis to 604a/604b)
# ---------------------------------------------------------------------------

ARMS: List[Dict[str, Any]] = [
    {
        "arm_id": "ARM_OFF",
        "label": "structured_curiosity_off_parent_anchor",
        "use_structured_curiosity": False,
        "use_curiosity_novelty": False,
        "use_curiosity_uncertainty": False,
        "use_curiosity_learning_progress": False,
    },
    {
        "arm_id": "ARM_ALL_ON",
        "label": "novelty_uncertainty_lp_all_on",
        "use_structured_curiosity": True,
        "use_curiosity_novelty": True,
        "use_curiosity_uncertainty": True,
        "use_curiosity_learning_progress": True,
    },
    {
        "arm_id": "ARM_NOVELTY_OFF",
        "label": "mech314a_off",
        "use_structured_curiosity": True,
        "use_curiosity_novelty": False,
        "use_curiosity_uncertainty": True,
        "use_curiosity_learning_progress": True,
    },
    {
        "arm_id": "ARM_UNCERTAINTY_OFF",
        "label": "mech314b_off",
        "use_structured_curiosity": True,
        "use_curiosity_novelty": True,
        "use_curiosity_uncertainty": False,
        "use_curiosity_learning_progress": True,
    },
    {
        "arm_id": "ARM_LP_OFF",
        "label": "mech314c_off",
        "use_structured_curiosity": True,
        "use_curiosity_novelty": True,
        "use_curiosity_uncertainty": True,
        "use_curiosity_learning_progress": False,
    },
]


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(seed=seed, **ENV_KWARGS)


def _make_agent(env: CausalGridWorldV2, arm: Dict[str, Any]) -> REEAgent:
    """Main-path SP-CEM + V_s + SD-056 stack with the now-VALIDATED GAP-A
    candidate source (e2.world_forward) + the modulatory-bias-selection-authority
    substrate ON (so the now action-conditionally-divergent curiosity bias can
    change the committed argmin) + the SD-056 rollout-norm clamp ON (so primary
    scores stay bounded and the authority fire is non-vacuous). All three are
    uniform across every arm -- the single varying axis is the structured-curiosity
    sub-flavour set. MECH-341 (E3 score diversity) and every other modulatory
    channel are OFF so curiosity is the only modulatory channel feeding the
    authority rescale (clean per-sub-flavour attribution).
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
        # ARC-065 SP-CEM (Layer A) -- main-path default (also in 604a/604b)
        use_support_preserving_cem=True,
        support_preserving_stratified_elites=True,
        support_preserving_ao_std_floor=0.2,
        support_preserving_min_first_action_classes=2,
        # Layer B / C OFF (structured curiosity is the single varying axis)
        use_e3_score_diversity=False,
        use_noise_floor=False,
        use_dacc=False,
        # V_s substrate (main-path default; identical across arms)
        use_per_stream_vs=True,
        use_per_region_vs=True,
        use_event_segmenter=True,
        use_invalidation_trigger=True,
        use_anchor_sets=True,
        # SD-056 -- uniform ON across all arms + ROLLOUT CLAMP (keeps the
        # online-trained E2 rollouts and primary E3 scores bounded).
        e2_action_contrastive_enabled=True,
        e2_action_contrastive_weight=SD056_WEIGHT,
        e2_rollout_output_norm_clamp_enabled=True,
        e2_rollout_output_norm_clamp_ratio=ROLLOUT_CLAMP_RATIO,
        # --- GAP-A substrate (the 604c load-bearing source fix) ---
        # curiosity_candidate_source feeds the per-candidate novelty (314a) from the
        # SD-056-trained action-conditional e2.world_forward(z0,a_i) predictions
        # (648a-validated; MECH-314a Phase-2 amend) -> the curiosity bias carries
        # cross-candidate RANGE. candidate_summary_source is the ARC-065 GAP-A
        # shared-channel re-source (649-validated, ree-v3 71dfb2b); inert here (the
        # shared E3-side channels are OFF) but set in lockstep for fidelity to the
        # validated GAP-A fix.
        curiosity_candidate_source="e2_world_forward",
        candidate_summary_source="e2_world_forward",
        # MECH-314 structured curiosity -- the varying axis
        use_structured_curiosity=bool(arm["use_structured_curiosity"]),
        use_curiosity_novelty=bool(arm["use_curiosity_novelty"]),
        use_curiosity_uncertainty=bool(arm["use_curiosity_uncertainty"]),
        use_curiosity_learning_progress=bool(arm["use_curiosity_learning_progress"]),
        curiosity_bias_scale=CURIOSITY_BIAS_SCALE,
        curiosity_novelty_weight=CURIOSITY_WEIGHT,
        curiosity_uncertainty_weight=CURIOSITY_WEIGHT,
        curiosity_learning_progress_weight=CURIOSITY_WEIGHT,
        # modulatory-bias-selection-authority -- VALIDATED (643a); ON uniformly so
        # the curiosity bias has competitive, bounded authority at E3.select.
        use_modulatory_selection_authority=True,
        modulatory_authority_gain=MODULATORY_AUTHORITY_GAIN,
        modulatory_authority_min_range_floor=MODULATORY_AUTHORITY_MIN_RANGE_FLOOR,
    )
    agent = REEAgent(cfg)
    # Enable per-channel score-bias decomposition so select_action records the
    # per-candidate curiosity_bias_range_mean (max-minus-min across K) that the
    # R2 readiness leg and C1 both route on (the V3-EXQ-571 / 648a pattern).
    agent.e3.e3_score_decomp_enabled = True
    return agent


# ---------------------------------------------------------------------------
# Per-tick measurement helpers (verbatim from V3-EXQ-569d / 604a / 604b)
# ---------------------------------------------------------------------------

def _trajectory_first_action_class(traj) -> int:
    return int(
        traj.actions[:, 0, :].argmax(dim=-1).detach().reshape(-1)[0].item()
    )


def _first_actions_K(candidates) -> torch.Tensor:
    rows = []
    for traj in candidates:
        rows.append(traj.actions[:, 0, :].detach().reshape(-1))
    return torch.stack(rows, dim=0)


def _entropy_from_counts(counts: Dict[int, int]) -> float:
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
    seen_classes: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for tup in pool:
        cls = int(tup[1].argmax().item())
        if cls not in seen_classes:
            seen_classes[cls] = tup
        if len(seen_classes) >= k:
            break
    if len(seen_classes) < MIN_CLASSES_FOR_TRAIN:
        return None
    samples = list(seen_classes.values())
    picked_ids = {id(s) for s in samples}
    for tup in pool:
        if len(samples) >= k:
            break
        if id(tup) in picked_ids:
            continue
        samples.append(tup)
        picked_ids.add(id(tup))
    return samples


def _e2_contrastive_step(
    agent: REEAgent,
    buffer: Deque[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    arm_weight: float,
    optimiser: torch.optim.Optimizer,
    rng: random.Random,
) -> Optional[float]:
    batch = _sample_class_diverse_batch(buffer, CONTRASTIVE_BATCH_K, rng)
    if batch is None:
        return None
    z0_K = torch.stack([t[0] for t in batch]).to(agent.device)
    actions_K = torch.stack([t[1] for t in batch]).to(agent.device)
    z1_K = torch.stack([t[2] for t in batch]).to(agent.device)
    optimiser.zero_grad(set_to_none=True)
    loss = agent.e2.world_forward_contrastive_loss(
        z_world_0=z0_K,
        actions=actions_K,
        z_world_1_targets=z1_K,
        simulation_mode=False,
    )
    if not torch.is_tensor(loss):
        return None
    loss_val = float(loss.detach().item())
    if not math.isfinite(loss_val):
        return loss_val
    if not loss.requires_grad or loss_val == 0.0:
        return loss_val
    weighted = float(arm_weight) * loss
    weighted.backward()
    torch.nn.utils.clip_grad_norm_(agent.e2.parameters(), max_norm=MAX_GRAD_NORM)
    optimiser.step()
    return loss_val


# ---------------------------------------------------------------------------
# Per-(seed, arm) runner
# ---------------------------------------------------------------------------

def _run_seed_arm(
    arm: Dict[str, Any],
    seed: int,
    p0_episodes: int,
    p1_episodes: int,
    steps_per_episode: int,
) -> Dict[str, Any]:
    # Complete RNG reset at cell entry so the cell is order-independent and the
    # arm_fingerprint is a pure function of (substrate, config, seed).
    reset_all_rng(seed)

    env = _make_env(seed)
    agent = _make_agent(env, arm)

    # SD-056 is ON in every arm -> every arm trains E2 online.
    e2_opt: torch.optim.Optimizer = torch.optim.Adam(
        agent.e2.parameters(), lr=E2_CONTRASTIVE_LR
    )

    transition_buffer: Deque[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ] = deque(maxlen=TRANSITION_BUFFER_MAX)
    sample_rng = random.Random(seed)
    n_buffer_appends = 0
    n_contrastive_skipped_nonfinite = 0
    n_contrastive_skipped_sparse = 0

    total_train_eps = p0_episodes + p1_episodes

    pairwise_dists: List[float] = []
    candidate_first_action_counts: Counter = Counter()
    candidate_unique_per_tick: List[float] = []
    candidate_entropy_per_tick: List[float] = []
    selected_class_counts: Counter = Counter()
    contrastive_loss_values: List[float] = []

    # Curiosity-channel diagnostics (P1) -- REAL get_state keys + the per-candidate
    # bias RANGE from agent._last_score_bias_decomp (the 604c readiness statistic).
    curiosity_bias_max_abs_values: List[float] = []   # reported only (magnitude; NOT a gate)
    curiosity_bias_range_values: List[float] = []      # R2 + C1 statistic (cross-candidate RANGE)
    curiosity_std_across_K_values: List[float] = []     # secondary range diagnostic
    curiosity_active_residue_centers: List[float] = []
    curiosity_subflavours_fired: List[float] = []

    # modulatory-bias-selection-authority diagnostics (P1).
    n_authority_active = 0
    authority_scale_factor_values: List[float] = []
    authority_range_values: List[float] = []

    # readiness diagnostics (P1).
    raw_score_ranges: List[float] = []
    n_raw_bounded = 0

    n_p0_ticks = 0
    n_p1_ticks = 0
    n_contrastive_steps = 0

    error_note: Optional[str] = None

    for ep in range(total_train_eps):
        is_p1 = ep >= p0_episodes
        phase_label = "P1" if is_p1 else "P0"

        _, obs_dict = env.reset()
        agent.reset()

        z_self_prev: Optional[torch.Tensor] = None
        action_prev: Optional[torch.Tensor] = None
        pending_capture: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        tick_in_ep = 0

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
                if (
                    torch.isfinite(z0_prev).all()
                    and torch.isfinite(a_prev).all()
                    and torch.isfinite(z1_obs).all()
                ):
                    transition_buffer.append((z0_prev, a_prev, z1_obs))
                    n_buffer_appends += 1
                pending_capture = None

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

            if is_p1 and candidates:
                pre_e3_classes = [_trajectory_first_action_class(t) for t in candidates]
                candidate_first_action_counts.update(pre_e3_classes)
                candidate_unique_per_tick.append(float(len(set(pre_e3_classes))))
                cnt: Counter = Counter(pre_e3_classes)
                candidate_entropy_per_tick.append(_entropy_from_counts(dict(cnt)))
                if len(candidates) >= 2:
                    actions_K = _first_actions_K(candidates).to(agent.device)
                    z0 = latent.z_world.detach()
                    with torch.no_grad():
                        # cand_world_pairwise_dist = the spread of the e2.world_forward
                        # predictions = the CONSUMED candidate-novelty representation
                        # (post-GAP-A, curiosity_candidate_source=e2_world_forward).
                        dist = float(
                            agent.e2.cand_world_pairwise_dist(z0, actions_K).item()
                        )
                    if math.isfinite(dist):
                        pairwise_dists.append(dist)

            if agent.goal_state is not None:
                try:
                    energy = float(body[0, 3].item())
                except Exception:
                    energy = 1.0
                drive_level = max(0.0, 1.0 - energy)
                agent.update_z_goal(benefit_exposure=0.0, drive_level=drive_level)

            action = agent.select_action(candidates, ticks)
            if action is None:
                idx = int(np.random.randint(0, env.action_dim))
                action = torch.zeros(1, env.action_dim, device=agent.device)
                action[0, idx] = 1.0
                agent._last_action = action
            if not torch.isfinite(action).all():
                if error_note is None:
                    error_note = (
                        f"non-finite action at arm={arm['arm_id']} seed={seed} "
                        f"phase={phase_label} ep={ep} step={_step}"
                    )
                break

            # ----- P1 measurement (post-select diagnostics) -----
            if is_p1:
                committed_class = int(action[0].argmax().item())
                selected_class_counts[committed_class] += 1
                n_p1_ticks += 1

                # Curiosity channel: REAL get_state magnitude (reported only) +
                # the per-candidate bias RANGE from the score-bias decomposition
                # (the R2 readiness leg + C1 statistic -- a cross-candidate range,
                # NOT a magnitude).
                cur = getattr(agent, "curiosity", None)
                if cur is not None:
                    st = cur.get_state()
                    bmax = st.get("last_bias_max_abs")
                    if bmax is not None and math.isfinite(float(bmax)):
                        curiosity_bias_max_abs_values.append(float(bmax))
                    nrc = st.get("last_n_active_residue_centers")
                    if nrc is not None:
                        curiosity_active_residue_centers.append(float(nrc))
                    nsf = st.get("last_n_subflavours_fired")
                    if nsf is not None:
                        curiosity_subflavours_fired.append(float(nsf))
                decomp = getattr(agent, "_last_score_bias_decomp", {}) or {}
                crange = decomp.get("curiosity_bias_range_mean")
                if crange is not None and math.isfinite(float(crange)):
                    curiosity_bias_range_values.append(float(crange))
                cstd = decomp.get("curiosity_std_across_K")
                if cstd is not None and math.isfinite(float(cstd)):
                    curiosity_std_across_K_values.append(float(cstd))

                # Authority mechanism + non-vacuity diagnostics.
                diag = getattr(agent.e3, "last_score_diagnostics", {}) or {}
                if bool(diag.get("modulatory_authority_active", False)):
                    n_authority_active += 1
                    sf = diag.get("modulatory_authority_scale_factor")
                    if sf is not None and math.isfinite(float(sf)) and float(sf) > 0.0:
                        authority_scale_factor_values.append(float(sf))
                mar = diag.get("modulatory_authority_range")
                if mar is not None and math.isfinite(float(mar)):
                    authority_range_values.append(float(mar))
                rsr = diag.get("e3_raw_score_range_mean")
                if rsr is not None and math.isfinite(float(rsr)):
                    raw_score_ranges.append(float(rsr))
                    if float(rsr) < RAW_SCORE_RANGE_BOUND:
                        n_raw_bounded += 1
            else:
                n_p0_ticks += 1

            if tick_in_ep % E2_TRAIN_EVERY_K_TICKS == 0:
                loss_val = _e2_contrastive_step(
                    agent=agent,
                    buffer=transition_buffer,
                    arm_weight=SD056_WEIGHT,
                    optimiser=e2_opt,
                    rng=sample_rng,
                )
                if loss_val is None:
                    n_contrastive_skipped_sparse += 1
                elif not math.isfinite(loss_val):
                    n_contrastive_skipped_nonfinite += 1
                elif is_p1:
                    contrastive_loss_values.append(loss_val)
                    n_contrastive_steps += 1

            if (
                torch.isfinite(latent.z_world).all()
                and torch.isfinite(action).all()
            ):
                pending_capture = (
                    latent.z_world.detach().reshape(-1).clone(),
                    action.detach().reshape(-1).clone(),
                )

            _, harm_signal, done, info, next_obs_dict = env.step(action)

            with torch.no_grad():
                agent.update_residue(
                    harm_signal=float(harm_signal),
                    world_delta=None,
                    hypothesis_tag=False,
                    owned=True,
                )

            z_self_prev = latent.z_self.detach()
            action_prev = action
            obs_dict = next_obs_dict
            tick_in_ep += 1
            if done:
                break

        if (ep + 1) % 10 == 0 or (ep + 1) == total_train_eps:
            print(
                f"  [train] arm={arm['arm_id']} seed={seed} phase={phase_label} "
                f"ep {ep + 1}/{total_train_eps}",
                flush=True,
            )

    def _mean(xs: List[float], default: float = 0.0) -> float:
        return float(sum(xs) / len(xs)) if xs else default

    def _maxx(xs: List[float], default: float = 0.0) -> float:
        return float(max(xs)) if xs else default

    def _minx(xs: List[float], default: float = 0.0) -> float:
        return float(min(xs)) if xs else default

    selected_action_entropy = _entropy_from_counts(dict(selected_class_counts))
    authority_active_frac = (
        float(n_authority_active) / n_p1_ticks if n_p1_ticks > 0 else 0.0
    )
    raw_bounded_frac = (
        float(n_raw_bounded) / n_p1_ticks if n_p1_ticks > 0 else 0.0
    )

    return {
        "arm_id": arm["arm_id"],
        "label": arm["label"],
        "seed": int(seed),
        "n_p0_ticks": int(n_p0_ticks),
        "n_p1_ticks": int(n_p1_ticks),
        "n_contrastive_steps": int(n_contrastive_steps),
        "n_buffer_appends": int(n_buffer_appends),
        "n_contrastive_skipped_sparse": int(n_contrastive_skipped_sparse),
        "n_contrastive_skipped_nonfinite": int(n_contrastive_skipped_nonfinite),
        "error_note": error_note,
        "cand_world_pairwise_dist_mean": round(_mean(pairwise_dists), 6),
        "cand_world_pairwise_dist_max": round(_maxx(pairwise_dists), 6),
        "cand_world_pairwise_dist_min": round(_minx(pairwise_dists), 6),
        "candidate_first_action_entropy_mean": round(_mean(candidate_entropy_per_tick), 6),
        "candidate_unique_first_action_classes_mean": round(_mean(candidate_unique_per_tick), 6),
        "candidate_first_action_counts": dict(sorted(candidate_first_action_counts.items())),
        "selected_action_class_entropy": round(selected_action_entropy, 6),
        "selected_class_counts": dict(sorted(selected_class_counts.items())),
        "selected_classes_n_unique": int(len(selected_class_counts)),
        # Curiosity channel: magnitude (reported only) + the cross-candidate RANGE.
        "curiosity_bias_max_abs_mean": round(_mean(curiosity_bias_max_abs_values), 6),
        "curiosity_bias_max_abs_peak": round(_maxx(curiosity_bias_max_abs_values), 6),
        "curiosity_bias_range_mean": round(_mean(curiosity_bias_range_values), 8),
        "curiosity_bias_range_peak": round(_maxx(curiosity_bias_range_values), 8),
        "curiosity_std_across_K_mean": round(_mean(curiosity_std_across_K_values), 8),
        "curiosity_active_residue_centers_mean": round(_mean(curiosity_active_residue_centers), 6),
        "curiosity_subflavours_fired_mean": round(_mean(curiosity_subflavours_fired), 6),
        # Authority mechanism + non-vacuity.
        "modulatory_authority_active_frac": round(authority_active_frac, 6),
        "modulatory_authority_scale_factor_mean": round(_mean(authority_scale_factor_values), 6),
        "modulatory_authority_range_mean": round(_mean(authority_range_values), 6),
        "raw_score_range_mean": round(_mean(raw_score_ranges), 6),
        "raw_score_range_max": round(_maxx(raw_score_ranges), 6),
        "raw_bounded_frac": round(raw_bounded_frac, 6),
        "contrastive_loss_mean": round(_mean(contrastive_loss_values), 6),
    }


# ---------------------------------------------------------------------------
# Cross-arm evaluation
# ---------------------------------------------------------------------------

def _arm_rows(rows: List[Dict[str, Any]], arm_id: str) -> List[Dict[str, Any]]:
    return [r for r in rows if r.get("arm_id") == arm_id]


def _n_seeds_above(rows: List[Dict[str, Any]], key: str, floor: float) -> int:
    return sum(1 for r in rows if float(r.get(key, 0.0)) > floor)


def _mean_key(rows: List[Dict[str, Any]], key: str) -> float:
    vals = [float(r.get(key, 0.0)) for r in rows]
    return float(sum(vals) / len(vals)) if vals else 0.0


def _evaluate(arm_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_id = {a["arm_id"]: _arm_rows(arm_results, a["arm_id"]) for a in ARMS}

    # --- Readiness preconditions (the 604c LOAD-BEARING FIX: RANGE not magnitude) ---
    # Positive control = ARM_ALL_ON (the arm where curiosity is loudest).
    # R1: consumed cand_world_pairwise_dist (the e2.world_forward per-candidate
    # spread the curiosity novelty consumes; the SAME statistic 649 cleared at 0.090).
    r1_seeds = _n_seeds_above(
        by_id["ARM_ALL_ON"], "cand_world_pairwise_dist_mean", CANDIDATE_SPREAD_FLOOR
    )
    r1_measured = _mean_key(by_id["ARM_ALL_ON"], "cand_world_pairwise_dist_mean")
    r1_met = r1_seeds >= MIN_SEEDS_PER_ARM_FOR_PASS

    # R2 (same-statistic gate C1 routes on): per-candidate curiosity_bias_range.
    # This is the leg 604b lacked -- a uniform per-tick offset has large magnitude
    # but ~0 range, so a magnitude floor passed while the argmin could not move.
    r2_seeds = _n_seeds_above(
        by_id["ARM_ALL_ON"], "curiosity_bias_range_mean", CURIOSITY_BIAS_RANGE_FLOOR
    )
    r2_measured = _mean_key(by_id["ARM_ALL_ON"], "curiosity_bias_range_mean")
    r2_met = r2_seeds >= MIN_SEEDS_PER_ARM_FOR_PASS

    # R3: primary scores numerically bounded across all arms (non-vacuity; a FLOOR
    # on a fraction -- no upper-bound directionality trap).
    r3_measured = _mean_key(arm_results, "raw_bounded_frac")
    r3_met = r3_measured >= BOUNDED_FRAC

    readiness_met = bool(r1_met and r2_met and r3_met)

    # Reported magnitude (NOT a gate -- kept for contrast with the 604b confound).
    curiosity_magnitude_measured = _mean_key(by_id["ARM_ALL_ON"], "curiosity_bias_max_abs_mean")

    # C0: substrate-readiness -- pairwise_dist > floor in majority of seeds in EACH arm.
    c0_per_arm = {
        aid: _n_seeds_above(rows, "cand_world_pairwise_dist_mean", C0_PAIRWISE_DIST_FLOOR)
        for aid, rows in by_id.items()
    }
    c0_pass = all(n >= MIN_SEEDS_PER_ARM_FOR_PASS for n in c0_per_arm.values())

    e_off = _mean_key(by_id["ARM_OFF"], "selected_action_class_entropy")
    e_all = _mean_key(by_id["ARM_ALL_ON"], "selected_action_class_entropy")
    e_nov = _mean_key(by_id["ARM_NOVELTY_OFF"], "selected_action_class_entropy")
    e_unc = _mean_key(by_id["ARM_UNCERTAINTY_OFF"], "selected_action_class_entropy")
    e_lp = _mean_key(by_id["ARM_LP_OFF"], "selected_action_class_entropy")

    # C1: MECH-314 parent effect -- curiosity ON differs from OFF.
    c1_delta = abs(e_all - e_off)
    c1_pass = c1_delta > DISTINCT_MARGIN

    # C2 (informative): at least one sub-flavour ablation differs from ALL_ON.
    sub_deltas = {
        "ARM_NOVELTY_OFF": abs(e_nov - e_all),
        "ARM_UNCERTAINTY_OFF": abs(e_unc - e_all),
        "ARM_LP_OFF": abs(e_lp - e_all),
    }
    c2_arms_passed = [a for a, d in sub_deltas.items() if d > DISTINCT_MARGIN]
    c2_pass = len(c2_arms_passed) >= 1

    overall_pass = bool(readiness_met and c0_pass and c1_pass)

    # C1 non-degeneracy: the test only meant something if the curiosity bias
    # genuinely varied across candidates (readiness_met -> bias range cleared) AND
    # the authority gate actually fired in ARM_ALL_ON.
    authority_active_all_on = _mean_key(by_id["ARM_ALL_ON"], "modulatory_authority_active_frac")
    c1_non_degenerate = bool(readiness_met and authority_active_all_on > 0.0)

    return {
        # Readiness (RANGE-based; the 604c fix)
        "candidate_spread_floor": CANDIDATE_SPREAD_FLOOR,
        "r1_consumed_spread_seeds_above": r1_seeds,
        "r1_consumed_spread_measured_mean": round(r1_measured, 6),
        "r1_met": bool(r1_met),
        "curiosity_bias_range_floor": CURIOSITY_BIAS_RANGE_FLOOR,
        "r2_bias_range_seeds_above": r2_seeds,
        "r2_bias_range_measured_mean": round(r2_measured, 8),
        "r2_met": bool(r2_met),
        "raw_score_range_bound": RAW_SCORE_RANGE_BOUND,
        "r3_bounded_measured": round(r3_measured, 6),
        "r3_met": bool(r3_met),
        "readiness_met": readiness_met,
        "curiosity_magnitude_measured_mean": round(curiosity_magnitude_measured, 8),  # reported only
        "authority_active_frac_all_on": round(authority_active_all_on, 6),
        # C0
        "c0_floor": C0_PAIRWISE_DIST_FLOOR,
        "c0_per_arm_n_seeds_above": c0_per_arm,
        "c0_min_seeds_required": MIN_SEEDS_PER_ARM_FOR_PASS,
        "c0_pass": bool(c0_pass),
        # C1 / C2
        "selected_entropy_ARM_OFF": round(e_off, 6),
        "selected_entropy_ARM_ALL_ON": round(e_all, 6),
        "selected_entropy_ARM_NOVELTY_OFF": round(e_nov, 6),
        "selected_entropy_ARM_UNCERTAINTY_OFF": round(e_unc, 6),
        "selected_entropy_ARM_LP_OFF": round(e_lp, 6),
        "c1_distinct_margin": DISTINCT_MARGIN,
        "c1_parent_delta_on_vs_off": round(c1_delta, 6),
        "c1_pass": bool(c1_pass),
        "c1_non_degenerate": c1_non_degenerate,
        "c2_sub_flavour_deltas_vs_all_on": {k: round(v, 6) for k, v in sub_deltas.items()},
        "c2_arms_passed": c2_arms_passed,
        "c2_pass": bool(c2_pass),
        # Per-arm channel summaries.
        "pairwise_dist_per_arm_mean": {
            aid: round(_mean_key(rows, "cand_world_pairwise_dist_mean"), 6)
            for aid, rows in by_id.items()
        },
        "curiosity_bias_range_per_arm_mean": {
            aid: round(_mean_key(rows, "curiosity_bias_range_mean"), 8)
            for aid, rows in by_id.items()
        },
        "curiosity_bias_max_abs_per_arm_mean": {
            aid: round(_mean_key(rows, "curiosity_bias_max_abs_mean"), 6)
            for aid, rows in by_id.items()
        },
        "authority_active_frac_per_arm_mean": {
            aid: round(_mean_key(rows, "modulatory_authority_active_frac"), 6)
            for aid, rows in by_id.items()
        },
        "authority_scale_factor_per_arm_mean": {
            aid: round(_mean_key(rows, "modulatory_authority_scale_factor_mean"), 6)
            for aid, rows in by_id.items()
        },
        "overall_pass": overall_pass,
    }


def _evidence_direction_per_claim(summary: Dict[str, Any]) -> Dict[str, str]:
    # readiness/substrate gate: the test could not let the claim express itself.
    if not (summary["readiness_met"] and summary["c0_pass"]):
        return {cid: "non_contributory" for cid in CLAIM_IDS}

    parent = "supports" if summary["c1_pass"] else "weakens"

    def _sub(arm_id: str) -> str:
        return "supports" if arm_id in summary["c2_arms_passed"] else "mixed"

    return {
        "MECH-314": parent,
        "MECH-314a": _sub("ARM_NOVELTY_OFF"),
        "MECH-314b": _sub("ARM_UNCERTAINTY_OFF"),
        "MECH-314c": _sub("ARM_LP_OFF"),
        "Q-044": "supports" if summary["c2_pass"] else "mixed",
    }


def _evidence_direction_overall(summary: Dict[str, Any]) -> str:
    if not (summary["readiness_met"] and summary["c0_pass"]):
        return "non_contributory"
    if summary["overall_pass"]:
        return "supports"
    return "mixed"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    seeds = DRY_RUN_SEEDS if dry_run else SEEDS
    p0 = DRY_RUN_P0 if dry_run else P0_WARMUP_EPISODES
    p1 = DRY_RUN_P1 if dry_run else P1_MEASUREMENT_EPISODES
    steps = DRY_RUN_STEPS if dry_run else STEPS_PER_EPISODE

    arm_results: List[Dict[str, Any]] = []
    for arm in ARMS:
        for seed in seeds:
            print(f"Seed {seed} Condition {arm['arm_id']}", flush=True)
            cell = _run_seed_arm(arm, seed, p0, p1, steps)
            # Arm fingerprint (instrument-only, Phase 0). Online SD-056 training
            # is stateful per cell -> never reuse-eligible.
            cell["arm_fingerprint"] = compute_arm_fingerprint(
                config_slice={
                    "arm": {k: arm[k] for k in (
                        "arm_id", "use_structured_curiosity", "use_curiosity_novelty",
                        "use_curiosity_uncertainty", "use_curiosity_learning_progress")},
                    "env_kwargs": ENV_KWARGS,
                    "sd056_weight": SD056_WEIGHT,
                    "rollout_clamp_ratio": ROLLOUT_CLAMP_RATIO,
                    "modulatory_authority_gain": MODULATORY_AUTHORITY_GAIN,
                    "curiosity_bias_scale": CURIOSITY_BIAS_SCALE,
                    "curiosity_weight": CURIOSITY_WEIGHT,
                    "curiosity_candidate_source": "e2_world_forward",
                    "candidate_summary_source": "e2_world_forward",
                    "p0_episodes": p0, "p1_episodes": p1, "steps_per_episode": steps,
                },
                seed=seed,
                script_path=Path(__file__),
                rng_fully_reset=True,
                extra_ineligible_reasons=["online_e2_training_stateful_per_cell"],
            )
            arm_results.append(cell)
            passed = cell.get("error_note") is None
            print(f"verdict: {'PASS' if passed else 'FAIL'}", flush=True)

    summary = _evaluate(arm_results)
    outcome = "PASS" if summary["overall_pass"] else "FAIL"
    edpc = _evidence_direction_per_claim(summary)
    edir = _evidence_direction_overall(summary)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{timestamp}_v3"

    manifest: Dict[str, Any] = {
        "schema_version": "v1",
        "run_id": run_id,
        "queue_id": QUEUE_ID,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": timestamp,
        "outcome": outcome,
        "result": outcome,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": CLAIM_IDS,
        "supersedes": SUPERSEDES,
        "evidence_direction": edir,
        "evidence_direction_per_claim": edpc,
        "evidence_direction_note": (
            "MECH-314 structured-curiosity sub-flavour ablation on the now-VALIDATED "
            "GAP-A behavioral-diversity substrate (ARC-065 shared candidate channel + "
            "MECH-314a Phase-2 e2.world_forward candidate source, validated by "
            "V3-EXQ-649 PASS + V3-EXQ-648a; consumed candidate spread cleared the 0.05 "
            "floor at 0.090) plus the modulatory-bias-selection-authority gate "
            "(V3-EXQ-643a PASS after the float32 cancellation fix). Supersedes "
            "V3-EXQ-604b, adjudicated NON_CONTRIBUTORY/substrate_ceiling: 604b ran "
            "2026-06-07T11:01Z BEFORE GAP-A validation, so the class-uniform candidate "
            "pool left the curiosity bias with magnitude (last_bias_max_abs=0.0177) but "
            "~zero cross-candidate RANGE -- a uniform per-tick offset that cannot move "
            "the E3 argmin even with authority ON. THE LOAD-BEARING FIX: the readiness "
            "gate now checks the cross-candidate RANGE statistic C1 routes on (R1 "
            "consumed cand_world_pairwise_dist >= 0.05; R2 per-candidate "
            "curiosity_bias_range >= 1e-4 from the score-bias decomposition) instead of "
            "bias magnitude (last_bias_max_abs) -- the magnitude-only gate was the "
            "same-statistic confound behind four prior nulls (604a/624a/614d/614e/604b). "
            "R3 (raw_bounded_frac >= 0.9) keeps the authority fire non-vacuous. Overall "
            "PASS = readiness AND C0 (per-arm candidate divergence) AND C1 (ARM_ALL_ON "
            "vs ARM_OFF selected-entropy delta > 0.03). C2 identifies which sub-flavours "
            "are selection-level load-bearing (314a/novelty expected; 314b/c broadcast "
            "scalars by design). Readiness-below-floor self-routes to "
            "substrate_not_ready_requeue, NEVER a MECH-314 falsification. claim_ids "
            "stay candidate / v3_pending with pending_retest_after_substrate=true; not "
            "weakened by this readiness gate."
        ),
        "interpretation": {
            "label": (
                "substrate_not_ready_requeue" if not summary["readiness_met"] else (
                    "curiosity_authority_supports_mech314" if summary["overall_pass"]
                    else "curiosity_authority_does_not_change_selection"
                )
            ),
            "preconditions": [
                {
                    "name": "consumed_candidate_spread_supra_floor",
                    "kind": "readiness",
                    "description": (
                        "ARM_ALL_ON mean cand_world_pairwise_dist (the e2.world_forward "
                        "per-candidate spread; post-GAP-A this IS the representation the "
                        "curiosity novelty consumes) clears the floor -- a cross-candidate "
                        "RANGE, the SAME statistic and floor V3-EXQ-649 cleared at 0.090."
                    ),
                    "control": (
                        "ARM_ALL_ON (all three sub-flavours on), P1 ticks; positive "
                        "control = the arm where the curiosity channel is loudest; "
                        "curiosity_candidate_source=e2_world_forward (SD-056-trained)"
                    ),
                    "measured": round(summary["r1_consumed_spread_measured_mean"], 6),
                    "threshold": CANDIDATE_SPREAD_FLOOR,
                    "met": bool(summary["r1_met"]),
                },
                {
                    "name": "curiosity_bias_range_supra_floor",
                    "kind": "readiness",
                    "description": (
                        "604c LOAD-BEARING same-statistic gate: ARM_ALL_ON mean "
                        "per-candidate curiosity_bias_range (max-minus-min of the "
                        "curiosity bias across the K candidates, from the score-bias "
                        "decomposition) -- the EXACT cross-candidate RANGE C1 routes on "
                        "and the authority rescales -- clears the floor. This is the leg "
                        "604b LACKED: 604b gated on bias MAGNITUDE (last_bias_max_abs), "
                        "and a uniform per-tick offset has large magnitude but ~0 range, "
                        "so the gate read READY while the argmin could not move "
                        "(same-statistic confound behind 604a/624a/614d/614e/604b). "
                        "Below-floor here -> the bias still collapsed (substrate not "
                        "ready / source not wired) -> substrate_not_ready_requeue, NOT a "
                        "MECH-314 falsification."
                    ),
                    "control": (
                        "ARM_ALL_ON positive control with curiosity_candidate_source="
                        "e2_world_forward; the SAME range statistic the load-bearing C1 "
                        "consumes (not a magnitude / mean_abs proxy)"
                    ),
                    "measured": round(summary["r2_bias_range_measured_mean"], 8),
                    "threshold": CURIOSITY_BIAS_RANGE_FLOOR,
                    "met": bool(summary["r2_met"]),
                },
                {
                    "name": "primary_scores_bounded_fraction_supra_floor",
                    "kind": "readiness",
                    "description": (
                        "fraction of P1 ticks (across all arms) with primary "
                        f"e3_raw_score_range_mean < {RAW_SCORE_RANGE_BOUND} -- non-vacuity: "
                        "an authority fire on exploded SD-056-online scores is a "
                        "degenerate artifact, not a real effect. A FLOOR on a fraction "
                        "(measured >= threshold), so no upper-bound directionality trap."
                    ),
                    "control": "SD-056 rollout-norm clamp enabled (e2_rollout_output_norm_clamp_enabled, 643a stability)",
                    "measured": round(summary["r3_bounded_measured"], 6),
                    "threshold": BOUNDED_FRAC,
                    "met": bool(summary["r3_met"]),
                },
            ],
            "criteria_non_degenerate": {
                "C0": bool(summary["c0_pass"]),
                "C1": bool(summary["c1_non_degenerate"]),
                "C2": bool(summary["c0_pass"] and summary["readiness_met"]),
            },
            "criteria": [
                {
                    "name": "C1_mech314_parent_effect_authority_on",
                    "load_bearing": True,
                    "passed": bool(summary["c1_pass"] and summary["readiness_met"]),
                },
            ],
        },
        "dry_run": bool(dry_run),
        "config": {
            "seeds": seeds,
            "p0_warmup_episodes": p0,
            "p1_measurement_episodes": p1,
            "steps_per_episode": steps,
            "env_kwargs": ENV_KWARGS,
            "arms": [{"arm_id": a["arm_id"], "label": a["label"],
                       "use_structured_curiosity": a["use_structured_curiosity"],
                       "use_curiosity_novelty": a["use_curiosity_novelty"],
                       "use_curiosity_uncertainty": a["use_curiosity_uncertainty"],
                       "use_curiosity_learning_progress": a["use_curiosity_learning_progress"]}
                      for a in ARMS],
            "curiosity_candidate_source": "e2_world_forward",
            "candidate_summary_source": "e2_world_forward",
            "sd056_weight": SD056_WEIGHT,
            "rollout_clamp_ratio": ROLLOUT_CLAMP_RATIO,
            "modulatory_authority_gain": MODULATORY_AUTHORITY_GAIN,
            "modulatory_authority_min_range_floor": MODULATORY_AUTHORITY_MIN_RANGE_FLOOR,
            "curiosity_bias_scale": CURIOSITY_BIAS_SCALE,
            "curiosity_weight": CURIOSITY_WEIGHT,
            "c0_pairwise_dist_floor": C0_PAIRWISE_DIST_FLOOR,
            "distinct_margin": DISTINCT_MARGIN,
            "min_seeds_per_arm_for_pass": MIN_SEEDS_PER_ARM_FOR_PASS,
            "candidate_spread_floor": CANDIDATE_SPREAD_FLOOR,
            "curiosity_bias_range_floor": CURIOSITY_BIAS_RANGE_FLOOR,
            "raw_score_range_bound": RAW_SCORE_RANGE_BOUND,
            "bounded_frac": BOUNDED_FRAC,
            "e2_contrastive_lr": E2_CONTRASTIVE_LR,
        },
        "acceptance_criteria": {
            "readiness_met": summary["readiness_met"],
            "C0_substrate_readiness_candidates_divergent": summary["c0_pass"],
            "C1_mech314_parent_effect_on_vs_off": summary["c1_pass"],
            "C2_subflavour_discriminability": summary["c2_pass"],
            "overall_pass": summary["overall_pass"],
        },
        "summary": summary,
        "arm_results": arm_results,
    }

    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.json"

    if not dry_run:
        out_path = write_flat_manifest(
            manifest,
            out_dir,
            dry_run=False,
            config=manifest.get("config"),
            seeds=SEEDS,
            script_path=Path(__file__),
        )
        print(f"Manifest written: {out_path}", flush=True)
    else:
        out_path = Path("/dev/null")
        print("Dry run -- manifest not written.", flush=True)

    print(f"Outcome: {outcome} (label={manifest['interpretation']['label']})", flush=True)
    for k, v in manifest["acceptance_criteria"].items():
        print(f"  {k}: {v}", flush=True)

    manifest["manifest_path"] = str(out_path)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="V3-EXQ-604c MECH-314 sub-flavour ablation on the validated "
                    "GAP-A substrate + modulatory-bias-selection-authority (supersedes 604b)"
    )
    parser.add_argument("--dry-run", action="store_true", help="Short smoke run.")
    args = parser.parse_args()

    result = run_experiment(dry_run=args.dry_run)

    if args.dry_run:
        sys.exit(0)

    _outcome_raw = str(result.get("outcome", "FAIL")).upper()
    _outcome = _outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL"
    emit_outcome(
        outcome=_outcome,
        manifest_path=str(result.get("manifest_path", Path("/dev/null"))),
    )
    sys.exit(0)
