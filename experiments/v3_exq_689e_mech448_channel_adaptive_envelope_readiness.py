"""V3-EXQ-689e -- MECH-448 / ARC-107 CHANNEL-ADAPTIVE eligibility-floor readiness.

SUBSTRATE-READINESS DIAGNOSTIC (experiment_purpose=diagnostic, claim_ids=[],
PROMOTES NOTHING). Validates the channel-adaptive (mean-relative) eligibility
floor amend landed this session on the MECH-448 rank-preserving F->eligibility
demotion lever.

THE PROBLEM (confirmed twice). The MECH-448 demotion envelope thresholds each
candidate's share of the competing F-merit against an ABSOLUTE floor
(f_eligibility_envelope_floor=0.30). That floor was tuned to PASS on the GAP-A
foraging bank (V3-EXQ-689d). Each downstream channel has a DIFFERENT F-merit
distribution, so the same fixed floor mis-fires:
  - V3-EXQ-654h (arc_062 rule-apprehension): every per-candidate share fell below
    0.30 -> the envelope admitted ALL candidates -> f_eligibility_excluded_count==0,
    an all-admit NO-OP (the lever never engaged).
  - V3-EXQ-485i -> 485j (OFC): needed a bespoke per-seed envelope-floor
    recalibration to engage; 485j then confirmed OFC discrimination CONVERTS under
    demotion -- the lever generalises off GAP-A; the residual was a separate
    devaluation test-design gap (485k), NOT the envelope.

THE FIX (landed this session, no-op default, bit-identical OFF). A new E3Config
flag use_f_eligibility_adaptive_floor replaces the fixed absolute floor with a
MEAN-RELATIVE one inside e3_selector._f_eligibility_envelope:
    floor = f_eligibility_adaptive_mean_factor * elig.mean()
A candidate is eligible iff its share of the competing merit exceeds mean_factor
(default 1.0) times the field's OWN mean share. Mean-relative is SCALE-INVARIANT
(auto-calibrates per channel -- no per-channel hand-tuning), CONFLICT-GRADED
(a decisive winner pulls the mean up so others fall below -> narrow; a near-tie
sits near the mean -> wide), and RANK-PRESERVING (still a threshold on elig
[monotone in merit] -> the eligible set stays an F-rank prefix). On any
non-uniform field at least one candidate is below the mean share, so excluded>0
by construction. The goal: collapse the ~5 per-channel hand-floor dances
(654h/485i/485j + pending 625/445/687) into ONE global knob.

WHAT THIS PROBES. f_eligibility_excluded_count > 0 lands in a PRODUCTIVE range
(1 <= envelope_size < K AND excluded_count > 0) across >= 2 REAL e3.select()
channel substrates with genuinely different F-merit distributions, driven by the
SAME global adaptive config (NO per-channel hand-tuning), where the FIXED 0.30
floor mis-fires; bit-identical OFF as the negative control. This is an
ENVELOPE-ENGAGEMENT readiness probe, NOT the full behavioural falsifier -- it
does not score the downstream conversion DVs; it reads the f_eligibility_*
diagnostics off the live select tick.

CHANNELS (2 real e3.select paths, each a GAP-A conversion stack -- SP-CEM +
SD-056 online + candidate_summary_source=e2_world_forward + route-range routing +
authority + shortlist-then-modulate top_k -- so the candidate pool is divergent,
the non-vacuity precondition; the per-channel modulatory head + env regime
present genuinely different F-merit distributions to the envelope):
  CHANNEL_A arc062_rule_apprehension -- lateral_pfc + gated_policy + mech295
    liking-bridge modulatory channel on the busy reef-bipartite foraging env
    (the 654h regime: spread / near-uniform competing merit -> the all-admit
    signature under the fixed 0.30 floor).
  CHANNEL_B ofc_outcome_value -- SD-033b OFC analog (ofc_train_state_bias_head,
    ofc_harm_dim>0) on a sharper-contrast foraging env (the 485i regime: a
    different F-merit spread that needed bespoke floor recalibration).

ARMS (the f_eligibility config axis, identical across both channels):
  ARM_OFF      use_f_eligibility_demotion=False -- the demotion block is skipped
               entirely (negative control: the lever absent; demotion never
               fires, the adaptive flag's presence perturbs nothing).
  ARM_FIXED    use_f_eligibility_demotion=True, adaptive OFF, floor=0.30 -- the
               689d config (expected to REPRODUCE the 654h no-op on CHANNEL_A).
  ARM_ADAPTIVE use_f_eligibility_demotion=True, adaptive ON, mean_factor=1.0 --
               the SAME global config on BOTH channels (expected excluded>0,
               productive, on BOTH).

INTERPRETATION GRID (self-routing):
| condition                                              | label                                    | direction        | next                                                |
|--------------------------------------------------------|------------------------------------------|------------------|-----------------------------------------------------|
| pool not divergent OR demotion never fires (a channel) | substrate_not_ready_requeue              | non_contributory | re-queue at a longer P0 warmup; NOT a verdict       |
| READINESS met, rank_preserving frac < 1.0              | rank_alteration_not_prefix_diagnose      | non_contributory | impl/design fault -> /diagnose-errors; NOT a verdict|
| READINESS met, ADAPTIVE still no-ops on a channel      | adaptive_floor_still_no_ops_requeue      | non_contributory | mean-relative insufficient -> mean_factor sweep /   |
|   (excluded_count==0 despite a divergent pool)         |                                          |                  | a different adaptive mode; NOT a falsification       |
| READINESS + rank met, ADAPTIVE excluded>0 productive   | adaptive_floor_engages_across_channels   | non_contributory | PASS -> the channel-adaptive floor is ready; the    |
|   on BOTH channels with the same global config         |                                          |                  | per-channel hand-floor sweeps retire                |

PROMOTES NOTHING: claim_ids=[]; MECH-448 stays candidate; ARC-107 untouched.

Usage:
  /opt/local/bin/python3 experiments/v3_exq_689e_mech448_channel_adaptive_envelope_readiness.py --dry-run
"""

import argparse
import json
import math
import random
import sys
from collections import Counter, deque
from datetime import datetime
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
from _metrics import check_degeneracy  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_689e_mech448_channel_adaptive_envelope_readiness"
QUEUE_ID = "V3-EXQ-689e"
SUPERSEDES: Optional[str] = None
CLAIM_IDS: List[str] = []  # substrate-readiness diagnostic; PROMOTES NOTHING
EXPERIMENT_PURPOSE = "diagnostic"

SEEDS = [42, 43, 44]
P0_WARMUP_EPISODES = 30           # SD-056 contrastive warmup -> divergent pool
P1_MEASUREMENT_EPISODES = 10
STEPS_PER_EPISODE = 150

DRY_RUN_SEEDS = [42]
DRY_RUN_P0 = 2
DRY_RUN_P1 = 2
DRY_RUN_STEPS = 30

# Pre-registered thresholds.
PAIRWISE_DIST_FLOOR = 0.03        # READINESS: ADAPTIVE arm e2.world_forward pool spread (range-kind)
DEMOTION_ACTIVE_FRAC_FLOOR = 0.8  # READINESS: fraction of demotion-arm P1 ticks with demotion active
EXCLUDED_GT0_FRAC_FLOOR = 0.5     # PRIMARY: fraction of ADAPTIVE demotion-active ticks with excluded>0
PRODUCTIVE_FRAC_FLOOR = 0.5       # PRIMARY: fraction of ADAPTIVE demotion-active ticks with 1<=size<K
RANK_PRESERVING_FRAC_REQUIRED = 1.0  # GUARDRAIL: every demotion-active tick rank-preserving
MIN_SEEDS_FOR_PASS = 2            # of 3

# MECH-448 lever config (ARC-107). Shared 689d-style; NO per-channel calibration.
F_ELIGIBILITY_ENVELOPE_FLOOR = 0.30   # fixed absolute DN-share floor (the 689d/ARM_FIXED config)
F_ELIGIBILITY_DN_SIGMA = 0.0
F_ELIGIBILITY_ADAPTIVE_MEAN_FACTOR = 1.0  # the SINGLE global adaptive knob (above-mean-share)

# Shared GAP-A conversion constant (ON all cells; the within-eligible arbitration).
MODULATORY_SHORTLIST_K = 3
MODULATORY_SHORTLIST_MODE = "top_k"
MODULATORY_AUTHORITY_GAIN = 2.0
MODULATORY_AUTHORITY_NORMALIZE_BASIS = "std"
MODULATORY_ROUTE_MIN_RANGE_FLOOR = 1e-6

# SD-056 online contrastive training (mirror 689d harness).
SD056_WEIGHT = 0.05
E2_CONTRASTIVE_LR = 1e-3
E2_TRAIN_EVERY_K_TICKS = 1
CONTRASTIVE_BATCH_K = 8
TRANSITION_BUFFER_MAX = 256
MIN_BUFFER_BEFORE_TRAIN = 16
MIN_CLASSES_FOR_TRAIN = 2
MAX_GRAD_NORM = 1.0

# OFC head (CHANNEL_B), mirroring V3-EXQ-485i.
OFC_STATE_DIM = 16
OFC_BIAS_SCALE = 0.1
Z_HARM_DIM = 32

# --- Env regimes (two genuinely different F-merit distributions) ---
# CHANNEL_A: busy reef-bipartite foraging (the 654h GAP-A bank: spread / near-uniform
# competing merit -> the all-admit signature under the fixed 0.30 floor).
ENV_KWARGS_A: Dict[str, Any] = dict(
    size=12, num_hazards=4, num_resources=5, hazard_harm=0.05,
    env_drift_interval=5, env_drift_prob=0.1,
    proximity_harm_scale=0.1, proximity_benefit_scale=0.05,
    proximity_approach_threshold=0.2, hazard_field_decay=0.5,
    resource_respawn_on_consume=True, toroidal=False, harm_history_len=10,
    reef_enabled=True, n_reef_patches=3, reef_patch_radius=2,
    hazard_food_attraction=0.7, reef_bipartite_layout=True,
    reef_bipartite_axis="horizontal", reef_bipartite_agent_band_radius=1,
)
# CHANNEL_B: sharper-contrast foraging (fewer hazards, higher harm -> a more
# decisive F-merit spread = a DIFFERENT channel F-distribution).
ENV_KWARGS_B: Dict[str, Any] = dict(
    size=12, num_hazards=2, num_resources=3, hazard_harm=0.1,
    env_drift_interval=5, env_drift_prob=0.1,
    proximity_harm_scale=0.2, proximity_benefit_scale=0.05,
    proximity_approach_threshold=0.2, hazard_field_decay=0.5,
    resource_respawn_on_consume=True, toroidal=False, harm_history_len=10,
    reef_enabled=True, n_reef_patches=3, reef_patch_radius=2,
    hazard_food_attraction=0.7, reef_bipartite_layout=True,
    reef_bipartite_axis="horizontal", reef_bipartite_agent_band_radius=1,
)

# --- Channels (modulatory stack + env regime) ---
CHANNELS: List[Dict[str, Any]] = [
    {
        "channel_id": "CHANNEL_A_arc062_rule_apprehension",
        "label": "arc_062 rule-apprehension bank (the 654h all-admit no-op regime)",
        "env_kwargs": ENV_KWARGS_A,
        "modulatory": "arc062",
    },
    {
        "channel_id": "CHANNEL_B_ofc_outcome_value",
        "label": "SD-033b OFC outcome-value bank (the 485i bespoke-recalibration regime)",
        "env_kwargs": ENV_KWARGS_B,
        "modulatory": "ofc",
    },
]

# --- Arms (the f_eligibility config axis; identical across channels) ---
OFF_ARM = "ARM_OFF"
FIXED_ARM = "ARM_FIXED"
ADAPTIVE_ARM = "ARM_ADAPTIVE"
ARMS: List[Dict[str, Any]] = [
    {
        "arm_id": OFF_ARM,
        "label": "demotion OFF (negative control; lever absent)",
        "use_f_eligibility_demotion": False,
        "use_f_eligibility_adaptive_floor": False,
    },
    {
        "arm_id": FIXED_ARM,
        "label": "demotion ON, FIXED absolute floor 0.30 (the 689d config)",
        "use_f_eligibility_demotion": True,
        "use_f_eligibility_adaptive_floor": False,
    },
    {
        "arm_id": ADAPTIVE_ARM,
        "label": "demotion ON, ADAPTIVE mean-relative floor (one global config)",
        "use_f_eligibility_demotion": True,
        "use_f_eligibility_adaptive_floor": True,
    },
]


def _make_env(channel: Dict[str, Any], seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(seed=seed, **channel["env_kwargs"])


def _make_agent(
    env: CausalGridWorldV2, channel: Dict[str, Any], arm: Dict[str, Any]
) -> REEAgent:
    """GAP-A conversion stack (SP-CEM + SD-056 online + e2_world_forward summary
    source + route-range routing + authority + shortlist-then-modulate top_k) with
    a per-channel modulatory head and the per-arm f_eligibility config. The ONLY
    axis swept across arms is the f_eligibility config; the ONLY axis across
    channels is the modulatory head + env regime (-> the F-merit distribution)."""
    is_arc062 = channel["modulatory"] == "arc062"
    is_ofc = channel["modulatory"] == "ofc"
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=32,
        alpha_world=0.9,
        alpha_self=0.3,
        use_harm_stream=True,
        z_harm_dim=Z_HARM_DIM,
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
        # ARC-065 SP-CEM (Layer A) -- main-path action-divergent pool.
        use_support_preserving_cem=True,
        support_preserving_stratified_elites=True,
        support_preserving_ao_std_floor=0.2,
        support_preserving_min_first_action_classes=2,
        # CHANNEL_A modulatory head: arc_062 rule-apprehension (lateral_pfc +
        # gated_policy) + mech295 liking-bridge (guaranteed-live modulatory channel).
        use_lateral_pfc_analog=is_arc062,
        use_gated_policy=is_arc062,
        use_mech295_liking_bridge=is_arc062,
        # CHANNEL_B modulatory head: SD-033b OFC analog (485i-faithful).
        use_ofc_analog=is_ofc,
        ofc_state_dim=OFC_STATE_DIM,
        ofc_harm_dim=(Z_HARM_DIM if is_ofc else 0),
        ofc_bias_scale=OFC_BIAS_SCALE,
        ofc_train_state_bias_head=is_ofc,
        # Other policy-layer regulators / CRF stack OFF (the envelope is the axis).
        use_structured_curiosity=False,
        use_e3_score_diversity=False,
        use_noise_floor=False,
        use_tonic_vigor=False,
        use_dacc=False,
        use_candidate_rule_field=False,
        # SD-056 substrate trained online on every cell (e2.world_forward divergence).
        e2_action_contrastive_enabled=True,
        e2_action_contrastive_weight=SD056_WEIGHT,
        e2_rollout_output_norm_clamp_enabled=True,
        e2_rollout_output_norm_clamp_ratio=2.0,
        # ARC-065 GAP-A: divergent eligible set (the non-vacuity precondition).
        candidate_summary_source="e2_world_forward",
        # Shared route-range routing + authority + shortlist-then-modulate constant.
        use_modulatory_channel_routing=True,
        modulatory_channel_route_source="cand_world_summary",
        modulatory_channel_route_weight=1.0,
        modulatory_channel_route_min_range_floor=MODULATORY_ROUTE_MIN_RANGE_FLOOR,
        use_modulatory_selection_authority=True,
        modulatory_authority_gain=MODULATORY_AUTHORITY_GAIN,
        modulatory_authority_normalize_basis=MODULATORY_AUTHORITY_NORMALIZE_BASIS,
        use_modulatory_shortlist_then_modulate=True,
        modulatory_shortlist_mode=MODULATORY_SHORTLIST_MODE,
        modulatory_shortlist_k=MODULATORY_SHORTLIST_K,
        # MECH-439 conflict-grade levers OFF (not the family under test).
        modulatory_shortlist_conflict_graded=False,
        use_gap_scaled_commit_temperature=False,
        # --- MECH-448 (ARC-107) f_eligibility demotion + channel-adaptive floor ---
        use_f_eligibility_demotion=bool(arm["use_f_eligibility_demotion"]),
        f_eligibility_envelope_floor=F_ELIGIBILITY_ENVELOPE_FLOOR,
        f_eligibility_dn_sigma=F_ELIGIBILITY_DN_SIGMA,
        use_f_eligibility_adaptive_floor=bool(arm["use_f_eligibility_adaptive_floor"]),
        f_eligibility_adaptive_mean_factor=F_ELIGIBILITY_ADAPTIVE_MEAN_FACTOR,
    )
    cfg.e3.commitment_threshold = 0.5
    return REEAgent(cfg)


# ---------------------------------------------------------------------------
# Measurement helpers (ported from V3-EXQ-689d)
# ---------------------------------------------------------------------------

def _obs(d: Dict[str, Any], key: str) -> Optional[torch.Tensor]:
    h = d.get(key)
    if h is None:
        return None
    return h.float().unsqueeze(0) if h.dim() == 1 else h.float()


def _first_actions_K(candidates) -> torch.Tensor:
    rows = [traj.actions[:, 0, :].detach().reshape(-1) for traj in candidates]
    return torch.stack(rows, dim=0)


def _entropy_from_counts(counts: Dict[int, int]) -> float:
    n = sum(counts.values())
    if n <= 0:
        return 0.0
    ent = 0.0
    for c in counts.values():
        if c > 0:
            p = c / n
            ent -= p * math.log(p)
    return ent


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


def _e2_contrastive_step(
    agent: REEAgent,
    buffer: Deque[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
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
    weighted = SD056_WEIGHT * loss
    weighted.backward()
    torch.nn.utils.clip_grad_norm_(agent.e2.parameters(), max_norm=MAX_GRAD_NORM)
    optimiser.step()
    return loss_val


# ---------------------------------------------------------------------------
# Per-(channel, arm, seed) runner
# ---------------------------------------------------------------------------

def _run_cell(
    channel: Dict[str, Any],
    arm: Dict[str, Any],
    seed: int,
    p0_episodes: int,
    p1_episodes: int,
    steps_per_episode: int,
) -> Dict[str, Any]:
    reset_all_rng(seed)

    env = _make_env(channel, seed)
    agent = _make_agent(env, channel, arm)
    e2_opt = torch.optim.Adam(agent.e2.parameters(), lr=E2_CONTRASTIVE_LR)

    transition_buffer: Deque[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ] = deque(maxlen=TRANSITION_BUFFER_MAX)
    sample_rng = random.Random(seed)

    demotion_on = bool(arm["use_f_eligibility_demotion"])
    total_train_eps = p0_episodes + p1_episodes

    pairwise_dists: List[float] = []
    k_values: List[int] = []
    demotion_active_ticks = 0
    excluded_counts: List[float] = []
    envelope_sizes: List[float] = []
    excluded_gt0_ticks = 0
    productive_ticks = 0          # demotion-active ticks with 1 <= size < K
    winner_neq_f_argmin_ticks = 0
    rank_preserving_active_ticks = 0
    selected_class_counts: Counter = Counter()
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

            if is_p1 and candidates and len(candidates) >= 2:
                actions_K = _first_actions_K(candidates).to(agent.device)
                z0 = latent.z_world.detach()
                with torch.no_grad():
                    pdist = float(
                        agent.e2.cand_world_pairwise_dist(z0, actions_K).item()
                    )
                if math.isfinite(pdist):
                    pairwise_dists.append(pdist)
                k_values.append(int(len(candidates)))

            if agent.goal_state is not None:
                try:
                    energy = float(body[0, 3].item())
                except Exception:
                    energy = 1.0
                drive_level = max(0.0, 1.0 - energy)
                agent.update_z_goal(benefit_exposure=0.0, drive_level=drive_level)

            action = agent.select_action(candidates, ticks, temperature=1.0)

            # MECH-448 readouts: read LIVE diagnostics at the select tick (P1 only).
            if is_p1:
                diag = agent.e3.last_score_diagnostics
                if bool(diag.get("f_eligibility_demotion_active", False)):
                    demotion_active_ticks += 1
                    env_size = float(diag.get("f_eligibility_envelope_size", -1))
                    excl = float(diag.get("f_eligibility_excluded_count", -1))
                    k_here = (
                        int(env_size + excl)
                        if (env_size >= 0 and excl >= 0) else -1
                    )
                    if math.isfinite(env_size) and env_size >= 0:
                        envelope_sizes.append(env_size)
                    if math.isfinite(excl) and excl >= 0:
                        excluded_counts.append(excl)
                        if excl > 0:
                            excluded_gt0_ticks += 1
                    if k_here >= 2 and 1 <= env_size < k_here:
                        productive_ticks += 1
                    if bool(diag.get("f_eligibility_winner_neq_f_argmin", False)):
                        winner_neq_f_argmin_ticks += 1
                    if bool(diag.get("f_eligibility_rank_preserving", True)):
                        rank_preserving_active_ticks += 1

            if action is None:
                idx = int(np.random.randint(0, env.action_dim))
                action = torch.zeros(1, env.action_dim, device=agent.device)
                action[0, idx] = 1.0
                agent._last_action = action
            if not torch.isfinite(action).all():
                if error_note is None:
                    error_note = (
                        f"non-finite action ch={channel['channel_id']} "
                        f"arm={arm['arm_id']} seed={seed} phase={phase_label} "
                        f"ep={ep} step={_step}"
                    )
                break

            if is_p1:
                selected_class_counts[int(action[0].argmax().item())] += 1
                n_p1_ticks += 1

            if (
                torch.isfinite(latent.z_world).all()
                and torch.isfinite(action).all()
            ):
                pending_capture = (
                    latent.z_world.detach().reshape(-1).clone(),
                    action.detach().reshape(-1).clone(),
                )

            if tick_in_ep % E2_TRAIN_EVERY_K_TICKS == 0:
                loss_val = _e2_contrastive_step(
                    agent=agent, buffer=transition_buffer,
                    optimiser=e2_opt, rng=sample_rng,
                )
                if loss_val is not None and math.isfinite(loss_val) and is_p1:
                    n_contrastive_steps += 1

            _, harm_signal, done, info, next_obs_dict = env.step(action)
            with torch.no_grad():
                agent.update_residue(
                    harm_signal=float(harm_signal), world_delta=None,
                    hypothesis_tag=False, owned=True,
                )

            z_self_prev = latent.z_self.detach()
            action_prev = action
            obs_dict = next_obs_dict
            tick_in_ep += 1
            if done:
                break

        if (ep + 1) % 10 == 0 or (ep + 1) == total_train_eps:
            print(
                f"  [train] ch={channel['channel_id']} arm={arm['arm_id']} "
                f"seed={seed} phase={phase_label} ep {ep + 1}/{total_train_eps}",
                flush=True,
            )

    def _mean(xs: List[float], default: float = 0.0) -> float:
        return float(sum(xs) / len(xs)) if xs else default

    demotion_active_frac = (
        float(demotion_active_ticks) / float(n_p1_ticks) if n_p1_ticks > 0 else 0.0
    )
    excluded_gt0_frac = (
        float(excluded_gt0_ticks) / float(demotion_active_ticks)
        if demotion_active_ticks > 0 else 0.0
    )
    productive_frac = (
        float(productive_ticks) / float(demotion_active_ticks)
        if demotion_active_ticks > 0 else 0.0
    )
    # Rank-preserving is vacuously true when demotion never fires (OFF arm).
    rank_preserving_frac = (
        float(rank_preserving_active_ticks) / float(demotion_active_ticks)
        if demotion_active_ticks > 0 else (1.0 if not demotion_on else 0.0)
    )
    modal_k = Counter(k_values).most_common(1)[0][0] if k_values else 0

    return {
        "channel_id": channel["channel_id"],
        "arm_id": arm["arm_id"],
        "label": arm["label"],
        "seed": int(seed),
        "use_f_eligibility_demotion": demotion_on,
        "use_f_eligibility_adaptive_floor": bool(
            arm["use_f_eligibility_adaptive_floor"]
        ),
        "n_p1_ticks": int(n_p1_ticks),
        "n_contrastive_steps": int(n_contrastive_steps),
        "error_note": error_note,
        # READINESS: divergent pool (range-kind statistic) + demotion engagement.
        "cand_world_pairwise_dist_mean": round(_mean(pairwise_dists), 6),
        "modal_k": int(modal_k),
        "f_eligibility_demotion_active_ticks": int(demotion_active_ticks),
        "f_eligibility_demotion_active_frac": round(demotion_active_frac, 6),
        # PRIMARY: the envelope ACTUALLY excludes, productively.
        "f_eligibility_excluded_count_mean": round(_mean(excluded_counts), 6),
        "f_eligibility_excluded_gt0_frac": round(excluded_gt0_frac, 6),
        "f_eligibility_envelope_size_mean": round(_mean(envelope_sizes), 6),
        "f_eligibility_productive_frac": round(productive_frac, 6),
        "f_eligibility_winner_neq_f_argmin_ticks": int(winner_neq_f_argmin_ticks),
        # GUARDRAIL.
        "f_eligibility_rank_preserving_frac": round(rank_preserving_frac, 6),
        # Informational.
        "selected_action_class_entropy": round(
            _entropy_from_counts(dict(selected_class_counts)), 6
        ),
        "selected_classes_n_unique": int(len(selected_class_counts)),
    }


# ---------------------------------------------------------------------------
# Cross-cell evaluation
# ---------------------------------------------------------------------------

def _cells(rows: List[Dict[str, Any]], channel_id: str, arm_id: str) -> List[Dict[str, Any]]:
    return [
        r for r in rows
        if r.get("channel_id") == channel_id and r.get("arm_id") == arm_id
    ]


def _n_seeds(cells: List[Dict[str, Any]], predicate) -> int:
    return sum(1 for c in cells if predicate(c))


def _evaluate(arm_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    any_error = any(r.get("error_note") for r in arm_results)

    preconditions: List[Dict[str, Any]] = []
    per_channel: Dict[str, Any] = {}
    readiness_ok = True
    adaptive_engages_all = True
    adaptive_still_no_ops = False
    rank_preserving_ok = True
    excluded_groups: List[List[float]] = []  # for check_degeneracy (per channel ADAPTIVE)

    for ch in CHANNELS:
        cid = ch["channel_id"]
        off = _cells(arm_results, cid, OFF_ARM)
        fixed = _cells(arm_results, cid, FIXED_ARM)
        adaptive = _cells(arm_results, cid, ADAPTIVE_ARM)

        # READINESS (range-kind): ADAPTIVE pool divergent + demotion engaged.
        n_pool_ok = _n_seeds(
            adaptive, lambda c: c["cand_world_pairwise_dist_mean"] >= PAIRWISE_DIST_FLOOR
        )
        n_demotion_ok = _n_seeds(
            adaptive, lambda c: c["f_eligibility_demotion_active_frac"] >= DEMOTION_ACTIVE_FRAC_FLOOR
        )
        ch_readiness_ok = (
            n_pool_ok >= MIN_SEEDS_FOR_PASS and n_demotion_ok >= MIN_SEEDS_FOR_PASS
        )
        readiness_ok = readiness_ok and ch_readiness_ok

        mean_pool = (
            float(np.mean([c["cand_world_pairwise_dist_mean"] for c in adaptive]))
            if adaptive else 0.0
        )
        preconditions.append({
            "name": f"pool_divergent_{cid}",
            "description": (
                "ADAPTIVE-arm e2_world_forward candidate-pool spread (mean pairwise "
                "distance, a RANGE-kind statistic matching the envelope's dependence "
                "on a non-uniform F field) clears the floor on >=2/3 seeds"
            ),
            "measured": round(mean_pool, 6),
            "threshold": PAIRWISE_DIST_FLOOR,
            "control": "ADAPTIVE arm divergent candidate pool (the non-vacuity precondition)",
            "met": bool(n_pool_ok >= MIN_SEEDS_FOR_PASS),
        })
        mean_demotion = (
            float(np.mean([c["f_eligibility_demotion_active_frac"] for c in adaptive]))
            if adaptive else 0.0
        )
        preconditions.append({
            "name": f"demotion_engaged_{cid}",
            "description": "ADAPTIVE-arm demotion block fires on the measurement ticks",
            "measured": round(mean_demotion, 6),
            "threshold": DEMOTION_ACTIVE_FRAC_FLOOR,
            "control": "ADAPTIVE arm with a live modulatory channel",
            "met": bool(n_demotion_ok >= MIN_SEEDS_FOR_PASS),
        })

        # PRIMARY: ADAPTIVE excludes (>0), productively, on >=2/3 seeds.
        n_excl = _n_seeds(
            adaptive, lambda c: c["f_eligibility_excluded_gt0_frac"] >= EXCLUDED_GT0_FRAC_FLOOR
        )
        n_prod = _n_seeds(
            adaptive, lambda c: c["f_eligibility_productive_frac"] >= PRODUCTIVE_FRAC_FLOOR
        )
        ch_adaptive_engages = (
            n_excl >= MIN_SEEDS_FOR_PASS and n_prod >= MIN_SEEDS_FOR_PASS
        )
        adaptive_engages_all = adaptive_engages_all and ch_adaptive_engages

        # ADAPTIVE still no-ops despite a divergent pool -> requeue (not a verdict).
        if ch_readiness_ok and not ch_adaptive_engages:
            adaptive_still_no_ops = True

        excluded_groups.append([c["f_eligibility_excluded_count_mean"] for c in adaptive])

        # GUARDRAIL: rank-preserving on every demotion-active cell (FIXED + ADAPTIVE).
        for c in fixed + adaptive:
            if c["f_eligibility_demotion_active_ticks"] > 0 and (
                c["f_eligibility_rank_preserving_frac"] < RANK_PRESERVING_FRAC_REQUIRED
            ):
                rank_preserving_ok = False

        # CORROBORATING (reported): does the FIXED floor mis-fire on this channel?
        fixed_no_op_seeds = _n_seeds(
            fixed, lambda c: c["f_eligibility_excluded_count_mean"] == 0.0
            and c["f_eligibility_demotion_active_ticks"] > 0
        )

        per_channel[cid] = {
            "label": ch["label"],
            "readiness_ok": ch_readiness_ok,
            "n_pool_divergent_seeds": n_pool_ok,
            "n_demotion_engaged_seeds": n_demotion_ok,
            "adaptive_excluded_gt0_seeds": n_excl,
            "adaptive_productive_seeds": n_prod,
            "adaptive_engages": ch_adaptive_engages,
            "adaptive_excluded_count_mean": round(
                float(np.mean([c["f_eligibility_excluded_count_mean"] for c in adaptive]))
                if adaptive else 0.0, 6
            ),
            "adaptive_envelope_size_mean": round(
                float(np.mean([c["f_eligibility_envelope_size_mean"] for c in adaptive]))
                if adaptive else 0.0, 6
            ),
            "fixed_excluded_count_mean": round(
                float(np.mean([c["f_eligibility_excluded_count_mean"] for c in fixed]))
                if fixed else 0.0, 6
            ),
            "fixed_no_op_seeds_corroborating": fixed_no_op_seeds,
            "off_demotion_active_ticks_total": sum(
                c["f_eligibility_demotion_active_ticks"] for c in off
            ),
        }

    deg = check_degeneracy({
        "f_eligibility_excluded_count_mean_adaptive": {"groups": excluded_groups},
    })

    # ---- self-route ----
    if any_error:
        label = "runtime_error"
        evidence_direction = "non_contributory"
        overall_pass = False
    elif not readiness_ok:
        label = "substrate_not_ready_requeue"
        evidence_direction = "non_contributory"
        overall_pass = False
    elif not rank_preserving_ok:
        label = "rank_alteration_not_prefix_diagnose"
        evidence_direction = "non_contributory"
        overall_pass = False
    elif adaptive_still_no_ops or not adaptive_engages_all:
        label = "adaptive_floor_still_no_ops_requeue"
        evidence_direction = "non_contributory"
        overall_pass = False
    else:
        label = "adaptive_floor_engages_across_channels"
        evidence_direction = "non_contributory"
        overall_pass = True

    criteria = [
        {
            "name": "PRIMARY_adaptive_excludes_productively_both_channels",
            "load_bearing": True,
            "passed": bool(adaptive_engages_all and readiness_ok),
        },
        {"name": "READINESS_pools_divergent_both_channels", "passed": bool(readiness_ok)},
        {"name": "GUARDRAIL_rank_preserving_all_demotion_ticks", "passed": bool(rank_preserving_ok)},
    ]
    criteria_non_degenerate = {
        "PRIMARY_excluded_count_has_spread": bool(deg.get("non_degenerate", True)),
        "READINESS_pool_divergent": bool(readiness_ok),
        "GUARDRAIL_rank_preserving": bool(rank_preserving_ok),
    }

    return {
        "label": label,
        "evidence_direction": evidence_direction,
        "overall_pass": overall_pass,
        "readiness_ok": readiness_ok,
        "adaptive_engages_all": adaptive_engages_all,
        "adaptive_still_no_ops": adaptive_still_no_ops,
        "rank_preserving_ok": rank_preserving_ok,
        "per_channel": per_channel,
        "preconditions": preconditions,
        "criteria": criteria,
        "criteria_non_degenerate": criteria_non_degenerate,
        "non_degenerate": deg.get("non_degenerate", True),
        "degeneracy_reason": deg.get("degeneracy_reason", ""),
    }


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    seeds = DRY_RUN_SEEDS if dry_run else SEEDS
    p0 = DRY_RUN_P0 if dry_run else P0_WARMUP_EPISODES
    p1 = DRY_RUN_P1 if dry_run else P1_MEASUREMENT_EPISODES
    steps = DRY_RUN_STEPS if dry_run else STEPS_PER_EPISODE

    arm_results: List[Dict[str, Any]] = []
    for channel in CHANNELS:
        for arm in ARMS:
            for seed in seeds:
                print(
                    f"Seed {seed} Condition {channel['channel_id']}/{arm['arm_id']}",
                    flush=True,
                )
                cell = _run_cell(channel, arm, seed, p0, p1, steps)
                cell["arm_fingerprint"] = compute_arm_fingerprint(
                    config_slice={
                        "channel": {
                            "channel_id": channel["channel_id"],
                            "modulatory": channel["modulatory"],
                            "env_kwargs": channel["env_kwargs"],
                        },
                        "arm": {
                            k: arm[k] for k in (
                                "arm_id", "use_f_eligibility_demotion",
                                "use_f_eligibility_adaptive_floor",
                            )
                        },
                        "f_eligibility_config": {
                            "f_eligibility_envelope_floor": F_ELIGIBILITY_ENVELOPE_FLOOR,
                            "f_eligibility_dn_sigma": F_ELIGIBILITY_DN_SIGMA,
                            "f_eligibility_adaptive_mean_factor": F_ELIGIBILITY_ADAPTIVE_MEAN_FACTOR,
                        },
                        "sd056_weight": SD056_WEIGHT,
                        "p0_episodes": p0, "p1_episodes": p1,
                        "steps_per_episode": steps,
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
    evidence_direction = summary["evidence_direction"]

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
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
        "evidence_direction": evidence_direction,
        "non_degenerate": summary.get("non_degenerate", True),
        "degeneracy_reason": summary.get("degeneracy_reason", ""),
        "evidence_direction_note": (
            "MECH-448 (ARC-107) CHANNEL-ADAPTIVE eligibility-floor substrate-readiness "
            "diagnostic. The fixed absolute share floor (0.30) was tuned to the GAP-A "
            "foraging bank (689d) and mis-fires per channel: 654h (arc_062) all-admit "
            "no-op; 485i (OFC) needed bespoke floor recalibration. The mean-relative "
            "adaptive floor (floor = mean_factor * elig.mean()) auto-calibrates per "
            "channel. PASS (adaptive_floor_engages_across_channels) = READINESS (both "
            "channel pools divergent + demotion engaged) AND the ADAPTIVE arm excludes "
            "productively (excluded_count>0, 1<=envelope_size<K) on BOTH channels with "
            "the SAME global config AND rank-preserving on every demotion-active tick. "
            "A non-divergent pool / demotion-inactive channel -> substrate_not_ready_requeue; "
            "ADAPTIVE still no-ops on a channel -> adaptive_floor_still_no_ops_requeue "
            "(mean-relative insufficient -> mean_factor sweep / different mode); a "
            "rank fraction < 1.0 -> rank_alteration_not_prefix_diagnose (/diagnose-errors). "
            "claim_ids=[]; PROMOTES NOTHING; MECH-448 stays candidate; ARC-107 untouched."
        ),
        "interpretation": {
            "label": summary["label"],
            "preconditions": summary["preconditions"],
            "criteria": summary["criteria"],
            "criteria_non_degenerate": summary["criteria_non_degenerate"],
            "routing": {
                "adaptive_floor_engages_across_channels": "PASS -> the channel-adaptive (mean-relative) floor is ready; the per-channel hand-floor sweeps (654h/485i/485j + pending 625/445/687) retire; downstream demotion retests use the adaptive floor",
                "substrate_not_ready_requeue": "a channel pool is not divergent OR demotion never fires; re-queue at a longer P0 warmup; do NOT weaken anything",
                "adaptive_floor_still_no_ops_requeue": "READINESS met but the ADAPTIVE floor still admits all (excluded_count==0) on a channel -> mean-relative insufficient -> mean_factor sweep / a different adaptive mode; NOT a falsification",
                "rank_alteration_not_prefix_diagnose": "the eligible set is NOT an F-rank prefix on a demotion-active tick -> implementation/design fault -> /diagnose-errors; NOT a verdict",
            },
        },
        "dry_run": bool(dry_run),
        "config": {
            "seeds": seeds,
            "p0_warmup_episodes": p0,
            "p1_measurement_episodes": p1,
            "steps_per_episode": steps,
            "channels": [
                {"channel_id": c["channel_id"], "label": c["label"],
                 "modulatory": c["modulatory"], "env_kwargs": c["env_kwargs"]}
                for c in CHANNELS
            ],
            "arms": [
                {k: a[k] for k in (
                    "arm_id", "label", "use_f_eligibility_demotion",
                    "use_f_eligibility_adaptive_floor",
                )}
                for a in ARMS
            ],
            "sd056_weight": SD056_WEIGHT,
            "f_eligibility_config": {
                "f_eligibility_envelope_floor": F_ELIGIBILITY_ENVELOPE_FLOOR,
                "f_eligibility_dn_sigma": F_ELIGIBILITY_DN_SIGMA,
                "f_eligibility_adaptive_mean_factor": F_ELIGIBILITY_ADAPTIVE_MEAN_FACTOR,
            },
            "conversion_constant": {
                "use_modulatory_channel_routing": True,
                "modulatory_channel_route_source": "cand_world_summary",
                "use_modulatory_selection_authority": True,
                "modulatory_authority_gain": MODULATORY_AUTHORITY_GAIN,
                "modulatory_authority_normalize_basis": MODULATORY_AUTHORITY_NORMALIZE_BASIS,
                "use_modulatory_shortlist_then_modulate": True,
                "modulatory_shortlist_mode": MODULATORY_SHORTLIST_MODE,
                "modulatory_shortlist_k": MODULATORY_SHORTLIST_K,
            },
            "thresholds": {
                "pairwise_dist_floor": PAIRWISE_DIST_FLOOR,
                "demotion_active_frac_floor": DEMOTION_ACTIVE_FRAC_FLOOR,
                "excluded_gt0_frac_floor": EXCLUDED_GT0_FRAC_FLOOR,
                "productive_frac_floor": PRODUCTIVE_FRAC_FLOOR,
                "rank_preserving_frac_required": RANK_PRESERVING_FRAC_REQUIRED,
                "min_seeds_for_pass": MIN_SEEDS_FOR_PASS,
            },
        },
        "acceptance_criteria": {
            "readiness_pools_divergent_both_channels": summary["readiness_ok"],
            "PRIMARY_adaptive_excludes_productively_both_channels": bool(
                summary["adaptive_engages_all"] and summary["readiness_ok"]
            ),
            "GUARDRAIL_rank_preserving_all_demotion_ticks": summary["rank_preserving_ok"],
            "overall_pass": summary["overall_pass"],
        },
        "summary": summary,
        "arm_results": arm_results,
    }

    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.json"

    if not dry_run:
        with open(out_path, "w") as fh:
            json.dump(manifest, fh, indent=2)
        print(f"Manifest written: {out_path}", flush=True)
    else:
        out_path = Path("/dev/null")
        print("Dry run -- manifest not written.", flush=True)

    print(
        f"Outcome: {outcome} (label={summary['label']}, "
        f"evidence_direction={evidence_direction})",
        flush=True,
    )
    for k, v in manifest["acceptance_criteria"].items():
        print(f"  {k}: {v}", flush=True)

    manifest["manifest_path"] = str(out_path)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="V3-EXQ-689e MECH-448 channel-adaptive eligibility-floor readiness diagnostic"
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
