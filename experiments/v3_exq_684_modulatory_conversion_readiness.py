"""V3-EXQ-684: modulatory-bias-selection-authority CONVERSION readiness validation
-- does the routed per-candidate channel range now MOVE the committed argmax under
the new gain/contrast + shortlist-then-modulate levers, beating a VERIFIED-LIFTING
matched-noise control?

behavioral_diversity_isolation:GAP-A. Claim-free substrate-readiness DIAGNOSTIC
(claim_ids=[]; experiment_purpose=diagnostic) for the conversion amend landed
2026-06-15 (e3_selector.py + config.py), routed by failure_autopsy_V3-EXQ-569g_2026-06-14
+ GAP-A governance_2026_06_15. NOT an evidence run: it cannot weaken ARC-065 either way;
it identifies WHICH conversion mechanism (if any) lifts committed-action diversity, to
select the config for the SEPARATE V3-EXQ-569h falsifier (which is GATED on this PASS).

WHY 684 (the gate before 569h): V3-EXQ-682 (no_collapse_reproduced, 2026-06-15) confirmed
the route-range amend SOLVED REACH -- ARM_1_E2WF applies in-arm route_range ~0.20 at the
live select tick, all four collapse causes ruled out incl seed 43. The residual gap 569g
isolated is CONVERSION: the gap-relative ADDITIVE authority at gain 0.5
(modrange = 0.5*raw_score_range) is subdominant to the F-dominated primary (88-89% of E3
variance, V3-EXQ-571), so the routed range flips only near-tie OUTLIERS, not near-decisive
winners (569g committed entropy 1/3 seeds strict-above a temperature-matched control; that
control UNDER-LIFTED to entropy==proposer, so its bar was vacuous). The conversion amend
adds two no-op-default levers; this validation discriminates which lifts.

DESIGN: 6-arm grid, matched seeds. The substrate stack is IDENTICAL to V3-EXQ-569g
(SP-CEM Layer A + shared E3-side bias channels lateral_pfc + mech295 + SD-056 online
contrastive with the rollout-norm clamp + use_modulatory_channel_routing
source=cand_world_summary + candidate_summary_source per arm) -- the ONLY new axes are the
conversion levers:
  ARM_PROPOSER      proposer source, T=1.0, additive range basis gain 0.5  (collapsed reference; route ~0)
  ARM_MATCHED_NOISE proposer source, T=2.5, additive range basis gain 0.5  (entropy control; MUST verify-lift over ARM_PROPOSER)
  ARM_LEGACY_E2WF   e2wf source, T=1.0, additive RANGE basis gain 0.5      (= 569g ARM_1; the no-conversion bar -- routed range present, but additive/near-tie-only)
  ARM_STD_G1        e2wf source, T=1.0, additive STD basis gain 1.0        (conversion lever (a))
  ARM_STD_G2        e2wf source, T=1.0, additive STD basis gain 2.0        (conversion lever (a))
  ARM_SHORTLIST     e2wf source, T=1.0, shortlist-then-modulate margin 0.25 (conversion lever (b); F filters to near-tie set, modulatory arbitrates within)

Treatment arms (e2wf, route-bearing): LEGACY_E2WF / STD_G1 / STD_G2 / SHORTLIST.
Conversion arms (the levers under test): STD_G1 / STD_G2 / SHORTLIST.

PRIMARY DV: committed (selected) first-action-class entropy per arm.

ACCEPTANCE (pre-registered; self-route grid; claim_ids=[] -> cannot weaken any claim):
  READINESS-1 (load-bearing, route-range; the SAME statistic the route-range substrate
    gates on -- range, NOT magnitude): the e2wf treatment arms apply in-arm
    modulatory_channel_route_range > ROUTE_RANGE_FLOOR on >= MIN_SEEDS seeds (checked on
    ARM_LEGACY_E2WF, the canonical e2wf channel). Below floor => routing not wired / e2
    under-trained => substrate_not_ready_requeue.
  READINESS-2 (load-bearing, the 569g gap): the MATCHED-NOISE control VERIFY-LIFTS --
    ARM_MATCHED_NOISE selected entropy STRICTLY ABOVE ARM_PROPOSER on the same seed on
    >= MIN_SEEDS seeds. A non-lifting temperature control cannot discriminate structured
    from random diversity (569g's ARM_2 under-lifted: entropy == proposer), so below this
    self-routes substrate_not_ready_requeue (control_not_lifting), NOT a verdict.
  C_CONVERSION (load-bearing): at least ONE conversion arm (STD_G1 / STD_G2 / SHORTLIST)
    produces committed-action entropy STRICTLY ABOVE BOTH ARM_LEGACY_E2WF (the additive
    no-conversion bar) AND ARM_MATCHED_NOISE (the verified-lifting noise control) on the
    same seed on >= MIN_SEEDS seeds -> the conversion lever MOVES committed diversity with
    structure a temperature control cannot replicate.

Self-route grid:
| outcome                                  | label                                   | outcome | next                                                  |
|------------------------------------------|-----------------------------------------|---------|-------------------------------------------------------|
| READINESS-1/2 unmet                      | substrate_not_ready_requeue             | FAIL    | non_contributory; re-queue at higher P0 / check wiring |
| READINESS met + >=1 conversion arm lifts | conversion_mechanism_identified         | PASS    | the winning arm config -> queue V3-EXQ-569h             |
| READINESS met + NO conversion arm lifts  | conversion_ceiling_persists_under_levers| FAIL    | /failure-autopsy; do NOT auto-promote 569h             |

Usage:
  /opt/local/bin/python3 experiments/v3_exq_684_modulatory_conversion_readiness.py --dry-run
"""

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

EXPERIMENT_TYPE = "v3_exq_684_modulatory_conversion_readiness"
QUEUE_ID = "V3-EXQ-684"
CLAIM_IDS: List[str] = []  # claim-free substrate-readiness diagnostic
EXPERIMENT_PURPOSE = "diagnostic"

SEEDS = [42, 43, 44]
P0_WARMUP_EPISODES = 60           # SD-056 contrastive warmup (V3-EXQ-649/569g proven budget)
P1_MEASUREMENT_EPISODES = 20
STEPS_PER_EPISODE = 200

DRY_RUN_SEEDS = [42]
DRY_RUN_P0 = 2
DRY_RUN_P1 = 2
DRY_RUN_STEPS = 30

# Acceptance thresholds (pre-registered).
ROUTE_RANGE_FLOOR = 0.01          # READINESS-1: e2wf-arm modulatory_channel_route_range (V3-EXQ-662 statistic)
C1_PAIRWISE_DIST_FLOOR = 0.03     # READINESS: ARM_LEGACY_E2WF e2.world_forward prediction spread (SD-056 trained)
C3_SELECTED_ENTROPY_FLOOR = 0.3   # conversion arms must clear a non-trivial entropy floor
MATCHED_ENTROPY_TEMPERATURE = 2.5
MIN_SEEDS_FOR_PASS = 2            # of 3

# SD-056 online contrastive training (mirror V3-EXQ-569g harness).
SD056_WEIGHT = 0.05
E2_CONTRASTIVE_LR = 1e-3
E2_TRAIN_EVERY_K_TICKS = 1
CONTRASTIVE_BATCH_K = 8
TRANSITION_BUFFER_MAX = 256
MIN_BUFFER_BEFORE_TRAIN = 16
MIN_CLASSES_FOR_TRAIN = 2
MAX_GRAD_NORM = 1.0

MODULATORY_ROUTE_MIN_RANGE_FLOOR = 1e-6  # substrate numerical active/inactive floor
MODULATORY_SHORTLIST_MARGIN = 0.25

# Behavioural-diversity env: SD-054 reef-bipartite hazard layout (matches the
# 569g FP-2 falsifier env exactly -- reef-vs-forage forces categorically opposite
# first actions).
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

# Per-arm conversion-lever config. Treatment arms (e2wf source) carry route-range;
# conversion arms set normalize_basis='std' (lever a) or shortlist (lever b).
ARMS: List[Dict[str, Any]] = [
    {
        "arm_id": "ARM_PROPOSER",
        "label": "proposer_collapsed_reference_additive_range_g0p5",
        "candidate_summary_source": "proposer",
        "temperature": 1.0,
        "normalize_basis": "range",
        "authority_gain": 0.5,
        "use_authority": True,
        "use_shortlist": False,
        "role": "control_reference",
    },
    {
        "arm_id": "ARM_MATCHED_NOISE",
        "label": "proposer_matched_entropy_temperature_control",
        "candidate_summary_source": "proposer",
        "temperature": MATCHED_ENTROPY_TEMPERATURE,
        "normalize_basis": "range",
        "authority_gain": 0.5,
        "use_authority": True,
        "use_shortlist": False,
        "role": "noise_control",
    },
    {
        "arm_id": "ARM_LEGACY_E2WF",
        "label": "e2wf_additive_range_basis_g0p5_no_conversion_bar",
        "candidate_summary_source": "e2_world_forward",
        "temperature": 1.0,
        "normalize_basis": "range",
        "authority_gain": 0.5,
        "use_authority": True,
        "use_shortlist": False,
        "role": "legacy_bar",
    },
    {
        "arm_id": "ARM_STD_G1",
        "label": "e2wf_conversion_a_std_basis_g1p0",
        "candidate_summary_source": "e2_world_forward",
        "temperature": 1.0,
        "normalize_basis": "std",
        "authority_gain": 1.0,
        "use_authority": True,
        "use_shortlist": False,
        "role": "conversion",
    },
    {
        "arm_id": "ARM_STD_G2",
        "label": "e2wf_conversion_a_std_basis_g2p0",
        "candidate_summary_source": "e2_world_forward",
        "temperature": 1.0,
        "normalize_basis": "std",
        "authority_gain": 2.0,
        "use_authority": True,
        "use_shortlist": False,
        "role": "conversion",
    },
    {
        "arm_id": "ARM_SHORTLIST",
        "label": "e2wf_conversion_b_shortlist_then_modulate_m0p25",
        "candidate_summary_source": "e2_world_forward",
        "temperature": 1.0,
        "normalize_basis": "range",
        "authority_gain": 0.5,
        "use_authority": False,   # shortlist selection takes precedence; authority off = clean lever-(b) test
        "use_shortlist": True,
        "role": "conversion",
    },
]

CONVERSION_ARM_IDS = ["ARM_STD_G1", "ARM_STD_G2", "ARM_SHORTLIST"]
TREATMENT_ARM_IDS = ["ARM_LEGACY_E2WF", "ARM_STD_G1", "ARM_STD_G2", "ARM_SHORTLIST"]


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(seed=seed, **ENV_KWARGS)


def _make_agent(env: CausalGridWorldV2, arm: Dict[str, Any]) -> REEAgent:
    """Main-path SP-CEM stack identical to V3-EXQ-569g (shared E3-side bias channels
    + SD-056 online contrastive + route-range substrate ON), with the conversion
    levers set per arm: modulatory_authority_normalize_basis (lever a) and
    use_modulatory_shortlist_then_modulate (lever b)."""
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
        # ARC-065 SP-CEM (Layer A) -- main-path default (action-divergent pool)
        use_support_preserving_cem=True,
        support_preserving_stratified_elites=True,
        support_preserving_ao_std_floor=0.2,
        support_preserving_min_first_action_classes=2,
        # SHARED E3-side bias channels (consume cand_world_summaries)
        use_lateral_pfc_analog=True,
        use_mech295_liking_bridge=True,
        # Other policy-layer regulators OFF (conversion levers are the axis)
        use_structured_curiosity=False,
        use_e3_score_diversity=False,
        use_noise_floor=False,
        use_tonic_vigor=False,
        use_dacc=False,
        use_ofc_analog=False,
        use_gated_policy=False,
        # SD-056 substrate trained online on every arm (e2.world_forward divergence)
        e2_action_contrastive_enabled=True,
        e2_action_contrastive_weight=SD056_WEIGHT,
        e2_rollout_output_norm_clamp_enabled=True,
        e2_rollout_output_norm_clamp_ratio=2.0,
        # GAP-A consumed-channel source (proposer collapses route ~0; e2wf carries range)
        candidate_summary_source=str(arm["candidate_summary_source"]),
        # Route-range substrate ON all arms (the 569g substrate; routes the consumed
        # cand_world_summaries cross-candidate range into the E3 modulatory accumulator)
        use_modulatory_channel_routing=True,
        modulatory_channel_route_source="cand_world_summary",
        modulatory_channel_route_weight=1.0,
        modulatory_channel_route_min_range_floor=MODULATORY_ROUTE_MIN_RANGE_FLOOR,
        # CONVERSION amend (569g/682): the levers under test, set per arm.
        use_modulatory_selection_authority=bool(arm["use_authority"]),
        modulatory_authority_gain=float(arm["authority_gain"]),
        modulatory_authority_normalize_basis=str(arm["normalize_basis"]),
        use_modulatory_shortlist_then_modulate=bool(arm["use_shortlist"]),
        modulatory_shortlist_margin=MODULATORY_SHORTLIST_MARGIN,
    )
    agent = REEAgent(cfg)
    return agent


# ---------------------------------------------------------------------------
# Measurement helpers (ported from V3-EXQ-569g)
# ---------------------------------------------------------------------------

def _trajectory_first_action_class(traj) -> int:
    return int(traj.actions[:, 0, :].argmax(dim=-1).detach().reshape(-1)[0].item())


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
    weighted = SD056_WEIGHT * loss
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
    reset_all_rng(seed)

    env = _make_env(seed)
    agent = _make_agent(env, arm)
    e2_opt = torch.optim.Adam(agent.e2.parameters(), lr=E2_CONTRASTIVE_LR)

    transition_buffer: Deque[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ] = deque(maxlen=TRANSITION_BUFFER_MAX)
    sample_rng = random.Random(seed)

    arm_temperature = float(arm["temperature"])
    total_train_eps = p0_episodes + p1_episodes

    pairwise_dists: List[float] = []
    route_ranges: List[float] = []          # in-arm modulatory_channel_route_range
    route_range_max = 0.0
    shortlist_sizes: List[float] = []
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
                # READINESS: e2.world_forward prediction spread (SD-056 trained).
                actions_K = _first_actions_K(candidates).to(agent.device)
                z0 = latent.z_world.detach()
                with torch.no_grad():
                    pdist = float(agent.e2.cand_world_pairwise_dist(z0, actions_K).item())
                if math.isfinite(pdist):
                    pairwise_dists.append(pdist)

            if agent.goal_state is not None:
                try:
                    energy = float(body[0, 3].item())
                except Exception:
                    energy = 1.0
                drive_level = max(0.0, 1.0 - energy)
                agent.update_z_goal(benefit_exposure=0.0, drive_level=drive_level)

            action = agent.select_action(
                candidates, ticks, temperature=arm_temperature
            )

            # READINESS-1 / route-range: in-arm RAW cross-candidate range of the routed
            # modulatory bias (V3-EXQ-662 statistic) + conversion-lever diagnostics.
            if is_p1:
                diag = agent.e3.last_score_diagnostics
                rr = float(diag.get("modulatory_channel_route_range", 0.0))
                if math.isfinite(rr):
                    route_ranges.append(rr)
                    route_range_max = max(route_range_max, rr)
                if bool(diag.get("modulatory_shortlist_active", False)):
                    shortlist_sizes.append(
                        float(diag.get("modulatory_shortlist_size", 0))
                    )

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

            # PRIMARY DV: committed first-action class diversity.
            if is_p1:
                committed_class = int(action[0].argmax().item())
                selected_class_counts[committed_class] += 1
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

    selected_action_entropy = _entropy_from_counts(dict(selected_class_counts))

    return {
        "arm_id": arm["arm_id"],
        "label": arm["label"],
        "role": arm["role"],
        "seed": int(seed),
        "candidate_summary_source": arm["candidate_summary_source"],
        "temperature": arm_temperature,
        "normalize_basis": arm["normalize_basis"],
        "authority_gain": float(arm["authority_gain"]),
        "use_authority": bool(arm["use_authority"]),
        "use_shortlist": bool(arm["use_shortlist"]),
        "n_p1_ticks": int(n_p1_ticks),
        "n_contrastive_steps": int(n_contrastive_steps),
        "error_note": error_note,
        # READINESS-1: in-arm routed range.
        "modulatory_channel_route_range_mean": round(_mean(route_ranges), 6),
        "modulatory_channel_route_range_max": round(route_range_max, 6),
        # READINESS: e2.world_forward prediction spread.
        "cand_world_pairwise_dist_mean": round(_mean(pairwise_dists), 6),
        # Lever (b) diagnostic.
        "modulatory_shortlist_size_mean": round(_mean(shortlist_sizes), 6),
        # PRIMARY DV.
        "selected_action_class_entropy": round(selected_action_entropy, 6),
        "selected_class_counts": dict(sorted(selected_class_counts.items())),
        "selected_classes_n_unique": int(len(selected_class_counts)),
    }


# ---------------------------------------------------------------------------
# Cross-arm evaluation
# ---------------------------------------------------------------------------

def _arm_rows(rows: List[Dict[str, Any]], arm_id: str) -> List[Dict[str, Any]]:
    return [r for r in rows if r.get("arm_id") == arm_id]


def _n_seeds(rows: List[Dict[str, Any]], predicate) -> int:
    return sum(1 for r in rows if predicate(r))


def _mean_key(rows: List[Dict[str, Any]], key: str) -> float:
    vals = [float(r.get(key, 0.0)) for r in rows]
    return float(sum(vals) / len(vals)) if vals else 0.0


def _evaluate(arm_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_arm = {a["arm_id"]: _arm_rows(arm_results, a["arm_id"]) for a in ARMS}
    by_arm_seed = {
        aid: {r["seed"]: r for r in rows} for aid, rows in by_arm.items()
    }

    RDIST = "modulatory_channel_route_range_mean"
    PDIST = "cand_world_pairwise_dist_mean"
    SENT = "selected_action_class_entropy"

    legacy = by_arm["ARM_LEGACY_E2WF"]
    proposer_by_seed = by_arm_seed["ARM_PROPOSER"]
    noise_by_seed = by_arm_seed["ARM_MATCHED_NOISE"]
    legacy_by_seed = by_arm_seed["ARM_LEGACY_E2WF"]

    # READINESS-1 (load-bearing, route-range, the SAME statistic the route-range
    # substrate gates on): ARM_LEGACY_E2WF (the canonical e2wf channel) carries
    # in-arm route_range > floor on >= MIN_SEEDS seeds.
    legacy_route_mean = _mean_key(legacy, RDIST)
    route_seeds_ok = _n_seeds(
        legacy, lambda r: float(r.get(RDIST, 0.0)) > ROUTE_RANGE_FLOOR
    )
    route_ready = bool(route_seeds_ok >= MIN_SEEDS_FOR_PASS)

    # READINESS (e2 divergence): ARM_LEGACY_E2WF e2.world_forward prediction spread.
    legacy_pdist_mean = _mean_key(legacy, PDIST)
    pdist_seeds_ok = _n_seeds(
        legacy, lambda r: float(r.get(PDIST, 0.0)) > C1_PAIRWISE_DIST_FLOOR
    )
    pdist_ready = bool(pdist_seeds_ok >= MIN_SEEDS_FOR_PASS)

    # READINESS-2 (load-bearing, the 569g gap): the matched-noise control verify-lifts
    # over ARM_PROPOSER on the same seed on >= MIN_SEEDS seeds.
    def _noise_lifts(seed: int) -> bool:
        rp = proposer_by_seed.get(seed)
        rn = noise_by_seed.get(seed)
        if rp is None or rn is None:
            return False
        return float(rn.get(SENT, 0.0)) > float(rp.get(SENT, 0.0))
    noise_lift_seeds = sum(1 for s in (r["seed"] for r in by_arm["ARM_MATCHED_NOISE"]) if _noise_lifts(s))
    control_ready = bool(noise_lift_seeds >= MIN_SEEDS_FOR_PASS)

    readiness_ok = bool(route_ready and pdist_ready and control_ready)

    # C_CONVERSION (load-bearing): a conversion arm's selected entropy STRICTLY ABOVE
    # BOTH ARM_LEGACY_E2WF AND ARM_MATCHED_NOISE on the same seed on >= MIN_SEEDS seeds.
    def _conversion_seeds(arm_id: str) -> int:
        rows = by_arm[arm_id]
        n = 0
        for r in rows:
            seed = r["seed"]
            rl = legacy_by_seed.get(seed)
            rn = noise_by_seed.get(seed)
            if rl is None or rn is None:
                continue
            e = float(r.get(SENT, 0.0))
            if (
                e > float(rl.get(SENT, 0.0))
                and e > float(rn.get(SENT, 0.0))
                and e > C3_SELECTED_ENTROPY_FLOOR
            ):
                n += 1
        return n

    conversion_arm_pass: Dict[str, int] = {
        aid: _conversion_seeds(aid) for aid in CONVERSION_ARM_IDS
    }
    winning_arms = [aid for aid, n in conversion_arm_pass.items() if n >= MIN_SEEDS_FOR_PASS]
    c_conversion_pass = bool(len(winning_arms) >= 1)

    # Non-degeneracy: every arm produced P1 ticks.
    non_degenerate = bool(
        all(len(rows) > 0 for rows in by_arm.values())
        and all(
            int(r.get("n_p1_ticks", 0)) > 0
            for rows in by_arm.values()
            for r in rows
        )
    )

    if not readiness_ok:
        label = "substrate_not_ready_requeue"
        overall_pass = False
        evidence_direction = "non_contributory"
    elif c_conversion_pass:
        label = "conversion_mechanism_identified"
        overall_pass = True
        evidence_direction = "diagnostic"
    else:
        label = "conversion_ceiling_persists_under_levers"
        overall_pass = False
        evidence_direction = "diagnostic"

    selected_entropy_per_arm = {
        aid: round(_mean_key(rows, SENT), 6) for aid, rows in by_arm.items()
    }
    route_range_per_arm = {
        aid: round(_mean_key(rows, RDIST), 6) for aid, rows in by_arm.items()
    }

    return {
        "readiness": {
            "route_range_floor": ROUTE_RANGE_FLOOR,
            "legacy_e2wf_route_range_mean": round(legacy_route_mean, 6),
            "legacy_e2wf_seeds_route_above_floor": int(route_seeds_ok),
            "route_ready": route_ready,
            "c1_pairwise_dist_floor": C1_PAIRWISE_DIST_FLOOR,
            "legacy_e2wf_pairwise_dist_mean": round(legacy_pdist_mean, 6),
            "legacy_e2wf_seeds_e2_divergent": int(pdist_seeds_ok),
            "pdist_ready": pdist_ready,
            "matched_noise_lift_seeds_over_proposer": int(noise_lift_seeds),
            "control_ready": control_ready,
            "min_seeds_required": MIN_SEEDS_FOR_PASS,
            "readiness_ok": readiness_ok,
        },
        "c_conversion": {
            "conversion_arm_pass_seed_count": conversion_arm_pass,
            "winning_arms": winning_arms,
            "c_conversion_pass": c_conversion_pass,
            "selected_entropy_floor": C3_SELECTED_ENTROPY_FLOOR,
        },
        "route_range_per_arm_mean": route_range_per_arm,
        "selected_action_entropy_per_arm_mean": selected_entropy_per_arm,
        "label": label,
        "evidence_direction": evidence_direction,
        "overall_pass": overall_pass,
        # Diagnostic-adjudication structures (the self-route is falsifiable).
        "preconditions": [
            {
                "name": "legacy_e2wf_modulatory_channel_route_range_supra_floor",
                "kind": "readiness",
                "description": (
                    "ARM_LEGACY_E2WF (e2_world_forward source) in-arm RAW cross-candidate "
                    "RANGE of the modulatory bias ROUTED into the E3 selection authority "
                    "(modulatory_channel_route_range, the V3-EXQ-662 statistic) clears the "
                    "floor -- the channel range REACHES the bias the conversion levers "
                    "rescale/arbitrate. SAME statistic the route-range substrate gates on "
                    "(range, NOT magnitude). Below floor => routing not wired / e2 "
                    "under-trained => substrate_not_ready_requeue, never a verdict."
                ),
                "control": (
                    "ARM_LEGACY_E2WF: use_modulatory_channel_routing + "
                    "source=cand_world_summary; candidate_summary_source=e2_world_forward "
                    "(genuinely action-divergent routed channel)"
                ),
                "measured": round(legacy_route_mean, 6),
                "threshold": ROUTE_RANGE_FLOOR,
                "met": route_ready,
            },
            {
                "name": "legacy_e2wf_e2_world_forward_prediction_spread_supra_floor",
                "kind": "readiness",
                "description": (
                    "ARM_LEGACY_E2WF e2.world_forward(z0, a_i) per-candidate prediction "
                    "spread (cand_world_pairwise_dist) clears the floor -- confirms SD-056 "
                    "trained the action-conditional divergence the routed channel "
                    "re-sources. Range statistic. Below floor => SD-056 under-trained => "
                    "substrate_not_ready_requeue."
                ),
                "control": "ARM_LEGACY_E2WF: agent.e2.cand_world_pairwise_dist on SP-CEM candidates",
                "measured": round(legacy_pdist_mean, 6),
                "threshold": C1_PAIRWISE_DIST_FLOOR,
                "met": pdist_ready,
            },
            {
                "name": "matched_noise_control_verify_lifts_over_proposer",
                "kind": "readiness",
                "description": (
                    "The matched-noise temperature control (ARM_MATCHED_NOISE, proposer @ "
                    "T=2.5) selected-action entropy STRICTLY ABOVE the proposer reference "
                    "(ARM_PROPOSER @ T=1.0) on >= MIN_SEEDS seeds -- the 569g gap: a "
                    "non-lifting control cannot discriminate structured from random "
                    "diversity (569g's ARM_2 under-lifted to entropy==proposer). Entropy "
                    "statistic (the SAME the C_CONVERSION criterion routes on). Below floor "
                    "=> control_not_lifting => substrate_not_ready_requeue."
                ),
                "control": "ARM_MATCHED_NOISE vs ARM_PROPOSER per-seed selected-action entropy",
                "measured": float(noise_lift_seeds),
                "threshold": float(MIN_SEEDS_FOR_PASS),
                "met": control_ready,
            },
        ],
        "criteria": [
            {
                "name": "C_CONVERSION_at_least_one_arm_strict_above_legacy_and_noise",
                "load_bearing": True,
                "passed": c_conversion_pass,
            },
        ],
        "criteria_non_degenerate": {"C_CONVERSION": non_degenerate},
    }


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
            cell["arm_fingerprint"] = compute_arm_fingerprint(
                config_slice={
                    "arm": {
                        k: arm[k]
                        for k in (
                            "arm_id", "candidate_summary_source", "temperature",
                            "normalize_basis", "authority_gain", "use_authority",
                            "use_shortlist",
                        )
                    },
                    "env_kwargs": ENV_KWARGS,
                    "sd056_weight": SD056_WEIGHT,
                    "shortlist_margin": MODULATORY_SHORTLIST_MARGIN,
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
    evidence_direction = summary["evidence_direction"]

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
        "evidence_direction": evidence_direction,
        "evidence_direction_note": (
            "CLAIM-FREE readiness validation for the modulatory-bias-selection-authority "
            "CONVERSION amend (gain/contrast normalize_basis + shortlist-then-modulate), "
            "routed by failure_autopsy_V3-EXQ-569g_2026-06-14 + GAP-A governance_2026_06_15 "
            "after V3-EXQ-682 (no_collapse_reproduced) confirmed REACH is solved. Tests "
            "whether the routed per-candidate channel range MOVES the committed argmax "
            "under the new levers, beating a VERIFIED-LIFTING matched-noise control. "
            "PASS (conversion_mechanism_identified) = at least one conversion arm (STD or "
            "SHORTLIST) committed-action entropy STRICTLY ABOVE BOTH ARM_LEGACY_E2WF (the "
            "additive no-conversion bar) AND ARM_MATCHED_NOISE (the verified-lifting noise "
            "control) on >=2/3 seeds, with route-range + e2-divergence + control-lift "
            "readiness met -- the conversion lever moves committed diversity with structure "
            "a temperature control cannot replicate; the winning arm config selects the "
            "V3-EXQ-569h falsifier. READINESS unmet (route below floor / e2 under-trained / "
            "matched-noise control does NOT lift over proposer) self-routes "
            "substrate_not_ready_requeue (non_contributory). READINESS met but NO conversion "
            "arm lifts => conversion_ceiling_persists_under_levers (FAIL; route to "
            "/failure-autopsy, do NOT auto-promote 569h). claim_ids=[] -> cannot weaken "
            "ARC-065 / MECH-341 / ARC-062 / MECH-309 / MECH-294 either way."
        ),
        "interpretation": {
            "label": summary["label"],
            "preconditions": summary["preconditions"],
            "criteria": summary["criteria"],
            "criteria_non_degenerate": summary["criteria_non_degenerate"],
            "routing": {
                "conversion_mechanism_identified": "PASS -> the winning conversion arm config selects V3-EXQ-569h (the GAP-A falsifier with an in-arm route-range gate)",
                "substrate_not_ready_requeue": "re-queue at higher P0 / check route-range wiring + that the matched-noise control lifts; do NOT promote to 569h",
                "conversion_ceiling_persists_under_levers": "FAIL -> /failure-autopsy: the conversion ceiling survives gain/contrast + shortlist; do NOT auto-promote 569h",
            },
        },
        "dry_run": bool(dry_run),
        "config": {
            "seeds": seeds,
            "p0_warmup_episodes": p0,
            "p1_measurement_episodes": p1,
            "steps_per_episode": steps,
            "env_kwargs": ENV_KWARGS,
            "arms": [
                {
                    k: a[k]
                    for k in (
                        "arm_id", "label", "role", "candidate_summary_source",
                        "temperature", "normalize_basis", "authority_gain",
                        "use_authority", "use_shortlist",
                    )
                }
                for a in ARMS
            ],
            "conversion_arm_ids": CONVERSION_ARM_IDS,
            "treatment_arm_ids": TREATMENT_ARM_IDS,
            "sd056_weight": SD056_WEIGHT,
            "matched_entropy_temperature": MATCHED_ENTROPY_TEMPERATURE,
            "shortlist_margin": MODULATORY_SHORTLIST_MARGIN,
            "thresholds": {
                "route_range_floor": ROUTE_RANGE_FLOOR,
                "c1_pairwise_dist_floor": C1_PAIRWISE_DIST_FLOOR,
                "c3_selected_entropy_floor": C3_SELECTED_ENTROPY_FLOOR,
                "min_seeds_for_pass": MIN_SEEDS_FOR_PASS,
            },
        },
        "acceptance_criteria": {
            "readiness_route_range_ready": summary["readiness"]["route_ready"],
            "readiness_e2_divergent_ready": summary["readiness"]["pdist_ready"],
            "readiness_matched_noise_control_lifts": summary["readiness"]["control_ready"],
            "readiness_substrate_ready": summary["readiness"]["readiness_ok"],
            "C_CONVERSION_at_least_one_arm_lifts": summary["c_conversion"]["c_conversion_pass"],
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
        description="V3-EXQ-684 modulatory conversion readiness validation (gain/contrast + shortlist)"
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
