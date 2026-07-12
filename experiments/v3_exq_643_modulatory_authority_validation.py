#!/opt/local/bin/python3
"""V3-EXQ-643 -- modulatory-bias-selection-authority falsification.

Validation experiment for the modulatory-bias-selection-authority substrate
(e3_selector.py:764-794, landed 2026-06-03). It is the outstanding "Next" step
in the design doc REE_assembly/evidence/planning/
modulatory_bias_selection_authority_design.md ("Falsification experiment (to be
queued)"). experiment_purpose=diagnostic, claim_ids=[].

THE PROBLEM THE FIX ADDRESSES
-----------------------------
Modulatory / secondary score-bias channels (MECH-314 curiosity, MECH-320 vigor,
MECH-341 within-class entropy, dACC / lateral_pfc / ofc / MECH-295 cortical
biases) are added to E3 candidate scores but DOMINATED by the primary F+M+phi
terms: fixed small bias magnitudes (~0.05-0.1) added to primary scores whose
raw_score_range is much larger (2.0-5.0) never change the argmin. Failure
cluster the fix targets: V3-EXQ-604a (curiosity_bias=0.0), 624a (vigor
action_density byte-identical ON==OFF), 614d (within-class temperature
byte-identical across T=0.5/1.0/2.0). The fix rescales the SUMMED modulatory
contribution so its range == gain * raw_score_range BEFORE selection.

DESIGN (604a-based, MECH-314a per-candidate curiosity instrumentation)
----------------------------------------------------------------------
All three arms share an IDENTICAL agent config (SD-056 action-contrastive
substrate ON + MECH-314 structured curiosity ALL_ON + MECH-341 entropy bonus ON,
stratified-select OFF) EXCEPT the modulatory-authority axis -- so the authority
flag is the single isolated variable. SD-056 is trained online (569d harness) so
candidates are action-divergent and the per-candidate curiosity channel has real
inputs (the precondition 604a established). MECH-341 entropy bonus floors the
modulatory signal: SP-CEM yields >= 2 first-action classes every tick, so the
combined modulatory_total has a non-degenerate range to rescale even if the
curiosity novelty channel is degenerate on a given seed.

  ARM_A  use_modulatory_selection_authority=False  (baseline; modulatory
         channels drowned, expect ~604a FAIL replication: authority never fires)
  ARM_B  use_modulatory_selection_authority=True, modulatory_authority_gain=0.5
  ARM_C  use_modulatory_selection_authority=True, modulatory_authority_gain=0.8

INSTRUMENTATION NOTE (604a stale-key fix)
-----------------------------------------
V3-EXQ-604a read agent.curiosity.get_state()["last_bias_abs_mean"], a key that
does NOT exist on StructuredCuriosity.get_state() -- so its curiosity_bias_abs_mean
was structurally 0.0 from a missing key, independent of any drowning. This script
reads the REAL keys: last_bias_max_abs (curiosity bias magnitude) and
last_n_active_residue_centers (whether MECH-314a has inputs). The load-bearing
authority signal is the e3_selector diagnostic
selected_candidate_rank_before_bias: > 0 means the modulatory bias pulled a
NON-raw-optimal candidate into selection (i.e. modulatory channels actually won
the argmin). In ARM_A this stays ~0 (bias drowned); in ARM_B/C it rises (bias has
authority in near-ties).

=== ACCEPTANCE (pre-registered) ===

  C0 (MODULATORY NON-DEGENERACY GUARD): in ARM_B and ARM_C, raw modulatory bias
     is present -- mean score_bias_abs_mean > MODULATORY_NONDEGEN_FLOOR on
     >= MIN_SEEDS / SEEDS seeds in EACH ON arm. If the modulatory channel is
     genuinely zero (the residue-degenerate curiosity case the substrate note
     flags: "scaling zero is still zero"), the run is non_contributory -> route
     to curiosity / modulatory activation, NOT an authority falsification.
  C1 (MECHANISM ACTIVE): ARM_A modulatory_authority_active_frac ~ 0.0 (flag OFF,
     bit-identical baseline); ARM_B and ARM_C authority_active_frac >
     AUTHORITY_ACTIVE_FRAC_FLOOR AND modulatory_authority_scale_factor_mean >
     SCALE_FACTOR_MIN on >= MIN_SEEDS / SEEDS seeds. Confirms the rescaling block
     fires and amplifies the modulatory range (the design-doc criterion 2).
  C2 (AUTHORITY REAL / BEHAVIOURAL): ARM_B and ARM_C give modulatory channels
     genuine selection authority AND change behaviour vs ARM_A --
     (a) bias_changed_selection_frac in B/C exceeds A by > RANK_CHANGE_DELTA, AND
     (b) observable action-distribution change vs A (visited_cells relative
         change > VISITED_CELLS_REL_DELTA OR mean_episode_length change >
         EPISODE_LEN_REL_DELTA), on >= MIN_SEEDS / SEEDS seeds per ON arm.
  C3 (DOSE-RESPONSE, informative): ARM_C (gain 0.8) deviates from ARM_A more than
     ARM_B (gain 0.5) -- higher scale_factor, more rank changes, larger action-
     distribution delta. Reported; NOT required for overall PASS.

Overall PASS (authority mechanism validated) = C0 AND C1 AND C2.

=== DIAGNOSTIC INTERPRETATION GRID ===

| Outcome                         | Reading                                          | Next action |
|---------------------------------|--------------------------------------------------|-------------|
| C0 + C1 + C2 (+ C3) hold        | The authority substrate gives modulatory signals | Keystone fix CONFIRMED. Unblocks: sequence the GAP-2 scaffold-readiness run + the SD-057 cue-recall -> approach-lift re-test (640a autopsy), and the per-claim evidence retests of MECH-314 / MECH-320 / MECH-341. |
|                                 | genuine, bounded selection authority + behaviour |             |
|                                 | changes; dose-response monotone in gain.         |             |
| C0 + C1 hold, C2 fails          | Authority fires + rescales but does NOT change    | Authority too weak at these gains OR primary harm/goal gaps exceed gain*range everywhere (no near-ties in this env). Sweep gain higher / use a near-tie env. NOT a substrate bug. |
|                                 | selection/behaviour.                              |             |
| C0 holds, C1 fails              | Modulatory bias present but authority block never  | /diagnose-errors on the e3_selector authority wiring (modulatory_range below floor? flag not reaching E3Config?). Check modulatory_authority_min_range_floor. |
|                                 | fires in ON arms.                                |             |
| C0 fails (B/C raw bias ~ 0)     | Curiosity/modulatory channel genuinely degenerate | non_contributory. "Scaling zero is still zero." Route to curiosity activation (residue centers / 314a inputs) BEFORE re-running the authority test. NOT an authority falsification. |

architecture_epoch: "ree_hybrid_guardrails_v1"

Smoke:
  /opt/local/bin/python3 experiments/v3_exq_643_modulatory_authority_validation.py --dry-run
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
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402


EXPERIMENT_TYPE = "v3_exq_643_modulatory_authority_validation"
QUEUE_ID = "V3-EXQ-643"
CLAIM_IDS: List[str] = []  # diagnostic / substrate-readiness; weights no claim
EXPERIMENT_PURPOSE = "diagnostic"

SEEDS = [42, 43, 44]
P0_WARMUP_EPISODES = 60          # mirror 569d/604a extended warmup (E2 contrastive train)
P1_MEASUREMENT_EPISODES = 20
STEPS_PER_EPISODE = 200

DRY_RUN_SEEDS = [42]
DRY_RUN_P0 = 2
DRY_RUN_P1 = 2
DRY_RUN_STEPS = 30

# Acceptance thresholds (pre-registered).
MODULATORY_NONDEGEN_FLOOR = 1e-4    # C0: raw modulatory bias present (score_bias_abs_mean)
AUTHORITY_ACTIVE_FRAC_FLOOR = 0.5   # C1: authority fires on majority of ON-arm ticks
SCALE_FACTOR_MIN = 1.0              # C1: rescaling amplifies the modulatory range
RANK_CHANGE_DELTA = 0.05            # C2a: B/C bias_changed_selection_frac over A
VISITED_CELLS_REL_DELTA = 0.05      # C2b: relative change in visited_cells vs A
EPISODE_LEN_REL_DELTA = 0.05        # C2b: relative change in mean_episode_length vs A
MIN_SEEDS = 2                       # of 3

# Per-arm authority configuration (the single isolated variable).
ARMS: List[Dict[str, Any]] = [
    {
        "arm_id": "ARM_A",
        "label": "authority_off_baseline_604a_replication",
        "use_modulatory_selection_authority": False,
        "modulatory_authority_gain": 0.5,   # unused when flag is False
    },
    {
        "arm_id": "ARM_B",
        "label": "authority_on_gain_0_5",
        "use_modulatory_selection_authority": True,
        "modulatory_authority_gain": 0.5,
    },
    {
        "arm_id": "ARM_C",
        "label": "authority_on_gain_0_8",
        "use_modulatory_selection_authority": True,
        "modulatory_authority_gain": 0.8,
    },
]

# SD-056 online contrastive training (569d/604a harness).
SD056_WEIGHT = 0.05
E2_CONTRASTIVE_LR = 1e-3
E2_TRAIN_EVERY_K_TICKS = 1
CONTRASTIVE_BATCH_K = 8
TRANSITION_BUFFER_MAX = 256
MIN_BUFFER_BEFORE_TRAIN = 16
MIN_CLASSES_FOR_TRAIN = 2
MAX_GRAD_NORM = 1.0

# Curiosity magnitudes (604a calibration).
CURIOSITY_BIAS_SCALE = 0.5
CURIOSITY_WEIGHT = 0.25

# ENV identical to V3-EXQ-604a / 569d so manifest-comparability holds.
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


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(seed=seed, **ENV_KWARGS)


def _make_agent(env: CausalGridWorldV2, arm: Dict[str, Any]) -> REEAgent:
    """Main-path SP-CEM + V_s stack, SD-056 ON, MECH-314 curiosity ALL_ON,
    MECH-341 entropy bonus ON (stratified OFF). The ONLY per-arm difference is
    the modulatory-selection-authority axis.
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
        # ARC-065 SP-CEM (main-path default; >= 2 first-action classes/tick)
        use_support_preserving_cem=True,
        support_preserving_stratified_elites=True,
        support_preserving_ao_std_floor=0.2,
        support_preserving_min_first_action_classes=2,
        # V_s substrate (main-path default; identical across arms)
        use_per_stream_vs=True,
        use_per_region_vs=True,
        use_event_segmenter=True,
        use_invalidation_trigger=True,
        use_anchor_sets=True,
        # SD-056 -- uniform ON (the substrate that makes candidates action-divergent)
        e2_action_contrastive_enabled=True,
        e2_action_contrastive_weight=SD056_WEIGHT,
        # MECH-314 structured curiosity -- ALL_ON (headline modulatory channel)
        use_structured_curiosity=True,
        use_curiosity_novelty=True,
        use_curiosity_uncertainty=True,
        use_curiosity_learning_progress=True,
        curiosity_bias_scale=CURIOSITY_BIAS_SCALE,
        curiosity_novelty_weight=CURIOSITY_WEIGHT,
        curiosity_uncertainty_weight=CURIOSITY_WEIGHT,
        curiosity_learning_progress_weight=CURIOSITY_WEIGHT,
        # MECH-341 entropy bonus ON (floors the modulatory signal); stratified
        # select OFF so the SELECTION RULE is identical across arms -- the only
        # selection difference is the authority rescaling.
        use_e3_score_diversity=True,
        use_e3_diversity_entropy_bonus=True,
        use_e3_diversity_stratified_select=False,
        # modulatory-bias-selection-authority -- the isolated per-arm variable
        use_modulatory_selection_authority=bool(arm["use_modulatory_selection_authority"]),
        modulatory_authority_gain=float(arm["modulatory_authority_gain"]),
        modulatory_authority_min_range_floor=1e-6,
    )
    return REEAgent(cfg)


# ---------------------------------------------------------------------------
# SD-056 online-contrastive helpers (verbatim from V3-EXQ-604a / 569d)
# ---------------------------------------------------------------------------

def _obs(d: Dict[str, Any], key: str) -> Optional[torch.Tensor]:
    h = d.get(key)
    if h is None:
        return None
    return h.float().unsqueeze(0) if h.dim() == 1 else h.float()


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
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = _make_env(seed)
    agent = _make_agent(env, arm)

    e2_opt: torch.optim.Optimizer = torch.optim.Adam(
        agent.e2.parameters(), lr=E2_CONTRASTIVE_LR
    )

    transition_buffer: Deque[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ] = deque(maxlen=TRANSITION_BUFFER_MAX)
    sample_rng = random.Random(seed)

    total_train_eps = p0_episodes + p1_episodes

    # P1 accumulators.
    selected_class_counts: Counter = Counter()
    visited_cells: set = set()
    episode_lengths: List[int] = []

    # Authority diagnostics (P1).
    n_select_ticks = 0
    n_authority_active = 0
    scale_factor_values: List[float] = []
    n_bias_changed_selection = 0       # selected_candidate_rank_before_bias > 0
    n_rank_valid = 0                   # ticks where rank diag is valid (>= 0)
    score_bias_abs_means: List[float] = []

    # Curiosity diagnostics (P1) -- REAL get_state keys.
    curiosity_bias_max_abs_values: List[float] = []
    curiosity_active_residue_centers: List[float] = []
    curiosity_subflavours_fired: List[float] = []

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
        ep_steps = 0

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
                visited_cells.add((int(env.agent_x), int(env.agent_y)))
                n_select_ticks += 1

                diag = getattr(agent.e3, "last_score_diagnostics", {}) or {}
                if bool(diag.get("modulatory_authority_active", False)):
                    n_authority_active += 1
                    sf = diag.get("modulatory_authority_scale_factor")
                    if sf is not None and math.isfinite(float(sf)) and float(sf) > 0.0:
                        scale_factor_values.append(float(sf))
                sbm = diag.get("score_bias_abs_mean")
                if sbm is not None and math.isfinite(float(sbm)):
                    score_bias_abs_means.append(float(sbm))
                rank_before = diag.get("selected_candidate_rank_before_bias", -1)
                if rank_before is not None and int(rank_before) >= 0:
                    n_rank_valid += 1
                    if int(rank_before) > 0:
                        n_bias_changed_selection += 1

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

            if tick_in_ep % E2_TRAIN_EVERY_K_TICKS == 0:
                _e2_contrastive_step(
                    agent=agent,
                    buffer=transition_buffer,
                    arm_weight=SD056_WEIGHT,
                    optimiser=e2_opt,
                    rng=sample_rng,
                )

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
            ep_steps += 1
            if done:
                break

        if is_p1:
            episode_lengths.append(int(ep_steps))

        if (ep + 1) % 10 == 0 or (ep + 1) == total_train_eps:
            print(
                f"  [train] arm={arm['arm_id']} seed={seed} phase={phase_label} "
                f"ep {ep + 1}/{total_train_eps}",
                flush=True,
            )

    def _mean(xs: List[float], default: float = 0.0) -> float:
        return float(sum(xs) / len(xs)) if xs else default

    selected_action_entropy = _entropy_from_counts(dict(selected_class_counts))
    authority_active_frac = (
        float(n_authority_active) / n_select_ticks if n_select_ticks > 0 else 0.0
    )
    bias_changed_selection_frac = (
        float(n_bias_changed_selection) / n_rank_valid if n_rank_valid > 0 else 0.0
    )

    return {
        "arm_id": arm["arm_id"],
        "label": arm["label"],
        "seed": int(seed),
        "use_modulatory_selection_authority": bool(arm["use_modulatory_selection_authority"]),
        "modulatory_authority_gain": float(arm["modulatory_authority_gain"]),
        "n_p1_select_ticks": int(n_select_ticks),
        "error_note": error_note,
        # Authority mechanism diagnostics.
        "modulatory_authority_active_frac": round(authority_active_frac, 6),
        "modulatory_authority_scale_factor_mean": round(_mean(scale_factor_values), 6),
        "score_bias_abs_mean": round(_mean(score_bias_abs_means), 6),
        "bias_changed_selection_frac": round(bias_changed_selection_frac, 6),
        # Curiosity-channel diagnostics (REAL keys; non-degeneracy guard inputs).
        "curiosity_bias_max_abs_mean": round(_mean(curiosity_bias_max_abs_values), 6),
        "curiosity_active_residue_centers_mean": round(_mean(curiosity_active_residue_centers), 6),
        "curiosity_subflavours_fired_mean": round(_mean(curiosity_subflavours_fired), 6),
        # Behavioural / action-distribution.
        "visited_cells": int(len(visited_cells)),
        "mean_episode_length": round(_mean([float(x) for x in episode_lengths]), 6),
        "selected_action_class_entropy": round(selected_action_entropy, 6),
        "selected_classes_n_unique": int(len(selected_class_counts)),
        "selected_class_counts": dict(sorted(selected_class_counts.items())),
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


def _seeds_with_behavioural_change(
    on_rows: List[Dict[str, Any]],
    off_by_seed: Dict[int, Dict[str, Any]],
) -> int:
    """Count ON-arm seeds whose action-distribution differs from the matched
    ARM_A seed beyond the relative-delta floor (visited_cells OR episode length).
    """
    n = 0
    for r in on_rows:
        ref = off_by_seed.get(int(r["seed"]))
        if ref is None:
            continue
        vc_ref = max(float(ref.get("visited_cells", 0.0)), 1.0)
        el_ref = max(float(ref.get("mean_episode_length", 0.0)), 1.0)
        vc_rel = abs(float(r.get("visited_cells", 0.0)) - vc_ref) / vc_ref
        el_rel = abs(float(r.get("mean_episode_length", 0.0)) - el_ref) / el_ref
        if vc_rel > VISITED_CELLS_REL_DELTA or el_rel > EPISODE_LEN_REL_DELTA:
            n += 1
    return n


def _seeds_rank_change_over_off(
    on_rows: List[Dict[str, Any]],
    off_by_seed: Dict[int, Dict[str, Any]],
) -> int:
    """Count ON-arm seeds whose bias_changed_selection_frac exceeds the matched
    ARM_A seed by > RANK_CHANGE_DELTA.
    """
    n = 0
    for r in on_rows:
        ref = off_by_seed.get(int(r["seed"]))
        if ref is None:
            continue
        delta = float(r.get("bias_changed_selection_frac", 0.0)) - float(
            ref.get("bias_changed_selection_frac", 0.0)
        )
        if delta > RANK_CHANGE_DELTA:
            n += 1
    return n


def _evaluate(arm_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    a_rows = _arm_rows(arm_results, "ARM_A")
    b_rows = _arm_rows(arm_results, "ARM_B")
    c_rows = _arm_rows(arm_results, "ARM_C")
    off_by_seed = {int(r["seed"]): r for r in a_rows}

    # C0: modulatory non-degeneracy in EACH ON arm.
    c0_b = _n_seeds_above(b_rows, "score_bias_abs_mean", MODULATORY_NONDEGEN_FLOOR)
    c0_c = _n_seeds_above(c_rows, "score_bias_abs_mean", MODULATORY_NONDEGEN_FLOOR)
    c0_pass = (c0_b >= MIN_SEEDS) and (c0_c >= MIN_SEEDS)

    # C1: mechanism active (ON arms fire + scale > 1) AND OFF arm silent.
    def _c1_arm(rows: List[Dict[str, Any]]) -> int:
        return sum(
            1 for r in rows
            if float(r.get("modulatory_authority_active_frac", 0.0)) > AUTHORITY_ACTIVE_FRAC_FLOOR
            and float(r.get("modulatory_authority_scale_factor_mean", 0.0)) > SCALE_FACTOR_MIN
        )
    c1_b = _c1_arm(b_rows)
    c1_c = _c1_arm(c_rows)
    a_active_frac_mean = _mean_key(a_rows, "modulatory_authority_active_frac")
    c1_off_silent = a_active_frac_mean < 1e-9
    c1_pass = (c1_b >= MIN_SEEDS) and (c1_c >= MIN_SEEDS) and c1_off_silent

    # C2: authority real + behavioural change vs ARM_A (per matched seed).
    c2_rank_b = _seeds_rank_change_over_off(b_rows, off_by_seed)
    c2_rank_c = _seeds_rank_change_over_off(c_rows, off_by_seed)
    c2_beh_b = _seeds_with_behavioural_change(b_rows, off_by_seed)
    c2_beh_c = _seeds_with_behavioural_change(c_rows, off_by_seed)
    c2_pass = (
        c2_rank_b >= MIN_SEEDS and c2_rank_c >= MIN_SEEDS
        and c2_beh_b >= MIN_SEEDS and c2_beh_c >= MIN_SEEDS
    )

    # C3 (informative): dose-response C > B.
    b_scale = _mean_key(b_rows, "modulatory_authority_scale_factor_mean")
    c_scale = _mean_key(c_rows, "modulatory_authority_scale_factor_mean")
    b_rank = _mean_key(b_rows, "bias_changed_selection_frac")
    c_rank = _mean_key(c_rows, "bias_changed_selection_frac")
    c3_pass = (c_scale > b_scale) and (c_rank >= b_rank)

    overall_pass = bool(c0_pass and c1_pass and c2_pass)

    return {
        "MODULATORY_NONDEGEN_FLOOR": MODULATORY_NONDEGEN_FLOOR,
        "AUTHORITY_ACTIVE_FRAC_FLOOR": AUTHORITY_ACTIVE_FRAC_FLOOR,
        "SCALE_FACTOR_MIN": SCALE_FACTOR_MIN,
        "RANK_CHANGE_DELTA": RANK_CHANGE_DELTA,
        "MIN_SEEDS_REQUIRED": MIN_SEEDS,
        # C0
        "c0_modulatory_nondegen_seeds_B": c0_b,
        "c0_modulatory_nondegen_seeds_C": c0_c,
        "c0_pass": bool(c0_pass),
        # C1
        "c1_mechanism_active_seeds_B": c1_b,
        "c1_mechanism_active_seeds_C": c1_c,
        "c1_off_arm_authority_active_frac_mean": round(a_active_frac_mean, 9),
        "c1_off_silent": bool(c1_off_silent),
        "c1_pass": bool(c1_pass),
        # C2
        "c2_rank_change_seeds_B": c2_rank_b,
        "c2_rank_change_seeds_C": c2_rank_c,
        "c2_behavioural_change_seeds_B": c2_beh_b,
        "c2_behavioural_change_seeds_C": c2_beh_c,
        "c2_pass": bool(c2_pass),
        # C3
        "c3_scale_factor_B_mean": round(b_scale, 6),
        "c3_scale_factor_C_mean": round(c_scale, 6),
        "c3_bias_changed_selection_frac_B_mean": round(b_rank, 6),
        "c3_bias_changed_selection_frac_C_mean": round(c_rank, 6),
        "c3_dose_response_pass": bool(c3_pass),
        # Per-arm summaries.
        "per_arm_means": {
            arm_id: {
                "modulatory_authority_active_frac": round(_mean_key(_arm_rows(arm_results, arm_id), "modulatory_authority_active_frac"), 6),
                "modulatory_authority_scale_factor_mean": round(_mean_key(_arm_rows(arm_results, arm_id), "modulatory_authority_scale_factor_mean"), 6),
                "score_bias_abs_mean": round(_mean_key(_arm_rows(arm_results, arm_id), "score_bias_abs_mean"), 6),
                "bias_changed_selection_frac": round(_mean_key(_arm_rows(arm_results, arm_id), "bias_changed_selection_frac"), 6),
                "curiosity_bias_max_abs_mean": round(_mean_key(_arm_rows(arm_results, arm_id), "curiosity_bias_max_abs_mean"), 6),
                "curiosity_active_residue_centers_mean": round(_mean_key(_arm_rows(arm_results, arm_id), "curiosity_active_residue_centers_mean"), 6),
                "visited_cells": round(_mean_key(_arm_rows(arm_results, arm_id), "visited_cells"), 6),
                "mean_episode_length": round(_mean_key(_arm_rows(arm_results, arm_id), "mean_episode_length"), 6),
                "selected_action_class_entropy": round(_mean_key(_arm_rows(arm_results, arm_id), "selected_action_class_entropy"), 6),
            }
            for arm_id in ("ARM_A", "ARM_B", "ARM_C")
        },
        "overall_pass": overall_pass,
    }


def _evidence_direction(summary: Dict[str, Any]) -> str:
    if summary["overall_pass"]:
        return "supports"
    if not summary["c0_pass"]:
        return "non_contributory"
    return "does_not_support"


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
            arm_results.append(cell)
            passed = cell.get("error_note") is None
            print(f"verdict: {'PASS' if passed else 'FAIL'}", flush=True)

    summary = _evaluate(arm_results)
    outcome = "PASS" if summary["overall_pass"] else "FAIL"
    edir = _evidence_direction(summary)

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
        "evidence_direction": edir,
        "evidence_direction_note": (
            "modulatory-bias-selection-authority falsification (604a-based). "
            "Three arms share an identical curiosity-ALL_ON + MECH-341-entropy + "
            "SD-056 config; the ONLY difference is use_modulatory_selection_authority "
            "(A=False / B=True gain0.5 / C=True gain0.8). PASS = C0 modulatory "
            "non-degeneracy AND C1 authority fires (scale_factor>1, OFF-arm silent) "
            "AND C2 authority changes selection (bias_changed_selection_frac) + "
            "action distribution (visited_cells / episode_length) vs ARM_A. "
            "C0 FAIL = curiosity/modulatory channel genuinely zero ('scaling zero "
            "is still zero') -> non_contributory, route to curiosity activation, "
            "NOT an authority falsification. Diagnostic (claim_ids=[]); does not "
            "weight MECH-314/320/341 confidence -- it unblocks their per-claim "
            "evidence retests."
        ),
        "dry_run": bool(dry_run),
        "config": {
            "seeds": seeds,
            "p0_warmup_episodes": p0,
            "p1_measurement_episodes": p1,
            "steps_per_episode": steps,
            "env_kwargs": ENV_KWARGS,
            "arms": [
                {
                    "arm_id": a["arm_id"],
                    "label": a["label"],
                    "use_modulatory_selection_authority": a["use_modulatory_selection_authority"],
                    "modulatory_authority_gain": a["modulatory_authority_gain"],
                }
                for a in ARMS
            ],
            "sd056_weight": SD056_WEIGHT,
            "curiosity_bias_scale": CURIOSITY_BIAS_SCALE,
            "curiosity_weight": CURIOSITY_WEIGHT,
            "thresholds": {
                "MODULATORY_NONDEGEN_FLOOR": MODULATORY_NONDEGEN_FLOOR,
                "AUTHORITY_ACTIVE_FRAC_FLOOR": AUTHORITY_ACTIVE_FRAC_FLOOR,
                "SCALE_FACTOR_MIN": SCALE_FACTOR_MIN,
                "RANK_CHANGE_DELTA": RANK_CHANGE_DELTA,
                "VISITED_CELLS_REL_DELTA": VISITED_CELLS_REL_DELTA,
                "EPISODE_LEN_REL_DELTA": EPISODE_LEN_REL_DELTA,
                "MIN_SEEDS": MIN_SEEDS,
            },
        },
        "acceptance_criteria": {
            "C0_modulatory_non_degeneracy": summary["c0_pass"],
            "C1_authority_mechanism_active": summary["c1_pass"],
            "C2_authority_changes_selection_and_behaviour": summary["c2_pass"],
            "C3_dose_response_informative": summary["c3_dose_response_pass"],
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

    print(f"Outcome: {outcome}", flush=True)
    for k, v in manifest["acceptance_criteria"].items():
        print(f"  {k}: {v}", flush=True)

    manifest["manifest_path"] = str(out_path)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="V3-EXQ-643 modulatory-bias-selection-authority falsification"
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
