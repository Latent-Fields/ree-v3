"""Canonical OFF/baseline arm for the V3-EXQ-643 modulatory-authority lineage.

Arm-reuse Phase 0 (instrument-only). Design plan:
REE_assembly/evidence/planning/arm_reuse_fingerprint_plan.md (sections 2, 7b).

WHAT THIS MODULE IS
-------------------
The single source of truth for the OFF / baseline arm of the modulatory-bias-
selection-authority validation lineage (V3-EXQ-643 -> 643a -> ...). 643a's ARM_A
("authority_off_baseline", use_modulatory_selection_authority=False) was a
hand-declared slice of the 643a script; here it is extracted verbatim so its
identity is **content-hashed** (this file is matched by the arm-fingerprint
substrate glob `experiments/_lib/**/*.py`, so any change to it correctly flips
the substrate hash and *refuses* a stale reuse).

THE CONTRACT (the part that must be exactly right)
--------------------------------------------------
A future 643b / 643c that wants to *reuse* a baseline minted from this module
MUST construct its OFF arm by importing `run_off_cell` (and the env / agent
builders) from here -- NOT by re-deriving the OFF path inline. The OFF arm's
computation is the conjunction of:
  * ENV_KWARGS                         -- identical to V3-EXQ-604a / 569d / 643/643a
  * the training schedule              -- P0_WARMUP_EPISODES=60, P1_MEASUREMENT_EPISODES=20,
                                          STEPS_PER_EPISODE=200, SEEDS=[42,43,44]
  * the substrate-operating config     -- CONFIG_FLAGS: SD-056 online contrastive
    (fires for ALL arms)                  + MECH-314 curiosity ALL_ON + MECH-341 entropy
                                          bonus + ARC-065 SP-CEM + V_s substrate, etc.
  * the SD-056 online-training params  -- SD056_WEIGHT, E2_CONTRASTIVE_LR, ...
  * the OFF arm's own flags            -- use_modulatory_selection_authority=False
                                          (gate inert), modulatory_authority_gain=0.5
                                          (unused when off), min_range_floor=1e-6.
It does **NOT** depend on the ON arms' gains (ARM_B 0.5 / ARM_C 0.8), the
acceptance thresholds, or the arm labels -- so none of those appear in the
fingerprint slice (`off_path_config_slice`). 643->643a changed only the ARM_A
label string; the OFF-path slice is invariant to that, which is exactly why a
narrowed canonical slice (not whole-config) is what makes cross-iteration reuse
hit (plan 7b constraint 2).

FIDELITY NOTE (reset_all_rng vs 643a's partial seed)
----------------------------------------------------
643a's `_run_seed_arm` seeded only `torch.manual_seed(seed)` + `np.random.seed(seed)`
at cell entry (and ARM_A always ran first, so global state was fresh). `run_off_cell`
here performs the COMPLETE per-cell reset (`reset_all_rng(seed)` -- torch+cuda+numpy+
python-random+harness `_action_random`) at cell entry. This is a strict superset:
the OFF loop draws only from torch (agent init + forward), numpy
(`np.random.randint` action fallback), and a LOCAL `random.Random(seed)` for
contrastive batch sampling -- it never consumes Python's global `random` or the
harness `_action_random`. So the extra resets touch RNGs the OFF arm never reads;
the computation is identical to 643a's ARM_A while becoming order-independent
(reuse_eligible). A reusing 643b must likewise reset at cell entry (the
/queue-experiment skill now requires reset_all_rng per cell for all multi-arm
experiments), so the two cells are the same random variable.

ASCII-only output (repo rule).
"""

from __future__ import annotations

import math
import random
import sys
from collections import Counter, deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import torch

# Make ree-v3 importable when this module is imported standalone.
_REPO_ROOT = Path(__file__).resolve().parents[3]  # experiments/_lib/baselines/ -> ree-v3
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments._lib.arm_fingerprint import reset_all_rng  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402

CANONICAL_BASELINE_ID = "exq643_modulatory_authority_off"
LINEAGE = "v3_exq_643_modulatory_authority_validation"

# --- Schedule (mirror 569d / 604a / 643 / 643a) ---------------------------------
SEEDS: List[int] = [42, 43, 44]
P0_WARMUP_EPISODES = 60          # extended warmup (E2 contrastive train)
P1_MEASUREMENT_EPISODES = 20
STEPS_PER_EPISODE = 200

# --- SD-056 online contrastive training (569d/604a/643 harness) + 643a clamp ----
SD056_WEIGHT = 0.05
E2_CONTRASTIVE_LR = 1e-3
E2_TRAIN_EVERY_K_TICKS = 1
CONTRASTIVE_BATCH_K = 8
TRANSITION_BUFFER_MAX = 256
MIN_BUFFER_BEFORE_TRAIN = 16
MIN_CLASSES_FOR_TRAIN = 2
MAX_GRAD_NORM = 1.0
ROLLOUT_CLAMP_RATIO = 2.0          # 643a: bound rollout z_world norm vs initial (SD-056 amend)

# --- Curiosity magnitudes (604a calibration) ------------------------------------
CURIOSITY_BIAS_SCALE = 0.5
CURIOSITY_WEIGHT = 0.25

# --- ENV identical to V3-EXQ-604a / 569d / 643 / 643a ---------------------------
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

# --- The canonical OFF arm (643a ARM_A, gate inert) -----------------------------
ARM_OFF: Dict[str, Any] = {
    "arm_id": "ARM_A",
    "label": "authority_off_baseline",
    "use_modulatory_selection_authority": False,
    "modulatory_authority_gain": 0.5,   # unused when the flag is False
}

# --- Substrate-operating config the OFF arm executes (fires for ALL arms) --------
# These are the explicit REEConfig.from_dims kwargs from 643a:_make_agent, with the
# OFF arm's own flags substituted. The env-dimension-derived kwargs (body_obs_dim,
# world_obs_dim, action_dim) are NOT here -- they are read from the env in
# make_off_agent(). Keeping the from_dims kwargs in ONE dict means the fingerprint
# slice and the actual agent build cannot drift apart.
CONFIG_FLAGS: Dict[str, Any] = dict(
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
    # SD-056 uniform ON + 643a rollout clamp (bound online-trained E2 rollouts)
    e2_action_contrastive_enabled=True,
    e2_action_contrastive_weight=SD056_WEIGHT,
    e2_rollout_output_norm_clamp_enabled=True,
    e2_rollout_output_norm_clamp_ratio=ROLLOUT_CLAMP_RATIO,
    # MECH-314 structured curiosity ALL_ON (z_world-derived modulatory channel)
    use_structured_curiosity=True,
    use_curiosity_novelty=True,
    use_curiosity_uncertainty=True,
    use_curiosity_learning_progress=True,
    curiosity_bias_scale=CURIOSITY_BIAS_SCALE,
    curiosity_novelty_weight=CURIOSITY_WEIGHT,
    curiosity_uncertainty_weight=CURIOSITY_WEIGHT,
    curiosity_learning_progress_weight=CURIOSITY_WEIGHT,
    # MECH-341 entropy bonus ON (cross-candidate range carrier); stratified select OFF
    use_e3_score_diversity=True,
    use_e3_diversity_entropy_bonus=True,
    use_e3_diversity_stratified_select=False,
    # modulatory-bias-selection-authority -- the OFF arm leaves the gate inert
    use_modulatory_selection_authority=bool(ARM_OFF["use_modulatory_selection_authority"]),
    modulatory_authority_gain=float(ARM_OFF["modulatory_authority_gain"]),
    modulatory_authority_min_range_floor=1e-6,
)


def make_off_env(seed: int) -> CausalGridWorldV2:
    """Build the OFF arm's env (identical to 643a _make_env)."""
    return CausalGridWorldV2(seed=seed, **ENV_KWARGS)


def make_off_agent(env: CausalGridWorldV2) -> REEAgent:
    """Build the OFF arm's agent (identical to 643a _make_agent with ARM_A flags).

    Main-path SP-CEM + V_s stack, SD-056 ON with rollout-norm clamp, MECH-314
    curiosity ALL_ON, MECH-341 entropy bonus ON (stratified OFF). The
    modulatory-selection-authority gate is inert (OFF arm).
    """
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        **CONFIG_FLAGS,
    )
    return REEAgent(cfg)


def off_path_config_slice(
    p0: int = P0_WARMUP_EPISODES,
    p1: int = P1_MEASUREMENT_EPISODES,
    steps: int = STEPS_PER_EPISODE,
) -> Dict[str, Any]:
    """The declared OFF-path config slice for the arm fingerprint.

    Deliberately NARROWED (plan decision 3 opt-in narrowing): only the params the
    OFF computation reads -- env_kwargs, schedule, the substrate-operating config
    (CONFIG_FLAGS, which already carries the OFF flags), and the SD-056 / curiosity
    run-loop constants. NO arm labels, NO ON-arm gains, NO acceptance thresholds --
    those do not change the OFF draw, so excluding them lets a 643b OFF arm match.
    """
    return {
        "baseline_id": CANONICAL_BASELINE_ID,
        "lineage": LINEAGE,
        "env_kwargs": dict(ENV_KWARGS),
        "schedule": {"p0": int(p0), "p1": int(p1), "steps_per_episode": int(steps)},
        "config_flags": dict(CONFIG_FLAGS),
        "sd056": {
            "weight": SD056_WEIGHT,
            "lr": E2_CONTRASTIVE_LR,
            "train_every_k_ticks": E2_TRAIN_EVERY_K_TICKS,
            "batch_k": CONTRASTIVE_BATCH_K,
            "buffer_max": TRANSITION_BUFFER_MAX,
            "min_buffer_before_train": MIN_BUFFER_BEFORE_TRAIN,
            "min_classes_for_train": MIN_CLASSES_FOR_TRAIN,
            "max_grad_norm": MAX_GRAD_NORM,
            "rollout_clamp_ratio": ROLLOUT_CLAMP_RATIO,
        },
        "curiosity": {"bias_scale": CURIOSITY_BIAS_SCALE, "weight": CURIOSITY_WEIGHT},
        "off_arm_flags": {
            "use_modulatory_selection_authority": bool(ARM_OFF["use_modulatory_selection_authority"]),
            "modulatory_authority_gain": float(ARM_OFF["modulatory_authority_gain"]),
            "modulatory_authority_min_range_floor": 1e-6,
        },
    }


# ---------------------------------------------------------------------------
# SD-056 online-contrastive helpers (verbatim from V3-EXQ-604a / 643 / 643a)
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
# Canonical OFF-arm per-seed runner (verbatim 643a ARM_A body; complete RNG reset)
# ---------------------------------------------------------------------------

def run_off_cell(
    seed: int,
    p0_episodes: int = P0_WARMUP_EPISODES,
    p1_episodes: int = P1_MEASUREMENT_EPISODES,
    steps_per_episode: int = STEPS_PER_EPISODE,
    arm: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run the canonical OFF/baseline cell for one seed and return its result row.

    Computation-identical to 643a `_run_seed_arm(ARM_A, seed, ...)`. The ONLY
    difference is that cell entry calls the COMPLETE RNG reset (reset_all_rng)
    instead of 643a's partial torch+numpy seed -- a strict superset that leaves
    the OFF draw unchanged (see module docstring FIDELITY NOTE) while making the
    cell order-independent so a Phase-0 fingerprint can mark it reuse_eligible.
    """
    if arm is None:
        arm = ARM_OFF

    # --- COMPLETE per-cell RNG reset at cell entry (order-independence) ---
    reset_all_rng(seed)

    env = make_off_env(seed)
    agent = make_off_agent(env)

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

    # 643a readiness diagnostics (P1).
    raw_score_ranges: List[float] = []                 # e3_raw_score_range_mean per tick
    n_raw_bounded = 0                                   # ticks with raw_range < bound
    modulatory_range_positive_control: List[float] = []  # range on >=2-class ticks (ON arms)

    # Curiosity diagnostics (P1) -- REAL get_state keys.
    curiosity_bias_max_abs_values: List[float] = []
    curiosity_active_residue_centers: List[float] = []

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
            n_first_action_classes = len({
                int(c.actions[:, 0, :].argmax(dim=-1).flatten()[0].item())
                for c in candidates
            }) if candidates else 0

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

                # 643a readiness signals.
                rsr = diag.get("e3_raw_score_range_mean")
                if rsr is not None and math.isfinite(float(rsr)):
                    raw_score_ranges.append(float(rsr))
                    if float(rsr) < 1e3:
                        n_raw_bounded += 1
                mar = diag.get("modulatory_authority_range")
                # Positive control for P-RANGE: ON arms, pool with >= 2 first-action
                # classes. The OFF arm never enters this branch (flag False), so the
                # list stays empty -- identical to 643a ARM_A.
                if (
                    bool(arm["use_modulatory_selection_authority"])
                    and n_first_action_classes >= 2
                    and mar is not None
                    and math.isfinite(float(mar))
                ):
                    modulatory_range_positive_control.append(float(mar))

                cur = getattr(agent, "curiosity", None)
                if cur is not None:
                    st = cur.get_state()
                    bmax = st.get("last_bias_max_abs")
                    if bmax is not None and math.isfinite(float(bmax)):
                        curiosity_bias_max_abs_values.append(float(bmax))
                    nrc = st.get("last_n_active_residue_centers")
                    if nrc is not None:
                        curiosity_active_residue_centers.append(float(nrc))

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
    raw_bounded_frac = (
        float(n_raw_bounded) / n_select_ticks if n_select_ticks > 0 else 0.0
    )

    return {
        "arm_id": arm["arm_id"],
        "label": arm["label"],
        "seed": int(seed),
        "use_modulatory_selection_authority": bool(arm["use_modulatory_selection_authority"]),
        "modulatory_authority_gain": float(arm["modulatory_authority_gain"]),
        "n_p1_select_ticks": int(n_select_ticks),
        "error_note": error_note,
        # Authority mechanism diagnostics (OFF arm: expected ~0 / inert).
        "modulatory_authority_active_frac": round(authority_active_frac, 6),
        "modulatory_authority_scale_factor_mean": round(_mean(scale_factor_values), 6),
        "modulatory_authority_range_mean": round(_mean(modulatory_range_positive_control), 6),
        "score_bias_abs_mean": round(_mean(score_bias_abs_means), 6),
        "bias_changed_selection_frac": round(bias_changed_selection_frac, 6),
        # 643a readiness diagnostics.
        "raw_score_range_mean": round(_mean(raw_score_ranges), 6),
        "raw_score_range_max": round(max(raw_score_ranges), 6) if raw_score_ranges else 0.0,
        "raw_bounded_frac": round(raw_bounded_frac, 6),
        "n_positive_control_ticks": int(len(modulatory_range_positive_control)),
        # Curiosity-channel diagnostics (REAL keys).
        "curiosity_bias_max_abs_mean": round(_mean(curiosity_bias_max_abs_values), 6),
        "curiosity_active_residue_centers_mean": round(_mean(curiosity_active_residue_centers), 6),
        # Behavioural / action-distribution.
        "visited_cells": int(len(visited_cells)),
        "mean_episode_length": round(_mean([float(x) for x in episode_lengths]), 6),
        "selected_action_class_entropy": round(selected_action_entropy, 6),
        "selected_classes_n_unique": int(len(selected_class_counts)),
        "selected_class_counts": dict(sorted(selected_class_counts.items())),
    }


__all__ = [
    "CANONICAL_BASELINE_ID",
    "LINEAGE",
    "SEEDS",
    "P0_WARMUP_EPISODES",
    "P1_MEASUREMENT_EPISODES",
    "STEPS_PER_EPISODE",
    "ENV_KWARGS",
    "ARM_OFF",
    "CONFIG_FLAGS",
    "make_off_env",
    "make_off_agent",
    "off_path_config_slice",
    "run_off_cell",
]
