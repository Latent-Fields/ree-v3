#!/opt/local/bin/python3
"""
V3-EXQ-554 -- Decoder + elite-refit collapse localization diagnostic.

Claims: none (proposer-fix probe; experiment_purpose=diagnostic; claim_ids=[]).

Purpose
-------
V3-EXQ-551a classified ``rollout_only_collapse`` in all 3 seeds:
    stage 1 ~2e-4, stage 2 < 1e-6, stage 3 ~1e-4-9e-4, stage 4 entropy=0.
V3-EXQ-553 (orthogonal CEM seeding) FAILed -- ARM_ORTHO stage1
indistinguishable from ARM_IID (~2e-4), even though
``ortho_diag_last`` confirmed the substrate fired with
``n_orthogonal_iters=3, n_iid_fallback_candidates=0``.

The collapse therefore happens BETWEEN the orthogonal noise injection
and the stage-1 measurement -- inside HippocampalModule.propose_trajectories.
This experiment localizes WHICH operation collapses the diversity by
instrumenting 5 measurement points along the per-iteration data path:

    M0  noise tensor (post-sampling, pre-add)
    M1  candidates_ao = ao_mean + ao_std * noise (post-add)
    M2  decoded actions (post-action_object_decoder)
    M3  E2 rollout final world_states (cross-check with 551a stage 2)
    M4  CEM iter-1 candidates_ao (post-elite-refit), measured on the
        SECOND CEM iteration's noise+add cycle

The 5 points are diff-able: a large gap between M_i and M_{i+1}
identifies the offending operation. M0-M2 are first-iteration only;
M3 is the first-iteration trajectory result; M4 is the second-iteration
ao tensor immediately after the refit + new noise + add step (i.e. it
captures what elite-refit did to the candidate diversity).

Implementation
--------------
Monkey-patches HippocampalModule.propose_trajectories at the experiment-
script level with an inlined copy that records the per-iteration tensors
into a logging dict. No ree_core edits. The patched copy preserves all
substrate semantics that EXQ-553 / EXQ-551a relied on:
  - MECH-267 operating_mode noise scale (off by default; preserved)
  - V3-EXQ-553 use_orthogonal_cem_seeding (consumed; flag-gated)
  - MECH-293 ghost-goal probes (off by default for this experiment;
    branch present but inert with use_mech293_ghost_probes=False)
  - MECH-269 / SD-029 / per-stream V_s flags wired identically to
    EXQ-553 so cross-comparable to the EXQ-553 manifest.

Arms
----
  ARM_IID:   use_orthogonal_cem_seeding=False (baseline).
  ARM_ORTHO: use_orthogonal_cem_seeding=True; otherwise identical.

3 seeds [42, 7, 17] x 2 arms = 6 cells, no training (P0=20 episodes
at 30 ticks/ep; the cliff exists at depth 0 per EXQ-551a / 553).
Mirrors EXQ-553 env wiring exactly so manifests are cross-comparable.

Metrics per (seed, arm)
-----------------------
For each measurement point M in {M0, M1, M2, M3, M4}, averaged across
per-tick samples in the run:
    M_pairwise_l2_mean   -- mean pairwise L2 across K candidates
    M_min_pairwise_l2_mean -- mean over ticks of min pairwise L2
    M_elementwise_var_mean -- mean elementwise variance across K
    M_n_samples           -- number of fresh-candidate ticks that
                              contributed (sanity).
plus end-to-end metrics also recorded in EXQ-553:
    stage1_pairwise_l2_mean  -- mirror of M2 first-step (sanity vs 553)
    action_class_entropy     -- the EXQ-550 monostrategy metric
    action_class_counts      -- raw histogram
    ortho_diag_last          -- substrate echo for ARM_ORTHO

Pre-registered interpretation grid (5 rows, baked into manifest)
----------------------------------------------------------------
R1 decoder_collapse:
   ARM_ORTHO M0 and M1 pairwise L2 > 1e-2 (substrate-fired diversity is
   real at noise + post-add levels) but M2 < 1e-3 (post-decoder
   collapses) -> action_object_decoder is the cliff. Next: probe the
   decoder weight singular values / output saturation.

R2 elite_refit_collapse:
   M0/M1/M2 all > 1e-2 (proposer-side diversity preserved through
   decode) but M4 < 1e-3 (elite-refit step collapses the candidate
   distribution by iteration 2) -> CEM elite-refit is the cliff.
   Next: probe elite_fraction / refit_std formula / single-elite
   degeneracy.

R3 noise_shape_irrelevant:
   ARM_ORTHO M0 approximately equal to ARM_IID M0 -> the orthogonal
   sampler is not actually producing distinct candidates at the noise
   tensor level (rescale broken, basis not applied, etc.). Next:
   re-audit ree_core orthogonal sampler implementation. This is the
   R3 of EXQ-553 surfaced inside the per-iteration view.

R4 e2_rollout_collapse:
   M0/M1/M2/M4 all preserve > 1e-2 diversity but M3 < 1e-6 (matches
   EXQ-551a stage 2 cliff) -> E2 dynamics maps diverse actions to
   near-identical world_state outcomes. Next: probe E2 forward
   weight rank / output saturation; route to E2-conditioning fix.

R5 other:
   Any pattern that does not match R1-R4. Route to /diagnose-errors.

Note on R1+R3 interaction: M0 captures the noise tensor BEFORE any
ao_std scaling. If ARM_ORTHO M0 already matches ARM_IID M0, R3 fires
first regardless of M1/M2/M3 outcome. The interpretation order is
R3 -> R1 -> R4 -> R2 -> R5.

experiment_purpose=diagnostic (substrate localization; output routes
to /diagnose-errors or claim-bearing follow-up depending on which row
fires).

claim_ids=[] intentionally: no single claim under test.

See ree-v3/CLAUDE.md MECH-269 / SD-029 sections; EXQ-553 manifest in
REE_assembly/evidence/experiments/v3_exq_553_orthogonal_cem_seeding/.
See REE_assembly/evidence/planning/arc_062_rule_apprehension_plan.md
for the cluster context.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from experiment_protocol import emit_outcome
from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.hippocampal.module import HippocampalModule
from ree_core.predictors.e2_fast import Trajectory
from ree_core.utils.config import REEConfig
from experiments.pack_writer import write_flat_manifest  # noqa: E402


EXPERIMENT_TYPE = "v3_exq_554_decoder_elite_refit_collapse_localization"
QUEUE_ID = "V3-EXQ-554"
CLAIM_IDS: List[str] = []
EXPERIMENT_PURPOSE = "diagnostic"

SEEDS = [42, 7, 17]
ARMS = ("ARM_IID", "ARM_ORTHO")
P0_EPISODES = 20
EPISODE_STEPS = 30

POINTS = ("M0", "M1", "M2", "M3", "M4")

# Used by the R-grid interpreter only as orientation thresholds; the
# manifest records the raw numbers so reviewers can re-classify.
LARGE_DIVERSITY = 1e-2
SMALL_DIVERSITY = 1e-3
ROLLOUT_COLLAPSE = 1e-6


# ---------------------------------------------------------------------------
# Instrumented copy of HippocampalModule.propose_trajectories.
#
# Mirrors ree_core/hippocampal/module.py lines ~449-700 exactly for the
# branches this experiment exercises (orthogonal/iid noise, value-flat
# CEM loop, no ghost-goal probes -- experiment config disables that
# branch). Records M0/M1/M2/M3 at iter=0 and M4 at iter=1.
#
# The patched method writes per-tick snapshots into the agent's
# hippocampal module via a side-channel attribute _exp554_logger which
# the experiment harness then drains and clears each tick.
# ---------------------------------------------------------------------------

def _install_instrumented_propose(hippocampal: HippocampalModule) -> None:
    """Monkey-patch propose_trajectories on a single HippocampalModule
    instance. Restores nothing -- the agent is constructed fresh per
    cell so isolation is per-cell."""

    original_propose = hippocampal.propose_trajectories

    def _logged_propose(
        z_world: torch.Tensor,
        z_self: Optional[torch.Tensor] = None,
        num_candidates: Optional[int] = None,
        e1_prior: Optional[torch.Tensor] = None,
        action_bias: Optional[torch.Tensor] = None,
        operating_mode: Optional[Dict[str, float]] = None,
        current_z_goal: Optional[torch.Tensor] = None,
    ) -> List[Trajectory]:
        # Local logger dict to be drained at the end of this call.
        snap: Dict[str, Dict[str, float]] = {}

        # Clear MECH-293 diagnostics from prior tick.
        hippocampal._last_propose_diagnostics = {}
        n = num_candidates or hippocampal.config.num_candidates
        num_elite = max(1, int(n * hippocampal.config.elite_fraction))
        batch_size = z_world.shape[0]
        device = z_world.device

        if z_self is None:
            z_self = torch.zeros(
                batch_size, hippocampal.e2.config.self_dim, device=device
            )

        ao_mean = hippocampal._get_terrain_action_object_mean(
            z_world, e1_prior=e1_prior
        )
        ao_std = torch.ones_like(ao_mean)

        # MECH-267 mode-conditioned noise scale. Off by default in this
        # experiment (mode_conditioning_enabled flag not set).
        hippocampal._last_operating_mode = (
            dict(operating_mode) if operating_mode is not None else None
        )
        mode_scale = hippocampal._compute_mode_noise_scale(operating_mode)
        hippocampal._last_mode_noise_scale = mode_scale
        if mode_scale is not None:
            ao_std = ao_std * mode_scale

        all_trajectories: List[Trajectory] = []
        use_orthogonal = bool(
            getattr(hippocampal.config, "use_orthogonal_cem_seeding", False)
        )
        ortho_diag = {
            "use_orthogonal_cem_seeding": use_orthogonal,
            "n_orthogonal_iters": 0,
            "n_iid_fallback_candidates": 0,
        }

        for _iteration in range(hippocampal.config.num_cem_iterations):
            trajectories: List[Trajectory] = []
            scores: List[torch.Tensor] = []

            ortho_noise: Optional[torch.Tensor] = None
            if use_orthogonal:
                flatten_dim = int(ao_mean.shape[1] * ao_mean.shape[2])
                if n <= flatten_dim:
                    raw = torch.randn(
                        n, flatten_dim, device=device, dtype=ao_mean.dtype
                    )
                    q, _ = torch.linalg.qr(raw.T, mode="reduced")
                    ortho_noise = q.T * float(flatten_dim) ** 0.5
                    ortho_noise = ortho_noise.view(
                        n, ao_mean.shape[1], ao_mean.shape[2]
                    ).unsqueeze(1).expand(
                        n, ao_mean.shape[0], ao_mean.shape[1], ao_mean.shape[2]
                    ).contiguous()
                    ortho_diag["n_orthogonal_iters"] += 1
                else:
                    ortho_noise = None
                    ortho_diag["n_iid_fallback_candidates"] += n

            # Per-iteration: build the [n, batch, H, ao_dim] noise stack
            # for measurement, then take per-candidate slices into the
            # legacy loop. This preserves bit-identical behaviour with
            # ree_core except for an extra accumulating tensor list.
            iter_noise_list: List[torch.Tensor] = []
            iter_cand_ao_list: List[torch.Tensor] = []
            iter_decoded_list: List[torch.Tensor] = []
            iter_world_final_list: List[torch.Tensor] = []

            for cand_idx in range(n):
                if ortho_noise is not None:
                    noise = ortho_noise[cand_idx]
                else:
                    noise = torch.randn_like(ao_mean)
                # M0: raw noise tensor (per-candidate slice from the
                # noise stack). Detach so logging does not leak
                # gradients.
                iter_noise_list.append(noise.detach().clone())

                action_objects_sample = ao_mean + ao_std * noise
                # M1: candidates_ao = ao_mean + ao_std * noise
                iter_cand_ao_list.append(action_objects_sample.detach().clone())

                actions = hippocampal._decode_action_objects(
                    action_objects_sample
                )
                # M2: decoded actions, post-action_object_decoder.
                iter_decoded_list.append(actions.detach().clone())

                traj = hippocampal.e2.rollout_with_world(
                    z_self, z_world, actions,
                    compute_action_objects=True,
                    action_bias=action_bias,
                )
                trajectories.append(traj)
                scores.append(hippocampal._score_trajectory(traj))

                # M3: trajectory final world_state. Trajectory.world_states
                # is List[Tensor[batch, world_dim]] (one per rollout step).
                if traj.world_states:
                    iter_world_final_list.append(
                        traj.world_states[-1].detach().clone()
                    )

            scores_tensor = torch.stack(scores)
            elite_indices = torch.argsort(scores_tensor)[:num_elite]

            elite_ao = []
            for i in elite_indices:
                ao_seq = trajectories[i].get_action_object_sequence()
                if ao_seq is not None:
                    elite_ao.append(ao_seq)
            if elite_ao:
                elite_ao_tensor = torch.stack(elite_ao)
                ao_mean = elite_ao_tensor.mean(dim=0)
                ao_std = elite_ao_tensor.std(dim=0) + 1e-6

            all_trajectories = trajectories

            # Record per-iteration measurements. Iteration 0 produces
            # M0..M3; iteration 1 produces the M4 snapshot (the new
            # candidates_ao stack constructed from the post-refit
            # ao_mean + ao_std).
            if _iteration == 0:
                snap["M0"] = _stats(iter_noise_list)
                snap["M1"] = _stats(iter_cand_ao_list)
                snap["M2"] = _stats(iter_decoded_list)
                snap["M3"] = _stats(iter_world_final_list)
            if _iteration == 1:
                snap["M4"] = _stats(iter_cand_ao_list)

        # MECH-293 ghost branch: only fires when config.use_mech293_ghost_probes
        # AND a bank exists AND current_z_goal is not None. Experiment
        # config disables MECH-293; falls through silently.
        if (
            getattr(hippocampal.config, "use_mech293_ghost_probes", False)
            and hippocampal.ghost_goal_bank is not None
        ):
            ghost_candidates = hippocampal._propose_ghost_seeded(
                current_z_goal=current_z_goal,
                n_total=n,
                z_self=z_self,
                e1_prior=e1_prior,
                action_bias=action_bias,
            )
            if ghost_candidates:
                all_trajectories = hippocampal._mix_value_flat_with_ghost(
                    value_flat=all_trajectories,
                    ghost=ghost_candidates,
                    replace_lowest=bool(
                        getattr(
                            hippocampal.config,
                            "mech293_replace_lowest_ranked",
                            True,
                        )
                    ),
                )

        if use_orthogonal:
            hippocampal._last_propose_diagnostics.update(ortho_diag)

        # Stash snapshot on the module for the harness to drain.
        hippocampal._exp554_last_snap = snap
        return all_trajectories

    # Bind onto the instance, preserving the API signature.
    hippocampal.propose_trajectories = _logged_propose  # type: ignore[assignment]
    # Reference original so smoke tests can compare wiring if needed.
    hippocampal._exp554_original_propose = original_propose


def _pairwise_l2_stats(stack: torch.Tensor) -> Tuple[float, float, float]:
    """Return (mean_pairwise_l2, min_pairwise_l2, elementwise_var) for
    a [K, ...] tensor. K must be >= 2."""
    K = stack.shape[0]
    flat = stack.reshape(K, -1)
    if K < 2:
        return (0.0, 0.0, float(flat.var(unbiased=False).item()))
    diffs = flat.unsqueeze(0) - flat.unsqueeze(1)
    l2 = diffs.norm(dim=-1)
    iu = torch.triu_indices(K, K, offset=1)
    pairs = l2[iu[0], iu[1]]
    return (
        float(pairs.mean().item()),
        float(pairs.min().item()),
        float(flat.var(dim=0, unbiased=False).mean().item()),
    )


def _stats(tensors: List[torch.Tensor]) -> Dict[str, float]:
    """Compute pairwise/var stats over a list of K tensors (one per
    candidate) of identical shape."""
    if not tensors:
        return {
            "pairwise_l2_mean": 0.0,
            "min_pairwise_l2": 0.0,
            "elementwise_var": 0.0,
            "K": 0,
        }
    stack = torch.stack(tensors, dim=0)
    pw_mean, pw_min, ew_var = _pairwise_l2_stats(stack)
    return {
        "pairwise_l2_mean": pw_mean,
        "min_pairwise_l2": pw_min,
        "elementwise_var": ew_var,
        "K": int(stack.shape[0]),
    }


# ---------------------------------------------------------------------------
# Run-cell harness (mirrors EXQ-553 structure)
# ---------------------------------------------------------------------------

def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed,
        size=10,
        num_hazards=2,
        num_resources=3,
        hazard_harm=0.04,
        proximity_harm_scale=0.12,
        proximity_benefit_scale=0.18,
        hazard_field_decay=0.5,
        energy_decay=0.005,
        use_proxy_fields=True,
        resource_respawn_on_consume=True,
        harm_history_len=10,
        scheduled_external_hazard_enabled=True,
        scheduled_external_hazard_interval=50,
        scheduled_external_hazard_prob=0.5,
        scheduled_external_hazard_adjacent_only=True,
    )


def _make_agent(env: CausalGridWorldV2, use_orthogonal: bool) -> REEAgent:
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=32,
        alpha_world=0.9,
        alpha_self=0.3,
        use_harm_stream=True,
        harm_obs_dim=51,
        use_affective_harm_stream=True,
        harm_obs_a_dim=50,
        harm_history_len=10,
        z_harm_a_dim=16,
        z_goal_enabled=False,
        drive_weight=0.0,
        e1_goal_conditioned=False,
        goal_weight=0.0,
        use_per_stream_vs=True,
        use_per_region_vs=True,
        use_event_segmenter=True,
        use_invalidation_trigger=True,
        use_anchor_sets=True,
        use_orthogonal_cem_seeding=use_orthogonal,
    )
    agent = REEAgent(cfg)
    _install_instrumented_propose(agent.hippocampal)
    return agent


def _shannon_entropy(counts: Dict[int, int]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    ent = 0.0
    for c in counts.values():
        if c <= 0:
            continue
        p = c / total
        ent -= p * math.log(p)
    return ent


def _accumulate_snap(
    accum: Dict[str, Dict[str, float]],
    snap: Dict[str, Dict[str, float]],
    n_seen: Dict[str, int],
) -> None:
    """Running-mean accumulation over per-tick snapshots."""
    for point, stats in snap.items():
        prev = accum.setdefault(
            point,
            {
                "pairwise_l2_mean": 0.0,
                "min_pairwise_l2": 0.0,
                "elementwise_var": 0.0,
                "K_last": 0,
            },
        )
        n_seen[point] = n_seen.get(point, 0) + 1
        n = n_seen[point]
        prev["pairwise_l2_mean"] += (
            stats["pairwise_l2_mean"] - prev["pairwise_l2_mean"]
        ) / n
        prev["min_pairwise_l2"] += (
            stats["min_pairwise_l2"] - prev["min_pairwise_l2"]
        ) / n
        prev["elementwise_var"] += (
            stats["elementwise_var"] - prev["elementwise_var"]
        ) / n
        prev["K_last"] = stats["K"]


def _run_cell(seed: int, arm: str) -> Dict:
    use_orthogonal = (arm == "ARM_ORTHO")
    torch.manual_seed(seed)
    env = _make_env(seed)
    agent = _make_agent(env, use_orthogonal=use_orthogonal)

    accum: Dict[str, Dict[str, float]] = {}
    n_seen: Dict[str, int] = {}
    n_ticks = 0
    n_fresh_propose_ticks = 0
    n_nans = 0
    error_note: Optional[str] = None
    action_counts: Dict[int, int] = {}
    ortho_diag_last: Dict = {}
    stage1_pw_per_tick: List[float] = []

    prev_candidates_id: Optional[int] = None

    for ep in range(P0_EPISODES):
        _, obs_dict = env.reset()
        agent.reset()
        prev_candidates_id = None
        for _step in range(EPISODE_STEPS):
            body = obs_dict["body_state"].float().unsqueeze(0)
            world = obs_dict["world_state"].float().unsqueeze(0)
            harm = obs_dict.get("harm_obs")
            if harm is not None:
                harm = harm.float().unsqueeze(0)
            harm_a = obs_dict.get("harm_obs_a")
            if harm_a is not None:
                harm_a = harm_a.float().unsqueeze(0)
            harm_hist = obs_dict.get("harm_history")
            if harm_hist is not None:
                harm_hist = harm_hist.float().unsqueeze(0)

            latent = agent.sense(
                obs_body=body, obs_world=world,
                obs_harm=harm, obs_harm_a=harm_a, obs_harm_history=harm_hist,
            )
            ticks = agent.clock.advance()
            wdim = latent.z_world.shape[-1]
            e1_prior = (
                agent._e1_tick(latent) if ticks.get("e1_tick", False)
                else torch.zeros(1, wdim, device=agent.device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)

            cur_id = id(candidates)
            fresh = (cur_id != prev_candidates_id)
            prev_candidates_id = cur_id

            if fresh and candidates:
                n_fresh_propose_ticks += 1
                # Drain the M-snap for this tick.
                snap = getattr(agent.hippocampal, "_exp554_last_snap", {})
                if snap:
                    _accumulate_snap(accum, snap, n_seen)
                # Stage 1 sanity proxy (mirrors EXQ-553): pairwise L2 of
                # first-step actions across candidates.
                firsts = [t.actions[:, 0, :].squeeze(0).detach() for t in candidates]
                if len(firsts) >= 2:
                    stacked = torch.stack(firsts, dim=0)
                    K = stacked.shape[0]
                    diffs = stacked.unsqueeze(0) - stacked.unsqueeze(1)
                    l2 = diffs.norm(dim=-1)
                    iu = torch.triu_indices(K, K, offset=1)
                    pairs = l2[iu[0], iu[1]]
                    stage1_pw_per_tick.append(float(pairs.mean().item()))
                # Substrate echo (ortho diagnostics).
                ortho_diag_last = dict(
                    getattr(agent.hippocampal, "_last_propose_diagnostics", {})
                )

            action = agent.select_action(candidates, ticks)
            if not torch.isfinite(action).all():
                n_nans += 1
                if error_note is None:
                    error_note = (
                        f"non-finite action at seed={seed} arm={arm} "
                        f"ep={ep} step={_step}"
                    )
                break
            a_idx = int(action[0].argmax().item())
            action_counts[a_idx] = action_counts.get(a_idx, 0) + 1
            _, _h, done, _info, obs_dict = env.step(action)
            n_ticks += 1
            if done:
                break
        if error_note is not None:
            break

    def _mean(xs: List[float]) -> float:
        return float(sum(xs) / len(xs)) if xs else 0.0

    # Flatten accum to manifest-friendly schema.
    per_point: Dict[str, Dict[str, float]] = {}
    for p in POINTS:
        if p in accum:
            per_point[p] = {
                "pairwise_l2_mean": accum[p]["pairwise_l2_mean"],
                "min_pairwise_l2_mean": accum[p]["min_pairwise_l2"],
                "elementwise_var_mean": accum[p]["elementwise_var"],
                "K_last": accum[p]["K_last"],
                "n_samples": n_seen.get(p, 0),
            }
        else:
            per_point[p] = {
                "pairwise_l2_mean": 0.0,
                "min_pairwise_l2_mean": 0.0,
                "elementwise_var_mean": 0.0,
                "K_last": 0,
                "n_samples": 0,
            }

    return {
        "seed": seed,
        "arm": arm,
        "use_orthogonal_cem_seeding": use_orthogonal,
        "n_ticks": n_ticks,
        "n_fresh_propose_ticks": n_fresh_propose_ticks,
        "n_nans": n_nans,
        "error_note": error_note,
        "per_point": per_point,
        "stage1_pairwise_l2_mean": _mean(stage1_pw_per_tick),
        "action_class_entropy": _shannon_entropy(action_counts),
        "action_class_counts": action_counts,
        "n_actions": sum(action_counts.values()),
        "ortho_diag_last": ortho_diag_last,
    }


def _interpret(per_arm: Dict[str, List[Dict]]) -> Tuple[str, str]:
    """Apply pre-registered 5-row interpretation grid.

    Order: R3 -> R1 -> R4 -> R2 -> R5 (R3 trumps when ARM_ORTHO M0 fails to
    show distinct candidates -- substrate sampler not actually firing).
    """
    iid_seeds = per_arm["ARM_IID"]
    ortho_seeds = per_arm["ARM_ORTHO"]
    if any(r.get("error_note") for r in iid_seeds + ortho_seeds):
        return ("ERROR", "agent_error")

    def _mean_point(seeds: List[Dict], point: str) -> float:
        vals = [s["per_point"][point]["pairwise_l2_mean"] for s in seeds]
        return float(sum(vals) / len(vals)) if vals else 0.0

    o_m0 = _mean_point(ortho_seeds, "M0")
    o_m1 = _mean_point(ortho_seeds, "M1")
    o_m2 = _mean_point(ortho_seeds, "M2")
    o_m3 = _mean_point(ortho_seeds, "M3")
    o_m4 = _mean_point(ortho_seeds, "M4")
    i_m0 = _mean_point(iid_seeds, "M0")

    # R3: noise_shape_irrelevant. ARM_ORTHO M0 must be substantially
    # higher than ARM_IID M0 -- if not, the substrate is producing
    # iid-equivalent noise and the proposer-fix path is closed.
    if o_m0 < 1.5 * max(i_m0, 1e-9):
        return ("FAIL", "R3_noise_shape_irrelevant_orthogonal_sampler_not_distinct")

    # R1: decoder_collapse. M0/M1 large; M2 small.
    if o_m1 > LARGE_DIVERSITY and o_m2 < SMALL_DIVERSITY:
        return ("FAIL", "R1_decoder_collapse")

    # R4: e2_rollout_collapse. M0..M2 + M4 preserve diversity but M3 ~ 0.
    if (
        o_m1 > LARGE_DIVERSITY
        and o_m2 > LARGE_DIVERSITY
        and o_m4 > LARGE_DIVERSITY
        and o_m3 < ROLLOUT_COLLAPSE
    ):
        return ("FAIL", "R4_e2_rollout_collapse")

    # R2: elite_refit_collapse. M0..M2 preserve diversity but M4 collapses.
    if (
        o_m1 > LARGE_DIVERSITY
        and o_m2 > LARGE_DIVERSITY
        and o_m4 < SMALL_DIVERSITY
    ):
        return ("FAIL", "R2_elite_refit_collapse")

    return ("FAIL", "R5_other_route_to_diagnose_errors")


def _print_plan() -> None:
    print(f"{EXPERIMENT_TYPE} -- decoder+elite-refit localization", flush=True)
    print(f"Queue ID: {QUEUE_ID}", flush=True)
    print(f"Seeds: {SEEDS}", flush=True)
    print(f"Arms: {ARMS}", flush=True)
    print(f"P0: {P0_EPISODES} ep x {EPISODE_STEPS} steps/ep -> "
          f"~{P0_EPISODES * EPISODE_STEPS} ticks/cell x 6 cells", flush=True)
    print(f"Measurement points: {POINTS}", flush=True)
    print(f"experiment_purpose={EXPERIMENT_PURPOSE}", flush=True)


def main() -> Tuple[Optional[str], Optional[str]]:
    parser = argparse.ArgumentParser(description=f"{EXPERIMENT_TYPE}")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print plan and exit 0; do not execute.",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="1 seed x 1 ep x 5 ticks per arm smoke; no manifest written.",
    )
    args = parser.parse_args()
    _run_started = datetime.now(timezone.utc)

    if args.dry_run:
        _print_plan()
        print("DRY RUN OK", flush=True)
        return (None, None)

    if args.smoke:
        global SEEDS, P0_EPISODES, EPISODE_STEPS
        SEEDS = [42]
        P0_EPISODES = 1
        EPISODE_STEPS = 5
        print("SMOKE: 1 seed x 1 ep x 5 ticks per arm; no manifest", flush=True)
        for arm in ARMS:
            r = _run_cell(SEEDS[0], arm)
            for p in POINTS:
                ps = r["per_point"][p]
                print(
                    f"  seed={SEEDS[0]} arm={arm} {p} "
                    f"pw_mean={ps['pairwise_l2_mean']:.4e} "
                    f"min_pw={ps['min_pairwise_l2_mean']:.4e} "
                    f"ew_var={ps['elementwise_var_mean']:.4e} "
                    f"K={ps['K_last']} n={ps['n_samples']}",
                    flush=True,
                )
            print(
                f"  seed={SEEDS[0]} arm={arm} stage1_pw_mean="
                f"{r['stage1_pairwise_l2_mean']:.4e} "
                f"entropy={r['action_class_entropy']:.4f} "
                f"ortho_diag={r['ortho_diag_last']}",
                flush=True,
            )
        print("SMOKE OK", flush=True)
        return (None, None)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"
    out_dir = (
        Path(__file__).resolve().parents[2]
        / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    per_arm: Dict[str, List[Dict]] = {a: [] for a in ARMS}
    for seed in SEEDS:
        for arm in ARMS:
            print(f"Seed {seed} Condition {arm}", flush=True)
            r = _run_cell(seed, arm)
            for p in POINTS:
                ps = r["per_point"][p]
                print(
                    f"  [train] seed={seed} arm={arm} {p} "
                    f"pw_mean={ps['pairwise_l2_mean']:.4e} "
                    f"min_pw={ps['min_pairwise_l2_mean']:.4e} "
                    f"ew_var={ps['elementwise_var_mean']:.4e} "
                    f"K={ps['K_last']} n={ps['n_samples']}",
                    flush=True,
                )
            print(
                f"  [train] seed={seed} arm={arm} stage1_pw="
                f"{r['stage1_pairwise_l2_mean']:.4e} "
                f"entropy={r['action_class_entropy']:.4f} "
                f"actions={r['action_class_counts']}",
                flush=True,
            )
            if r["error_note"] is not None:
                print(f"  ERROR: {r['error_note']}", flush=True)
            per_arm[arm].append(r)

    outcome, label = _interpret(per_arm)
    print(f"\nOutcome: {outcome}", flush=True)
    print(f"Interpretation: {label}", flush=True)

    summary = {
        "gate_rule": (
            "Diagnostic: 5-row interpretation grid over per-point pairwise L2 "
            "across 5 measurement points (M0 noise, M1 cand_ao, M2 decoded, "
            "M3 e2 world_final, M4 iter2 cand_ao post-refit)."
        ),
        "large_diversity_threshold": LARGE_DIVERSITY,
        "small_diversity_threshold": SMALL_DIVERSITY,
        "rollout_collapse_threshold": ROLLOUT_COLLAPSE,
        "interpretation_label": label,
        "n_seeds": len(SEEDS),
    }

    output = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": ts,
        "outcome": outcome,
        "evidence_direction": "non_contributory",
        "evidence_direction_note": (
            "Decoder + elite-refit collapse localization. Instruments 5 "
            "measurement points inside HippocampalModule.propose_trajectories "
            "via experiment-side monkey-patch (no ree_core modification): "
            "M0 noise tensor; M1 candidates_ao = ao_mean + ao_std * noise; "
            "M2 decoded actions post-action_object_decoder; M3 trajectory "
            "final world_state; M4 iter-2 candidates_ao post-elite-refit. "
            "Pre-registered 5-row interpretation grid (evaluated in order "
            "R3 -> R1 -> R4 -> R2 -> R5): "
            "R3 noise_shape_irrelevant -- ARM_ORTHO M0 not distinct from "
            "ARM_IID M0; orthogonal sampler is not producing distinct "
            "candidates at the noise level. Routes to ree_core orthogonal "
            "sampler re-audit. "
            "R1 decoder_collapse -- M0/M1 > 1e-2 but M2 < 1e-3; "
            "action_object_decoder is the cliff. Routes to decoder weight "
            "/ saturation probe. "
            "R4 e2_rollout_collapse -- M0..M2 + M4 preserve diversity but "
            "M3 < 1e-6 (matches V3-EXQ-551a stage 2 cliff); E2 dynamics "
            "maps diverse actions to near-identical world_state outcomes. "
            "Routes to E2 forward rank / saturation probe. "
            "R2 elite_refit_collapse -- M0..M2 preserve diversity but M4 "
            "< 1e-3; CEM elite-refit step is the cliff. Routes to "
            "elite_fraction / refit_std formula / single-elite degeneracy. "
            "R5 other -- routes to /diagnose-errors. "
            "experiment_purpose=diagnostic with claim_ids=[] intentional: "
            "this is a localization probe, not a single-claim test. "
            "Successor experiment chosen by which row fires."
        ),
        "pass_criteria_summary": summary,
        "per_arm_results": per_arm,
        "config": {
            "seeds": SEEDS,
            "arms": list(ARMS),
            "p0_episodes": P0_EPISODES,
            "episode_steps": EPISODE_STEPS,
            "manipulated_variable": "use_orthogonal_cem_seeding",
            "env_kwargs_match": "V3-EXQ-551 / V3-EXQ-553",
            "no_training": True,
            "measurement_points": list(POINTS),
            "supersedes": None,
        },
    }

    out_file = write_flat_manifest(
        output,
        out_dir,
        dry_run=False,
        config=output.get("config"),
        seeds=SEEDS,
        script_path=Path(__file__),
        elapsed_seconds=(datetime.now(timezone.utc) - _run_started).total_seconds(),
    )
    print(f"Result written to: {out_file}", flush=True)

    return (outcome, str(out_file))


if __name__ == "__main__":
    _outcome, _manifest_path = main()
    if _outcome is not None and _manifest_path is not None:
        emit_outcome(outcome=_outcome, manifest_path=_manifest_path)
    sys.exit(0)
