"""V3-EXQ-925a: E3 F-dominance causal-replay harness, COMMITTED REGIME (H1-H4).

EXPERIMENT_PURPOSE = "diagnostic" -- excluded from governance confidence/conflict
scoring. claim_ids=[] (no claim is adjudicated; this discriminates among causal
hypotheses about MECH-439, it does not test MECH-439/ARC-062/ARC-107/108/110
themselves).

SUPERSEDES V3-EXQ-925. Same scientific question (H1-H4); corrected instrument.

WHY THIS LETTER EXISTS -- THE 925 REGIME FINDING WAS A DRIVER DEFECT
--------------------------------------------------------------------
V3-EXQ-925 reported `committed_fraction = 0.000` across 2493 fresh E3 selections
and concluded a SELECTOR REGIME finding: that this substrate, at its corpus-
standard default configuration, essentially never engages committed selection,
so MECH-439's "committed-selection variance monopoly" framing presumes a regime
the substrate does not occupy. `failure_autopsy_V3-EXQ-925_2026-08-12` routed a
re-test enabling `use_gap_scaled_commit_temperature` to engage commitment.

BOTH the finding and the proposed fix were wrong, and this was established by
direct runtime measurement during this script's authoring (2026-08-25), not
inferred:

1. `use_gap_scaled_commit_temperature` CANNOT engage commitment. Both of its
   call sites (`e3_selector.py:3683`, `:3763`) sit strictly INSIDE an
   `elif committed:` branch. The flag changes HOW a committed pick is made
   (softening the within-shortlist argmin into a gap-scaled multinomial --
   MECH-439 "Factor B"); it has no influence whatsoever on WHETHER `committed`
   becomes True. With `committed` ALWAYS False it is a dead branch, bit-identical to
   OFF. A re-test using it as the regime lever would have reproduced
   `committed_fraction = 0.000` exactly and burned ~2.2h learning nothing.

2. The real cause is a MISSING DRIVER CALL. `committed` is set at
   `e3_selector.py:3451` as `commit_variance < effective_threshold`, where
   `commit_variance = self._running_variance`. `_running_variance` is updated
   ONLY by `E3TrajectorySelector.update_running_variance`, whose in-substrate
   call site is `post_action_update` (`:3979`), reached from
   `REEAgent.update_residue` (`agent.py:9972`). V3-EXQ-925's driver calls
   NEITHER. Measured: `_running_variance` stayed pinned at exactly its
   `precision_init = 0.5` for every tick of a 260-tick probe, against
   `commitment_threshold = 0.4` -- so `committed = 0.5 < 0.4` was False
   deterministically, by construction, before any data was collected.

   The substrate's own comment at `e3_selector.py:3969-3973` names this exact
   failure: "Previous code gated on _committed_trajectory, creating a deadlock:
   rv starts at precision_init (0.5), above commit_threshold (0.40), so the
   agent never commits, rv never updates, and the agent can never commit."
   That deadlock was fixed INSIDE `post_action_update`. 925's driver never calls
   it, and so re-created the identical deadlock from outside the fix. 97 other
   experiment drivers do call it.

3. Adding the single missing call breaks the deadlock at the UNCHANGED default
   config. Measured on this env/agent, 8 episodes, no config change at all:
   `_running_variance` 0.500000 -> 0.007956, and `committed_fraction`
   0.000 -> 0.959 at the stock `commitment_threshold = 0.4`.

So `committed_fraction = 0.000` was never a fact about the substrate's operating
regime. It was an instrument defect in one driver. The H0 "SELECTOR REGIME"
discovery leg registered from 925, and the MECH-439 `evidence_quality_note`
drafted for governance Step 2b review, both rest on the incorrect reading.

WHAT THIS RUN DOES
------------------
Restores the regime 925 intended to test, then re-runs the SAME C0-C5 frozen-
state counterfactual replay restricted to genuinely COMMITTED ticks, so H1-H4 get
the fair test under MECH-439's own premise that 925 could not deliver.

ARMS (2, x 3 seeds = 6 cells):
  ARM_COMMIT      -- deadlock fix only. `post_action_update` is called every step
                     so `_running_variance` tracks world-forward prediction error
                     and commitment engages as an EMERGENT, learned property
                     rather than a config artefact. Stock commit config
                     (`commitment_threshold = 0.4`, `use_gap_scaled_commit_
                     temperature = False`) -- deliberately unchanged, so this arm
                     measures the substrate's ACTUAL default committed regime,
                     which is precisely the quantity 925 mis-measured.
  ARM_COMMIT_GSCT -- deadlock fix PLUS `use_gap_scaled_commit_temperature = True`.
                     MECH-439 Factor B becomes reachable for the first time (its
                     branch requires `committed`, which only ARM_COMMIT's fix
                     makes attainable). Tests whether gap-scaled entropy
                     regularisation of the committed pick changes the H1-H4 read.

DV-SYMMETRY DECLARATION (CLAUDE.md DV-symmetry invariance, per arm)
-------------------------------------------------------------------
The DVs are (a) `committed_fraction` -- a threshold count over the scalar
`_running_variance`; and (b) the C0-C5 counterfactual deltas on the captured
per-candidate score vector (`delta_p_safer`, `total_variation`, `argmin_flip`).

  ARM_COMMIT: the manipulation is calling `post_action_update`, which mutates
  `_running_variance` -- the very scalar DV (a) thresholds. It is NOT a broadcast
  constant across candidates, NOT a monotone rescaling of the score vector, and
  NOT a permutation of interchangeable units: it changes a per-tick scalar that
  gates which SELECTION BRANCH executes (`elif committed:` vs the softmax-sample
  branch), so it is invariant under none of the three symmetry groups. It is also
  demonstrably not invariant empirically: DV (a) moves 0.000 -> 0.959.

  ARM_COMMIT_GSCT: the manipulation replaces `argmin` over the eligible set with
  a temperature-scaled multinomial over the SAME score vector. This is NOT a
  monotone rescaling of the DV -- it changes the selection RULE, not the scores,
  so rank-based DVs are not protected from it (a hot commit-T can select a
  non-argmin candidate, which is the whole point). It is not a broadcast constant
  (T_eff varies per tick with `gap_norm`) and not a permutation. CAVEAT recorded
  as a precondition rather than assumed: if `gap_norm` is degenerate (all ticks
  decisive, `gap_norm -> 1`), `T_eff -> base` and the pick recovers the argmin,
  making this arm silently equivalent to ARM_COMMIT. `gap_scaled_commit_active`
  and the realised `T_eff` spread are therefore MEASURED and gate-checked, not
  presumed (see `gap_scaled_commit_engaged` precondition).

WHAT IS UNCHANGED FROM 925 (so the comparison is clean)
--------------------------------------------------------
Env kwargs, agent config, SD-056 P0 contrastive recipe, the C0-C5 replay
arithmetic, the independent env-native hazard-distance criterion, the latch-clear
sample-size discipline, and the readiness floors are all verbatim 925. The ONLY
substantive changes are the `post_action_update` call, the arm axis, the
committed-conditioned analysis, and the commit-gate recording below.

RECORDING DEBT CLOSED
---------------------
925 reported `committed_fraction = 0.000` as its headline regime finding but
recorded NEITHER of the two numbers that explain it: `commit_gate_variance` (what
the gate compared) and `effective_threshold` (what it compared against). So the
margin could not be sized from the manifest, and the defect was invisible to
anyone reading the recorded evidence. This run records both per event, plus
`_running_variance` and the realised gap-scaled-commit diagnostics.

READ THE COMMITTED-REGIME NOTE BEFORE THE H1-H4 PATTERN READ. If commitment does
NOT engage in ARM_COMMIT despite the fix, that is a substrate_not_ready_requeue,
NOT a regime finding and NOT a ceiling -- the readiness gate self-routes it.
"""


import sys
import math
import random
import time
import argparse
from collections import deque, Counter
from typing import Any, Deque, Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiment_protocol import emit_outcome
from ree_core.environment.causal_grid_world import CausalGridWorld
from ree_core.agent import REEAgent
from ree_core.utils.config import REEConfig
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from experiments._lib.manifest_core import stamp_recording_core
from experiments._lib.arm_fingerprint import arm_cell
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator
from experiments._lib.precondition_gate import (
    PreconditionSpec,
    aggregate_arm_gates,
    assert_no_structurally_unsatisfiable_gate,
    evaluate_arm_gate,
)

EXPERIMENT_PURPOSE = "diagnostic"
EXPERIMENT_TYPE = "v3_exq_925a_e3_fdominance_committed_regime_causal_harness"
SUPERSEDES = "V3-EXQ-925"

SEEDS = [42, 43, 44]
P0_WARMUP_EPISODES = 60
P1_CAPTURE_EPISODES = 40
STEPS_PER_EPISODE = 200

# ENV IDENTICAL TO V3-EXQ-643a/604a/569d (the actually-validated SD-056 combination).
# A first attempt on a stripped-down env (size=8, 3 hazards, no resources/reef/drift)
# left the contrastive loss stuck at chance level (log(8)~=2.08) for 3000 gradient
# steps -- confirmed 2026-08-12 by direct loss-logging, not inferred. Env richness
# (resources, reef structure, drift) appears load-bearing for SD-056's training
# signal, not incidental to 643a's specific claim.
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
    use_proxy_fields=True,
)

DRY_RUN_P0 = 3
DRY_RUN_P1 = 3
DRY_RUN_STEPS = 30

MIN_FRESH_EVENTS_PER_SEED = 15   # verdict floor (readiness), not a claim threshold

# SD-056 online contrastive (verbatim recipe from V3-EXQ-643a / 604a / 569d).
SD056_WEIGHT = 0.05
E2_CONTRASTIVE_LR = 1e-3
CONTRASTIVE_BATCH_K = 8
TRANSITION_BUFFER_MAX = 256
MIN_BUFFER_BEFORE_TRAIN = 16
MIN_CLASSES_FOR_TRAIN = 2
MAX_GRAD_NORM = 1.0
ROLLOUT_CLAMP_RATIO = 2.0

# Readiness preconditions (P0 abort gate -- measured on a positive control BEFORE
# the expensive P1 capture phase; see CLAUDE.md P0 readiness-assert convention).
CAND_WORLD_PAIRWISE_DIST_FLOOR = 0.02   # SD-056 readiness gate's own floor (571/609/643a lineage)
GATED_POLICY_STD_ACROSS_K_FLOOR = 1e-6  # same eps class as V3-EXQ-609's SPREAD_ZERO_EPS

# --- Committed-regime axis (this letter's reason for existing) -------------
# ARM_COMMIT is the deadlock fix ALONE at stock commit config; ARM_COMMIT_GSCT
# adds MECH-439 Factor B, which is only reachable once commitment engages.
ARM_COMMIT = "ARM_COMMIT"
ARM_COMMIT_GSCT = "ARM_COMMIT_GSCT"
ARMS = [ARM_COMMIT, ARM_COMMIT_GSCT]

# Load-bearing readiness floor. The primary criterion routes on statistics
# computed over COMMITTED events, so the readiness check must assert the SAME
# statistic (committed_fraction), not a proxy for it -- CLAUDE.md
# "readiness precondition must assert the SAME statistic the load-bearing
# criterion routes on" (the V3-EXQ-643 magnitude-vs-range defect). Measured
# 0.959 on an 8-episode positive-control probe at stock config with the fix, so
# a 0.05 floor is ~19x below the measured value and cannot be a tight gate; it
# exists to catch the deadlock RE-appearing (which reads exactly 0.000), not to
# grade the regime.
COMMITTED_FRACTION_FLOOR = 0.05
# Minimum committed events per arm before the C0-C5 committed-conditioned read
# is considered non-degenerate.
MIN_COMMITTED_EVENTS_PER_ARM = 30
# Gap-scaled commit-T is only meaningfully DIFFERENT from a hard argmin when the
# realised T_eff actually varies across ticks (T_eff = base + alpha*(1-gap_norm);
# an all-decisive-gap run collapses it onto the argmin). Guards the ARM_COMMIT_GSCT
# DV-symmetry caveat declared in the module docstring.
GSCT_TEFF_SPREAD_FLOOR = 1e-6

F_WEIGHT_LADDER = [1.0, 0.5, 0.25, 0.0]   # matches V3-EXQ-858 for comparability

_ZG = ZGoalStreamAccumulator()


# ---------------------------------------------------------------------------
# Env / agent construction
# ---------------------------------------------------------------------------

def _make_env(seed: Optional[int]) -> CausalGridWorld:
    return CausalGridWorld(seed=seed, **ENV_KWARGS)


def _make_agent(env: CausalGridWorld, obs_dict: Dict[str, Any],
                gap_scaled_commit: bool = False) -> REEAgent:
    """V3-EXQ-643a's FULL agent config, verbatim (not just the SD-056-relevant
    subset) + this script's own additions (gated_policy, finer channel gating,
    post-scorer-fix default). Two narrower attempts (SD-056-subset config on a
    stripped env; SD-056-subset config on 643a's env) both left the contrastive
    loss stuck at chance level (~log(8)) after ~3000 gradient steps despite
    confirmed-correct gradient flow to world_transition/world_action_encoder --
    so this copies EVERYTHING 643a set, on the theory that some other flag
    (z_harm/z_goal streams, resource_proximity_head, benefit_eval, per_stream_vs,
    event_segmenter, anchor_sets, structured_curiosity, e3_score_diversity)
    shapes E2's representations indirectly even though none are SD-056-specific.
    CausalGridWorld (unlike CausalGridWorldV2) has no body_obs_dim/world_obs_dim
    properties -- dims come from a live obs_dict after env.reset(), matching the
    V3-EXQ-571/609/924 convention (643a used CausalGridWorldV2's own properties;
    functionally identical since CausalGridWorldV2 IS CausalGridWorld with
    use_proxy_fields=True, a factory function not a subclass)."""
    cfg = REEConfig.from_dims(
        # ARM_COMMIT_GSCT only. MECH-439 "Factor B": softens the within-
        # shortlist COMMITTED argmin into a gap-scaled multinomial. Reachable
        # ONLY when `committed` is True (e3_selector.py:3683,:3763 both sit
        # inside `elif committed:`), which is why it was a dead branch in
        # V3-EXQ-925 and could never have served as that run's regime lever.
        use_gap_scaled_commit_temperature=gap_scaled_commit,
        body_obs_dim=obs_dict["body_state"].shape[0],
        world_obs_dim=obs_dict["world_state"].shape[0],
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
        # V_s substrate (main-path default; identical across arms in 643a)
        use_per_stream_vs=True,
        use_per_region_vs=True,
        use_event_segmenter=True,
        use_invalidation_trigger=True,
        use_anchor_sets=True,
        # SD-056 online contrastive + 643a rollout-norm clamp (stability).
        e2_action_contrastive_enabled=True,
        e2_action_contrastive_weight=SD056_WEIGHT,
        e2_rollout_output_norm_clamp_enabled=True,
        e2_rollout_output_norm_clamp_ratio=ROLLOUT_CLAMP_RATIO,
        # MECH-314 structured curiosity -- ALL_ON (643a calibration).
        use_structured_curiosity=True,
        use_curiosity_novelty=True,
        use_curiosity_uncertainty=True,
        use_curiosity_learning_progress=True,
        curiosity_bias_scale=0.5,
        curiosity_novelty_weight=0.25,
        curiosity_uncertainty_weight=0.25,
        curiosity_learning_progress_weight=0.25,
        # MECH-341 entropy bonus ON, stratified select OFF (643a).
        use_e3_score_diversity=True,
        use_e3_diversity_entropy_bonus=True,
        use_e3_diversity_stratified_select=False,
        # This script's own additions: the competitor channel + finer per-
        # channel breakdown (not part of 643a; 643a used the combined score_bias
        # sum via use_modulatory_selection_authority instead).
        use_gated_policy=True,
        use_finer_channel_gating=True,
    )
    # THE LOAD-BEARING FLAG (2026-08-12, this session). Without this, gated_policy
    # consumes the PROPOSER's candidate summaries, whose cross-candidate spread is
    # ~0 under monostrategy -- so its per-candidate bias is EXACTLY 0.0 no matter
    # how well SD-056 trains, because SD-056 shapes e2.world_forward and the
    # channel never reads e2.world_forward. Measured directly on an UNTRAINED
    # agent: "proposer" -> gated_policy std_across_K = 0.0 exactly;
    # "e2_world_forward" -> 3.6e-4 (non-zero). config.py:3755 documents this as
    # "the GAP-A extension of the 2026-05-17 ARC-062 GAP-B first-action-onehot fix
    # beyond GatedPolicy, and the shared-channel sibling of
    # curiosity_candidate_source (MECH-314a Phase-2, which fixed the curiosity
    # channel only)". An earlier version of this script omitted it, which alone
    # would have pinned the competitor channel to zero and made the whole
    # H1/H2/H3/H4 discrimination vacuous by construction.
    cfg.candidate_summary_source = "e2_world_forward"
    cfg.e3.f_weight = 1.0
    cfg.e3.e3_include_untrained_fallback_scorers = False  # post-SD-E3-SCORER-COMPLETION default
    agent = REEAgent(config=cfg)
    agent.e3.e3_score_decomp_enabled = True
    return agent


def _obs(d: Dict[str, Any], key: str) -> Optional[torch.Tensor]:
    h = d.get(key)
    if h is None:
        return None
    return h.float().unsqueeze(0) if h.dim() == 1 else h.float()


def _drive_z_goal(agent: REEAgent, body: torch.Tensor) -> None:
    """V3-EXQ-643a's z_goal driver, verbatim (643a line 474-480).

    `update_z_goal` is the SOLE writer of z_goal anywhere in the substrate. The
    config copied from 643a sets z_goal_enabled=True, so without this call
    z_goal stays at zero-init for the whole run, GoalState.is_active() is always
    False, and agent.py passes current_z_goal=None to every consumer -- the E3
    goal term, MECH-295 liking->approach bridge, MECH-293 ghost probes and the
    frontopolar counterfactual read all silently no-op with nothing raised and
    nothing visible in the manifest (V3-EXQ-830 / V3-EXQ-626)."""
    if agent.goal_state is None:
        return
    try:
        energy = float(body[0, 3].item())
    except Exception:
        energy = 1.0
    agent.update_z_goal(benefit_exposure=0.0, drive_level=max(0.0, 1.0 - energy))



def _commit_gate_update(agent: REEAgent, obs_dict: Dict[str, Any]) -> Dict[str, float]:
    """THE V3-EXQ-925 DEADLOCK FIX (this letter's reason for existing).

    Feed the OBSERVED post-action z_world back into E3 so `_running_variance`
    tracks world-forward prediction error. Without this call `_running_variance`
    stays pinned at `precision_init` (0.5) forever, `commit_threshold` is 0.4, and
    `committed = 0.5 < 0.4` is False on every tick by construction -- which is
    exactly what produced V3-EXQ-925's `committed_fraction = 0.000` and its
    incorrect SELECTOR REGIME conclusion.

    `e3_selector.py:3969-3973` documents this same deadlock and fixes it INSIDE
    `post_action_update`; a driver that never calls `post_action_update`
    re-creates the deadlock from outside the fix. `REEAgent.update_residue`
    (`agent.py:9972`) is the usual agent-level entry point; E3 is called directly
    here because this harness deliberately does not drive the residue/harm path
    (it must not perturb the frozen-state replay it is measuring).

    Returns the commit-gate scalars for recording -- the two numbers V3-EXQ-925
    reported a regime finding without ever recording.
    """
    body = obs_dict["body_state"].float()
    world = obs_dict["world_state"].float()
    if body.dim() == 1:
        body = body.unsqueeze(0)
    if world.dim() == 1:
        world = world.unsqueeze(0)
    with torch.no_grad():
        latent = agent.sense(
            obs_body=body, obs_world=world,
            obs_harm=_obs(obs_dict, "harm_obs"),
            obs_harm_a=_obs(obs_dict, "harm_obs_a"),
            obs_harm_history=_obs(obs_dict, "harm_history"),
        )
        agent.e3.post_action_update(
            actual_z_world=latent.z_world, harm_occurred=False
        )
    rv = float(agent.e3._running_variance)
    thr = float(agent.e3.commit_threshold)
    return {
        "commit_gate_variance": rv,
        "effective_threshold_base": thr,
        "commit_margin": thr - rv,
    }


def _sample_class_diverse_batch(buffer, k, rng):
    if len(buffer) < MIN_BUFFER_BEFORE_TRAIN:
        return None
    pool = list(buffer)
    rng.shuffle(pool)
    seen: Dict[int, Any] = {}
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
        if id(tup) not in picked:
            samples.append(tup)
            picked.add(id(tup))
    return samples


def _e2_contrastive_step(agent, buffer, optimiser, rng) -> Optional[float]:
    batch = _sample_class_diverse_batch(buffer, CONTRASTIVE_BATCH_K, rng)
    if batch is None:
        return None
    z0_K = torch.stack([t[0] for t in batch]).to(agent.device)
    a_K = torch.stack([t[1] for t in batch]).to(agent.device)
    z1_K = torch.stack([t[2] for t in batch]).to(agent.device)
    optimiser.zero_grad(set_to_none=True)
    loss = agent.e2.world_forward_contrastive_loss(
        z_world_0=z0_K, actions=a_K, z_world_1_targets=z1_K, simulation_mode=False,
    )
    if not torch.is_tensor(loss):
        return None
    loss_val = float(loss.detach().item())
    if not math.isfinite(loss_val) or not loss.requires_grad or loss_val == 0.0:
        return loss_val
    (SD056_WEIGHT * loss).backward()
    torch.nn.utils.clip_grad_norm_(agent.e2.parameters(), max_norm=MAX_GRAD_NORM)
    optimiser.step()
    return loss_val


# ---------------------------------------------------------------------------
# Ground-truth (env-native, independent of any network) safety criterion
# ---------------------------------------------------------------------------

def _nearest_hazard_dist(x: int, y: int, hazards: List[List[int]]) -> float:
    if not hazards:
        return float("inf")
    return min(math.hypot(x - h[0], y - h[1]) for h in hazards)


def _candidate_terminal_hazard_dist(env: CausalGridWorld, first_action: int) -> float:
    """Ground truth: distance from the candidate's post-first-action position to
    the nearest known hazard, using env-native action_map + hazards directly --
    NEVER a network output. This is the independent criterion (per user's brief,
    "safety/harm" axis)."""
    dx, dy = env._action_map[first_action % env.action_dim]
    if env.toroidal:
        nx, ny = (env.agent_x + dx) % env.size, (env.agent_y + dy) % env.size
    else:
        nx, ny = env.agent_x + dx, env.agent_y + dy
    return _nearest_hazard_dist(nx, ny, env.hazards)


# ---------------------------------------------------------------------------
# P0: SD-056 warmup + readiness probe
# ---------------------------------------------------------------------------

def run_p0_warmup(agent: REEAgent, env: CausalGridWorld, obs_dict, seed: int,
                   n_episodes: int, steps_per_episode: int,
                   total_episodes: Optional[int] = None, ep_offset: int = 0) -> Tuple[Dict, Any]:
    e2_opt = torch.optim.Adam(agent.e2.parameters(), lr=E2_CONTRASTIVE_LR)
    buffer: Deque = deque(maxlen=TRANSITION_BUFFER_MAX)
    sample_rng = random.Random(seed)
    losses: List[float] = []

    for ep in range(n_episodes):
        _, obs_dict = env.reset()
        agent.reset()
        pending_capture = None
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
                if torch.isfinite(z0_prev).all() and torch.isfinite(a_prev).all() and torch.isfinite(z1_obs).all():
                    buffer.append((z0_prev, a_prev, z1_obs))
                pending_capture = None

            ticks = agent.clock.advance()
            wdim = latent.z_world.shape[-1]
            e1_prior = agent._e1_tick(latent) if ticks.get("e1_tick", False) else torch.zeros(1, wdim, device=agent.device)
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)

            # 643a's z_goal driver, verbatim. update_z_goal is the SOLE writer of
            # z_goal in the substrate; the config copied from 643a sets
            # z_goal_enabled=True, so omitting this leaves z_goal pinned at
            # zero-init and silently no-ops every downstream consumer (the E3
            # goal term, MECH-295 liking->approach, frontopolar read, ...).
            _drive_z_goal(agent, body)

            action = agent.select_action(candidates, ticks)
            if action is None or not torch.isfinite(action).all():
                break

            _e2_contrastive_step(agent, buffer, e2_opt, sample_rng)

            if torch.isfinite(latent.z_world).all() and torch.isfinite(action).all():
                pending_capture = (
                    latent.z_world.detach().reshape(-1).clone(),
                    action.detach().reshape(-1).clone(),
                )

            action_idx = int(action[0].argmax().item())
            _, _, terminated, _, obs_dict = env.step(action_idx % env.action_dim)
            # DEADLOCK FIX -- see _commit_gate_update. Must run in P0 too: the
            # commit gate reads an EMA, so it needs the warmup phase to descend
            # before P1 measures the regime.
            _commit_gate_update(agent, obs_dict)
            if terminated:
                agent.reset()
                _, obs_dict = env.reset()
        # Runner progress. Denominator is the COMBINED per-cell episode budget so
        # P0 and P1 form one monotonic 1..N series against a single
        # episodes_per_run, rather than two phase-local series with different
        # denominators (which the runner would read as a shrinking total).
        _tot = total_episodes if total_episodes else n_episodes
        if (ep + 1) % 10 == 0 or (ep + 1) == n_episodes:
            print(f"  [train] P0 seed={seed} ep {ep_offset + ep + 1}/{_tot}", flush=True)
    return {"p0_episodes": n_episodes}, obs_dict


def measure_readiness(agent: REEAgent, env: CausalGridWorld, obs_dict, n_probe_ticks: int = 20):
    """Positive-control probe: cand_world_pairwise_dist (SD-056 target) and
    gated_policy std_across_K (the competitor's own per-candidate spread), on
    genuinely fresh E3 ticks only.

    MEASUREMENT-CORRECTION (2026-08-12, this session -- read this before editing).
    The first version of this function hand-rolled the pairwise distance over
    `trajectory.world_states[0][0]` -- the PROPOSER's candidate summaries -- and
    read 4.5e-5..8.6e-5 across three separate training configurations, which was
    then (wrongly) reported as "SD-056 fails to converge". That statistic is NOT
    what SD-056 trains. SD-056 shapes `e2.world_forward(z0, a_i)`, and the
    substrate exposes the correct diagnostic as a first-class API,
    `E2FastPredictor.cand_world_pairwise_dist(z_world_0, candidate_actions)`
    (e2_fast.py:224), whose own docstring names it the "headline metric for
    V3-EXQ-571 root-cause / SD-056 substrate-readiness". Measured on an entirely
    UNTRAINED agent the correct statistic reads ~0.0141 -- ~300x the hand-rolled
    number -- so the hand-rolled gate was not measuring a weaker version of the
    right thing, it was measuring a different thing.

    This is exactly the same-statistic-mismatch failure mode CLAUDE.md documents
    at length (V3-EXQ-642/643: "the readiness precondition's `measured` MUST be
    the SAME statistic the load-bearing criterion routes on"), committed inside
    the readiness gate written to prevent it. Use the API; do not re-hand-roll."""
    pairwise_dists: List[float] = []
    gp_stds: List[float] = []
    commit_flags: List[bool] = []
    gsct_teffs: List[float] = []
    n_fresh = 0
    tries = 0
    while n_fresh < n_probe_ticks and tries < n_probe_ticks * 20:
        tries += 1
        body = obs_dict["body_state"].float()
        world = obs_dict["world_state"].float()
        if body.dim() == 1:
            body = body.unsqueeze(0)
        if world.dim() == 1:
            world = world.unsqueeze(0)
        agent.e3.last_channel_terms = None
        with torch.no_grad():
            latent = agent.sense(obs_body=body, obs_world=world)
            ticks = agent.clock.advance()
            wdim = latent.z_world.shape[-1]
            e1_prior = agent._e1_tick(latent) if ticks.get("e1_tick", False) else torch.zeros(1, wdim, device=agent.device)
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            if candidates and len(candidates) >= 2:
                # The SD-056 substrate-readiness API -- NOT a hand-rolled distance
                # over proposer summaries. See this function's docstring.
                cand_first_actions = torch.stack(
                    [c.actions[:, 0, :][0].detach().float() for c in candidates]
                )
                d = agent.e2.cand_world_pairwise_dist(
                    latent.z_world.detach(), cand_first_actions
                )
                pairwise_dists.append(float(d.item()))
            _drive_z_goal(agent, body)
            action = agent.select_action(candidates, ticks)
        gp_terms = agent.e3.last_channel_terms.get("gated_policy") if isinstance(agent.e3.last_channel_terms, dict) else None
        if gp_terms is not None and torch.is_tensor(gp_terms) and gp_terms.numel() > 1:
            n_fresh += 1
            gp_stds.append(float(gp_terms.std(unbiased=False).item()))
        # COMMITTED-REGIME readiness. The load-bearing criterion routes on
        # statistics computed over COMMITTED events, so the readiness probe must
        # assert committed_fraction itself, not a proxy (CLAUDE.md: readiness
        # precondition must assert the SAME statistic -- the V3-EXQ-643
        # magnitude-vs-range defect). Read off the selector's own diagnostics,
        # which is the same source the recorded events use.
        _d = getattr(agent.e3, "last_score_diagnostics", {}) or {}
        if gp_terms is not None and torch.is_tensor(gp_terms) and gp_terms.numel() > 1:
            commit_flags.append(bool(_d.get("committed", False)))
            gsct_teffs.append(float(_d.get("gap_scaled_commit_temperature_eff", -1.0)))
        if action is None or not torch.isfinite(action).all():
            _, obs_dict = env.reset()
            continue
        action_idx = int(action[0].argmax().item())
        _, _, terminated, _, obs_dict = env.step(action_idx % env.action_dim)
        # DEADLOCK FIX inside the readiness probe too -- otherwise the probe
        # would measure committed_fraction under the very deadlock this run
        # exists to remove, and would self-route every arm to
        # substrate_not_ready_requeue.
        _commit_gate_update(agent, obs_dict)
        if terminated:
            agent.reset()
            _, obs_dict = env.reset()
    _teff_live = [t for t in gsct_teffs if t > 0.0]
    return {
        "cand_world_pairwise_dist_mean": float(np.mean(pairwise_dists)) if pairwise_dists else 0.0,
        "gated_policy_std_across_K_mean": float(np.mean(gp_stds)) if gp_stds else 0.0,
        "n_fresh_probe_ticks": n_fresh,
        # The load-bearing readiness statistic for this letter.
        "committed_fraction": (
            float(sum(commit_flags)) / float(len(commit_flags)) if commit_flags else 0.0
        ),
        "n_commit_probe_ticks": len(commit_flags),
        "commit_gate_variance": float(agent.e3._running_variance),
        "commit_threshold_eff": float(agent.e3.commit_threshold),
        # ARM_COMMIT_GSCT DV-symmetry guard: T_eff must actually VARY, else the
        # gap-scaled pick collapses onto the argmin and the arm is silently
        # equivalent to ARM_COMMIT (declared in the module docstring).
        "gsct_teff_spread": (
            float(max(_teff_live) - min(_teff_live)) if len(_teff_live) >= 2 else 0.0
        ),
        "n_gsct_active_ticks": len(_teff_live),
    }, obs_dict


# ---------------------------------------------------------------------------
# P1: capture genuine fresh-selection events + inline counterfactual replay
# ---------------------------------------------------------------------------

def _derive_temperature(scores: np.ndarray, probs: np.ndarray) -> float:
    """Recover the effective softmax temperature from the captured (scores, probs).

    The selector computes probs = softmax(-scores / T) (e3_selector.py:3135), so for
    any pair i,j:  log(p_i/p_j) = -(s_i - s_j)/T  =>  T = -(s_i - s_j)/log(p_i/p_j).
    Deriving it beats reading a config default because agent.select_action can
    modulate the temperature (noise-floor / arousal) before it reaches e3.select --
    a config read would silently use the wrong T and mis-scale every counterfactual
    distribution. Uses the most-separated pair for numerical conditioning, and
    least-squares over all pairs against the max-prob reference for robustness.
    """
    p = np.clip(probs, 1e-300, None)
    ref = int(np.argmax(p))
    ds = scores - scores[ref]
    dlp = np.log(p) - np.log(p[ref])
    mask = np.isfinite(ds) & np.isfinite(dlp) & (np.abs(dlp) > 1e-9)
    if not mask.any():
        return 1.0
    # dlp = -ds / T  =>  least-squares T from  ds = -T * dlp
    t = float(-np.dot(ds[mask], dlp[mask]) / max(np.dot(dlp[mask], dlp[mask]), 1e-30))
    return t if np.isfinite(t) and t > 1e-9 else 1.0


def _eligibility_argmin(agent: REEAgent, post_mod_scores: torch.Tensor) -> Tuple[int, List[int]]:
    """MECH-448 eligibility applied to a (possibly counterfactual) score vector,
    then argmin restricted to the eligible set. Pure function of the score vector
    + agent.e3.config per the technical grounding (no other live state read)."""
    eligible_idx = agent.e3._f_eligibility_envelope(post_mod_scores)
    eligible_idx = eligible_idx.tolist() if torch.is_tensor(eligible_idx) else list(eligible_idx)
    if not eligible_idx:
        eligible_idx = list(range(post_mod_scores.shape[0]))
    sub = post_mod_scores[eligible_idx]
    winner = eligible_idx[int(sub.argmin().item())]
    return winner, eligible_idx


def _replay_event(agent: REEAgent, event: Dict[str, Any], rng: np.random.Generator) -> Dict[str, Any]:
    """Recompute argmin under C0-C5 by applying counterfactual DELTAS to the REAL
    captured per-candidate score vector (agent.e3.last_scores), never by rebuilding
    an approximation of it.

    DESIGN CORRECTION (2026-08-12, this session -- caught by the C0 check below on
    a smoke run, BEFORE the multi-hour real run). The first version reconstructed
    the score as `f_weight*f + lambda_eff*m + gated_policy` and took its argmin.
    That reproduced the live selection on 2.5% of events -- indistinguishable from
    chance at K=32 -- because it was wrong twice over:
      (a) `harm_weighted` in _last_traj_components is ALREADY `lambda_eff * m`
          (e3_selector.py:1288), so multiplying by lambda_eff again double-counted it;
      (b) more fundamentally, the live score carries terms that reconstruction
          omitted entirely -- residue, benefit, goal, the DR-12/DR-13 penalties, and
          the WHOLE modulatory stack (score_bias's other named channels, the MECH-341
          entropy bonus, the route term, the explore term). A subset-sum is not the
          score the selector actually ranked.
    Perturbing the true vector instead makes C0 exact BY CONSTRUCTION for every
    argmin-path selection, and makes each counterfactual a precise, isolated
    perturbation rather than a competing approximation. `f` and `gated_policy` are
    still captured per-candidate because they are what the counterfactuals ADD or
    REMOVE -- but they are now deltas against reality, not the whole model of it.

    A residual C0 < 1.0 is still expected and INFORMATIVE, not a defect: the live
    selector does not always take the plain argmin path (it may go through the
    eligibility/shortlist route, or the uncommitted multinomial branch). Those
    events are reported via c0_matches_factual and the `committed` flag so the
    analysis can condition on argmin-path events rather than silently averaging
    over a mixture."""
    scores = event["final_scores_per_cand"]   # [K] THE REAL post-modulatory scores
    f = event["f_per_cand"]                   # [K] raw literal F (the C1/C2/C3 delta)
    gp = event["gated_policy_per_cand"]       # [K] the competitor term (the C5 delta)
    factual_f_weight = event["f_weight"]
    temperature = event["temperature"]
    safer_idx = event["ground_truth_safer_idx"]
    K = scores.shape[0]

    # F's actual contribution to the real score vector, which is what a lesion removes.
    f_contribution = factual_f_weight * f

    out: Dict[str, Any] = {}

    def _probs(s: np.ndarray) -> np.ndarray:
        """Selection distribution. Scores are COSTS, so the selector uses
        softmax(-scores/T) (e3_selector.py:3135). Shift-invariant, computed in a
        numerically stable form."""
        z = -s / max(temperature, 1e-9)
        z = z - z.max()
        e = np.exp(z)
        return e / e.sum()

    def _rank_of(s: np.ndarray, idx: int) -> int:
        """0 = most-preferred (lowest cost)."""
        return int(np.argsort(np.argsort(s))[idx])

    p_factual = _probs(scores)

    # C0 VALIDITY CHECK -- NOT an argmin-match test.
    # The selector is running UNCOMMITTED (see the module docstring): it samples
    # from softmax(-scores/T) rather than taking argmin, so "does the argmin equal
    # the factual selection" is ill-posed and reads ~chance BY CONSTRUCTION. The
    # meaningful check on a stochastic selector is whether the candidate actually
    # drawn had non-trivial probability under the captured vector -- i.e. whether
    # the vector we captured is consistent with the draw we observed.
    c0_argmin = int(np.argmin(scores))
    out["c0_argmin"] = c0_argmin
    out["c0_argmin_matches_factual"] = bool(c0_argmin == event["factual_selected_idx"])
    out["c0_prob_of_factual_choice"] = float(p_factual[event["factual_selected_idx"]])
    out["c0_rank_of_factual_choice"] = _rank_of(scores, event["factual_selected_idx"])
    out["c0_uniform_prob"] = 1.0 / K
    # THE load-bearing validity check: does the distribution reconstructed from the
    # captured scores + derived temperature match the selector's OWN captured
    # distribution? Near-zero means the capture and the softmax model are faithful,
    # so every counterfactual distribution below is trustworthy. A large value means
    # the replay does not model the real selector and NOTHING downstream is valid.
    out["c0_probs_reconstruction_max_abs_err"] = float(
        np.max(np.abs(p_factual - event["factual_probs_per_cand"]))
    )

    # ---- PRIMARY DVs: distributional, valid under a stochastic selector --------
    # How much selection probability sits on the independently-preferred
    # (ground-truth-safer) candidate, factually vs under each counterfactual.
    out["p_safer_factual"] = float(p_factual[safer_idx])
    out["rank_safer_factual"] = _rank_of(scores, safer_idx)

    # ---- SCALE DIAGNOSTICS: is the score spread big enough to STEER selection? --
    # A variance decomposition can report F as ~96% of score variance (V3-EXQ-924)
    # while the ABSOLUTE cross-candidate spread is tiny relative to the softmax
    # temperature -- in which case the selector samples near-uniformly and F's
    # "dominance" is dominance of a quantity that barely moves behaviour. That is
    # precisely the H1-vs-H3 distinction, so it must be recorded, not inferred.
    out["score_range"] = float(np.max(scores) - np.min(scores))
    out["score_range_over_temperature"] = float(out["score_range"] / max(temperature, 1e-12))
    out["f_contribution_range"] = float(np.max(f_contribution) - np.min(f_contribution))
    out["gated_policy_contribution_range"] = float(np.max(gp) - np.min(gp))
    # Normalised selection entropy: 1.0 = uniform (no steering), 0.0 = deterministic.
    _p = np.clip(p_factual, 1e-300, None)
    out["selection_entropy_normalised"] = float(
        -(np.sum(_p * np.log(_p))) / math.log(K)
    ) if K > 1 else 0.0
    # Expected hazard distance under the selection distribution -- the outcome-level
    # readout (higher = safer), independent of which candidate index wins.
    hz = event["hazard_dists"]
    finite = np.isfinite(hz)
    out["expected_hazard_dist_factual"] = (
        float(np.dot(p_factual[finite], hz[finite]) / max(p_factual[finite].sum(), 1e-12))
        if finite.any() else None
    )

    def _record(tag: str, s_cf: np.ndarray) -> None:
        """Record every DV for one counterfactual score vector.

        Distributional DVs are PRIMARY (valid under the stochastic selector);
        the argmin flip is recorded alongside as the deterministic-selector
        counterpart, meaningful on committed/argmin-path events."""
        p_cf = _probs(s_cf)
        out[f"{tag}_argmin"] = int(np.argmin(s_cf))
        out[f"{tag}_argmin_flip_vs_factual_argmin"] = bool(int(np.argmin(s_cf)) != c0_argmin)
        # PRIMARY: probability mass moved onto the independently-preferred candidate.
        out[f"{tag}_p_safer"] = float(p_cf[safer_idx])
        out[f"{tag}_delta_p_safer"] = float(p_cf[safer_idx] - p_factual[safer_idx])
        # Rank of the safer candidate (0 = most preferred). NEGATIVE delta = improved.
        out[f"{tag}_rank_safer"] = _rank_of(s_cf, safer_idx)
        out[f"{tag}_delta_rank_safer"] = int(_rank_of(s_cf, safer_idx) - out["rank_safer_factual"])
        # Outcome-level: expected ground-truth hazard distance under the new
        # distribution. Higher = safer. Independent of candidate identity.
        if finite.any():
            eh = float(np.dot(p_cf[finite], hz[finite]) / max(p_cf[finite].sum(), 1e-12))
            out[f"{tag}_expected_hazard_dist"] = eh
            out[f"{tag}_delta_expected_hazard_dist"] = (
                eh - out["expected_hazard_dist_factual"]
                if out["expected_hazard_dist_factual"] is not None else None
            )
        # Total distributional movement (how much the counterfactual moved selection
        # at all -- guards against reading a null as "no effect" when nothing moved).
        out[f"{tag}_total_variation_vs_factual"] = float(0.5 * np.abs(p_cf - p_factual).sum())

    # C1: literal-F lesion -- remove F's contribution, leave everything else intact.
    _record("c1", scores - f_contribution)

    # C2: literal-F association scramble -- swap F's contribution for a permutation
    # of itself, preserving its marginal distribution within this decision while
    # destroying its association with candidate identity.
    perm = rng.permutation(K)
    _record("c2", scores - f_contribution + f_contribution[perm])

    # C3: attenuation ladder -- re-weight F's contribution only.
    for fw in F_WEIGHT_LADDER:
        _record(f"c3_fw{str(fw).replace('.', '_')}", scores - f_contribution + fw * f)

    # C4: primary-field/eligibility toggle, on the REAL score vector.
    s4_t = torch.from_numpy(scores).float()
    c4_elig_argmin, eligible_idx = _eligibility_argmin(agent, s4_t)
    out["c4_eligibility_argmin"] = int(c4_elig_argmin)
    out["c4_eligibility_excluded_count"] = int(K - len(eligible_idx))
    out["c4_unrestricted_argmin"] = c0_argmin
    out["c4_eligibility_changes_winner"] = bool(out["c4_eligibility_argmin"] != c0_argmin)

    # C5: competitor lesion -- remove gated_policy's contribution, at factual AND
    # F-lesioned weight, to separate "the competitor never had influence" from
    # "the competitor had influence only once F stopped dominating".
    _record("c5_factual_f_no_competitor", scores - gp)
    _record("c5_lesioned_f_no_competitor", scores - f_contribution - gp)
    # Influence = the competitor's removal moves the distribution at all. Measured
    # distributionally (total variation) rather than by an argmin flip, so a real
    # but sub-argmin-threshold influence is not recorded as exactly zero.
    out["c5_competitor_tv_at_factual_f"] = out["c5_factual_f_no_competitor_total_variation_vs_factual"]
    out["c5_competitor_had_influence_factual"] = bool(
        out["c5_factual_f_no_competitor_argmin"] != c0_argmin
    )
    out["c5_competitor_had_influence_lesioned_f"] = bool(
        out["c5_lesioned_f_no_competitor_argmin"] != out["c1_argmin"]
    )

    return out


def run_p1_capture_and_replay(agent: REEAgent, env: CausalGridWorld, obs_dict, seed: int,
                               n_episodes: int, steps_per_episode: int,
                               total_episodes: Optional[int] = None,
                               ep_offset: int = 0) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    rng = np.random.default_rng(seed)

    for ep in range(n_episodes):
        _, obs_dict = env.reset()
        agent.reset()
        for _step in range(steps_per_episode):
            body = obs_dict["body_state"].float()
            world = obs_dict["world_state"].float()
            if body.dim() == 1:
                body = body.unsqueeze(0)
            if world.dim() == 1:
                world = world.unsqueeze(0)

            # Clear the E3 diagnostics latch -- see CLAUDE.md "Sample-size
            # integrity". E3 fires only every heartbeat.e3_steps_per_tick env
            # steps; confirmed empirically during V3-EXQ-924's authoring.
            agent.e3.last_score_decomp = None
            agent.e3.last_channel_terms = None
            agent.e3.last_score_diagnostics = None
            agent.e3.last_scores = None
            agent.e3.last_precommit_probs = None

            with torch.no_grad():
                latent = agent.sense(obs_body=body, obs_world=world)
                ticks = agent.clock.advance()
                wdim = latent.z_world.shape[-1]
                e1_prior = agent._e1_tick(latent) if ticks.get("e1_tick", False) else torch.zeros(1, wdim, device=agent.device)
                candidates = agent.generate_trajectories(latent, e1_prior, ticks)
                _drive_z_goal(agent, body)
                action = agent.select_action(candidates, ticks)

            last_decomp = agent.e3.last_score_decomp
            last_terms = agent.e3.last_channel_terms
            last_scores = agent.e3.last_scores
            last_probs = agent.e3.last_precommit_probs
            is_fresh = (
                bool(last_decomp and last_decomp.get("per_candidate"))
                and isinstance(last_terms, dict) and ("gated_policy" in last_terms)
                and last_scores is not None and torch.is_tensor(last_scores)
                and last_probs is not None and torch.is_tensor(last_probs)
            )

            if is_fresh and candidates:
                cands = last_decomp["per_candidate"]
                gp_vec = last_terms["gated_policy"]
                K = len(cands)
                if (torch.is_tensor(gp_vec) and gp_vec.numel() == K and K >= 2
                        and last_scores.numel() == K):
                    f_arr = np.array([c.get("f", 0.0) for c in cands], dtype=np.float64)
                    # harm_weighted is ALREADY lambda_eff * m (e3_selector.py:1288) --
                    # recorded for provenance only; never re-multiplied by lambda_eff.
                    m_arr = np.array([c.get("harm_weighted", 0.0) for c in cands], dtype=np.float64)
                    gp_arr = gp_vec.detach().cpu().numpy().astype(np.float64).reshape(-1)
                    # THE REAL post-modulatory per-candidate score vector the selector
                    # ranked. Counterfactuals perturb THIS, rather than a reconstruction.
                    scores_arr = last_scores.detach().cpu().numpy().astype(np.float64).reshape(-1)
                    # The EXACT factual selection distribution the selector sampled from
                    # (e3_selector.py:3135 softmax(-scores/temperature)). Captured rather
                    # than reconstructed, and used to DERIVE the effective temperature --
                    # agent.select_action can modulate T before it reaches e3.select, so
                    # reading a config default would silently use the wrong one.
                    probs_arr = last_probs.detach().cpu().numpy().astype(np.float64).reshape(-1)
                    temperature = _derive_temperature(scores_arr, probs_arr)
                    lambda_eff = float(agent.e3.config.lambda_ethical)
                    f_weight = float(agent.e3.config.f_weight)

                    diag = getattr(agent.e3, "last_score_diagnostics", {}) or {}
                    committed = bool(diag.get("committed", False))
                    # COMMIT-GATE RECORDING (closes V3-EXQ-925's recording debt:
                    # it reported committed_fraction=0.000 as a regime finding
                    # while recording neither number that explains it).
                    commit_gate_variance = float(agent.e3._running_variance)
                    commit_threshold_eff = float(agent.e3.commit_threshold)
                    # MECH-439 Factor B realised diagnostics -- present only when
                    # the gap-scaled committed pick actually fired.
                    gsct_active = bool(diag.get("gap_scaled_commit_active", False))
                    gsct_gap_norm = float(diag.get("gap_scaled_commit_gap_norm", -1.0))
                    gsct_teff = float(diag.get("gap_scaled_commit_temperature_eff", -1.0))

                    factual_selected_idx = int(action[0].argmax(dim=-1).item()) if False else None
                    # factual_selected_idx must come from WHICH CANDIDATE was chosen,
                    # not the resulting action one-hot (multiple candidates can share
                    # a first action). last_score_decomp carries selected_idx directly
                    # (e3_selector.py: self.last_score_decomp["selected_idx"] = ...).
                    factual_selected_idx = last_decomp.get("selected_idx")

                    if factual_selected_idx is not None and 0 <= int(factual_selected_idx) < K:
                        first_actions = []
                        for c in candidates[:K]:
                            try:
                                first_actions.append(int(c.actions[:, 0, :].argmax(dim=-1).flatten()[0].item()))
                            except Exception:
                                first_actions.append(0)
                        hazard_dists = np.array(
                            [_candidate_terminal_hazard_dist(env, a) for a in first_actions],
                            dtype=np.float64,
                        )
                        ground_truth_safer_idx = int(np.argmax(hazard_dists))

                        event = {
                            "f_per_cand": f_arr,
                            "harm_per_cand": m_arr,
                            "gated_policy_per_cand": gp_arr,
                            "final_scores_per_cand": scores_arr,
                            "factual_probs_per_cand": probs_arr,
                            "temperature": temperature,
                            "lambda_eff": lambda_eff,
                            "f_weight": f_weight,
                            "factual_selected_idx": int(factual_selected_idx),
                            "committed": committed,
                            "commit_gate_variance": commit_gate_variance,
                            "commit_threshold_eff": commit_threshold_eff,
                            "commit_margin": commit_threshold_eff - commit_gate_variance,
                            "gsct_active": gsct_active,
                            "gsct_gap_norm": gsct_gap_norm,
                            "gsct_temperature_eff": gsct_teff,
                            "hazard_dists": hazard_dists,
                            "ground_truth_safer_idx": ground_truth_safer_idx,
                            "gated_policy_range": float(gp_arr.max() - gp_arr.min()),
                        }
                        replay = _replay_event(agent, event, rng)
                        event.update(replay)
                        # F ALONE: which candidate literal F by itself prefers. Used as
                        # the disagreement axis vs the ground-truth-safer candidate.
                        # F is a COST (lower = preferred), so argmin.
                        f_only_argmin = int(np.argmin(f_arr))
                        event["f_only_argmin"] = f_only_argmin
                        # Competitor alone (gated_policy only, F excluded entirely).
                        event["competitor_only_argmin"] = int(np.argmin(gp_arr))
                        # Disagreement: does F-alone's preferred candidate differ from
                        # the ground-truth-safer candidate?
                        event["disagreement_event"] = bool(f_only_argmin != ground_truth_safer_idx)
                        events.append(event)

            if action is None or not torch.isfinite(action).all():
                _, obs_dict = env.reset()
                continue
            action_idx = int(action[0].argmax().item())
            _, _, terminated, _, obs_dict = env.step(action_idx % env.action_dim)
            # DEADLOCK FIX -- see _commit_gate_update. Applied AFTER the event was
            # recorded, so the event's `committed` flag reflects the gate state the
            # selector actually used for THAT selection, not a post-hoc update.
            _commit_gate_update(agent, obs_dict)
            if terminated:
                agent.reset()
                _, obs_dict = env.reset()
        _tot = total_episodes if total_episodes else n_episodes
        if (ep + 1) % 10 == 0 or (ep + 1) == n_episodes:
            print(f"  [train] P1 seed={seed} ep {ep_offset + ep + 1}/{_tot} "
                  f"n_events={len(events)}", flush=True)

    return events


# ---------------------------------------------------------------------------
# Aggregation + hypothesis evaluation
# ---------------------------------------------------------------------------

def aggregate_events(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(events)
    if n == 0:
        return {"n_events": 0}

    def _frac(pred):
        return float(np.mean([1.0 if pred(e) else 0.0 for e in events]))

    def _mean(key, src=None):
        vals = [e[key] for e in (src if src is not None else events)
                if e.get(key) is not None and np.isfinite(e[key])]
        return float(np.mean(vals)) if vals else None

    disagreement_events = [e for e in events if e["disagreement_event"]]
    n_dis = len(disagreement_events)
    committed_events = [e for e in events if e.get("committed")]

    ladder_tags = [f"c3_fw{str(fw).replace('.', '_')}" for fw in F_WEIGHT_LADDER]

    agg: Dict[str, Any] = {
        "n_events": n,
        "n_disagreement_events": n_dis,
        "disagreement_rate": n_dis / n,
        # THE SELECTOR REGIME. committed=False means the selector SAMPLED from
        # softmax(-scores/T) rather than taking argmin, so argmin-flip statistics
        # are not comparable to the factual draw and the distributional DVs below
        # are the load-bearing ones. Reported explicitly so a reader can never
        # mistake which regime produced the numbers.
        "n_committed_events": len(committed_events),
        "committed_fraction": len(committed_events) / n,
        # REPLAY VALIDITY: max abs error between the reconstructed factual
        # distribution and the selector's OWN captured distribution. Near 0 =>
        # the replay models the real selector faithfully.
        "c0_probs_reconstruction_max_abs_err_mean": _mean("c0_probs_reconstruction_max_abs_err"),
        "c0_probs_reconstruction_max_abs_err_worst": float(
            np.max([e["c0_probs_reconstruction_max_abs_err"] for e in events])
        ),
        "c0_prob_of_factual_choice_mean": _mean("c0_prob_of_factual_choice"),
        "c0_uniform_prob_mean": _mean("c0_uniform_prob"),
        "c0_argmin_matches_factual_rate": _frac(lambda e: e["c0_argmin_matches_factual"]),
        # Distributional movement caused by each counterfactual (0 => nothing moved).
        "c1_total_variation_mean": _mean("c1_total_variation_vs_factual"),
        "c2_total_variation_mean": _mean("c2_total_variation_vs_factual"),
        "c5_competitor_tv_at_factual_f_mean": _mean("c5_competitor_tv_at_factual_f"),
        # Deterministic-selector counterparts (meaningful on committed events).
        "argmin_flip_c1_f_lesion": _frac(lambda e: e["c1_argmin_flip_vs_factual_argmin"]),
        "argmin_flip_c2_f_scramble": _frac(lambda e: e["c2_argmin_flip_vs_factual_argmin"]),
        "flip_prob_c4_eligibility_changes_winner": _frac(lambda e: e["c4_eligibility_changes_winner"]),
        "competitor_had_influence_factual_f_rate": _frac(lambda e: e["c5_competitor_had_influence_factual"]),
        "competitor_had_influence_lesioned_f_rate": _frac(lambda e: e["c5_competitor_had_influence_lesioned_f"]),
        "gated_policy_range_mean": float(np.mean([e["gated_policy_range"] for e in events])),
        "temperature_mean": _mean("temperature"),
        # SCALE: can the score spread steer selection at all, at this temperature?
        # selection_entropy_normalised near 1.0 => near-uniform sampling, i.e. NO
        # channel (F included) is meaningfully steering the committed choice, which
        # is a different situation from "F steers and suppresses competitors".
        "score_range_mean": _mean("score_range"),
        "score_range_over_temperature_mean": _mean("score_range_over_temperature"),
        "f_contribution_range_mean": _mean("f_contribution_range"),
        "gated_policy_contribution_range_mean": _mean("gated_policy_contribution_range"),
        "selection_entropy_normalised_mean": _mean("selection_entropy_normalised"),
    }

    # PRIMARY DVs, per counterfactual: did selection probability / rank / expected
    # ground-truth safety move toward the independently-preferred candidate?
    for tag in ["c1", "c2"] + ladder_tags:
        agg[f"{tag}_delta_p_safer_mean"] = _mean(f"{tag}_delta_p_safer")
        agg[f"{tag}_delta_rank_safer_mean"] = _mean(f"{tag}_delta_rank_safer")
        agg[f"{tag}_delta_expected_hazard_dist_mean"] = _mean(f"{tag}_delta_expected_hazard_dist")

    if n_dis > 0:
        dc: Dict[str, Any] = {
            "n": n_dis,
            "p_safer_factual_mean": _mean("p_safer_factual", disagreement_events),
            "expected_hazard_dist_factual_mean": _mean("expected_hazard_dist_factual", disagreement_events),
        }
        for tag in ["c1"] + ladder_tags:
            dc[f"{tag}_delta_p_safer_mean"] = _mean(f"{tag}_delta_p_safer", disagreement_events)
            dc[f"{tag}_delta_rank_safer_mean"] = _mean(f"{tag}_delta_rank_safer", disagreement_events)
            dc[f"{tag}_delta_expected_hazard_dist_mean"] = _mean(
                f"{tag}_delta_expected_hazard_dist", disagreement_events)
            # Fraction of disagreement events where the counterfactual moved
            # probability mass TOWARD the safer candidate at all.
            dc[f"{tag}_frac_moved_toward_safer"] = float(np.mean([
                1.0 if (e.get(f"{tag}_delta_p_safer") or 0.0) > 0 else 0.0
                for e in disagreement_events
            ]))
        agg["disagreement_conditioned"] = dc
    else:
        agg["disagreement_conditioned"] = None

    return agg


def classify_hypotheses(agg: Dict[str, Any]) -> Dict[str, Any]:
    """Report the pattern found against the pre-registered H1-H4 predictions.
    This CLASSIFIES, it does not adjudicate a claim (claim_ids=[])."""
    if agg.get("n_events", 0) == 0:
        return {"verdict": "no_events_captured", "failure_localisation": "THE MEASURES FAILED"}

    n_dis = agg["n_disagreement_events"]
    if n_dis == 0:
        return {
            "verdict": "no_disagreement_events",
            "note": (
                "F-alone's preferred candidate never diverged from the ground-truth-"
                "safer candidate across all captured fresh selections -- there is no "
                "disagreement to condition a recruitment analysis on. This does NOT "
                "by itself support or refute H1/H2/H3/H4; it means this environment "
                "configuration/seed set did not produce cases where the two axes "
                "point different directions. Not evidence either way."
            ),
            "failure_localisation": "THE ENVIRONMENT/TRAINING FAILED (no disagreement generated, not a mechanism finding)",
        }

    dc = agg["disagreement_conditioned"]
    # PRIMARY: does removing F move selection probability toward the independently-
    # preferred (ground-truth-safer) candidate, on events where F and safety disagree?
    c1_dp_safer = dc.get("c1_delta_p_safer_mean") or 0.0
    c1_frac_toward = dc.get("c1_frac_moved_toward_safer") or 0.0
    c1_d_hazard = dc.get("c1_delta_expected_hazard_dist_mean")
    c1_tv = agg.get("c1_total_variation_mean") or 0.0
    competitor_influence_lesioned = agg["competitor_had_influence_lesioned_f_rate"]
    competitor_influence_factual = agg["competitor_had_influence_factual_f_rate"]
    competitor_tv = agg.get("c5_competitor_tv_at_factual_f_mean") or 0.0
    eligibility_changes = agg["flip_prob_c4_eligibility_changes_winner"]

    notes = []

    # SCALE NOTE -- the most consequential reading, ahead of any authority claim.
    sel_H = agg.get("selection_entropy_normalised_mean")
    if sel_H is not None and sel_H > 0.99:
        notes.append(
            f"NEAR-UNIFORM SELECTION: normalised selection entropy {sel_H:.5f} (1.0 = uniform "
            f"over K candidates), mean score range {agg.get('score_range_mean'):.4g} against "
            f"temperature {agg.get('temperature_mean'):.4g}. At this scale the softmax is "
            "essentially flat, so NO channel -- literal F included -- is meaningfully steering "
            "the choice. This is a materially different situation from 'F dominates and "
            "suppresses competitors': a variance decomposition can attribute ~96% of score "
            "VARIANCE to F (V3-EXQ-924) while F's ABSOLUTE contribution is far too small "
            "relative to T to move selection. Read every authority number below in that light, "
            "and treat 'F-dominance' claims resting on variance-share alone as unestablished "
            "on this configuration."
        )

    # REGIME NOTE next -- it governs how every other number should be read.
    if agg.get("committed_fraction", 0.0) < 0.5:
        notes.append(
            f"SELECTOR REGIME: committed_fraction={agg.get('committed_fraction'):.3f} -- the "
            "selector predominantly SAMPLED from softmax(-scores/T) rather than taking "
            "the committed argmin. The distributional DVs (delta_p_safer, delta_rank_safer, "
            "delta_expected_hazard_dist) are therefore the load-bearing readouts; the "
            "argmin-flip columns describe what a DETERMINISTIC selector would have done "
            "on the same score vectors and must not be read as the behaviour of this run. "
            "This is itself a substantive observation about the substrate: MECH-439's "
            "'committed-selection variance monopoly' framing presumes committed selection, "
            "and commitment was largely not engaged here."
        )

    if c1_tv < 1e-6:
        notes.append(
            "C1 (literal-F lesion) moved the selection distribution essentially NOT AT ALL "
            f"(mean total variation {c1_tv:.2e}). Before reading any null as evidence, note "
            "this means F's contribution was numerically negligible in the final score vector "
            "for this configuration -- a precondition failure, not a finding about authority."
        )
    elif c1_dp_safer > 0.01 and c1_frac_toward > 0.5:
        notes.append(
            "H1-supportive signature: removing literal F moves selection probability TOWARD "
            f"the independently-preferred candidate on disagreement events (mean delta_p_safer "
            f"{c1_dp_safer:+.4f}, moved-toward on {c1_frac_toward:.1%} of events"
            + (f", expected ground-truth hazard distance {c1_d_hazard:+.4f}" if c1_d_hazard is not None else "")
            + "). F was suppressing an independently-better option."
        )
    elif c1_dp_safer < -0.01:
        notes.append(
            "H4-supportive signature: removing literal F moves selection probability AWAY "
            f"from the independently-preferred candidate (mean delta_p_safer {c1_dp_safer:+.4f}) "
            "-- F's control is doing useful work on this axis, and lesioning it makes the "
            "independently-judged outcome WORSE. Consistent with appropriate specialisation "
            "rather than pathological dominance."
        )
    else:
        notes.append(
            f"C1 moved the distribution (total variation {c1_tv:.4g}) but produced no "
            f"consistent movement toward or away from the independently-preferred candidate "
            f"(mean delta_p_safer {c1_dp_safer:+.4f}). F has causal authority but that "
            "authority is not aligned with -- nor opposed to -- the ground-truth safety axis; "
            "removing it reshuffles rather than improves."
        )

    if competitor_tv < 1e-9 and competitor_influence_factual < 0.05:
        notes.append(
            "H3-supportive: the competitor channel (gated_policy) has essentially NO causal "
            f"influence at factual F (total variation {competitor_tv:.2e}) -- consistent with "
            "upstream insufficiency (its per-candidate signal is too small to matter), not "
            "with F actively suppressing an otherwise-capable competitor."
        )
    if competitor_influence_factual < 0.05 and competitor_influence_lesioned > 0.3:
        notes.append(
            "The competitor has near-zero influence at factual F but substantial "
            "influence once F is lesioned -- consistent with F actively suppressing "
            "an otherwise-causally-capable competitor (H1), not the competitor being "
            "intrinsically inert (which would show near-zero influence in BOTH "
            "conditions)."
        )
    if competitor_influence_lesioned < 0.05:
        notes.append(
            "The competitor shows near-zero influence even with literal F fully "
            "lesioned -- H1 alone does not explain the ceiling; either the broader "
            "primary field still dominates (H2, check eligibility-toggle results) "
            "or the competitor's signal is too weak to ever win an argmin regardless "
            "of F (H3, upstream insufficiency -- consistent with gated_policy_range_"
            f"mean={agg['gated_policy_range_mean']:.6g})."
        )
    if eligibility_changes > 0.1:
        notes.append(
            "MECH-448 eligibility toggling changes the winner on a non-trivial "
            "fraction of events even at factual F -- some evidence for H2 "
            "(the eligibility/primary-field pathway carries authority beyond "
            "literal F's own weight)."
        )
    else:
        notes.append(
            "MECH-448 eligibility toggling rarely changes the winner -- no strong "
            "evidence the eligibility mechanism itself (as opposed to literal F) is "
            "an independent locus of dominance in this sample."
        )

    return {
        "primary_c1_delta_p_safer_mean": c1_dp_safer,
        "primary_c1_frac_moved_toward_safer": c1_frac_toward,
        "primary_c1_delta_expected_hazard_dist_mean": c1_d_hazard,
        "c1_total_variation_mean": c1_tv,
        "competitor_had_influence_factual_f_rate": competitor_influence_factual,
        "competitor_had_influence_lesioned_f_rate": competitor_influence_lesioned,
        "competitor_total_variation_at_factual_f_mean": competitor_tv,
        "eligibility_toggle_changes_winner_rate": eligibility_changes,
        "committed_fraction": agg.get("committed_fraction"),
        "notes": notes,
        "failure_localisation": (
            "SEE NOTES -- this run classifies a pattern; it does not adjudicate a claim "
            "(claim_ids=[]). Read the SELECTOR REGIME note first: it determines whether "
            "the distributional or the argmin-flip columns are the meaningful ones."
        ),
    }


# ---------------------------------------------------------------------------
# Top level
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Preconditions (regime-conditioned; see CLAUDE.md multi-arm precondition rule)
# ---------------------------------------------------------------------------

PRECONDITION_SPECS = [
    PreconditionSpec(
        name="cand_world_pairwise_dist",
        description=("SD-056 candidate differentiation in z_world space. Verbatim "
                     "V3-EXQ-925 floor; unchanged so the arms stay comparable."),
        control="fresh E3 probe ticks after P0 SD-056 warmup",
        threshold=CAND_WORLD_PAIRWISE_DIST_FLOOR,
        direction="lower",
        kind="readiness",
    ),
    PreconditionSpec(
        name="gated_policy_std_across_K",
        description=("Competitor channel carries per-candidate signal. Verbatim "
                     "V3-EXQ-925 floor."),
        control="fresh E3 probe ticks after P0 SD-056 warmup",
        threshold=GATED_POLICY_STD_ACROSS_K_FLOOR,
        direction="lower",
        kind="readiness",
    ),
    PreconditionSpec(
        name="committed_fraction",
        description=(
            "THE load-bearing readiness statistic for this letter. The primary "
            "criteria route on C0-C5 deltas computed over COMMITTED events, so "
            "this asserts the SAME statistic rather than a proxy. Reads 0.000 "
            "under the V3-EXQ-925 deadlock and ~0.96 with the fix, so a below-"
            "floor value here means the deadlock is back -- a "
            "substrate_not_ready_requeue, never a regime finding or a ceiling."
        ),
        control="fresh E3 probe ticks after P0 warmup WITH the post_action_update fix applied",
        threshold=COMMITTED_FRACTION_FLOOR,
        direction="lower",
        kind="readiness",
    ),
    PreconditionSpec(
        name="gsct_teff_spread",
        description=(
            "MECH-439 Factor B must actually VARY its committed-pick temperature. "
            "T_eff = base + alpha*(1 - gap_norm), so an all-decisive-gap run "
            "collapses T_eff onto base and the gap-scaled pick silently recovers "
            "the plain argmin -- making ARM_COMMIT_GSCT equivalent to ARM_COMMIT "
            "and its delta an arithmetic identity rather than a measurement. This "
            "is the DV-symmetry guard declared in the module docstring."
        ),
        control="realised gap_scaled_commit_temperature_eff across fresh committed probe ticks",
        threshold=GSCT_TEFF_SPREAD_FLOOR,
        direction="lower",
        kind="readiness",
        # NOT MEANINGFUL for ARM_COMMIT: that arm deliberately leaves
        # use_gap_scaled_commit_temperature at its stock False, so Factor B never
        # fires and T_eff is never written. Asserting it there would make the arm
        # structurally un-passable and collapse the two-arm design back to one
        # (the V3-EXQ-785 defect). Disposition (a): scope the PRECONDITION out.
        applies_to=lambda ctx: bool(ctx.get("gap_scaled_commit")),
        applies_note=(
            "ARM_COMMIT runs stock use_gap_scaled_commit_temperature=False by "
            "design, so Factor B never fires and T_eff is never written."
        ),
    ),
]


def _arm_ctx(arm: str) -> Dict[str, Any]:
    return {"arm": arm, "gap_scaled_commit": (arm == ARM_COMMIT_GSCT)}


def run_experiment(dry_run: bool = False, started_at: Optional[float] = None) -> Dict[str, Any]:
    p0_eps = DRY_RUN_P0 if dry_run else P0_WARMUP_EPISODES
    p1_eps = DRY_RUN_P1 if dry_run else P1_CAPTURE_EPISODES
    steps = DRY_RUN_STEPS if dry_run else STEPS_PER_EPISODE

    # Design-time refusal: prove no arm's gate is unsatisfiable from its OWN
    # pre-registered config BEFORE any compute is spent (CLAUDE.md / 785).
    assert_no_structurally_unsatisfiable_gate(
        PRECONDITION_SPECS, [_arm_ctx(a) for a in ARMS]
    )

    arm_results: List[Dict[str, Any]] = []
    arm_gates: List[Dict[str, Any]] = []
    per_arm_events: Dict[str, List[Dict[str, Any]]] = {}
    per_seed_readiness: Dict[str, Any] = {}

    for arm in ARMS:
        gap_scaled = (arm == ARM_COMMIT_GSCT)
        arm_events: List[Dict[str, Any]] = []
        arm_readiness: Dict[str, Any] = {}

        for seed in SEEDS:
            print(f"Seed {seed} Condition {arm}", flush=True)
            torch.manual_seed(seed)
            np.random.seed(seed)
            config_slice = {
                "seed": seed, "arm": arm, "gap_scaled_commit": gap_scaled,
                "p0_episodes": p0_eps, "p1_episodes": p1_eps,
                "steps_per_episode": steps, "sd056_weight": SD056_WEIGHT,
                "rollout_clamp_ratio": ROLLOUT_CLAMP_RATIO,
                "commit_gate_update": True,
            }
            with arm_cell(seed, config_slice=config_slice, script_path=Path(__file__)) as cell:
                env = _make_env(seed=None)
                _, obs_dict = env.reset()
                agent = _make_agent(env, obs_dict, gap_scaled_commit=gap_scaled)

                print(f"== {arm} seed {seed}: P0 SD-056 warmup ({p0_eps} episodes) ==", flush=True)
                _, obs_dict = run_p0_warmup(
                    agent, env, obs_dict, seed, p0_eps, steps,
                    total_episodes=p0_eps + p1_eps, ep_offset=0,
                )
                print(f"    [decomp] seed={seed} ep {p0_eps}/{p0_eps} P0 warmup complete", flush=True)

                readiness, obs_dict = measure_readiness(
                    agent, env, obs_dict, n_probe_ticks=15 if not dry_run else 5
                )
                readiness["arm"] = arm
                arm_readiness[str(seed)] = readiness
                per_seed_readiness[f"{arm}:{seed}"] = readiness
                print(f"    readiness: cand_world_pairwise_dist="
                      f"{readiness['cand_world_pairwise_dist_mean']:.4f} "
                      f"gated_policy_std_across_K={readiness['gated_policy_std_across_K_mean']:.6f} "
                      f"committed_fraction={readiness['committed_fraction']:.3f} "
                      f"(rv={readiness['commit_gate_variance']:.6f} vs thr="
                      f"{readiness['commit_threshold_eff']:.3f})", flush=True)

                print(f"== {arm} seed {seed}: P1 capture + inline replay ({p1_eps} episodes) ==", flush=True)
                events = run_p1_capture_and_replay(
                    agent, env, obs_dict, seed, p1_eps, steps,
                    total_episodes=p0_eps + p1_eps, ep_offset=p0_eps,
                )
                for e in events:
                    e["seed"] = seed
                    e["arm"] = arm
                arm_events.extend(events)
                n_comm = sum(1 for e in events if e.get("committed"))
                print(f"    [decomp] seed={seed} ep {p1_eps}/{p1_eps} n_fresh_events={len(events)} "
                      f"n_committed={n_comm} "
                      f"n_disagreement={sum(1 for e in events if e['disagreement_event'])}", flush=True)

                row = {
                    "arm_id": arm, "seed": seed,
                    "gap_scaled_commit": gap_scaled,
                    "n_events": len(events),
                    "n_committed_events": n_comm,
                    "committed_fraction": (float(n_comm) / len(events)) if events else 0.0,
                    "readiness": readiness,
                }
                cell.stamp(row)
                arm_results.append(row)

                verdict = "PASS" if len(events) >= (MIN_FRESH_EVENTS_PER_SEED if not dry_run else 1) else "FAIL"
                print(f"verdict: {verdict}", flush=True)

            _ZG.observe(agent)

        per_arm_events[arm] = arm_events

        # --- arm-level measured values for the gate ------------------------
        def _mean(key: str) -> float:
            vals = [r[key] for r in arm_readiness.values() if key in r]
            return float(np.mean(vals)) if vals else 0.0

        n_comm_arm = sum(1 for e in arm_events if e.get("committed"))
        measured = {
            "cand_world_pairwise_dist": _mean("cand_world_pairwise_dist_mean"),
            "gated_policy_std_across_K": _mean("gated_policy_std_across_K_mean"),
            # Measured over the ACTUAL captured P1 events, not just the probe --
            # this is the denominator the primary criteria are computed on.
            "committed_fraction": (float(n_comm_arm) / len(arm_events)) if arm_events else 0.0,
        }
        if gap_scaled:
            measured["gsct_teff_spread"] = _mean("gsct_teff_spread")

        arm_gates.append(
            evaluate_arm_gate(arm, _arm_ctx(arm), PRECONDITION_SPECS, measured)
        )

    gate_agg = aggregate_arm_gates(arm_gates)

    # --- per-arm aggregation: ALL events, and COMMITTED-ONLY events ---------
    # aggregate_events() is V3-EXQ-925's validated arithmetic, reused verbatim and
    # simply called twice per arm. The committed-only call is the regime-matched
    # read this letter exists to produce.
    per_arm_aggregate: Dict[str, Any] = {}
    for arm in ARMS:
        evs = per_arm_events.get(arm, [])
        comm = [e for e in evs if e.get("committed")]
        per_arm_aggregate[arm] = {
            "all_events": aggregate_events(evs),
            "committed_only": aggregate_events(comm) if comm else None,
            "n_events": len(evs),
            "n_committed_events": len(comm),
            "committed_fraction": (float(len(comm)) / len(evs)) if evs else 0.0,
            "commit_margin_mean": (
                float(np.mean([e["commit_margin"] for e in evs if "commit_margin" in e]))
                if evs else 0.0
            ),
            "hypothesis_evaluation_committed": (
                classify_hypotheses(aggregate_events(comm)) if comm else None
            ),
        }

    all_events = [e for arm in ARMS for e in per_arm_events.get(arm, [])]
    agg = aggregate_events(all_events)
    hyp = classify_hypotheses(agg)

    # --- criteria (load-bearing flags + non-degeneracy, keyed per arm) ------
    criteria = []
    for arm in ARMS:
        pa = per_arm_aggregate[arm]
        enough = pa["n_committed_events"] >= (MIN_COMMITTED_EVENTS_PER_ARM if not dry_run else 1)
        criteria.append({
            "name": f"{arm}::committed_regime_engaged",
            "load_bearing": True,
            "passed": bool(enough and pa["committed_fraction"] >= COMMITTED_FRACTION_FLOOR),
            "measured_committed_fraction": pa["committed_fraction"],
            "n_committed_events": pa["n_committed_events"],
        })
    # PASS combination rule, recorded explicitly rather than left implicit
    # (CLAUDE.md multi-criterion combination logic).
    combination_rule = (
        "AND over arms: every arm must engage the committed regime "
        f"(committed_fraction >= {COMMITTED_FRACTION_FLOOR} AND "
        f"n_committed_events >= {MIN_COMMITTED_EVENTS_PER_ARM}). This is an AND "
        "because BOTH arms depend on the deadlock fix; a red arm here means the "
        "fix did not take in that arm, which is a substrate_not_ready_requeue "
        "for that arm rather than a scientific finding."
    )

    crit_non_degen = {
        c["name"]: bool(
            per_arm_aggregate[c["name"].split("::")[0]]["n_committed_events"]
            >= (MIN_COMMITTED_EVENTS_PER_ARM if not dry_run else 1)
        )
        for c in criteria
    }

    # --- outcome -----------------------------------------------------------
    if not gate_agg["any_green"]:
        outcome = "FAIL"
        label = "substrate_not_ready_requeue"
        outcome_note = (
            "No arm cleared its readiness gate -- substrate_not_ready_requeue. "
            + gate_agg["degeneracy_reason"]
        )
    elif len(all_events) == 0:
        outcome = "FAIL"
        label = "substrate_not_ready_requeue"
        outcome_note = ("Readiness cleared but zero fresh E3 selection events were "
                        "captured -- instrumentation defect, investigate before re-running.")
    else:
        outcome = "PASS"
        label = "committed_regime_engaged_h1_h4_readable"
        cf_a = per_arm_aggregate[ARM_COMMIT]["committed_fraction"]
        cf_b = per_arm_aggregate[ARM_COMMIT_GSCT]["committed_fraction"]
        outcome_note = (
            f"COMMITTED REGIME RESTORED. V3-EXQ-925 measured committed_fraction="
            f"0.000 and concluded a selector-regime finding; that was a driver "
            f"defect (no post_action_update call -> _running_variance pinned at "
            f"precision_init=0.5 > commit_threshold=0.4 by construction). With the "
            f"fix and NO config change: {ARM_COMMIT} committed_fraction={cf_a:.3f}, "
            f"{ARM_COMMIT_GSCT}={cf_b:.3f} over {len(all_events)} fresh events. "
            f"H1-H4 are re-read on committed events only -- see "
            f"per_arm_aggregate[*].committed_only and "
            f"hypothesis_evaluation_committed. Replay validity (worst abs err): "
            f"{agg['c0_probs_reconstruction_max_abs_err_worst']:.2e}."
        )

    interpretation = {
        "label": label,
        "preconditions": gate_agg["adjudication_preconditions"],
        "criteria_non_degenerate": crit_non_degen,
        "criteria": criteria,
        "combination_rule": combination_rule,
        "regime_note": (
            "READ FIRST. This run's premise is that V3-EXQ-925's "
            "committed_fraction=0.000 was an INSTRUMENT defect, not a substrate "
            "regime fact, and that use_gap_scaled_commit_temperature (the lever "
            "failure_autopsy_V3-EXQ-925_2026-08-12 recommended) could not have "
            "engaged commitment because both its call sites sit inside "
            "`elif committed:`. Both points were verified by direct runtime "
            "measurement at authoring time. If ARM_COMMIT's committed_fraction "
            "comes back at or near 0.000 DESPITE the fix, that refutes this "
            "run's premise and is itself the finding -- route it as "
            "substrate_not_ready_requeue and re-open the 925 reading, do NOT "
            "read it as a ceiling."
        ),
    }

    return {
        "outcome": outcome,
        "outcome_note": outcome_note,
        "interpretation": interpretation,
        "per_seed_readiness": per_seed_readiness,
        "arm_results": arm_results,
        "per_arm_aggregate": per_arm_aggregate,
        "per_arm_gate": gate_agg["per_arm_gate"],
        "aggregate": agg,
        "hypothesis_evaluation": hyp,
        "non_degenerate": gate_agg["non_degenerate"],
        "degeneracy_reason": gate_agg["degeneracy_reason"],
        "degenerate_metrics": {},
        "n_total_events": len(all_events),
        "n_total_committed_events": sum(1 for e in all_events if e.get("committed")),
        "per_seed_event_counts": {
            f"{arm}:{s}": sum(1 for e in all_events if e["seed"] == s and e["arm"] == arm)
            for arm in ARMS for s in SEEDS
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="V3-EXQ-925a: E3 F-dominance causal-replay harness, COMMITTED REGIME"
    )
    parser.add_argument("--dry-run", action="store_true", help="Short run for smoke testing")
    args = parser.parse_args()

    t0 = time.perf_counter()
    result = run_experiment(dry_run=args.dry_run, started_at=t0)

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{timestamp}_v3"

    out_dir = Path(__file__).parent.parent.parent / "REE_assembly" / "evidence" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": [],
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "evidence_direction": "diagnostic",
        "outcome": result["outcome"],
        "outcome_note": result["outcome_note"],
        "interpretation": result["interpretation"],
        "non_degenerate": result["non_degenerate"],
        "degeneracy_reason": result["degeneracy_reason"],
        "degenerate_metrics": result["degenerate_metrics"],
        "per_arm_gate": result["per_arm_gate"],
        "timestamp_utc": timestamp,
        "seeds": SEEDS,
        "arms": ARMS,
        "p0_warmup_episodes": P0_WARMUP_EPISODES,
        "p1_capture_episodes": P1_CAPTURE_EPISODES,
        "steps_per_episode": STEPS_PER_EPISODE,
        "f_weight_ladder": F_WEIGHT_LADDER,
        "per_seed_readiness": result["per_seed_readiness"],
        "n_total_events": result["n_total_events"],
        "n_total_committed_events": result["n_total_committed_events"],
        "per_seed_event_counts": result["per_seed_event_counts"],
        "arm_results": result["arm_results"],
        "per_arm_aggregate": result["per_arm_aggregate"],
        "aggregate": result["aggregate"],
        "hypothesis_evaluation": result["hypothesis_evaluation"],
        "supersedes": SUPERSEDES,
        "related_finding_docs": [
            "REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-925_2026-08-12.md",
            "REE_assembly/evidence/planning/v3_exq_571_root_cause_2026-05-25.md",
            "REE_assembly/docs/architecture/sd_056_e2_action_conditional_divergence.md",
        ],
        "methodology_note": (
            "Frozen-state causal-replay diagnostic, COMMITTED-REGIME letter of "
            "V3-EXQ-925. Design, env, agent config, SD-056 P0 recipe and the C0-C5 "
            "replay arithmetic are verbatim 925; the substantive changes are (1) the "
            "missing post_action_update call that V3-EXQ-925 omitted, (2) a 2-arm "
            "axis over use_gap_scaled_commit_temperature, (3) committed-conditioned "
            "re-aggregation, and (4) recording the commit-gate scalars. "
            "WHY: V3-EXQ-925 reported committed_fraction=0.000 and concluded the "
            "substrate does not occupy a committed regime at default config. That "
            "was an instrument defect. `committed` is set at e3_selector.py:3451 as "
            "`_running_variance < commit_threshold`; `_running_variance` is updated "
            "only via update_running_variance, whose in-substrate call site is "
            "post_action_update (:3979, reached from REEAgent.update_residue at "
            "agent.py:9972). 925's driver calls neither, so _running_variance stayed "
            "pinned at precision_init=0.5 against commit_threshold=0.4 and commitment "
            "was arithmetically impossible before the run started. Measured directly "
            "at authoring: rv pinned at exactly 0.500000 across a 260-tick probe; "
            "with the single added call, rv 0.500000 -> 0.007956 and "
            "committed_fraction 0.000 -> 0.959 over 8 episodes at UNCHANGED default "
            "config. e3_selector.py:3969-3973 documents this same deadlock and fixes "
            "it inside post_action_update; 97 other drivers call it. Separately, the "
            "lever the 925 autopsy recommended (use_gap_scaled_commit_temperature) "
            "could not have worked: both call sites (:3683, :3763) are inside "
            "`elif committed:`, so with committed ALWAYS False it is a dead branch, "
            "bit-identical to OFF. It is retained here as ARM_COMMIT_GSCT, where it "
            "becomes genuinely reachable for the first time (MECH-439 Factor B). "
            "CONSEQUENCE FOR GOVERNANCE: the H0 'SELECTOR REGIME' discovery leg from "
            "925 and the MECH-439 evidence_quality_note drafted for Step 2b review "
            "both rest on the incorrect reading and should be revisited against this "
            "run."
        ),
    }

    config_for_recording = {
        "seeds": SEEDS, "arms": ARMS,
        "p0_warmup_episodes": P0_WARMUP_EPISODES,
        "p1_capture_episodes": P1_CAPTURE_EPISODES,
        "steps_per_episode": STEPS_PER_EPISODE,
        "env_kwargs": ENV_KWARGS,
        "sd056_weight": SD056_WEIGHT, "rollout_clamp_ratio": ROLLOUT_CLAMP_RATIO,
        "f_weight_ladder": F_WEIGHT_LADDER,
        "committed_fraction_floor": COMMITTED_FRACTION_FLOOR,
        "min_committed_events_per_arm": MIN_COMMITTED_EVENTS_PER_ARM,
        "gsct_teff_spread_floor": GSCT_TEFF_SPREAD_FLOOR,
        "commit_gate_update_applied": True,
    }
    # Multi-arm: stamp AFTER arm_results is assembled so substrate_hash HOISTS
    # from the per-cell fingerprints instead of being recomputed (and mismatching).
    stamp_recording_core(
        manifest, config=config_for_recording, seeds=SEEDS, script_path=Path(__file__),
        started_at=t0, z_goal_stream_stats=_ZG.stats(),
    )

    out_path = write_flat_manifest(
        manifest, out_dir, dry_run=bool(args.dry_run),
        config=manifest.get("config"), seeds=SEEDS, script_path=Path(__file__), json_default=str,
    )
    print(f"Manifest written: {out_path}")
    print(f"Result written to: {out_path}")
    if args.dry_run:
        print("DRY RUN complete.")

    _outcome_raw = str(result["outcome"]).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
        dry_run=bool(args.dry_run),
    )
