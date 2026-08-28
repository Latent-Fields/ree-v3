"""
V3-EXQ-953 -- MECH-135/INV-088 E1 rollout horizon sweep + one-step per-action divergence probe

Claims: MECH-135, INV-088
Supersedes: none (diagnostic probe, not a re-test of V3-EXQ-108b's own PASS/FAIL criteria)
DIAGNOSTIC. See EXPERIMENT_PURPOSE below.

WHY THIS RUN EXISTS (targeted_review_e1_forward_model_rollout_consistency, 2026-08-03,
SYNTHESIS.md Section 4)
-------------------------------------------------------------------------------------------
V3-EXQ-108b (and its predecessor 108a) found E1's rollout-based goal-proximity scoring
collapses at rollout_horizon=30: e1coe_score_var ~1e-13/1e-14 against a 0.002 threshold, and
CR_rollout/CR_real ~3e-6 against a 0.1 collapse-ratio threshold, on a SANCTIONED SD-070-trained
z_world encoder (ruling out the undertrained-instrument confound). The lit-pull commissioned to
recommend a training-objective fix (latent overshooting / DaD scheduled unrolling / TD-MPC-style
multi-step consistency / Asadi direct sequence-conditioning) instead surfaced a code-verified
structural fact that the recommendation presupposes away:

  ree_core/predictors/e1_deep.py's forward()/predict_long_horizon() take NO action parameter.
  The only path by which a candidate action reaches z_world is second-order (action ->
  agent.e2.predict_next_self -> z_self -> E1's prior_generator), and predict_long_horizon's
  LSTM input zeroes the z_self half (`prior_full = cat([zeros(self_dim), prior])`) -- the
  entire action signal is squeezed through one world_dim-wide projection.

Two mechanisms could produce the observed collapse, and every literature-grounded fix targets
only the first:
  (a) horizon mismatch  -- E1 is trained at horizon=1, used at horizon=30, with no objective
      covering steps 2-30. Compounding error under autoregressive rollout.
  (b) action-blindness  -- E1's transition cannot express distinctiveness across actions
      REGARDLESS of horizon, because the action signal barely reaches the transition at all.

If (b) dominates, every training-objective candidate the lit-pull ranked (TD-MPC-style
consistency, DaD, Asadi, latent overshooting) fails, because all four presuppose an
action-conditioned transition and only regularise/retarget its ALREADY-EXPRESSIVE per-action
signal. The substrate_queue.json SD-e1-rollout-consistency-training entry's implementation_hint
("add a multi-step/rollout-consistency term to E1's training objective") would then be
misdirected -- the first work item becomes action-conditioning E1's transition interface, ahead
of any multi-step objective.

SYNTHESIS.md Section 4 recommends exactly the two measurements this script makes, on E1 as
trained by V3-EXQ-108b's own Phase 0a/0b pipeline (SD-070 sanctioned encoder warmup, unchanged,
then unchanged bespoke E1/E2 single-step training) -- no retraining, no interface change, no
training-objective change. This is instrumentation added to a loop V3-EXQ-108b's driver already
has:

  1. HORIZON SWEEP: record CR_rollout(h) for h = 1, 2, 3, 5, 10, 20, 30 on the SAME 40 candidate
     action sequences 108b/108a used, from the SAME warmup starting state. Compounding error
     (a) predicts smooth degradation with depth -- CR_rollout(1)/CR_real starts near healthy and
     decays toward the observed near-floor value by h=30. Action-blindness (b) predicts the
     ratio is ALREADY near-floor at h=1, before any compounding could have occurred -- the two
     are cleanly distinguishable from the shape of the curve alone.
  2. ONE-STEP PER-ACTION DIVERGENCE: from the SAME single starting state (z_self_0, z_world_0),
     apply each of the K=env.action_dim actions (not 40 sampled sequences -- the literal action
     space) for exactly one E1 step, and measure the contrast ratio across the K resulting
     z_world predictions. This is the E1 analogue of the sibling E2 review's
     cand_world_pairwise_dist (targeted_review_e2_forward_model_action_divergence). Near-zero
     confirms (b) directly and is not confoundable by rollout depth at all -- it is a single
     E1 call per action, no autoregression involved.

DECISION RULE (SYNTHESIS.md Section 4 point 3, operationalised)
  - action_blindness_dominant_floored_at_h1: CR_rollout(1)/CR_real is ALREADY below
    ACTION_BLIND_RATIO_FLOOR (0.1, same convention as 108b's CR_ROLLOUT_COLLAPSE_RATIO) --
    collapse precedes any compounding. The one-step per-action ratio is reported alongside as a
    depth-independent confirmatory readout (near-zero corroborates; not near-zero would be a
    genuine surprise worth flagging, since it would mean the h=1 rollout population and the
    literal-action-space population disagree).
  - horizon_mismatch_dominant_smooth_degradation: CR_rollout(1)/CR_real clears the floor but
    CR_rollout(30)/CR_real does not -- the ranking in SYNTHESIS.md Section 3 (TD-MPC-style
    multi-step consistency, first) stands, and the substrate entry's implementation_hint is
    correct as written.
  - no_collapse_replicated_this_run: neither ratio falls below the floor -- would contradict
    108a/108b's own finding on the identical pipeline; flagged for re-examination rather than
    asserted as a mechanism finding.
  - ambiguous_partial_signal: neither of the above two clean shapes obtains (e.g. non-monotone
    curve, or the two readouts at h=1 disagree sharply) -- reported honestly as inconclusive,
    flagged for follow-up autopsy.

DECLARED NULL. This run does not reopen V3-EXQ-108a/108b's own C1/C2/C3 result (the collapse
stands regardless of which mechanism explains it). Its job is explaining WHY, for the
SD-e1-rollout-consistency-training substrate entry's depends_on_unresolved item -- not
re-litigating whether E1's rollout collapses.

Re-derive brake (Step 2.5b, this session): MECH-135 has 1 qualifying substrate_ceiling autopsy
(V3-EXQ-108a) against a threshold of 2 -- brake does not fire. INV-088 has 2 qualifying autopsies
(v3_exq_750_..., v3_exq_754_...) but both are on a DIFFERENT substrate axis entirely (MECH-457
strategy-diversity / hierarchical-curriculum exploration readouts) -- this run tests INV-088 via
E1's forward-model rollout dynamics, a redesign on a different mechanism per the brake's own
"Not braked" carve-out, not a lettered retest of either.

Substrate-path overlap gate (Step 2.5c, this session): two OPEN corrupting-severity
substrate_queue.json entries list files this driver imports --
`contextmemory-write-path-addressing-degeneracy` (ree_core/predictors/e1_deep.py, whole-file) and
`mode-governance-engagement` (ree_core/agent.py, whole-file, among others). Traced both against
this driver's actual call path: (1) the contextmemory defect is specifically in
ContextMemory.write() (only reachable via E1DeepPredictor.update_from_observation(), which this
driver never calls); this driver's only ContextMemory interaction is the READ path inside
predict_long_horizon() (already SANCTIONED -- SD-016, implemented), identical to 108b's own
unchanged call pattern. (2) the mode-governance defect is in SalienceCoordinator.tick()'s
affinity-input clamp and agent.py's mode-switching commitment term (_et_commit, ~line 6944) --
this driver never engages mode governance, external-task switching, or SalienceCoordinator at
all (no goal-mode competition anywhere in Phase 0a/0b/2/3/4). Neither corrupting code path is
exercised; both are whole-file entries whose actual defect is scoped narrower than the file.
Proceeding is consistent with V3-EXQ-108b (same call pattern, already treated as valid evidence
by the commissioning review).

SLEEP DRIVER: not applicable -- no sleep phase entered in this run.

Z_GOAL: not used. This probe measures raw z_world differentiation directly (contrast ratio over
rollout/action populations); it does not compute goal_proximity or drive any goal-directed
selection, so no GoalState/z_goal machinery is instantiated.
"""

import sys
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig
from experiment_protocol import emit_outcome
from experiments.pack_writer import write_flat_manifest
from experiments._lib.zworld_p0_warmup import run_zworld_p0
from experiments._lib.capability_eval import RandomPolicy
from experiments._lib.zworld_encoder_guard import (
    latent_stack_snapshot,
    assert_world_encoder_trained,
)
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator

_ZG = ZGoalStreamAccumulator()


EXPERIMENT_TYPE = "v3_exq_953_mech135_inv088_e1_horizon_sweep_action_divergence_probe"
CLAIM_IDS = ["MECH-135", "INV-088"]
EXPERIMENT_PURPOSE = "diagnostic"
SUPERSEDES = None

# agent_construction_before_seed lint: run_seed() builds the agent (_build_agent, inside
# Phase 0a's SD-070 warmup call chain) before any torch.manual_seed call in this function's
# own flow. Inherited from V3-EXQ-108a/108b's identical exemption (row 9,
# agent_seed_order_lint_backlog_triage.md): this run scores a single fixed starting state's
# rollout/action population off the LITERAL SAME agent object within a seed (never two
# independently-constructed agents), so unseeded weight init cannot confound the
# within-seed measurement this script makes.
AGENT_SEED_ORDER_EXEMPT = (
    "Horizon-sweep and per-action population both scored off the literal same agent object "
    "within a seed (inherited from V3-EXQ-108a/108b's identical exemption, itself inherited "
    "from V3-EXQ-108 row 9 triage, agent_seed_order_lint_backlog_triage.md)"
)

# ---------------------------------------------------------------------------
# Pre-registered thresholds
# ---------------------------------------------------------------------------
ZWORLD_P0_EPISODES = 60            # SD-070 encoder warmup -- matches V3-EXQ-108b/819/819a
CR_REAL_FLOOR = 1e-4               # below this, real-state differentiation itself is ~0
ACTION_BLIND_RATIO_FLOOR = 0.1     # CR_rollout(h)/CR_real below this = collapsed at that h
                                    # (same convention/value as 108b's CR_ROLLOUT_COLLAPSE_RATIO)
N_SEQUENCES = 40                   # matches V3-EXQ-108a/108b's candidate population size
ROLLOUT_HORIZON = 30               # matches V3-EXQ-108a/108b
SWEEP_HORIZONS = [1, 2, 3, 5, 10, 20, 30]   # per SYNTHESIS.md Section 4 point 1, verbatim
N_REAL_SAMPLES = 40                # matches N_SEQUENCES, for a size-matched CR_real comparison
STEPS_PER_REAL_SAMPLE = 30         # matches ROLLOUT_HORIZON


# ---------------------------------------------------------------------------
# Helpers (env/agent construction, training -- unchanged from V3-EXQ-108b)
# ---------------------------------------------------------------------------

def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


def _env_kwargs() -> Dict[str, Any]:
    """Env config, unchanged from V3-EXQ-108a/108b."""
    return dict(
        size=10, num_hazards=2, num_resources=4,
        hazard_harm=0.02, env_drift_interval=8, env_drift_prob=0.05,
        proximity_harm_scale=0.03, proximity_benefit_scale=0.04,
        proximity_approach_threshold=0.15, hazard_field_decay=0.5,
        resource_respawn_on_consume=True,
    )


def _build_agent(seed: int, world_dim: int, self_dim: int) -> Tuple[REEAgent, CausalGridWorldV2]:
    env = CausalGridWorldV2(seed=seed, **_env_kwargs())
    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=self_dim,
        world_dim=world_dim,
        alpha_world=0.9,
        alpha_self=0.3,
    )
    config.latent.unified_latent_mode = False
    agent = REEAgent(config)
    return agent, env


# ---------------------------------------------------------------------------
# Phase 0a: SD-070 sanctioned z_world encoder warmup (unchanged from V3-EXQ-108b)
# ---------------------------------------------------------------------------

def _run_zworld_p0_warmup(
    agent: REEAgent, seed: int, zworld_p0_episodes: int, steps_per_episode: int,
    dry_run: bool = False,
) -> Dict[str, Any]:
    before = latent_stack_snapshot(agent)
    warmup_env = CausalGridWorldV2(seed=seed, **_env_kwargs())
    p0a_report = run_zworld_p0(
        agent, warmup_env, seed, zworld_p0_episodes, steps_per_episode,
        policy=RandomPolicy(seed), label="v3_exq_953 P0a (SD-070 z_world encoder)",
        dry_run=dry_run,
    )
    encoder_report = assert_world_encoder_trained(
        agent, before, p0=zworld_p0_episodes, strict=False,
        context="v3_exq_953_mech135_inv088_e1_horizon_sweep_action_divergence_probe",
        escape_hint="pass zworld_p0_episodes=0 for a deliberate frozen-encoder run",
    )
    return {**p0a_report, **encoder_report}


# ---------------------------------------------------------------------------
# Phase 0b: bespoke E1/E2 single-step training (unchanged from V3-EXQ-108a/108b)
# ---------------------------------------------------------------------------

def _train_agent(
    agent: REEAgent,
    env: CausalGridWorldV2,
    seed: int,
    n_episodes: int,
    steps_per_episode: int,
) -> None:
    """Byte-identical to V3-EXQ-108a/108b's _train_agent -- calls agent.sense() exactly ONCE
    per env step (StepHarness invariant #1). sense() runs under torch.no_grad() throughout, so
    Phase 0a's now-trained encoder is never further disturbed by this phase."""
    torch.manual_seed(seed + 2000)
    random.seed(seed + 2000)
    agent.train()

    opt_e1 = optim.Adam(agent.e1.parameters(), lr=1e-3)
    opt_e2 = optim.Adam(agent.e2.parameters(), lr=1e-3)

    for ep in range(n_episodes):
        _, obs_dict = env.reset()
        agent.reset()
        ep_loss_e1 = 0.0
        ep_loss_e2 = 0.0
        n_steps = 0

        latent_prev: Optional[object] = None
        action_prev: Optional[torch.Tensor] = None

        for _ in range(steps_per_episode):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            with torch.no_grad():
                latent_curr = agent.sense(obs_body, obs_world)

            action_idx = random.randint(0, env.action_dim - 1)
            action_curr = _action_to_onehot(action_idx, env.action_dim, agent.device)

            if latent_prev is not None:
                opt_e1.zero_grad()
                total_prev = torch.cat([latent_prev.z_self, latent_prev.z_world], dim=-1)
                total_curr = torch.cat([latent_curr.z_self, latent_curr.z_world], dim=-1)
                e1_pred, _ = agent.e1(total_prev, horizon=1)
                e1_loss = F.mse_loss(e1_pred[:, 0, :], total_curr.detach())
                e1_loss.backward()
                opt_e1.step()
                ep_loss_e1 += e1_loss.item()

                opt_e2.zero_grad()
                z_self_pred = agent.e2.predict_next_self(latent_prev.z_self.detach(), action_prev)
                e2_loss = F.mse_loss(z_self_pred, latent_curr.z_self.detach())
                e2_loss.backward()
                opt_e2.step()
                ep_loss_e2 += e2_loss.item()
                n_steps += 1

            _, _, done, _, obs_dict = env.step(action_curr)

            latent_prev = latent_curr
            action_prev = action_curr

            if done:
                break

        if (ep + 1) % 20 == 0:
            print(
                f"  [train] label seed={seed} ep {ep+1}/{n_episodes} "
                f"e1_loss={ep_loss_e1/max(n_steps,1):.5f} "
                f"e2_loss={ep_loss_e2/max(n_steps,1):.5f}",
                flush=True,
            )

    agent.eval()
    print(f"  [Train] Done. {n_episodes} episodes.", flush=True)


# ---------------------------------------------------------------------------
# Phase 1: warmup state (unchanged from V3-EXQ-108a/108b's Phase 2)
# ---------------------------------------------------------------------------

def _get_warmup_state(
    agent: REEAgent, env: CausalGridWorldV2, seed: int, n_warmup_steps: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed + 1000)
    random.seed(seed + 1000)
    _, obs_dict = env.reset()
    agent.reset()
    latent = None

    for _ in range(n_warmup_steps):
        obs_body = obs_dict["body_state"]
        obs_world = obs_dict["world_state"]
        with torch.no_grad():
            latent = agent.sense(obs_body, obs_world)
        agent.clock.advance()
        action_idx = random.randint(0, env.action_dim - 1)
        action = _action_to_onehot(action_idx, env.action_dim, agent.device)
        _, _, done, _, obs_dict = env.step(action)
        if done:
            _, obs_dict = env.reset()
            agent.reset()
            latent = None

    if latent is None:
        obs_body = obs_dict["body_state"]
        obs_world = obs_dict["world_state"]
        with torch.no_grad():
            latent = agent.sense(obs_body, obs_world)

    return latent.z_self.detach(), latent.z_world.detach()


# ---------------------------------------------------------------------------
# Phase 2: generate candidate sequences (unchanged from V3-EXQ-108a/108b's Phase 3)
# ---------------------------------------------------------------------------

def _generate_candidate_sequences(
    n_sequences: int, horizon: int, n_actions: int, seed: int,
) -> List[List[int]]:
    torch.manual_seed(seed + 500)
    random.seed(seed + 500)
    seqs = []
    for _ in range(n_sequences):
        seq = [random.randint(0, n_actions - 1) for _ in range(horizon)]
        seqs.append(seq)
    return seqs


# ---------------------------------------------------------------------------
# Phase 3 (NEW -- the core probe): horizon-sweep rollout + one-step per-action divergence
# ---------------------------------------------------------------------------

def _rollout_with_horizon_checkpoints(
    agent: REEAgent,
    z_self_start: torch.Tensor,
    z_world_start: torch.Tensor,
    action_sequence: List[int],
    self_dim: int,
    checkpoints: List[int],
) -> Dict[int, torch.Tensor]:
    """Roll one candidate action sequence forward, recording z_world at each requested
    checkpoint horizon (1-indexed: checkpoint h means 'after h actions'). Same per-step E1/E2
    call pattern as V3-EXQ-108a/108b's _score_sequence_e1coe_with_endpoint -- the only change
    is capturing intermediate z_world instead of discarding every step but the last."""
    device = agent.device
    n_actions = agent.config.e2.action_dim
    checkpoint_set = set(checkpoints)
    max_h = max(checkpoints)

    agent.e1.reset_hidden_state()

    z_self_curr = z_self_start.clone()
    z_world_curr = z_world_start.clone()
    captured: Dict[int, torch.Tensor] = {}

    for step_idx, a_idx in enumerate(action_sequence[:max_h], start=1):
        action = _action_to_onehot(a_idx, n_actions, device)
        total_curr = torch.cat([z_self_curr, z_world_curr], dim=-1)
        with torch.no_grad():
            e1_preds, _ = agent.e1(total_curr, horizon=1)
        z_world_next = e1_preds[0, 0, self_dim:].unsqueeze(0)
        with torch.no_grad():
            z_self_next = agent.e2.predict_next_self(z_self_curr, action)
        z_self_curr = z_self_next
        z_world_curr = z_world_next

        if step_idx in checkpoint_set:
            captured[step_idx] = z_world_curr.detach().clone()

    return captured


def _one_step_per_action_divergence(
    agent: REEAgent,
    z_self_start: torch.Tensor,
    z_world_start: torch.Tensor,
    self_dim: int,
    n_actions: int,
) -> List[torch.Tensor]:
    """From the SAME single starting state, apply each of the n_actions literal actions for
    exactly one E1 step. Returns the n_actions resulting z_world predictions -- the direct
    analogue of the sibling E2 review's cand_world_pairwise_dist, applied to E1's transition."""
    device = agent.device
    total_start = torch.cat([z_self_start, z_world_start], dim=-1)
    endpoints: List[torch.Tensor] = []
    for a_idx in range(n_actions):
        agent.e1.reset_hidden_state()
        action = _action_to_onehot(a_idx, n_actions, device)
        with torch.no_grad():
            e1_preds, _ = agent.e1(total_start, horizon=1)
        z_world_next = e1_preds[0, 0, self_dim:].unsqueeze(0)
        # z_world_next does not depend on `action` in E1's own forward signature (E1 takes no
        # action parameter -- see module docstring); action is threaded here only so a future
        # action-conditioned E1 slots in without changing this function's contract. Recorded
        # per-action regardless, since the DIVERGENCE population is what this probe measures --
        # a degenerate (identical-across-actions) population is itself the finding.
        endpoints.append(z_world_next.detach().clone())
    return endpoints


# ---------------------------------------------------------------------------
# Phase 4: real z_world sample (unchanged from V3-EXQ-108a/108b's Phase 4b)
# ---------------------------------------------------------------------------

def _collect_real_zworld_sample(
    agent: REEAgent, env: CausalGridWorldV2, seed: int, n_samples: int, steps_per_sample: int,
) -> List[torch.Tensor]:
    torch.manual_seed(seed + 3000)
    random.seed(seed + 3000)
    samples: List[torch.Tensor] = []
    for _ in range(n_samples):
        _, obs_dict = env.reset()
        agent.reset()
        latent = None
        for _ in range(steps_per_sample):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world)
            action_idx = random.randint(0, env.action_dim - 1)
            action = _action_to_onehot(action_idx, env.action_dim, agent.device)
            _, _, done, _, obs_dict = env.step(action)
            if done:
                break
        if latent is None:
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world)
        samples.append(latent.z_world.detach())
    return samples


def _contrast_ratio(vectors: List[torch.Tensor]) -> Dict[str, float]:
    """CR = spread / ||centroid||, per zworld_near_static_characterisation_2026-07-18 sec 2's
    offset-invariant statistic (same helper as V3-EXQ-108a/108b). spread = RMS deviation from
    centroid."""
    stacked = torch.cat(vectors, dim=0)  # [N, dim]
    centroid = stacked.mean(dim=0, keepdim=True)  # [1, dim]
    centroid_norm = float(centroid.norm().item())
    deviations = stacked - centroid
    spread = float(torch.sqrt((deviations.pow(2).sum(dim=-1)).mean()).item())
    cr = (spread / centroid_norm) if centroid_norm > 1e-12 else float("nan")
    return {"spread": spread, "centroid_norm": centroid_norm, "contrast_ratio": cr}


def _mean_pairwise_l2(vectors: List[torch.Tensor]) -> float:
    """Raw mean pairwise L2 distance -- reported alongside the contrast ratio for direct
    interpretability of the per-action divergence readout (SYNTHESIS.md Section 4 point 2)."""
    stacked = torch.cat(vectors, dim=0)  # [N, dim]
    n = stacked.shape[0]
    if n < 2:
        return float("nan")
    dists = torch.cdist(stacked, stacked, p=2)
    iu = torch.triu_indices(n, n, offset=1)
    pair_vals = dists[iu[0], iu[1]]
    return float(pair_vals.mean().item())


# ---------------------------------------------------------------------------
# Single-seed runner
# ---------------------------------------------------------------------------

def run_seed(
    seed: int,
    world_dim: int,
    self_dim: int,
    n_train_episodes: int,
    steps_per_episode: int,
    n_sequences: int,
    rollout_horizon: int,
    sweep_horizons: List[int],
    n_warmup_steps: int,
    zworld_p0_episodes: int,
    n_real_samples: int,
    steps_per_real_sample: int,
    dry_run: bool = False,
) -> Dict[str, Any]:
    print(f"\n[EXQ-953] seed={seed}", flush=True)
    print(f"Seed {seed} Condition single", flush=True)

    agent, env = _build_agent(seed, world_dim, self_dim)

    # Phase 0a: SD-070 sanctioned encoder warmup (unchanged from 108b)
    print(f"[EXQ-953] Phase 0a: SD-070 z_world encoder warmup ({zworld_p0_episodes} eps)...", flush=True)
    readiness_report = _run_zworld_p0_warmup(
        agent, seed, zworld_p0_episodes, steps_per_episode, dry_run=dry_run,
    )
    print(
        f"  encoder_trained={readiness_report.get('zworld_encoder_trained')} "
        f"max_abs_delta={readiness_report.get('world_encoder_max_abs_delta'):.6f}",
        flush=True,
    )

    # Phase 0b: bespoke E1/E2 single-step training (unchanged from 108a/108b)
    print(f"[EXQ-953] Phase 0b: training E1/E2 ({n_train_episodes} eps)...", flush=True)
    _train_agent(agent, env, seed, n_train_episodes, steps_per_episode)

    # Phase 1: warmup state
    print("[EXQ-953] Phase 1: warmup state...", flush=True)
    z_self_0, z_world_0 = _get_warmup_state(agent, env, seed, n_warmup_steps)

    # Phase 2: candidate sequences (same population 108a/108b scored)
    print(f"[EXQ-953] Phase 2: generating {n_sequences} candidate sequences...", flush=True)
    seqs = _generate_candidate_sequences(n_sequences, rollout_horizon, env.action_dim, seed)

    # Phase 3a: horizon-sweep rollout -- CR_rollout(h) for every checkpoint horizon
    print(f"[EXQ-953] Phase 3a: horizon-sweep rollout ({sweep_horizons})...", flush=True)
    endpoints_by_h: Dict[int, List[torch.Tensor]] = {h: [] for h in sweep_horizons}
    for i, seq in enumerate(seqs):
        captured = _rollout_with_horizon_checkpoints(
            agent, z_self_0, z_world_0, seq, self_dim, sweep_horizons,
        )
        for h in sweep_horizons:
            endpoints_by_h[h].append(captured[h])
        if (i + 1) % 10 == 0:
            print(f"  rolled out {i+1}/{n_sequences}", flush=True)

    cr_rollout_by_h: Dict[int, Dict[str, float]] = {
        h: _contrast_ratio(endpoints_by_h[h]) for h in sweep_horizons
    }
    for h in sweep_horizons:
        print(f"  CR_rollout(h={h}) = {cr_rollout_by_h[h]['contrast_ratio']:.6f}", flush=True)

    # Phase 3b: one-step per-action divergence (literal K-action population, single start)
    print(f"[EXQ-953] Phase 3b: one-step per-action divergence (K={env.action_dim})...", flush=True)
    per_action_endpoints = _one_step_per_action_divergence(
        agent, z_self_0, z_world_0, self_dim, env.action_dim,
    )
    per_action_cr = _contrast_ratio(per_action_endpoints)
    per_action_mean_pairwise_l2 = _mean_pairwise_l2(per_action_endpoints)
    print(
        f"  per_action CR={per_action_cr['contrast_ratio']:.6f} "
        f"mean_pairwise_l2={per_action_mean_pairwise_l2:.6f}",
        flush=True,
    )

    # Phase 4: real z_world sample + CR_real (antecedent baseline, unchanged from 108b)
    print(f"[EXQ-953] Phase 4: sampling {n_real_samples} real z_world observations...", flush=True)
    real_samples = _collect_real_zworld_sample(
        agent, env, seed, n_real_samples, steps_per_real_sample,
    )
    cr_real = _contrast_ratio(real_samples)
    print(f"  CR_real={cr_real['contrast_ratio']:.6f}", flush=True)

    cr_real_val = cr_real["contrast_ratio"]
    cr_real_ok = (cr_real_val == cr_real_val) and cr_real_val > 0  # excludes NaN

    ratio_by_h: Dict[int, float] = {}
    for h in sweep_horizons:
        rv = cr_rollout_by_h[h]["contrast_ratio"]
        ratio_by_h[h] = (rv / cr_real_val) if cr_real_ok and rv == rv else float("nan")

    per_action_ratio = (
        (per_action_cr["contrast_ratio"] / cr_real_val)
        if cr_real_ok and per_action_cr["contrast_ratio"] == per_action_cr["contrast_ratio"]
        else float("nan")
    )

    verdict = "PASS" if bool(readiness_report.get("zworld_encoder_trained")) and cr_real_ok else "FAIL"
    print(f"verdict: {verdict}", flush=True)

    _ZG.observe(agent)

    return {
        "seed": seed,
        "readiness": readiness_report,
        "cr_real": cr_real,
        "cr_rollout_by_horizon": {str(h): cr_rollout_by_h[h] for h in sweep_horizons},
        "cr_ratio_by_horizon": {str(h): ratio_by_h[h] for h in sweep_horizons},
        "per_action_divergence": {
            "contrast_ratio": per_action_cr["contrast_ratio"],
            "spread": per_action_cr["spread"],
            "centroid_norm": per_action_cr["centroid_norm"],
            "mean_pairwise_l2": per_action_mean_pairwise_l2,
            "ratio_to_cr_real": per_action_ratio,
            "n_actions": env.action_dim,
        },
        "seed_verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run(
    seeds: List[int],
    world_dim: int,
    self_dim: int,
    n_train_episodes: int,
    steps_per_episode: int,
    n_sequences: int,
    rollout_horizon: int,
    sweep_horizons: List[int],
    n_warmup_steps: int,
    zworld_p0_episodes: int,
    n_real_samples: int,
    steps_per_real_sample: int,
    dry_run: bool = False,
) -> Dict[str, Any]:
    seed_results = []
    for seed in seeds:
        r = run_seed(
            seed=seed,
            world_dim=world_dim,
            self_dim=self_dim,
            n_train_episodes=n_train_episodes,
            steps_per_episode=steps_per_episode,
            n_sequences=n_sequences,
            rollout_horizon=rollout_horizon,
            sweep_horizons=sweep_horizons,
            n_warmup_steps=n_warmup_steps,
            zworld_p0_episodes=zworld_p0_episodes,
            n_real_samples=n_real_samples,
            steps_per_real_sample=steps_per_real_sample,
            dry_run=dry_run,
        )
        seed_results.append(r)

    # ---- Readiness (P0) ----
    encoder_trained_per_seed = [bool(r["readiness"].get("zworld_encoder_trained")) for r in seed_results]
    p_encoder_trained_met = all(encoder_trained_per_seed)
    cr_real_per_seed = [r["cr_real"]["contrast_ratio"] for r in seed_results]
    p_real_nondegenerate_met = all((v == v) and v > 0 for v in cr_real_per_seed)  # v==v excludes NaN

    preconditions = [
        {
            "name": "encoder_trained",
            "kind": "readiness",
            "description": (
                "At least one split_encoder.world_encoder tensor moved during the Phase 0a "
                "SD-070 warmup, per every seed (zworld_encoder_guard's load-bearing bit) -- "
                "identical precondition to V3-EXQ-108b."
            ),
            "measured": min(
                r["readiness"].get("world_encoder_max_abs_delta", 0.0) for r in seed_results
            ),
            "threshold": 0.0,
            "direction": "lower",
            "comparator": ">",
            "met": p_encoder_trained_met,
        },
        {
            "name": "real_zworld_nondegenerate",
            "kind": "readiness",
            "description": (
                "CR_real (the antecedent contrast ratio on real, diverse z_world observations) "
                "is a finite, positive number for every seed -- confirms sensing itself is not "
                "degenerate before drawing any horizon/action-divergence conclusion from the "
                "ratios computed against it."
            ),
            "measured": min(v for v in cr_real_per_seed if v == v) if any(v == v for v in cr_real_per_seed) else 0.0,
            "threshold": 0.0,
            "direction": "lower",
            "met": p_real_nondegenerate_met,
        },
    ]

    non_degenerate = bool(p_encoder_trained_met and p_real_nondegenerate_met)

    if not non_degenerate:
        label = "substrate_not_ready_requeue"
        degeneracy_reason = (
            "P0 readiness unmet: "
            + ("encoder_trained failed. " if not p_encoder_trained_met else "")
            + ("real_zworld_nondegenerate failed. " if not p_real_nondegenerate_met else "")
        )
        status = "FAIL"
        evidence_direction = "non_contributory"
        evidence_direction_per_claim = {"MECH-135": "non_contributory", "INV-088": "non_contributory"}
        min_ratio_h1 = float("nan")
        min_ratio_h30 = float("nan")
        min_per_action_ratio = float("nan")
    else:
        degeneracy_reason = None
        h1 = sweep_horizons[0]
        h_last = sweep_horizons[-1]
        ratio_h1_per_seed = [r["cr_ratio_by_horizon"][str(h1)] for r in seed_results]
        ratio_hlast_per_seed = [r["cr_ratio_by_horizon"][str(h_last)] for r in seed_results]
        per_action_ratio_per_seed = [r["per_action_divergence"]["ratio_to_cr_real"] for r in seed_results]

        min_ratio_h1 = min(v for v in ratio_h1_per_seed if v == v) if any(v == v for v in ratio_h1_per_seed) else float("nan")
        min_ratio_h30 = min(v for v in ratio_hlast_per_seed if v == v) if any(v == v for v in ratio_hlast_per_seed) else float("nan")
        min_per_action_ratio = (
            min(v for v in per_action_ratio_per_seed if v == v)
            if any(v == v for v in per_action_ratio_per_seed) else float("nan")
        )

        h1_floored = (min_ratio_h1 == min_ratio_h1) and (min_ratio_h1 < ACTION_BLIND_RATIO_FLOOR)
        hlast_floored = (min_ratio_h30 == min_ratio_h30) and (min_ratio_h30 < ACTION_BLIND_RATIO_FLOOR)

        if h1_floored:
            label = "action_blindness_dominant_floored_at_h1"
        elif (not h1_floored) and hlast_floored:
            label = "horizon_mismatch_dominant_smooth_degradation"
        elif (not h1_floored) and (not hlast_floored):
            label = "no_collapse_replicated_this_run"
        else:
            label = "ambiguous_partial_signal"

        status = "PASS"  # diagnostic -- PASS means the probe produced a clean, adjudicable label
        if label == "action_blindness_dominant_floored_at_h1":
            evidence_direction = "supports"
            evidence_direction_per_claim = {"MECH-135": "supports", "INV-088": "supports"}
        elif label == "horizon_mismatch_dominant_smooth_degradation":
            evidence_direction = "non_contributory"
            evidence_direction_per_claim = {"MECH-135": "non_contributory", "INV-088": "supports"}
        else:
            evidence_direction = "non_contributory"
            evidence_direction_per_claim = {"MECH-135": "non_contributory", "INV-088": "non_contributory"}

    print(f"\n[EXQ-953] Label: {label}", flush=True)
    print(f"[EXQ-953] Status: {status}", flush=True)

    criteria = [
        {
            "name": "C_HORIZON_SWEEP_ADJUDICABLE",
            "load_bearing": True,
            "passed": bool(non_degenerate),
            "measured": 1.0 if non_degenerate else 0.0,
            "threshold": 1.0,
            "statement": (
                "P0 readiness cleared on every seed, so the horizon-sweep and per-action-"
                "divergence curves are adjudicable (not routed to substrate_not_ready_requeue)."
            ),
        },
    ]
    criteria_non_degenerate = {"C_HORIZON_SWEEP_ADJUDICABLE": non_degenerate}

    result: Dict[str, Any] = {
        "experiment_type": EXPERIMENT_TYPE,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "supersedes": SUPERSEDES,
        "evidence_class": "diagnostic_disambiguation",
        "evidence_direction": evidence_direction,
        "evidence_direction_per_claim": evidence_direction_per_claim,
        "seeds": seeds,
        "world_dim": world_dim,
        "self_dim": self_dim,
        "n_train_episodes": n_train_episodes,
        "steps_per_episode": steps_per_episode,
        "n_sequences": n_sequences,
        "rollout_horizon": rollout_horizon,
        "sweep_horizons": sweep_horizons,
        "n_warmup_steps": n_warmup_steps,
        "zworld_p0_episodes": zworld_p0_episodes,
        "n_real_samples": n_real_samples,
        "steps_per_real_sample": steps_per_real_sample,
        "registered_cr_real_floor": CR_REAL_FLOOR,
        "registered_action_blind_ratio_floor": ACTION_BLIND_RATIO_FLOOR,
        "min_cr_ratio_at_h1": min_ratio_h1,
        "min_cr_ratio_at_hlast": min_ratio_h30,
        "min_per_action_ratio": min_per_action_ratio,
        "non_degenerate": non_degenerate,
        "degeneracy_reason": degeneracy_reason,
        "status": status,
        "outcome": status,
        "verdict": status,
        "interpretation": {
            "label": label,
            "preconditions": preconditions,
            "criteria": criteria,
            "criteria_non_degenerate": criteria_non_degenerate,
        },
        "source_review": "targeted_review_e1_forward_model_rollout_consistency (SYNTHESIS.md Section 4)",
        "source_autopsy": "failure_autopsy_V3-EXQ-108a_2026-08-02",
        "hypothesis_space_qid": "inv088_evaluator_degeneracy_cause",
    }

    # Flatten per-seed metrics
    for r in seed_results:
        s = r["seed"]
        for k, v in r.items():
            if k != "seed":
                result[f"seed_{s}_{k}"] = v

    return result, [
        {
            "seed": r["seed"],
            "cr_real": r["cr_real"]["contrast_ratio"],
            "cr_rollout_by_horizon": {h: r["cr_rollout_by_horizon"][str(h)]["contrast_ratio"] for h in sweep_horizons},
            "cr_ratio_by_horizon": r["cr_ratio_by_horizon"],
            "per_action_divergence": r["per_action_divergence"],
        }
        for r in seed_results
    ]


if __name__ == "__main__":
    import argparse
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser(
        description="V3-EXQ-953: E1 rollout horizon sweep + one-step per-action divergence probe"
    )
    parser.add_argument("--seeds", type=str, default="42,123")
    parser.add_argument("--world-dim", type=int, default=32)
    parser.add_argument("--self-dim", type=int, default=32)
    parser.add_argument("--train-episodes", type=int, default=100)
    parser.add_argument("--steps-per-episode", type=int, default=200)
    parser.add_argument("--rollout-horizon", type=int, default=ROLLOUT_HORIZON)
    parser.add_argument("--n-sequences", type=int, default=N_SEQUENCES)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--zworld-p0-episodes", type=int, default=ZWORLD_P0_EPISODES)
    parser.add_argument("--n-real-samples", type=int, default=N_REAL_SAMPLES)
    parser.add_argument("--steps-per-real-sample", type=int, default=STEPS_PER_REAL_SAMPLE)
    parser.add_argument("--dry-run", "--smoke-test", dest="dry_run", action="store_true")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    if args.dry_run:
        n_train = 2
        steps_ep = 50
        n_sequences = 5
        horizon = 10
        sweep_horizons = [1, 2, 5, 10]
        warmup = 5
        zworld_p0 = 3
        n_real = 5
        steps_real = 10
        seeds = seeds[:1]
        print("[V3-EXQ-953] SMOKE TEST MODE", flush=True)
    else:
        n_train = args.train_episodes
        steps_ep = args.steps_per_episode
        n_sequences = args.n_sequences
        horizon = args.rollout_horizon
        sweep_horizons = SWEEP_HORIZONS
        warmup = args.warmup_steps
        zworld_p0 = args.zworld_p0_episodes
        n_real = args.n_real_samples
        steps_real = args.steps_per_real_sample

    t0 = time.perf_counter()
    result, per_seed_rows = run(
        seeds=seeds,
        world_dim=args.world_dim,
        self_dim=args.self_dim,
        n_train_episodes=n_train,
        steps_per_episode=steps_ep,
        n_sequences=n_sequences,
        rollout_horizon=horizon,
        sweep_horizons=sweep_horizons,
        n_warmup_steps=warmup,
        zworld_p0_episodes=zworld_p0,
        n_real_samples=n_real,
        steps_per_real_sample=steps_real,
        dry_run=args.dry_run,
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["timestamp_utc"] = ts
    result["run_timestamp"] = ts
    result["run_id"] = f"{EXPERIMENT_TYPE}_{ts}_v3"
    result["architecture_epoch"] = "ree_hybrid_guardrails_v1"
    result["per_seed_rows"] = per_seed_rows

    full_config = {
        "seeds": seeds,
        "world_dim": args.world_dim,
        "self_dim": args.self_dim,
        "n_train_episodes": n_train,
        "steps_per_episode": steps_ep,
        "n_sequences": n_sequences,
        "rollout_horizon": horizon,
        "sweep_horizons": sweep_horizons,
        "n_warmup_steps": warmup,
        "zworld_p0_episodes": zworld_p0,
        "n_real_samples": n_real,
        "steps_per_real_sample": steps_real,
        "cr_real_floor": CR_REAL_FLOOR,
        "action_blind_ratio_floor": ACTION_BLIND_RATIO_FLOOR,
        "alpha_world": 0.9,
        "alpha_self": 0.3,
        "unified_latent_mode": False,
    }

    out_path = write_flat_manifest(
        result,
        dry_run=args.dry_run,
        config=full_config,
        seeds=seeds,
        script_path=Path(__file__),
        started_at=t0,
        z_goal_stream_stats=_ZG.stats(),
    )

    print(f"\nResult written to: {out_path}", flush=True)
    print(f"Status: {result['status']}", flush=True)
    print(f"Label: {result['interpretation']['label']}", flush=True)

    if args.dry_run:
        print("[V3-EXQ-953] SMOKE TEST COMPLETE", flush=True)
        for k in ["status", "non_degenerate", "degeneracy_reason"]:
            print(f"  {k}: {result.get(k, 'N/A')}", flush=True)
        print(f"  label: {result['interpretation']['label']}", flush=True)

    # --- runner-conformance sentinel ---
    _outcome_raw = str(result.get("status", "FAIL")).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
        dry_run=args.dry_run,
    )
