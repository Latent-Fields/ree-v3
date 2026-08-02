"""
V3-EXQ-108b -- MECH-135/INV-088 z_world evaluator-degeneracy DISAMBIGUATION

Claims: MECH-135, INV-088
Supersedes: none (companion diagnostic to V3-EXQ-108a, not a re-test of its own criteria)
DIAGNOSTIC. See EXPERIMENT_PURPOSE below.

WHY THIS RUN EXISTS (failure_autopsy_V3-EXQ-108a_2026-08-02, confirmed)
-------------------------------------------------------------------------
V3-EXQ-108a (StepHarness-confound-fixed re-run of V3-EXQ-108) found e1coe_score_var
~1.5e-14 -- across 40 genuinely different 30-step candidate action sequences from the
same starting state, E1's rollout-based goal_proximity scoring assigned functionally
identical scores. This replicated the original V3-EXQ-108's real full-production FAIL
almost exactly (c1=0.00219 vs 0.00229; c3 var ~1.5e-14 vs ~1.4e-14) even after fixing
108's documented training-time confound -- ruling that confound out.

The autopsy flagged this as potentially the starkest confirmation yet of INV-088's
("world_goal_evaluator_bounded_by_z_world_differentiation") coupling-leg prediction --
more direct than V3-EXQ-744a (real-but-weak) or V3-EXQ-819/819a (AUC 0.90 vs a ~48-57
ceiling on a SANCTIONED-trained z_world). But INV-088's own claims.yaml evidence_quality_
note records an EXPLICITLY UNRESOLVED confound: every cell in the 2026-07-18
zworld_near_static_characterisation diagnostic was UNTRAINED (0/61 latent_stack tensors
moved during the x734 P0 warmup), so "(a1) encoder untrained" and "(a2) world_dim=32
discriminative-granularity ceiling" are "perfectly confounded in this data." 108a's
z_world IS trained (via its own bespoke single-step E1 MSE loss) -- a new data point on
the confound-breaking axis -- but that bespoke loss is NOT the sanctioned
sd_zworld_warmup_optimizer_group (SD-070) training route V3-EXQ-819/819a already
confirmed adequately-trained, so 108a cannot by itself rule out "(a1) this particular
bespoke loss is inadequate" versus "(a2) genuine dimension ceiling."

Pre-registered hypotheses (hypothesis_space_registry.v1.json qid
inv088_evaluator_degeneracy_cause):
  H-undertrained-instrument (a1, axis=representation): a properly-trained z_world (the
    sanctioned sd_zworld_warmup_optimizer_group route) would show materially higher
    e1coe_score_var, clearing 108a's own C3 threshold (>=0.002).
  H-dimension-ceiling (a2, axis=measurement): even with the sanctioned-trained z_world,
    e1coe_score_var stays near-degenerate -- world_dim=32 structurally cannot support
    enough discriminative capacity.

THIS SCRIPT'S DESIGN -- resolves BOTH probes in one run, plus a third possibility
-----------------------------------------------------------------------------------
Phase 0 is restructured into two sub-phases:
  Phase 0a (NEW): the SANCTIONED SD-070 z_world encoder warmup, via
    experiments/_lib/zworld_p0_warmup.run_zworld_p0 -- the exact route
    V3-EXQ-819/819a validated as training split_encoder.world_encoder (confirmed
    3/3 seeds, weight-delta > 0). Runs on a DEDICATED warmup env (random policy,
    agent not driven) so it never touches residue/goal/clock state, per that
    function's own docstring.
  Phase 0b (UNCHANGED from V3-EXQ-108a): the existing bespoke E1/E2 single-step-MSE
    training loop, byte-identical code, now operating on TOP of the now-prediction-
    trained encoder (sense() is called under torch.no_grad() throughout Phase 0b, so
    the encoder is never further disturbed by it -- exactly as in 108a).

Phases 1-6 are UNCHANGED from V3-EXQ-108a (goal template, warmup state, candidate
generation, FROZEN/E1_COE rollout scoring, real-env execution) -- except Phase 4's
E1_COE scoring is extended to ALSO capture each candidate's final z_world ENDPOINT
vector (not just its scalar goal_proximity score), and a NEW Phase 4b samples 40
REAL, diverse z_world observations via independent random-policy rollouts from
reset -- the antecedent/static-differentiation readout, mirroring
zworld_near_static_characterisation_2026-07-18's own real-state sampling
methodology (diverse real states under a policy, as opposed to E1-imagined rollout
endpoints).

CONTRAST RATIO (offset-invariant, per zworld_near_static_characterisation_2026-07-18
sec 2): CR = spread / ||centroid||, where centroid = mean over the population and
spread = RMS deviation from centroid. Computed on TWO populations per seed:
  CR_real    -- the 40 REAL z_world observations (Phase 4b). The antecedent check:
                does the SANCTIONED-trained encoder differentiate real states at all?
  CR_rollout -- the 40 E1_COE candidate sequences' predicted z_world endpoints
                (Phase 4). The coupling-leg check: does E1's forward-rollout DYNAMICS
                preserve that differentiation across 30 imagined steps?

Both are computed WITHIN THE SAME RUN, SAME SEED, SAME 40 action sequences generating
the rollout population (CR_rollout) and an independently-sampled but identically-sized
population of real trajectories (CR_real) -- a self-contained comparison that needs no
cross-run or cross-family threshold borrowing from the 2026-07-18 characterisation
(a different env family, x734's rungs). Only the STATISTIC (CR = spread/||centroid||)
is reused from that methodology, not its absolute numbers.

DECISION TREE (both P0 preconditions must pass first -- see READINESS below)
  1. CR_real < CR_REAL_FLOOR (1e-4)
       -> "dimension_ceiling_confirmed": even a genuinely-trained encoder shows
          near-zero real-state contrast ratio. H-dimension-ceiling CONFIRMED;
          H-undertrained-instrument ELIMINATED (encoder training succeeded per the
          P_ENCODER_TRAINED precondition, yet real differentiation is still ~0).
  2. elif C3 (e1coe_score_var >= 0.002, 108a's own threshold) PASSES
       -> "undertrained_instrument_confirmed": with the SAME sanctioned-trained
          encoder, scoring now discriminates where 108a's bespoke-trained one could
          not. H-undertrained-instrument CONFIRMED; H-dimension-ceiling ELIMINATED.
  3. elif (CR_rollout / CR_real) < CR_ROLLOUT_COLLAPSE_RATIO (0.1)
       -> "downstream_dynamics_collapse": the encoder differentiates real states
          fine (CR_real clears the floor) but E1's ROLLOUT DYNAMICS collapse
          distinct imagined futures toward near-identical predictions. NEITHER
          pre-registered hypothesis -- a third, new finding pointing at E1's own
          forward-model training/dynamics as the bottleneck, independent of the
          STATIC representation's capacity.
  4. else
       -> "unresolved_partial_signal": real and rollout differentiation are both
          roughly healthy, yet C3 still fails -- points to a projection/measurement-
          level issue (goal_proximity's own sensitivity to z_world variation) rather
          than a representation or dynamics collapse. Reported honestly as
          inconclusive on the a1/a2 question; flagged for a follow-up autopsy.

READINESS (P0, both existential, gate below-floor to substrate_not_ready_requeue)
  P_ENCODER_TRAINED: at least one split_encoder.world_encoder tensor moved during
    Phase 0a (zworld_encoder_guard's own load-bearing bit). If this fails, the
    SD-070 warmup itself did not engage this run (unexpected given V3-EXQ-819/819a
    already validated the route, but checked for auditability) -- self-route
    substrate_not_ready_requeue, never a1/a2 verdict.
  P_REAL_ZWORLD_NONDEGENERATE: CR_real computed on a genuinely non-degenerate
    real-state sample (CR_real is a finite number, i.e. centroid_norm > 1e-12 and
    the sample is not literally empty/constant). If this fails, sensing itself is
    broken this run (an environment/config issue, not a ceiling finding) --
    self-route substrate_not_ready_requeue.

DECLARED NULL. A FAIL here (any label) does not reopen MECH-135's original C1/C2/C3
result -- 108a's weakens stands regardless of which explanation this run confirms.
This run's job is explaining WHY, for governance and the INV-088 thread, not
re-litigating whether.

Re-derive brake (Step 2.5b, this session): 0 autopsies count a substrate_ceiling hit
against MECH-135 or INV-088 at this granularity (checked via the standard grep-count
method) -- brake does not fire. This is also explicitly NOT a re-derive-braked lettered
retest of 108a's own criteria: it is a diagnostic on a DIFFERENT substrate axis
(SD-070 encoder training) than 108a tested, addressing a confound 108a itself could
not resolve.

SLEEP DRIVER: not applicable -- no sleep phase entered in this run.

Z_GOAL: real (GoalState with a genuine collected template, unchanged from 108a) --
recorded via z_goal_stream_stats at manifest-write time.
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
from ree_core.goal import GoalConfig, GoalState
from ree_core.utils.config import REEConfig
from experiment_protocol import emit_outcome
from experiments.pack_writer import write_flat_manifest
from experiments._metrics import check_degeneracy
from experiments._lib.zworld_p0_warmup import run_zworld_p0
from experiments._lib.capability_eval import RandomPolicy
from experiments._lib.zworld_encoder_guard import (
    latent_stack_snapshot,
    assert_world_encoder_trained,
)


EXPERIMENT_TYPE = "v3_exq_108b_mech135_inv088_zworld_disambiguation"
CLAIM_IDS = ["MECH-135", "INV-088"]
EXPERIMENT_PURPOSE = "diagnostic"
SUPERSEDES = None

# agent_construction_before_seed lint: run_seed() builds the agent (_build_agent,
# inside Phase 0a's _run_zworld_p0_warmup call chain) before any torch.manual_seed
# call in this function's own flow. Inherited from V3-EXQ-108a's identical
# exemption (row 9, agent_seed_order_lint_backlog_triage.md): FROZEN and E1_COE are
# both scored off the LITERAL SAME agent object within a seed (never two
# independently-constructed agents), so unseeded weight init cannot confound the
# within-seed arm comparison this script tests. This script's Phase 0a/0b/1-6 use
# the same single-agent-per-seed shape as 108a, unchanged.
AGENT_SEED_ORDER_EXEMPT = (
    "FROZEN/E1_COE scored off the literal same agent object within a seed "
    "(inherited from V3-EXQ-108a's identical exemption, itself inherited from "
    "V3-EXQ-108 row 9 triage, agent_seed_order_lint_backlog_triage.md)"
)

# ---------------------------------------------------------------------------
# Pre-registered thresholds
# ---------------------------------------------------------------------------
C1_PROX_DELTA_THRESHOLD = 0.05     # unchanged from V3-EXQ-108/108a
C2_SEEDS_NEEDED = 1                # unchanged from V3-EXQ-108/108a
C3_VAR_THRESHOLD = 0.002           # unchanged from V3-EXQ-108/108a
ZWORLD_P0_EPISODES = 60            # SD-070 encoder warmup -- matches V3-EXQ-819/819a's
                                    # ZWORLD_P0_EPISODES (competence_floor_zworld.py),
                                    # the sanctioned sd_zworld_warmup_optimizer_group route
CR_REAL_FLOOR = 1e-4               # below this, real-state differentiation itself is ~0
CR_ROLLOUT_COLLAPSE_RATIO = 0.1    # CR_rollout / CR_real below this = dynamics collapse
N_REAL_SAMPLES = 40                # matches n_sequences, for a size-matched CR comparison
STEPS_PER_REAL_SAMPLE = 30         # matches rollout_horizon, for a horizon-matched comparison


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


def _env_kwargs() -> Dict[str, Any]:
    """Env config, unchanged from V3-EXQ-108a."""
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
# Phase 0a: SD-070 sanctioned z_world encoder warmup (NEW)
# ---------------------------------------------------------------------------

def _run_zworld_p0_warmup(
    agent: REEAgent, seed: int, zworld_p0_episodes: int, steps_per_episode: int,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """SD-070 P0a encoder warmup on a DEDICATED env (random policy, agent not driven),
    per run_zworld_p0's own docstring requirement. Returns the combined p0a training
    report + weight-delta readiness audit for the manifest."""
    before = latent_stack_snapshot(agent)
    warmup_env = CausalGridWorldV2(seed=seed, **_env_kwargs())
    p0a_report = run_zworld_p0(
        agent, warmup_env, seed, zworld_p0_episodes, steps_per_episode,
        policy=RandomPolicy(seed), label="v3_exq_108b P0a (SD-070 z_world encoder)",
        dry_run=dry_run,
    )
    encoder_report = assert_world_encoder_trained(
        agent, before, p0=zworld_p0_episodes, strict=False,
        context="v3_exq_108b_mech135_inv088_zworld_disambiguation",
        escape_hint="pass zworld_p0_episodes=0 for a deliberate frozen-encoder run",
    )
    return {**p0a_report, **encoder_report}


# ---------------------------------------------------------------------------
# Phase 0b: bespoke E1/E2 single-step training (UNCHANGED from V3-EXQ-108a)
# ---------------------------------------------------------------------------

def _train_agent(
    agent: REEAgent,
    env: CausalGridWorldV2,
    seed: int,
    n_episodes: int,
    steps_per_episode: int,
) -> None:
    """Train agent with random policy (E1 + E2 only). Byte-identical to
    V3-EXQ-108a's _train_agent -- calls agent.sense() exactly ONCE per env step
    (StepHarness invariant #1). sense() runs under torch.no_grad() throughout, so
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
                f"  [Train] ep {ep+1}/{n_episodes} "
                f"e1_loss={ep_loss_e1/max(n_steps,1):.5f} "
                f"e2_loss={ep_loss_e2/max(n_steps,1):.5f}",
                flush=True,
            )

    agent.eval()
    print(f"  [Train] Done. {n_episodes} episodes.", flush=True)


# ---------------------------------------------------------------------------
# Phase 1: goal template (unchanged from V3-EXQ-108a)
# ---------------------------------------------------------------------------

def _collect_goal_template(
    agent: REEAgent, env: CausalGridWorldV2, seed: int, max_steps: int,
) -> Tuple[torch.Tensor, str]:
    torch.manual_seed(seed)
    random.seed(seed)
    _, obs_dict = env.reset()
    agent.reset()

    for _ in range(max_steps):
        obs_body = obs_dict["body_state"]
        obs_world = obs_dict["world_state"]
        with torch.no_grad():
            latent = agent.sense(obs_body, obs_world)
        agent.clock.advance()
        action_idx = random.randint(0, env.action_dim - 1)
        action = _action_to_onehot(action_idx, env.action_dim, agent.device)
        _, _, done, info, obs_dict = env.step(action)
        if info.get("transition_type", "none") == "resource":
            print(
                f"  [Phase1] Resource contact, z_world_norm={latent.z_world.norm().item():.3f}",
                flush=True,
            )
            return latent.z_world.detach(), "resource_contact"
        if done:
            _, obs_dict = env.reset()
            agent.reset()

    print("  [Phase1] WARNING: no resource contact -- using fallback unit vector", flush=True)
    z_goal = torch.randn(1, agent.config.latent.world_dim)
    z_goal = F.normalize(z_goal, dim=-1)
    return z_goal, "fallback_unit_vector"


# ---------------------------------------------------------------------------
# Phase 2: warmup state (unchanged from V3-EXQ-108a)
# ---------------------------------------------------------------------------

def _get_warmup_state(
    agent: REEAgent, env: CausalGridWorldV2, seed: int, n_warmup_steps: int,
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    torch.manual_seed(seed + 1000)
    random.seed(seed + 1000)
    _, obs_dict = env.reset()
    agent.reset()
    latent = None
    warmup_actions: List[int] = []

    for _ in range(n_warmup_steps):
        obs_body = obs_dict["body_state"]
        obs_world = obs_dict["world_state"]
        with torch.no_grad():
            latent = agent.sense(obs_body, obs_world)
        agent.clock.advance()
        action_idx = random.randint(0, env.action_dim - 1)
        warmup_actions.append(action_idx)
        action = _action_to_onehot(action_idx, env.action_dim, agent.device)
        _, _, done, _, obs_dict = env.step(action)
        if done:
            _, obs_dict = env.reset()
            agent.reset()
            latent = None
            warmup_actions = []

    if latent is None:
        obs_body = obs_dict["body_state"]
        obs_world = obs_dict["world_state"]
        with torch.no_grad():
            latent = agent.sense(obs_body, obs_world)

    return latent.z_self.detach(), latent.z_world.detach(), warmup_actions


# ---------------------------------------------------------------------------
# Phase 3: generate candidate sequences (unchanged from V3-EXQ-108a)
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
# Phase 4: score sequences (E1_COE extended to also return the z_world endpoint)
# ---------------------------------------------------------------------------

def _score_sequence_frozen(z_world_start: torch.Tensor, goal_state: GoalState) -> float:
    return float(goal_state.goal_proximity(z_world_start).item())


def _score_sequence_e1coe_with_endpoint(
    agent: REEAgent,
    z_self_start: torch.Tensor,
    z_world_start: torch.Tensor,
    action_sequence: List[int],
    goal_state: GoalState,
    self_dim: int,
) -> Tuple[float, torch.Tensor]:
    """Same rollout logic as V3-EXQ-108a's _score_sequence_e1coe, but additionally
    returns the final z_world_curr endpoint tensor (needed for the CR_rollout
    contrast-ratio diagnostic -- the only change from 108a's original)."""
    device = agent.device
    n_actions = agent.config.e2.action_dim

    agent.e1.reset_hidden_state()

    z_self_curr = z_self_start.clone()
    z_world_curr = z_world_start.clone()

    for a_idx in action_sequence:
        action = _action_to_onehot(a_idx, n_actions, device)
        total_curr = torch.cat([z_self_curr, z_world_curr], dim=-1)
        with torch.no_grad():
            e1_preds, _ = agent.e1(total_curr, horizon=1)
        z_world_next = e1_preds[0, 0, self_dim:].unsqueeze(0)
        with torch.no_grad():
            z_self_next = agent.e2.predict_next_self(z_self_curr, action)
        z_self_curr = z_self_next
        z_world_curr = z_world_next

    score = float(goal_state.goal_proximity(z_world_curr).item())
    return score, z_world_curr.detach()


# ---------------------------------------------------------------------------
# Phase 4b: real z_world sample (NEW -- the antecedent/static-differentiation readout)
# ---------------------------------------------------------------------------

def _collect_real_zworld_sample(
    agent: REEAgent, env: CausalGridWorldV2, seed: int, n_samples: int, steps_per_sample: int,
) -> List[torch.Tensor]:
    """Sample n_samples REAL, diverse z_world observations via independent
    random-policy rollouts from reset -- mirrors zworld_near_static_characterisation_
    2026-07-18's own real-state sampling methodology (diverse real states under a
    policy), as the antecedent counterpart to the E1-imagined rollout endpoints.
    Uses its own seed offset (+3000) so it does not disturb the deterministic
    warmup_actions replay Phase 6 depends on -- every phase in this script manages
    its own RNG state explicitly at entry, so ordering is safe."""
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
    """CR = spread / ||centroid||, per zworld_near_static_characterisation_2026-07-18
    sec 2's offset-invariant statistic. spread = RMS deviation from centroid."""
    stacked = torch.cat(vectors, dim=0)  # [N, dim]
    centroid = stacked.mean(dim=0, keepdim=True)  # [1, dim]
    centroid_norm = float(centroid.norm().item())
    deviations = stacked - centroid
    spread = float(torch.sqrt((deviations.pow(2).sum(dim=-1)).mean()).item())
    cr = (spread / centroid_norm) if centroid_norm > 1e-12 else float("nan")
    return {"spread": spread, "centroid_norm": centroid_norm, "contrast_ratio": cr}


# ---------------------------------------------------------------------------
# Phase 5: execute best sequence in real env (unchanged from V3-EXQ-108a)
# ---------------------------------------------------------------------------

def _execute_in_real_env(
    agent: REEAgent,
    env: CausalGridWorldV2,
    seed: int,
    warmup_actions: List[int],
    action_sequence: List[int],
    goal_state: GoalState,
) -> Tuple[float, bool]:
    device = agent.device

    torch.manual_seed(seed + 1000)
    random.seed(seed + 1000)
    _, obs_dict = env.reset()
    agent.reset()
    latent = None

    for a_idx in warmup_actions:
        obs_body = obs_dict["body_state"]
        obs_world = obs_dict["world_state"]
        with torch.no_grad():
            latent = agent.sense(obs_body, obs_world)
        agent.clock.advance()
        action = _action_to_onehot(a_idx, env.action_dim, device)
        _, _, done, _, obs_dict = env.step(action)
        if done:
            break

    resource_contacted = False
    for a_idx in action_sequence:
        obs_body = obs_dict["body_state"]
        obs_world = obs_dict["world_state"]
        with torch.no_grad():
            latent = agent.sense(obs_body, obs_world)
        agent.clock.advance()
        action = _action_to_onehot(a_idx, env.action_dim, device)
        _, _, done, info, obs_dict = env.step(action)
        if info.get("transition_type", "none") == "resource":
            resource_contacted = True
        if done:
            break

    obs_body = obs_dict["body_state"]
    obs_world = obs_dict["world_state"]
    with torch.no_grad():
        latent_final = agent.sense(obs_body, obs_world)

    final_prox = float(goal_state.goal_proximity(latent_final.z_world).item())
    return final_prox, resource_contacted


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
    n_warmup_steps: int,
    goal_max_steps: int,
    zworld_p0_episodes: int,
    n_real_samples: int,
    steps_per_real_sample: int,
    c1_threshold: float,
    c3_var_threshold: float,
    dry_run: bool = False,
) -> Dict[str, Any]:
    print(f"\n[EXQ-108b] seed={seed}", flush=True)
    print(f"Seed {seed} Condition single", flush=True)

    agent, env = _build_agent(seed, world_dim, self_dim)

    # Phase 0a: SD-070 sanctioned encoder warmup (NEW)
    print(f"[EXQ-108b] Phase 0a: SD-070 z_world encoder warmup ({zworld_p0_episodes} eps)...", flush=True)
    readiness_report = _run_zworld_p0_warmup(
        agent, seed, zworld_p0_episodes, steps_per_episode, dry_run=dry_run,
    )
    print(
        f"  encoder_trained={readiness_report.get('zworld_encoder_trained')} "
        f"max_abs_delta={readiness_report.get('world_encoder_max_abs_delta'):.6f}",
        flush=True,
    )

    # Phase 0b: bespoke E1/E2 single-step training (unchanged from 108a)
    print(f"[EXQ-108b] Phase 0b: training E1/E2 ({n_train_episodes} eps)...", flush=True)
    _train_agent(agent, env, seed, n_train_episodes, steps_per_episode)

    # Phase 1: goal template
    print("[EXQ-108b] Phase 1: goal template...", flush=True)
    z_goal_tensor, goal_template_source = _collect_goal_template(agent, env, seed, goal_max_steps)
    goal_config = GoalConfig(goal_dim=world_dim, z_goal_enabled=True, goal_weight=1.0)
    goal_state = GoalState(goal_config, agent.device)
    goal_state._z_goal = z_goal_tensor.to(agent.device)
    print(f"  z_goal_norm={goal_state.goal_norm():.4f} source={goal_template_source}", flush=True)

    # Phase 2: warmup state
    print("[EXQ-108b] Phase 2: warmup state...", flush=True)
    z_self_0, z_world_0, warmup_actions = _get_warmup_state(agent, env, seed, n_warmup_steps)
    base_prox = float(goal_state.goal_proximity(z_world_0).item())
    print(f"  base_prox={base_prox:.4f}", flush=True)

    # Phase 3: candidate sequences
    print(f"[EXQ-108b] Phase 3: generating {n_sequences} candidate sequences...", flush=True)
    seqs = _generate_candidate_sequences(n_sequences, rollout_horizon, env.action_dim, seed)

    # Phase 4: score under FROZEN and E1_COE (E1_COE now also captures endpoints)
    print("[EXQ-108b] Phase 4: scoring sequences...", flush=True)
    frozen_scores: List[float] = []
    e1coe_scores: List[float] = []
    e1coe_endpoints: List[torch.Tensor] = []

    for i, seq in enumerate(seqs):
        frozen_scores.append(_score_sequence_frozen(z_world_0, goal_state))
        score, endpoint = _score_sequence_e1coe_with_endpoint(
            agent, z_self_0, z_world_0, seq, goal_state, self_dim
        )
        e1coe_scores.append(score)
        e1coe_endpoints.append(endpoint)

        if (i + 1) % 10 == 0:
            print(f"  scored {i+1}/{n_sequences}", flush=True)

    e1coe_scores_t = torch.tensor(e1coe_scores)
    e1coe_score_var = float(e1coe_scores_t.var().item()) if len(e1coe_scores) > 1 else 0.0
    e1coe_score_min = float(e1coe_scores_t.min().item())
    e1coe_score_max = float(e1coe_scores_t.max().item())
    e1coe_score_mean = float(e1coe_scores_t.mean().item())

    print(
        f"  E1_COE scores: min={e1coe_score_min:.4f} max={e1coe_score_max:.4f} "
        f"mean={e1coe_score_mean:.4f} var={e1coe_score_var:.6f}",
        flush=True,
    )
    print(f"  FROZEN scores: all={frozen_scores[0]:.4f} (constant)", flush=True)

    # Phase 4b: real z_world sample + contrast ratios (NEW)
    print(f"[EXQ-108b] Phase 4b: sampling {n_real_samples} real z_world observations...", flush=True)
    real_samples = _collect_real_zworld_sample(
        agent, env, seed, n_real_samples, steps_per_real_sample,
    )
    cr_real = _contrast_ratio(real_samples)
    cr_rollout = _contrast_ratio(e1coe_endpoints)
    cr_ratio = (
        (cr_rollout["contrast_ratio"] / cr_real["contrast_ratio"])
        if cr_real["contrast_ratio"] and cr_real["contrast_ratio"] == cr_real["contrast_ratio"]
        and cr_real["contrast_ratio"] > 0
        else float("nan")
    )
    print(
        f"  CR_real={cr_real['contrast_ratio']:.6f} CR_rollout={cr_rollout['contrast_ratio']:.6f} "
        f"ratio={cr_ratio:.6f}",
        flush=True,
    )

    # Phase 5: select best sequences
    frozen_best_idx = int(torch.tensor(frozen_scores).argmax().item())
    e1coe_best_idx = int(e1coe_scores_t.argmax().item())
    print(
        f"  FROZEN best_idx={frozen_best_idx} (random) "
        f"E1_COE best_idx={e1coe_best_idx} score={e1coe_scores[e1coe_best_idx]:.4f}",
        flush=True,
    )

    # Phase 6: execute best sequences in real env
    print("[EXQ-108b] Phase 6: real-env execution...", flush=True)
    agent.eval()
    real_prox_frozen, resource_frozen = _execute_in_real_env(
        agent, env, seed, warmup_actions, seqs[frozen_best_idx], goal_state
    )
    real_prox_e1coe, resource_e1coe = _execute_in_real_env(
        agent, env, seed, warmup_actions, seqs[e1coe_best_idx], goal_state
    )

    prox_delta = real_prox_e1coe - real_prox_frozen
    print(
        f"  FROZEN: real_prox={real_prox_frozen:.4f} contact={resource_frozen}",
        flush=True,
    )
    print(
        f"  E1_COE: real_prox={real_prox_e1coe:.4f} contact={resource_e1coe} "
        f"delta={prox_delta:+.4f}",
        flush=True,
    )
    c3_pass_this_seed = e1coe_score_var >= c3_var_threshold
    verdict = "PASS" if prox_delta >= c1_threshold and c3_pass_this_seed else "FAIL"
    print(f"verdict: {verdict}", flush=True)

    return {
        "seed": seed,
        "readiness": readiness_report,
        "goal_template_source": goal_template_source,
        "z_goal_norm": goal_state.goal_norm(),
        "base_prox": base_prox,
        "e1coe_score_min": e1coe_score_min,
        "e1coe_score_max": e1coe_score_max,
        "e1coe_score_mean": e1coe_score_mean,
        "e1coe_score_var": e1coe_score_var,
        "e1coe_best_score": float(e1coe_scores[e1coe_best_idx]),
        "frozen_base_score": frozen_scores[0],
        "real_prox_frozen": real_prox_frozen,
        "real_prox_e1coe": real_prox_e1coe,
        "prox_delta": prox_delta,
        "resource_frozen": resource_frozen,
        "resource_e1coe": resource_e1coe,
        "cr_real": cr_real,
        "cr_rollout": cr_rollout,
        "cr_ratio": cr_ratio,
        "c3_pass_this_seed": c3_pass_this_seed,
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
    n_warmup_steps: int,
    goal_max_steps: int,
    zworld_p0_episodes: int,
    n_real_samples: int,
    steps_per_real_sample: int,
    c1_threshold: float,
    c3_var_threshold: float,
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
            n_warmup_steps=n_warmup_steps,
            goal_max_steps=goal_max_steps,
            zworld_p0_episodes=zworld_p0_episodes,
            n_real_samples=n_real_samples,
            steps_per_real_sample=steps_per_real_sample,
            c1_threshold=c1_threshold,
            c3_var_threshold=c3_var_threshold,
            dry_run=dry_run,
        )
        seed_results.append(r)

    # ---- Readiness (P0) ----
    encoder_trained_per_seed = [bool(r["readiness"].get("zworld_encoder_trained")) for r in seed_results]
    p_encoder_trained_met = all(encoder_trained_per_seed)
    cr_real_per_seed = [r["cr_real"]["contrast_ratio"] for r in seed_results]
    p_real_nondegenerate_met = all(
        (v == v) and v > 0 for v in cr_real_per_seed  # v==v excludes NaN
    )

    preconditions = [
        {
            "name": "encoder_trained",
            "kind": "readiness",
            "description": (
                "At least one split_encoder.world_encoder tensor moved during the "
                "Phase 0a SD-070 warmup, per every seed (zworld_encoder_guard's "
                "load-bearing bit)."
            ),
            "measured": min(
                r["readiness"].get("world_encoder_max_abs_delta", 0.0) for r in seed_results
            ),
            "threshold": 0.0,
            "direction": "lower",
            "met": p_encoder_trained_met,
        },
        {
            "name": "real_zworld_nondegenerate",
            "kind": "readiness",
            "description": (
                "CR_real (the antecedent contrast ratio on 40 real, diverse z_world "
                "observations) is a finite, positive number for every seed -- "
                "confirms sensing itself is not degenerate before drawing any a1/a2 "
                "conclusion from it."
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
    else:
        degeneracy_reason = None
        # ---- Decision tree ----
        cr_real_vals = [r["cr_real"]["contrast_ratio"] for r in seed_results]
        cr_ratio_vals = [r["cr_ratio"] for r in seed_results]
        c3_per_seed = [r["c3_pass_this_seed"] for r in seed_results]
        c3_pass = all(c3_per_seed)
        min_cr_real = min(cr_real_vals)
        min_cr_ratio = min(v for v in cr_ratio_vals if v == v) if any(v == v for v in cr_ratio_vals) else float("nan")

        if min_cr_real < CR_REAL_FLOOR:
            label = "dimension_ceiling_confirmed"
        elif c3_pass:
            label = "undertrained_instrument_confirmed"
        elif (min_cr_ratio == min_cr_ratio) and (min_cr_ratio < CR_ROLLOUT_COLLAPSE_RATIO):
            label = "downstream_dynamics_collapse"
        else:
            label = "unresolved_partial_signal"

        prox_deltas = [r["prox_delta"] for r in seed_results]
        c1_val = sum(prox_deltas) / len(prox_deltas)
        c1_pass = c1_val >= c1_threshold
        c2_per_seed = [int(r["resource_e1coe"]) >= int(r["resource_frozen"]) for r in seed_results]
        c2_pass = sum(c2_per_seed) >= C2_SEEDS_NEEDED

        status = "PASS" if (label == "undertrained_instrument_confirmed" and c1_pass and c2_pass) else "FAIL"

        if label == "undertrained_instrument_confirmed" and status == "PASS":
            evidence_direction = "supports"
            evidence_direction_per_claim = {"MECH-135": "supports", "INV-088": "weakens"}
        elif label == "dimension_ceiling_confirmed":
            evidence_direction = "non_contributory"
            evidence_direction_per_claim = {"MECH-135": "non_contributory", "INV-088": "supports"}
        elif label == "downstream_dynamics_collapse":
            evidence_direction = "non_contributory"
            evidence_direction_per_claim = {"MECH-135": "non_contributory", "INV-088": "non_contributory"}
        else:
            evidence_direction = "non_contributory"
            evidence_direction_per_claim = {"MECH-135": "non_contributory", "INV-088": "non_contributory"}

    print(f"\n[EXQ-108b] Label: {label}", flush=True)
    print(f"[EXQ-108b] Status: {status}", flush=True)

    criteria = [
        {
            "name": "C3_E1COE_SCORE_DISCRIMINATES",
            "load_bearing": True,
            "passed": bool(non_degenerate and all(r["c3_pass_this_seed"] for r in seed_results)),
            "measured": min(r["e1coe_score_var"] for r in seed_results),
            "threshold": c3_var_threshold,
            "statement": (
                "e1coe_score_var >= C3_VAR_THRESHOLD in every seed -- the same "
                "non-degeneracy criterion V3-EXQ-108/108a used, now on a "
                "sanctioned-trained z_world."
            ),
        },
    ]
    criteria_non_degenerate = {"C3_E1COE_SCORE_DISCRIMINATES": non_degenerate}

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
        "n_warmup_steps": n_warmup_steps,
        "zworld_p0_episodes": zworld_p0_episodes,
        "n_real_samples": n_real_samples,
        "steps_per_real_sample": steps_per_real_sample,
        "registered_c1_threshold": c1_threshold,
        "registered_c3_var_threshold": c3_var_threshold,
        "registered_c2_seeds_needed": C2_SEEDS_NEEDED,
        "registered_cr_real_floor": CR_REAL_FLOOR,
        "registered_cr_rollout_collapse_ratio": CR_ROLLOUT_COLLAPSE_RATIO,
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
        "source_autopsy": "failure_autopsy_V3-EXQ-108a_2026-08-02",
        "hypothesis_space_qid": "inv088_evaluator_degeneracy_cause",
    }

    # Flatten per-seed metrics
    for r in seed_results:
        s = r["seed"]
        for k, v in r.items():
            if k != "seed":
                result[f"seed_{s}_{k}"] = v

    return result


if __name__ == "__main__":
    import argparse
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser(
        description="V3-EXQ-108b: MECH-135/INV-088 z_world evaluator-degeneracy disambiguation"
    )
    parser.add_argument("--seeds", type=str, default="42,123")
    parser.add_argument("--world-dim", type=int, default=32)
    parser.add_argument("--self-dim", type=int, default=32)
    parser.add_argument("--train-episodes", type=int, default=100)
    parser.add_argument("--steps-per-episode", type=int, default=200)
    parser.add_argument("--rollout-horizon", type=int, default=30)
    parser.add_argument("--n-sequences", type=int, default=40)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--goal-max-steps", type=int, default=2000)
    parser.add_argument("--zworld-p0-episodes", type=int, default=ZWORLD_P0_EPISODES)
    parser.add_argument("--n-real-samples", type=int, default=N_REAL_SAMPLES)
    parser.add_argument("--steps-per-real-sample", type=int, default=STEPS_PER_REAL_SAMPLE)
    parser.add_argument("--c1-threshold", type=float, default=C1_PROX_DELTA_THRESHOLD)
    parser.add_argument("--c3-var-threshold", type=float, default=C3_VAR_THRESHOLD)
    parser.add_argument("--dry-run", "--smoke-test", dest="dry_run", action="store_true")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    if args.dry_run:
        n_train = 2
        steps_ep = 50
        n_sequences = 5
        horizon = 10
        warmup = 5
        goal_max = 300
        zworld_p0 = 3
        n_real = 5
        steps_real = 10
        c1_thresh = -99.0
        c3_thresh = 0.0
        seeds = seeds[:1]
        print("[V3-EXQ-108b] SMOKE TEST MODE", flush=True)
    else:
        n_train = args.train_episodes
        steps_ep = args.steps_per_episode
        n_sequences = args.n_sequences
        horizon = args.rollout_horizon
        warmup = args.warmup_steps
        goal_max = args.goal_max_steps
        zworld_p0 = args.zworld_p0_episodes
        n_real = args.n_real_samples
        steps_real = args.steps_per_real_sample
        c1_thresh = args.c1_threshold
        c3_thresh = args.c3_var_threshold

    t0 = time.perf_counter()
    result = run(
        seeds=seeds,
        world_dim=args.world_dim,
        self_dim=args.self_dim,
        n_train_episodes=n_train,
        steps_per_episode=steps_ep,
        n_sequences=n_sequences,
        rollout_horizon=horizon,
        n_warmup_steps=warmup,
        goal_max_steps=goal_max,
        zworld_p0_episodes=zworld_p0,
        n_real_samples=n_real,
        steps_per_real_sample=steps_real,
        c1_threshold=c1_thresh,
        c3_var_threshold=c3_thresh,
        dry_run=args.dry_run,
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["timestamp_utc"] = ts
    result["run_timestamp"] = ts
    result["run_id"] = f"{EXPERIMENT_TYPE}_{ts}_v3"
    result["architecture_epoch"] = "ree_hybrid_guardrails_v1"

    full_config = {
        "seeds": seeds,
        "world_dim": args.world_dim,
        "self_dim": args.self_dim,
        "n_train_episodes": n_train,
        "steps_per_episode": steps_ep,
        "n_sequences": n_sequences,
        "rollout_horizon": horizon,
        "n_warmup_steps": warmup,
        "goal_max_steps": goal_max,
        "zworld_p0_episodes": zworld_p0,
        "n_real_samples": n_real,
        "steps_per_real_sample": steps_real,
        "c1_threshold": c1_thresh,
        "c3_var_threshold": c3_thresh,
        "cr_real_floor": CR_REAL_FLOOR,
        "cr_rollout_collapse_ratio": CR_ROLLOUT_COLLAPSE_RATIO,
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
    )

    print(f"\nResult written to: {out_path}", flush=True)
    print(f"Status: {result['status']}", flush=True)
    print(f"Label: {result['interpretation']['label']}", flush=True)

    if args.dry_run:
        print("[V3-EXQ-108b] SMOKE TEST COMPLETE", flush=True)
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
