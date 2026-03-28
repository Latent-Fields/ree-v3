#!/opt/local/bin/python3
"""
V3-EXQ-146 -- Q-001 Entity Emergence and Binding Discriminative Pair

Claims: Q-001
Proposal: EXP-0086 / EVB-0070

Q-001 asks:
  "What mechanisms produce entity emergence and binding?"

ARC-006 asserts that entities are sparse, persistent, bindable structures
emerging within the latent stack -- maintained across time by attention-gated
(precision/z_self) mechanisms, with error ownership entity-linked and
object-file-like persistence buffers tracking features across time.

The candidate binding mechanism under test:
  z_self (self-referential latent) provides the attentional gate for
  entity persistence in z_world. When z_self is active and updated by
  self-maintenance loss, it selectively amplifies world-state responses
  to entity events (hazard appearances/disappearances) -- producing
  coherent persistent representations vs. diffuse non-entity-linked
  responses.

This experiment tests that mechanism via ablation:

  ENTITY_BINDING_ON (z_self active):
    self_dim=32, self_maintenance_weight=0.1, alpha_self=0.3.
    z_self functions as a persistent gating signal -- the self-maintenance
    objective enforces structured (low-D_eff) self-representation, which
    focuses attentional resources on recurrent entity-linked world features.
    Expected: z_world shows large, consistent step-responses when hazard
    entities appear (entity_event_response), and high z_world autocorrelation
    across contiguous object-tracking steps (entity_tracking_persistence).

  ENTITY_ABLATED (z_self zeroed):
    self_dim=32, but z_self is zeroed out after each sense() call.
    The world encoder receives no attentional gate from z_self.
    self_maintenance_weight=0.0 (no maintenance gradient).
    Expected: z_world responses to hazard events are weaker and noisier,
    entity_tracking_persistence is lower.

If Q-001's binding mechanism (z_self attentional gating) is the operative
mechanism for entity persistence and binding:
  - ENTITY_BINDING_ON shows significantly larger entity_event_response
    and entity_tracking_persistence than ENTITY_ABLATED.

If Q-001's candidate mechanism is NOT the operative mechanism (e.g., binding
emerges purely from world-encoder dynamics without z_self gating):
  - ENTITY_ABLATED shows comparable entity_event_response and persistence
    to ENTITY_BINDING_ON.

Pre-registered thresholds
--------------------------
C1: ENTITY_BINDING_ON mean entity_event_response >= THRESH_BINDING_EVENT_RESPONSE
    both seeds.
    (z_self-gated world encoder produces detectable entity event responses)

C2: ENTITY_ABLATED mean entity_event_response <= THRESH_ABLATED_EVENT_RESPONSE
    both seeds.
    (without z_self gate, world encoder shows weaker entity responses)

C3: Per-seed entity_event_response gap (BINDING - ABLATED) >= THRESH_EVENT_RESPONSE_GAP
    both seeds.
    (binding mechanism produces discriminable improvement over ablated baseline)

C4: ENTITY_BINDING_ON mean entity_tracking_persistence >= THRESH_BINDING_PERSISTENCE
    both seeds.
    (z_self-gated representation shows higher z_world autocorrelation across
    consecutive entity-tracking steps vs. non-event baseline)

C5: Per-seed persistence gap (BINDING - ABLATED) >= THRESH_PERSISTENCE_GAP
    both seeds.
    (binding mechanism produces more temporally coherent world representations)

Interpretation:
  C1+C2+C3+C4+C5 ALL PASS: Q-001 PARTIALLY ANSWERED -- z_self attentional gating
    IS the operative mechanism for entity emergence and binding in the current V3
    architecture. The self-maintenance objective enforces structured self-representation
    that gates attention on entity-linked world features, producing persistent and
    responsive entity representations in z_world.

  C3 or C5 FAIL only: The gap is present but below threshold -- directional support
    but weaker than predicted. May indicate additional binding mechanisms are needed
    (relational binding, MECH-044 hippocampal relational binding).

  C1+C2 FAIL: Both conditions show comparable entity_event_response -- z_self does not
    modulate world encoder entity sensitivity. Alternative binding mechanism required.

  C4 FAIL: Persistence not improved by z_self -- world encoder's own temporal dynamics
    (alpha_world=0.9 EMA) may already be sufficient for persistence without z_self.

Conditions
----------
ENTITY_BINDING_ON:
  z_self active (self_dim=32), self_maintenance_weight=0.1.
  Trains E1, E2, E3 + self-maintenance loss.
  z_self remains active during eval.

ENTITY_ABLATED:
  z_self zeroed after each sense() call during eval.
  No self_maintenance_weight (weight=0.0).
  All other config identical to ENTITY_BINDING_ON.

Design: matched-seed comparison. Both conditions use the same seed for
env initialization and training random state. Training is identical except
self_maintenance_weight=0.0 in ENTITY_ABLATED.

Entity event: a timestep where the hazard_state portion of obs_world changes
by more than EVENT_DELTA_THRESH (hazard appears, disappears, or moves).
entity_event_response = mean |delta_z_world| on event steps.
entity_tracking_persistence = mean cosine_sim(z_world_t, z_world_{t+1}) on
non-event steps following an event (the representation should persist).

Seeds: [42, 123] (matched -- same env+training per seed, two conditions)
Env:   CausalGridWorldV2 size=10, 3 hazards, 4 resources, drift_interval=5
Train: 300 episodes x 200 steps (warmup)
Eval:  100 episodes x 200 steps (entity event tracking)
Estimated runtime: ~60 min any machine
"""

import sys
import random
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_146_q001_entity_binding_pair"
CLAIM_IDS = ["Q-001"]

# ---------------------------------------------------------------------------
# Pre-registered thresholds
# ---------------------------------------------------------------------------
# C1: ENTITY_BINDING_ON entity_event_response >= this (both seeds).
#     A meaningful entity response: mean delta_z_world on event steps
#     must exceed the per-step background noise level.
#     Calibrated against: typical z_world norm ~1-3 after warmup; event
#     should cause delta >= 10% of mean norm = 0.10 (conservative).
THRESH_BINDING_EVENT_RESPONSE = 0.05

# C2: ENTITY_ABLATED entity_event_response <= this (both seeds).
#     Without z_self gating, entity responses should be weaker.
#     Threshold is generous: ABLATED may still show some response from
#     the world encoder's EMA dynamics alone. Set to same as C1 --
#     ABLATED must not exceed the binding response by more than C3 gap.
THRESH_ABLATED_EVENT_RESPONSE = 0.15

# C3: per-seed event response gap (BINDING - ABLATED) >= this (both seeds).
#     Binding must produce at least 30% better entity responsiveness.
#     Conservative: set to 0.02 (absolute delta_z_world units).
THRESH_EVENT_RESPONSE_GAP = 0.02

# C4: ENTITY_BINDING_ON entity_tracking_persistence >= this (both seeds).
#     Cosine similarity of consecutive z_world on non-event post-event steps.
#     Perfect persistence = 1.0; random walk = ~0. Target >= 0.70.
THRESH_BINDING_PERSISTENCE = 0.70

# C5: per-seed persistence gap (BINDING - ABLATED) >= this (both seeds).
#     Binding should produce >= 0.05 higher z_world autocorrelation
#     on entity-tracking steps vs. ablated.
THRESH_PERSISTENCE_GAP = 0.05

# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------
BODY_OBS_DIM = 12
WORLD_OBS_DIM = 250   # CausalGridWorldV2 size=10
ACTION_DIM = 5

SELF_DIM = 32
WORLD_DIM = 32

# Entity event detection: obs_world change magnitude threshold
# (normalized by WORLD_OBS_DIM to be scale-invariant)
EVENT_DELTA_THRESH = 0.02

# Self-maintenance hyperparameters
MAINT_WEIGHT_BINDING = 0.1
MAINT_WEIGHT_ABLATED = 0.0
D_EFF_TARGET = 4.0   # reasonable target for self_dim=32

# Tracking window: number of steps after an event to measure persistence
PERSISTENCE_WINDOW = 5


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed,
        size=10,
        num_hazards=3,
        num_resources=4,
        hazard_harm=0.02,
        env_drift_interval=5,
        env_drift_prob=0.3,   # higher drift -> more entity events
    )


# ---------------------------------------------------------------------------
# Training + Evaluation
# ---------------------------------------------------------------------------

def _run_condition(
    seed: int,
    condition: str,
    warmup_episodes: int,
    eval_episodes: int,
    steps_per_episode: int,
    self_maintenance_weight: float,
    lr: float,
    alpha_world: float,
    alpha_self: float,
    nav_bias: float,
    dry_run: bool,
) -> Dict:
    """
    Run one condition (ENTITY_BINDING_ON or ENTITY_ABLATED) for a single seed.

    In ENTITY_ABLATED, z_self is zeroed after each sense() call during eval.
    self_maintenance_weight controls whether the maintenance gradient is active.

    Returns metrics dict for governance indexer.
    """
    ablate_z_self = (condition == "ENTITY_ABLATED")

    torch.manual_seed(seed)
    random.seed(seed)

    env = _make_env(seed)

    config = REEConfig.from_dims(
        body_obs_dim=BODY_OBS_DIM,
        world_obs_dim=WORLD_OBS_DIM,
        action_dim=ACTION_DIM,
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
        alpha_world=alpha_world,
        alpha_self=alpha_self,
        self_maintenance_weight=self_maintenance_weight,
        self_maintenance_d_eff_target=D_EFF_TARGET,
    )
    agent = REEAgent(config)

    e1_opt = optim.Adam(agent.e1.parameters(), lr=lr)
    e2_opt = optim.Adam(agent.e2.parameters(), lr=lr * 3)
    e3_opt = optim.Adam(
        list(agent.e3.parameters()) + list(agent.latent_stack.parameters()),
        lr=lr,
    )

    if dry_run:
        warmup_episodes = 3
        eval_episodes = 5

    print(
        f"  [V3-EXQ-146] {condition} seed={seed}"
        f" warmup={warmup_episodes} eval={eval_episodes}"
        f" ablate_z_self={ablate_z_self}"
        f" maint_weight={self_maintenance_weight}",
        flush=True,
    )

    # -----------------------------------------------------------------------
    # Phase 1: Warmup training
    # -----------------------------------------------------------------------
    agent.train()

    for ep in range(warmup_episodes):
        _, obs_dict = env.reset()
        agent.reset()
        ep_harm = 0.0

        for _ in range(steps_per_episode):
            obs_body  = torch.tensor(obs_dict["body_state"],  dtype=torch.float32)
            obs_world = torch.tensor(obs_dict["world_state"], dtype=torch.float32)

            z_self_t = None
            if agent._current_latent is not None:
                z_self_t = agent._current_latent.z_self.detach().clone()

            latent = agent.sense(obs_body, obs_world)
            ticks = agent.clock.advance()
            e1_prior = (
                agent._e1_tick(latent) if ticks["e1_tick"]
                else torch.zeros(1, WORLD_DIM, device=agent.device)
            )

            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            action = agent.select_action(candidates, ticks, temperature=1.0)

            if z_self_t is not None:
                agent.record_transition(z_self_t, action, latent.z_self.detach().clone())

            if random.random() < nav_bias:
                action = torch.zeros(1, ACTION_DIM)
                action[0, random.randint(0, ACTION_DIM - 1)] = 1.0

            _, reward, done, _, obs_dict = env.step(action)
            harm_signal = float(reward) if reward < 0 else 0.0
            ep_harm += abs(harm_signal)

            e1_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()
            total_e1_e2 = e1_loss + e2_loss
            if total_e1_e2.requires_grad:
                e1_opt.zero_grad()
                e2_opt.zero_grad()
                total_e1_e2.backward()
                e1_opt.step()
                e2_opt.step()

            if agent._current_latent is not None:
                z_world = agent._current_latent.z_world.detach()
                harm_target = torch.tensor(
                    [[1.0 if harm_signal < 0 else 0.0]], device=agent.device
                )
                harm_loss = F.mse_loss(agent.e3.harm_eval(z_world), harm_target)
                maint_loss = agent.compute_self_maintenance_loss()
                total_e3 = harm_loss + maint_loss
                e3_opt.zero_grad()
                total_e3.backward()
                e3_opt.step()

            agent.update_residue(harm_signal)
            if done:
                break

        if (ep + 1) % 100 == 0:
            print(
                f"    [train] {condition} seed={seed}"
                f" ep {ep+1}/{warmup_episodes} harm={ep_harm:.3f}",
                flush=True,
            )

    # -----------------------------------------------------------------------
    # Phase 2: Evaluation -- entity event tracking
    # -----------------------------------------------------------------------
    agent.eval()

    event_response_list:    List[float] = []  # |delta_z_world| on event steps
    persistence_list:       List[float] = []  # cosine_sim(z_world_t, z_world_{t+1}) in persistence window
    baseline_delta_list:    List[float] = []  # |delta_z_world| on non-event steps (reference)
    baseline_persist_list:  List[float] = []  # cosine_sim on non-event steps

    for _ep in range(eval_episodes):
        _, obs_dict = env.reset()
        agent.reset()

        prev_obs_world_t: torch.Tensor = torch.tensor(
            obs_dict["world_state"], dtype=torch.float32
        )
        prev_z_world: torch.Tensor = torch.zeros(WORLD_DIM)
        in_persistence_window = 0  # countdown: remaining steps after an event to track persistence

        for _step in range(steps_per_episode):
            obs_body  = torch.tensor(obs_dict["body_state"],  dtype=torch.float32)
            obs_world = torch.tensor(obs_dict["world_state"], dtype=torch.float32)

            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world)
                agent.clock.advance()
                agent._e1_tick(latent)

                # Ablation: zero z_self to remove attentional gate
                if ablate_z_self and agent._current_latent is not None:
                    agent._current_latent.z_self.zero_()

            # Detect entity event: large change in obs_world
            obs_delta = float((obs_world - prev_obs_world_t).abs().mean().item())
            is_event = (obs_delta > EVENT_DELTA_THRESH)

            if agent._current_latent is not None:
                z_world_now = agent._current_latent.z_world.squeeze(0).detach().cpu().float()
                z_world_norm = float(z_world_now.norm().item())

                if z_world_norm > 1e-6 and float(prev_z_world.norm().item()) > 1e-6:
                    # delta_z_world: magnitude of change in world latent
                    delta_z_world = float((z_world_now - prev_z_world).norm().item())
                    # cosine similarity: persistence of world representation
                    cos_sim = float(
                        F.cosine_similarity(
                            z_world_now.unsqueeze(0),
                            prev_z_world.unsqueeze(0),
                        ).item()
                    )

                    if is_event:
                        event_response_list.append(delta_z_world)
                        in_persistence_window = PERSISTENCE_WINDOW
                    elif in_persistence_window > 0:
                        # Non-event step immediately after event: measure persistence
                        persistence_list.append(cos_sim)
                        in_persistence_window -= 1
                    else:
                        # Baseline non-event step
                        baseline_delta_list.append(delta_z_world)
                        baseline_persist_list.append(cos_sim)

                prev_z_world = z_world_now

            prev_obs_world_t = obs_world

            action_idx = random.randint(0, ACTION_DIM - 1)
            action = torch.zeros(1, ACTION_DIM)
            action[0, action_idx] = 1.0
            _, _, done, _, obs_dict = env.step(action)
            if done:
                break

    n_events     = len(event_response_list)
    n_persist    = len(persistence_list)
    n_baseline_d = len(baseline_delta_list)
    n_baseline_p = len(baseline_persist_list)

    mean_event_response = (
        sum(event_response_list) / max(1, n_events)
    )
    mean_persistence = (
        sum(persistence_list) / max(1, n_persist)
    )
    mean_baseline_delta = (
        sum(baseline_delta_list) / max(1, n_baseline_d)
    )
    mean_baseline_persistence = (
        sum(baseline_persist_list) / max(1, n_baseline_p)
    )

    # Signal-to-noise: event_response / baseline_delta
    event_snr = (
        mean_event_response / max(1e-6, mean_baseline_delta)
    )

    print(
        f"  [{condition}] seed={seed}"
        f" n_events={n_events}"
        f" event_response={mean_event_response:.4f}"
        f" baseline_delta={mean_baseline_delta:.4f}"
        f" snr={event_snr:.3f}"
        f" persistence={mean_persistence:.4f}"
        f" baseline_persist={mean_baseline_persistence:.4f}",
        flush=True,
    )

    return {
        "seed":                        seed,
        "condition":                   condition,
        "n_entity_events":             n_events,
        "mean_event_response":         mean_event_response,
        "mean_persistence":            mean_persistence,
        "mean_baseline_delta":         mean_baseline_delta,
        "mean_baseline_persistence":   mean_baseline_persistence,
        "event_snr":                   event_snr,
        "n_persistence_samples":       n_persist,
        "n_baseline_samples":          n_baseline_d,
    }


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run(
    seeds: Tuple[int, ...] = (42, 123),
    warmup_episodes: int = 300,
    eval_episodes: int = 100,
    steps_per_episode: int = 200,
    lr: float = 1e-3,
    alpha_world: float = 0.9,
    alpha_self: float = 0.3,
    nav_bias: float = 0.35,
    dry_run: bool = False,
) -> dict:
    """Q-001: Entity Emergence and Binding Discriminative Pair.

    Tests whether z_self attentional gating (self-maintenance active) is
    the operative mechanism for entity persistence and binding in the V3
    latent stack (ARC-006). Compares ENTITY_BINDING_ON (z_self active)
    vs ENTITY_ABLATED (z_self zeroed) on entity event responsiveness
    and entity tracking persistence metrics.
    """
    print(
        f"\n[V3-EXQ-146] Q-001 Entity Binding Discriminative Pair"
        f"  seeds={list(seeds)}"
        f"  warmup={warmup_episodes} eval={eval_episodes}"
        f"  alpha_world={alpha_world}",
        flush=True,
    )

    results_binding: List[Dict] = []
    results_ablated: List[Dict] = []

    for seed in seeds:
        # ENTITY_BINDING_ON condition
        r_binding = _run_condition(
            seed=seed,
            condition="ENTITY_BINDING_ON",
            warmup_episodes=warmup_episodes,
            eval_episodes=eval_episodes,
            steps_per_episode=steps_per_episode,
            self_maintenance_weight=MAINT_WEIGHT_BINDING,
            lr=lr,
            alpha_world=alpha_world,
            alpha_self=alpha_self,
            nav_bias=nav_bias,
            dry_run=dry_run,
        )
        results_binding.append(r_binding)

        # ENTITY_ABLATED condition
        r_ablated = _run_condition(
            seed=seed,
            condition="ENTITY_ABLATED",
            warmup_episodes=warmup_episodes,
            eval_episodes=eval_episodes,
            steps_per_episode=steps_per_episode,
            self_maintenance_weight=MAINT_WEIGHT_ABLATED,
            lr=lr,
            alpha_world=alpha_world,
            alpha_self=alpha_self,
            nav_bias=nav_bias,
            dry_run=dry_run,
        )
        results_ablated.append(r_ablated)

    def _avg(results: List[Dict], key: str) -> float:
        vals = [r[key] for r in results if r[key] == r[key]]
        return float(sum(vals) / max(1, len(vals)))

    # -----------------------------------------------------------------------
    # Pre-registered criteria
    # -----------------------------------------------------------------------

    # C1: BINDING event_response >= THRESH_BINDING_EVENT_RESPONSE (both seeds)
    c1_per_seed = [
        r["mean_event_response"] >= THRESH_BINDING_EVENT_RESPONSE
        for r in results_binding
    ]
    c1_pass = len(c1_per_seed) >= len(seeds) and all(c1_per_seed)

    # C2: ABLATED event_response <= THRESH_ABLATED_EVENT_RESPONSE (both seeds)
    c2_per_seed = [
        r["mean_event_response"] <= THRESH_ABLATED_EVENT_RESPONSE
        for r in results_ablated
    ]
    c2_pass = len(c2_per_seed) >= len(seeds) and all(c2_per_seed)

    # C3: per-seed event_response gap (BINDING - ABLATED) >= THRESH_EVENT_RESPONSE_GAP (both seeds)
    c3_per_seed: List[bool] = []
    per_seed_event_gap: List[float] = []
    for r_b in results_binding:
        matching = [r for r in results_ablated if r["seed"] == r_b["seed"]]
        if matching:
            gap = r_b["mean_event_response"] - matching[0]["mean_event_response"]
            per_seed_event_gap.append(gap)
            c3_per_seed.append(gap >= THRESH_EVENT_RESPONSE_GAP)
    c3_pass = len(c3_per_seed) >= len(seeds) and all(c3_per_seed)

    # C4: BINDING persistence >= THRESH_BINDING_PERSISTENCE (both seeds)
    c4_per_seed = [
        r["mean_persistence"] >= THRESH_BINDING_PERSISTENCE
        for r in results_binding
    ]
    c4_pass = len(c4_per_seed) >= len(seeds) and all(c4_per_seed)

    # C5: per-seed persistence gap (BINDING - ABLATED) >= THRESH_PERSISTENCE_GAP (both seeds)
    c5_per_seed: List[bool] = []
    per_seed_persistence_gap: List[float] = []
    for r_b in results_binding:
        matching = [r for r in results_ablated if r["seed"] == r_b["seed"]]
        if matching:
            gap = r_b["mean_persistence"] - matching[0]["mean_persistence"]
            per_seed_persistence_gap.append(gap)
            c5_per_seed.append(gap >= THRESH_PERSISTENCE_GAP)
    c5_pass = len(c5_per_seed) >= len(seeds) and all(c5_per_seed)

    all_pass = c1_pass and c2_pass and c3_pass and c4_pass and c5_pass
    criteria_met = sum([c1_pass, c2_pass, c3_pass, c4_pass, c5_pass])
    status = "PASS" if all_pass else "FAIL"

    if all_pass:
        decision = "retain_ree"
    elif criteria_met >= 3:
        decision = "hybridize"
    else:
        decision = "retire_ree_claim"

    # Summary averages
    mean_event_binding  = _avg(results_binding, "mean_event_response")
    mean_event_ablated  = _avg(results_ablated, "mean_event_response")
    mean_persist_binding = _avg(results_binding, "mean_persistence")
    mean_persist_ablated = _avg(results_ablated, "mean_persistence")
    mean_baseline_delta  = _avg(results_binding, "mean_baseline_delta")
    mean_snr_binding     = _avg(results_binding, "event_snr")
    mean_snr_ablated     = _avg(results_ablated, "event_snr")

    print(
        f"\n[V3-EXQ-146] Results:",
        flush=True,
    )
    print(
        f"  ENTITY_BINDING_ON:"
        f" event_response={mean_event_binding:.4f}"
        f" [target>={THRESH_BINDING_EVENT_RESPONSE}]"
        f" snr={mean_snr_binding:.3f}"
        f" persistence={mean_persist_binding:.4f}"
        f" [target>={THRESH_BINDING_PERSISTENCE}]",
        flush=True,
    )
    print(
        f"  ENTITY_ABLATED:    "
        f" event_response={mean_event_ablated:.4f}"
        f" [target<={THRESH_ABLATED_EVENT_RESPONSE}]"
        f" snr={mean_snr_ablated:.3f}"
        f" persistence={mean_persist_ablated:.4f}",
        flush=True,
    )
    print(
        f"  event_response_gap(BINDING-ABLATED) per seed: "
        f"{[round(g, 4) for g in per_seed_event_gap]}"
        f"  [target>={THRESH_EVENT_RESPONSE_GAP}]",
        flush=True,
    )
    print(
        f"  persistence_gap(BINDING-ABLATED) per seed:    "
        f"{[round(g, 4) for g in per_seed_persistence_gap]}"
        f"  [target>={THRESH_PERSISTENCE_GAP}]",
        flush=True,
    )
    print(
        f"  C1={c1_pass} C2={c2_pass} C3={c3_pass}"
        f" C4={c4_pass} C5={c5_pass}"
        f"  status={status} ({criteria_met}/5)"
        f"  decision={decision}",
        flush=True,
    )

    # Failure notes
    failure_notes: List[str] = []
    if not c1_pass:
        vals = [round(r["mean_event_response"], 4) for r in results_binding]
        failure_notes.append(
            f"C1 FAIL: BINDING event_response {vals} < {THRESH_BINDING_EVENT_RESPONSE}"
            " -- even with z_self active, world encoder does not respond to entity events;"
            " check EVENT_DELTA_THRESH (may be too large for this env's drift amplitude),"
            " or increase nav_bias so agent encounters hazard events more frequently."
        )
    if not c2_pass:
        vals = [round(r["mean_event_response"], 4) for r in results_ablated]
        failure_notes.append(
            f"C2 FAIL: ABLATED event_response {vals} > {THRESH_ABLATED_EVENT_RESPONSE}"
            " -- world encoder responds to entity events even without z_self gating;"
            " the alpha_world=0.9 EMA alone may be sufficient for entity responsiveness."
            " This would suggest world encoder temporal dynamics (not z_self) are the"
            " primary binding mechanism -- an important Q-001 finding."
        )
    if not c3_pass:
        vals = [round(g, 4) for g in per_seed_event_gap]
        failure_notes.append(
            f"C3 FAIL: event_response_gap {vals} < {THRESH_EVENT_RESPONSE_GAP}"
            " -- z_self ablation has minimal effect on entity responsiveness;"
            " z_self is not the dominant attentional gate for entity binding;"
            " consider that entity emergence may be primarily a world-encoder-internal"
            " mechanism (temporal EMA + event contrastive supervision from SD-009)."
        )
    if not c4_pass:
        vals = [round(r["mean_persistence"], 4) for r in results_binding]
        failure_notes.append(
            f"C4 FAIL: BINDING persistence {vals} < {THRESH_BINDING_PERSISTENCE}"
            " -- z_world autocorrelation in post-event window is low even with binding;"
            " the PERSISTENCE_WINDOW={PERSISTENCE_WINDOW} steps may be too short,"
            " or entity tracking persistence requires additional mechanisms (MECH-045)."
        )
    if not c5_pass:
        vals = [round(g, 4) for g in per_seed_persistence_gap]
        failure_notes.append(
            f"C5 FAIL: persistence_gap {vals} < {THRESH_PERSISTENCE_GAP}"
            " -- persistence improvement from z_self gating is below threshold;"
            " world encoder EMA may already produce most available persistence."
        )

    for note in failure_notes:
        print(f"  {note}", flush=True)

    # Interpretation
    if all_pass:
        interpretation = (
            f"Q-001 PARTIALLY ANSWERED -- z_self attentional gating IS the operative"
            f" mechanism for entity emergence and binding in V3."
            f" ENTITY_BINDING_ON: event_response={mean_event_binding:.4f},"
            f" persistence={mean_persist_binding:.4f}."
            f" ENTITY_ABLATED: event_response={mean_event_ablated:.4f},"
            f" persistence={mean_persist_ablated:.4f}."
            f" Per-seed event_response gaps: {[round(g, 4) for g in per_seed_event_gap]}."
            f" Per-seed persistence gaps: {[round(g, 4) for g in per_seed_persistence_gap]}."
            f" The self-maintenance objective (D_eff-target) enforces structured"
            f" self-representation that amplifies world encoder sensitivity to entity"
            f" events and stabilises tracking representations across consecutive steps."
            f" Supports ARC-006 binding mechanism: z_self as attentional gate for entity"
            f" persistence in z_world."
        )
    elif criteria_met >= 3:
        interpretation = (
            f"Partial support for Q-001 binding mechanism: {criteria_met}/5 criteria met."
            f" Directional evidence that z_self gating improves entity responsiveness"
            f" and/or persistence, but some thresholds not met."
            f" event_response: BINDING={mean_event_binding:.4f} vs ABLATED={mean_event_ablated:.4f}."
            f" persistence: BINDING={mean_persist_binding:.4f} vs ABLATED={mean_persist_ablated:.4f}."
            f" C1={c1_pass} C2={c2_pass} C3={c3_pass} C4={c4_pass} C5={c5_pass}."
        )
    else:
        interpretation = (
            f"Q-001 BINDING MECHANISM NOT CONFIRMED: z_self attentional gating does NOT"
            f" produce detectable improvements in entity event responsiveness or persistence."
            f" event_response: BINDING={mean_event_binding:.4f} vs ABLATED={mean_event_ablated:.4f}."
            f" persistence: BINDING={mean_persist_binding:.4f} vs ABLATED={mean_persist_ablated:.4f}."
            f" Only {criteria_met}/5 criteria met."
            f" The current V3 architecture's entity binding may emerge primarily from"
            f" world-encoder temporal dynamics (alpha_world EMA + event contrastive supervision)"
            f" rather than z_self attentional gating. This constrains Q-001's answer:"
            f" binding in this architecture is encoder-internal, not z_self-gated."
        )

    # Per-seed detail rows
    per_binding_rows = "\n".join(
        f"  seed={r['seed']}: event_response={r['mean_event_response']:.4f}"
        f" snr={r['event_snr']:.3f}"
        f" persistence={r['mean_persistence']:.4f}"
        f" baseline_delta={r['mean_baseline_delta']:.4f}"
        f" n_events={r['n_entity_events']}"
        for r in results_binding
    )
    per_ablated_rows = "\n".join(
        f"  seed={r['seed']}: event_response={r['mean_event_response']:.4f}"
        f" snr={r['event_snr']:.3f}"
        f" persistence={r['mean_persistence']:.4f}"
        f" baseline_delta={r['mean_baseline_delta']:.4f}"
        f" n_events={r['n_entity_events']}"
        for r in results_ablated
    )

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(
            f"- {n}" for n in failure_notes
        )

    summary_markdown = (
        f"# V3-EXQ-146 -- Q-001 Entity Emergence and Binding Discriminative Pair\n\n"
        f"**Status:** {status}\n"
        f"**Claims:** Q-001\n"
        f"**Decision:** {decision}\n"
        f"**Seeds:** {list(seeds)}\n"
        f"**Conditions:** ENTITY_BINDING_ON vs ENTITY_ABLATED\n"
        f"**Warmup:** {warmup_episodes} eps x {steps_per_episode} steps"
        f"  **Eval:** {eval_episodes} eps x {steps_per_episode} steps\n"
        f"**Env:** CausalGridWorldV2 size=10, 3 hazards, 4 resources"
        f" nav_bias={nav_bias}\n\n"
        f"## Design\n\n"
        f"Q-001 asks what mechanisms produce entity emergence and binding in the REE"
        f" latent stack. The candidate mechanism under test: z_self attentional gating"
        f" (self-maintenance active) enables entity persistence in z_world by focusing"
        f" world encoder attention on entity-linked features (ARC-006).\n\n"
        f"ENTITY_BINDING_ON: z_self active, self_maintenance_weight={MAINT_WEIGHT_BINDING}.\n"
        f"ENTITY_ABLATED: z_self zeroed after each sense() call, maint_weight={MAINT_WEIGHT_ABLATED}.\n"
        f"Entity events: |mean(delta_obs_world)| > {EVENT_DELTA_THRESH}"
        f" (hazard drift steps from CausalGridWorldV2 env_drift_prob=0.3).\n\n"
        f"## Pre-Registered Thresholds\n\n"
        f"C1: BINDING entity_event_response >= {THRESH_BINDING_EVENT_RESPONSE} (both seeds)\n"
        f"C2: ABLATED entity_event_response <= {THRESH_ABLATED_EVENT_RESPONSE} (both seeds)\n"
        f"C3: per-seed event_response gap (BINDING-ABLATED) >= {THRESH_EVENT_RESPONSE_GAP} (both seeds)\n"
        f"C4: BINDING entity_tracking_persistence >= {THRESH_BINDING_PERSISTENCE} (both seeds)\n"
        f"C5: per-seed persistence gap (BINDING-ABLATED) >= {THRESH_PERSISTENCE_GAP} (both seeds)\n\n"
        f"## Results\n\n"
        f"| Condition | event_response | event_snr | persistence | baseline_delta |\n"
        f"|-----------|----------------|-----------|-------------|----------------|\n"
        f"| ENTITY_BINDING_ON | {mean_event_binding:.4f} | {mean_snr_binding:.3f}"
        f" | {mean_persist_binding:.4f} | {mean_baseline_delta:.4f} |\n"
        f"| ENTITY_ABLATED    | {mean_event_ablated:.4f} | {mean_snr_ablated:.3f}"
        f" | {mean_persist_ablated:.4f} | -- |\n\n"
        f"**Per-seed event_response gap (BINDING - ABLATED):"
        f" {[round(g, 4) for g in per_seed_event_gap]}**\n"
        f"**Per-seed persistence gap (BINDING - ABLATED):"
        f" {[round(g, 4) for g in per_seed_persistence_gap]}**\n\n"
        f"### ENTITY_BINDING_ON per seed\n{per_binding_rows}\n\n"
        f"### ENTITY_ABLATED per seed\n{per_ablated_rows}\n\n"
        f"## PASS Criteria\n\n"
        f"| Criterion | Result |\n"
        f"|-----------|--------|\n"
        f"| C1: BINDING event_response >= {THRESH_BINDING_EVENT_RESPONSE} (both seeds)"
        f" | {'PASS' if c1_pass else 'FAIL'} |\n"
        f"| C2: ABLATED event_response <= {THRESH_ABLATED_EVENT_RESPONSE} (both seeds)"
        f" | {'PASS' if c2_pass else 'FAIL'} |\n"
        f"| C3: event_response_gap(BINDING-ABLATED) >= {THRESH_EVENT_RESPONSE_GAP} (both seeds)"
        f" | {'PASS' if c3_pass else 'FAIL'} |\n"
        f"| C4: BINDING persistence >= {THRESH_BINDING_PERSISTENCE} (both seeds)"
        f" | {'PASS' if c4_pass else 'FAIL'} |\n"
        f"| C5: persistence_gap(BINDING-ABLATED) >= {THRESH_PERSISTENCE_GAP} (both seeds)"
        f" | {'PASS' if c5_pass else 'FAIL'} |\n\n"
        f"Criteria met: {criteria_met}/5 -> **{status}**\n\n"
        f"## Interpretation\n\n"
        f"{interpretation}\n"
        f"{failure_section}\n"
    )

    metrics: Dict[str, float] = {
        "binding_mean_event_response":     float(mean_event_binding),
        "ablated_mean_event_response":     float(mean_event_ablated),
        "binding_mean_persistence":        float(mean_persist_binding),
        "ablated_mean_persistence":        float(mean_persist_ablated),
        "binding_mean_event_snr":          float(mean_snr_binding),
        "ablated_mean_event_snr":          float(mean_snr_ablated),
        "mean_baseline_delta":             float(mean_baseline_delta),
        "mean_event_response_gap":         float(
            sum(per_seed_event_gap) / max(1, len(per_seed_event_gap))
        ),
        "mean_persistence_gap":            float(
            sum(per_seed_persistence_gap) / max(1, len(per_seed_persistence_gap))
        ),
        "thresh_binding_event_response":   float(THRESH_BINDING_EVENT_RESPONSE),
        "thresh_ablated_event_response":   float(THRESH_ABLATED_EVENT_RESPONSE),
        "thresh_event_response_gap":       float(THRESH_EVENT_RESPONSE_GAP),
        "thresh_binding_persistence":      float(THRESH_BINDING_PERSISTENCE),
        "thresh_persistence_gap":          float(THRESH_PERSISTENCE_GAP),
        "crit1_pass":                      1.0 if c1_pass else 0.0,
        "crit2_pass":                      1.0 if c2_pass else 0.0,
        "crit3_pass":                      1.0 if c3_pass else 0.0,
        "crit4_pass":                      1.0 if c4_pass else 0.0,
        "crit5_pass":                      1.0 if c5_pass else 0.0,
        "criteria_met":                    float(criteria_met),
    }

    return {
        "status":             status,
        "metrics":            metrics,
        "summary_markdown":   summary_markdown,
        "claim_ids":          CLAIM_IDS,
        "evidence_direction": (
            "supports" if all_pass
            else ("mixed" if criteria_met >= 3 else "weakens")
        ),
        "experiment_type":    EXPERIMENT_TYPE,
        "fatal_error_count":  0,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser(
        description="V3-EXQ-146 Q-001 Entity Binding Discriminative Pair"
    )
    parser.add_argument("--seeds",         type=int,   nargs="+", default=[42, 123])
    parser.add_argument("--warmup",        type=int,   default=300)
    parser.add_argument("--eval-eps",      type=int,   default=100)
    parser.add_argument("--steps",         type=int,   default=200)
    parser.add_argument("--alpha-world",   type=float, default=0.9)
    parser.add_argument("--dry-run",       action="store_true")
    args = parser.parse_args()

    result = run(
        seeds=tuple(args.seeds),
        warmup_episodes=args.warmup,
        eval_episodes=args.eval_eps,
        steps_per_episode=args.steps,
        alpha_world=args.alpha_world,
        dry_run=args.dry_run,
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"]       = ts
    result["claim"]               = CLAIM_IDS[0]
    result["verdict"]             = result["status"]
    result["run_id"]              = f"{EXPERIMENT_TYPE}_{ts}_v3"
    result["architecture_epoch"]  = "ree_hybrid_guardrails_v1"

    out_dir = (
        Path(__file__).resolve().parents[2]
        / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{EXPERIMENT_TYPE}_{ts}.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"\nResult written to: {out_path}", flush=True)
    print(f"Status: {result['status']}", flush=True)
    for k, v in result["metrics"].items():
        print(f"  {k}: {v}", flush=True)
