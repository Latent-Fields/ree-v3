#!/opt/local/bin/python3
"""
V3-EXQ-418l -- SD-017 Context-Conditioned Action: Phase 2 substrate retest.
SLEEP DRIVER: K=1 single-fire (SleepLoopManager, default cycle_every_k_episodes=1, fires every episode)

SUPERSEDES: V3-EXQ-418a (FAIL under pre-Phase-2 substrate -- action_bias_div
flat at 0.0003 across all seeds; same FAIL signature as the 418/418a/436
cohort that motivated the Phase 2 substrate template propagation).

ID NOTE: this iteration was originally planned as V3-EXQ-418c per the
sleep_substrate_plan.md GAP-2 owner-list, but V3-EXQ-418c is already in
ree-v3/runner_status.json completed list (cleared as superseded in the
2026-04-25 governance cycle). Using 418c would have been silently skipped
by the runner. 418{a..k} are all used; 418l is the next available letter.
The plan-of-record GAP-2 row should be updated 418c -> 418l on completion.

MECHANISM UNDER TEST: SD-017 (sleep_phase.minimal_sleep_infrastructure_v3)
  Discriminative pair test of whether SD-017 sleep cycling produces
  measurable cue_action_proj differentiation (action_bias_divergence
  between SAFE and DANGEROUS contexts), now under the Phase 2 substrate
  stack:
    - SD-016 Path 1 diversification loss ON (sd016_diversification_weight=0.5,
      A2_div_only equivalent: writepath_mode=off so only the div loss fires).
    - MECH-269 Phase 1 per-stream V_s ON.
    - MECH-269 Phase 2 (ii) anchor sets ON (dual-trace).
    - SD-039 anchor goal-snapshot payload ON (substrate-level).
    - SD-017 SleepLoopManager ON (use_sleep_loop=True), SWS+REM enabled.
  Existing 418a substrate (sd016_enabled=True, shy_enabled=False,
  LAMBDA_TERRAIN=0.1, terrain_loss in E1 training loop) preserved -- the
  5 Phase 2 flags are layered on top. sd016_enabled=True and
  sd016_writepath_mode="off" are NOT contradictory: writepath_mode="off"
  is the A2_div_only equivalent (writes-off, div-on) per EXQ-265a's
  pattern.

EXPERIMENT_PURPOSE: evidence

WHY THE RETEST: V3-EXQ-265a PASS (2026-05-09T20:12Z) validated the Phase 2
substrate template end-to-end on the SD-017 methods-validation experiment.
Per sleep_substrate_plan.md GAP-2 plan-of-record (decision-log entries
2026-05-09T19:49Z and 2026-05-09T20:14Z), the four Tier-1 successors
mechanically apply the validated 5-flag template to their respective base
scripts. 418l is the action_bias_div discriminative pair successor.
Original 418/418a/436 cohort consistently FAILed C1 (action_bias_div<0.05)
because cue_action_proj gradient never reached the discriminator under
the SD-016-confounded baseline (per ree-v3/CLAUDE.md SD-016 section,
EXP-0155 diagnostic confirmed cue_action_proj.weight grad ~0). The
hypothesis under Phase 2 substrate: V_s + anchor sets + SD-039 payload
+ SD-016 div loss break the slot-collapse confound that masked
cue_action_proj training gradient.

CLAIM_IDS RE-EVALUATION (per CLAUDE.md accuracy rule): the 418a tagged
["SD-017"] for an SD-017 sleep-driven action-bias differentiation test.
Under the template change the mechanism under test is unchanged -- still
sleep-cycle driven cue_action_proj differentiation, just now exercised
under the Phase 2 substrate stack. claim_ids=["SD-017"] preserved.
evidence_direction_per_claim is not strictly required (single claim) but
emitted for indexer consistency.

DESIGN (preserved from 418a):
  Two conditions, 3 seeds each [42, 49, 56]:
    WITH_SLEEP_SD016:    sws_enabled=True, rem_enabled=True, sd016_enabled=True
    WITHOUT_SLEEP_SD016: sws_enabled=False, rem_enabled=False, sd016_enabled=True
  Both conditions: 5 Phase 2 flags (sd016_writepath_mode="off",
    sd016_diversification_weight=0.5, use_per_stream_vs=True,
    use_anchor_sets=True, use_sd039_anchor_payload=True). use_sleep_loop
    follows the per-condition with_sleep boolean (matches sws/rem gating).
  Training:
    P0 (P0_EPISODES=50): Encoder warmup. Both conditions. Alternating contexts.
    P1 (P1_EPISODES=150): WITH_SLEEP gets sleep every SLEEP_INTERVAL=10 eps.
  Evaluation: EVAL_EPISODES eval episodes; measure action_bias_divergence.

ACCEPTANCE CRITERIA:
  C1: action_bias_divergence(WITH_SLEEP) >= 0.05 in >= 2/3 seeds
      (the discriminative metric the original 418/418a/436 cohort identified
       as flat at 0.0; passes only if Phase 2 substrate breaks the
       cue_action_proj training-gradient confound).
  C2: action_bias_divergence(WITH_SLEEP) > action_bias_divergence(WITHOUT_SLEEP)
      in >= 2/3 seeds (sleep-specific advantage; without-sleep arm controls
      for SD-016 + V_s + anchor set + payload contribution).
  C3: slot_diversity(WITH_SLEEP) > slot_diversity(WITHOUT_SLEEP) in >= 2/3 seeds
      (secondary check on slot differentiation).
  C4: signed |action_bias_div WITH_SLEEP - action_bias_div WITHOUT_SLEEP|
      > 0.05 in >= 2/3 seeds (Phase 2 acceptance shape per EXQ-265a /
      EXQ-500a precedent: either direction informative -- sleep adding
      action-bias divergence OR sleep collapsing it relative to a
      waking-only attractor under the new substrate).

PASS: C1 AND C2 AND C4.
  C3 is reported as a secondary diagnostic (substrate-side dissociation
  check); it is not part of PASS. The C2/C4 distinction is meaningful:
  C2 is direction-only ("sleep does better"), C4 is signed-magnitude
  ("the two arms differ enough to matter"). Both serve different
  diagnostic functions.

INTERPRETATION GRID (for the discussant reviewing this):
  All PASS                     -> Phase 2 substrate breaks the
                                  cue_action_proj confound on the
                                  action_bias_div metric. SD-017 sleep
                                  cycling produces a measurable
                                  context-conditioned action signature.
                                  Roll GAP-2 owner-list forward; 418l
                                  closes.
  C1 PASS, C2 FAIL, C4 PASS    -> WITH_SLEEP clears 0.05 absolute
                                  threshold but WITHOUT_SLEEP also clears
                                  it (no sleep-specific advantage); C4
                                  signed-diff still confirms the two arms
                                  differ. Read: Phase 2 substrate alone
                                  (without sleep) produces the
                                  cue_action_proj differentiation; sleep
                                  is not the operative mechanism. SD-017
                                  evidence weakens; SD-016 evidence
                                  strengthens. Possibly route to a
                                  no-sleep-only successor (418m).
  C1 PASS, C2 PASS, C4 FAIL    -> WITH_SLEEP > WITHOUT_SLEEP in direction
                                  but |diff| < 0.05. Differentiation is
                                  real but small. Indicates Phase 2
                                  substrate + sleep produces a
                                  sleep-specific signal but the magnitude
                                  is below the substrate-comparable
                                  tolerance. Possibly tighten threshold
                                  in a successor (418n) and/or extend
                                  P1 episodes for fuller training.
  C1 FAIL                      -> action_bias_div(WITH_SLEEP) < 0.05.
                                  Phase 2 substrate did not break the
                                  cue_action_proj confound on this
                                  metric. Substrate hypothesis weakens
                                  for the action-bias pathway.
                                  Architectural concern: cue_action_proj
                                  may need EXP-0155-equivalent gradient
                                  diagnostic OR supervised loss
                                  (per ree-v3/CLAUDE.md SD-016 section
                                  open question). Route via /diagnose-errors.
  C1 FAIL, C4 PASS              -> Both arms produce action_bias_div
                                  below 0.05 absolute threshold but the
                                  arms differ in signed magnitude.
                                  Indicates Phase 2 substrate produces
                                  arm-distinct dynamics but neither
                                  reaches the absolute threshold.
                                  Possibly the absolute 0.05 is
                                  miscalibrated for the Phase 2 regime;
                                  the cohort threshold was set against
                                  the pre-Phase-2 substrate baseline.

claim_ids: ["SD-017"]
architecture_epoch: "ree_hybrid_guardrails_v1"
run_id: ends _v3
supersedes: V3-EXQ-418a
"""

import os
import sys
import json
import argparse
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig
from experiment_protocol import emit_outcome


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPERIMENT_TYPE = "v3_exq_418l_sd017_action_bias_div_phase2"
QUEUE_ID        = "V3-EXQ-418l"
SUPERSEDES_ID   = "V3-EXQ-418a"
CLAIM_IDS       = ["SD-017"]
EXPERIMENT_PURPOSE = "evidence"

P0_EPISODES       = 50          # encoder warmup (no sleep in either condition)
P1_EPISODES       = 150         # P1: WITH_SLEEP gets sleep cycles
EVAL_EPISODES     = 30          # evaluation episodes per seed per condition
STEPS_PER_EPISODE = 150
SLEEP_INTERVAL    = 10          # sleep cycle every N P1 episodes (WITH_SLEEP only)
CONTEXT_SWITCH_EVERY = 5        # alternate SAFE/DANGEROUS every N episodes
LAMBDA_TERRAIN    = 0.1         # terrain_loss weight for SD-016 cue_terrain_proj training

LR = 1e-4

SEEDS = [42, 49, 56]

MIN_CONTEXT_SAMPLES = 10

# C1 absolute threshold (preserved from 418a).
C1_BIAS_ABSOLUTE_THRESHOLD = 0.05

# C4 signed-difference tolerance (matches EXQ-265a / EXQ-500a Phase 2 default).
# Per the cohort starter prompt, tightening to 0.03 is appropriate if smoke
# saturates at the 0.05 boundary; default 0.05 chosen to match 265a.
C4_DIFF_TOLERANCE = 0.05

# Phase 2 substrate template constants (validated by EXQ-265a PASS 2026-05-09T20:12Z).
SD016_DIVERSIFICATION_WEIGHT = 0.5  # matches EXQ-418e LAMBDA_DIVERSIFY / EXQ-265a / EXQ-500a


# ---------------------------------------------------------------------------
# Terrain loss helpers (copied from v3_exq_182_sd016_terrain_calibration.py
# via 418a; preserved verbatim).
# ---------------------------------------------------------------------------

def get_hazard_max(obs_dict: Dict, world_obs: Optional[torch.Tensor]) -> float:
    """Extract hazard_field_view.max() from observation dict."""
    if "harm_obs" in obs_dict:
        harm_obs = obs_dict["harm_obs"]
        if hasattr(harm_obs, 'shape') and harm_obs.shape[-1] >= 26:
            return float(harm_obs[..., :25].max().item())
    if "hazard_field_view" in obs_dict:
        hfv = obs_dict["hazard_field_view"]
        if hasattr(hfv, 'shape'):
            return float(hfv.max().item())
    if world_obs is not None and world_obs.shape[-1] >= 225:
        return float(world_obs[..., 200:225].max().item())
    return 0.0


def compute_terrain_loss(agent: REEAgent, z_world: torch.Tensor, hazard_max: float) -> torch.Tensor:
    """Supervised terrain_loss for cue_terrain_proj (extract_cue_context WITH gradients)."""
    _, terrain_weight = agent.e1.extract_cue_context(z_world)
    w_harm_target = 0.8 if hazard_max > 0.3 else 0.2
    w_goal_target = 0.8 if hazard_max < 0.1 else 0.3
    target = torch.tensor(
        [[w_harm_target, w_goal_target]],
        dtype=terrain_weight.dtype,
        device=terrain_weight.device,
    )
    return F.mse_loss(terrain_weight, target)


# ---------------------------------------------------------------------------
# Environment factories (preserved from 418a)
# ---------------------------------------------------------------------------

def _make_env_safe(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed,
        size=8,
        num_hazards=1,
        num_resources=3,
        hazard_harm=0.02,
        use_proxy_fields=True,
        resource_respawn_on_consume=True,
    )


def _make_env_dangerous(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed + 1000,
        size=8,
        num_hazards=5,
        num_resources=3,
        hazard_harm=0.04,
        use_proxy_fields=True,
        resource_respawn_on_consume=True,
    )


# ---------------------------------------------------------------------------
# Agent factory -- extends 418a with 5 Phase 2 flags + use_sleep_loop
# ---------------------------------------------------------------------------

def _make_agent(env: CausalGridWorldV2, with_sleep: bool) -> REEAgent:
    """Build REEAgent with 418a substrate + 5 Phase 2 flags applied.

    Existing 418a substrate preserved:
      - sd016_enabled=True (SD-016 cue indexing on; both conditions).
      - shy_enabled=False (prevents 0.85^15 slot collapse).
      - LAMBDA_TERRAIN=0.1 applied in training loop (cue_terrain_proj training).

    Phase 2 substrate template added (validated by EXQ-265a PASS):
      - sd016_writepath_mode="off"          (A2_div_only: writes off, div on)
      - sd016_diversification_weight=0.5     (matches EXQ-418e / EXQ-265a)
      - use_per_stream_vs=True               (MECH-269 Phase 1)
      - use_anchor_sets=True                 (MECH-269 Phase 2 ii; precondition
                                              raises if use_per_stream_vs=False)
      - use_sd039_anchor_payload=True        (SD-039 substrate-side payload)
      - use_sleep_loop=with_sleep            (SD-017 SleepLoopManager; tracks
                                              sws/rem gating per-condition)

    Note: sd016_enabled=True and sd016_writepath_mode="off" are NOT
    contradictory. writepath_mode="off" disables the SD-016 write path
    while preserving the SD-016 ContextMemory query path; the
    diversification loss still trains the slots. This is the A2_div_only
    pattern from EXQ-418e.
    """
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        alpha_world=0.9,
        alpha_self=0.3,
        # SD-016: frontal cue-indexed context retrieval (BOTH conditions, 418a)
        sd016_enabled=True,
        # Phase 2 substrate template (5 flags, mechanically applied per
        # sleep_substrate_plan.md GAP-2 propagation pattern).
        sd016_writepath_mode="off",
        sd016_diversification_weight=SD016_DIVERSIFICATION_WEIGHT,
        use_per_stream_vs=True,
        use_anchor_sets=True,
        use_sd039_anchor_payload=True,
        # SD-017: sleep phase switches (only WITH_SLEEP) + Phase A SleepLoopManager
        sws_enabled=with_sleep,
        sws_consolidation_steps=8,
        sws_schema_weight=0.1,
        rem_enabled=with_sleep,
        rem_attribution_steps=6,
        use_sleep_loop=with_sleep,
        # SHY disabled: isolates sleep -> slots -> action pathway (fix from 418a).
        shy_enabled=False,
    )
    return REEAgent(cfg)


# ---------------------------------------------------------------------------
# Slot diversity helper (preserved from 418a)
# ---------------------------------------------------------------------------

def _compute_slot_diversity(agent: REEAgent) -> float:
    """Mean pairwise cosine distance between E1 ContextMemory slots."""
    with torch.no_grad():
        mem = agent.e1.context_memory.memory  # [num_slots, memory_dim]
        n = mem.shape[0]
        if n < 2:
            return 0.0
        normed = F.normalize(mem, dim=-1)
        sim = torch.mm(normed, normed.t())  # [n, n]
        mask = ~torch.eye(n, dtype=torch.bool, device=mem.device)
        dist = 1.0 - sim[mask]
        return float(dist.mean().item())


# ---------------------------------------------------------------------------
# Action bias extraction (preserved from 418a)
# ---------------------------------------------------------------------------

def _extract_action_bias(agent: REEAgent, z_world: torch.Tensor) -> Optional[torch.Tensor]:
    """Extract action_bias from E1 ContextMemory via SD-016 cue projection."""
    if not hasattr(agent.e1, 'world_query_proj'):
        return None
    with torch.no_grad():
        z_w = z_world.detach()
        if z_w.dim() == 1:
            z_w = z_w.unsqueeze(0)
        action_bias, _ = agent.e1.extract_cue_context(z_w)
        return action_bias.squeeze(0)


def _action_bias_divergence(
    safe_biases: List[torch.Tensor],
    dang_biases: List[torch.Tensor],
) -> float:
    """Mean cosine distance between action_bias vectors from SAFE vs DANGEROUS."""
    if len(safe_biases) < MIN_CONTEXT_SAMPLES or len(dang_biases) < MIN_CONTEXT_SAMPLES:
        return 0.0
    with torch.no_grad():
        safe_mat = torch.stack(safe_biases[:50])
        dang_mat = torch.stack(dang_biases[:50])
        safe_norm = F.normalize(safe_mat, dim=-1)
        dang_norm = F.normalize(dang_mat, dim=-1)
        sim_mat = torch.mm(safe_norm, dang_norm.t())
        mean_sim = float(sim_mat.mean().item())
        return max(0.0, 1.0 - mean_sim)


# ---------------------------------------------------------------------------
# One-hot helper (preserved from 418a)
# ---------------------------------------------------------------------------

def _onehot(idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, idx] = 1.0
    return v


# ---------------------------------------------------------------------------
# Training episode (preserved from 418a -- terrain_loss + pred_loss)
# ---------------------------------------------------------------------------

def _run_training_episode(
    agent: REEAgent,
    env: CausalGridWorldV2,
    optimizer: optim.Optimizer,
) -> Tuple[float, float]:
    """Train for one episode with random actions. Returns (harm_rate, pred_loss).

    terrain_loss applied (LAMBDA_TERRAIN=0.1) when sd016_enabled=True;
    gradient flows through cue_terrain_proj -> output_proj -> world_query_proj.
    """
    device = agent.device
    _, obs_dict = env.reset()
    agent.reset()
    agent.e1.reset_hidden_state()

    ep_harm = 0.0
    ep_steps = 0
    pred_losses: List[float] = []

    for _step in range(STEPS_PER_EPISODE):
        obs_body = obs_dict["body_state"]
        obs_world = obs_dict["world_state"]
        if not torch.is_tensor(obs_body):
            obs_body = torch.tensor(obs_body, dtype=torch.float32, device=device)
        else:
            obs_body = obs_body.to(device)
        if obs_body.dim() == 1:
            obs_body = obs_body.unsqueeze(0)

        if not torch.is_tensor(obs_world):
            obs_world = torch.tensor(obs_world, dtype=torch.float32, device=device)
        else:
            obs_world = obs_world.to(device)
        if obs_world.dim() == 1:
            obs_world = obs_world.unsqueeze(0)

        latent = agent.sense(obs_body, obs_world)
        agent.clock.advance()

        # Capture hazard_max BEFORE env.step (uses current obs)
        hazard_max = get_hazard_max(obs_dict, obs_world)

        action_idx = random.randint(0, env.action_dim - 1)
        action = _onehot(action_idx, env.action_dim, device)

        if agent._current_latent is not None:
            z_self_prev = agent._current_latent.z_self.detach().clone()
            agent.record_transition(z_self_prev, action, latent.z_self.detach())

        _, harm_signal, done, info, obs_dict = env.step(action)
        ep_harm += max(0.0, float(-harm_signal))
        ep_steps += 1

        # Combined loss: prediction + terrain_loss for SD-016 training
        pred_loss = agent.compute_prediction_loss()
        t_loss = compute_terrain_loss(agent, latent.z_world, hazard_max)
        total_loss = pred_loss + LAMBDA_TERRAIN * t_loss
        if total_loss.requires_grad:
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
            optimizer.step()
            pred_losses.append(float(pred_loss.item()))

        agent.update_residue(float(harm_signal) if float(harm_signal) < 0 else 0.0)

        if done:
            break

    harm_rate = ep_harm / max(1, ep_steps)
    mean_pred_loss = float(sum(pred_losses) / len(pred_losses)) if pred_losses else 0.0
    return harm_rate, mean_pred_loss


# ---------------------------------------------------------------------------
# Evaluation episode (preserved from 418a)
# ---------------------------------------------------------------------------

def _run_eval_episode(
    agent: REEAgent,
    env: CausalGridWorldV2,
    env_label: str,
    safe_biases: List[torch.Tensor],
    dang_biases: List[torch.Tensor],
) -> float:
    """Run one eval episode (no training). Collect action_bias by context."""
    device = agent.device
    _, obs_dict = env.reset()
    agent.reset()
    agent.e1.reset_hidden_state()

    ep_harm = 0.0
    ep_steps = 0

    for _step in range(STEPS_PER_EPISODE):
        obs_body = obs_dict["body_state"]
        obs_world = obs_dict["world_state"]
        if not torch.is_tensor(obs_body):
            obs_body = torch.tensor(obs_body, dtype=torch.float32, device=device)
        else:
            obs_body = obs_body.to(device)
        if obs_body.dim() == 1:
            obs_body = obs_body.unsqueeze(0)

        if not torch.is_tensor(obs_world):
            obs_world = torch.tensor(obs_world, dtype=torch.float32, device=device)
        else:
            obs_world = obs_world.to(device)
        if obs_world.dim() == 1:
            obs_world = obs_world.unsqueeze(0)

        latent = agent.sense(obs_body, obs_world)
        agent.clock.advance()

        if latent.z_world is not None:
            bias = _extract_action_bias(agent, latent.z_world)
            if bias is not None:
                if env_label == "SAFE":
                    safe_biases.append(bias.cpu())
                else:
                    dang_biases.append(bias.cpu())

        action_idx = random.randint(0, env.action_dim - 1)
        action = _onehot(action_idx, env.action_dim, device)

        _, harm_signal, done, _info, obs_dict = env.step(action)
        ep_harm += max(0.0, float(-harm_signal))
        ep_steps += 1

        if done:
            break

    return ep_harm / max(1, ep_steps)


# ---------------------------------------------------------------------------
# Condition runner (preserved from 418a, identical control flow)
# ---------------------------------------------------------------------------

def run_condition(
    condition_name: str,
    with_sleep: bool,
    seed: int,
    dry_run: bool = False,
) -> Dict:
    """Run one condition x seed. Returns result dict."""
    torch.manual_seed(seed)
    random.seed(seed)

    p0_eps   = P0_EPISODES   if not dry_run else 3
    p1_eps   = P1_EPISODES   if not dry_run else 5
    eval_eps = EVAL_EPISODES if not dry_run else 4
    total_eps = p0_eps + p1_eps

    env_safe = _make_env_safe(seed)
    env_dang = _make_env_dangerous(seed)

    agent = _make_agent(env_safe, with_sleep)
    optimizer = optim.Adam(
        list(agent.e1.parameters()) + list(agent.latent_stack.parameters()),
        lr=LR,
    )

    print(
        f"  seed boundary: cond={condition_name} seed={seed}",
        flush=True,
    )

    # ---- P0: encoder warmup (no sleep in either condition) ----
    for ep in range(p0_eps):
        use_dangerous = (ep // CONTEXT_SWITCH_EVERY) % 2 == 1
        env = env_dang if use_dangerous else env_safe
        harm_rate, pred_loss = _run_training_episode(agent, env, optimizer)

        report_interval = max(1, p0_eps // 2)
        if (ep + 1) % report_interval == 0 or ep == 0:
            print(
                f"  [train] cond={condition_name} seed={seed}"
                f" ep {ep + 1}/{total_eps} phase=P0"
                f" harm_rate={harm_rate:.4f} pred_loss={pred_loss:.4f}",
                flush=True,
            )

    # ---- P1: WITH_SLEEP gets sleep cycles ----
    for ep in range(p1_eps):
        abs_ep = p0_eps + ep
        use_dangerous = (abs_ep // CONTEXT_SWITCH_EVERY) % 2 == 1
        env = env_dang if use_dangerous else env_safe
        harm_rate, pred_loss = _run_training_episode(agent, env, optimizer)

        if with_sleep and (ep + 1) % SLEEP_INTERVAL == 0:
            sleep_m = agent.run_sleep_cycle()
            sws_writes = sleep_m.get("sws_n_writes", 0.0)
            sws_div    = sleep_m.get("sws_slot_diversity", 0.0)
            rem_rolls  = sleep_m.get("rem_n_rollouts", 0.0)
            print(
                f"  [sleep] cond={condition_name} seed={seed}"
                f" ep {abs_ep + 1}/{total_eps}"
                f" sws_writes={sws_writes:.0f}"
                f" slot_div={sws_div:.4f}"
                f" rem_rolls={rem_rolls:.0f}",
                flush=True,
            )

        report_interval = max(1, p1_eps // 3)
        if (ep + 1) % report_interval == 0 or ep == p1_eps - 1:
            print(
                f"  [train] cond={condition_name} seed={seed}"
                f" ep {abs_ep + 1}/{total_eps} phase=P1"
                f" harm_rate={harm_rate:.4f} pred_loss={pred_loss:.4f}",
                flush=True,
            )

    slot_diversity = _compute_slot_diversity(agent)

    # ---- Evaluation ----
    safe_biases: List[torch.Tensor] = []
    dang_biases: List[torch.Tensor] = []
    eval_harm_rates: List[float] = []

    for ev_ep in range(eval_eps):
        if ev_ep % 2 == 0:
            h = _run_eval_episode(agent, env_safe, "SAFE", safe_biases, dang_biases)
        else:
            h = _run_eval_episode(agent, env_dang, "DANGEROUS", safe_biases, dang_biases)
        eval_harm_rates.append(h)

    eval_harm_rate = float(sum(eval_harm_rates) / len(eval_harm_rates)) if eval_harm_rates else 0.0
    action_bias_div = _action_bias_divergence(safe_biases, dang_biases)
    n_safe_samples  = len(safe_biases)
    n_dang_samples  = len(dang_biases)

    print(
        f"  verdict: cond={condition_name} seed={seed}"
        f" slot_div={slot_diversity:.4f}"
        f" action_bias_div={action_bias_div:.4f}"
        f" eval_harm_rate={eval_harm_rate:.4f}"
        f" n_safe={n_safe_samples} n_dang={n_dang_samples}",
        flush=True,
    )

    return {
        "condition": condition_name,
        "seed": seed,
        "slot_diversity": slot_diversity,
        "action_bias_divergence": action_bias_div,
        "eval_harm_rate": eval_harm_rate,
        "n_safe_bias_samples": n_safe_samples,
        "n_dang_bias_samples": n_dang_samples,
        "p0_episodes": p0_eps,
        "p1_episodes": p1_eps,
        "eval_episodes": eval_eps,
    }


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

def main(dry_run: bool = False) -> Tuple[Dict, Optional[str]]:
    conditions = [
        ("WITH_SLEEP_SD016",    True),
        ("WITHOUT_SLEEP_SD016", False),
    ]
    all_results: Dict[str, List[Dict]] = {cname: [] for cname, _ in conditions}

    n_seeds   = len(SEEDS)
    total_runs = len(conditions) * n_seeds

    run_num = 0
    for cond_name, with_sleep in conditions:
        for seed in SEEDS:
            run_num += 1
            print(
                f"[train] ep {run_num}/{total_runs}"
                f" cond={cond_name} seed={seed}",
                flush=True,
            )
            res = run_condition(cond_name, with_sleep, seed, dry_run=dry_run)
            all_results[cond_name].append(res)

    c1_wins = 0
    c2_wins = 0
    c3_wins = 0
    c4_wins = 0

    per_seed_comparisons = []
    for with_r, wo_r in zip(
        all_results["WITH_SLEEP_SD016"], all_results["WITHOUT_SLEEP_SD016"]
    ):
        assert with_r["seed"] == wo_r["seed"], "Seed mismatch in comparison"
        s = with_r["seed"]

        bias_threshold_win   = with_r["action_bias_divergence"] >= C1_BIAS_ABSOLUTE_THRESHOLD
        bias_improvement_win = with_r["action_bias_divergence"] > wo_r["action_bias_divergence"]
        div_win              = with_r["slot_diversity"] > wo_r["slot_diversity"]
        signed_diff          = with_r["action_bias_divergence"] - wo_r["action_bias_divergence"]
        c4_win               = abs(signed_diff) > C4_DIFF_TOLERANCE

        c1_wins += int(bias_threshold_win)
        c2_wins += int(bias_improvement_win)
        c3_wins += int(div_win)
        c4_wins += int(c4_win)

        per_seed_comparisons.append({
            "seed": s,
            "with_action_bias_div": with_r["action_bias_divergence"],
            "without_action_bias_div": wo_r["action_bias_divergence"],
            "signed_diff": signed_diff,
            "abs_diff": abs(signed_diff),
            "c1_bias_threshold_win": bias_threshold_win,
            "c2_bias_improvement_win": bias_improvement_win,
            "c4_signed_abs_diff_gt_tolerance": c4_win,
            "with_slot_diversity": with_r["slot_diversity"],
            "without_slot_diversity": wo_r["slot_diversity"],
            "c3_diversity_win": div_win,
            "with_eval_harm_rate": with_r["eval_harm_rate"],
            "without_eval_harm_rate": wo_r["eval_harm_rate"],
            "with_n_safe_samples": with_r["n_safe_bias_samples"],
            "with_n_dang_samples": with_r["n_dang_bias_samples"],
        })

    threshold = 2  # >= 2 of 3 seeds
    c1_pass = c1_wins >= threshold
    c2_pass = c2_wins >= threshold
    c3_pass = c3_wins >= threshold
    c4_pass = c4_wins >= threshold

    # PASS = C1 AND C2 AND C4. C3 is secondary.
    outcome = "PASS" if (c1_pass and c2_pass and c4_pass) else "FAIL"

    print(
        f"verdict: C1_bias_threshold={c1_pass} ({c1_wins}/{n_seeds}),"
        f" C2_bias_improvement={c2_pass} ({c2_wins}/{n_seeds}),"
        f" C3_slot_div={c3_pass} ({c3_wins}/{n_seeds}),"
        f" C4_signed_diff={c4_pass} ({c4_wins}/{n_seeds})"
        f" => {outcome}",
        flush=True,
    )

    ts     = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"
    direction = "supports" if outcome == "PASS" else "does_not_support"

    output = {
        "schema_version": "v1",
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "run_id": run_id,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "claim_ids_tested": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "evidence_class": "ablation_pair",
        "outcome": outcome,
        "result": outcome,
        "timestamp_utc": ts,
        "evidence_direction": direction,
        "evidence_direction_per_claim": {"SD-017": direction},
        "supersedes": SUPERSEDES_ID,
        "experiment_version": "l",
        "fix_description": (
            "Phase 2 substrate retest of SD-017 action_bias_div discriminative pair. "
            "Successor to V3-EXQ-418a (FAIL action_bias_div=0.0003) under the validated "
            "5-flag substrate template (SD-016 div loss + MECH-269 V_s + anchor sets + "
            "SD-039 payload + SD-017 sleep loop). Existing 418a substrate "
            "(sd016_enabled=True, shy_enabled=False, LAMBDA_TERRAIN=0.1, terrain_loss "
            "in E1 training) preserved -- the 5 Phase 2 flags layer on top. "
            "ID is 418l rather than 418c because 418c is in runner_status completed "
            "list (would be silently skipped); 418{a..k} all used."
        ),
        "acceptance_checks": {
            "C1_action_bias_div_gte_0.05_in_2of3_seeds": c1_pass,
            "C1_wins": c1_wins,
            "C2_action_bias_div_improvement_2of3_seeds": c2_pass,
            "C2_wins": c2_wins,
            "C3_slot_diversity_improvement_2of3_seeds": c3_pass,
            "C3_wins": c3_wins,
            "C4_signed_abs_diff_gt_0.05_in_2of3_seeds": c4_pass,
            "C4_wins": c4_wins,
            "primary_pass": c1_pass and c2_pass and c4_pass,
            "secondary_pass": c3_pass,
        },
        "registered_thresholds": {
            "C1_BIAS_ABSOLUTE_THRESHOLD": C1_BIAS_ABSOLUTE_THRESHOLD,
            "C4_DIFF_TOLERANCE": C4_DIFF_TOLERANCE,
            "MIN_CONTEXT_SAMPLES": MIN_CONTEXT_SAMPLES,
        },
        "per_seed_comparisons": per_seed_comparisons,
        "all_results": {
            cond: results for cond, results in all_results.items()
        },
        "params": {
            "p0_episodes": P0_EPISODES if not dry_run else 3,
            "p1_episodes": P1_EPISODES if not dry_run else 5,
            "eval_episodes": EVAL_EPISODES if not dry_run else 4,
            "steps_per_episode": STEPS_PER_EPISODE,
            "sleep_interval": SLEEP_INTERVAL,
            "context_switch_every": CONTEXT_SWITCH_EVERY,
            "lambda_terrain": LAMBDA_TERRAIN,
            "seeds": SEEDS,
            "sd016_enabled": True,
            "sws_consolidation_steps": 8,
            "rem_attribution_steps": 6,
            "shy_enabled": False,
            "shy_decay_rate": "disabled",
            # Phase 2 substrate template (recorded for indexer audit).
            "sd016_writepath_mode": "off",
            "sd016_diversification_weight": SD016_DIVERSIFICATION_WEIGHT,
            "use_per_stream_vs": True,
            "use_anchor_sets": True,
            "use_sd039_anchor_payload": True,
            "use_sleep_loop_with_sleep_arm": True,
            "min_context_samples": MIN_CONTEXT_SAMPLES,
            "dry_run": dry_run,
        },
        "notes": (
            "Phase 2 retest of SD-017 action_bias_div discriminative pair. "
            "Cohort starter prompt: /tmp/sleep_substrate_phase2_remaining_starter_prompt.md. "
            "Plan-of-record: REE_assembly/evidence/planning/sleep_substrate_plan.md GAP-2."
        ),
    }

    out_path: Optional[str] = None
    if not dry_run:
        out_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "..", "REE_assembly", "evidence", "experiments",
        )
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{run_id}.json")
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Results written to {out_path}", flush=True)
    else:
        print(f"[DRY RUN] run_id={run_id} outcome={outcome}", flush=True)

    print(f"Outcome: {outcome}", flush=True)
    return output, out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run minimal episodes to verify wiring",
    )
    args = parser.parse_args()
    result, out_path = main(dry_run=args.dry_run)

    _outcome_raw = str(result.get("outcome", "FAIL")).upper()
    _outcome_clean = _outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL"
    emit_outcome(
        outcome=_outcome_clean,
        manifest_path=out_path,
    )
