#!/opt/local/bin/python3
"""
V3-EXQ-1106 -- MECH-287 option B (PAG descending release): Stage-0 precondition gate.

SLEEP DRIVER: not applicable (no sleep flags set).
RED-TEAM (fable, Step 4.5): CONTESTED. F1 (S0-3 non-degeneracy read A's lock,
  not the reach arm's) FIXED: reach_arm_frozen_in_eval precondition + labels
  reach_arm_never_frozen_in_eval / stage0_pass_reach_arm_lock_weak + non-load-
  bearing S0_5. F2 (per-step fraction conflates late lock entry with no lock)
  FIXED as a split label + freeze_fraction_after_entry readout; the pre-registered
  S0-1 statistic is unchanged (same instrument as V3-EXQ-475). F3 (H route
  suppressed while frozen under MECH-284) NOTED: eval_h_events recorded; a zero
  is an expected outcome. F4 (identity precondition tautological) NOTED.

WHAT THIS IS. The pre-registered Stage 0 of the lock-DV test in
REE_assembly/evidence/planning/mech287_anchor_freeze_exit_design_20260925.md
section 6 (892a53036a + 1e0d9d935d). It runs NO path contrast. It measures, on
today's substrate (freeze no-op = STAY, ree-v3 1fc881692d), whether the
Stage-1 lock-DV test has anything to test:

  S0-1 LOCK REPRODUCES. Comparator arm A_BOTH_OFF (V3-EXQ-1097 lineage flags,
       descending-release path OFF) at V3-EXQ-475 scale: eval freeze-active
       fraction >= 0.80 on >= 2 of 3 seeds. The 475 reference (1.0) was recorded
       under the UP no-op defect, so whether the lock reproduces with STAY is
       unmeasured. FAIL -> PRECONDITION-FAILED: the lock DV has no comparator;
       MECH-287's lock falsifier routes back to the user (option A or C of the
       2026-09-25 decision). It is NOT a falsification.
  S0-2 LOCK RATIO. Distribution of z_harm_a_norm / (theta_freeze * gaba_tone)
       on frozen eval gate ticks of A_BOTH_OFF (pooled over seeds). Fixes alpha
       by the pre-registered rule: alpha = smallest of {1, 2, 4} with
       1 + alpha >= median lock ratio. Median > 5 -> INERT-BY-MAGNITUDE (the path
       cannot reach the lock at a bounded gain) and Stage 1 does not run.
  S0-3 REACH IN REGIME. Arm D_BOTH_ON with the path ON at alpha 0 (behaviour
       bit-identical to path OFF by construction, contract C1 of
       tests/contracts/test_pag_descending_release.py):
       _pag_desc_n_drive_steps_while_frozen > 0 in EVAL on >= 2 of 3 seeds.
       Zero -> built but not reached in this regime; not a falsification.
  S0-4 BUDGET. Wall-clock per warmup episode (per cell), to size Stage 1. The
       old ~20 min/seed estimate is void. Informational, not load-bearing.

OUTCOME. PASS iff S0-1 AND S0-2 AND S0-3 (plain AND). Every outcome is
evidence_direction non_contributory (a precondition diagnostic; it tests no
claim hypothesis). The label routes the next action:
  precondition_failed_no_lock_comparator  -> user decision (option A / C)
  inert_by_magnitude                      -> report; Stage 1 does not run
  path_built_not_reached_in_regime        -> report; Stage 1 does not run
  stage0_pass                             -> queue Stage 1 at the recorded alpha
  substrate_not_ready_requeue             -> the PAG gate never ticked in eval

MEASUREMENT NOTES (all fixed before the run):
- Freeze-active fraction is per ENV STEP over eval (primary, as V3-EXQ-1097's
  freeze_active_steps / n_eval_steps). The gate ticks only on E3 ticks and its
  state persists between ticks; the per-GATE-TICK fraction is also recorded.
- Lock ratio is read ONLY on fresh gate ticks (pag diagnostics n_ticks
  advanced this env step) that end frozen, never on a latched last_output --
  no pseudo-replication. It is computed as z_harm_a_norm / exit_threshold,
  which equals z_harm_a_norm / (theta_freeze * gaba_tone) EXACTLY here because
  override_signal is 0.0 (SD-037 not constructed in this lineage) and
  alpha_descending is 0.0 in both arms. The driver asserts both per tick and
  records n_ticks where the identity check failed (expected 0).
- The reach counter is cumulative across episodes by design
  (REEAgent.pag_descending_release_diagnostics docstring), so the eval value
  is the delta from a snapshot taken at eval entry.
- Warmup is the V3-EXQ-1097 / V3-EXQ-475 protocol verbatim (including its
  two sense() calls per step); eval is no_grad, one sense() per step.

Claim tag: MECH-287 (this is the precondition gate for its lock falsifier).
EXPERIMENT_PURPOSE diagnostic -- excluded from governance scoring.
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig

from experiment_protocol import emit_outcome
from experiments._lib.arm_fingerprint import arm_cell
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator
from experiments.pack_writer import write_flat_manifest


EXPERIMENT_TYPE = "v3_exq_1106_mech287b_stage0_lock_precondition"
EXPERIMENT_PURPOSE = "diagnostic"
CLAIM_IDS = ["MECH-287"]
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"
QUEUE_ID = "V3-EXQ-1106"

# --------------------------------------------------------------------------- #
# Pre-registered thresholds (design doc section 6 Stage 0, verbatim)           #
# --------------------------------------------------------------------------- #
LOCK_FREEZE_FRACTION_FLOOR = 0.80   # S0-1, per seed
SEED_MAJORITY = 2                   # ">= 2 of 3 seeds"
ALPHA_LADDER = (1.0, 2.0, 4.0)      # S0-2 rule
LOCK_RATIO_INERT_ABOVE = 5.0        # S0-2: median > 5 -> INERT-BY-MAGNITUDE
REACH_FLOOR = 1                     # S0-3: n_drive_steps_while_frozen > 0 (>= 1)
MIN_FROZEN_GATE_TICKS = 10          # S0-2 cannot-determine floor on pooled n
IDENTITY_TOL = 1e-9
ANCHOR_REACHABILITY_EXEMPT = (
    "each readiness predicate IS its degeneracy definition, reachable by "
    "construction: gate n_ticks >= 1 (the gate ticks on every E3 tick when "
    "use_pag_freeze_gate is on), path enabled at alpha 0 (a config readback), "
    "identity violations == 0 (an arithmetic identity at override 0 / alpha 0)")

# --------------------------------------------------------------------------- #
# Lineage config (V3-EXQ-475 via V3-EXQ-1097, verbatim)                        #
# --------------------------------------------------------------------------- #
HARM_MODE_THRESH   = 0.25
EXPLORE_ERR_THRESH = 0.10

ENV_KWARGS = dict(
    size=10, num_hazards=3, num_resources=5, hazard_harm=0.05,
    env_drift_interval=5, env_drift_prob=0.1,
    proximity_harm_scale=0.1, proximity_benefit_scale=0.05,
    proximity_approach_threshold=0.2, hazard_field_decay=0.5,
    resource_respawn_on_consume=True, use_proxy_fields=True, toroidal=False,
    harm_history_len=10, limb_damage_enabled=True, damage_increment=0.15,
    failure_prob_scale=0.3, heal_rate=0.002, n_landmarks_b=2,
)

WARMUP_EPISODES   = 60
EVAL_EPISODES     = 5
STEPS_PER_EPISODE = 200
WORLD_DIM = 32
SELF_DIM  = 32
HARM_DIM  = 32
HARM_A_DIM = 16
HARM_HISTORY_LEN = 10

WF_BUF_MAX = 2000
HARM_EVAL_BUF_MAX = 2000
BATCH_SIZE = 32
LR_E1 = 1e-4
LR_E2_WF = 3e-4
LR_E3_HARM = 1e-3
LR_ENC_AUX = 5e-4

SEEDS = [0, 1, 2]

# arm -> (MECH-287 chain flags, descending-release path)
ARMS: Dict[str, Dict] = {
    # comparator: V3-EXQ-1097 A_BOTH_OFF lineage flags, path OFF
    "A_BOTH_OFF": dict(trigger=False, accumulator=False, segmenter=True,
                       path_on=False),
    # treatment chain, path ON at alpha 0 (bit-identical behaviour)
    "D_BOTH_ON":  dict(trigger=True,  accumulator=True,  segmenter=True,
                       path_on=True),
}
COMPARATOR_ARM = "A_BOTH_OFF"
REACH_ARM = "D_BOTH_ON"
PATH_ALPHA_STAGE0 = 0.0
PATH_DECAY = 0.95
PATH_SOURCE = "invalidation"

_ZG = ZGoalStreamAccumulator()


# --------------------------------------------------------------------------- #
# Helpers (lineage-faithful, from V3-EXQ-1097)                                 #
# --------------------------------------------------------------------------- #
def _action_to_onehot(idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, idx] = 1.0
    return v


def _obs_harm(d):            return d.get("harm_obs")
def _obs_harm_a(d):          return d.get("harm_obs_a")
def _obs_harm_history(d):    return d.get("harm_history")


def _obs_accum(d) -> float:
    v = d.get("accumulated_harm")
    return float(v) if v is not None else 0.0


def _obs_resource_prox(d) -> float:
    rv = d.get("resource_field_view")
    if rv is None:
        return 0.0
    return float(rv.max().item()) if isinstance(rv, torch.Tensor) else float(np.max(rv))


def arm_config_slice(arm: str) -> Dict:
    """Everything the cell's computation reads. Never thresholds or labels."""
    a = ARMS[arm]
    return {
        "lineage": "v3_exq_475_sd036_decay_unlocks_exq471",
        "env_kwargs": dict(ENV_KWARGS),
        "schedule": {
            "warmup_episodes": WARMUP_EPISODES,
            "eval_episodes": EVAL_EPISODES,
            "steps_per_episode": STEPS_PER_EPISODE,
        },
        "dims": {"world": WORLD_DIM, "self": SELF_DIM,
                 "harm": HARM_DIM, "harm_a": HARM_A_DIM},
        "optim": {"lr_e1": LR_E1, "lr_e2_wf": LR_E2_WF,
                  "lr_e3_harm": LR_E3_HARM, "lr_enc_aux": LR_ENC_AUX,
                  "batch_size": BATCH_SIZE,
                  "wf_buf_max": WF_BUF_MAX,
                  "harm_eval_buf_max": HARM_EVAL_BUF_MAX},
        "substrate_operating": {
            "use_gabaergic_decay": True,
            "use_pag_freeze_gate": True,
            "use_per_stream_vs": True,
            "use_anchor_sets": True,
            "use_per_region_vs": True,
            "use_vs_commit_release": True,
            "commitment_threshold": 0.5,
            "beta_gate_bistable": True,
            "harm_descending_mod_enabled": True,
            "descending_attenuation_factor": 0.5,
        },
        "readout": {"lock_ratio_identity_tol": IDENTITY_TOL},
        "arm_flags": {
            "use_invalidation_trigger": a["trigger"],
            "use_staleness_accumulator": a["accumulator"],
            "use_mech284_hysteresis": a["accumulator"],
            "use_event_segmenter": a["segmenter"],
            "use_pag_descending_release": a["path_on"],
            "pag_descending_release_alpha": PATH_ALPHA_STAGE0,
            "pag_descending_release_decay": PATH_DECAY,
            "pag_descending_release_source": PATH_SOURCE,
        },
    }


def _make_agent_and_env(arm: str, seed: int) -> Tuple[REEAgent, CausalGridWorldV2]:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    env = CausalGridWorldV2(seed=seed, **ENV_KWARGS)
    a = ARMS[arm]
    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=SELF_DIM, world_dim=WORLD_DIM, harm_dim=HARM_DIM,
        alpha_world=0.9, alpha_self=0.3,
        reafference_action_dim=env.action_dim,
        use_harm_stream=True, z_harm_dim=HARM_DIM,
        use_affective_harm_stream=True, z_harm_a_dim=HARM_A_DIM,
        harm_history_len=HARM_HISTORY_LEN,
        use_resource_proximity_head=True, resource_proximity_weight=0.5,
        benefit_eval_enabled=True, benefit_weight=1.0,
        z_goal_enabled=True, goal_weight=0.5, drive_weight=2.0,
        e1_goal_conditioned=True,
        limb_damage_enabled=True, damage_increment=0.15,
        failure_prob_scale=0.3, heal_rate=0.002,
        # SD-036 / MECH-279 -- the V3-EXQ-475 comparator regime
        use_gabaergic_decay=True,
        use_pag_freeze_gate=True,
        # V_s substrate: held ON and identical in both arms
        use_per_stream_vs=True,
        use_anchor_sets=True,
        use_per_region_vs=True,
        use_vs_commit_release=True,
        # MECH-287 chain factors
        use_invalidation_trigger=a["trigger"],
        use_staleness_accumulator=a["accumulator"],
        use_mech284_hysteresis=a["accumulator"],
        use_event_segmenter=a["segmenter"],
        # MECH-287 option B path (alpha 0 -> behaviour bit-identical)
        use_pag_descending_release=a["path_on"],
        pag_descending_release_alpha=PATH_ALPHA_STAGE0,
        pag_descending_release_decay=PATH_DECAY,
        pag_descending_release_source=PATH_SOURCE,
    )
    config.e3.commitment_threshold = 0.5
    config.heartbeat.beta_gate_bistable = True
    config.harm_descending_mod_enabled = True
    config.descending_attenuation_factor = 0.5
    return REEAgent(config), env


# --------------------------------------------------------------------------- #
# Phase 0: warmup (V3-EXQ-475 protocol via V3-EXQ-1097, verbatim)              #
# --------------------------------------------------------------------------- #
def _warmup_train(agent, env, num_episodes: int, steps_per_episode: int,
                  total_eps: int, arm: str, seed: int) -> Dict:
    device = agent.device
    action_dim = env.action_dim

    e1_optimizer = optim.Adam(agent.e1.parameters(), lr=LR_E1)
    e2_wf_optimizer = optim.Adam(
        list(agent.e2.world_transition.parameters())
        + list(agent.e2.world_action_encoder.parameters()), lr=LR_E2_WF)
    harm_eval_optimizer = optim.Adam(agent.e3.harm_eval_head.parameters(), lr=LR_E3_HARM)
    aux_params = list(agent.latent_stack.parameters())
    aux_optimizer = optim.Adam(aux_params, lr=LR_ENC_AUX)

    wf_buf: List = []
    harm_eval_buf: List = []
    reward_log: List[float] = []
    ep_seconds: List[float] = []
    warm_freeze_active_steps = 0
    warm_steps = 0

    agent.train()
    for ep in range(num_episodes):
        t_ep = time.perf_counter()
        flat_obs, obs_dict = env.reset()
        agent.reset()
        z_world_prev = z_self_prev = action_prev = None
        ep_reward = 0.0

        for _ in range(steps_per_episode):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            obs_h, obs_h_a = _obs_harm(obs_dict), _obs_harm_a(obs_dict)
            obs_h_h = _obs_harm_history(obs_dict)
            prox_t, accum_t = _obs_resource_prox(obs_dict), _obs_accum(obs_dict)

            latent = agent.sense(obs_body, obs_world, obs_harm=obs_h,
                                 obs_harm_a=obs_h_a, obs_harm_history=obs_h_h)
            z_world_curr = latent.z_world.detach()

            aux_terms = []
            prox_loss = agent.compute_resource_proximity_loss(
                torch.tensor([[prox_t]], device=device), latent)
            if prox_loss is not None and prox_loss.requires_grad:
                aux_terms.append(prox_loss)
            harm_accum_loss = agent.compute_harm_accum_loss(
                torch.tensor([[accum_t]], device=device), latent)
            if harm_accum_loss is not None and harm_accum_loss.requires_grad:
                aux_terms.append(harm_accum_loss)
            if aux_terms:
                aux_loss = sum(aux_terms)
                aux_optimizer.zero_grad()
                aux_loss.backward(retain_graph=False)
                torch.nn.utils.clip_grad_norm_(aux_params, 1.0)
                aux_optimizer.step()

            if z_self_prev is not None and action_prev is not None:
                agent.record_transition(z_self_prev, action_prev, latent.z_self.detach())

            latent = agent.sense(obs_body, obs_world, obs_harm=obs_h,
                                 obs_harm_a=obs_h_a, obs_harm_history=obs_h_h)
            ticks = agent.clock.advance()
            e1_prior = (agent._e1_tick(latent) if ticks.get("e1_tick", False)
                        else torch.zeros(1, WORLD_DIM, device=device))
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)

            agent.update_z_goal(
                benefit_exposure=max(0.0, float(obs_dict.get("benefit_exposure", 0.0))),
                drive_level=REEAgent.compute_drive_level(obs_body))

            action = agent.select_action(candidates, ticks, temperature=1.0)
            if action is None:
                action = _action_to_onehot(random.randint(0, action_dim - 1),
                                           action_dim, device)
                agent._last_action = action

            if agent.pag_freeze_gate is not None:
                if agent.pag_freeze_gate.last_output.freeze_active:
                    warm_freeze_active_steps += 1
            warm_steps += 1

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)
            ep_reward += float(harm_signal)

            if z_world_prev is not None and action_prev is not None:
                wf_buf.append((z_world_prev.cpu(), action_prev.cpu(), z_world_curr.cpu()))
                if len(wf_buf) > WF_BUF_MAX:
                    wf_buf = wf_buf[-WF_BUF_MAX:]

            harm_target = abs(float(harm_signal)) if float(harm_signal) < 0 else 0.0
            harm_eval_buf.append((z_world_curr.cpu(), torch.tensor([harm_target])))
            if len(harm_eval_buf) > HARM_EVAL_BUF_MAX:
                harm_eval_buf = harm_eval_buf[-HARM_EVAL_BUF_MAX:]

            if len(wf_buf) >= BATCH_SIZE:
                idxs = torch.randperm(len(wf_buf))[:BATCH_SIZE].tolist()
                zw_b = torch.cat([wf_buf[i][0] for i in idxs]).to(device)
                a_b = torch.cat([wf_buf[i][1] for i in idxs]).to(device)
                zw1_b = torch.cat([wf_buf[i][2] for i in idxs]).to(device)
                wf_pred = agent.e2.world_forward(zw_b, a_b)
                wf_loss = F.mse_loss(wf_pred, zw1_b)
                if wf_loss.requires_grad:
                    e2_wf_optimizer.zero_grad()
                    wf_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(agent.e2.world_transition.parameters())
                        + list(agent.e2.world_action_encoder.parameters()), 1.0)
                    e2_wf_optimizer.step()
                with torch.no_grad():
                    agent.e3.update_running_variance((wf_pred.detach() - zw1_b).detach())

            if len(harm_eval_buf) >= BATCH_SIZE:
                idxs = torch.randperm(len(harm_eval_buf))[:BATCH_SIZE].tolist()
                zw_b = torch.cat([harm_eval_buf[i][0] for i in idxs]).to(device)
                ht_b = torch.cat([harm_eval_buf[i][1] for i in idxs]).to(device)
                hp = agent.e3.harm_eval(zw_b)
                he_loss = F.mse_loss(hp.squeeze(), ht_b.squeeze())
                if he_loss.requires_grad:
                    harm_eval_optimizer.zero_grad()
                    he_loss.backward()
                    harm_eval_optimizer.step()

            if len(agent._world_experience_buffer) >= 2:
                e1_loss = agent.compute_prediction_loss()
                if e1_loss.requires_grad:
                    e1_optimizer.zero_grad()
                    e1_loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.e1.parameters(), 1.0)
                    e1_optimizer.step()

            z_world_prev = z_world_curr
            z_self_prev = latent.z_self.detach()
            action_prev = action.detach()
            if done:
                break

        reward_log.append(ep_reward)
        ep_seconds.append(time.perf_counter() - t_ep)
        if (ep + 1) % 5 == 0 or ep == num_episodes - 1:
            print(f"  [train] arm={arm} seed={seed} ep {ep+1}/{total_eps}"
                  f"  rv={agent.e3._running_variance:.4f}"
                  f"  ep_reward={ep_reward:.4f}"
                  f"  ep_sec={ep_seconds[-1]:.1f}", flush=True)

    first10 = float(np.mean(reward_log[:10])) if len(reward_log) >= 10 else float(np.mean(reward_log))
    last10 = float(np.mean(reward_log[-10:])) if len(reward_log) >= 10 else float(np.mean(reward_log))
    return {
        "final_running_variance": float(agent.e3._running_variance),
        "first10_reward": first10,
        "last10_reward": last10,
        "per_episode_seconds": [float(s) for s in ep_seconds],
        "mean_episode_seconds": float(np.mean(ep_seconds)) if ep_seconds else float("nan"),
        "freeze_active_fraction": (float(warm_freeze_active_steps) / warm_steps
                                   if warm_steps else float("nan")),
        "pag_desc_diag_end": agent.pag_descending_release_diagnostics(),
    }


# --------------------------------------------------------------------------- #
# Phase 1: eval with the Stage-0 instruments                                   #
# --------------------------------------------------------------------------- #
def _eval_agent(agent, env, num_episodes: int, steps_per_episode: int,
                warmup_eps: int, total_eps: int, arm: str, seed: int) -> Dict:
    action_dim = env.action_dim
    device = agent.device
    gate = agent.pag_freeze_gate

    desc0 = agent.pag_descending_release_diagnostics()
    diag0 = dict(gate.diagnostics) if gate is not None else {}

    freeze_active_steps = 0
    n_steps = 0
    n_gate_ticks = 0
    n_gate_ticks_frozen = 0
    lock_ratios: List[float] = []
    identity_violations = 0
    per_episode_freeze_frac: List[float] = []
    episodes_ending_frozen = 0
    per_episode_frac_after_entry: List[float] = []
    first_freeze_step: List[Optional[int]] = []
    first_release_step: List[Optional[int]] = []
    episode_rewards: List[float] = []
    prev_ticks = int(diag0.get("n_ticks", 0))

    agent.eval()
    for ep_idx in range(num_episodes):
        t_ep = time.perf_counter()
        flat_obs, obs_dict = env.reset()
        agent.reset()
        z_self_prev = action_prev = None
        ep_reward = 0.0
        ep_frozen = 0
        ep_steps = 0
        ep_first_release: Optional[int] = None
        ep_first_freeze: Optional[int] = None
        last_active = False
        prev_ticks = int(gate.diagnostics.get("n_ticks", 0)) if gate is not None else 0

        for step in range(steps_per_episode):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            obs_h, obs_h_a = _obs_harm(obs_dict), _obs_harm_a(obs_dict)
            obs_h_h = _obs_harm_history(obs_dict)

            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world, obs_harm=obs_h,
                                     obs_harm_a=obs_h_a, obs_harm_history=obs_h_h)
                if z_self_prev is not None and action_prev is not None:
                    agent.record_transition(z_self_prev, action_prev,
                                            latent.z_self.detach())
                ticks = agent.clock.advance()
                e1_prior = (agent._e1_tick(latent) if ticks.get("e1_tick", False)
                            else torch.zeros(1, WORLD_DIM, device=device))
                candidates = agent.generate_trajectories(latent, e1_prior, ticks)
                agent.update_z_goal(
                    benefit_exposure=max(0.0, float(obs_dict.get("benefit_exposure", 0.0))),
                    drive_level=REEAgent.compute_drive_level(obs_body))
                action = agent.select_action(candidates, ticks, temperature=1.0)
                if action is None:
                    action = _action_to_onehot(random.randint(0, action_dim - 1),
                                               action_dim, device)
                    agent._last_action = action

            if gate is not None:
                out = gate.last_output
                now_ticks = int(gate.diagnostics.get("n_ticks", 0))
                fresh = now_ticks > prev_ticks
                prev_ticks = now_ticks
                if out.freeze_active:
                    freeze_active_steps += 1
                    ep_frozen += 1
                    if ep_first_freeze is None:
                        ep_first_freeze = step
                if fresh:
                    n_gate_ticks += 1
                    if out.freeze_release and ep_first_release is None:
                        ep_first_release = step
                    if out.freeze_active:
                        n_gate_ticks_frozen += 1
                        # identity: exit_threshold == theta * gaba_tone here
                        tone = (float(agent.gabaergic_decay.gaba_tone)
                                if agent.gabaergic_decay is not None else 1.0)
                        expected = float(gate.config.theta_freeze) * max(0.0, tone)
                        if abs(float(out.exit_threshold) - expected) > IDENTITY_TOL * max(1.0, expected):
                            identity_violations += 1
                        if float(out.exit_threshold) > 0.0:
                            lock_ratios.append(float(out.z_harm_a_norm)
                                               / float(out.exit_threshold))
                last_active = bool(out.freeze_active)
            n_steps += 1
            ep_steps += 1

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)
            ep_reward += float(harm_signal)
            z_self_prev = latent.z_self.detach()
            action_prev = action.detach()
            if done:
                break

        episode_rewards.append(ep_reward)
        per_episode_freeze_frac.append(float(ep_frozen) / ep_steps if ep_steps else float("nan"))
        # red-team F2: separate lock ENTRY latency from lock PERSISTENCE
        if ep_first_freeze is not None and ep_steps > ep_first_freeze:
            per_episode_frac_after_entry.append(float(ep_frozen) / (ep_steps - ep_first_freeze))
        first_freeze_step.append(ep_first_freeze)
        if last_active:
            episodes_ending_frozen += 1
        first_release_step.append(ep_first_release)
        print(f"  [train] arm={arm} seed={seed} ep {warmup_eps+ep_idx+1}/{total_eps}"
              f"  eval_freeze_frac={per_episode_freeze_frac[-1]:.3f}"
              f"  ep_sec={time.perf_counter()-t_ep:.1f}", flush=True)

    desc1 = agent.pag_descending_release_diagnostics()
    diag1 = dict(gate.diagnostics) if gate is not None else {}
    hip = agent.hippocampal
    trig = getattr(hip, "invalidation_trigger", None) if hip is not None else None

    def _d(k):
        return int(desc1.get(k, 0)) - int(desc0.get(k, 0))

    lr = np.array(lock_ratios, dtype=float)
    censored = sum(1 for s in first_release_step if s is None)
    return {
        "arm": arm,
        "seed": seed,
        "n_eval_steps": int(n_steps),
        "freeze_active_steps": int(freeze_active_steps),
        "eval_freeze_active_fraction": (float(freeze_active_steps) / n_steps
                                        if n_steps else float("nan")),
        "per_episode_freeze_active_fraction": per_episode_freeze_frac,
        "n_gate_ticks": int(n_gate_ticks),
        "n_gate_ticks_frozen": int(n_gate_ticks_frozen),
        "gate_tick_freeze_fraction": (float(n_gate_ticks_frozen) / n_gate_ticks
                                      if n_gate_ticks else float("nan")),
        "lock_ratios": [float(x) for x in lock_ratios],
        "lock_ratio_median": float(np.median(lr)) if lr.size else float("nan"),
        "lock_ratio_p25": float(np.percentile(lr, 25)) if lr.size else float("nan"),
        "lock_ratio_p75": float(np.percentile(lr, 75)) if lr.size else float("nan"),
        "lock_ratio_identity_violations": int(identity_violations),
        "episodes_ending_frozen": int(episodes_ending_frozen),
        "lock_persistence": float(episodes_ending_frozen) / max(1, num_episodes),
        "per_episode_freeze_fraction_after_entry": per_episode_frac_after_entry,
        "freeze_fraction_after_entry": (float(np.mean(per_episode_frac_after_entry))
                                        if per_episode_frac_after_entry else float("nan")),
        "first_freeze_step": [s for s in first_freeze_step],
        "episodes_never_frozen": int(sum(1 for s in first_freeze_step if s is None)),
        "time_to_first_release_steps": [s for s in first_release_step],
        "time_to_first_release_censored": int(censored),
        "pag_eval_commits": int(diag1.get("n_commits", 0)) - int(diag0.get("n_commits", 0)),
        "pag_eval_releases": int(diag1.get("n_releases", 0)) - int(diag0.get("n_releases", 0)),
        "desc_enabled": bool(desc1.get("enabled", False)),
        "desc_alpha": float(desc1.get("alpha", 0.0)),
        "eval_drive_steps": _d("n_drive_steps"),
        "eval_drive_steps_while_frozen": _d("n_drive_steps_while_frozen"),
        "eval_t3_events": _d("n_t3_events"),
        "eval_h_events": _d("n_h_events"),
        "n_broadcast_last_episode": int(trig.get_stats().get("n_broadcast", 0)) if trig is not None else 0,
        "vs_commit_release_count": int(getattr(agent, "_vs_commit_release_count", 0)),
        "mean_eval_reward": float(np.mean(episode_rewards)) if episode_rewards else float("nan"),
    }


# --------------------------------------------------------------------------- #
def run_cell(arm: str, seed: int, warmup_eps: int, eval_eps: int, steps: int) -> Dict:
    print(f"\nSeed {seed} Condition {arm}", flush=True)
    total = warmup_eps + eval_eps
    t_cell = time.perf_counter()
    with arm_cell(seed, config_slice=arm_config_slice(arm),
                  script_path=Path(__file__),
                  include_driver_script_in_hash=False) as cell:
        agent, env = _make_agent_and_env(arm, seed)
        warm = _warmup_train(agent, env, warmup_eps, steps, total, arm, seed)
        row = _eval_agent(agent, env, eval_eps, steps, warmup_eps, total, arm, seed)
        row.update({f"warmup_{k}": v for k, v in warm.items()})
        row["cell_seconds"] = float(time.perf_counter() - t_cell)
        _ZG.observe(agent)
        cell.stamp(row)
    print(f"  arm={arm} seed={seed} eval_freeze_frac={row['eval_freeze_active_fraction']:.4f}"
          f" frozen_gate_ticks={row['n_gate_ticks_frozen']}"
          f" lock_ratio_median={row['lock_ratio_median']:.4f}"
          f" drive_while_frozen={row['eval_drive_steps_while_frozen']}"
          f" warm_ep_sec={row['warmup_mean_episode_seconds']:.1f}", flush=True)
    return row


def _finite(x) -> bool:
    return bool(isinstance(x, (int, float)) and not isinstance(x, bool)
                and np.isfinite(x))


def choose_alpha(median_ratio: float) -> Optional[float]:
    """Pre-registered rule. None = INERT-BY-MAGNITUDE (median > 5)."""
    if not _finite(median_ratio) or median_ratio > LOCK_RATIO_INERT_ABOVE:
        return None
    for a in ALPHA_LADDER:
        if 1.0 + a >= median_ratio:
            return a
    return None


def analyse(rows: List[Dict]) -> Dict:
    comp = sorted([r for r in rows if r["arm"] == COMPARATOR_ARM], key=lambda r: r["seed"])
    reach = sorted([r for r in rows if r["arm"] == REACH_ARM], key=lambda r: r["seed"])

    # readiness: the gate ticked in eval at all (instrument live)
    all_ticks = [r["n_gate_ticks"] for r in rows]
    worst_ticks = min(all_ticks) if all_ticks else 0
    worst_ticks_cell = None
    for r in rows:
        if r["n_gate_ticks"] == worst_ticks:
            worst_ticks_cell = f"{r['arm']}/seed={r['seed']}"
            break
    gate_live = worst_ticks > 0

    # S0-1
    fracs = [r["eval_freeze_active_fraction"] for r in comp]
    s01_seeds = sum(1 for f in fracs if _finite(f) and f >= LOCK_FREEZE_FRACTION_FLOOR)
    s01 = s01_seeds >= SEED_MAJORITY
    fin = sorted([f for f in fracs if _finite(f)], reverse=True)
    # the 2nd-best seed is the statistic the ">= 2 of 3" rule turns on
    s01_measured = fin[SEED_MAJORITY - 1] if len(fin) >= SEED_MAJORITY else float("nan")

    # S0-2
    pooled = [x for r in comp for x in r["lock_ratios"]]
    n_pooled = len(pooled)
    median_ratio = float(np.median(pooled)) if pooled else float("nan")
    s02_determinable = n_pooled >= MIN_FROZEN_GATE_TICKS
    alpha = choose_alpha(median_ratio) if s02_determinable else None
    s02 = bool(s02_determinable and alpha is not None)
    identity_bad = sum(r["lock_ratio_identity_violations"] for r in rows)

    # S0-3
    reach_vals = [r["eval_drive_steps_while_frozen"] for r in reach]
    s03_seeds = sum(1 for v in reach_vals if v >= REACH_FLOOR)
    s03 = s03_seeds >= SEED_MAJORITY
    rs = sorted(reach_vals, reverse=True)
    s03_measured = float(rs[SEED_MAJORITY - 1]) if len(rs) >= SEED_MAJORITY else float("nan")
    reach_path_live = all(r["desc_enabled"] and r["desc_alpha"] == 0.0 for r in reach) and bool(reach)
    # red-team F1: S0-3's counter only increments while frozen, so its
    # non-degeneracy needs the REACH arm to freeze in eval -- not A's lock.
    reach_frozen_seeds = sum(1 for r in reach if r["n_gate_ticks_frozen"] >= 1)
    reach_arm_frozen = reach_frozen_seeds >= SEED_MAJORITY
    reach_fracs = sorted([r["eval_freeze_active_fraction"] for r in reach
                          if _finite(r["eval_freeze_active_fraction"])], reverse=True)
    reach_lock_2nd = (reach_fracs[SEED_MAJORITY - 1]
                      if len(reach_fracs) >= SEED_MAJORITY else float("nan"))
    reach_lock_ok = _finite(reach_lock_2nd) and reach_lock_2nd >= LOCK_FREEZE_FRACTION_FLOOR
    # red-team F2: late lock ENTRY vs no lock
    persist_seeds = sum(1 for r in comp if r["lock_persistence"] >= LOCK_FREEZE_FRACTION_FLOOR)
    late_entry_lock = persist_seeds >= SEED_MAJORITY

    # S0-4 budget
    warm_sec = [r["warmup_mean_episode_seconds"] for r in rows if _finite(r["warmup_mean_episode_seconds"])]
    budget = float(max(warm_sec)) if warm_sec else float("nan")

    overall = bool(gate_live and s01 and s02 and s03)

    if not gate_live:
        label = "substrate_not_ready_requeue"
    elif not s01 and late_entry_lock:
        label = "precondition_failed_no_lock_comparator_late_entry"
    elif not s01:
        label = "precondition_failed_no_lock_comparator"
    elif not s02_determinable:
        label = "lock_ratio_cannot_determine"
    elif alpha is None:
        label = "inert_by_magnitude"
    elif not s03 and not reach_arm_frozen:
        label = "reach_arm_never_frozen_in_eval"
    elif not s03:
        label = "path_built_not_reached_in_regime"
    elif not reach_lock_ok:
        label = "stage0_pass_reach_arm_lock_weak"
    else:
        label = "stage0_pass"

    criteria = [
        {"name": "S0_1_lock_reproduces", "load_bearing": True, "passed": s01,
         "measured": s01_measured, "threshold": LOCK_FREEZE_FRACTION_FLOOR,
         "direction": "lower", "seeds_clearing": s01_seeds,
         "seeds_required": SEED_MAJORITY,
         "per_seed": {str(r["seed"]): r["eval_freeze_active_fraction"] for r in comp},
         "detail": "A_BOTH_OFF eval freeze-active fraction (per env step) >= 0.80 "
                   "on >= 2 of 3 seeds; measured = 2nd-best seed"},
        {"name": "S0_2_lock_ratio_bounded", "load_bearing": True, "passed": s02,
         "measured": median_ratio, "threshold": LOCK_RATIO_INERT_ABOVE,
         "direction": "upper", "n_frozen_gate_ticks_pooled": n_pooled,
         "min_frozen_gate_ticks": MIN_FROZEN_GATE_TICKS,
         "alpha_selected": alpha,
         "detail": "median of z_harm_a_norm/(theta*gaba_tone) on fresh frozen gate "
                   "ticks, A_BOTH_OFF pooled over seeds; alpha = smallest of "
                   "{1,2,4} with 1+alpha >= median; median > 5 -> inert"},
        {"name": "S0_3_reach_in_regime", "load_bearing": True, "passed": s03,
         "measured": s03_measured, "threshold": float(REACH_FLOOR),
         "direction": "lower", "seeds_clearing": s03_seeds,
         "seeds_required": SEED_MAJORITY,
         "per_seed": {str(r["seed"]): r["eval_drive_steps_while_frozen"] for r in reach},
         "detail": "D_BOTH_ON, path ON alpha 0: eval delta of "
                   "_pag_desc_n_drive_steps_while_frozen >= 1 on >= 2 of 3 seeds; "
                   "measured = 2nd-best seed"},
        {"name": "S0_5_reach_arm_lock_headroom", "load_bearing": False,
         "passed": bool(reach_lock_ok), "measured": reach_lock_2nd,
         "threshold": LOCK_FREEZE_FRACTION_FLOOR, "direction": "lower",
         "detail": "NOT pre-registered (red-team F1, added pre-run): D_BOTH_ON eval "
                   "freeze-active fraction, 2nd-best seed. Stage 1's P1 lives in "
                   "D_BOTH_ON; below 0.80 PASS is sub-labelled "
                   "stage0_pass_reach_arm_lock_weak. Does not change PASS."},
        {"name": "S0_4_budget_warmup_episode_seconds", "load_bearing": False,
         "passed": _finite(budget), "measured": budget,
         "threshold_not_applicable": "informational sizing measurement for Stage 1",
         "detail": "worst-cell mean wall-clock seconds per warmup episode"},
    ]

    preconditions = [
        {"name": "pag_gate_ticked_in_eval",
         "description": "every cell's PAG freeze gate ticked at least once in eval "
                        "(the instrument all three criteria read is live)",
         "measured": float(worst_ticks), "threshold": 1.0, "direction": "lower",
         "offending_cell": worst_ticks_cell,
         "control": "the gate ticks on every E3 tick when use_pag_freeze_gate is on",
         "met": gate_live},
        {"name": "reach_arm_path_constructed_at_alpha0",
         "description": "D_BOTH_ON reports the descending-release path enabled with "
                        "alpha 0 (so S0-3 reads a live counter at bit-identical behaviour)",
         "measured": 1.0 if reach_path_live else 0.0, "threshold": 1.0,
         "direction": "lower",
         "control": "use_pag_descending_release=True, pag_descending_release_alpha=0.0",
         "met": reach_path_live},
        {"name": "lock_ratio_identity_holds",
         "description": "exit_threshold == theta_freeze*gaba_tone on every frozen tick "
                        "(override 0, alpha 0), so the ratio is the pre-registered one",
         "measured": float(identity_bad), "threshold": 0.0, "direction": "upper",
         "control": "SD-037 not constructed; alpha_descending 0 (a consistency "
                    "check, tautological at this config -- not evidence)",
         "met": identity_bad == 0},
        {"name": "reach_arm_frozen_in_eval",
         "description": "D_BOTH_ON froze (>= 1 frozen gate tick) in eval on >= 2 of 3 "
                        "seeds; S0-3's counter increments only while frozen, so a "
                        "zero with this unmet reads 'never frozen', not 'not reached'",
         "measured": float(reach_frozen_seeds), "threshold": float(SEED_MAJORITY),
         "direction": "lower",
         "control": "red-team F1 (added pre-run)",
         "met": bool(reach_arm_frozen)},
    ]

    criteria_non_degenerate = {
        "S0_1_lock_reproduces": bool(gate_live and len(fin) == len(comp) and comp),
        "S0_2_lock_ratio_bounded": bool(s02_determinable),
        "S0_3_reach_in_regime": bool(reach_path_live and reach_arm_frozen),
    }

    per_seed_alpha = {}
    for r in comp:
        per_seed_alpha[str(r["seed"])] = choose_alpha(r["lock_ratio_median"])

    return {
        "outcome": "PASS" if overall else "FAIL",
        "label": label,
        "criteria": criteria,
        "combination_rule": ("PASS = pag_gate_ticked_in_eval AND S0_1 AND S0_2 AND S0_3 "
                             "(plain AND); S0_4 informational"),
        "preconditions": preconditions,
        "criteria_non_degenerate": criteria_non_degenerate,
        "alpha_selected": alpha,
        "median_lock_ratio": median_ratio,
        "per_seed_alpha_if_rule_applied": per_seed_alpha,
        "budget_worst_warmup_episode_seconds": budget,
    }


def _flat_scalar(rows: List[Dict], ana: Dict) -> Dict:
    out: Dict[str, float] = {}
    for c in ana["criteria"]:
        out[f"{c['name']}__passed"] = int(bool(c["passed"]))
        for k in ("measured", "threshold"):
            v = c.get(k)
            if _finite(v):
                out[f"{c['name']}__{k}"] = float(v)
    for p in ana["preconditions"]:
        out[f"precondition__{p['name']}__met"] = int(bool(p["met"]))
    if _finite(ana["alpha_selected"]):
        out["alpha_selected"] = float(ana["alpha_selected"])
    if _finite(ana["median_lock_ratio"]):
        out["median_lock_ratio_comparator"] = float(ana["median_lock_ratio"])
    for r in rows:
        key = f"{r['arm']}__seed{r['seed']}"
        for f in ("eval_freeze_active_fraction", "lock_ratio_median",
                  "freeze_fraction_after_entry", "eval_h_events",
                  "eval_drive_steps_while_frozen", "lock_persistence",
                  "warmup_mean_episode_seconds", "warmup_freeze_active_fraction"):
            v = r.get(f)
            if _finite(v):
                out[f"{f}__{key}"] = float(v)
    return out


# --------------------------------------------------------------------------- #
def run(dry_run: bool = False):
    t0 = time.perf_counter()
    warmup_eps = 2 if dry_run else WARMUP_EPISODES
    eval_eps = 2 if dry_run else EVAL_EPISODES
    steps = 30 if dry_run else STEPS_PER_EPISODE
    seeds = SEEDS[:2] if dry_run else SEEDS

    print(f"{EXPERIMENT_TYPE} -- MECH-287 option B Stage-0 precondition gate", flush=True)
    print(f"Arms: {list(ARMS)}  Seeds: {seeds}  warmup={warmup_eps}"
          f"  eval={eval_eps}  steps/ep={steps}", flush=True)

    rows: List[Dict] = []
    # comparator first: S0-1 is readable from the first three cells
    for arm in ARMS:
        for seed in seeds:
            row = run_cell(arm, seed, warmup_eps, eval_eps, steps)
            rows.append(row)
            print(f"verdict: {'PASS' if row['n_gate_ticks'] > 0 else 'FAIL'}", flush=True)

    ana = analyse(rows)
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    manifest = {
        "run_id": f"{EXPERIMENT_TYPE}_{ts}_v3",
        "queue_id": QUEUE_ID,
        "experiment_type": EXPERIMENT_TYPE,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "claim_ids": CLAIM_IDS,
        "timestamp_utc": ts,
        "outcome": ana["outcome"],
        "evidence_direction": "non_contributory",
        "evidence_direction_per_claim": {"MECH-287": "non_contributory"},
        "criteria": ana["criteria"],
        "combination_rule": ana["combination_rule"],
        "interpretation": {
            "label": ana["label"],
            "preconditions": ana["preconditions"],
            "criteria_non_degenerate": ana["criteria_non_degenerate"],
            "routing": {
                "precondition_failed_no_lock_comparator":
                    "lock DV has no comparator on STAY substrate; MECH-287 lock "
                    "falsifier routes back to the user (option A or C of the "
                    "2026-09-25 decision). Not a falsification.",
                "inert_by_magnitude":
                    "median lock ratio > 5: path cannot reach the lock at a bounded "
                    "gain; Stage 1 does not run.",
                "precondition_failed_no_lock_comparator_late_entry":
                    "S0-1 per-step fraction < 0.80 but lock_persistence >= 0.80 on >= 2 "
                    "seeds: the lock exists once entered, entry is late. Same route "
                    "(user decision) but the fraction-after-entry readout is the lead.",
                "reach_arm_never_frozen_in_eval":
                    "D_BOTH_ON did not freeze in eval; S0-3 zero is uninformative "
                    "about reach. Not a falsification.",
                "stage0_pass_reach_arm_lock_weak":
                    "Stage 0 passed as pre-registered, but D_BOTH_ON's lock is below "
                    "0.80 so Stage 1's P1 may lack headroom; flag before queuing Stage 1.",
                "path_built_not_reached_in_regime":
                    "no invalidation while frozen in eval; built but not reached. "
                    "Not a falsification.",
                "stage0_pass": "queue Stage 1 (2x2x3, path eval-only ON at alpha_selected).",
                "lock_ratio_cannot_determine":
                    "fewer than MIN_FROZEN_GATE_TICKS frozen gate ticks pooled.",
                "substrate_not_ready_requeue": "PAG gate never ticked in eval.",
            },
        },
        "alpha_selected": ana["alpha_selected"],
        "median_lock_ratio_comparator": ana["median_lock_ratio"],
        "per_seed_alpha_if_rule_applied": ana["per_seed_alpha_if_rule_applied"],
        "budget_worst_warmup_episode_seconds": ana["budget_worst_warmup_episode_seconds"],
        "arm_results": rows,
        "per_seed_results": rows,
        "sleep_driver_pattern": "not_applicable",
        "design_doc": ("REE_assembly/evidence/planning/"
                       "mech287_anchor_freeze_exit_design_20260925.md section 6 Stage 0"),
        "threshold_derivation": {
            "LOCK_FREEZE_FRACTION_FLOOR": "design section 6 Stage 0 item 1 (0.80; 475 ref 1.0 under UP defect)",
            "ALPHA_LADDER": "design section 6 Stage 0 item 2 ({1,2,4}, 1+alpha >= median)",
            "LOCK_RATIO_INERT_ABOVE": "design section 6 Stage 0 item 2 (median > 5 -> inert)",
            "REACH_FLOOR": "design section 6 Stage 0 item 3 (> 0)",
            "MIN_FROZEN_GATE_TICKS": "driver-level cannot-determine floor for a median (10)",
        },
        "ethics_preflight": {
            "involves_negative_valence": False, "involves_suffering_like_state": False,
            "involves_self_model": False, "involves_inescapability_or_helplessness": False,
            "involves_offline_replay_over_harm": False,
            "involves_social_mind_or_language": False,
            "involves_human_data_or_clinical_context": False, "decision": "allow",
        },
        "known_substrate_limitations": [
            "mech005-betagate-decommit-counter-and-commit-ceiling (degrading, open)",
            "SD-091 / mech290 / suffering-derivative-comparator-refractory (degrading, open; agent.select_action)",
            "SD-018 / SD-106 (degrading, open): resource-proximity head is armed here.",
        ],
    }
    manifest["readout"] = _flat_scalar(rows, ana)
    full_config = {
        "env_kwargs": ENV_KWARGS,
        "arms": {k: dict(v) for k, v in ARMS.items()},
        "warmup_episodes": warmup_eps,
        "eval_episodes": eval_eps,
        "steps_per_episode": steps,
        "dims": {"world": WORLD_DIM, "self": SELF_DIM, "harm": HARM_DIM, "harm_a": HARM_A_DIM},
        "substrate_operating": arm_config_slice(COMPARATOR_ARM)["substrate_operating"],
        "path": {"alpha_stage0": PATH_ALPHA_STAGE0, "decay": PATH_DECAY, "source": PATH_SOURCE},
        "dry_run": bool(dry_run),
    }
    return manifest, full_config, seeds, t0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    manifest, full_config, _seeds, _t0 = run(dry_run=args.dry_run)
    out_path = write_flat_manifest(
        manifest,
        dry_run=args.dry_run,
        config=full_config,
        seeds=_seeds,
        script_path=Path(__file__),
        started_at=_t0,
        z_goal_stream_stats=_ZG.stats(),
    )
    print(f"\noutcome: {manifest['outcome']}", flush=True)
    print(f"label: {manifest['interpretation']['label']}", flush=True)
    print(f"alpha_selected: {manifest['alpha_selected']}", flush=True)
    for c in manifest["criteria"]:
        print(f"  {c['name']}: passed={c['passed']} measured={c['measured']}"
              f" threshold={c.get('threshold')}", flush=True)
    print(f"manifest: {out_path}", flush=True)

    _o = str(manifest["outcome"]).upper()
    emit_outcome(outcome=_o if _o in ("PASS", "FAIL") else "FAIL",
                 manifest_path=str(out_path), dry_run=args.dry_run)
