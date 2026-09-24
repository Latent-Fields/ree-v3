#!/opt/local/bin/python3
"""
V3-EXQ-1097 -- MECH-287 four-arm trigger-vs-accumulator invalidation factorial.

Claims: MECH-287 (primary), MECH-284 (coupled accumulator half)

This is the four-arm dissociation MECH-287's evidence_quality_note names as the
outstanding promote-to-active gate (historically "V3-EXQ-476"), which has never
run on the Phase-3 substrate.

PURPOSE
-------
MECH-287 asserts a broadcast "anchor may be wrong" trigger that drives MECH-284
staleness accumulation over the active anchor set, and thence a MECH-269 anchor
reset. Its secondary falsifiable is a 2x2 dissociation:

    trigger-loss     -> single-event blindness (no broadcasts at all)
    accumulator-loss -> V3-EXQ-475's phenotype (broadcasts that never integrate)
    both-lesioned    -> rigid perseveration (the EXQ-471/475 catatonic lock)
    both-intact      -> the lock is released

ARMS (5 cells x 3 seeds; use_vs_commit_release held ON and IDENTICAL in all)
---------------------------------------------------------------------------
    A_BOTH_OFF   trigger OFF, accumulator OFF, segmenter ON   <- comparator
    B_TRIG_ONLY  trigger ON,  accumulator OFF, segmenter ON
    C_ACC_ONLY   trigger OFF, accumulator ON,  segmenter ON
    D_BOTH_ON    trigger ON,  accumulator ON,  segmenter ON
    E_SEG_OFF    trigger ON,  accumulator ON,  segmenter OFF  <- verdict-3 control

use_staleness_accumulator and use_mech284_hysteresis move together as the single
"MECH-284 online readout" factor, exactly as the claim's manipulation specifies.

WHY use_vs_commit_release IS ON IN EVERY ARM (user decision, 2026-09-24, Option A)
---------------------------------------------------------------------------------
MECH-287's DV is behavioural, but its registered flag set contains no read-side
consumer: the ONLY path from anchor state to behaviour in the agent is the
commit-release hook (agent.py:7375), gated on use_vs_commit_release, whose own
docstring (config.py:3121 block) states "with flag off, EXQ-478/480
wired-but-inert behaviour reproduces". The proposer never consults anchors
(module.py:2037; its anchor branch is MECH-293 ghost seeding, gated on
use_mech292_ghost_bank / use_mech293_ghost_probes, both default False and both
OFF here). With the read-side hook off, every arm is behaviourally identical by
construction and the run would satisfy MECH-287's FALSIFYING clause as an
artifact.

The flag is therefore held ON and IDENTICAL across all five arms, so it cannot
confound the trigger x accumulator contrast. CONSEQUENCE FOR INTERPRETATION: a
positive result supports MECH-287 + MECH-284 + the MECH-269/MECH-090 read-side
release hook JOINTLY, not MECH-287 alone. In-tree precedent for arming it:
V3-EXQ-490b / 490c / 490e / 490f / 596 / 601.
Analysis: REE_assembly evidence/planning/mech287_four_arm_factorial_readside_gap_20260924.md

COMPARATOR CONFIG (pre-flight NAMED CHANGE, orchestrate-20260924-b)
-------------------------------------------------------------------
Built on V3-EXQ-475's config (SD-036 GABAergic decay + MECH-279 PAG freeze gate
ON, 60 warmup episodes), NOT V3-EXQ-478's. 475 measured pag_n_commits 71/70/64
against pag_n_releases 6/5/5 -- ~12.9 re-commits per release, the catatonic-lock
regime with real dynamic range. 478 ran with no PAG gate, no decay and no warmup
training; its freeze_recommit_count sat at 1 in all four cells with
action_class_entropy 0.0 -- a saturated ceiling misread as a floor, and a
different quantity from this claim's DV.

RED-TEAM: see the Step 4.5 line at the end of this docstring.

SLEEP DRIVER: not applicable (no sleep flags set).
"""

from __future__ import annotations

import argparse
import json
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
from experiments._lib.manifest_core import stamp_recording_core

try:
    from _manifest import write_manifest
except Exception:  # pragma: no cover
    def write_manifest(path, manifest):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(manifest, fh, indent=2)


EXPERIMENT_TYPE = "v3_exq_1097_mech287_invalidation_trigger_accumulator_factorial"
EXPERIMENT_PURPOSE = "evidence"
CLAIM_IDS = ["MECH-287", "MECH-284"]
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"

# --------------------------------------------------------------------------- #
# Pre-registered thresholds (constants -- never derived from this run's stats)  #
# --------------------------------------------------------------------------- #
# Derivation of every number here is in the module docstring of the analysis
# doc and restated in `threshold_derivation` in the manifest.
#
# V3-EXQ-475 measured re-commits per PAG release, per seed:
#     71/6 = 11.8333, 70/5 = 14.0000, 64/5 = 12.8000
#     mean 12.8778, sample SD 1.0854, DV floor 1.0 (one commit per release)
LINEAGE_475_RECOMMITS_PER_RELEASE = (11.8333, 14.0000, 12.8000)
LINEAGE_475_MEAN = 12.8778
LINEAGE_475_SD = 1.0854
DV_FLOOR = 1.0

# Non-degeneracy precondition (the claim's own): the comparator arm must
# reproduce the catatonic-lock regime. 8.0 is ~32 pct below 475's observed
# minimum (11.8333) and ~4.5 SD below its mean -- margin for adding the
# anchor/segmenter/per-region substrate, while still 7.0 above the DV floor so
# the effect gate below stays jointly satisfiable.
LOCK_REGIME_FLOOR = 8.0

# Effect-size gate, standing convention (scale on the SD of the paired delta
# plus an absolute floor -- hippocampal_campaign_assay_specifications_20260910
# section 1.6). Absolute floor = one third of the measured elevation of the
# lock regime over the DV floor: (12.8778 - 1.0) / 3 = 3.959 -> 4.0.
EFFECT_FLOOR_ABS = 4.0
EFFECT_SD_MULTIPLIER = 2.0
SEED_MAJORITY = 2

# Mode-flip budget, counted in RE-COMMITS after the 5th PAG release (the claim
# says "within a bounded re-commit budget after roughly five PAG releases").
# One lock-regime release-cycle's worth of re-commits: round(12.8778) = 13.
MODE_FLIP_RELEASE_INDEX = 5
MODE_FLIP_RECOMMIT_BUDGET = 13
MODE_FLIP_RUN = 3          # consecutive non-avoid steps that count as a flip

# Dissociation gates
BROADCAST_SILENT_MAX = 0   # trigger-lesioned arms must be exactly silent
STALENESS_ZERO_TOL = 1e-9  # accumulator-absent arms integrate nothing

# --------------------------------------------------------------------------- #
# Lineage config (V3-EXQ-475, verbatim)                                        #
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

# arm -> (use_invalidation_trigger, MECH-284 online readout, use_event_segmenter)
ARMS: Dict[str, Dict[str, bool]] = {
    "A_BOTH_OFF":  dict(trigger=False, accumulator=False, segmenter=True),
    "B_TRIG_ONLY": dict(trigger=True,  accumulator=False, segmenter=True),
    "C_ACC_ONLY":  dict(trigger=False, accumulator=True,  segmenter=True),
    "D_BOTH_ON":   dict(trigger=True,  accumulator=True,  segmenter=True),
    "E_SEG_OFF":   dict(trigger=True,  accumulator=True,  segmenter=False),
}
COMPARATOR_ARM = "A_BOTH_OFF"
TREATMENT_ARM  = "D_BOTH_ON"
TRIGGER_LESIONED_ARMS = ("A_BOTH_OFF", "C_ACC_ONLY")


# --------------------------------------------------------------------------- #
# Helpers (lineage-faithful)                                                   #
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


def _classify_mode(z_harm_norm: float, world_change_norm: float, harm_signal: float) -> str:
    if z_harm_norm > HARM_MODE_THRESH:
        return "avoid"
    if harm_signal > 0.01:
        return "approach"
    if world_change_norm > EXPLORE_ERR_THRESH:
        return "explore"
    return "neutral"


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
                  "batch_size": BATCH_SIZE},
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
        "arm_flags": {
            "use_invalidation_trigger": a["trigger"],
            "use_staleness_accumulator": a["accumulator"],
            "use_mech284_hysteresis": a["accumulator"],
            "use_event_segmenter": a["segmenter"],
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
        # V_s substrate: held ON and identical in EVERY arm
        use_per_stream_vs=True,
        use_anchor_sets=True,
        use_per_region_vs=True,
        use_vs_commit_release=True,
        # the manipulated factors
        use_invalidation_trigger=a["trigger"],
        use_staleness_accumulator=a["accumulator"],
        use_mech284_hysteresis=a["accumulator"],
        use_event_segmenter=a["segmenter"],
    )
    config.e3.commitment_threshold = 0.5
    config.heartbeat.beta_gate_bistable = True
    config.harm_descending_mod_enabled = True
    config.descending_attenuation_factor = 0.5
    return REEAgent(config), env


# --------------------------------------------------------------------------- #
# Phase 0: warmup (V3-EXQ-475 protocol, identical for every arm)               #
# --------------------------------------------------------------------------- #
def _warmup_train(agent, env, num_episodes: int, steps_per_episode: int,
                  arm: str, seed: int) -> Dict:
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

    agent.train()
    for ep in range(num_episodes):
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
        if (ep + 1) % 10 == 0 or ep == num_episodes - 1:
            print(f"  [train] arm={arm} seed={seed} ep {ep+1}/{num_episodes}"
                  f"  rv={agent.e3._running_variance:.4f}"
                  f"  ep_reward={ep_reward:.4f}", flush=True)

    first10 = float(np.mean(reward_log[:10])) if len(reward_log) >= 10 else float(np.mean(reward_log))
    last10 = float(np.mean(reward_log[-10:])) if len(reward_log) >= 10 else float(np.mean(reward_log))
    return {
        "final_running_variance": float(agent.e3._running_variance),
        "warmup_first10_reward": first10,
        "warmup_last10_reward": last10,
    }


# --------------------------------------------------------------------------- #
# Phase 2: eval with the MECH-287 factorial instrumentation                    #
# --------------------------------------------------------------------------- #
def _eval_agent(agent, env, num_episodes: int, steps_per_episode: int,
                arm: str, seed: int) -> Dict:
    action_dim = env.action_dim
    device = agent.device
    hip = agent.hippocampal

    episode_rewards: List[float] = []
    episode_harms: List[float] = []
    freeze_commit_count = 0
    freeze_active_steps = 0
    anchor_reset_count = 0
    staleness_peaks: List[float] = []

    # release-indexed timeline for the mode-flip DV
    global_step = 0
    release_steps: List[int] = []          # global step index of each PAG release
    recommit_steps: List[int] = []         # global step index of each freeze commit
    mode_timeline: List[Tuple[int, str]] = []
    prev_releases = 0

    agent.eval()
    for ep_idx in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        z_self_prev = z_world_prev = action_prev = None
        ep_reward = ep_harm = 0.0
        ep_peak = 0.0
        prev_active_keys = set()
        if hip is not None and getattr(hip, "anchor_set", None) is not None:
            prev_active_keys = {a.key for a in hip.anchor_set.active_anchors()}

        for _ in range(steps_per_episode):
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

            # MECH-279 PAG freeze gate: poll AFTER select_action (475 protocol)
            if agent.pag_freeze_gate is not None:
                last = agent.pag_freeze_gate.last_output
                if last.freeze_commit:
                    freeze_commit_count += 1
                    recommit_steps.append(global_step)
                if last.freeze_active:
                    freeze_active_steps += 1
                now_releases = int(agent.pag_freeze_gate.diagnostics.get("n_releases", 0))
                if now_releases > prev_releases:
                    for _r in range(now_releases - prev_releases):
                        release_steps.append(global_step)
                    prev_releases = now_releases

            # MECH-269 anchor resets (active -> inactive transitions)
            if hip is not None and getattr(hip, "anchor_set", None) is not None:
                active_now = {a.key for a in hip.anchor_set.active_anchors()}
                anchor_reset_count += len(prev_active_keys - active_now)
                prev_active_keys = active_now

            # MECH-284 staleness integral
            sa = getattr(hip, "staleness_accumulator", None) if hip is not None else None
            if sa is not None:
                snap = sa.snapshot()
                if snap:
                    ep_peak = max(ep_peak, max(snap.values()))

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)
            ep_reward += float(harm_signal)
            ep_harm += abs(min(0.0, float(harm_signal)))

            z_harm_norm = float(latent.z_harm_s.norm().item()) if getattr(
                latent, "z_harm_s", None) is not None else 0.0
            world_change_norm = (
                float((latent.z_world.detach() - z_world_prev).norm().item())
                if z_world_prev is not None else 0.0)
            mode_timeline.append(
                (global_step, _classify_mode(z_harm_norm, world_change_norm,
                                             float(harm_signal))))

            z_world_prev = latent.z_world.detach()
            z_self_prev = latent.z_self.detach()
            action_prev = action.detach()
            global_step += 1
            if done:
                break

        episode_rewards.append(ep_reward)
        episode_harms.append(ep_harm)
        staleness_peaks.append(ep_peak)

    pag_diag = (agent.pag_freeze_gate.diagnostics
                if agent.pag_freeze_gate is not None else {})
    trig = getattr(hip, "invalidation_trigger", None) if hip is not None else None
    trig_stats = trig.get_stats() if trig is not None else {}
    sa = getattr(hip, "staleness_accumulator", None) if hip is not None else None
    sa_stats = sa.get_stats() if sa is not None else {}

    n_releases = int(pag_diag.get("n_releases", 0))
    n_commits = int(pag_diag.get("n_commits", 0))
    recommits_per_release = (float(n_commits) / float(n_releases)
                             if n_releases > 0 else float("nan"))

    return {
        "arm": arm,
        "seed": seed,
        "mean_reward": float(np.mean(episode_rewards)),
        "mean_harm": float(np.mean(episode_harms)),
        "per_episode_reward": [float(x) for x in episode_rewards],
        "per_episode_harm": [float(x) for x in episode_harms],
        # --- primary DV ---
        "pag_n_ticks": int(pag_diag.get("n_ticks", 0)),
        "pag_n_commits": n_commits,
        "pag_n_releases": n_releases,
        "recommits_per_release": recommits_per_release,
        # --- mechanism instruments ---
        "n_broadcast": int(trig_stats.get("n_broadcast", 0)),
        "n_suppressed": int(trig_stats.get("n_suppressed", 0)),
        "staleness_n_integrations": int(sa_stats.get("n_integrations", 0)),
        "staleness_max": float(sa_stats.get("max_staleness", 0.0)),
        "staleness_mean": float(sa_stats.get("mean_staleness", 0.0)),
        "mean_staleness_peak": float(np.mean(staleness_peaks)) if staleness_peaks else 0.0,
        "anchor_reset_count": int(anchor_reset_count),
        # --- read-side manipulation check (Option A) ---
        "vs_commit_release_count": int(getattr(agent, "_vs_commit_release_count", 0)),
        # --- freeze / mode ---
        "freeze_commit_count": int(freeze_commit_count),
        "freeze_active_steps": int(freeze_active_steps),
        "n_eval_steps": int(global_step),
        "time_to_mode_flip_recommits": _time_to_mode_flip(
            release_steps, recommit_steps, mode_timeline),
        "mode_fraction_avoid": (
            float(sum(1 for _s, m in mode_timeline if m == "avoid") / len(mode_timeline))
            if mode_timeline else float("nan")),
    }


def _time_to_mode_flip(release_steps: List[int], recommit_steps: List[int],
                       mode_timeline: List[Tuple[int, str]]) -> float:
    """Re-commits between the Nth PAG release and the first sustained non-avoid run.

    Returns NaN when the Nth release never happened (the DV is undefined, not
    zero -- an absent release is not a fast flip).
    """
    if len(release_steps) < MODE_FLIP_RELEASE_INDEX:
        return float("nan")
    anchor_step = release_steps[MODE_FLIP_RELEASE_INDEX - 1]
    run = 0
    flip_step: Optional[int] = None
    for step, mode in mode_timeline:
        if step < anchor_step:
            continue
        if mode != "avoid":
            run += 1
            if run >= MODE_FLIP_RUN:
                flip_step = step
                break
        else:
            run = 0
    if flip_step is None:
        return float("inf")   # never flipped within the run
    return float(sum(1 for s in recommit_steps if anchor_step <= s <= flip_step))


# --------------------------------------------------------------------------- #
# Cell driver                                                                  #
# --------------------------------------------------------------------------- #
def run_cell(arm: str, seed: int, warmup_eps: int, eval_eps: int,
             steps: int) -> Dict:
    print(f"\nSeed {seed} Condition {arm}", flush=True)
    with arm_cell(seed, config_slice=arm_config_slice(arm),
                  script_path=Path(__file__),
                  include_driver_script_in_hash=False) as cell:
        agent, env = _make_agent_and_env(arm, seed)
        warm = _warmup_train(agent, env, warmup_eps, steps, arm, seed)
        row = _eval_agent(agent, env, eval_eps, steps, arm, seed)
        row.update({f"warmup_{k}": v for k, v in warm.items()})
        cell.stamp(row)
    rpr = row["recommits_per_release"]
    print(f"  arm={arm} seed={seed} pag_commits={row['pag_n_commits']}"
          f" releases={row['pag_n_releases']}"
          f" recommits_per_release={rpr:.4f}"
          f" broadcasts={row['n_broadcast']}"
          f" staleness_max={row['staleness_max']:.6f}"
          f" anchor_resets={row['anchor_reset_count']}"
          f" vs_releases={row['vs_commit_release_count']}", flush=True)
    return row


# --------------------------------------------------------------------------- #
# Analysis                                                                     #
# --------------------------------------------------------------------------- #
def _by(rows: List[Dict], arm: str) -> Dict[int, Dict]:
    return {r["seed"]: r for r in rows if r["arm"] == arm}


def _finite(x) -> bool:
    return isinstance(x, float) and np.isfinite(x)


def analyse(rows: List[Dict]) -> Dict:
    comp = _by(rows, COMPARATOR_ARM)
    treat = _by(rows, TREATMENT_ARM)
    seeds = sorted(set(comp) & set(treat))

    # ---- non-degeneracy precondition: comparator reproduces the lock regime
    comp_vals = [comp[s]["recommits_per_release"] for s in sorted(comp)]
    comp_finite = [v for v in comp_vals if _finite(v)]
    worst_comp = min(comp_finite) if comp_finite else float("nan")
    worst_comp_seed = (sorted(comp)[comp_vals.index(worst_comp)]
                       if comp_finite and worst_comp in comp_vals else None)
    lock_met = bool(comp_finite) and len(comp_finite) == len(comp_vals) and worst_comp >= LOCK_REGIME_FLOOR

    # ---- C1 unlock effect (paired per-seed delta)
    deltas = [comp[s]["recommits_per_release"] - treat[s]["recommits_per_release"]
              for s in seeds
              if _finite(comp[s]["recommits_per_release"])
              and _finite(treat[s]["recommits_per_release"])]
    mean_d = float(np.mean(deltas)) if deltas else float("nan")
    sd_d = float(np.std(deltas, ddof=1)) if len(deltas) > 1 else float("nan")
    sd_gate = EFFECT_SD_MULTIPLIER * sd_d if _finite(sd_d) else float("nan")
    seeds_clearing = sum(1 for d in deltas if d >= EFFECT_FLOOR_ABS)
    c1 = bool(
        _finite(mean_d) and mean_d >= EFFECT_FLOOR_ABS
        and _finite(sd_gate) and mean_d >= sd_gate
        and seeds_clearing >= SEED_MAJORITY)

    # ---- C2 trigger-loss dissociation: trigger-lesioned arms are silent
    trig_les = [r for r in rows if r["arm"] in TRIGGER_LESIONED_ARMS]
    worst_les_bcast = max([r["n_broadcast"] for r in trig_les], default=0)
    c2 = bool(trig_les) and worst_les_bcast <= BROADCAST_SILENT_MAX

    # ---- C3 accumulator-loss dissociation: broadcasts that never integrate
    b_rows = [r for r in rows if r["arm"] == "B_TRIG_ONLY"]
    d_rows = [r for r in rows if r["arm"] == TREATMENT_ARM]
    b_min_bcast = min([r["n_broadcast"] for r in b_rows], default=0)
    b_max_stale = max([r["staleness_max"] for r in b_rows], default=0.0)
    d_min_bcast = min([r["n_broadcast"] for r in d_rows], default=0)
    d_min_stale = min([r["staleness_max"] for r in d_rows], default=0.0)
    c3 = bool(b_rows and d_rows
              and b_min_bcast > 0 and b_max_stale <= STALENESS_ZERO_TOL
              and d_min_bcast > 0 and d_min_stale > STALENESS_ZERO_TOL)

    # ---- C4 verdict-3 control: trigger present, no boundary input -> silent
    e_rows = [r for r in rows if r["arm"] == "E_SEG_OFF"]
    e_max_bcast = max([r["n_broadcast"] for r in e_rows], default=0)
    c4 = bool(e_rows) and e_max_bcast <= BROADCAST_SILENT_MAX

    # ---- C5 mode flip within a bounded re-commit budget
    flips = [(s, treat[s]["time_to_mode_flip_recommits"]) for s in seeds]
    flips_ok = sum(1 for _s, f in flips
                   if _finite(f) and f <= MODE_FLIP_RECOMMIT_BUDGET)
    worst_flip = max([f for _s, f in flips if _finite(f)], default=float("nan"))
    c5 = flips_ok >= SEED_MAJORITY

    # ---- manipulation check: did the read-side hook fire at all?
    treat_vs = min([r["vs_commit_release_count"] for r in d_rows], default=0)
    readside_live = treat_vs > 0

    overall = bool(lock_met and c1 and c2 and c3 and c4 and c5)

    criteria = [
        {"name": "C1_unlock_recommits_per_release", "load_bearing": True,
         "passed": c1, "measured": mean_d, "threshold": EFFECT_FLOOR_ABS,
         "measured_sd_of_delta": sd_d, "threshold_sd_gate": sd_gate,
         "seeds_clearing": seeds_clearing, "seeds_required": SEED_MAJORITY,
         "detail": "mean paired delta (A_BOTH_OFF - D_BOTH_ON) must clear BOTH "
                   "the absolute floor and 2x SD(delta), on >=2 of 3 seeds"},
        {"name": "C2_trigger_lesion_silent", "load_bearing": True,
         "passed": c2, "measured": float(worst_les_bcast),
         "threshold": float(BROADCAST_SILENT_MAX), "direction": "upper",
         "detail": "worst-cell n_broadcast across trigger-lesioned arms A and C"},
        {"name": "C3_accumulator_lesion_broadcasts_never_integrate",
         "load_bearing": True, "passed": c3,
         "measured": float(b_max_stale), "threshold": float(STALENESS_ZERO_TOL),
         "direction": "upper",
         "measured_b_min_broadcast": float(b_min_bcast),
         "measured_d_min_staleness": float(d_min_stale),
         "detail": "B_TRIG_ONLY: broadcasts fire but staleness never rises; "
                   "D_BOTH_ON: both rise"},
        {"name": "C4_segmenter_off_trigger_silent", "load_bearing": True,
         "passed": c4, "measured": float(e_max_bcast),
         "threshold": float(BROADCAST_SILENT_MAX), "direction": "upper",
         "detail": "verdict-3 control: trigger armed but no BoundaryEvent input"},
        {"name": "C5_mode_flip_within_recommit_budget", "load_bearing": True,
         "passed": c5, "measured": float(worst_flip),
         "threshold": float(MODE_FLIP_RECOMMIT_BUDGET), "direction": "upper",
         "seeds_clearing": flips_ok, "seeds_required": SEED_MAJORITY,
         "detail": "re-commits between the 5th PAG release and the first "
                   "sustained non-avoid run, in D_BOTH_ON"},
    ]

    preconditions = [
        {"name": "comparator_reproduces_catatonic_lock_regime",
         "description": "A_BOTH_OFF recommits_per_release, worst seed, must clear "
                        "the lock-regime floor derived from V3-EXQ-475",
         "measured": worst_comp, "threshold": LOCK_REGIME_FLOOR,
         "direction": "lower",
         "offending_cell": (f"{COMPARATOR_ARM}/seed={worst_comp_seed}"
                            if worst_comp_seed is not None else None),
         "control": "V3-EXQ-475 measured 11.8333/14.0000/12.8000 on this config "
                    "without the anchor substrate",
         "met": lock_met},
        {"name": "readside_commit_release_fires",
         "description": "use_vs_commit_release actually released a commitment in "
                        "the both-intact arm (worst seed); if 0, the read-side "
                        "hook never engaged and the DV could not move",
         "measured": float(treat_vs), "threshold": 1.0, "direction": "lower",
         "control": "Option A arms this hook identically in every arm",
         "met": readside_live},
    ]

    non_degenerate = bool(lock_met and readside_live)
    degeneracy_reason = None
    if not lock_met:
        degeneracy_reason = (
            f"comparator arm {COMPARATOR_ARM} did not reproduce the catatonic-lock "
            f"regime (worst-seed recommits_per_release {worst_comp} < "
            f"{LOCK_REGIME_FLOOR}); the unlock DV has no headroom")
    elif not readside_live:
        degeneracy_reason = (
            "use_vs_commit_release never fired in the both-intact arm, so anchor "
            "invalidation had no behavioural path and no arm could differ")

    if not non_degenerate:
        direction = "non_contributory"
        per_claim = {"MECH-287": "non_contributory", "MECH-284": "non_contributory"}
        label = "substrate_not_ready_requeue"
    elif overall:
        direction = "supports"
        per_claim = {"MECH-287": "supports", "MECH-284": "supports"}
        label = "trigger_accumulator_dissociation_confirmed"
    elif c2 and c3 and c4 and not (c1 or c5):
        direction = "weakens"
        per_claim = {"MECH-287": "weakens", "MECH-284": "weakens"}
        label = "mechanism_fires_but_no_behavioural_unlock"
    else:
        direction = "mixed"
        per_claim = {
            "MECH-287": "supports" if c1 else "mixed",
            "MECH-284": "supports" if c3 else "mixed",
        }
        label = "partial_dissociation"

    return {
        "outcome": "PASS" if overall else "FAIL",
        "criteria": criteria,
        "combination_rule": ("PASS = precondition(lock regime) AND C1 AND C2 AND "
                             "C3 AND C4 AND C5 (all load-bearing, plain AND)"),
        "interpretation": {
            "label": label,
            "preconditions": preconditions,
            "criteria_non_degenerate": {
                "C1_unlock_recommits_per_release": bool(len(deltas) == len(seeds) and lock_met),
                "C2_trigger_lesion_silent": bool(trig_les),
                "C3_accumulator_lesion_broadcasts_never_integrate": bool(b_rows and d_rows),
                "C4_segmenter_off_trigger_silent": bool(e_rows),
                "C5_mode_flip_within_recommit_budget": bool(
                    any(_finite(f) for _s, f in flips)),
            },
        },
        "evidence_direction": direction,
        "evidence_direction_per_claim": per_claim,
        "non_degenerate": non_degenerate,
        "degeneracy_reason": degeneracy_reason,
        "paired_deltas_recommits_per_release": {str(s): float(
            comp[s]["recommits_per_release"] - treat[s]["recommits_per_release"])
            for s in seeds
            if _finite(comp[s]["recommits_per_release"])
            and _finite(treat[s]["recommits_per_release"])},
    }


def _flat_scalar(rows: List[Dict], ana: Dict) -> Dict:
    out: Dict[str, float] = {}
    for c in ana["criteria"]:
        out[f"{c['name']}__passed"] = int(bool(c["passed"]))
        for k in ("measured", "threshold"):
            v = c.get(k)
            if isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v):
                out[f"{c['name']}__{k}"] = float(v)
    for p in ana["interpretation"]["preconditions"]:
        out[f"precondition__{p['name']}__met"] = int(bool(p["met"]))
        if isinstance(p.get("measured"), (int, float)) and np.isfinite(p["measured"]):
            out[f"precondition__{p['name']}__measured"] = float(p["measured"])
    for arm in ARMS:
        vals = [r["recommits_per_release"] for r in rows if r["arm"] == arm]
        fin = [v for v in vals if _finite(v)]
        if fin:
            out[f"recommits_per_release__{arm}__mean"] = float(np.mean(fin))
        bc = [r["n_broadcast"] for r in rows if r["arm"] == arm]
        if bc:
            out[f"n_broadcast__{arm}__min"] = float(min(bc))
        vs = [r["vs_commit_release_count"] for r in rows if r["arm"] == arm]
        if vs:
            out[f"vs_commit_release_count__{arm}__min"] = float(min(vs))
    out["non_degenerate"] = int(bool(ana["non_degenerate"]))
    return out


# --------------------------------------------------------------------------- #
def run(dry_run: bool = False) -> Dict:
    t0 = time.perf_counter()
    warmup_eps = 3 if dry_run else WARMUP_EPISODES
    eval_eps = 2 if dry_run else EVAL_EPISODES
    steps = 30 if dry_run else STEPS_PER_EPISODE
    seeds = SEEDS[:2] if dry_run else SEEDS

    print(f"{EXPERIMENT_TYPE} -- MECH-287 four-arm invalidation factorial", flush=True)
    print(f"Arms: {list(ARMS)}", flush=True)
    print(f"Seeds: {seeds}  warmup={warmup_eps}  eval={eval_eps}  steps/ep={steps}",
          flush=True)
    print("use_vs_commit_release=True held IDENTICAL in every arm (Option A)", flush=True)

    rows: List[Dict] = []
    for arm in ARMS:
        for seed in seeds:
            row = run_cell(arm, seed, warmup_eps, eval_eps, steps)
            rows.append(row)
            print(f"verdict: {'PASS' if _finite(row['recommits_per_release']) else 'FAIL'}",
                  flush=True)

    ana = analyse(rows)

    manifest = {
        "run_id": f"{EXPERIMENT_TYPE}_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_v3",
        "experiment_type": EXPERIMENT_TYPE,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "claim_ids": CLAIM_IDS,
        "timestamp_utc": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
        "outcome": ana["outcome"],
        "evidence_direction": ana["evidence_direction"],
        "evidence_direction_per_claim": ana["evidence_direction_per_claim"],
        "non_degenerate": ana["non_degenerate"],
        "degeneracy_reason": ana["degeneracy_reason"],
        "criteria": ana["criteria"],
        "combination_rule": ana["combination_rule"],
        "interpretation": ana["interpretation"],
        "paired_deltas_recommits_per_release": ana["paired_deltas_recommits_per_release"],
        "arm_results": rows,
        "per_seed_results": rows,
        "sleep_driver_pattern": "not_applicable",
        "threshold_derivation": {
            "lineage_run": "v3_exq_475_sd036_decay_unlocks_exq471_20260422T173839Z_v3",
            "lineage_recommits_per_release": list(LINEAGE_475_RECOMMITS_PER_RELEASE),
            "lineage_mean": LINEAGE_475_MEAN,
            "lineage_sd": LINEAGE_475_SD,
            "dv_floor": DV_FLOOR,
            "LOCK_REGIME_FLOOR": LOCK_REGIME_FLOOR,
            "EFFECT_FLOOR_ABS": EFFECT_FLOOR_ABS,
            "EFFECT_SD_MULTIPLIER": EFFECT_SD_MULTIPLIER,
            "MODE_FLIP_RECOMMIT_BUDGET": MODE_FLIP_RECOMMIT_BUDGET,
            "note": (
                "Effect-size gate follows the standing convention (scale on the SD "
                "of the paired delta plus an absolute floor). Absolute floor = one "
                "third of the lineage's measured elevation over the DV floor: "
                "(12.8778 - 1.0)/3 = 3.959 -> 4.0. Lock-regime floor 8.0 is ~32 pct "
                "below the lineage minimum, leaving the effect gate jointly "
                "satisfiable (at the floor, D must reach 4.0, still 3.0 above the "
                "DV floor). Mode-flip budget = round(lineage mean) = 13 re-commits, "
                "one lock-regime release-cycle's worth."),
        },
        "interpretation_caveat": (
            "use_vs_commit_release is held ON in every arm, so a positive result "
            "supports MECH-287 + MECH-284 + the MECH-269/MECH-090 read-side commit-"
            "release hook JOINTLY, not MECH-287 alone. The trigger x accumulator "
            "contrast is unconfounded by it because it is identical across arms."),
        "known_substrate_limitations": [
            "mech005-betagate-decommit-counter-and-commit-ceiling (degrading, open): "
            "BetaGate.release() increments no counter, so the de-commit leg is read "
            "here via the use_vs_commit_release counter instead.",
            "residue-integrate-sampling-collapse-world-dim-32 (degrading, open): "
            "residue integration is weak at world_dim=32, which this lineage uses.",
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
        "dims": {"world": WORLD_DIM, "self": SELF_DIM,
                 "harm": HARM_DIM, "harm_a": HARM_A_DIM},
        "substrate_operating": arm_config_slice(COMPARATOR_ARM)["substrate_operating"],
        "dry_run": bool(dry_run),
    }
    stamp_recording_core(manifest, config=full_config, seeds=seeds,
                         script_path=Path(__file__), started_at=t0)
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    manifest = run(dry_run=args.dry_run)

    out_dir = Path("/Users/dgolden/REE_Working/REE_assembly/evidence/experiments")
    out_path = out_dir / f"{manifest['run_id']}.json"
    write_manifest(out_path, manifest)

    print(f"\noutcome: {manifest['outcome']}", flush=True)
    print(f"evidence_direction: {manifest['evidence_direction']}", flush=True)
    print(f"non_degenerate: {manifest['non_degenerate']}", flush=True)
    for c in manifest["criteria"]:
        print(f"  {c['name']}: passed={c['passed']} "
              f"measured={c['measured']} threshold={c['threshold']}", flush=True)
    print(f"manifest: {out_path}", flush=True)

    _o = str(manifest["outcome"]).upper()
    emit_outcome(outcome=_o if _o in ("PASS", "FAIL") else "FAIL",
                 manifest_path=str(out_path), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
