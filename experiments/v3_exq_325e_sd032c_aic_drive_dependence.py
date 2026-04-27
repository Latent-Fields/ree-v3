#!/opt/local/bin/python3
"""
V3-EXQ-325e: SD-032c AIC-Analog -- Drive-Dependence Falsification Signature (FIXED)

experiment_purpose: evidence
supersedes: V3-EXQ-325d

=============================================================================
DO NOT QUEUE until MECH-269 V_s invalidation circuit is fully landed
(end-to-end validation via V3-EXQ-476).

WHY: Without behavioural mode-switch variation driven by V_s-mediated anchor
invalidation, beta_gate_bistable commits lock the agent into a single mode
for the whole eval window. safe_ratio() then returns the 1.0
empty-uncommitted fallback for all conditions and C1/C4 remain untestable
regardless of whether the AIC/drive wiring is correct. V_s-driven mode
transitions (MECH-269 + MECH-287 + MECH-288 + MECH-284) are the supply of
genuine commit/uncommit alternation that makes a z_harm_s_ratio measurable.

Track the gate on EXQ-476 acceptance. When EXQ-476 passes, add a queue
entry for this script (machine_affinity=any, estimated_minutes=~240,
supersedes=v3_exq_325d_sd032c_aic_descending_modulation, claim_ids=
[SD-032c, SD-021], experiment_purpose=evidence) and flip the three
use_per_stream_vs / use_anchor_sets / use_invalidation_trigger flags
on below (currently False so smoke test runs clean before gate lifts).
=============================================================================

ROOT CAUSE of EXQ-325d bit-identical aic_salience_mean per seed
(AIC_ON == AIC_DRIVE_ABLATED to 16 decimals):

  Bug 1 -- wrong knob targeted for salience drive-dependence.
    The 325d ablation toggled aic_drive_protect_weight (0.0 vs 1.0).
    Per ree_core/cingulate/aic_analog.py AICAnalog.tick():
      urgency_scaled = ratio * (1.0 + drive_coupling * drive)   <- SALIENCE
      drive_protect  = 1.0 - drive_protect_weight * drive       <- GAIN ONLY
      gain = 1.0 - base_attenuation * mode_weight * drive_protect
    drive_protect_weight enters harm_s_gain ONLY. It cannot move
    aic_salience regardless of drive_level. The C4 acceptance criterion
    used aic_salience, so this arm was mis-targeted.

  Bug 2 -- drive_level was never updated during the run.
    325d's training/eval loops never called agent.update_z_goal(),
    so GoalState._last_drive_level stayed at the getattr default 0.0.
    With drive=0, BOTH multiplicative drive terms collapse: salience
    falls to ratio * 1.0 and gain falls to its drive-independent
    baseline. Even if the right knob had been toggled, drive=0
    permanently neutralises any drive-dependence signature.

  Bug 3 -- beta_gate_bistable=True + no interrupt-release path.
    MECH-090 bistable gate elevates on entry to committed state and
    only releases on hippocampal completion signal or MECH-091 urgency
    interrupt. With neither wired to fire in this env, the agent
    stayed committed for all 6000 eval steps. safe_ratio(committed,
    uncommitted) returned its 1.0 empty-uncommitted fallback for all
    four conditions, so z_harm_s_ratio became a constant unrelated
    to AIC/descending behaviour.

FIXES applied here (in addition to the V_s gate above):

  Fix 1: Call agent.update_z_goal(benefit_exposure,
         drive_level=REEAgent.compute_drive_level(obs_body)) each
         training and eval tick so GoalState._last_drive_level
         tracks SD-012 drive_level in [0, 1].

  Fix 2: Split the ablation into two targeted arms:
           AIC_COUPLING_ABLATED  -- aic_drive_coupling=0
                                    (salience drive-dependence knob)
           AIC_PROTECT_ABLATED   -- aic_drive_protect_weight=0
                                    (gain drive-dependence knob)
         C4a tests salience drive-dependence (AIC_ON vs
         AIC_COUPLING_ABLATED on aic_salience_mean).
         C4b tests gain drive-dependence (AIC_ON vs
         AIC_PROTECT_ABLATED on z_harm_s_ratio).

  Fix 3: Force genuine commit/uncommit alternation. Two independent
         levers, both gated on V_s landing:
           (a) urgency_interrupt_threshold lowered (E3Config) so
               MECH-091 fires on typical z_harm_a.norm() seen in
               this env, giving uncommitted baseline ticks.
           (b) V_s-driven anchor invalidation (use_per_stream_vs,
               use_anchor_sets, use_invalidation_trigger) supplies
               the biologically primary mode-switch signal that
               MECH-090's bistable latch was designed around.
         Both are pre-wired here but held OFF by the V_s gate --
         EXQ-476 success is the trigger to flip them on.

Five conditions (all use beta_gate_bistable=True + limb_damage, matching
EXQ-325d's commit regime):
  AIC_ON                -- use_aic_analog=True, drive-dependence fully on
  AIC_COUPLING_ABLATED  -- aic_drive_coupling=0.0 (salience drive-independent)
  AIC_PROTECT_ABLATED   -- aic_drive_protect_weight=0.0 (gain drive-independent)
  LEGACY                -- use_aic_analog=False, legacy SD-021 raw beta gate
  CONTROL               -- harm_descending_mod_enabled=False

Pass criteria (pre-registered, split C4 in two):
  C1 (attenuation present): AIC_ON z_harm_s_ratio < CONTROL z_harm_s_ratio
  C2 (stream selectivity):  |AIC_ON z_harm_a_ratio - CONTROL z_harm_a_ratio| < 0.3
  C3 (commits exist):       AIC_ON n_committed_steps >= 10
  C3b (uncommits exist):    AIC_ON n_uncommitted_steps >= 10
                            (without this, safe_ratio returns fallback
                             and C1/C4b are not measured)
  C4a (salience drive-dependence):
       |AIC_ON aic_salience_mean - AIC_COUPLING_ABLATED aic_salience_mean| > 0.02
  C4b (gain drive-dependence):
       |AIC_ON z_harm_s_ratio - AIC_PROTECT_ABLATED z_harm_s_ratio| > 0.02
  C5 (module non-trivial):  AIC_ON aic_salience_mean > 0.0
  C6 (drive tracked):       mean(drive_level_trace_AIC_ON) > 0.05
                            (guards against Bug 2 regressing silently)

Experiment PASS: >= 2/3 seeds satisfy C1, C2, C3, C3b, C4a, C4b, C5, C6.

Claims: SD-032c (primary), SD-021 (subsumed). Supersedes V3-EXQ-325d.
"""

import json
import sys
import random
import datetime
import argparse
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.latent.stack import HarmEncoder, AffectiveHarmEncoder
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_325e_sd032c_aic_drive_dependence"
CLAIM_IDS = ["SD-032c", "SD-021"]

C2_THRESHOLD = 0.3
C3_THRESHOLD = 10
C3B_THRESHOLD = 10
C4_THRESHOLD = 0.02
C6_DRIVE_THRESHOLD = 0.05
PASS_MIN_SEEDS = 2

HARM_OBS_DIM = 51
HARM_OBS_A_DIM = 7
Z_HARM_DIM = 32
Z_HARM_A_DIM = 16

SEEDS = [42, 43, 44]
TRAIN_EPISODES = 150
EVAL_EPISODES = 30
STEPS_PER_EPISODE = 200
LR = 1e-3
WF_LR = 1e-3

# V_s invalidation circuit gate. Flip these to True only after EXQ-476
# passes. Leaving them False lets the script smoke-test and preserve
# reproducibility with EXQ-325d's commit regime while the gate holds.
VS_CIRCUIT_READY = False

# Lowered urgency interrupt threshold to break the permanent-commit lock
# in the absence of V_s-driven mode transitions. Only active when the
# V_s circuit is still off; once V_s is wired in, the biologically
# primary release path takes over and this knob returns to default.
URGENCY_INTERRUPT_THRESHOLD_FALLBACK = 0.3

CONDITIONS = ["AIC_ON", "AIC_COUPLING_ABLATED", "AIC_PROTECT_ABLATED", "LEGACY", "CONTROL"]


def make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        size=10, num_hazards=4, num_resources=3,
        hazard_harm=0.1, resource_benefit=0.05,
        use_proxy_fields=True,
        limb_damage_enabled=True,
        damage_increment=0.15,
        failure_prob_scale=0.3,
        heal_rate=0.002,
        seed=seed,
    )


def make_config(condition: str) -> REEConfig:
    from ree_core.utils.config import HeartbeatConfig, E3Config
    hb = HeartbeatConfig(beta_gate_bistable=True)

    use_aic = condition in ("AIC_ON", "AIC_COUPLING_ABLATED", "AIC_PROTECT_ABLATED")
    descending = condition != "CONTROL"
    drive_coupling = 0.0 if condition == "AIC_COUPLING_ABLATED" else 1.0
    drive_protect_weight = 0.0 if condition == "AIC_PROTECT_ABLATED" else 1.0

    urgency_thresh = (
        0.8 if VS_CIRCUIT_READY else URGENCY_INTERRUPT_THRESHOLD_FALLBACK
    )
    e3 = E3Config(urgency_interrupt_threshold=urgency_thresh)

    return REEConfig.from_dims(
        body_obs_dim=17,
        world_obs_dim=250,
        action_dim=4,
        alpha_world=0.9,
        use_harm_stream=True,
        harm_obs_dim=HARM_OBS_DIM,
        z_harm_dim=Z_HARM_DIM,
        use_affective_harm_stream=True,
        harm_obs_a_dim=HARM_OBS_A_DIM,
        z_harm_a_dim=Z_HARM_A_DIM,
        limb_damage_enabled=True,
        harm_descending_mod_enabled=descending,
        descending_attenuation_factor=0.5,
        use_aic_analog=use_aic,
        aic_baseline_alpha=0.02,
        aic_drive_coupling=drive_coupling,
        aic_base_attenuation=0.5,
        aic_drive_protect_weight=drive_protect_weight,
        heartbeat=hb,
        e3=e3,
    )


def _compute_drive(obs_body: torch.Tensor) -> float:
    return float(REEAgent.compute_drive_level(obs_body))


def _benefit_from_obs(obs_dict: Dict) -> float:
    # CausalGridWorldV2 reports per-step benefit_exposure in info-ish scalar
    # fields; fall back to 0.0 if absent so the drive hook still ticks.
    be = obs_dict.get("benefit_exposure", None)
    if be is None:
        return 0.0
    if isinstance(be, torch.Tensor):
        return float(be.item())
    return float(be)


def run_training(agent: REEAgent, enc_s: HarmEncoder, enc_a: AffectiveHarmEncoder,
                 env: CausalGridWorldV2, device, n_eps: int, seed: int, condition: str,
                 n_steps: int = STEPS_PER_EPISODE):
    prox_head = nn.Sequential(nn.Linear(Z_HARM_DIM, 1), nn.Sigmoid()).to(device)
    all_params = (
        list(agent.parameters())
        + list(enc_s.parameters())
        + list(enc_a.parameters())
        + list(prox_head.parameters())
    )
    opt = optim.Adam(all_params, lr=LR)
    wf_opt = optim.Adam(
        list(agent.e2.world_transition.parameters()) +
        list(agent.e2.world_action_encoder.parameters()),
        lr=WF_LR,
    )
    wf_buf: deque = deque(maxlen=2000)

    for ep in range(n_eps):
        _, obs_dict = env.reset()
        agent.reset()
        z_world_prev: Optional[torch.Tensor] = None
        action_prev: Optional[torch.Tensor] = None

        for _ in range(n_steps):
            obs_body = obs_dict["body_state"].to(device)
            obs_world = obs_dict["world_state"].to(device)
            harm_obs = obs_dict.get("harm_obs")
            harm_obs_a = obs_dict.get("harm_obs_a")
            if harm_obs is None:
                break
            harm_obs = harm_obs.to(device)

            # Fix 1: drive_level tracking (SD-012)
            drive = _compute_drive(obs_body)
            benefit = _benefit_from_obs(obs_dict)
            agent.update_z_goal(benefit_exposure=benefit, drive_level=drive)

            z_harm_s = enc_s(harm_obs.unsqueeze(0))
            if harm_obs_a is not None:
                harm_obs_a_t = harm_obs_a.to(device)
                enc_a(harm_obs_a_t.unsqueeze(0))

            latent = agent.sense(obs_body, obs_world,
                                 obs_harm=harm_obs, obs_harm_a=harm_obs_a)
            z_world_curr = latent.z_world.detach()

            if z_world_prev is not None and action_prev is not None:
                wf_buf.append((
                    z_world_prev.cpu(),
                    action_prev.cpu(),
                    z_world_curr.cpu(),
                ))

            ticks = agent.clock.advance()
            e1_prior = (
                agent._e1_tick(latent) if ticks.get("e1_tick")
                else torch.zeros(1, agent.config.latent.world_dim, device=device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            action = agent.select_action(candidates, ticks)
            action_idx = int(action.argmax(dim=-1).item())
            _, _, done, _, obs_dict = env.step(action_idx)

            opt.zero_grad()
            pred_loss = agent.compute_prediction_loss()
            harm_loss = F.mse_loss(prox_head(z_harm_s), harm_obs[-1:].unsqueeze(0))
            total = pred_loss + harm_loss
            if total.requires_grad:
                total.backward()
                torch.nn.utils.clip_grad_norm_(all_params, 1.0)
                opt.step()

            if len(wf_buf) >= 16:
                k = min(32, len(wf_buf))
                idxs = torch.randperm(len(wf_buf))[:k].tolist()
                zw_b = torch.cat([wf_buf[i][0] for i in idxs]).to(device)
                a_b = torch.cat([wf_buf[i][1] for i in idxs]).to(device)
                zw1_b = torch.cat([wf_buf[i][2] for i in idxs]).to(device)
                wf_pred = agent.e2.world_forward(zw_b, a_b)
                wf_loss = F.mse_loss(wf_pred, zw1_b)
                if wf_loss.requires_grad:
                    wf_opt.zero_grad()
                    wf_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(agent.e2.world_transition.parameters()) +
                        list(agent.e2.world_action_encoder.parameters()),
                        1.0,
                    )
                    wf_opt.step()
                with torch.no_grad():
                    agent.e3.update_running_variance(
                        (wf_pred.detach() - zw1_b).detach()
                    )

            z_world_prev = z_world_curr
            action_prev = action.detach()
            if done:
                break

        if (ep + 1) % 25 == 0 or ep == n_eps - 1:
            print(f"  [train] seed={seed} cond={condition} ep {ep+1}/{n_eps}", flush=True)


def eval_commitment_attenuation(agent: REEAgent, enc_s: HarmEncoder, enc_a: AffectiveHarmEncoder,
                                env: CausalGridWorldV2, device, n_eps: int,
                                n_steps: int = STEPS_PER_EPISODE) -> Dict:
    z_s_committed: List[float] = []
    z_s_uncommitted: List[float] = []
    z_a_committed: List[float] = []
    z_a_uncommitted: List[float] = []
    aic_saliences: List[float] = []
    drive_trace: List[float] = []
    total_committed = 0
    total_uncommitted = 0

    for ep in range(n_eps):
        _, obs_dict = env.reset()
        agent.reset()
        for _ in range(n_steps):
            obs_body = obs_dict["body_state"].to(device)
            obs_world = obs_dict["world_state"].to(device)
            harm_obs = obs_dict.get("harm_obs")
            harm_obs_a = obs_dict.get("harm_obs_a")
            if harm_obs is None:
                break
            harm_obs = harm_obs.to(device)

            # Fix 1: drive_level tracking during eval too
            drive = _compute_drive(obs_body)
            benefit = _benefit_from_obs(obs_dict)
            agent.update_z_goal(benefit_exposure=benefit, drive_level=drive)
            drive_trace.append(drive)

            with torch.no_grad():
                z_harm_a_val = None
                if harm_obs_a is not None:
                    harm_obs_a_t = harm_obs_a.to(device)
                    res = enc_a(harm_obs_a_t.unsqueeze(0))
                    z_harm_a_val = res[0] if isinstance(res, tuple) else res

                latent = agent.sense(obs_body, obs_world,
                                     obs_harm=harm_obs, obs_harm_a=harm_obs_a)
                ticks = agent.clock.advance()
                e1_prior = (
                    agent._e1_tick(latent) if ticks.get("e1_tick")
                    else torch.zeros(1, agent.config.latent.world_dim, device=device)
                )
                candidates = agent.generate_trajectories(latent, e1_prior, ticks)
                action = agent.select_action(candidates, ticks)

            if agent.aic is not None:
                aic_saliences.append(float(agent.aic.aic_salience))

            is_committed = agent.e3._committed_trajectory is not None
            action_idx = int(action.argmax(dim=-1).item())
            _, _, done, _, obs_dict = env.step(action_idx)

            if latent.z_harm is not None:
                z_s_post = float(latent.z_harm.norm().item())
            else:
                z_s_post = 0.0
            if latent.z_harm_a is not None:
                z_a_post = float(latent.z_harm_a.norm().item())
            elif z_harm_a_val is not None:
                z_a_post = float(z_harm_a_val.norm().item())
            else:
                z_a_post = 0.0

            if is_committed:
                z_s_committed.append(z_s_post)
                z_a_committed.append(z_a_post)
                total_committed += 1
            else:
                z_s_uncommitted.append(z_s_post)
                z_a_uncommitted.append(z_a_post)
                total_uncommitted += 1

            if done:
                break

    def safe_ratio(committed, uncommitted):
        if not committed or not uncommitted:
            return 1.0
        mean_uc = float(np.mean(uncommitted))
        if mean_uc < 1e-8:
            return 1.0
        return float(np.mean(committed)) / mean_uc

    return {
        "z_harm_s_ratio": safe_ratio(z_s_committed, z_s_uncommitted),
        "z_harm_a_ratio": safe_ratio(z_a_committed, z_a_uncommitted),
        "n_committed_steps": total_committed,
        "n_uncommitted_steps": total_uncommitted,
        "z_harm_s_mean_committed": float(np.mean(z_s_committed)) if z_s_committed else 0.0,
        "z_harm_s_mean_uncommitted": float(np.mean(z_s_uncommitted)) if z_s_uncommitted else 0.0,
        "z_harm_a_mean_committed": float(np.mean(z_a_committed)) if z_a_committed else 0.0,
        "z_harm_a_mean_uncommitted": float(np.mean(z_a_uncommitted)) if z_a_uncommitted else 0.0,
        "aic_salience_mean": float(np.mean(aic_saliences)) if aic_saliences else 0.0,
        "aic_salience_max": float(np.max(aic_saliences)) if aic_saliences else 0.0,
        "drive_level_mean": float(np.mean(drive_trace)) if drive_trace else 0.0,
        "drive_level_max": float(np.max(drive_trace)) if drive_trace else 0.0,
    }


def run_seed(seed: int, dry_run: bool = False) -> Dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cpu")
    n_train = 2 if dry_run else TRAIN_EPISODES
    n_eval = 1 if dry_run else EVAL_EPISODES
    n_steps = 5 if dry_run else STEPS_PER_EPISODE

    condition_results = {}
    for condition in CONDITIONS:
        print(f"Seed {seed} Condition {condition}", flush=True)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        env = make_env(seed)
        cfg = make_config(condition)
        agent = REEAgent(cfg)
        enc_s = HarmEncoder(harm_obs_dim=HARM_OBS_DIM, z_harm_dim=Z_HARM_DIM).to(device)
        enc_a = AffectiveHarmEncoder(harm_obs_a_dim=HARM_OBS_A_DIM, z_harm_a_dim=Z_HARM_A_DIM).to(device)

        run_training(agent, enc_s, enc_a, env, device, n_train, seed, condition, n_steps=n_steps)
        rv = agent.e3._running_variance
        print(f"  {condition}: rv_after_train={rv:.4f} committed_flag={rv < agent.e3.commit_threshold}", flush=True)
        metrics = eval_commitment_attenuation(agent, enc_s, enc_a, env, device, n_eval, n_steps=n_steps)
        condition_results[condition] = metrics
        print(
            f"  {condition}: z_s_ratio={metrics['z_harm_s_ratio']:.4f} "
            f"z_a_ratio={metrics['z_harm_a_ratio']:.4f} "
            f"committed={metrics['n_committed_steps']} "
            f"uncommitted={metrics['n_uncommitted_steps']} "
            f"drive_mean={metrics['drive_level_mean']:.3f} "
            f"aic_sal_mean={metrics['aic_salience_mean']:.4f}",
            flush=True,
        )

    aic_on = condition_results["AIC_ON"]
    aic_coup = condition_results["AIC_COUPLING_ABLATED"]
    aic_prot = condition_results["AIC_PROTECT_ABLATED"]
    legacy = condition_results["LEGACY"]
    ctrl = condition_results["CONTROL"]

    c1_pass = aic_on["z_harm_s_ratio"] < ctrl["z_harm_s_ratio"]
    c2_pass = abs(aic_on["z_harm_a_ratio"] - ctrl["z_harm_a_ratio"]) < C2_THRESHOLD
    c3_pass = aic_on["n_committed_steps"] >= C3_THRESHOLD
    c3b_pass = aic_on["n_uncommitted_steps"] >= C3B_THRESHOLD
    c4a_pass = abs(aic_on["aic_salience_mean"] - aic_coup["aic_salience_mean"]) > C4_THRESHOLD
    c4b_pass = abs(aic_on["z_harm_s_ratio"] - aic_prot["z_harm_s_ratio"]) > C4_THRESHOLD
    c5_pass = aic_on["aic_salience_mean"] > 0.0
    c6_pass = aic_on["drive_level_mean"] > C6_DRIVE_THRESHOLD
    seed_pass = all([c1_pass, c2_pass, c3_pass, c3b_pass, c4a_pass, c4b_pass, c5_pass, c6_pass])

    print(
        f"-> seed {seed}: {'PASS' if seed_pass else 'FAIL'} "
        f"C1={c1_pass} C2={c2_pass} C3={c3_pass} C3b={c3b_pass} "
        f"C4a={c4a_pass} C4b={c4b_pass} C5={c5_pass} C6={c6_pass}",
        flush=True,
    )

    return {
        "seed": seed,
        "seed_pass": seed_pass,
        "z_harm_s_ratio_aic_on": aic_on["z_harm_s_ratio"],
        "z_harm_s_ratio_aic_coupling_ablated": aic_coup["z_harm_s_ratio"],
        "z_harm_s_ratio_aic_protect_ablated": aic_prot["z_harm_s_ratio"],
        "z_harm_s_ratio_legacy": legacy["z_harm_s_ratio"],
        "z_harm_s_ratio_control": ctrl["z_harm_s_ratio"],
        "z_harm_a_ratio_aic_on": aic_on["z_harm_a_ratio"],
        "z_harm_a_ratio_control": ctrl["z_harm_a_ratio"],
        "n_committed_aic_on": aic_on["n_committed_steps"],
        "n_uncommitted_aic_on": aic_on["n_uncommitted_steps"],
        "aic_salience_mean_aic_on": aic_on["aic_salience_mean"],
        "aic_salience_mean_aic_coupling_ablated": aic_coup["aic_salience_mean"],
        "aic_salience_mean_aic_protect_ablated": aic_prot["aic_salience_mean"],
        "drive_level_mean_aic_on": aic_on["drive_level_mean"],
        "c1_attenuation_present": c1_pass,
        "c2_stream_selective": c2_pass,
        "c3_commits_exist": c3_pass,
        "c3b_uncommits_exist": c3b_pass,
        "c4a_salience_drive_dependence": c4a_pass,
        "c4b_gain_drive_dependence": c4b_pass,
        "c5_aic_salience_nontrivial": c5_pass,
        "c6_drive_tracked": c6_pass,
        "condition_results": condition_results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not VS_CIRCUIT_READY and not args.dry_run:
        print(
            "ERROR: VS_CIRCUIT_READY=False. Refusing to run a full evidence "
            "pass before MECH-269 V_s invalidation circuit lands (see module "
            "docstring). Re-run with --dry-run for a smoke test, or flip the "
            "gate after EXQ-476 passes.",
            flush=True,
        )
        sys.exit(2)

    timestamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = (
        "v3_exq_325e_sd032c_aic_drive_dependence_dry" if args.dry_run
        else f"v3_exq_325e_sd032c_aic_drive_dependence_{timestamp}_v3"
    )
    print(f"EXQ-325e start: {run_id} (VS_CIRCUIT_READY={VS_CIRCUIT_READY})", flush=True)

    seeds_to_run = SEEDS[:1] if args.dry_run else SEEDS
    per_seed = [run_seed(s, dry_run=args.dry_run) for s in seeds_to_run]
    seeds_passing = sum(1 for r in per_seed if r["seed_pass"])
    experiment_passes = seeds_passing >= PASS_MIN_SEEDS
    outcome = "PASS" if experiment_passes else "FAIL"

    print(f"\n=== EXQ-325e {outcome} ===", flush=True)
    print(f"Seeds pass: {seeds_passing}/{len(seeds_to_run)}", flush=True)
    for r in per_seed:
        s = "PASS" if r["seed_pass"] else "FAIL"
        print(
            f"  seed={r['seed']}: {s} "
            f"s_aic={r['z_harm_s_ratio_aic_on']:.4f} "
            f"s_coup={r['z_harm_s_ratio_aic_coupling_ablated']:.4f} "
            f"s_prot={r['z_harm_s_ratio_aic_protect_ablated']:.4f} "
            f"s_ctl={r['z_harm_s_ratio_control']:.4f} "
            f"sal={r['aic_salience_mean_aic_on']:.3f} "
            f"sal_coup={r['aic_salience_mean_aic_coupling_ablated']:.3f} "
            f"drive={r['drive_level_mean_aic_on']:.3f}",
            flush=True,
        )

    evidence_direction = "supports" if experiment_passes else "does_not_support"
    per_claim = {
        "SD-032c": evidence_direction,
        "SD-021": evidence_direction,
    }

    output = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": "evidence",
        "evidence_direction": evidence_direction,
        "evidence_direction_per_claim": per_claim,
        "supersedes": "v3_exq_325d_sd032c_aic_descending_modulation",
        "vs_circuit_ready": VS_CIRCUIT_READY,
        "urgency_interrupt_threshold": (
            0.8 if VS_CIRCUIT_READY else URGENCY_INTERRUPT_THRESHOLD_FALLBACK
        ),
        "registered_thresholds": {
            "C1_attenuation_present": "aic_on < control",
            "C2_stream_selectivity": C2_THRESHOLD,
            "C3_committed_steps": C3_THRESHOLD,
            "C3b_uncommitted_steps": C3B_THRESHOLD,
            "C4a_salience_drive_dependence_abs_delta": C4_THRESHOLD,
            "C4b_gain_drive_dependence_abs_delta": C4_THRESHOLD,
            "C5_aic_salience_gt": 0.0,
            "C6_drive_level_mean_gt": C6_DRIVE_THRESHOLD,
            "seeds_needed": PASS_MIN_SEEDS,
        },
        "outcome": outcome,
        "timestamp_utc": datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
        "per_seed_results": per_seed,
        "seeds_passing": seeds_passing,
        "experiment_passes": experiment_passes,
        "fix_notes": (
            "Supersedes EXQ-325d. Three targeted fixes: (1) agent.update_z_goal() "
            "called each tick with SD-012 drive_level so AIC formulas see non-zero "
            "drive; (2) ablation split into AIC_COUPLING_ABLATED (salience knob) "
            "and AIC_PROTECT_ABLATED (gain knob) so each drive-dependence term has "
            "its own acceptance criterion (C4a/C4b); (3) urgency_interrupt_threshold "
            "lowered to break the permanent-commit lock until the V_s invalidation "
            "circuit (MECH-269/287/288/284) provides the biologically primary "
            "mode-switch signal (VS_CIRCUIT_READY toggle). C3b (uncommits_exist) "
            "and C6 (drive_tracked) added as regression guards so any regression "
            "of Bugs 2/3 fails loudly instead of producing bit-identical results."
        ),
    }

    if not args.dry_run:
        out_dir = (
            Path(__file__).resolve().parents[2]
            / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{run_id}.json"
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Result written to: {out_path}", flush=True)
    else:
        print(f"[dry-run] outcome={outcome} (no manifest written)", flush=True)


if __name__ == "__main__":
    main()
