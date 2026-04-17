#!/opt/local/bin/python3
"""
V3-EXQ-431: SD-003 Causal Attribution Discrimination Probe (Signed Directionality Test)

experiment_purpose: evidence

Scientific question: Does the z_harm_s attribution pipeline correctly attribute
DIRECTION of harm -- i.e., does it predict that the actual action caused MORE harm
than a counterfactual action when harm actually occurs?

Background / motivation:
  EXQ-329 and EXQ-353 both PASS with sign_correct=1.0. However, those experiments
  computed sign_correct as (gap > 0), where gap = ||z_pred_act - z_pred_cf|| (unsigned
  L2 norm). This is TRIVIALLY true for any model that produces different outputs for
  different actions. It does not test the key SD-003 claim: that the actual action was
  predicted to lead to MORE harm than the counterfactual action.

  The genuine directionality test is:
    signed_causal = z_pred_act.norm() - z_pred_cf.norm()
    sign_correct = 1 if signed_causal > 0 (actual > CF harm)

  When harm events occur (agent at or near hazard), the actual action moved the agent
  TOWARD the hazard. A counterfactual action would have moved the agent elsewhere.
  E2_harm_s should predict: ||z_harm_s(a_actual)|| > ||z_harm_s(a_cf)||.
  With RANDOM weights this is ~0.5 (coin flip). TRAINED weights should show > 0.5.

  Extends EXQ-329/353 by:
  1. Computing signed directionality (not trivially-positive unsigned gap).
  2. Breaking down by transition_type: env_caused_hazard, hazard_approach,
     agent_caused_hazard (contamination -- expected to be rare, treated as diagnostic).
  3. Comparing TRAINED vs RANDOM sign_correct to confirm attribution is learned.

  Predicted pattern (env_caused_hazard):
    TRAINED sign_correct > 0.65 (actual action reliably leads to more predicted harm)
    RANDOM sign_correct ~= 0.5 (random model is noise)
  Predicted pattern (hazard_approach):
    TRAINED sign_correct > 0.55 (weaker: agent is already near, some directions similar)
    RANDOM sign_correct ~= 0.5
  Predicted pattern (agent_caused_hazard / contamination):
    UNCERTAIN -- contamination not in harm_obs -> z_harm_s near-zero before contact
    -> forward model has no harm signal to discriminate -> sign_correct may be ~0.5
    This is the key substrate-gap diagnostic for SD-003 coverage.

Conditions: TRAINED (P0+P1+P2) vs RANDOM (P2 only, fresh weights)

P0 (100 ep): HarmEncoder + agent encoder warmup (TRAINED only)
P1 (100 ep): E2HarmSForward interventional training fraction=0.5 (TRAINED only)
P2 (100 ep): Evaluation -- compute signed_causal and sign_correct per transition type

Pass criteria:
  C1: TRAINED forward_r2 >= 0.7 (E2_harm_s trained, in >= 2/3 seeds)
  C2: TRAINED sign_correct_env >= 0.65 (directional attribution for env hazard harm,
      significantly above 0.5 random baseline) in >= 2/3 seeds
  C3: TRAINED sign_correct_env > RANDOM sign_correct_env + 0.1 (learned, not structural)
      in >= 2/3 seeds
  C4: n_env >= 20 AND n_approach >= 20 per seed in TRAINED (sufficient stats)

Diagnostic (not gating PASS):
  D1: agent_caused sign_correct (if n_agent >= 3) -- contamination coverage gap test
  D2: approach sign_correct (TRAINED vs RANDOM) -- proximity attribution quality
  D3: unsigned causal_sig by type (for comparison with EXQ-329/353 values)

PASS: C1 AND C2 AND C3 AND C4 across >= 2/3 seeds.

Expected evidence implication:
  PASS: z_harm_s pipeline correctly attributes harm direction for env-hazard events.
  Contamination gap (D1): if agent_caused sign_correct ~= 0.5, this reveals z_harm_s
  does not cover contamination attribution -- needed for full SD-003 scope. Documents
  the substrate gap without failing the claim for the env-hazard case.
  FAIL: z_harm_s pipeline cannot attribute harm direction -- SD-003 needs redesign.

Claims: SD-003 (primary). SD-013 (interventional training used at fraction=0.5,
per EXQ-330a PASS).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import argparse
import random
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig
from ree_core.agent import REEAgent
from ree_core.predictors.e2_harm_s import E2HarmSForward, E2HarmSConfig

# ---------------------------------------------------------------------------
# Experiment metadata
# ---------------------------------------------------------------------------
EXPERIMENT_TYPE    = "v3_exq_431_sd003_causal_discrimination"
CLAIM_IDS          = ["SD-003", "SD-013"]
EXPERIMENT_PURPOSE = "evidence"

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
SEEDS      = [42, 7, 13]
CONDITIONS = ["TRAINED", "RANDOM"]

P0_EPISODES  = 100    # HarmEncoder + agent encoder warmup (TRAINED only)
P1_EPISODES  = 100    # E2HarmSForward interventional training (TRAINED only)
P2_EPISODES  = 100    # Evaluation (both conditions)
STEPS_PER_EP = 200

GRID_SIZE     = 8
NUM_RESOURCES = 3
NUM_HAZARDS   = 3     # extra hazard vs baseline for higher env_caused contact rate
HAZARD_HARM   = 0.3
# contaminated_harm uses env default (0.4)

Z_HARM_DIM   = 32
HARM_OBS_DIM = 51
ACTION_DIM   = 5

LR_AGENT    = 3e-4
LR_HARM_FWD = 5e-4

REPLAY_BUF_MAX = 5000
BATCH_SIZE     = 32

# SD-013 interventional settings (per EXQ-330a PASS)
INTERVENTIONAL_FRACTION = 0.5
INTERVENTIONAL_MARGIN   = 0.1

# Pass thresholds
C1_FORWARD_R2_THRESH       = 0.7
C2_SIGN_CORRECT_ENV_THRESH = 0.65   # well above 0.5 random baseline
C3_LEARNED_LIFT            = 0.10   # TRAINED sign_correct_env - RANDOM > this
C4_MIN_ENV_EVENTS          = 20
C4_MIN_APPROACH_EVENTS     = 20
MIN_SEEDS_PASS             = 2

HARM_TYPES = ["env_caused_hazard", "agent_caused_hazard", "hazard_approach"]

DRY_RUN_EPISODES = 3
DRY_RUN_STEPS    = 20


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed,
        size=GRID_SIZE,
        num_resources=NUM_RESOURCES,
        num_hazards=NUM_HAZARDS,
        hazard_harm=HAZARD_HARM,
        resource_benefit=0.3,
        resource_respawn_on_consume=True,
        proximity_harm_scale=0.1,
        proximity_approach_threshold=0.2,
        use_proxy_fields=True,
    )


def _make_agent(env: CausalGridWorldV2, seed: int) -> REEAgent:
    torch.manual_seed(seed)
    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=ACTION_DIM,
        self_dim=32,
        world_dim=32,
        alpha_world=0.9,
        use_harm_stream=True,
        harm_obs_dim=HARM_OBS_DIM,
        z_harm_dim=Z_HARM_DIM,
        use_resource_proximity_head=True,
        resource_proximity_weight=0.5,
        use_event_classifier=True,
    )
    return REEAgent(config)


def _make_harm_fwd(device, use_interventional: bool) -> E2HarmSForward:
    cfg = E2HarmSConfig(
        use_e2_harm_s_forward=True,
        z_harm_dim=Z_HARM_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=128,
        use_interventional=use_interventional,
        interventional_fraction=INTERVENTIONAL_FRACTION,
        interventional_margin=INTERVENTIONAL_MARGIN,
    )
    return E2HarmSForward(cfg).to(device)


def _onehot(idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, idx] = 1.0
    return v


def _random_cf_action(a_actual_idx: int, n_actions: int, device) -> torch.Tensor:
    choices = [i for i in range(n_actions) if i != a_actual_idx]
    cf_idx = random.choice(choices)
    return _onehot(cf_idx, n_actions, device)


# ---------------------------------------------------------------------------
# Training (TRAINED condition only)
# ---------------------------------------------------------------------------

def run_training(
    seed: int,
    env: CausalGridWorldV2,
    agent: REEAgent,
    harm_fwd: E2HarmSForward,
    dry_run: bool,
) -> float:
    """P0 + P1 training. Returns forward_r2 after P1."""
    total_p0  = DRY_RUN_EPISODES if dry_run else P0_EPISODES
    total_p1  = DRY_RUN_EPISODES if dry_run else P1_EPISODES
    steps_per = DRY_RUN_STEPS    if dry_run else STEPS_PER_EP
    total_eps = total_p0 + total_p1

    agent_opt    = optim.Adam(list(agent.parameters()), lr=LR_AGENT)
    harm_fwd_opt = optim.Adam(harm_fwd.parameters(), lr=LR_HARM_FWD)
    replay_buf: List[Tuple[torch.Tensor, int, torch.Tensor]] = []
    device     = agent.device
    prev_ttype = "none"

    eval_preds:   List[float] = []
    eval_targets: List[float] = []

    z_harm_s_prev:   Optional[torch.Tensor] = None
    action_prev_idx: Optional[int]          = None

    for ep in range(total_eps):
        _, obs_dict = env.reset()
        agent.reset()
        z_harm_s_prev   = None
        action_prev_idx = None

        phase = "P0" if ep < total_p0 else "P1"
        in_p1      = (phase == "P1")
        p1_eval_ep = in_p1 and ep >= (total_p0 + total_p1 - 20)

        for step in range(steps_per):
            obs_body  = obs_dict["body_state"].to(device)
            obs_world = obs_dict["world_state"].to(device)
            obs_harm  = obs_dict.get("harm_obs", None)
            if obs_harm is not None:
                obs_harm = (obs_harm.to(device).unsqueeze(0)
                            if obs_harm.dim() == 1 else obs_harm.to(device))

            z_self_prev_t: Optional[torch.Tensor] = None
            if agent._current_latent is not None:
                z_self_prev_t = agent._current_latent.z_self.detach().clone()

            latent     = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
            ticks      = agent.clock.advance()
            e1_prior   = (
                agent._e1_tick(latent)
                if ticks.get("e1_tick", True)
                else torch.zeros(1, 32, device=device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            action     = agent.select_action(candidates, ticks)
            action_idx = int(action.argmax(dim=-1).item())

            z_harm_s_now = latent.z_harm
            if z_harm_s_now is not None:
                z_hs_d = z_harm_s_now.detach().clone()
                if z_harm_s_prev is not None and action_prev_idx is not None:
                    replay_buf.append((z_harm_s_prev, action_prev_idx, z_hs_d))
                    if len(replay_buf) > REPLAY_BUF_MAX:
                        replay_buf = replay_buf[-REPLAY_BUF_MAX:]

            flat_next, harm_signal, done, info, obs_dict_next = env.step(action_idx)
            ttype = info.get("transition_type", "none")
            agent.update_residue(float(harm_signal))
            if z_self_prev_t is not None:
                agent.record_transition(z_self_prev_t, action, latent.z_self.detach())

            # P0: encoder warmup
            if phase == "P0":
                agent_opt.zero_grad()
                loss = agent.compute_prediction_loss() + agent.compute_e2_loss()
                rfv = obs_dict.get("resource_field_view", None)
                if rfv is not None:
                    rp_t = float(rfv.max().item())
                    loss = loss + agent.compute_resource_proximity_loss(rp_t, latent)
                lat2 = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
                loss = loss + agent.compute_event_contrastive_loss(prev_ttype, lat2)
                if loss.requires_grad:
                    loss.backward()
                    nn.utils.clip_grad_norm_(list(agent.parameters()), 1.0)
                    agent_opt.step()

            # P1: E2HarmSForward interventional training
            if in_p1 and len(replay_buf) >= BATCH_SIZE:
                batch_idx  = random.sample(range(len(replay_buf)), BATCH_SIZE)
                z_s_b  = torch.cat([replay_buf[i][0] for i in batch_idx], dim=0).detach()
                a_idxs = [replay_buf[i][1] for i in batch_idx]
                z_s1_b = torch.cat([replay_buf[i][2] for i in batch_idx], dim=0).detach()

                a_b = torch.zeros(BATCH_SIZE, ACTION_DIM, device=device)
                for bi, ai in enumerate(a_idxs):
                    a_b[bi, ai] = 1.0

                harm_fwd_opt.zero_grad()
                z_pred = harm_fwd(z_s_b, a_b)
                fwd_loss = harm_fwd.compute_loss(z_pred, z_s1_b)

                if random.random() < INTERVENTIONAL_FRACTION:
                    a_cf_b = torch.zeros(BATCH_SIZE, ACTION_DIM, device=device)
                    for bi, ai in enumerate(a_idxs):
                        cfs = [j for j in range(ACTION_DIM) if j != ai]
                        a_cf_b[bi, random.choice(cfs)] = 1.0
                    fwd_loss = fwd_loss + harm_fwd.compute_interventional_loss(
                        z_s_b, a_b, a_cf_b
                    )

                fwd_loss.backward()
                harm_fwd_opt.step()

            # P1 forward_r2 eval slice
            if p1_eval_ep and z_harm_s_now is not None and z_harm_s_prev is not None:
                with torch.no_grad():
                    a_act = _onehot(action_idx, ACTION_DIM, device)
                    z_pred_act = harm_fwd(z_harm_s_prev.detach(), a_act)
                for d in range(Z_HARM_DIM):
                    eval_preds.append(float(z_pred_act[0, d].item()))
                    eval_targets.append(float(z_harm_s_now.detach()[0, d].item()))

            z_harm_s_prev   = z_harm_s_now.detach().clone() if z_harm_s_now is not None else None
            action_prev_idx = action_idx
            prev_ttype      = ttype
            obs_dict        = obs_dict_next

            if done:
                break

        if (ep + 1) % 50 == 0:
            print(f"  [train] seed={seed} ep {ep+1}/{total_eps} "
                  f"phase={phase} replay={len(replay_buf)}", flush=True)

    forward_r2 = 0.0
    if len(eval_preds) >= 10:
        try:
            tgt    = np.array(eval_targets)
            prd    = np.array(eval_preds)
            ss_res = float(np.sum((tgt - prd) ** 2))
            ss_tot = float(np.sum((tgt - tgt.mean()) ** 2))
            forward_r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-10 else 0.0
        except Exception:
            pass

    return forward_r2


# ---------------------------------------------------------------------------
# Evaluation (both conditions)
# ---------------------------------------------------------------------------

def run_evaluation(
    seed: int,
    env: CausalGridWorldV2,
    agent: REEAgent,
    harm_fwd: E2HarmSForward,
    dry_run: bool,
) -> Dict:
    """
    Collect signed causal_sig per transition type over P2 episodes.
    signed_causal = z_pred_act.norm() - z_pred_cf.norm()
    sign_correct = fraction of harm events where signed_causal > 0.
    """
    total_p2  = DRY_RUN_EPISODES if dry_run else P2_EPISODES
    steps_per = DRY_RUN_STEPS    if dry_run else STEPS_PER_EP
    device    = agent.device

    # Per-type accumulators
    signed_causals: Dict[str, List[float]] = {t: [] for t in HARM_TYPES}
    unsigned_causals: Dict[str, List[float]] = {t: [] for t in HARM_TYPES}

    z_harm_s_prev:   Optional[torch.Tensor] = None

    for ep in range(total_p2):
        _, obs_dict = env.reset()
        agent.reset()
        z_harm_s_prev = None

        for step in range(steps_per):
            obs_body  = obs_dict["body_state"].to(device)
            obs_world = obs_dict["world_state"].to(device)
            obs_harm  = obs_dict.get("harm_obs", None)
            if obs_harm is not None:
                obs_harm = (obs_harm.to(device).unsqueeze(0)
                            if obs_harm.dim() == 1 else obs_harm.to(device))

            latent     = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
            ticks      = agent.clock.advance()
            e1_prior   = (
                agent._e1_tick(latent)
                if ticks.get("e1_tick", True)
                else torch.zeros(1, 32, device=device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            action     = agent.select_action(candidates, ticks)
            action_idx = int(action.argmax(dim=-1).item())

            z_harm_s_now = latent.z_harm

            flat_next, harm_signal, done, info, obs_dict_next = env.step(action_idx)
            ttype = info.get("transition_type", "none")
            agent.update_residue(float(harm_signal))

            # Compute signed causal_sig for harm events
            if harm_signal < 0.0 and ttype in HARM_TYPES and z_harm_s_now is not None:
                z_hs = z_harm_s_now.detach()
                a_act = _onehot(action_idx, ACTION_DIM, device)
                a_cf  = _random_cf_action(action_idx, ACTION_DIM, device)

                with torch.no_grad():
                    z_pred_act = harm_fwd(z_hs, a_act)
                    z_pred_cf  = harm_fwd(z_hs, a_cf)

                # Signed: positive = actual action predicts MORE harm than CF action
                signed_causal   = float(z_pred_act.norm(dim=-1).mean().item()
                                        - z_pred_cf.norm(dim=-1).mean().item())
                unsigned_causal = float((z_pred_act - z_pred_cf).norm(dim=-1).mean().item())

                signed_causals[ttype].append(signed_causal)
                unsigned_causals[ttype].append(unsigned_causal)

            z_harm_s_prev = z_harm_s_now.detach().clone() if z_harm_s_now is not None else None
            obs_dict      = obs_dict_next

            if done:
                break

    # Summarise per type
    sign_correct:      Dict[str, float] = {}
    causal_sig_mean:   Dict[str, float] = {}
    causal_sig_counts: Dict[str, int]   = {}

    for ttype in HARM_TYPES:
        vals = signed_causals[ttype]
        n    = len(vals)
        causal_sig_counts[ttype] = n
        sign_correct[ttype]    = float(sum(v > 0 for v in vals) / max(1, n))
        causal_sig_mean[ttype] = float(np.mean(unsigned_causals[ttype])) if unsigned_causals[ttype] else 0.0

    return {
        "sign_correct":     sign_correct,
        "causal_sig_mean":  causal_sig_mean,
        "causal_sig_counts": causal_sig_counts,
    }


# ---------------------------------------------------------------------------
# Single condition run
# ---------------------------------------------------------------------------

def run_condition(seed: int, condition: str, dry_run: bool) -> Dict:
    print(f"Seed {seed}  Condition {condition}")
    env   = _make_env(seed)
    agent = _make_agent(env, seed)
    device = agent.device

    is_trained   = (condition == "TRAINED")
    harm_fwd     = _make_harm_fwd(device, use_interventional=is_trained)
    forward_r2   = 0.0

    if is_trained:
        forward_r2 = run_training(seed, env, agent, harm_fwd, dry_run)
        print(f"  [P1 done] seed={seed} forward_r2={forward_r2:.3f}")
    else:
        print(f"  [RANDOM] seed={seed} no training")

    # Separate env for clean evaluation (different seed offset)
    eval_env   = _make_env(seed + 1000)
    eval_agent = _make_agent(eval_env, seed)
    if is_trained:
        eval_agent.load_state_dict(agent.state_dict())

    eval_metrics = run_evaluation(seed, eval_env, eval_agent, harm_fwd, dry_run)

    sc  = eval_metrics["sign_correct"]
    cnt = eval_metrics["causal_sig_counts"]
    print(
        f"  [eval] seed={seed} cond={condition} "
        f"env_sc={sc.get('env_caused_hazard', 0.0):.3f}(n={cnt.get('env_caused_hazard', 0)}) "
        f"agent_sc={sc.get('agent_caused_hazard', 0.0):.3f}(n={cnt.get('agent_caused_hazard', 0)}) "
        f"approach_sc={sc.get('hazard_approach', 0.0):.3f}(n={cnt.get('hazard_approach', 0)})"
    )

    return {
        "seed":         seed,
        "condition":    condition,
        "forward_r2":   forward_r2,
        **eval_metrics,
    }


# ---------------------------------------------------------------------------
# Criteria
# ---------------------------------------------------------------------------

def evaluate_criteria(all_results: List[Dict]) -> Dict:
    by_cond: Dict[str, List[Dict]] = defaultdict(list)
    for r in all_results:
        by_cond[r["condition"]].append(r)

    trained_list = sorted(by_cond.get("TRAINED", []), key=lambda x: x["seed"])
    random_list  = sorted(by_cond.get("RANDOM",  []), key=lambda x: x["seed"])

    # C1: TRAINED forward_r2 >= 0.7
    c1_vals  = [r["forward_r2"] for r in trained_list]
    c1_seeds = sum(v >= C1_FORWARD_R2_THRESH for v in c1_vals)
    c1_pass  = c1_seeds >= MIN_SEEDS_PASS

    # C2: TRAINED sign_correct_env >= 0.65
    c2_vals  = [r["sign_correct"].get("env_caused_hazard", 0.0) for r in trained_list]
    c2_seeds = sum(v >= C2_SIGN_CORRECT_ENV_THRESH for v in c2_vals)
    c2_pass  = c2_seeds >= MIN_SEEDS_PASS

    # C3: TRAINED sign_correct_env > RANDOM sign_correct_env + 0.1 (learned)
    c3_seeds  = 0
    c3_lifts  = []
    for t, rnd in zip(trained_list, random_list):
        t_env   = t["sign_correct"].get("env_caused_hazard", 0.0)
        rnd_env = rnd["sign_correct"].get("env_caused_hazard", 0.0)
        lift    = t_env - rnd_env
        c3_lifts.append(lift)
        if lift >= C3_LEARNED_LIFT:
            c3_seeds += 1
    c3_pass = c3_seeds >= MIN_SEEDS_PASS

    # C4: sufficient stats (n_env >= 20, n_approach >= 20 in TRAINED)
    c4_seeds   = 0
    c4_details = []
    for t in trained_list:
        n_env      = t["causal_sig_counts"].get("env_caused_hazard", 0)
        n_approach = t["causal_sig_counts"].get("hazard_approach", 0)
        ok         = (n_env >= C4_MIN_ENV_EVENTS and n_approach >= C4_MIN_APPROACH_EVENTS)
        c4_details.append({"n_env": n_env, "n_approach": n_approach, "ok": ok})
        if ok:
            c4_seeds += 1
    c4_pass = c4_seeds >= MIN_SEEDS_PASS

    overall_pass = c1_pass and c2_pass and c3_pass and c4_pass

    # Diagnostics
    d1_agent_sc = [r["sign_correct"].get("agent_caused_hazard", 0.0) for r in trained_list]
    d1_n_agent  = [r["causal_sig_counts"].get("agent_caused_hazard", 0) for r in trained_list]
    d2_approach_trained = [r["sign_correct"].get("hazard_approach", 0.0) for r in trained_list]
    d2_approach_random  = [r["sign_correct"].get("hazard_approach", 0.0) for r in random_list]

    return {
        "c1_forward_r2_pass":  c1_pass,
        "c1_vals":             c1_vals,
        "c1_seeds_pass":       c1_seeds,
        "c2_sign_correct_env_pass": c2_pass,
        "c2_vals":             c2_vals,
        "c2_seeds_pass":       c2_seeds,
        "c3_learned_lift_pass": c3_pass,
        "c3_lifts":            c3_lifts,
        "c3_seeds_pass":       c3_seeds,
        "c4_sufficient_stats_pass": c4_pass,
        "c4_details":          c4_details,
        "c4_seeds_pass":       c4_seeds,
        "d1_agent_caused_sign_correct":   d1_agent_sc,
        "d1_agent_caused_n":              d1_n_agent,
        "d2_approach_sign_correct_trained": d2_approach_trained,
        "d2_approach_sign_correct_random":  d2_approach_random,
        "overall_pass":        overall_pass,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = (
        f"v3_exq_431_sd003_causal_discrimination_dry_{ts}_v3"
        if args.dry_run
        else f"v3_exq_431_sd003_causal_discrimination_{ts}_v3"
    )
    print(f"EXQ-431 start: {run_id}")

    all_results: List[Dict] = []
    for seed in SEEDS:
        for condition in CONDITIONS:
            result = run_condition(seed, condition, dry_run=args.dry_run)
            all_results.append(result)

    criteria = evaluate_criteria(all_results)
    outcome  = "PASS" if criteria["overall_pass"] else "FAIL"

    print(f"\n=== EXQ-431 {outcome} ===")
    print(f"C1 forward_r2 (>={C1_FORWARD_R2_THRESH}): {criteria['c1_forward_r2_pass']} "
          f"(vals={[f'{v:.3f}' for v in criteria['c1_vals']]})")
    print(f"C2 sign_correct_env (>={C2_SIGN_CORRECT_ENV_THRESH}): "
          f"{criteria['c2_sign_correct_env_pass']} "
          f"(vals={[f'{v:.3f}' for v in criteria['c2_vals']]})")
    print(f"C3 learned lift (>={C3_LEARNED_LIFT}): {criteria['c3_learned_lift_pass']} "
          f"(lifts={[f'{v:.3f}' for v in criteria['c3_lifts']]})")
    print(f"C4 sufficient stats: {criteria['c4_sufficient_stats_pass']} "
          f"(details={criteria['c4_details']})")
    print(f"D1 agent_caused sign_correct: "
          f"{[f'{v:.3f}' for v in criteria['d1_agent_caused_sign_correct']]} "
          f"(n={criteria['d1_agent_caused_n']})")
    print(f"D2 approach sign_correct trained/random: "
          f"{[f'{v:.3f}' for v in criteria['d2_approach_sign_correct_trained']]} / "
          f"{[f'{v:.3f}' for v in criteria['d2_approach_sign_correct_random']]}")

    # Per-claim direction
    # SD-003: attribution directionality -- C2 + C3 (trained > random, above threshold)
    # SD-013: forward model with interventional training -- C1
    sd003_pass = criteria["c2_sign_correct_env_pass"] and criteria["c3_learned_lift_pass"]
    sd013_pass = criteria["c1_forward_r2_pass"]

    output = {
        "run_id":            run_id,
        "experiment_type":   EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids":         CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "evidence_direction_per_claim": {
            "SD-003": "supports" if sd003_pass  else "does_not_support",
            "SD-013": "supports" if sd013_pass  else "does_not_support",
        },
        "evidence_direction": "supports" if criteria["overall_pass"] else "does_not_support",
        "outcome":           outcome,
        "criteria":          criteria,
        "results_per_condition": all_results,
        "config": {
            "seeds":                     SEEDS,
            "conditions":                CONDITIONS,
            "p0_episodes":               P0_EPISODES,
            "p1_episodes":               P1_EPISODES,
            "p2_episodes":               P2_EPISODES,
            "steps_per_ep":              STEPS_PER_EP,
            "grid_size":                 GRID_SIZE,
            "num_hazards":               NUM_HAZARDS,
            "hazard_harm":               HAZARD_HARM,
            "interventional_fraction":   INTERVENTIONAL_FRACTION,
            "c2_sign_correct_env_thresh": C2_SIGN_CORRECT_ENV_THRESH,
            "c3_learned_lift":           C3_LEARNED_LIFT,
            "c4_min_env_events":         C4_MIN_ENV_EVENTS,
            "c4_min_approach_events":    C4_MIN_APPROACH_EVENTS,
        },
        "timestamp_utc": ts,
    }

    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "..", "REE_assembly", "evidence", "experiments",
        EXPERIMENT_TYPE,
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{run_id}.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results -> {out_path}")


if __name__ == "__main__":
    main()
