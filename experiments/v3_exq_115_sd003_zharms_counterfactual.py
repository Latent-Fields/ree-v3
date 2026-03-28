#!/opt/local/bin/python3
"""
V3-EXQ-115 -- SD-003 z_harm_s Counterfactual Attribution Discriminative Pair

Claims: SD-003, ARC-033
Proposal: EXP-0009 / EVB-0008
Supersedes: EXQ-109 (pre-SD-011 era, single-seed, no matched-seed design)

SD-003 asserts: the agent attributes harm causally via a counterfactual comparison
entirely within the sensory-discriminative harm stream (z_harm_s):

    causal_sig = E3.harm_eval(z_harm_s_actual) - mean(E3.harm_eval(E2_harm_s(z_harm_s, a_cf)))

where E2_harm_s = HarmForwardModel(z_harm_s, action) -> z_harm_s_next  (ARC-033).

Key architectural context (SD-011):
  - z_harm_s: sensory-discriminative nociceptive stream (HarmEncoder output, Adelta-analog)
    Encodes immediate proximity/intensity; FORWARD-PREDICTABLE across actions.
  - z_harm_a: affective-motivational stream (AffectiveHarmEncoder output, C-fiber analog)
    Encodes accumulated threat state; NOT counterfactually modeled.
  - HarmBridge (z_world -> z_harm) is architecturally infeasible (EXQ-093/094: bridge_r2=0;
    z_world perp z_harm by SD-010 design). Do NOT use HarmBridge.
  - SD-003 redesigned post-SD-011: counterfactual operates on z_harm_s only.

EXQ-109 was the first SD-003 z_harm_s probe (single-seed, 3 seeds 42/43/44).
EXQ-115 redesign:
  1. Matched seeds [42, 123] across TRAINED vs ABLATED conditions (proposal requirement)
  2. Explicit TRAINED vs ABLATED discriminative pair per seed:
       TRAINED:  HarmForwardModel trained on (z_harm_s_t, action_t, z_harm_s_{t+1})
       ABLATED:  Same architecture, random init (never trained)
  3. Both conditions share identical environment seed, config, and evaluation
     trajectories -- same seed = same obstacle layout, same stochastic steps
  4. Aggregate verdict: mean metrics across matched seeds, same thresholds

Physics / expected direction:
  Moving toward a hazard increases z_harm_s. The trained HarmForwardModel predicts that
  counterfactual (away) actions would lead to LOWER z_harm_s_next. So for approach events:
    E3(z_harm_s_actual) is HIGH (proximate)
    mean(E3(z_harm_s_cf)) is LOW (counterfactual = staying away)
    causal_sig_trained >> causal_sig_random (random model predicts noise)
  => delta_approach = causal_approach_trained - causal_approach_ablated > 0

PRE-REGISTERED ACCEPTANCE CRITERIA (ALL required for PASS):
  C1 (gate): harm_fwd_r2_mean > 0.10
    Trained HarmForwardModel must learn z_harm_s dynamics before attribution can work.
    Without this gate, C2/C3 are uninterpretable.
  C2 (discrimination): delta_approach_mean > 0.005
    TRAINED causal_sig at approach events > ABLATED causal_sig, averaged across seeds.
    This is the primary SD-003 discriminative test.
  C3 (gradient ordering): causal_approach_trained_mean > causal_none_trained_mean
    Trained causal signal is larger at approach events than baseline (none events).
  C4 (escalation ordering): causal_contact_trained_mean > causal_approach_trained_mean
    Contact events produce larger causal signal than approach events (MECH-102 ordering).
  C5 (data quality): n_approach_eval_mean > 20
    Sufficient approach events per seed for reliable mean estimates.
  C6 (seed consistency): delta_approach > 0.0 for BOTH individual seeds
    Direction is consistent across seeds, not an artifact of one seed.

Decision scoring:
  retain_ree: C1 AND C2 AND C3 AND C4 AND C5 AND C6  -- full SD-003 + ARC-033 validation
  hybridize:  C1 AND C2 AND C5 AND C6, C3 or C4 fail  -- discrimination works, ordering unclear
  retire_ree_claim: C1 passes but C2 fails (delta_approach <= 0) AND C5 passes
                    -- trained model learns dynamics but adds no attribution signal
  inconclusive: C1 fails OR C5 fails  -- prerequisite unmet, result uninterpretable
"""

import sys
import random
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.latent.stack import HarmEncoder, HarmForwardModel
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_115_sd003_zharms_counterfactual"
CLAIM_IDS = ["SD-003", "ARC-033"]

# Pre-registered thresholds (MUST NOT be changed post-hoc)
THRESH_C1_HARM_FWD_R2      = 0.10   # trained forward model R2 -- gate
THRESH_C2_DELTA_APPROACH   = 0.005  # TRAINED - ABLATED causal_approach delta
THRESH_C3_GRAD_ORDERING    = 0.0    # causal_approach_trained > causal_none_trained
THRESH_C4_ESCALATION       = 0.0    # causal_contact_trained > causal_approach_trained
THRESH_C5_MIN_APPROACH_EVT = 20     # min approach events per seed eval
THRESH_C6_BOTH_SEEDS       = 0.0    # delta_approach per seed > 0 (sign consistent)

HARM_OBS_DIM = 51
Z_HARM_DIM   = 32
MAX_HF_DATA  = 3000


def _action_to_onehot(idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, idx] = 1.0
    return v


def _ttype_bucket(ttype: str) -> str:
    if ttype in ("env_caused_hazard", "agent_caused_hazard"):
        return "contact"
    elif ttype == "hazard_approach":
        return "approach"
    return "none"


def _hazard_approach_action(env: CausalGridWorldV2, n_actions: int) -> int:
    """Return action index that moves toward the nearest hazard gradient.
    Falls back to random if proxy fields are unavailable."""
    obs_dict = env._get_observation_dict()
    world_state = obs_dict.get("world_state", None)
    if world_state is None or not env.use_proxy_fields:
        return random.randint(0, n_actions - 1)
    # world_state[225:250] = hazard_field_view (5x5 flattened, proxy channel)
    field_view = world_state[225:250].numpy().reshape(5, 5)
    # Agent at center (2,2); actions: 0=up(-1,0), 1=down(+1,0), 2=left(0,-1), 3=right(0,+1)
    deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    vals = []
    for dr, dc in deltas:
        r, c = 2 + dr, 2 + dc
        if 0 <= r < 5 and 0 <= c < 5:
            vals.append(float(field_view[r, c]))
        else:
            vals.append(-1.0)
    return int(np.argmax(vals))


def _run_single(
    seed: int,
    phase1_episodes: int,
    phase2_episodes: int,
    eval_episodes: int,
    steps_per_episode: int,
    world_dim: int,
    self_dim: int,
    nav_bias: float,
    dry_run: bool,
) -> Dict:
    """Run one matched seed. Returns metrics for TRAINED and ABLATED conditions.

    Both conditions share the same seed, same HarmEncoder, same E3 harm head trained
    in P1/P2. Only the counterfactual model differs: TRAINED uses the learned
    HarmForwardModel; ABLATED uses a random-init model.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    if dry_run:
        phase1_episodes = min(3, phase1_episodes)
        phase2_episodes = min(2, phase2_episodes)
        eval_episodes   = min(2, eval_episodes)

    env = CausalGridWorldV2(
        seed=seed,
        size=6,
        num_hazards=4,
        num_resources=2,
        hazard_harm=0.02,
        env_drift_interval=10,
        env_drift_prob=0.1,
        proximity_harm_scale=0.05,
        proximity_benefit_scale=0.03,
        proximity_approach_threshold=0.15,
        hazard_field_decay=0.5,
        use_proxy_fields=True,
    )
    n_actions = env.action_dim

    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=self_dim,
        world_dim=world_dim,
        alpha_world=0.9,
        alpha_self=0.3,
        reafference_action_dim=env.action_dim,
        use_harm_stream=True,
        harm_obs_dim=HARM_OBS_DIM,
        z_harm_dim=Z_HARM_DIM,
    )
    agent = REEAgent(config)

    # SD-010: sensory-discriminative harm encoder (HarmEncoder = HarmEncoderS)
    harm_enc = HarmEncoder(harm_obs_dim=HARM_OBS_DIM, z_harm_dim=Z_HARM_DIM)
    # ARC-033: E2_harm_s analog -- trained forward model
    harm_fwd_trained = HarmForwardModel(z_harm_dim=Z_HARM_DIM, action_dim=n_actions)
    # ABLATED control: same architecture, random init, never trained
    harm_fwd_ablated = HarmForwardModel(z_harm_dim=Z_HARM_DIM, action_dim=n_actions)

    opt_harm = optim.Adam(
        list(harm_enc.parameters())
        + list(agent.e3.harm_eval_z_harm_head.parameters()),
        lr=1e-3,
    )
    opt_hf = optim.Adam(harm_fwd_trained.parameters(), lr=1e-3)
    opt_std = optim.Adam(
        [p for n, p in agent.named_parameters()
         if "harm_eval" not in n
         and "world_transition" not in n
         and "world_action_encoder" not in n],
        lr=1e-3,
    )

    hf_data: List = []
    n_approach_p1 = 0; n_contact_p1 = 0; n_none_p1 = 0

    # ---- Phase 1: train terrain + HarmEncoder + HarmForwardModel ---
    print(
        f"[EXQ-115 seed={seed}] P1: terrain+fwd training ({phase1_episodes} eps)...",
        flush=True,
    )
    agent.train(); harm_enc.train(); harm_fwd_trained.train()

    for ep in range(phase1_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()

        for _ in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            latent    = agent.sense(obs_body, obs_world)
            agent.clock.advance()

            harm_obs = obs_dict.get("harm_obs", torch.zeros(HARM_OBS_DIM)).float()
            with torch.no_grad():
                z_harm_curr = harm_enc(harm_obs.unsqueeze(0)).detach()

            # Navigation: biased toward hazard gradient to ensure approach/contact data
            if random.random() < nav_bias:
                action_idx = _hazard_approach_action(env, n_actions)
            else:
                action_idx = random.randint(0, n_actions - 1)
            action_oh = _action_to_onehot(action_idx, n_actions, agent.device)
            agent._last_action = action_oh

            flat_obs, _, done, info, obs_dict = env.step(action_oh)

            ttype = info.get("transition_type", "none")
            if ttype == "hazard_approach":   n_approach_p1 += 1
            elif ttype in ("env_caused_hazard", "agent_caused_hazard"): n_contact_p1 += 1
            else: n_none_p1 += 1

            # Collect (z_harm_t, action_t, z_harm_{t+1}) for HarmForwardModel
            harm_obs_next = obs_dict.get("harm_obs", torch.zeros(HARM_OBS_DIM)).float()
            with torch.no_grad():
                z_harm_next_target = harm_enc(harm_obs_next.unsqueeze(0)).detach()

            hf_data.append((z_harm_curr.cpu(), action_oh.detach().cpu(),
                            z_harm_next_target.cpu()))
            if len(hf_data) > MAX_HF_DATA:
                hf_data = hf_data[-MAX_HF_DATA:]

            # Train HarmEncoder + E3 harm head on proximity label (harm_obs[12])
            label = harm_obs[12].unsqueeze(0).unsqueeze(0)
            z_for_train = harm_enc(harm_obs.unsqueeze(0))
            pred_harm = agent.e3.harm_eval_z_harm(z_for_train)
            loss_he = F.mse_loss(pred_harm, label)
            opt_harm.zero_grad(); loss_he.backward(); opt_harm.step()

            # Train HarmForwardModel on accumulated replay buffer
            if len(hf_data) >= 16:
                k = min(32, len(hf_data))
                idxs = torch.randperm(len(hf_data))[:k].tolist()
                zh_b  = torch.cat([hf_data[i][0] for i in idxs]).to(agent.device)
                a_b   = torch.cat([hf_data[i][1] for i in idxs]).to(agent.device)
                zh1_b = torch.cat([hf_data[i][2] for i in idxs]).to(agent.device)
                hf_loss = F.mse_loss(harm_fwd_trained(zh_b, a_b), zh1_b)
                if hf_loss.requires_grad:
                    opt_hf.zero_grad(); hf_loss.backward(); opt_hf.step()

            # Standard agent losses (E1 + E2 world)
            e1_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()
            total = e1_loss + e2_loss
            if total.requires_grad:
                opt_std.zero_grad(); total.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                opt_std.step()

            if done: break

        if (ep + 1) % 50 == 0 or ep == phase1_episodes - 1:
            print(
                f"  [P1 seed={seed}] ep {ep+1}/{phase1_episodes}"
                f"  hf_buf={len(hf_data)}"
                f"  approach={n_approach_p1} contact={n_contact_p1} none={n_none_p1}",
                flush=True,
            )

    # --- Evaluate HarmForwardModel R2 on held-out steps -----------------
    harm_fwd_r2 = 0.0
    if len(hf_data) >= 50:
        with torch.no_grad():
            harm_enc.eval(); harm_fwd_trained.eval()
            flat_obs, obs_dict = env.reset()
            agent.reset()
            zh_list = []; zh_pred_list = []
            prev_zh = None; prev_a = None
            for _ in range(min(300, steps_per_episode * 5)):
                harm_obs = obs_dict.get("harm_obs", torch.zeros(HARM_OBS_DIM)).float()
                zh_curr = harm_enc(harm_obs.unsqueeze(0))
                if prev_zh is not None and prev_a is not None:
                    zh_pred = harm_fwd_trained(prev_zh, prev_a)
                    zh_list.append(zh_curr.cpu())
                    zh_pred_list.append(zh_pred.detach().cpu())
                prev_zh = zh_curr.detach()
                if random.random() < nav_bias:
                    a_idx = _hazard_approach_action(env, n_actions)
                else:
                    a_idx = random.randint(0, n_actions - 1)
                a_oh = _action_to_onehot(a_idx, n_actions, agent.device)
                agent._last_action = a_oh
                prev_a = a_oh.detach()
                latent = agent.sense(obs_dict["body_state"], obs_dict["world_state"])
                agent.clock.advance()
                flat_obs, _, done, _, obs_dict = env.step(a_oh)
                if done: break
            if len(zh_list) >= 20:
                zh_actual  = torch.cat(zh_list, dim=0)
                zh_pred_t  = torch.cat(zh_pred_list, dim=0)
                ss_res = ((zh_actual - zh_pred_t) ** 2).sum()
                ss_tot = ((zh_actual - zh_actual.mean(0, keepdim=True)) ** 2).sum()
                harm_fwd_r2 = float((1.0 - ss_res / (ss_tot + 1e-8)).item())
            harm_enc.train(); harm_fwd_trained.train()
    print(f"  [seed={seed}] HarmForwardModel R2 (trained): {harm_fwd_r2:.4f}", flush=True)

    # ---- Phase 2: calibrate E3 harm head (stratified, frozen everything else) ---
    print(
        f"[EXQ-115 seed={seed}] P2: E3 calibration ({phase2_episodes} eps)...",
        flush=True,
    )
    for p in agent.parameters(): p.requires_grad_(False)
    for p in harm_enc.parameters(): p.requires_grad_(False)
    for p in harm_fwd_trained.parameters(): p.requires_grad_(False)
    for p in agent.e3.harm_eval_z_harm_head.parameters(): p.requires_grad_(True)

    opt_e3 = optim.Adam(agent.e3.harm_eval_z_harm_head.parameters(), lr=1e-4)
    strat: Dict[str, list] = {"none": [], "approach": [], "contact": []}
    STRAT_MAX   = 500
    MIN_BUCKET  = 4
    SAMP_BUCKET = 8

    for ep in range(phase2_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        for _ in range(steps_per_episode):
            with torch.no_grad():
                latent = agent.sense(obs_dict["body_state"], obs_dict["world_state"])
                agent.clock.advance()
            if random.random() < nav_bias:
                a_idx = _hazard_approach_action(env, n_actions)
            else:
                a_idx = random.randint(0, n_actions - 1)
            a_oh = _action_to_onehot(a_idx, n_actions, agent.device)
            agent._last_action = a_oh
            flat_obs, _, done, info, obs_dict = env.step(a_oh)

            ttype    = info.get("transition_type", "none")
            harm_obs = obs_dict.get("harm_obs", torch.zeros(HARM_OBS_DIM)).float()
            label    = float(harm_obs[12].item())
            with torch.no_grad():
                zh = harm_enc(harm_obs.unsqueeze(0))
            bucket = _ttype_bucket(ttype)
            strat[bucket].append((zh.detach(), label))
            if len(strat[bucket]) > STRAT_MAX:
                strat[bucket] = strat[bucket][-STRAT_MAX:]

            buckets_ready = [b for b in strat if len(strat[b]) >= MIN_BUCKET]
            if len(buckets_ready) >= 2:
                zh_list_t = []; lbl_list = []
                for bk in strat:
                    buf = strat[bk]
                    if len(buf) < MIN_BUCKET: continue
                    k = min(SAMP_BUCKET, len(buf))
                    for i in random.sample(range(len(buf)), k):
                        zh_list_t.append(buf[i][0]); lbl_list.append(buf[i][1])
                if len(zh_list_t) >= 6:
                    zh_b  = torch.cat(zh_list_t, dim=0).to(agent.device)
                    lbl_b = torch.tensor(lbl_list, dtype=torch.float32,
                                         device=agent.device).unsqueeze(1)
                    pred = agent.e3.harm_eval_z_harm(zh_b)
                    loss = F.mse_loss(pred, lbl_b)
                    if loss.requires_grad:
                        opt_e3.zero_grad(); loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            agent.e3.harm_eval_z_harm_head.parameters(), 0.5)
                        opt_e3.step()
            if done: break

        if (ep + 1) % 25 == 0 or ep == phase2_episodes - 1:
            buf_sz = {k: len(v) for k, v in strat.items()}
            print(
                f"  [P2 seed={seed}] ep {ep+1}/{phase2_episodes}  strat={buf_sz}",
                flush=True,
            )

    # ---- Phase 3: Attribution eval (TRAINED vs ABLATED) --------------------
    print(
        f"[EXQ-115 seed={seed}] P3: attribution eval ({eval_episodes} eps)...",
        flush=True,
    )
    agent.eval(); harm_enc.eval(); harm_fwd_trained.eval()
    # harm_fwd_ablated was never trained -- stays at random init throughout

    causal_trained: Dict[str, List[float]] = {}
    causal_ablated: Dict[str, List[float]] = {}

    for ep in range(eval_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        for _ in range(steps_per_episode):
            with torch.no_grad():
                latent = agent.sense(obs_dict["body_state"], obs_dict["world_state"])
                agent.clock.advance()

                # z_harm_s BEFORE the action
                harm_obs_t = obs_dict.get("harm_obs", torch.zeros(HARM_OBS_DIM)).float()
                z_harm_t   = harm_enc(harm_obs_t.unsqueeze(0))

            if random.random() < nav_bias:
                action_idx = _hazard_approach_action(env, n_actions)
            else:
                action_idx = random.randint(0, n_actions - 1)
            a_oh = _action_to_onehot(action_idx, n_actions, agent.device)
            agent._last_action = a_oh
            flat_obs, _, done, info, obs_dict = env.step(a_oh)

            ttype  = info.get("transition_type", "none")
            bucket = _ttype_bucket(ttype)

            with torch.no_grad():
                # Actual outcome: z_harm_s AFTER the action
                harm_obs_t1 = obs_dict.get("harm_obs", torch.zeros(HARM_OBS_DIM)).float()
                z_harm_t1   = harm_enc(harm_obs_t1.unsqueeze(0))
                harm_eval_actual = float(agent.e3.harm_eval_z_harm(z_harm_t1).item())

                # TRAINED counterfactual: SD-003 causal_sig formula
                #   causal_sig = E3(z_harm_s_actual) - mean(E3(E2_harm_s(z_harm_s, a_cf)))
                cf_vals_trained = []
                for ci in range(n_actions):
                    a_cf = _action_to_onehot(ci, n_actions, agent.device)
                    z_cf = harm_fwd_trained(z_harm_t, a_cf)
                    cf_vals_trained.append(
                        float(agent.e3.harm_eval_z_harm(z_cf).item()))
                causal_sig_trained = harm_eval_actual - float(np.mean(cf_vals_trained))

                # ABLATED counterfactual: same formula, random-init forward model
                cf_vals_ablated = []
                for ci in range(n_actions):
                    a_cf = _action_to_onehot(ci, n_actions, agent.device)
                    z_cf = harm_fwd_ablated(z_harm_t, a_cf)
                    cf_vals_ablated.append(
                        float(agent.e3.harm_eval_z_harm(z_cf).item()))
                causal_sig_ablated = harm_eval_actual - float(np.mean(cf_vals_ablated))

            causal_trained.setdefault(bucket, []).append(causal_sig_trained)
            causal_ablated.setdefault(bucket, []).append(causal_sig_ablated)
            if done: break

    def _m(lst): return float(np.mean(lst)) if lst else 0.0

    causal_approach_trained = _m(causal_trained.get("approach", []))
    causal_none_trained     = _m(causal_trained.get("none",     []))
    causal_contact_trained  = _m(causal_trained.get("contact",  []))
    causal_approach_ablated = _m(causal_ablated.get("approach", []))

    n_approach_eval = len(causal_trained.get("approach", []))
    n_contact_eval  = len(causal_trained.get("contact",  []))
    n_none_eval     = len(causal_trained.get("none",     []))
    delta_approach  = causal_approach_trained - causal_approach_ablated

    print(f"\n  --- EXQ-115 seed={seed} results ---", flush=True)
    print(f"  harm_fwd_r2:              {harm_fwd_r2:.4f}", flush=True)
    print(f"  causal_approach_trained:  {causal_approach_trained:.6f}", flush=True)
    print(f"  causal_approach_ablated:  {causal_approach_ablated:.6f}", flush=True)
    print(f"  delta_approach (C2):      {delta_approach:.6f}", flush=True)
    print(f"  causal_none_trained:      {causal_none_trained:.6f}", flush=True)
    print(f"  causal_contact_trained:   {causal_contact_trained:.6f}", flush=True)
    print(f"  n_approach_eval:          {n_approach_eval}", flush=True)
    print(f"  n_contact_eval:           {n_contact_eval}", flush=True)
    print(f"  n_none_eval:              {n_none_eval}", flush=True)

    return {
        "seed":                      seed,
        "harm_fwd_r2":               float(harm_fwd_r2),
        "causal_approach_trained":   float(causal_approach_trained),
        "causal_approach_ablated":   float(causal_approach_ablated),
        "delta_approach":            float(delta_approach),
        "causal_none_trained":       float(causal_none_trained),
        "causal_contact_trained":    float(causal_contact_trained),
        "n_approach_eval":           float(n_approach_eval),
        "n_contact_eval":            float(n_contact_eval),
        "n_none_eval":               float(n_none_eval),
        "n_approach_p1":             float(n_approach_p1),
        "n_contact_p1":              float(n_contact_p1),
        "n_none_p1":                 float(n_none_p1),
    }


def main():
    import json
    import sys
    import time
    from datetime import datetime, timezone

    dry_run = "--dry-run" in sys.argv

    SEEDS = [42, 123]  # matched seeds (EVB-0008 min_shared_seeds=2)

    PHASE1_EPISODES    = 5 if dry_run else 150
    PHASE2_EPISODES    = 2 if dry_run else 75
    EVAL_EPISODES      = 2 if dry_run else 50
    STEPS_PER_EPISODE  = 200
    WORLD_DIM          = 32
    SELF_DIM           = 32
    NAV_BIAS           = 0.60  # 60% toward-hazard, 40% random

    OUT_DIR = (
        Path(__file__).resolve().parents[1].parent
        / "REE_assembly" / "evidence" / "experiments"
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(
        f"[V3-EXQ-115] SD-003 z_harm_s counterfactual discriminative pair\n"
        f"  Seeds: {SEEDS}  nav_bias={NAV_BIAS}  dry_run={dry_run}\n"
        f"  Grid: 6x6, 4 hazards\n"
        f"  Pre-registered thresholds:\n"
        f"    C1 harm_fwd_r2>{THRESH_C1_HARM_FWD_R2}  C2 delta_approach>{THRESH_C2_DELTA_APPROACH}\n"
        f"    C3 approach>none  C4 contact>approach  C5 n_approach>{THRESH_C5_MIN_APPROACH_EVT}"
        f"  C6 both seeds delta>0",
        flush=True,
    )

    all_results = []
    for seed in SEEDS:
        print(f"\n{'='*60}", flush=True)
        print(f"SEED {seed}", flush=True)
        print(f"{'='*60}", flush=True)
        result = _run_single(
            seed=seed,
            phase1_episodes=PHASE1_EPISODES,
            phase2_episodes=PHASE2_EPISODES,
            eval_episodes=EVAL_EPISODES,
            steps_per_episode=STEPS_PER_EPISODE,
            world_dim=WORLD_DIM,
            self_dim=SELF_DIM,
            nav_bias=NAV_BIAS,
            dry_run=dry_run,
        )
        all_results.append(result)

    # ---- Aggregate across matched seeds ---
    def _agg(key):
        vals = [r[key] for r in all_results if key in r]
        return float(np.mean(vals)) if vals else 0.0

    harm_fwd_r2_mean           = _agg("harm_fwd_r2")
    delta_approach_mean        = _agg("delta_approach")
    causal_approach_t_mean     = _agg("causal_approach_trained")
    causal_none_t_mean         = _agg("causal_none_trained")
    causal_contact_t_mean      = _agg("causal_contact_trained")
    causal_approach_abl_mean   = _agg("causal_approach_ablated")
    n_approach_mean            = _agg("n_approach_eval")

    c1 = harm_fwd_r2_mean        > THRESH_C1_HARM_FWD_R2
    c2 = delta_approach_mean     > THRESH_C2_DELTA_APPROACH
    c3 = causal_approach_t_mean  > causal_none_t_mean    + THRESH_C3_GRAD_ORDERING
    c4 = causal_contact_t_mean   > causal_approach_t_mean + THRESH_C4_ESCALATION
    c5 = n_approach_mean         > THRESH_C5_MIN_APPROACH_EVT
    c6 = all(r["delta_approach"] > THRESH_C6_BOTH_SEEDS for r in all_results)

    n_met  = sum([c1, c2, c3, c4, c5, c6])
    status = "PASS" if all([c1, c2, c3, c4, c5, c6]) else "FAIL"

    failure_notes = []
    if not c1: failure_notes.append(
        f"C1 FAIL: harm_fwd_r2_mean={harm_fwd_r2_mean:.4f} <= {THRESH_C1_HARM_FWD_R2}"
        f" (HarmForwardModel not learning z_harm_s dynamics -- ARC-033 gate not met)")
    if not c2: failure_notes.append(
        f"C2 FAIL: delta_approach_mean={delta_approach_mean:.6f} <= {THRESH_C2_DELTA_APPROACH}"
        f" (trained counterfactual adds no signal vs ablated for approach events)")
    if not c3: failure_notes.append(
        f"C3 FAIL: causal_approach_trained={causal_approach_t_mean:.6f}"
        f" not > causal_none_trained={causal_none_t_mean:.6f}"
        f" (gradient ordering broken: approach events not elevated vs baseline)")
    if not c4: failure_notes.append(
        f"C4 FAIL: causal_contact_trained={causal_contact_t_mean:.6f}"
        f" not > causal_approach_trained={causal_approach_t_mean:.6f}"
        f" (escalation ordering: contact events should exceed approach)")
    if not c5: failure_notes.append(
        f"C5 FAIL: n_approach_eval_mean={n_approach_mean:.1f} <= {THRESH_C5_MIN_APPROACH_EVT}"
        f" (insufficient approach events -- result not reliable)")
    if not c6:
        per_seed_deltas = {r['seed']: r['delta_approach'] for r in all_results}
        failure_notes.append(
            f"C6 FAIL: seed consistency broken -- per-seed deltas: {per_seed_deltas}"
            f" (direction not consistent across both seeds)")

    print(f"\nV3-EXQ-115 aggregate verdict: {status}  ({n_met}/6 criteria)", flush=True)
    for note in failure_notes:
        print(f"  {note}", flush=True)

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(
            f"- {n}" for n in failure_notes)

    per_seed_table = ""
    for r in all_results:
        per_seed_table += (
            f"| {r['seed']} | {r['harm_fwd_r2']:.4f} |"
            f" {r['causal_approach_trained']:.5f} | {r['causal_approach_ablated']:.5f} |"
            f" {r['delta_approach']:.5f} | {int(r['n_approach_eval'])} |\n"
        )

    summary_markdown = f"""# V3-EXQ-115 -- SD-003 z_harm_s Counterfactual Attribution

**Status:** {status}
**Claims:** SD-003, ARC-033
**Supersedes:** EXQ-109 (pre-SD-011 era, single z_harm stream, no matched-seed design)

## Design

Discriminative pair: TRAINED vs ABLATED HarmForwardModel (E2_harm_s analog).
Both conditions use the same HarmEncoder + E3 harm_eval head trained identically in P1/P2.
Seeds matched across conditions: {SEEDS}.

SD-003 pipeline:
  causal_sig = E3(z_harm_s_actual) - mean(E3(E2_harm_s(z_harm_s, a_cf)))
ABLATED: random-init forward model never trained -- produces noise predictions.

## Pre-Registered Criteria

| Criterion | Threshold | Value | Pass |
|-----------|-----------|-------|------|
| C1 harm_fwd_r2 | >{THRESH_C1_HARM_FWD_R2} | {harm_fwd_r2_mean:.4f} | {'yes' if c1 else 'no'} |
| C2 delta_approach | >{THRESH_C2_DELTA_APPROACH} | {delta_approach_mean:.6f} | {'yes' if c2 else 'no'} |
| C3 approach>none (trained) | >0 | {causal_approach_t_mean - causal_none_t_mean:.6f} | {'yes' if c3 else 'no'} |
| C4 contact>approach (trained) | >0 | {causal_contact_t_mean - causal_approach_t_mean:.6f} | {'yes' if c4 else 'no'} |
| C5 n_approach_eval | >{THRESH_C5_MIN_APPROACH_EVT} | {n_approach_mean:.1f} | {'yes' if c5 else 'no'} |
| C6 both seeds delta>0 | sign consistent | {'yes' if c6 else 'no'} | {'yes' if c6 else 'no'} |

## Pairwise Comparison

| Seed | harm_fwd_r2 | causal_approach_trained | causal_approach_ablated | delta | n_approach |
|------|-------------|-------------------------|-------------------------|-------|------------|
{per_seed_table}
| **mean** | **{harm_fwd_r2_mean:.4f}** | **{causal_approach_t_mean:.5f}** | **{causal_approach_abl_mean:.5f}** | **{delta_approach_mean:.5f}** | **{n_approach_mean:.1f}** |
{failure_section}
"""

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"

    pack = {
        "experiment_type":    EXPERIMENT_TYPE,
        "run_id":             run_id,
        "run_timestamp":      ts,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids":          CLAIM_IDS,
        "claim_ids_tested":   CLAIM_IDS,
        "evidence_class":     "discriminative_pair",
        "evidence_direction": "supports" if status == "PASS" else "weakens",
        "seeds":              SEEDS,
        "n_seeds":            float(len(SEEDS)),
        "grid_size":          6,
        "n_hazards":          4,
        "nav_bias":           NAV_BIAS,
        "registered_thresholds": {
            "c1_harm_fwd_r2":       THRESH_C1_HARM_FWD_R2,
            "c2_delta_approach":    THRESH_C2_DELTA_APPROACH,
            "c3_grad_ordering":     THRESH_C3_GRAD_ORDERING,
            "c4_escalation":        THRESH_C4_ESCALATION,
            "c5_min_approach_evt":  THRESH_C5_MIN_APPROACH_EVT,
            "c6_seed_consistency":  THRESH_C6_BOTH_SEEDS,
        },
        "metrics": {
            "harm_fwd_r2":                   harm_fwd_r2_mean,
            "delta_approach":                delta_approach_mean,
            "causal_approach_trained":       causal_approach_t_mean,
            "causal_approach_ablated":       causal_approach_abl_mean,
            "causal_none_trained":           causal_none_t_mean,
            "causal_contact_trained":        causal_contact_t_mean,
            "n_approach_eval":               n_approach_mean,
            "crit1_pass":                    1.0 if c1 else 0.0,
            "crit2_pass":                    1.0 if c2 else 0.0,
            "crit3_pass":                    1.0 if c3 else 0.0,
            "crit4_pass":                    1.0 if c4 else 0.0,
            "crit5_pass":                    1.0 if c5 else 0.0,
            "crit6_pass":                    1.0 if c6 else 0.0,
            "criteria_met":                  float(n_met),
            "fatal_error_count":             0.0,
        },
        "status":   status,
        "verdict":  status,
        "scenario": "SD-003 z_harm_s counterfactual: TRAINED vs ABLATED HarmForwardModel",
        "interpretation": (
            "PASS: learned E2_harm_s (HarmForwardModel) adds systematic causal signal"
            " for approach events vs ablated baseline -- SD-003 z_harm_s pipeline"
            " validated, ARC-033 forward model trainable."
            " FAIL C1: HarmForwardModel not learning z_harm_s dynamics (ARC-033 gate"
            " not met; deeper architectural issue)."
            " FAIL C2 with C1 pass: forward model learns dynamics but pipeline produces"
            " no discrimination -- SD-003 z_harm_s architecture insufficient."
        ),
        "pairwise_deltas": {
            "causal_approach_trained_minus_ablated": float(delta_approach_mean),
        },
        "per_seed_results": all_results,
        "summary": summary_markdown,
    }

    exp_dir = OUT_DIR / EXPERIMENT_TYPE
    exp_dir.mkdir(exist_ok=True)
    out_path = exp_dir / f"{run_id}.json"
    with open(out_path, "w") as f:
        json.dump(pack, f, indent=2)

    print(f"\n{'='*60}", flush=True)
    print(f"AGGREGATE ({len(SEEDS)} seeds) verdict: {status}", flush=True)
    print(f"  harm_fwd_r2_mean:          {harm_fwd_r2_mean:.4f}", flush=True)
    print(f"  delta_approach_mean (C2):  {delta_approach_mean:.6f}", flush=True)
    print(f"  causal_approach_trained:   {causal_approach_t_mean:.6f}", flush=True)
    print(f"  causal_approach_ablated:   {causal_approach_abl_mean:.6f}", flush=True)
    print(f"  causal_none_trained:       {causal_none_t_mean:.6f}", flush=True)
    print(f"  causal_contact_trained:    {causal_contact_t_mean:.6f}", flush=True)
    print(f"  n_approach_eval_mean:      {n_approach_mean:.1f}", flush=True)
    print(f"  criteria_met:              {n_met}/6", flush=True)
    print(f"\nResults written to: {out_path}", flush=True)


if __name__ == "__main__":
    main()
