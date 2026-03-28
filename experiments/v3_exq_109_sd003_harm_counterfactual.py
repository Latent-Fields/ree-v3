"""
V3-EXQ-109 -- SD-003 Redesign: Harm-Stream Counterfactual Attribution (Discriminative Pair)

Claims: SD-003, ARC-033

This experiment provides decision-grade discriminative evidence for SD-003 and ARC-033
by comparing a TRAINED HarmForwardModel counterfactual against an UNTRAINED (random-init)
baseline within the same run.

Background:
SD-003 redesign (post SD-010 + SD-011): the counterfactual pipeline operates entirely
within the harm stream. HarmBridge (z_world -> z_harm) was confirmed infeasible in
EXQ-093/094 (bridge_r2=0; z_world perp z_harm by SD-010 design).

Post-SD-011 pipeline:
    z_harm_cf = HarmForwardModel(z_harm_actual, a_cf)
    causal_sig = E3.harm_eval_z_harm(z_harm_actual) - mean(E3.harm_eval_z_harm(z_harm_cf))

EXQ-095/095a/095b showed harm_fwd_r2=0 with a 12x12 grid and random navigation:
hazard coverage too sparse for the forward model to learn action-conditional structure.
Root cause confirmed: harm_obs_s sparsity -> R2=0 trivially from conditional mean.

Fixes in EXQ-109:
  1. Smaller grid (6x6 instead of 12x12) -> denser hazard coverage per episode
  2. Navigation bias: 60% proximity-following (move toward nearest hazard gradient),
     40% random -> ensures approach/contact transitions in training data
  3. Fewer hazards (4) to avoid additive field clip saturation (EXQ-102: 6 hazards
     causes clip inversion with clipped fields)
  4. Within-run discriminative pair: both TRAINED and RANDOM forward models evaluated
     on the same eval trajectories (same trained HarmEncoder + E3 harm head)

Discriminative pair design:
  TRAINED: causal_sig = harm_eval(z_harm_actual) - mean(harm_eval(harm_fwd_trained(z_harm, a_cf)))
  RANDOM:  causal_sig = harm_eval(z_harm_actual) - mean(harm_eval(harm_fwd_random(z_harm, a_cf)))

For approach transitions, TRAINED should give systematically larger positive causal_sig than
RANDOM. The physics: moving toward a hazard increases z_harm; the trained model predicts the
counterfactual (other actions lead to lower z_harm); the random model predicts noise.

PRE-REGISTERED ACCEPTANCE CRITERIA (ALL required for PASS):
  C1 (gate): harm_fwd_r2 > 0.10          -- trained model must learn harm dynamics
  C2: delta_approach > 0.005              -- TRAINED causal_approach > RANDOM causal_approach
  C3: causal_approach_trained > causal_none_trained  -- gradient ordering preserved
  C4: causal_contact_trained > causal_approach_trained  -- MECH-102 escalation ordering
  C5: n_approach_eval > 20               -- enough approach events to trust averages

If C1 fails even with the denser grid + navigation bias: indicates a deeper issue with
HarmForwardModel trainability at this scale (ARC-033 not yet achievable). Expected
diagnostic value regardless of PASS/FAIL.
"""

import sys
import random
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.latent.stack import HarmEncoder, HarmForwardModel
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_109_sd003_harm_counterfactual"
CLAIM_IDS = ["SD-003", "ARC-033"]

HARM_OBS_DIM = 51
Z_HARM_DIM   = 32
MAX_HF_DATA  = 3000


# Pre-registered thresholds (must NOT be changed post-hoc)
THRESH_C1_HARM_FWD_R2      = 0.10   # trained forward model R2
THRESH_C2_DELTA_APPROACH   = 0.005  # TRAINED - RANDOM causal_approach delta
THRESH_C3_GRAD_ORDERING    = 0.0    # causal_approach_trained > causal_none_trained
THRESH_C4_ESCALATION       = 0.0    # causal_contact_trained > causal_approach_trained
THRESH_C5_MIN_APPROACH_EVT = 20     # minimum approach events in eval


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


def _hazard_approach_action(env, n_actions: int) -> int:
    """Return action index that moves the agent toward the nearest hazard gradient.
    Falls back to random if no gradient information available."""
    obs_dict = env._get_observation_dict()
    # hazard_field_view is the last 25 dims of world_state (indices 225:250)
    world_state = obs_dict.get("world_state", None)
    if world_state is None or not env.use_proxy_fields:
        return random.randint(0, n_actions - 1)
    # world_state[225:250] = hazard_field_view (5x5 flattened)
    field_view = world_state[225:250].numpy().reshape(5, 5)
    # Agent is at center (2,2); check adjacent cells to pick highest hazard
    # Actions: 0=up(-1,0), 1=down(+1,0), 2=left(0,-1), 3=right(0,+1)
    deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    vals = []
    for dr, dc in deltas:
        r, c = 2 + dr, 2 + dc
        if 0 <= r < 5 and 0 <= c < 5:
            vals.append(float(field_view[r, c]))
        else:
            vals.append(-1.0)
    best = int(np.argmax(vals))
    # Use the toward-hazard action with 60% probability, random otherwise
    return best


def run(
    seed: int = 0,
    phase1_episodes: int = 150,
    phase2_episodes: int = 75,
    eval_episodes: int = 50,
    steps_per_episode: int = 200,
    world_dim: int = 32,
    self_dim: int = 32,
    nav_bias: float = 0.60,
) -> dict:
    """
    nav_bias: probability of taking toward-hazard action instead of random.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

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
    n_actions = env.action_dim

    # SD-010: sensory-discriminative harm encoder
    harm_enc = HarmEncoder(harm_obs_dim=HARM_OBS_DIM, z_harm_dim=Z_HARM_DIM)
    # ARC-033: HarmForwardModel (E2_harm_s analog)
    harm_fwd_trained = HarmForwardModel(z_harm_dim=Z_HARM_DIM, action_dim=n_actions)
    # ABLATION: random-init forward model (never trained) -- same architecture
    harm_fwd_random = HarmForwardModel(z_harm_dim=Z_HARM_DIM, action_dim=n_actions)

    print(
        f"[V3-EXQ-109] SD-003 discriminative pair: trained vs random HarmForwardModel\n"
        f"  Grid: 6x6, 4 hazards, nav_bias={nav_bias}\n"
        f"  Phases: P1 terrain+fwd({phase1_episodes} eps) ->"
        f" P2 E3 calib({phase2_episodes} eps) ->"
        f" P3 eval({eval_episodes} eps)\n"
        f"  Pre-registered thresholds:"
        f" C1 harm_fwd_r2>{THRESH_C1_HARM_FWD_R2},"
        f" C2 delta_approach>{THRESH_C2_DELTA_APPROACH},"
        f" C3 approach>none, C4 contact>approach, C5 n_approach>{THRESH_C5_MIN_APPROACH_EVT}",
        flush=True,
    )

    opt_harm = optim.Adam(
        list(harm_enc.parameters())
        + list(agent.e3.harm_eval_z_harm_head.parameters()),
        lr=1e-3,
    )
    opt_hf = optim.Adam(harm_fwd_trained.parameters(), lr=1e-3)
    opt_std = optim.Adam(
        [p for n, p in agent.named_parameters()
         if "harm_eval" not in n and "world_transition" not in n
         and "world_action_encoder" not in n],
        lr=1e-3,
    )

    hf_data: List = []

    # ---- Phase 1: terrain + HarmForwardModel training -----------------------
    print(f"\n[P1] Terrain + HarmForwardModel training ({phase1_episodes} eps)...",
          flush=True)
    agent.train(); harm_enc.train(); harm_fwd_trained.train()
    n_approach_p1 = 0; n_contact_p1 = 0; n_none_p1 = 0

    for ep in range(phase1_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        z_harm_prev = None
        a_prev_oh   = None

        for _ in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            latent    = agent.sense(obs_body, obs_world)
            agent.clock.advance()

            harm_obs = obs_dict.get("harm_obs", torch.zeros(HARM_OBS_DIM)).float()
            with torch.no_grad():
                z_harm_curr = harm_enc(harm_obs.unsqueeze(0)).detach()

            # Navigation: biased toward hazard gradient
            if random.random() < nav_bias:
                action_idx = _hazard_approach_action(env, n_actions)
            else:
                action_idx = random.randint(0, n_actions - 1)
            action_oh = _action_to_onehot(action_idx, n_actions, agent.device)
            agent._last_action = action_oh

            flat_obs, _, done, info, obs_dict = env.step(action_oh)

            ttype = info.get("transition_type", "none")
            if ttype == "hazard_approach": n_approach_p1 += 1
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

            # Train HarmEncoder + E3 harm head on proximity label
            label = harm_obs[12].unsqueeze(0).unsqueeze(0)
            z_for_train = harm_enc(harm_obs.unsqueeze(0))
            pred_harm = agent.e3.harm_eval_z_harm(z_for_train)
            loss_he = F.mse_loss(pred_harm, label)
            opt_harm.zero_grad(); loss_he.backward(); opt_harm.step()

            # Train HarmForwardModel on accumulated buffer
            if len(hf_data) >= 16:
                k = min(32, len(hf_data))
                idxs = torch.randperm(len(hf_data))[:k].tolist()
                zh_b  = torch.cat([hf_data[i][0] for i in idxs]).to(agent.device)
                a_b   = torch.cat([hf_data[i][1] for i in idxs]).to(agent.device)
                zh1_b = torch.cat([hf_data[i][2] for i in idxs]).to(agent.device)
                hf_loss = F.mse_loss(harm_fwd_trained(zh_b, a_b), zh1_b)
                if hf_loss.requires_grad:
                    opt_hf.zero_grad(); hf_loss.backward(); opt_hf.step()

            # Standard agent losses
            e1_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()
            total = e1_loss + e2_loss
            if total.requires_grad:
                opt_std.zero_grad(); total.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                opt_std.step()

            z_harm_prev = z_harm_curr
            a_prev_oh   = action_oh.detach()
            if done: break

        if (ep + 1) % 50 == 0 or ep == phase1_episodes - 1:
            print(
                f"  [P1] ep {ep+1}/{phase1_episodes}"
                f"  hf_buf={len(hf_data)}"
                f"  approach={n_approach_p1} contact={n_contact_p1} none={n_none_p1}",
                flush=True,
            )

    # Evaluate HarmForwardModel R2
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
                zh_actual = torch.cat(zh_list, dim=0)
                zh_pred_t  = torch.cat(zh_pred_list, dim=0)
                ss_res = ((zh_actual - zh_pred_t) ** 2).sum()
                ss_tot = ((zh_actual - zh_actual.mean(0, keepdim=True)) ** 2).sum()
                harm_fwd_r2 = float((1.0 - ss_res / (ss_tot + 1e-8)).item())
            harm_enc.train(); harm_fwd_trained.train()
    print(f"  HarmForwardModel R2 (trained): {harm_fwd_r2:.4f}", flush=True)

    # ---- Phase 2: E3 harm head calibration (stratified) --------------------
    print(f"\n[P2] E3 calibration ({phase2_episodes} eps)...", flush=True)
    for p in agent.parameters(): p.requires_grad_(False)
    for p in harm_enc.parameters(): p.requires_grad_(False)
    for p in harm_fwd_trained.parameters(): p.requires_grad_(False)
    for p in agent.e3.harm_eval_z_harm_head.parameters(): p.requires_grad_(True)

    opt_e3 = optim.Adam(agent.e3.harm_eval_z_harm_head.parameters(), lr=1e-4)
    strat: Dict[str, list] = {"none": [], "approach": [], "contact": []}
    STRAT_MAX = 500
    MIN_BUCKET = 4
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
            print(f"  [P2] ep {ep+1}/{phase2_episodes}  strat={buf_sz}", flush=True)

    # ---- Phase 3: Attribution eval (TRAINED vs RANDOM) ----------------------
    print(f"\n[P3] Attribution eval ({eval_episodes} eps)...", flush=True)
    agent.eval(); harm_enc.eval(); harm_fwd_trained.eval()
    # harm_fwd_random was never trained -- stays at random init

    causal_trained: Dict[str, List[float]] = {}
    causal_random:  Dict[str, List[float]] = {}
    harm_by:        Dict[str, List[float]] = {}

    for ep in range(eval_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        for _ in range(steps_per_episode):
            with torch.no_grad():
                latent = agent.sense(obs_dict["body_state"], obs_dict["world_state"])
                agent.clock.advance()

                # Record z_harm BEFORE the action (state from which agent acts)
                harm_obs_t = obs_dict.get("harm_obs", torch.zeros(HARM_OBS_DIM)).float()
                z_harm_t   = harm_enc(harm_obs_t.unsqueeze(0))

            if random.random() < nav_bias:
                a_idx = _hazard_approach_action(env, n_actions)
            else:
                a_idx = random.randint(0, n_actions - 1)
            a_oh = _action_to_onehot(a_idx, n_actions, agent.device)
            agent._last_action = a_oh
            flat_obs, _, done, info, obs_dict = env.step(a_oh)

            ttype = info.get("transition_type", "none")
            bucket = _ttype_bucket(ttype)

            with torch.no_grad():
                # Actual outcome: z_harm AFTER taking the action
                harm_obs_t1 = obs_dict.get("harm_obs", torch.zeros(HARM_OBS_DIM)).float()
                z_harm_t1   = harm_enc(harm_obs_t1.unsqueeze(0))
                harm_eval_actual = float(agent.e3.harm_eval_z_harm(z_harm_t1).item())

                # TRAINED: counterfactual via trained HarmForwardModel
                # Compare actual outcome to mean predicted CF from state z_harm_t
                # (SD-003: causal_sig = E3(actual_next) - mean(E3(harm_fwd(z_t, a_cf))))
                cf_vals_trained = []
                for ci in range(n_actions):
                    a_cf = _action_to_onehot(ci, n_actions, agent.device)
                    z_cf = harm_fwd_trained(z_harm_t, a_cf)
                    cf_vals_trained.append(float(agent.e3.harm_eval_z_harm(z_cf).item()))
                causal_sig_trained = harm_eval_actual - float(np.mean(cf_vals_trained))

                # RANDOM: counterfactual via random-init HarmForwardModel
                cf_vals_random = []
                for ci in range(n_actions):
                    a_cf = _action_to_onehot(ci, n_actions, agent.device)
                    z_cf = harm_fwd_random(z_harm_t, a_cf)
                    cf_vals_random.append(float(agent.e3.harm_eval_z_harm(z_cf).item()))
                causal_sig_random = harm_eval_actual - float(np.mean(cf_vals_random))

            causal_trained.setdefault(bucket, []).append(causal_sig_trained)
            causal_random.setdefault(bucket,  []).append(causal_sig_random)
            harm_by.setdefault(bucket,         []).append(harm_eval_actual)
            if done: break

    def _m(lst): return float(np.mean(lst)) if lst else 0.0

    cat = causal_trained; car = causal_random
    causal_approach_trained = _m(cat.get("approach", []))
    causal_none_trained     = _m(cat.get("none",     []))
    causal_contact_trained  = _m(cat.get("contact",  []))
    causal_approach_random  = _m(car.get("approach", []))

    n_approach_eval = len(cat.get("approach", []))
    n_contact_eval  = len(cat.get("contact",  []))
    n_none_eval     = len(cat.get("none",     []))
    delta_approach  = causal_approach_trained - causal_approach_random

    print(f"\n  --- V3-EXQ-109 results ---", flush=True)
    print(f"  harm_fwd_r2:              {harm_fwd_r2:.4f}", flush=True)
    print(f"  causal_approach_trained:  {causal_approach_trained:.6f}", flush=True)
    print(f"  causal_approach_random:   {causal_approach_random:.6f}", flush=True)
    print(f"  delta_approach (C2):      {delta_approach:.6f}", flush=True)
    print(f"  causal_none_trained:      {causal_none_trained:.6f}", flush=True)
    print(f"  causal_contact_trained:   {causal_contact_trained:.6f}", flush=True)
    print(f"  n_approach_eval:          {n_approach_eval}", flush=True)
    print(f"  n_contact_eval:           {n_contact_eval}", flush=True)
    print(f"  n_none_eval:              {n_none_eval}", flush=True)

    c1 = harm_fwd_r2               > THRESH_C1_HARM_FWD_R2
    c2 = delta_approach             > THRESH_C2_DELTA_APPROACH
    c3 = causal_approach_trained    > causal_none_trained + THRESH_C3_GRAD_ORDERING
    c4 = causal_contact_trained     > causal_approach_trained + THRESH_C4_ESCALATION
    c5 = n_approach_eval            > THRESH_C5_MIN_APPROACH_EVT

    n_met   = sum([c1, c2, c3, c4, c5])
    status  = "PASS" if all([c1, c2, c3, c4, c5]) else "FAIL"

    failure_notes = []
    if not c1: failure_notes.append(
        f"C1 FAIL: harm_fwd_r2={harm_fwd_r2:.4f} <= {THRESH_C1_HARM_FWD_R2}"
        f" (forward model not learning harm dynamics even with denser grid)")
    if not c2: failure_notes.append(
        f"C2 FAIL: delta_approach={delta_approach:.6f} <= {THRESH_C2_DELTA_APPROACH}"
        f" (trained CF does not add signal vs random CF for approach events)")
    if not c3: failure_notes.append(
        f"C3 FAIL: causal_approach_trained={causal_approach_trained:.6f}"
        f" not > causal_none_trained={causal_none_trained:.6f}"
        f" (gradient ordering broken)")
    if not c4: failure_notes.append(
        f"C4 FAIL: causal_contact_trained={causal_contact_trained:.6f}"
        f" not > causal_approach_trained={causal_approach_trained:.6f}"
        f" (MECH-102 escalation not recovered)")
    if not c5: failure_notes.append(
        f"C5 FAIL: n_approach_eval={n_approach_eval} <= {THRESH_C5_MIN_APPROACH_EVT}"
        f" (too few approach events for reliable estimates)")

    print(f"\nV3-EXQ-109 verdict: {status}  ({n_met}/5 criteria)", flush=True)
    for note in failure_notes:
        print(f"  {note}", flush=True)

    metrics = {
        "harm_fwd_r2":               float(harm_fwd_r2),
        "causal_approach_trained":   float(causal_approach_trained),
        "causal_approach_random":    float(causal_approach_random),
        "delta_approach":            float(delta_approach),
        "causal_none_trained":       float(causal_none_trained),
        "causal_contact_trained":    float(causal_contact_trained),
        "n_approach_eval":           float(n_approach_eval),
        "n_contact_eval":            float(n_contact_eval),
        "n_none_eval":               float(n_none_eval),
        "n_approach_p1":             float(n_approach_p1),
        "n_contact_p1":              float(n_contact_p1),
        "n_none_p1":                 float(n_none_p1),
        "crit1_pass":                1.0 if c1 else 0.0,
        "crit2_pass":                1.0 if c2 else 0.0,
        "crit3_pass":                1.0 if c3 else 0.0,
        "crit4_pass":                1.0 if c4 else 0.0,
        "crit5_pass":                1.0 if c5 else 0.0,
        "criteria_met":              float(n_met),
        "fatal_error_count":         0.0,
    }

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(f"- {n}" for n in failure_notes)

    summary_markdown = f"""# V3-EXQ-109 -- SD-003 Harm-Stream Counterfactual Attribution

**Status:** {status}
**Claims:** SD-003, ARC-033
**Supersedes:** EXQ-095/095a/095b (harm_fwd_r2=0 root cause: 12x12 sparse grid + random nav)

## Design

Discriminative pair within a single run:
- TRAINED: HarmForwardModel trained on (z_harm_t, action_t, z_harm_t+1) transitions
- RANDOM: Same architecture, random init (never trained)
Both use the same HarmEncoder and E3 harm_eval head trained in P1/P2.

Fixes over EXQ-095:
1. Grid 6x6 (was 12x12): denser hazard coverage
2. Navigation bias {nav_bias}: 60% toward-hazard, 40% random
3. 4 hazards (was 6): avoids additive-field clip saturation (EXQ-102 finding)

## Pre-Registered Criteria

| Criterion | Threshold | Value | Pass |
|-----------|-----------|-------|------|
| C1 harm_fwd_r2 | >{THRESH_C1_HARM_FWD_R2} | {harm_fwd_r2:.4f} | {'yes' if c1 else 'no'} |
| C2 delta_approach | >{THRESH_C2_DELTA_APPROACH} | {delta_approach:.6f} | {'yes' if c2 else 'no'} |
| C3 approach>none (trained) | >0 | {causal_approach_trained - causal_none_trained:.6f} | {'yes' if c3 else 'no'} |
| C4 contact>approach (trained) | >0 | {causal_contact_trained - causal_approach_trained:.6f} | {'yes' if c4 else 'no'} |
| C5 n_approach_eval | >{THRESH_C5_MIN_APPROACH_EVT} | {n_approach_eval} | {'yes' if c5 else 'no'} |

## Pairwise Comparison (Scenario: SD-003 discrimination)

| Condition | approach | none | contact |
|-----------|----------|------|---------|
| TRAINED   | {causal_approach_trained:.5f} | {causal_none_trained:.5f} | {causal_contact_trained:.5f} |
| RANDOM    | {causal_approach_random:.5f}  | n/a  | n/a     |
| delta     | {delta_approach:.5f}          | --   | --      |
{failure_section}
"""

    return {
        "experiment_type":    EXPERIMENT_TYPE,
        "claim_ids":          CLAIM_IDS,
        "seed":               seed,
        "phase1_episodes":    phase1_episodes,
        "phase2_episodes":    phase2_episodes,
        "eval_episodes":      eval_episodes,
        "steps_per_episode":  steps_per_episode,
        "nav_bias":           nav_bias,
        "grid_size":          6,
        "n_hazards":          4,
        "registered_thresholds": {
            "c1_harm_fwd_r2":       THRESH_C1_HARM_FWD_R2,
            "c2_delta_approach":    THRESH_C2_DELTA_APPROACH,
            "c3_grad_ordering":     THRESH_C3_GRAD_ORDERING,
            "c4_escalation":        THRESH_C4_ESCALATION,
            "c5_min_approach_evt":  THRESH_C5_MIN_APPROACH_EVT,
        },
        "metrics":  metrics,
        "status":   status,
        "verdict":  status,
        "summary":  summary_markdown,
        "scenario": "SD-003 counterfactual attribution: TRAINED vs RANDOM HarmForwardModel",
        "interpretation": (
            "PASS confirms SD-003 harm-stream pipeline: learned E2_harm_s adds"
            " attribution signal beyond random noise for approach events. "
            "FAIL with C1=0 indicates forward model still not learning: needs further"
            " investigation (ARC-033 blocked at current architecture/scale)."
        ),
        "pairwise_deltas": {
            "causal_approach_trained_minus_random": float(delta_approach),
        },
    }


def main():
    import json
    import time
    from datetime import datetime, timezone

    SEEDS = [42, 43, 44]
    OUT_DIR = Path(__file__).resolve().parents[1].parent / "REE_assembly" / "evidence" / "experiments"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []
    for seed in SEEDS:
        print(f"\n{'='*60}", flush=True)
        print(f"SEED {seed}", flush=True)
        print(f"{'='*60}", flush=True)
        result = run(seed=seed)
        all_results.append(result)

    # Aggregate
    def _agg(key):
        vals = [r["metrics"][key] for r in all_results if key in r.get("metrics", {})]
        return float(np.mean(vals)) if vals else 0.0

    harm_fwd_r2_mean         = _agg("harm_fwd_r2")
    delta_approach_mean      = _agg("delta_approach")
    causal_approach_t_mean   = _agg("causal_approach_trained")
    causal_none_t_mean       = _agg("causal_none_trained")
    causal_contact_t_mean    = _agg("causal_contact_trained")
    n_approach_mean          = _agg("n_approach_eval")
    criteria_met_mean        = _agg("criteria_met")

    c1_agg = harm_fwd_r2_mean      > THRESH_C1_HARM_FWD_R2
    c2_agg = delta_approach_mean   > THRESH_C2_DELTA_APPROACH
    c3_agg = causal_approach_t_mean > causal_none_t_mean + THRESH_C3_GRAD_ORDERING
    c4_agg = causal_contact_t_mean  > causal_approach_t_mean + THRESH_C4_ESCALATION
    c5_agg = n_approach_mean        > THRESH_C5_MIN_APPROACH_EVT
    all_pass_agg = all([c1_agg, c2_agg, c3_agg, c4_agg, c5_agg])
    status_agg = "PASS" if all_pass_agg else "FAIL"

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"

    pack = {
        "experiment_type":        EXPERIMENT_TYPE,
        "run_id":                 run_id,
        "run_timestamp":          ts,
        "architecture_epoch":     "ree_hybrid_guardrails_v1",
        "claim_ids":              CLAIM_IDS,
        "claim_ids_tested":       CLAIM_IDS,
        "evidence_class":         "discriminative_pair",
        "evidence_direction":     "supports" if all_pass_agg else "weakens",
        "seeds":                  SEEDS,
        "n_seeds":                float(len(SEEDS)),
        "grid_size":              6,
        "n_hazards":              4,
        "registered_thresholds": {
            "c1_harm_fwd_r2":       THRESH_C1_HARM_FWD_R2,
            "c2_delta_approach":    THRESH_C2_DELTA_APPROACH,
            "c3_grad_ordering":     THRESH_C3_GRAD_ORDERING,
            "c4_escalation":        THRESH_C4_ESCALATION,
            "c5_min_approach_evt":  THRESH_C5_MIN_APPROACH_EVT,
        },
        "metrics": {
            "harm_fwd_r2":               harm_fwd_r2_mean,
            "delta_approach":            delta_approach_mean,
            "causal_approach_trained":   causal_approach_t_mean,
            "causal_none_trained":       causal_none_t_mean,
            "causal_contact_trained":    causal_contact_t_mean,
            "n_approach_eval":           n_approach_mean,
            "crit1_pass":                1.0 if c1_agg else 0.0,
            "crit2_pass":                1.0 if c2_agg else 0.0,
            "crit3_pass":                1.0 if c3_agg else 0.0,
            "crit4_pass":                1.0 if c4_agg else 0.0,
            "crit5_pass":                1.0 if c5_agg else 0.0,
            "criteria_met":              float(sum([c1_agg, c2_agg, c3_agg, c4_agg, c5_agg])),
            "fatal_error_count":         0.0,
        },
        "status":   status_agg,
        "verdict":  status_agg,
        "scenario": "SD-003 counterfactual attribution: TRAINED vs RANDOM HarmForwardModel",
        "interpretation": (
            "PASS: learned HarmForwardModel adds systematic causal signal for approach"
            " events beyond random-init baseline -- SD-003 harm-stream pipeline validated."
            " FAIL C1: forward model still not learning (ARC-033 blocked)."
            " FAIL C2 with C1 pass: pipeline architecture correct but signal too weak."
        ),
        "pairwise_deltas": {
            "causal_approach_trained_minus_random": float(delta_approach_mean),
        },
        "per_seed_results": all_results,
    }

    exp_dir = OUT_DIR / EXPERIMENT_TYPE
    exp_dir.mkdir(exist_ok=True)
    out_path = exp_dir / f"{run_id}.json"
    with open(out_path, "w") as f:
        json.dump(pack, f, indent=2)

    print(f"\n{'='*60}", flush=True)
    print(f"AGGREGATE ({len(SEEDS)} seeds) verdict: {status_agg}", flush=True)
    print(f"  harm_fwd_r2:             {harm_fwd_r2_mean:.4f}", flush=True)
    print(f"  delta_approach (C2):     {delta_approach_mean:.6f}", flush=True)
    print(f"  causal_approach_trained: {causal_approach_t_mean:.6f}", flush=True)
    print(f"  causal_none_trained:     {causal_none_t_mean:.6f}", flush=True)
    print(f"  causal_contact_trained:  {causal_contact_t_mean:.6f}", flush=True)
    print(f"  n_approach_eval:         {n_approach_mean:.1f}", flush=True)
    print(f"  criteria_met:            {sum([c1_agg,c2_agg,c3_agg,c4_agg,c5_agg])}/5", flush=True)
    print(f"\nResults written to: {out_path}", flush=True)


if __name__ == "__main__":
    main()
