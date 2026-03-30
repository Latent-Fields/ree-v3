#!/opt/local/bin/python3
"""
V3-EXQ-166e -- SD-003 Harm-Delta Predictor (ARC-033)

Claims: SD-003, ARC-033
Supersedes: EXQ-166d (dry-run only; decoder discrimination cannot fix latent identity collapse)

Root cause of EXQ-166b/c/d failures:
  The latent-space HarmForwardModel predicts z_harm_next (32-dim vector). Because z_harm_s
  barely changes step-to-step (harm field is spatially smooth, temporal autocorrelation ~0.9),
  MSE minimization converges to identity: fwd(z_harm_t, a) ~= z_harm_t for ALL actions.
  R2=0.62 (EXQ-166b/c) is measuring the quality of the identity predictor, not action-conditional
  prediction. Once the forward model outputs the same z_harm for all actions, no downstream
  fix (decoder substitution, shuffled ablation) can recover causal discrimination.

Fix -- predict harm DELTA, not absolute z_harm_next:
  Target = harm_obs_s_next[CENTER_CELL] - harm_obs_s_curr[CENTER_CELL]  (scalar)
  Model: HarmDeltaPredictor(z_harm_s_t, action_onehot) -> scalar_delta
  Loss: MSE on scalar_delta

Why identity collapse cannot recur:
  - Predicting delta=0 for all actions is optimal ONLY if the training set is perfectly
    balanced (equal approach/retreat steps). In practice with nav_bias=0.65 toward hazards,
    approach events dominate -> mean delta > 0 near hazards -> zero prediction is suboptimal.
  - More importantly: the SIGN of the delta is action-dependent.
    Near hazard + approach action: delta > 0 (center cell hazard increases)
    Near hazard + retreat action:  delta < 0 (center cell hazard decreases)
    Empty space + any action:      delta ~= 0 (no hazard gradient)
  - The model MUST learn the action-conditional sign to minimize MSE.
  - Predicting the mean across all actions is wrong for approach AND retreat events.

Causal signal pipeline (SD-003 compliant):
  delta_actual = HarmDeltaPredictor(z_harm_s_t, a_actual)
  delta_cf     = mean([HarmDeltaPredictor(z_harm_s_t, a_cf) for a_cf in CF_ACTIONS])
  causal_gap   = delta_actual - delta_cf

At approach events (agent moves toward hazard):
  delta_actual > 0  (harm will increase)
  delta_cf < 0      (counterfactual: retreating would decrease harm)
  causal_gap > 0    <-- this is the SD-003 causal signature

Phases:
  P0 (warmup, phase0_episodes): Train HarmEncoder (autoencoder + center-cell regression).
     Ensures z_harm_s encodes harm-discriminative structure.
  P1 (delta training, phase1_episodes): Freeze HarmEncoder. Train HarmDeltaPredictor on
     (z_harm_s_t, action) -> harm_delta_scalar. Stratified replay: 50% approach/contact,
     50% neutral to prevent collapse to zero prediction.
  P2 (eval, eval_episodes): Compute causal_gap at approach / contact / neutral events.

PRE-REGISTERED ACCEPTANCE CRITERIA (ALL required for PASS):
  C1 (delta_r2 > 0.05):
    HarmDeltaPredictor R2 on held-out eval transitions. Low threshold (0.05) because
    predicting a scalar change is harder than absolute z_harm; even R2=0.05 confirms
    the model captures some action-conditional signal above the mean baseline.
  C2 (causal_gap_approach_mean > 0.0):
    Primary SD-003 criterion. Trained model produces higher harm-delta prediction for
    the actual approach action vs. mean of counterfactual actions at approach events.
  C3 (causal_gap_approach_mean > causal_gap_neutral_mean):
    Ordering: causal gap larger at approach events than neutral events.
  C4 (causal_gap_contact_mean > causal_gap_approach_mean):
    Escalation: contact events produce larger causal gap than approach events.
  C5 (n_approach_eval_mean > 20):
    Data quality gate: sufficient approach events for reliable estimates.
  C6 (causal_gap_approach > 0 for >= 3/4 seeds):
    Seed consistency.

Decision scoring:
  retain_ree:       ALL criteria met -- full SD-003 + ARC-033 validation via delta pipeline
  hybridize:        C1+C2+C5+C6 pass, C3 or C4 fail
  retire_ree_claim: C1 passes but C2 <= 0 AND C5 passes
  inconclusive:     C1 fails -- delta predictor did not learn action-conditional harm signal
"""

import sys
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.latent.stack import HarmEncoder
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_166e_sd003_harm_delta_predictor"
CLAIM_IDS = ["SD-003", "ARC-033"]

# Pre-registered thresholds (MUST NOT be changed post-hoc)
THRESH_C1_DELTA_R2         = 0.05
THRESH_C2_CAUSAL_APPROACH  = 0.0
THRESH_C3_ORDERING         = 0.0   # approach gap > neutral gap (diff)
THRESH_C4_ESCALATION       = 0.0   # contact gap > approach gap (diff)
THRESH_C5_MIN_APPROACH_EVT = 20
THRESH_C6_MAJORITY_FRAC    = 0.75  # >= 75% of seeds

HARM_OBS_DIM   = 51
Z_HARM_DIM     = 32
CENTER_CELL    = 12    # index 12 = center of 5x5 hazard_field_view within harm_obs
ACTION_DIM     = 4     # CausalGridWorldV2 default
P0_VAR_THRESH  = 0.005  # degenerate encoder gate


class HarmDeltaPredictor(nn.Module):
    """
    Predict the scalar harm delta: harm_obs_s_next[CENTER_CELL] - harm_obs_s_curr[CENTER_CELL].
    Input:  z_harm_s_t (Z_HARM_DIM) + action_onehot (ACTION_DIM)
    Output: scalar predicted delta (harm change at center cell)

    The action input is critical: the same z_harm_s state will produce positive delta
    for approach actions and negative delta for retreat actions near hazards.
    This action-conditional asymmetry is what prevents identity collapse.
    """

    def __init__(self, z_harm_dim: int = 32, action_dim: int = 4, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_harm_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, z_harm: torch.Tensor, action_oh: torch.Tensor) -> torch.Tensor:
        """Returns (batch, 1) scalar delta predictions."""
        x = torch.cat([z_harm, action_oh], dim=-1)
        return self.net(x)


class HarmDecoder(nn.Module):
    """Decoder for P0 autoencoder pre-training only."""

    def __init__(self, z_harm_dim: int = 32, harm_obs_dim: int = 51, hidden_dim: int = 64):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(z_harm_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, harm_obs_dim),
        )

    def forward(self, z_harm: torch.Tensor) -> torch.Tensor:
        return self.decoder(z_harm)


def _action_to_onehot(idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, idx] = 1.0
    return v


def _hazard_approach_action(env: CausalGridWorldV2, n_actions: int) -> int:
    """Return action index toward nearest hazard gradient. Falls back to random."""
    obs_dict = env._get_observation_dict()
    world_state = obs_dict.get("world_state", None)
    if world_state is None or not env.use_proxy_fields:
        return random.randint(0, n_actions - 1)
    field_view = world_state[225:250].numpy().reshape(5, 5)
    deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    vals = []
    for dr, dc in deltas:
        r, c = 2 + dr, 2 + dc
        if 0 <= r < 5 and 0 <= c < 5:
            vals.append(float(field_view[r, c]))
        else:
            vals.append(-1.0)
    return int(np.argmax(vals))


def _ttype_bucket(ttype: str) -> str:
    if ttype in ("env_caused_hazard", "agent_caused_hazard"):
        return "contact"
    elif ttype == "hazard_approach":
        return "approach"
    return "none"


def _run_single(
    seed: int,
    phase0_episodes: int,
    phase1_episodes: int,
    eval_episodes: int,
    steps_per_episode: int,
    world_dim: int,
    self_dim: int,
    nav_bias: float,
    dry_run: bool,
) -> Dict:
    """Run one seed. Returns metrics dict."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    if dry_run:
        phase0_episodes = min(3, phase0_episodes)
        phase1_episodes = min(3, phase1_episodes)
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
    device = agent.device

    harm_enc    = HarmEncoder(harm_obs_dim=HARM_OBS_DIM, z_harm_dim=Z_HARM_DIM)
    harm_dec_p0 = HarmDecoder(z_harm_dim=Z_HARM_DIM, harm_obs_dim=HARM_OBS_DIM)
    delta_pred  = HarmDeltaPredictor(z_harm_dim=Z_HARM_DIM, action_dim=n_actions, hidden_dim=64)

    opt_enc = optim.Adam(
        list(harm_enc.parameters())
        + list(harm_dec_p0.parameters())
        + list(agent.e3.harm_eval_z_harm_head.parameters()),
        lr=1e-3,
    )
    opt_delta = optim.Adam(delta_pred.parameters(), lr=1e-3)

    # Replay: (harm_obs_t, action_idx, harm_obs_next, transition_bucket)
    # Stratified by bucket: "approach", "contact", "none"
    replay_approach: List[Tuple] = []
    replay_contact:  List[Tuple] = []
    replay_neutral:  List[Tuple] = []
    MAX_PER_BUCKET = 2000

    # ---- Phase 0: Pre-train HarmEncoder as autoencoder + center-cell regression -------
    print(
        f"[EXQ-166e seed={seed}] P0: HarmEncoder warm-up ({phase0_episodes} eps)...",
        flush=True,
    )
    agent.train(); harm_enc.train(); harm_dec_p0.train(); delta_pred.train()

    for ep in range(phase0_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()

        for _ in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            agent.sense(obs_body, obs_world)
            agent.clock.advance()

            harm_obs_t = obs_dict.get("harm_obs", torch.zeros(HARM_OBS_DIM)).float()

            if random.random() < nav_bias:
                action_idx = _hazard_approach_action(env, n_actions)
            else:
                action_idx = random.randint(0, n_actions - 1)
            action_oh = _action_to_onehot(action_idx, n_actions, device)
            agent._last_action = action_oh

            flat_obs, _, done, info, obs_dict = env.step(action_oh)
            harm_obs_next = obs_dict.get("harm_obs", torch.zeros(HARM_OBS_DIM)).float()
            ttype = info.get("transition_type", "none")
            bucket = _ttype_bucket(ttype)

            # Store for P1 buffer (raw obs, not latent -- encoder not frozen yet)
            tup = (harm_obs_t.cpu(), action_idx, harm_obs_next.cpu(), bucket)
            if bucket == "approach" and len(replay_approach) < MAX_PER_BUCKET:
                replay_approach.append(tup)
            elif bucket == "contact" and len(replay_contact) < MAX_PER_BUCKET:
                replay_contact.append(tup)
            elif bucket == "none" and len(replay_neutral) < MAX_PER_BUCKET:
                replay_neutral.append(tup)

            # P0 loss: autoencoder + center-cell scalar regression
            h_t = harm_obs_t.unsqueeze(0)
            z_t = harm_enc(h_t)
            recon = harm_dec_p0(z_t)
            loss_ae  = ((recon - h_t) ** 2).mean()
            label    = h_t[0, CENTER_CELL].unsqueeze(0).unsqueeze(0)
            z_harm_t_eval = agent.e3.harm_eval_z_harm_head(z_t)
            loss_reg = ((z_harm_t_eval - label) ** 2).mean()
            loss_p0  = loss_ae + loss_reg

            opt_enc.zero_grad()
            loss_p0.backward()
            opt_enc.step()

            if done:
                flat_obs, obs_dict = env.reset()
                agent.reset()

    # Freeze HarmEncoder; clear replay and re-fill with fresh latent-encoded transitions
    for p in harm_enc.parameters():
        p.requires_grad_(False)
    harm_enc.eval()

    # Check encoder quality
    p0_z_harm_var: float = 0.0
    if len(replay_approach) + len(replay_contact) + len(replay_neutral) > 0:
        sample_obs = []
        for tup in (replay_approach + replay_contact + replay_neutral)[:200]:
            sample_obs.append(tup[0].unsqueeze(0))
        with torch.no_grad():
            zs = harm_enc(torch.cat(sample_obs, dim=0))
        p0_z_harm_var = float(zs.var().item())

    print(
        f"[EXQ-166e seed={seed}] P0 done. z_harm_var={p0_z_harm_var:.4f}"
        f" approach={len(replay_approach)} contact={len(replay_contact)}"
        f" neutral={len(replay_neutral)}",
        flush=True,
    )

    if p0_z_harm_var < P0_VAR_THRESH:
        print(f"[EXQ-166e seed={seed}] WARNING: degenerate encoder (var={p0_z_harm_var:.5f})")

    # Discard stale P0 buffer (transitions collected while encoder was updating).
    # Re-collect fresh P1 data with frozen encoder.
    replay_approach.clear()
    replay_contact.clear()
    replay_neutral.clear()

    # ---- Phase 1: Train HarmDeltaPredictor with frozen encoder ----------------------
    print(
        f"[EXQ-166e seed={seed}] P1: HarmDeltaPredictor training ({phase1_episodes} eps)...",
        flush=True,
    )
    harm_enc.eval(); delta_pred.train()

    for ep in range(phase1_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()

        for _ in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            agent.sense(obs_body, obs_world)
            agent.clock.advance()

            harm_obs_t = obs_dict.get("harm_obs", torch.zeros(HARM_OBS_DIM)).float()

            if random.random() < nav_bias:
                action_idx = _hazard_approach_action(env, n_actions)
            else:
                action_idx = random.randint(0, n_actions - 1)
            action_oh = _action_to_onehot(action_idx, n_actions, device)
            agent._last_action = action_oh

            flat_obs, _, done, info, obs_dict = env.step(action_oh)
            harm_obs_next = obs_dict.get("harm_obs", torch.zeros(HARM_OBS_DIM)).float()
            ttype  = info.get("transition_type", "none")
            bucket = _ttype_bucket(ttype)

            # Compute z_harm with frozen encoder
            with torch.no_grad():
                z_harm_t = harm_enc(harm_obs_t.unsqueeze(0))

            tup = (z_harm_t.squeeze(0).cpu(), action_idx, harm_obs_t.cpu(), harm_obs_next.cpu(), bucket)
            if bucket == "approach" and len(replay_approach) < MAX_PER_BUCKET:
                replay_approach.append(tup)
            elif bucket == "contact" and len(replay_contact) < MAX_PER_BUCKET:
                replay_contact.append(tup)
            elif bucket == "none" and len(replay_neutral) < MAX_PER_BUCKET:
                replay_neutral.append(tup)

            # Online delta training on this step
            delta_target = float(harm_obs_next[CENTER_CELL].item()) - float(harm_obs_t[CENTER_CELL].item())
            a_oh_train = _action_to_onehot(action_idx, n_actions, device)
            delta_pred_val = delta_pred(z_harm_t.detach(), a_oh_train)
            loss_delta = ((delta_pred_val - torch.tensor([[delta_target]], device=device)) ** 2).mean()

            opt_delta.zero_grad()
            loss_delta.backward()
            opt_delta.step()

            if done:
                flat_obs, obs_dict = env.reset()
                agent.reset()

    # Replay-based training on stratified batch (oversample approach + contact)
    print(
        f"[EXQ-166e seed={seed}] P1 replay: approach={len(replay_approach)}"
        f" contact={len(replay_contact)} neutral={len(replay_neutral)}",
        flush=True,
    )
    n_replay_steps = min(len(replay_approach) + len(replay_contact), 200) if not dry_run else 5
    if n_replay_steps > 0:
        delta_pred.train()
        for _ in range(n_replay_steps):
            # Stratified batch: half from approach/contact, half from neutral
            batch_size = 32
            n_event   = batch_size // 2
            n_neutral = batch_size - n_event
            pool_event = replay_approach + replay_contact
            batch_ev  = random.choices(pool_event,    k=min(n_event,   len(pool_event)))
            batch_neu = random.choices(replay_neutral, k=min(n_neutral, len(replay_neutral))) if replay_neutral else []

            batch = batch_ev + batch_neu
            if not batch:
                break

            z_harms = torch.stack([b[0] for b in batch], dim=0).to(device)
            a_idxs  = [b[1] for b in batch]
            obs_t   = torch.stack([b[2] for b in batch], dim=0)
            obs_next= torch.stack([b[3] for b in batch], dim=0)

            a_ohs = torch.zeros(len(batch), n_actions, device=device)
            for i, ai in enumerate(a_idxs):
                a_ohs[i, ai] = 1.0

            targets = (obs_next[:, CENTER_CELL] - obs_t[:, CENTER_CELL]).unsqueeze(1).to(device)
            preds   = delta_pred(z_harms.detach(), a_ohs)
            loss_r  = ((preds - targets) ** 2).mean()

            opt_delta.zero_grad()
            loss_r.backward()
            opt_delta.step()

    # ---- Evaluation: compute R2 and causal gap ------------------------------------
    print(
        f"[EXQ-166e seed={seed}] Eval ({eval_episodes} eps)...",
        flush=True,
    )
    harm_enc.eval(); delta_pred.eval()

    eval_targets:  List[float] = []
    eval_preds:    List[float] = []
    gap_approach:  List[float] = []
    gap_contact:   List[float] = []
    gap_neutral:   List[float] = []
    n_approach_eval = 0
    n_contact_eval  = 0

    CF_ACTIONS = list(range(n_actions))  # all actions as counterfactuals

    for ep in range(eval_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()

        for _ in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            agent.sense(obs_body, obs_world)
            agent.clock.advance()

            harm_obs_t = obs_dict.get("harm_obs", torch.zeros(HARM_OBS_DIM)).float()

            if random.random() < nav_bias:
                action_idx = _hazard_approach_action(env, n_actions)
            else:
                action_idx = random.randint(0, n_actions - 1)
            action_oh = _action_to_onehot(action_idx, n_actions, device)
            agent._last_action = action_oh

            flat_obs, _, done, info, obs_dict = env.step(action_oh)
            harm_obs_next = obs_dict.get("harm_obs", torch.zeros(HARM_OBS_DIM)).float()
            ttype  = info.get("transition_type", "none")
            bucket = _ttype_bucket(ttype)

            delta_actual_obs = float(harm_obs_next[CENTER_CELL].item()) - float(harm_obs_t[CENTER_CELL].item())

            with torch.no_grad():
                z_harm_t = harm_enc(harm_obs_t.unsqueeze(0))
                a_oh_actual = _action_to_onehot(action_idx, n_actions, device)
                delta_actual_pred = float(delta_pred(z_harm_t, a_oh_actual).item())

                # Counterfactual: mean delta pred over ALL actions
                cf_preds = []
                for a_cf in CF_ACTIONS:
                    a_oh_cf = _action_to_onehot(a_cf, n_actions, device)
                    cf_preds.append(float(delta_pred(z_harm_t, a_oh_cf).item()))
                delta_cf_mean = float(np.mean(cf_preds))

                causal_gap = delta_actual_pred - delta_cf_mean

            # R2 accumulation
            eval_targets.append(delta_actual_obs)
            eval_preds.append(delta_actual_pred)

            # Gap by bucket
            if bucket == "approach":
                gap_approach.append(causal_gap)
                n_approach_eval += 1
            elif bucket == "contact":
                gap_contact.append(causal_gap)
                n_contact_eval += 1
            else:
                gap_neutral.append(causal_gap)

            if done:
                flat_obs, obs_dict = env.reset()
                agent.reset()

    # Compute R2
    t_arr = np.array(eval_targets)
    p_arr = np.array(eval_preds)
    ss_res = float(((t_arr - p_arr) ** 2).sum())
    ss_tot = float(((t_arr - t_arr.mean()) ** 2).sum())
    delta_r2 = float(1.0 - ss_res / (ss_tot + 1e-8))

    causal_gap_approach = float(np.mean(gap_approach)) if gap_approach else 0.0
    causal_gap_contact  = float(np.mean(gap_contact))  if gap_contact  else 0.0
    causal_gap_neutral  = float(np.mean(gap_neutral))  if gap_neutral  else 0.0

    print(
        f"[EXQ-166e seed={seed}] delta_r2={delta_r2:.4f}"
        f" gap_approach={causal_gap_approach:.4f}"
        f" gap_contact={causal_gap_contact:.4f}"
        f" gap_neutral={causal_gap_neutral:.4f}"
        f" n_approach={n_approach_eval}",
        flush=True,
    )

    c1 = delta_r2 > THRESH_C1_DELTA_R2
    c2 = causal_gap_approach > THRESH_C2_CAUSAL_APPROACH
    c3 = (causal_gap_approach - causal_gap_neutral) > THRESH_C3_ORDERING
    c4 = (causal_gap_contact - causal_gap_approach) > THRESH_C4_ESCALATION
    c5 = n_approach_eval > THRESH_C5_MIN_APPROACH_EVT
    # C6 evaluated across seeds in aggregate

    return {
        "seed": seed,
        "delta_r2": delta_r2,
        "causal_gap_approach": causal_gap_approach,
        "causal_gap_contact":  causal_gap_contact,
        "causal_gap_neutral":  causal_gap_neutral,
        "n_approach_eval":     n_approach_eval,
        "n_contact_eval":      n_contact_eval,
        "p0_z_harm_var":       p0_z_harm_var,
        "crit1_pass":          int(c1),
        "crit2_pass":          int(c2),
        "crit3_pass":          int(c3),
        "crit4_pass":          int(c4),
        "crit5_pass":          int(c5),
    }


if __name__ == "__main__":
    import argparse
    import json
    import datetime

    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run",        action="store_true")
    parser.add_argument("--seeds",          type=int, nargs="+", default=[42, 123, 7, 13])
    parser.add_argument("--phase0-episodes",type=int, default=200)
    parser.add_argument("--phase1-episodes",type=int, default=300)
    parser.add_argument("--eval-episodes",  type=int, default=100)
    parser.add_argument("--steps-per-episode", type=int, default=200)
    parser.add_argument("--world-dim",      type=int, default=32)
    parser.add_argument("--self-dim",       type=int, default=16)
    parser.add_argument("--nav-bias",       type=float, default=0.65)
    args = parser.parse_args()

    seeds = args.seeds
    results_by_seed = {}

    for seed in seeds:
        r = _run_single(
            seed=seed,
            phase0_episodes=args.phase0_episodes,
            phase1_episodes=args.phase1_episodes,
            eval_episodes=args.eval_episodes,
            steps_per_episode=args.steps_per_episode,
            world_dim=args.world_dim,
            self_dim=args.self_dim,
            nav_bias=args.nav_bias,
            dry_run=args.dry_run,
        )
        results_by_seed[str(seed)] = r

    # Aggregate
    def _mean(key): return float(np.mean([results_by_seed[str(s)][key] for s in seeds]))

    delta_r2_mean            = _mean("delta_r2")
    causal_gap_approach_mean = _mean("causal_gap_approach")
    causal_gap_contact_mean  = _mean("causal_gap_contact")
    causal_gap_neutral_mean  = _mean("causal_gap_neutral")
    n_approach_eval_mean     = _mean("n_approach_eval")

    c1 = delta_r2_mean > THRESH_C1_DELTA_R2
    c2 = causal_gap_approach_mean > THRESH_C2_CAUSAL_APPROACH
    c3 = (causal_gap_approach_mean - causal_gap_neutral_mean) > THRESH_C3_ORDERING
    c4 = (causal_gap_contact_mean - causal_gap_approach_mean) > THRESH_C4_ESCALATION
    c5 = n_approach_eval_mean > THRESH_C5_MIN_APPROACH_EVT
    c6_count = sum(
        1 for s in seeds if results_by_seed[str(s)]["causal_gap_approach"] > THRESH_C2_CAUSAL_APPROACH
    )
    c6 = (c6_count / len(seeds)) >= THRESH_C6_MAJORITY_FRAC

    all_criteria = [c1, c2, c3, c4, c5, c6]
    criteria_met = sum(all_criteria)
    outcome = "PASS" if all(all_criteria) else "FAIL"

    # Decision scoring
    if all(all_criteria):
        decision = "retain_ree"
    elif c1 and c2 and c5 and c6:
        decision = "hybridize"
    elif c1 and not c2 and c5:
        decision = "retire_ree_claim"
    else:
        decision = "inconclusive"

    timestamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{timestamp}_v3"

    manifest = {
        "schema_version": "v1",
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "claim_ids": CLAIM_IDS,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "outcome": outcome,
        "criteria_met": criteria_met,
        "total_criteria": len(all_criteria),
        "dry_run": args.dry_run,
        "supersedes": "v3_exq_166d_sd003_harm_decoder_discrimination",
        "notes": (
            "EXQ-166e: harm-delta predictor approach. Fixes fundamental identity collapse in"
            " latent-space HarmForwardModel (EXQ-166b/c/d). Predicts scalar harm delta"
            " (harm_obs_next[CENTER_CELL] - harm_obs_curr[CENTER_CELL]) from (z_harm_s, action)."
            " Identity collapse cannot recur: predicting delta=0 for all actions is wrong at"
            " approach/contact events where the delta has a signed action-dependent component."
            " ARC-033 causal signal: delta_actual - mean(delta_cf_over_actions)."
        ),
        "metrics": {
            "delta_r2_mean":             delta_r2_mean,
            "causal_gap_approach_mean":  causal_gap_approach_mean,
            "causal_gap_contact_mean":   causal_gap_contact_mean,
            "causal_gap_neutral_mean":   causal_gap_neutral_mean,
            "n_approach_eval_mean":      n_approach_eval_mean,
            "crit1_pass": float(c1),
            "crit2_pass": float(c2),
            "crit3_pass": float(c3),
            "crit4_pass": float(c4),
            "crit5_pass": float(c5),
            "crit6_pass": float(c6),
        },
        "per_seed": {
            str(s): {k: v for k, v in results_by_seed[str(s)].items() if k != "seed"}
            for s in seeds
        },
        "decision": decision,
        "config": {
            "seeds":              seeds,
            "phase0_episodes":    args.phase0_episodes,
            "phase1_episodes":    args.phase1_episodes,
            "eval_episodes":      args.eval_episodes,
            "steps_per_episode":  args.steps_per_episode,
            "world_dim":          args.world_dim,
            "self_dim":           args.self_dim,
            "nav_bias":           args.nav_bias,
            "harm_obs_dim":       HARM_OBS_DIM,
            "z_harm_dim":         Z_HARM_DIM,
            "center_cell_idx":    CENTER_CELL,
            "fwd_hidden_dim":     64,
            "grid_size":          6,
            "num_hazards":        4,
            "p0_var_thresh":      P0_VAR_THRESH,
        },
    }

    print(f"\n[EXQ-166e] outcome={outcome} criteria={criteria_met}/{len(all_criteria)}", flush=True)
    print(f"  delta_r2={delta_r2_mean:.4f}  gap_approach={causal_gap_approach_mean:.4f}"
          f"  gap_contact={causal_gap_contact_mean:.4f}"
          f"  gap_neutral={causal_gap_neutral_mean:.4f}", flush=True)
    print(f"  C1={int(c1)} C2={int(c2)} C3={int(c3)} C4={int(c4)} C5={int(c5)} C6={int(c6)}", flush=True)

    if not args.dry_run:
        out_dir = (
            Path(__file__).resolve().parents[2]
            / "REE_assembly"
            / "evidence"
            / "experiments"
            / EXPERIMENT_TYPE
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{run_id}.json"
        with open(out_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"[EXQ-166e] written -> {out_path}", flush=True)
        print(f"Status: {outcome}", flush=True)
    else:
        print("[EXQ-166e] dry-run: no output written.", flush=True)
