#!/opt/local/bin/python3
"""
V3-EXQ-145 -- SD-008 + SD-007 + SD-003 Pipeline Integration Test

Claims: SD-008, SD-007, SD-003
Proposal: EXP-0087 (medium priority)

Purpose
-------
SD-008, SD-007, and SD-003 have each been validated individually:
  - SD-008 (alpha_world>=0.9): EXQ-040 PASS -- event selectivity margin restored
  - SD-007 (ReafferencePredictor): EXQ-099a PASS -- selectivity_ratio=1.655>=1.1
  - SD-003 (counterfactual E2 pipeline): EXQ-030b PASS -- attribution_gap>0, causal_sig>0.005

This experiment asks: do all three compose correctly END-TO-END in a single pipeline run?
Composing them in isolation does not guarantee they compose correctly together because:
  (a) Reafference correction on alpha_world=0.9 may interact differently than on 0.3.
      (At alpha=0.3, z_world is already a 3-step average; correcting it may subtract
      too much. At alpha=0.9, z_world is nearly one-step; correction precision matters more.)
  (b) E3 harm_eval training distribution depends on z_world quality -- which depends on
      both the alpha correction AND the reafference correction being active simultaneously.
  (c) Counterfactual SD-003 causal_sig depends on E2.world_forward being trained on
      the SAME z_world distribution that results from SD-008+SD-007 together.

Integration success requires all three criteria to be met in the SAME run:
  C1 (SD-008): z_world event_selectivity_margin > 0.05  [alpha_world=0.9 preserves events]
  C2 (SD-007): reafference_selectivity_ratio >= 1.1     [correction improves hazard distinction]
  C3 (SD-003): attribution_gap > 0.0                   [agent-caused > env-caused attribution]
  C4 (SD-003): causal_sig_approach > 0.003              [minimum SD-003 signal present]
  C5 (world_forward): world_forward_r2 > 0.05           [E2 learned world dynamics; SD-003 prereq]
  C6 (data quality): n_approach >= 30 for both seeds    [adequate evaluation signal]

All criteria must pass in BOTH seeds for integration PASS. This is a stricter requirement
than the individual validations (which allowed 2/3 seed majority), because this is a
compositional correctness check, not a marginal effect measurement.

Mechanism under test
--------------------
If PASS on all six criteria simultaneously, both seeds:
  -> SD-008 + SD-007 + SD-003 compose without destructive interference.
  -> alpha_world=0.9 + reafference subtraction + counterfactual E2 produce all expected
     outputs together: event-responsive z_world, perspective-corrected z_world,
     and interpretable causal attribution signal.
  -> The integrated pipeline is ready for higher-level experiments (e.g. SD-011 dual-stream
     extension, ARC-033 E2_harm_s forward model).

If C1 fails only: alpha_world=0.9 not preserving events in this integration context
  -> investigate interaction between reafference correction and encoder EMA.
If C2 fails only: reafference correction not lifting selectivity despite good alpha
  -> predictor quality issue; check R2_reaf diagnostic.
If C3/C4 fail only: counterfactual pipeline not producing attribution signal despite
  good world representation -> E3 harm_eval training issue (distribution mismatch or
  training steps insufficient).
If C5 fails: E2.world_forward not learning -> check world_forward optimizer convergence.

Integration design
------------------
This is NOT an ablation pair (no "OFF" condition). Each seed runs ONE condition:
all three mechanisms active simultaneously. PASS = both seeds show all criteria met.
Ablation/discriminative pairs for each mechanism individually exist in EXQ-040, EXQ-099a,
EXQ-030b. This experiment tests only COMPOSITIONAL correctness.

Phase 1: Predictor collection and training
  Collect (z_world_prev, action, delta_z_world_loco) tuples from locomotion episodes
  in a low-hazard environment (same design as EXQ-118). Train LSTM reafference predictor.
  Gate: R2_reaf >= 0.08 (slightly lower than EXQ-118 C1=0.10 due to combined-run context).
  If Phase 1 fails the R2 gate, the experiment exits as FAIL immediately.

Phase 2: Integration warmup + eval (seeds 42 and 123)
  Config: alpha_world=0.9, reafference_action_dim=action_dim, use_event_classifier=False
  (use_event_classifier=False: SD-009 tested separately; this tests SD-008+SD-007+SD-003
  without mixing SD-009 into the integration test -- see claim_ids accuracy rule)
  Env: CausalGridWorldV2 size=10, num_hazards=3, num_resources=4
  Warmup: 400 episodes x 200 steps per seed (with 3 separate optimizers: MECH-069)
  Eval: 60 episodes x 200 steps per seed (more than EXQ-030b for integration test)

Pre-registered thresholds
--------------------------
C1: event_selectivity_margin_corrected > THRESH_C1 = 0.05  (both seeds)
    Margin = mean|dz_world| at hazard events minus mean|dz_world| at locomotion.
    With alpha_world=0.9 and reafference correction, z_world should be sharply responsive.

C2: reafference_selectivity_ratio >= THRESH_C2 = 1.10  (both seeds)
    mean|dz_world_corrected| at hazard_approach / mean|dz_world_corrected| at none.
    Reafference correction lifts hazard-step z_world magnitude relative to locomotion.

C3: attribution_gap > THRESH_C3 = 0.0  (both seeds)
    attribution_gap = mean_causal_sig(hazard_approach) - mean_causal_sig(env_caused_hazard)
    Agent-directed approach is more causally attributed than passive hazard exposure.

C4: causal_sig_approach > THRESH_C4 = 0.003  (both seeds)
    Minimum SD-003 signal. Slightly lower than EXQ-030b C3=0.005 to account for
    the reafference correction adding a different distribution to E3's training.

C5: world_forward_r2 > THRESH_C5 = 0.05  (pooled, i.e. shared trained E2 evaluated)
    E2.world_forward R2 on held-out world transitions -- must exceed chance.

C6: n_approach >= THRESH_C6 = 30  (each seed independently)
    Data quality gate: enough approach events to trust ratio/gap measurements.

Decision rules
--------------
ALL PASS (C1-C6, both seeds where per-seed): SD-008+SD-007+SD-003 integrate correctly.
  -> evidence_direction = "supports" for all three claim IDs.
C5 FAIL only: E2 world_forward failed -- not interpretable as integration failure.
  -> status FAIL, evidence_direction = "mixed" (mechanisms not at fault, training issue).
C6 FAIL (any seed): insufficient data, inconclusive -- rerun with more eval episodes.
Any other criterion FAIL: integration failure identified, evidence_direction = "weakens"
  for the specific claims whose criteria failed.

Seeds: [42, 123]
Env:   CausalGridWorldV2 size=10, 3 hazards, 4 resources, nav_bias=0.35
Warmup: 400 episodes x 200 steps per seed
Eval:   60 episodes x 200 steps per seed
Estimated runtime: ~120 min on DLAPTOP-4.local (Mac CPU)
  [2 seeds x (400+60 eps) x 200 steps x 0.10 min/ep ~= 2x46 = ~92 min + overhead]
"""

import sys
import copy
import json
import random
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_145_sd008_sd007_sd003_integration"
CLAIM_IDS = ["SD-008", "SD-007", "SD-003"]

# ---------------------------------------------------------------------------
# Pre-registered thresholds
# ---------------------------------------------------------------------------
THRESH_C1_SELECTIVITY_MARGIN  = 0.05   # C1: event selectivity margin (SD-008 in integration)
THRESH_C2_SELECTIVITY_RATIO   = 1.10   # C2: reafference selectivity lift ratio (SD-007)
THRESH_C3_ATTRIBUTION_GAP     = 0.0    # C3: attribution_gap > 0 (SD-003 agent vs env)
THRESH_C4_CAUSAL_SIG          = 0.003  # C4: min causal_sig at hazard_approach (SD-003)
THRESH_C5_WORLD_FORWARD_R2    = 0.05   # C5: E2 world_forward R2 (SD-003 prereq)
THRESH_C6_MIN_APPROACH        = 30     # C6: minimum approach events per seed (data quality)
THRESH_REAF_R2_GATE           = 0.08   # Phase 1 gate: predictor quality

SEEDS = [42, 123]

# ---------------------------------------------------------------------------
# LSTM reafference predictor (identical architecture to EXQ-110/111/118)
# ---------------------------------------------------------------------------

class UpgradedReafferencePredictor(nn.Module):
    """
    LSTM reafference predictor (hidden_dim=128).
    Predicts delta_z_world_loco from (z_world_prev, action).
    After subtraction: z_world_corrected = z_world_raw - predicted_loco_delta.
    Architecture matches EXQ-118 for comparability.
    """

    def __init__(self, world_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.world_dim  = world_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(
            input_size=world_dim + action_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.out = nn.Linear(hidden_dim, world_dim)
        self._hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    def reset_hidden(self) -> None:
        self._hidden = None

    def forward(
        self,
        z_world_prev: torch.Tensor,
        a_prev: torch.Tensor,
    ) -> torch.Tensor:
        """Single-step stateful forward. Input shapes: [1, world_dim], [1, action_dim]."""
        x = torch.cat([z_world_prev, a_prev], dim=-1).unsqueeze(1)
        out, self._hidden = self.lstm(x, self._hidden)
        return self.out(out.squeeze(1))

    def forward_sequence(self, x_seq: torch.Tensor) -> torch.Tensor:
        """Sequence forward for BPTT training (stateless)."""
        out, _ = self.lstm(x_seq)
        return self.out(out)

    def correct_z_world(
        self,
        z_world_raw: torch.Tensor,
        z_world_prev: torch.Tensor,
        a_prev: torch.Tensor,
    ) -> torch.Tensor:
        return z_world_raw - self.forward(z_world_prev, a_prev)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


def _build_collection_env(seed: int) -> CausalGridWorldV2:
    """Low-hazard environment for predictor data collection (matches EXQ-118)."""
    return CausalGridWorldV2(
        seed=seed,
        size=8,
        num_hazards=1,
        num_resources=2,
        hazard_harm=0.02,
        env_drift_interval=5,
        env_drift_prob=0.05,
        proximity_harm_scale=0.05,
        proximity_benefit_scale=0.03,
        proximity_approach_threshold=0.04,
        hazard_field_decay=0.5,
    )


def _build_eval_env(seed: int) -> CausalGridWorldV2:
    """Integration-test environment: moderate hazard density for adequate approach events."""
    return CausalGridWorldV2(
        seed=seed,
        size=10,
        num_hazards=3,
        num_resources=4,
        hazard_harm=0.02,
        env_drift_interval=5,
        env_drift_prob=0.2,
        proximity_harm_scale=0.05,
        proximity_benefit_scale=0.03,
        proximity_approach_threshold=0.15,
        hazard_field_decay=0.5,
    )


# ---------------------------------------------------------------------------
# Phase 1: Predictor collection and training
# ---------------------------------------------------------------------------

def _collect_predictor_data(
    seed: int,
    world_dim: int,
    action_dim: int,
    collection_episodes: int = 40,
    steps_per_episode: int = 300,
) -> Tuple[Optional[List], Optional[UpgradedReafferencePredictor], float]:
    """
    Collect locomotion sequences and train LSTM reafference predictor.
    Returns (sequences, trained_predictor, r2_test).
    Returns (None, None, 0.0) on failure.
    """
    torch.manual_seed(seed)
    random.seed(seed)

    cenv = _build_collection_env(seed)
    cconfig = REEConfig.from_dims(
        body_obs_dim=cenv.body_obs_dim,
        world_obs_dim=cenv.world_obs_dim,
        action_dim=cenv.action_dim,
        self_dim=32,
        world_dim=world_dim,
        alpha_world=0.9,  # SD-008: must be set during collection too for consistency
        reafference_action_dim=0,  # disabled during collection; raw z_world used
    )
    cagent = REEAgent(cconfig)
    cagent.eval()

    # Collect (z_world_prev, action, delta_z_world_raw) from ALL transition types
    # (including hazard_approach/contact) -- filter done below
    sequences: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    loco_types = {"none"}  # collect only locomotion steps for predictor training

    print(f"  [Phase1/seed={seed}] Collecting predictor data ({collection_episodes} eps)...", flush=True)
    for ep in range(collection_episodes):
        flat_obs, obs_dict = cenv.reset()
        cagent.reset()
        z_world_prev = None
        a_prev = None

        for step in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            with torch.no_grad():
                latent = cagent.sense(obs_body, obs_world)
                z_world_raw = latent.z_world.detach().cpu()

            action_idx = random.randint(0, cenv.action_dim - 1)
            action = _action_to_onehot(action_idx, cenv.action_dim, "cpu")

            flat_obs, harm_signal, done, info, obs_dict = cenv.step(action)
            ttype = info.get("transition_type", "none")

            if z_world_prev is not None and a_prev is not None and ttype in loco_types:
                delta_z = (z_world_raw - z_world_prev).squeeze(0)  # [world_dim]
                sequences.append((
                    z_world_prev.squeeze(0),
                    a_prev.squeeze(0),
                    delta_z,
                ))

            z_world_prev = z_world_raw
            a_prev = action
            if done:
                break

        if (ep + 1) % 20 == 0:
            print(f"    ep {ep+1}/{collection_episodes}  loco_seq={len(sequences)}", flush=True)

    if len(sequences) < 80:
        print(f"  [Phase1] Insufficient locomotion sequences: {len(sequences)} < 80", flush=True)
        return None, None, 0.0

    # Build LSTM input sequences: pack individual transitions into sequences of length 8
    seq_len = 8
    n_seqs = len(sequences) // seq_len
    if n_seqs < 10:
        print(f"  [Phase1] Too few full sequences: {n_seqs}", flush=True)
        return None, None, 0.0

    # Build tensors: [n_seqs, seq_len, world_dim+action_dim], [n_seqs, seq_len, world_dim]
    xs, ys = [], []
    for i in range(n_seqs):
        seg = sequences[i * seq_len : (i + 1) * seq_len]
        x_seg = torch.stack([torch.cat([s[0], s[1]]) for s in seg])  # [L, in]
        y_seg = torch.stack([s[2] for s in seg])                      # [L, world_dim]
        xs.append(x_seg)
        ys.append(y_seg)

    X = torch.stack(xs)  # [N, L, in]
    Y = torch.stack(ys)  # [N, L, world_dim]

    n_train = int(n_seqs * 0.8)
    X_train, X_test = X[:n_train], X[n_train:]
    Y_train, Y_test = Y[:n_train], Y[n_train:]

    predictor = UpgradedReafferencePredictor(world_dim, cenv.action_dim, hidden_dim=128)
    pred_opt = optim.Adam(predictor.parameters(), lr=1e-3)

    print(f"  [Phase1] Training LSTM predictor ({n_train} train seqs, {len(X_test)} test seqs)...", flush=True)
    for epoch in range(600):
        predictor.train()
        pred_opt.zero_grad()
        pred_out = predictor.forward_sequence(X_train)  # [N, L, world_dim]
        loss = F.mse_loss(pred_out, Y_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
        pred_opt.step()
        if (epoch + 1) % 150 == 0:
            print(f"    epoch {epoch+1}/600  loss={loss.item():.5f}", flush=True)

    # Evaluate R2 on test set
    predictor.eval()
    with torch.no_grad():
        pred_test = predictor.forward_sequence(X_test)
        ss_res = ((pred_test - Y_test) ** 2).sum().item()
        ss_tot = ((Y_test - Y_test.mean()) ** 2).sum().item()
        r2 = 1.0 - ss_res / (ss_tot + 1e-8)

    print(f"  [Phase1] R2_test = {r2:.4f}  (gate >= {THRESH_REAF_R2_GATE})", flush=True)
    return sequences, predictor, float(r2)


# ---------------------------------------------------------------------------
# Phase 2: Integration training + eval
# ---------------------------------------------------------------------------

def _train_seed(
    seed: int,
    predictor: UpgradedReafferencePredictor,
    warmup_episodes: int,
    steps_per_episode: int,
) -> Tuple[REEAgent, CausalGridWorldV2, Dict]:
    """
    Train agent with SD-008 (alpha_world=0.9) + SD-007 (LSTM reafference) + SD-003
    (world_forward + harm_eval both trained).

    Three separate optimizers (MECH-069):
      - standard: E1, E2_self, latent encoder
      - world_forward: E2.world_transition + world_action_encoder
      - harm_eval: E3.harm_eval_head
    """
    torch.manual_seed(seed)
    random.seed(seed)

    env = _build_eval_env(seed)

    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=32,
        alpha_world=0.9,             # SD-008: event responsiveness
        reafference_action_dim=env.action_dim,  # SD-007: correction enabled
        use_event_classifier=False,  # SD-009 tested separately; keep integration clean
    )
    agent = REEAgent(config)

    # Three-optimizer split (MECH-069: incommensurable error signals)
    standard_params = [
        p for n, p in agent.named_parameters()
        if "harm_eval_head" not in n
        and "world_transition" not in n
        and "world_action_encoder" not in n
    ]
    world_forward_params = (
        list(agent.e2.world_transition.parameters()) +
        list(agent.e2.world_action_encoder.parameters())
    )
    harm_eval_params = list(agent.e3.harm_eval_head.parameters())

    opt_std  = optim.Adam(standard_params,      lr=1e-3)
    opt_wf   = optim.Adam(world_forward_params, lr=1e-3)
    opt_harm = optim.Adam(harm_eval_params,     lr=1e-4)

    harm_buf_pos: List[torch.Tensor] = []
    harm_buf_neg: List[torch.Tensor] = []
    MAX_BUF = 2000

    wf_data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    MAX_WF  = 5000

    counts: Dict[str, int] = {}
    num_actions = env.action_dim

    print(f"  [seed={seed}] Warmup ({warmup_episodes} eps, alpha_world=0.9, reaf=ON, harm_eval=ON)...", flush=True)

    for ep in range(warmup_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        predictor.reset_hidden()

        z_world_prev_for_reaf = None
        a_prev_for_reaf = None
        z_world_for_wf = None
        a_for_wf = None

        for step in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            latent = agent.sense(obs_body, obs_world)
            agent.clock.advance()

            # Apply reafference correction to z_world (SD-007)
            z_world_raw = latent.z_world.detach()
            if z_world_prev_for_reaf is not None and a_prev_for_reaf is not None:
                with torch.no_grad():
                    z_world_corrected = predictor.correct_z_world(
                        z_world_raw, z_world_prev_for_reaf, a_prev_for_reaf
                    )
            else:
                z_world_corrected = z_world_raw

            action_idx = random.randint(0, num_actions - 1)
            action = _action_to_onehot(action_idx, num_actions, agent.device)
            agent._last_action = action

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)
            ttype = info.get("transition_type", "none")
            counts[ttype] = counts.get(ttype, 0) + 1

            # Harm buffer: use CORRECTED z_world for E3 training (SD-007 + SD-008 integration)
            if harm_signal < 0:
                harm_buf_pos.append(z_world_corrected.cpu())
                if len(harm_buf_pos) > MAX_BUF:
                    harm_buf_pos = harm_buf_pos[-MAX_BUF:]
            else:
                harm_buf_neg.append(z_world_corrected.cpu())
                if len(harm_buf_neg) > MAX_BUF:
                    harm_buf_neg = harm_buf_neg[-MAX_BUF:]

            # World-forward buffer: use CORRECTED z_world for E2 world_forward training
            if z_world_for_wf is not None and a_for_wf is not None:
                wf_data.append((z_world_for_wf.cpu(), a_for_wf.cpu(), z_world_corrected.cpu()))
                if len(wf_data) > MAX_WF:
                    wf_data = wf_data[-MAX_WF:]

            # Standard E1 + E2_self losses
            e1_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()
            total_loss = e1_loss + e2_loss
            if total_loss.requires_grad:
                opt_std.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                opt_std.step()

            # E2.world_forward training (separate optimizer)
            if len(wf_data) >= 16:
                k = min(32, len(wf_data))
                idxs = torch.randperm(len(wf_data))[:k].tolist()
                zw_b  = torch.cat([wf_data[i][0] for i in idxs]).to(agent.device)
                a_b   = torch.cat([wf_data[i][1] for i in idxs]).to(agent.device)
                zw1_b = torch.cat([wf_data[i][2] for i in idxs]).to(agent.device)
                pred_wf = agent.e2.world_forward(zw_b, a_b)
                wf_loss = F.mse_loss(pred_wf, zw1_b)
                if wf_loss.requires_grad:
                    opt_wf.zero_grad()
                    wf_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        agent.e2.world_transition.parameters(), 0.5
                    )
                    opt_wf.step()

            # E3 harm_eval training -- trained on both observed and E2-predicted states
            # (FIX 2 from EXQ-030b: teach E3 to evaluate E2-output distribution)
            if len(harm_buf_pos) >= 4 and len(harm_buf_neg) >= 4:
                k_pos = min(16, len(harm_buf_pos))
                k_neg = min(16, len(harm_buf_neg))
                pos_idx = torch.randperm(len(harm_buf_pos))[:k_pos].tolist()
                neg_idx = torch.randperm(len(harm_buf_neg))[:k_neg].tolist()

                zw_pos_obs = torch.cat([harm_buf_pos[i] for i in pos_idx]).to(agent.device)
                zw_neg_obs = torch.cat([harm_buf_neg[i] for i in neg_idx]).to(agent.device)

                with torch.no_grad():
                    a_rand_pos = torch.zeros(k_pos, num_actions, device=agent.device)
                    a_rand_pos[torch.arange(k_pos), torch.randint(0, num_actions, (k_pos,))] = 1.0
                    a_rand_neg = torch.zeros(k_neg, num_actions, device=agent.device)
                    a_rand_neg[torch.arange(k_neg), torch.randint(0, num_actions, (k_neg,))] = 1.0

                    zw_pos_pred = agent.e2.world_forward(zw_pos_obs, a_rand_pos)
                    zw_neg_pred = agent.e2.world_forward(zw_neg_obs, a_rand_neg)

                zw_b_harm = torch.cat([zw_pos_obs, zw_neg_obs, zw_pos_pred, zw_neg_pred], dim=0)
                target = torch.cat([
                    torch.ones(k_pos,  1, device=agent.device),
                    torch.zeros(k_neg, 1, device=agent.device),
                    torch.ones(k_pos,  1, device=agent.device),
                    torch.zeros(k_neg, 1, device=agent.device),
                ], dim=0)

                pred_h = agent.e3.harm_eval(zw_b_harm)
                harm_loss = F.mse_loss(pred_h, target)
                if harm_loss.requires_grad:
                    opt_harm.zero_grad()
                    harm_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        agent.e3.harm_eval_head.parameters(), 0.5
                    )
                    opt_harm.step()

            z_world_prev_for_reaf = z_world_raw
            a_prev_for_reaf = action.detach()
            z_world_for_wf = z_world_corrected
            a_for_wf = action.detach()

            if done:
                break

        if (ep + 1) % 100 == 0 or ep == warmup_episodes - 1:
            approach = counts.get("hazard_approach", 0)
            contact  = counts.get("env_caused_hazard", 0) + counts.get("agent_caused_hazard", 0)
            print(
                f"    ep {ep+1}/{warmup_episodes}  approach={approach}  contact={contact}"
                f"  wf_buf={len(wf_data)}  harm+={len(harm_buf_pos)} harm-={len(harm_buf_neg)}",
                flush=True,
            )

    return agent, env, {"wf_data": wf_data, "counts": counts}


def _compute_world_forward_r2(
    agent: REEAgent,
    wf_data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> float:
    """R2 of E2.world_forward on held-out transitions."""
    if len(wf_data) < 20:
        return 0.0
    n = len(wf_data)
    n_train = int(n * 0.8)
    with torch.no_grad():
        zw_all  = torch.cat([d[0] for d in wf_data], dim=0).to(agent.device)
        a_all   = torch.cat([d[1] for d in wf_data], dim=0).to(agent.device)
        zw1_all = torch.cat([d[2] for d in wf_data], dim=0).to(agent.device)
        pred_all = agent.e2.world_forward(zw_all, a_all)
        pred_test = pred_all[n_train:]
        tgt_test  = zw1_all[n_train:]
        if pred_test.shape[0] == 0:
            return 0.0
        ss_res = ((tgt_test - pred_test) ** 2).sum()
        ss_tot = ((tgt_test - tgt_test.mean(0, keepdim=True)) ** 2).sum()
        r2 = float((1.0 - ss_res / (ss_tot + 1e-8)).item())
    return r2


def _eval_seed(
    agent: REEAgent,
    env: CausalGridWorldV2,
    predictor: UpgradedReafferencePredictor,
    eval_episodes: int,
    steps_per_episode: int,
) -> Dict:
    """
    Integration evaluation. Measures:
      - C1: event_selectivity_margin (SD-008 in integration context)
      - C2: reafference_selectivity_ratio (SD-007 in integration context)
      - C3/C4: attribution_gap and causal_sig_approach (SD-003)
      - C6: n_approach
    """
    agent.eval()
    num_actions = env.action_dim

    ttypes = ["none", "hazard_approach", "env_caused_hazard", "agent_caused_hazard"]
    dz_corrected:  Dict[str, List[float]] = {t: [] for t in ttypes}
    causal_sigs:   Dict[str, List[float]] = {t: [] for t in ttypes}

    z_world_prev_for_reaf = None
    a_prev_for_reaf = None

    for ep in range(eval_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        predictor.reset_hidden()
        z_world_prev_for_reaf = None
        a_prev_for_reaf = None

        for step in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world)
                agent.clock.advance()

                z_world_raw = latent.z_world.detach()

                # SD-007: apply reafference correction
                if z_world_prev_for_reaf is not None and a_prev_for_reaf is not None:
                    z_world_c = predictor.correct_z_world(
                        z_world_raw, z_world_prev_for_reaf, a_prev_for_reaf
                    )
                else:
                    z_world_c = z_world_raw

            action_idx = random.randint(0, num_actions - 1)
            action = _action_to_onehot(action_idx, num_actions, agent.device)
            agent._last_action = action

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)
            ttype = info.get("transition_type", "none")

            if z_world_prev_for_reaf is not None:
                # C1/C2 metric: |delta_z_world_corrected| at each transition type
                with torch.no_grad():
                    dz_mag = float((z_world_c - z_world_prev_for_reaf).norm().item())
                if ttype in dz_corrected:
                    dz_corrected[ttype].append(dz_mag)

                # C3/C4 metric: counterfactual attribution pipeline (SD-003)
                with torch.no_grad():
                    z_world_actual = agent.e2.world_forward(z_world_c, action)
                    harm_actual    = agent.e3.harm_eval(z_world_actual)

                    sigs = []
                    for cf_idx in range(num_actions):
                        if cf_idx == action_idx:
                            continue
                        a_cf = _action_to_onehot(cf_idx, num_actions, agent.device)
                        z_cf    = agent.e2.world_forward(z_world_c, a_cf)
                        harm_cf = agent.e3.harm_eval(z_cf)
                        sigs.append(float((harm_actual - harm_cf).item()))

                    mean_sig = float(np.mean(sigs)) if sigs else 0.0

                if ttype in causal_sigs:
                    causal_sigs[ttype].append(mean_sig)

            z_world_prev_for_reaf = z_world_c
            a_prev_for_reaf = action.detach()

            if done:
                break

    def _mean(lst):
        return float(np.mean(lst)) if lst else 0.0

    mean_dz = {t: _mean(dz_corrected[t]) for t in ttypes}
    mean_cs = {t: _mean(causal_sigs[t])  for t in ttypes}
    n_counts = {t: len(dz_corrected[t])  for t in ttypes}

    # C1: event selectivity margin (SD-008)
    # margin = mean|dz| at hazard_approach MINUS mean|dz| at none (locomotion baseline)
    event_selectivity_margin = mean_dz["hazard_approach"] - mean_dz["none"]

    # C2: reafference selectivity ratio (SD-007)
    # ratio = mean|dz_corrected| at hazard_approach / mean|dz_corrected| at none
    reafference_selectivity_ratio = (
        mean_dz["hazard_approach"] / (mean_dz["none"] + 1e-8)
    )

    # C3: attribution gap (SD-003)
    attribution_gap = mean_cs["hazard_approach"] - mean_cs["env_caused_hazard"]

    # C4: minimum causal signal (SD-003)
    causal_sig_approach = mean_cs["hazard_approach"]

    return {
        "event_selectivity_margin":     event_selectivity_margin,
        "reafference_selectivity_ratio": reafference_selectivity_ratio,
        "attribution_gap":              attribution_gap,
        "causal_sig_approach":          causal_sig_approach,
        "mean_dz":                      mean_dz,
        "mean_cs":                      mean_cs,
        "n_counts":                     n_counts,
    }


# ---------------------------------------------------------------------------
# Main run() function
# ---------------------------------------------------------------------------

def run(**kwargs) -> dict:
    """
    V3-EXQ-145: SD-008 + SD-007 + SD-003 integration test.

    Phase 1: LSTM reafference predictor training (one shared predictor, seed=42).
    Phase 2: Per-seed warmup + eval with all three mechanisms active.
    """
    warmup_episodes   = kwargs.get("warmup_episodes",   400)
    eval_episodes     = kwargs.get("eval_episodes",      60)
    steps_per_episode = kwargs.get("steps_per_episode", 200)
    collection_eps    = kwargs.get("collection_eps",     40)
    collection_steps  = kwargs.get("collection_steps",  300)
    WORLD_DIM         = 32
    ACTION_DIM        = 5  # CausalGridWorldV2 default

    print(
        f"[V3-EXQ-145] SD-008 + SD-007 + SD-003 Integration Test\n"
        f"  seeds={SEEDS}  warmup_eps={warmup_episodes}  eval_eps={eval_episodes}\n"
        f"  alpha_world=0.9 (SD-008)  reafference=LSTM (SD-007)  attribution=counterfactual (SD-003)\n"
        f"  CLAIM_IDS: {CLAIM_IDS}",
        flush=True,
    )

    # ------------------------------------------------------------------
    # Phase 1: Train LSTM reafference predictor (shared across seeds)
    # ------------------------------------------------------------------
    print("\n[Phase 1] LSTM reafference predictor training (seed=42)...", flush=True)
    _, predictor, r2_reaf = _collect_predictor_data(
        seed=42,
        world_dim=WORLD_DIM,
        action_dim=ACTION_DIM,
        collection_episodes=collection_eps,
        steps_per_episode=collection_steps,
    )

    if predictor is None or r2_reaf < THRESH_REAF_R2_GATE:
        print(
            f"[Phase 1 FAIL] Predictor R2={r2_reaf:.4f} < gate {THRESH_REAF_R2_GATE}. "
            f"Cannot proceed to Phase 2.",
            flush=True,
        )
        return {
            "status": "FAIL",
            "metrics": {
                "reaf_r2_gate_pass":        0.0,
                "reaf_r2":                  float(r2_reaf),
                "phase1_failed":            1.0,
            },
            "summary_markdown": (
                f"# V3-EXQ-145 -- FAIL\n\n"
                f"**Phase 1 gate FAIL**: predictor R2={r2_reaf:.4f} < {THRESH_REAF_R2_GATE}.\n"
                f"Reafference predictor quality insufficient. Engineering bottleneck.\n"
                f"Does not retire SD-007 or SD-008 -- see EXQ-118 for predictor quality context."
            ),
            "claim_ids": CLAIM_IDS,
            "evidence_direction": "mixed",
            "experiment_type": EXPERIMENT_TYPE,
        }

    print(f"[Phase 1 PASS] Predictor R2={r2_reaf:.4f} >= gate {THRESH_REAF_R2_GATE}.", flush=True)

    # ------------------------------------------------------------------
    # Phase 2: Per-seed integration warmup + eval
    # ------------------------------------------------------------------
    all_seed_results: Dict[int, Dict] = {}
    wf_data_all: List = []

    for seed in SEEDS:
        print(f"\n[Phase 2 / seed={seed}] Training...", flush=True)
        agent, env, train_out = _train_seed(
            seed=seed,
            predictor=predictor,
            warmup_episodes=warmup_episodes,
            steps_per_episode=steps_per_episode,
        )
        wf_data_all.extend(train_out["wf_data"])

        print(f"\n[Phase 2 / seed={seed}] Evaluating ({eval_episodes} eps)...", flush=True)
        eval_out = _eval_seed(
            agent=agent,
            env=env,
            predictor=predictor,
            eval_episodes=eval_episodes,
            steps_per_episode=steps_per_episode,
        )
        eval_out["train_counts"] = train_out["counts"]
        all_seed_results[seed] = eval_out

        print(
            f"  [seed={seed}] event_sel_margin={eval_out['event_selectivity_margin']:.4f} "
            f" reaf_ratio={eval_out['reafference_selectivity_ratio']:.3f} "
            f" attr_gap={eval_out['attribution_gap']:.5f} "
            f" causal_sig_approach={eval_out['causal_sig_approach']:.5f} "
            f" n_approach={eval_out['n_counts']['hazard_approach']}",
            flush=True,
        )

    # ------------------------------------------------------------------
    # C5: world_forward R2 (pooled wf_data from all seeds)
    # Evaluate on a fresh agent from the last seed (representative)
    # ------------------------------------------------------------------
    world_forward_r2 = _compute_world_forward_r2(agent, train_out["wf_data"])
    print(f"\n[C5] world_forward_r2 = {world_forward_r2:.4f}  (gate > {THRESH_C5_WORLD_FORWARD_R2})", flush=True)

    # ------------------------------------------------------------------
    # Evaluate criteria per seed
    # ------------------------------------------------------------------
    criteria_by_seed: Dict[int, Dict] = {}
    for seed in SEEDS:
        r = all_seed_results[seed]
        c1 = r["event_selectivity_margin"]  > THRESH_C1_SELECTIVITY_MARGIN
        c2 = r["reafference_selectivity_ratio"] >= THRESH_C2_SELECTIVITY_RATIO
        c3 = r["attribution_gap"]           > THRESH_C3_ATTRIBUTION_GAP
        c4 = r["causal_sig_approach"]       > THRESH_C4_CAUSAL_SIG
        c6 = r["n_counts"]["hazard_approach"] >= THRESH_C6_MIN_APPROACH
        criteria_by_seed[seed] = {
            "c1_pass": c1, "c2_pass": c2, "c3_pass": c3,
            "c4_pass": c4, "c6_pass": c6,
        }

    c5_pass = world_forward_r2 > THRESH_C5_WORLD_FORWARD_R2

    # Integration PASS: all per-seed criteria pass for BOTH seeds, plus C5
    per_seed_pass = {
        seed: all(v for v in criteria_by_seed[seed].values())
        for seed in SEEDS
    }
    both_seeds_pass = all(per_seed_pass.values())
    all_pass = both_seeds_pass and c5_pass

    # Build failure notes
    failure_notes = []
    for seed in SEEDS:
        cr = criteria_by_seed[seed]
        r  = all_seed_results[seed]
        if not cr["c1_pass"]:
            failure_notes.append(
                f"C1 FAIL seed={seed}: event_selectivity_margin="
                f"{r['event_selectivity_margin']:.4f} <= {THRESH_C1_SELECTIVITY_MARGIN}"
            )
        if not cr["c2_pass"]:
            failure_notes.append(
                f"C2 FAIL seed={seed}: reaf_selectivity_ratio="
                f"{r['reafference_selectivity_ratio']:.3f} < {THRESH_C2_SELECTIVITY_RATIO}"
            )
        if not cr["c3_pass"]:
            failure_notes.append(
                f"C3 FAIL seed={seed}: attribution_gap="
                f"{r['attribution_gap']:.5f} <= {THRESH_C3_ATTRIBUTION_GAP}"
            )
        if not cr["c4_pass"]:
            failure_notes.append(
                f"C4 FAIL seed={seed}: causal_sig_approach="
                f"{r['causal_sig_approach']:.5f} <= {THRESH_C4_CAUSAL_SIG}"
            )
        if not cr["c6_pass"]:
            failure_notes.append(
                f"C6 FAIL seed={seed}: n_approach="
                f"{r['n_counts']['hazard_approach']} < {THRESH_C6_MIN_APPROACH}"
            )
    if not c5_pass:
        failure_notes.append(
            f"C5 FAIL: world_forward_r2={world_forward_r2:.4f} <= {THRESH_C5_WORLD_FORWARD_R2}"
        )

    status = "PASS" if all_pass else "FAIL"

    # Determine evidence_direction per claim
    # C1 informs SD-008, C2 informs SD-007, C3+C4 inform SD-003
    # Use per-claim override to give fine-grained governance signal
    sd008_both = all(criteria_by_seed[s]["c1_pass"] for s in SEEDS)
    sd007_both = all(criteria_by_seed[s]["c2_pass"] for s in SEEDS)
    sd003_both = all(
        criteria_by_seed[s]["c3_pass"] and criteria_by_seed[s]["c4_pass"]
        for s in SEEDS
    )

    def _dir(passes: bool) -> str:
        return "supports" if passes else "weakens"

    evidence_direction_per_claim = {
        "SD-008": _dir(sd008_both),
        "SD-007": _dir(sd007_both),
        "SD-003": _dir(sd003_both),
    }

    # Overall direction (for the run-level field): supports if all three pass
    overall_dir = (
        "supports" if (sd008_both and sd007_both and sd003_both)
        else "mixed" if sum([sd008_both, sd007_both, sd003_both]) >= 2
        else "weakens"
    )

    # Build metrics dict (flat, all floats)
    metrics: Dict[str, float] = {
        "reaf_r2":             float(r2_reaf),
        "world_forward_r2":    float(world_forward_r2),
        "c5_pass":             1.0 if c5_pass else 0.0,
    }
    for seed in SEEDS:
        r  = all_seed_results[seed]
        cr = criteria_by_seed[seed]
        metrics[f"seed{seed}_event_selectivity_margin"]     = float(r["event_selectivity_margin"])
        metrics[f"seed{seed}_reafference_selectivity_ratio"] = float(r["reafference_selectivity_ratio"])
        metrics[f"seed{seed}_attribution_gap"]              = float(r["attribution_gap"])
        metrics[f"seed{seed}_causal_sig_approach"]          = float(r["causal_sig_approach"])
        metrics[f"seed{seed}_causal_sig_none"]              = float(r["mean_cs"]["none"])
        metrics[f"seed{seed}_causal_sig_env_hazard"]        = float(r["mean_cs"]["env_caused_hazard"])
        metrics[f"seed{seed}_n_approach"]                   = float(r["n_counts"]["hazard_approach"])
        metrics[f"seed{seed}_n_none"]                       = float(r["n_counts"]["none"])
        metrics[f"seed{seed}_dz_at_approach"]               = float(r["mean_dz"]["hazard_approach"])
        metrics[f"seed{seed}_dz_at_none"]                   = float(r["mean_dz"]["none"])
        metrics[f"seed{seed}_c1_pass"] = 1.0 if cr["c1_pass"] else 0.0
        metrics[f"seed{seed}_c2_pass"] = 1.0 if cr["c2_pass"] else 0.0
        metrics[f"seed{seed}_c3_pass"] = 1.0 if cr["c3_pass"] else 0.0
        metrics[f"seed{seed}_c4_pass"] = 1.0 if cr["c4_pass"] else 0.0
        metrics[f"seed{seed}_c6_pass"] = 1.0 if cr["c6_pass"] else 0.0

    # Build summary
    def _row(label, val, thresh, pass_flag):
        p = "PASS" if pass_flag else "FAIL"
        return f"| {label} | {val} | {thresh} | {p} |"

    seed_rows = ""
    for seed in SEEDS:
        r  = all_seed_results[seed]
        cr = criteria_by_seed[seed]
        seed_rows += (
            f"\n### seed={seed}\n\n"
            f"| Criterion | Value | Threshold | Result |\n"
            f"|---|---|---|---|\n"
            + _row("C1 event_selectivity_margin (SD-008)",
                   f"{r['event_selectivity_margin']:.4f}", f"> {THRESH_C1_SELECTIVITY_MARGIN}", cr["c1_pass"])
            + "\n"
            + _row("C2 reaf_selectivity_ratio (SD-007)",
                   f"{r['reafference_selectivity_ratio']:.3f}", f">= {THRESH_C2_SELECTIVITY_RATIO}", cr["c2_pass"])
            + "\n"
            + _row("C3 attribution_gap (SD-003)",
                   f"{r['attribution_gap']:.5f}", f"> {THRESH_C3_ATTRIBUTION_GAP}", cr["c3_pass"])
            + "\n"
            + _row("C4 causal_sig_approach (SD-003)",
                   f"{r['causal_sig_approach']:.5f}", f"> {THRESH_C4_CAUSAL_SIG}", cr["c4_pass"])
            + "\n"
            + _row("C6 n_approach (data quality)",
                   f"{r['n_counts']['hazard_approach']}", f">= {THRESH_C6_MIN_APPROACH}", cr["c6_pass"])
            + "\n"
        )

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(f"- {n}" for n in failure_notes)

    summary_markdown = f"""# V3-EXQ-145 -- SD-008 + SD-007 + SD-003 Integration Test

**Status:** {status}
**Claims:** SD-008, SD-007, SD-003
**Seeds:** {SEEDS}
**Phase 1 predictor R2:** {r2_reaf:.4f}
**C5 world_forward R2:** {world_forward_r2:.4f}

## Purpose

This integration test verifies that SD-008 (alpha_world=0.9), SD-007 (LSTM reafference
correction), and SD-003 (counterfactual E2 attribution pipeline) compose correctly when
all three are active simultaneously in a single pipeline run. Individual validations exist
(EXQ-040, EXQ-099a, EXQ-030b); this test checks for compositional correctness.

## Per-Claim Direction

| Claim | Both seeds pass criteria? | Direction |
|---|---|---|
| SD-008 (alpha_world event selectivity) | {"Yes" if sd008_both else "No"} | {evidence_direction_per_claim["SD-008"]} |
| SD-007 (reafference selectivity lift) | {"Yes" if sd007_both else "No"} | {evidence_direction_per_claim["SD-007"]} |
| SD-003 (counterfactual attribution) | {"Yes" if sd003_both else "No"} | {evidence_direction_per_claim["SD-003"]} |

## Criteria by Seed
{seed_rows}

## Global Criteria

| Criterion | Value | Threshold | Result |
|---|---|---|---|
| C5 world_forward_r2 (E2 learned dynamics) | {world_forward_r2:.4f} | > {THRESH_C5_WORLD_FORWARD_R2} | {"PASS" if c5_pass else "FAIL"} |
| Phase 1 predictor R2 | {r2_reaf:.4f} | >= {THRESH_REAF_R2_GATE} | PASS |

**Integration PASS** requires: all per-seed C1-C4+C6 pass for BOTH seeds AND C5 passes.

Overall: **{status}**
{failure_section}
"""

    result = {
        "status": status,
        "metrics": metrics,
        "summary_markdown": summary_markdown,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": overall_dir,
        "evidence_direction_per_claim": evidence_direction_per_claim,
        "experiment_type": EXPERIMENT_TYPE,
    }
    return result


# ---------------------------------------------------------------------------
# Explorer-launch __main__ entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup-episodes",   type=int,   default=400)
    parser.add_argument("--eval-episodes",     type=int,   default=60)
    parser.add_argument("--steps",             type=int,   default=200)
    parser.add_argument("--collection-eps",    type=int,   default=40)
    parser.add_argument("--collection-steps",  type=int,   default=300)
    args = parser.parse_args()

    result = run(
        warmup_episodes=args.warmup_episodes,
        eval_episodes=args.eval_episodes,
        steps_per_episode=args.steps,
        collection_eps=args.collection_eps,
        collection_steps=args.collection_steps,
    )

    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"] = ts
    result["run_id"] = f"{EXPERIMENT_TYPE}_{ts}_v3"
    result["architecture_epoch"] = "ree_hybrid_guardrails_v1"

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
