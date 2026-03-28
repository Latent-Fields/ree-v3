"""
V3-EXQ-111 -- SD-007 Discriminative Pair (REAF_ON vs REAF_OFF)

Claim: SD-007
Proposal: EVB-0003 / EXP-0003
Decision deadline: 2026-03-31

SD-007 asserts: "ReafferencePredictor provides perspective-corrected world latent by
subtracting self-caused sensory change." This is a DESIGN DECISION -- does enabling
the reafference gate (REAF_ON) improve z_world event selectivity compared to using
the uncorrected raw latent (REAF_OFF)?

Relationship to EXQ-110:
  EXQ-110 tests MECH-098 (the specific mechanism: z_world_raw_prev as predictor input).
  EXQ-111 tests SD-007 (the design decision: is the reafference gate beneficial at all?).
  Both compare REAF_ON vs REAF_OFF, but the governance question differs:
  SD-007 requires a retain_ree | hybridize | retire_ree_claim decision.

Historical context:
  EXQ-027b (PASS, diagnostic): correction_delta = -0.0450 -- correction was HURTING
  E3 calibration under the old MLP predictor trained on empty-steps-only.
  Root cause: insufficient training data (ttype=="none" filter), poor predictor quality.
  EXQ-110 fixed the predictor (LSTM, all-transitions training).
  EXQ-111 asks whether the fixed predictor makes the gate beneficial.

PRE-REGISTERED THRESHOLDS (hardcoded, not inferred post-hoc):
  C1 (predictor gate):     R2_test >= 0.10
    Same gate as EXQ-110. Predictor must capture at least some locomotion variance.
  C2 (selectivity lift):   mean_selectivity_ON >= 1.15 * mean_selectivity_OFF
    15% selectivity improvement required for retain_ree.
  C3 (consistency):        selectivity_ON > selectivity_OFF for BOTH seeds (pair_pass_count >= 2)
    Consistent direction across seeds required for retain_ree.

DECISION MAPPING (SD-007 governance outcomes):
  C1 AND C2 AND C3 -> PASS -> supports -> retain_ree
    Correction improves z_world selectivity robustly.
  C1 AND (sel_ON > sel_OFF mean) but C2 fails -> FAIL -> weakens -> hybridize
    Correction is directionally positive but below threshold (lambda scaling may help).
  C1 AND (sel_ON < sel_OFF mean) -> FAIL -> contradicts -> retire_ree_claim
    Correction actively hurts selectivity -- reafference gate is not beneficial.
  C1 AND C2 AND NOT C3 (inconsistent across seeds) -> FAIL -> ambiguous -> hybridize
    Selectivity mean passes but direction inconsistent -- needs redesign.
  NOT C1 -> FAIL -> not_applicable
    Engineering bottleneck -- predictor quality insufficient. Delays, does not retire SD-007.

Phase 1 -- Predictor collection + training:
  COLLECTION_ENV: size=8, num_hazards=1, proximity_approach_threshold=0.04
  collection_episodes=30, steps_per_episode=300, collect ALL transitions (no ttype filter)
  LSTM predictor: hidden_dim=128, 500 epochs, Adam lr=1e-3, gradient clip 1.0
  80/20 sequence-level train/test split.

Phase 2 -- Discriminative pair (only if Phase 1 passes):
  EVAL_ENV: size=8, num_hazards=4, proximity_approach_threshold=0.15
  Seeds: [42, 123] (matched, both run under both conditions)
  Conditions: REAF_ON (LSTM predictor active) vs REAF_OFF (no correction, z_world_raw)
  150 warmup + 50 eval episodes per seed per condition.
  Primary metric: event_selectivity = mean|delta_z_world| at hazard_approach /
                                       mean|delta_z_world| at none
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

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "discriminative_pair_sd007_reafference_gate"
CLAIM_IDS = ["SD-007"]

# Pre-registered thresholds (must be defined before execution, not post-hoc)
C1_R2_THRESHOLD = 0.10           # predictor quality gate
C2_SELECTIVITY_THRESHOLD = 1.15  # mean_sel_ON >= C2 * mean_sel_OFF for retain_ree
# C3 is consistency: sel_ON > sel_OFF for both seeds (pair_pass_count >= 2)


# ---------------------------------------------------------------------------
# LSTM reafference predictor (same architecture as EXQ-110)
# ---------------------------------------------------------------------------

class UpgradedReafferencePredictor(nn.Module):
    """
    LSTM reafference predictor (hidden_dim=128).

    Predicts delta_z_world (locomotion-induced perspective shift component)
    from (z_world_prev, action). After subtraction:
      z_world_corrected = z_world_raw - predicted_loco_delta
    The residual should be larger at genuine external events (hazard approach/contact)
    than at pure locomotion.

    Public interface (matches ReafferencePredictor in stack.py):
      forward(z_world_prev, a_prev) -> delta_z_world_loco  [single-step, stateful]
      forward_sequence(x_seq)       -> delta_seq            [BPTT training only]
      correct_z_world(z_world_raw, z_world_prev, a_prev) -> z_world_corrected
      reset_hidden()                                        [call at episode boundary]
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
        """Single-step stateful forward."""
        x = torch.cat([z_world_prev, a_prev], dim=-1).unsqueeze(1)  # [B, 1, in]
        out, self._hidden = self.lstm(x, self._hidden)
        return self.out(out.squeeze(1))  # [B, world_dim]

    def forward_sequence(self, x_seq: torch.Tensor) -> torch.Tensor:
        """Sequence forward for BPTT training (does not use internal hidden state)."""
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
    """Low-hazard environment for predictor data collection."""
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
    """Standard evaluation environment with 4 hazards."""
    return CausalGridWorldV2(
        seed=seed,
        size=8,
        num_hazards=4,
        num_resources=4,
        hazard_harm=0.02,
        env_drift_interval=5,
        env_drift_prob=0.1,
        proximity_harm_scale=0.05,
        proximity_benefit_scale=0.03,
        proximity_approach_threshold=0.15,
        hazard_field_decay=0.5,
    )


def _build_config(
    env: CausalGridWorldV2,
    self_dim: int,
    world_dim: int,
    alpha_world: float,
    alpha_self: float,
    reafference: bool,
) -> REEConfig:
    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=self_dim,
        world_dim=world_dim,
        alpha_world=alpha_world,
        alpha_self=alpha_self,
        reafference_action_dim=env.action_dim if reafference else 0,
    )
    config.latent.unified_latent_mode = False  # SD-005 split always on
    return config


# ---------------------------------------------------------------------------
# Phase 1: collect data + train predictor
# ---------------------------------------------------------------------------

def _collect_locomotion_sequences(
    seed: int,
    n_episodes: int,
    steps_per_episode: int,
    self_dim: int,
    world_dim: int,
    alpha_world: float,
    alpha_self: float,
    max_seq_len: int = 60,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Collect ALL transitions (no ttype filter) from the low-hazard collection environment.

    The predictor learns E[delta_z_world | z_world_prev, action] across all transition
    types. At genuine event steps the predictor underestimates because the event component
    is not predictable from (z_world_prev, action) alone. The residual captures unexpected
    world changes -- larger at genuine external events.

    Reafference is DISABLED so z_world_raw reflects the uncorrected encoder output.
    """
    torch.manual_seed(seed)
    random.seed(seed)

    env    = _build_collection_env(seed)
    config = _build_config(env, self_dim, world_dim, alpha_world, alpha_self, reafference=False)
    agent  = REEAgent(config)
    agent.eval()

    X_seqs: List[torch.Tensor] = []
    Y_seqs: List[torch.Tensor] = []

    for _ in range(n_episodes):
        _, obs_dict = env.reset()
        agent.reset()

        prev_z_world_raw: Optional[torch.Tensor] = None
        ep_inputs: List[torch.Tensor] = []
        ep_targets: List[torch.Tensor] = []

        for _ in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]

            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world)
            agent.clock.advance()

            z_world_raw_curr = latent.z_world_raw
            if z_world_raw_curr is None:
                z_world_raw_curr = latent.z_world

            action_idx = random.randint(0, env.action_dim - 1)
            action = _action_to_onehot(action_idx, env.action_dim, agent.device)
            agent._last_action = action

            _, _, done, _, obs_dict = env.step(action)

            # Collect ALL transitions (no ttype filter)
            if prev_z_world_raw is not None:
                delta = (z_world_raw_curr - prev_z_world_raw).detach()
                inp   = torch.cat(
                    [prev_z_world_raw.detach(), action.detach()], dim=-1
                )
                ep_inputs.append(inp.squeeze(0))
                ep_targets.append(delta.squeeze(0))

                if len(ep_inputs) >= max_seq_len:
                    X_seqs.append(torch.stack(ep_inputs))
                    Y_seqs.append(torch.stack(ep_targets))
                    ep_inputs  = []
                    ep_targets = []

            prev_z_world_raw = z_world_raw_curr.detach()

            if done:
                break

        if len(ep_inputs) >= 2:
            X_seqs.append(torch.stack(ep_inputs))
            Y_seqs.append(torch.stack(ep_targets))

    return X_seqs, Y_seqs


def _r2_on_sequences(
    pred: UpgradedReafferencePredictor,
    X_seqs: List[torch.Tensor],
    Y_seqs: List[torch.Tensor],
) -> float:
    """Compute R2 across all test sequences (flattened)."""
    all_pred: List[torch.Tensor] = []
    all_true: List[torch.Tensor] = []
    with torch.no_grad():
        for X_seq, Y_seq in zip(X_seqs, Y_seqs):
            p = pred.forward_sequence(X_seq.unsqueeze(0)).squeeze(0)
            all_pred.append(p)
            all_true.append(Y_seq)
    if not all_pred:
        return 0.0
    preds = torch.cat(all_pred, dim=0)
    trues = torch.cat(all_true, dim=0)
    ss_res = ((preds - trues) ** 2).sum().item()
    ss_tot = ((trues - trues.mean(dim=0)) ** 2).sum().item()
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-10 else 0.0


def _train_predictor(
    X_seqs: List[torch.Tensor],
    Y_seqs: List[torch.Tensor],
    world_dim: int,
    action_dim: int,
    n_epochs: int,
    lr: float,
) -> Tuple[UpgradedReafferencePredictor, float, int, int]:
    """
    Train LSTM predictor via BPTT. 80/20 sequence-level split.
    Returns (predictor, r2_test, n_train_steps, n_test_steps).
    """
    n_seqs  = len(X_seqs)
    n_steps = sum(len(x) for x in X_seqs)
    print(
        f"  [Phase1] Training on {n_seqs} sequences ({n_steps} steps)"
        f" ({n_epochs} epochs, lr={lr})",
        flush=True,
    )

    pred = UpgradedReafferencePredictor(world_dim, action_dim)

    if n_seqs < 2:
        print(
            f"  [Phase1] WARNING: only {n_seqs} sequences, insufficient."
            " R2 forced to 0.0.",
            flush=True,
        )
        return pred, 0.0, 0, 0

    perm    = list(range(n_seqs))
    random.shuffle(perm)
    n_train = max(1, int(0.8 * n_seqs))
    train_idx = perm[:n_train]
    test_idx  = perm[n_train:] if n_train < n_seqs else perm[:1]

    X_train = [X_seqs[i] for i in train_idx]
    Y_train = [Y_seqs[i] for i in train_idx]
    X_test  = [X_seqs[i] for i in test_idx]
    Y_test  = [Y_seqs[i] for i in test_idx]

    n_train_steps = sum(len(x) for x in X_train)
    n_test_steps  = sum(len(x) for x in X_test)

    opt = optim.Adam(pred.parameters(), lr=lr)

    pred.train()
    for epoch in range(n_epochs):
        epoch_loss  = 0.0
        epoch_steps = 0
        for X_seq, Y_seq in zip(X_train, Y_train):
            x = X_seq.unsqueeze(0)
            y = Y_seq.unsqueeze(0)
            pred_out = pred.forward_sequence(x)
            loss = F.mse_loss(pred_out, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pred.parameters(), 1.0)
            opt.step()
            epoch_loss  += loss.item() * len(X_seq)
            epoch_steps += len(X_seq)

        if (epoch + 1) % 100 == 0 or epoch == n_epochs - 1:
            pred.eval()
            r2 = _r2_on_sequences(pred, X_test, Y_test)
            pred.train()
            avg_loss = epoch_loss / max(1, epoch_steps)
            print(
                f"  [Phase1] epoch {epoch+1}/{n_epochs}"
                f" loss={avg_loss:.6f} R2_test={r2:.4f}",
                flush=True,
            )

    pred.eval()
    r2_final = _r2_on_sequences(pred, X_test, Y_test)
    print(
        f"  [Phase1] Final R2_test={r2_final:.4f}"
        f" (gate threshold={C1_R2_THRESHOLD:.2f})",
        flush=True,
    )
    return pred, r2_final, n_train_steps, n_test_steps


# ---------------------------------------------------------------------------
# Phase 2: single-condition run
# ---------------------------------------------------------------------------

def _run_single(
    seed: int,
    reafference: bool,
    upgraded_predictor: Optional[UpgradedReafferencePredictor],
    warmup_episodes: int,
    eval_episodes: int,
    steps_per_episode: int,
    self_dim: int,
    world_dim: int,
    lr: float,
    alpha_world: float,
    alpha_self: float,
) -> Dict:
    """
    Run one condition (REAF_ON or REAF_OFF) in the evaluation environment.

    Returns dict with:
      event_selectivity:   mean|delta_z_world| at hazard_approach /
                           mean|delta_z_world| at none
      harm_avoidance_rate: fraction of hazard_approach steps where harm_eval > 0.5
    """
    torch.manual_seed(seed)
    random.seed(seed)

    cond_label = "REAF_ON" if reafference else "REAF_OFF"
    print(f"\n  [Phase2] seed={seed} condition={cond_label}", flush=True)

    env    = _build_eval_env(seed)
    config = _build_config(env, self_dim, world_dim, alpha_world, alpha_self, reafference)
    agent  = REEAgent(config)

    # Swap in upgraded predictor for REAF_ON
    if reafference and upgraded_predictor is not None:
        pred_copy = copy.deepcopy(upgraded_predictor).to(agent.device)
        agent.latent_stack.reafference_predictor = pred_copy

    harm_buf_pos: List[torch.Tensor] = []
    harm_buf_neg: List[torch.Tensor] = []
    MAX_BUF = 2000

    standard_params  = [p for n, p in agent.named_parameters() if "harm_eval_head" not in n]
    harm_eval_params = list(agent.e3.harm_eval_head.parameters())
    optimizer          = optim.Adam(standard_params, lr=lr)
    harm_eval_optimizer = optim.Adam(harm_eval_params, lr=1e-4)

    train_approach = 0
    train_contact  = 0

    # --- TRAIN ---
    agent.train()
    for ep in range(warmup_episodes):
        _, obs_dict = env.reset()
        agent.reset()
        if reafference and upgraded_predictor is not None:
            agent.latent_stack.reafference_predictor.reset_hidden()

        for _ in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            latent = agent.sense(obs_body, obs_world)
            agent.clock.advance()

            z_world_curr = latent.z_world.detach()

            action_idx = random.randint(0, env.action_dim - 1)
            action = _action_to_onehot(action_idx, env.action_dim, agent.device)
            agent._last_action = action

            _, harm_signal, done, info, obs_dict = env.step(action)
            ttype = info.get("transition_type", "none")
            if ttype == "hazard_approach":
                train_approach += 1
            elif ttype in ("env_caused_hazard", "agent_caused_hazard"):
                train_contact += 1

            is_harm = float(harm_signal) < 0
            if is_harm:
                harm_buf_pos.append(z_world_curr)
                if len(harm_buf_pos) > MAX_BUF:
                    harm_buf_pos = harm_buf_pos[-MAX_BUF:]
            else:
                harm_buf_neg.append(z_world_curr)
                if len(harm_buf_neg) > MAX_BUF:
                    harm_buf_neg = harm_buf_neg[-MAX_BUF:]

            e1_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()
            total_loss = e1_loss + e2_loss
            if total_loss.requires_grad:
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                optimizer.step()

            if len(harm_buf_pos) >= 4 and len(harm_buf_neg) >= 4:
                k_pos = min(16, len(harm_buf_pos))
                k_neg = min(16, len(harm_buf_neg))
                pos_idx = torch.randperm(len(harm_buf_pos))[:k_pos].tolist()
                neg_idx = torch.randperm(len(harm_buf_neg))[:k_neg].tolist()
                zw_pos = torch.cat([harm_buf_pos[i] for i in pos_idx], dim=0)
                zw_neg = torch.cat([harm_buf_neg[i] for i in neg_idx], dim=0)
                zw_b   = torch.cat([zw_pos, zw_neg], dim=0)
                target_h = torch.cat([
                    torch.ones(k_pos,  1, device=agent.device),
                    torch.zeros(k_neg, 1, device=agent.device),
                ], dim=0)
                pred_harm = agent.e3.harm_eval(zw_b)
                harm_loss = F.mse_loss(pred_harm, target_h)
                if harm_loss.requires_grad:
                    harm_eval_optimizer.zero_grad()
                    harm_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        agent.e3.harm_eval_head.parameters(), 0.5,
                    )
                    harm_eval_optimizer.step()

            if done:
                break

        if (ep + 1) % 50 == 0 or ep == warmup_episodes - 1:
            print(
                f"  [train] seed={seed} cond={cond_label}"
                f" ep {ep+1}/{warmup_episodes}"
                f" approach={train_approach} contact={train_contact}"
                f" buf_pos={len(harm_buf_pos)} buf_neg={len(harm_buf_neg)}",
                flush=True,
            )

    # --- EVAL ---
    agent.eval()

    deltas_approach: List[float] = []
    deltas_none:     List[float] = []
    approach_detected = 0
    approach_total    = 0

    for _ in range(eval_episodes):
        _, obs_dict = env.reset()
        agent.reset()
        if reafference and upgraded_predictor is not None:
            agent.latent_stack.reafference_predictor.reset_hidden()
        prev_z_world: Optional[torch.Tensor] = None

        for _ in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]

            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world)
            agent.clock.advance()

            z_world_curr = latent.z_world.detach()

            action_idx = random.randint(0, env.action_dim - 1)
            action = _action_to_onehot(action_idx, env.action_dim, agent.device)
            agent._last_action = action

            _, _, done, info, obs_dict = env.step(action)
            ttype = info.get("transition_type", "none")

            if prev_z_world is not None:
                delta_mag = float(
                    (z_world_curr - prev_z_world).abs().mean().item()
                )
                if ttype == "hazard_approach":
                    deltas_approach.append(delta_mag)
                elif ttype == "none":
                    deltas_none.append(delta_mag)

            try:
                with torch.no_grad():
                    harm_score = float(agent.e3.harm_eval(z_world_curr).item())
                if ttype == "hazard_approach":
                    approach_total += 1
                    if harm_score > 0.5:
                        approach_detected += 1
            except Exception:
                pass

            prev_z_world = z_world_curr

            if done:
                break

    mean_delta_approach = float(sum(deltas_approach) / max(1, len(deltas_approach)))
    mean_delta_none     = float(sum(deltas_none)     / max(1, len(deltas_none)))
    event_selectivity   = mean_delta_approach / max(mean_delta_none, 1e-8)
    harm_avoidance_rate = float(approach_detected) / max(approach_total, 1)

    print(
        f"  [eval] seed={seed} cond={cond_label}"
        f" selectivity={event_selectivity:.4f}"
        f" (approach={mean_delta_approach:.4f}/none={mean_delta_none:.4f})"
        f" harm_avoidance={harm_avoidance_rate:.4f}"
        f" ({approach_detected}/{approach_total})"
        f" n_approach={len(deltas_approach)} n_none={len(deltas_none)}",
        flush=True,
    )

    return {
        "seed": seed,
        "condition": cond_label,
        "event_selectivity": float(event_selectivity),
        "harm_avoidance_rate": float(harm_avoidance_rate),
        "mean_delta_approach": float(mean_delta_approach),
        "mean_delta_none": float(mean_delta_none),
        "n_approach": int(approach_total),
        "n_approach_detected": int(approach_detected),
        "n_none_deltas": int(len(deltas_none)),
        "train_approach_events": int(train_approach),
        "train_contact_events": int(train_contact),
    }


# ---------------------------------------------------------------------------
# Output writer
# ---------------------------------------------------------------------------

def _write_output(result: dict) -> None:
    out_dir = Path(__file__).resolve().parents[2] / "REE_assembly" / "evidence" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.utcnow().strftime("T%H%M%SZ")
    fname = out_dir / f"{EXPERIMENT_TYPE}_{ts}.json"
    with open(fname, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[EXQ-111] Output written to: {fname}", flush=True)


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------

def run(
    collection_episodes: int = 30,
    collection_steps: int = 300,
    warmup_episodes: int = 150,
    eval_episodes: int = 50,
    steps_per_episode: int = 200,
    self_dim: int = 32,
    world_dim: int = 32,
    lr: float = 1e-3,
    alpha_world: float = 0.9,
    alpha_self: float = 0.3,
    predictor_epochs: int = 500,
    predictor_lr: float = 1e-3,
    seeds: Optional[List[int]] = None,
    dry_run: bool = False,
    **kwargs,
) -> dict:
    """
    Two-phase SD-007 discriminative pair.

    Phase 1: train upgraded LSTM predictor using low-hazard collection env.
    Phase 2: discriminative pair REAF_ON vs REAF_OFF (only if Phase 1 passes).
    """
    if seeds is None:
        seeds = [42, 123]

    if dry_run:
        collection_episodes = 2
        collection_steps    = 50
        warmup_episodes     = 5
        eval_episodes       = 3
        steps_per_episode   = 50
        predictor_epochs    = 5
        print("[EXQ-111] DRY RUN mode -- short parameters.", flush=True)

    print(
        f"\n[V3-EXQ-111] SD-007 discriminative pair (REAF_ON vs REAF_OFF)"
        f" alpha_world={alpha_world}"
        f" collection={collection_episodes}eps x {collection_steps}steps"
        f" warmup={warmup_episodes} eval={eval_episodes}"
        f" seeds={seeds}",
        flush=True,
    )

    ts_start = datetime.datetime.utcnow()

    env_dummy = _build_collection_env(seeds[0])

    # ---- Phase 1: collect + train ----
    print("\n[V3-EXQ-111] Phase 1 -- collecting locomotion sequences...", flush=True)
    X_seqs, Y_seqs = _collect_locomotion_sequences(
        seed=seeds[0],
        n_episodes=collection_episodes,
        steps_per_episode=collection_steps,
        self_dim=self_dim,
        world_dim=world_dim,
        alpha_world=alpha_world,
        alpha_self=alpha_self,
    )
    n_seqs  = len(X_seqs)
    n_steps = sum(len(x) for x in X_seqs)
    print(f"  [Phase1] Collected {n_seqs} sequences ({n_steps} steps).", flush=True)

    upgraded_predictor, r2_test, n_train_steps, n_test_steps = _train_predictor(
        X_seqs=X_seqs,
        Y_seqs=Y_seqs,
        world_dim=world_dim,
        action_dim=env_dummy.action_dim,
        n_epochs=predictor_epochs,
        lr=predictor_lr,
    )

    c1_pass = r2_test >= C1_R2_THRESHOLD
    print(
        f"  [Phase1] C1 {'PASS' if c1_pass else 'FAIL'}"
        f" R2_test={r2_test:.4f} threshold={C1_R2_THRESHOLD:.2f}",
        flush=True,
    )

    if not c1_pass:
        result = {
            "run_id": f"v3_exq_111_c1fail_{ts_start.strftime('%Y%m%dT%H%M%SZ')}_v3",
            "architecture_epoch": "ree_hybrid_guardrails_v1",
            "experiment_type": EXPERIMENT_TYPE,
            "claim_ids": CLAIM_IDS,
            "claim_ids_tested": CLAIM_IDS,
            "status": "FAIL",
            "fail_reason": "C1_FAIL_predictor_quality_insufficient",
            "r2_test": float(r2_test),
            "n_train_samples": int(n_train_steps),
            "n_test_samples": int(n_test_steps),
            "n_locomotion_sequences": int(n_seqs),
            "n_locomotion_steps": int(n_steps),
            "c1_pass": False,
            "c2_pass": False,
            "c3_pass": False,
            "pair_pass_count": 0,
            "registered_c1_threshold": C1_R2_THRESHOLD,
            "registered_c2_threshold": C2_SELECTIVITY_THRESHOLD,
            "evidence_class": "discriminative_pair",
            "evidence_direction": "not_applicable",
            "recommended_outcome": "hold",
            "outcome_scores": {
                "retain_ree": False,
                "hybridize": False,
                "retire_ree_claim": False,
            },
            "summary": (
                f"SD-007 Phase 1 gate FAIL: R2_test={r2_test:.4f} < threshold={C1_R2_THRESHOLD:.2f}. "
                f"Engineering bottleneck prevents Phase 2. Collected {n_steps} steps from "
                f"{collection_episodes} episodes in low-hazard env (size=8, 1 hazard). "
                "Does NOT retire SD-007 -- predictor quality insufficient to test design decision."
            ),
        }
        _write_output(result)
        return result

    # ---- Phase 2: discriminative pair ----
    print(
        f"\n[V3-EXQ-111] Phase 2 -- discriminative pair"
        f" seeds={seeds}"
        f" C2_threshold={C2_SELECTIVITY_THRESHOLD}",
        flush=True,
    )

    condition_results: List[Dict] = []
    for seed in seeds:
        for reaf_on in [True, False]:
            res = _run_single(
                seed=seed,
                reafference=reaf_on,
                upgraded_predictor=upgraded_predictor if reaf_on else None,
                warmup_episodes=warmup_episodes,
                eval_episodes=eval_episodes,
                steps_per_episode=steps_per_episode,
                self_dim=self_dim,
                world_dim=world_dim,
                lr=lr,
                alpha_world=alpha_world,
                alpha_self=alpha_self,
            )
            condition_results.append(res)

    # Aggregate results
    def _get(cond_results, seed, cond):
        for r in cond_results:
            if r["seed"] == seed and r["condition"] == cond:
                return r
        return {}

    sel_on_42   = _get(condition_results, 42,  "REAF_ON").get("event_selectivity", 0.0)
    sel_off_42  = _get(condition_results, 42,  "REAF_OFF").get("event_selectivity", 1e-8)
    sel_on_123  = _get(condition_results, 123, "REAF_ON").get("event_selectivity", 0.0)
    sel_off_123 = _get(condition_results, 123, "REAF_OFF").get("event_selectivity", 1e-8)

    harm_on_42   = _get(condition_results, 42,  "REAF_ON").get("harm_avoidance_rate", 0.0)
    harm_off_42  = _get(condition_results, 42,  "REAF_OFF").get("harm_avoidance_rate", 0.0)
    harm_on_123  = _get(condition_results, 123, "REAF_ON").get("harm_avoidance_rate", 0.0)
    harm_off_123 = _get(condition_results, 123, "REAF_OFF").get("harm_avoidance_rate", 0.0)

    mean_sel_on  = (sel_on_42  + sel_on_123)  / 2.0
    mean_sel_off = (sel_off_42 + sel_off_123) / 2.0
    selectivity_ratio = mean_sel_on / max(mean_sel_off, 1e-8)
    sel_delta = mean_sel_on - mean_sel_off

    mean_harm_on  = (harm_on_42  + harm_on_123)  / 2.0
    mean_harm_off = (harm_off_42 + harm_off_123) / 2.0

    # C3: consistent direction across BOTH seeds
    pair_pass_count = int(sel_on_42 > sel_off_42) + int(sel_on_123 > sel_off_123)
    c3_pass = pair_pass_count >= 2  # both seeds must show positive direction

    c2_pass = mean_sel_on >= C2_SELECTIVITY_THRESHOLD * mean_sel_off
    overall_pass = c1_pass and c2_pass and c3_pass

    # Decision mapping for SD-007 governance
    correction_positive = sel_delta > 0
    if c2_pass and c3_pass:
        evidence_direction = "supports"
        recommended_outcome = "retain_ree"
    elif correction_positive and not c2_pass:
        evidence_direction = "weakens"
        recommended_outcome = "hybridize"  # directional but below threshold
    elif not correction_positive:
        evidence_direction = "contradicts"
        recommended_outcome = "retire_ree_claim"  # correction actively hurts
    else:
        # c2 passes but c3 fails (inconsistent across seeds)
        evidence_direction = "ambiguous"
        recommended_outcome = "hybridize"

    outcome_scores = {
        "retain_ree": bool(c2_pass and c3_pass),
        "hybridize": bool(recommended_outcome == "hybridize"),
        "retire_ree_claim": bool(recommended_outcome == "retire_ree_claim"),
    }

    run_id = (
        f"v3_exq_111_{'pass' if overall_pass else 'fail'}"
        f"_{ts_start.strftime('%Y%m%dT%H%M%SZ')}_v3"
    )

    summary = (
        f"SD-007 reafference gate discriminative pair. "
        f"C1 PASS: R2_test={r2_test:.4f} (collected {n_steps} steps, {n_seqs} seqs). "
        f"Selectivity REAF_ON={mean_sel_on:.4f} vs REAF_OFF={mean_sel_off:.4f} "
        f"(ratio={selectivity_ratio:.3f}, delta={sel_delta:+.4f}). "
        f"C2 {'PASS' if c2_pass else 'FAIL'} (threshold={C2_SELECTIVITY_THRESHOLD}). "
        f"C3 {'PASS' if c3_pass else 'FAIL'} (pair_pass_count={pair_pass_count}/2). "
        f"Harm avoidance ON={mean_harm_on:.4f} vs OFF={mean_harm_off:.4f} "
        f"(delta={mean_harm_on - mean_harm_off:+.4f}). "
        f"Overall: {'PASS' if overall_pass else 'FAIL'}. "
        f"Recommended outcome: {recommended_outcome}."
    )

    print(f"\n[EXQ-111] C1=PASS R2={r2_test:.4f}", flush=True)
    print(
        f"[EXQ-111] C2={'PASS' if c2_pass else 'FAIL'}"
        f" sel_ON={mean_sel_on:.4f} vs OFF={mean_sel_off:.4f}"
        f" ratio={selectivity_ratio:.3f}",
        flush=True,
    )
    print(
        f"[EXQ-111] C3={'PASS' if c3_pass else 'FAIL'}"
        f" pair_pass_count={pair_pass_count}/2",
        flush=True,
    )
    print(f"[EXQ-111] Overall: {'PASS' if overall_pass else 'FAIL'}", flush=True)
    print(f"[EXQ-111] Recommended outcome: {recommended_outcome}", flush=True)
    print(f"[EXQ-111] {summary}", flush=True)

    result = {
        "run_id": run_id,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "experiment_type": EXPERIMENT_TYPE,
        "claim_ids": CLAIM_IDS,
        "claim_ids_tested": CLAIM_IDS,
        "status": "PASS" if overall_pass else "FAIL",
        # Phase 1 predictor metrics
        "r2_test": float(r2_test),
        "n_train_samples": int(n_train_steps),
        "n_test_samples": int(n_test_steps),
        "n_locomotion_sequences": int(n_seqs),
        "n_locomotion_steps": int(n_steps),
        # Phase 2 per-seed selectivity
        "selectivity_on_s42":   float(sel_on_42),
        "selectivity_off_s42":  float(sel_off_42),
        "selectivity_on_s123":  float(sel_on_123),
        "selectivity_off_s123": float(sel_off_123),
        "mean_selectivity_on":  float(mean_sel_on),
        "mean_selectivity_off": float(mean_sel_off),
        "selectivity_ratio":    float(selectivity_ratio),
        "selectivity_delta":    float(sel_delta),
        # Phase 2 per-seed harm avoidance
        "harm_avoidance_on_s42":   float(harm_on_42),
        "harm_avoidance_off_s42":  float(harm_off_42),
        "harm_avoidance_on_s123":  float(harm_on_123),
        "harm_avoidance_off_s123": float(harm_off_123),
        "harm_avoidance_mean_on":  float(mean_harm_on),
        "harm_avoidance_mean_off": float(mean_harm_off),
        # Criteria
        "c1_pass": bool(c1_pass),
        "c2_pass": bool(c2_pass),
        "c3_pass": bool(c3_pass),
        "pair_pass_count": int(pair_pass_count),
        "registered_c1_threshold": C1_R2_THRESHOLD,
        "registered_c2_threshold": C2_SELECTIVITY_THRESHOLD,
        # Governance
        "evidence_class": "discriminative_pair",
        "evidence_direction": evidence_direction,
        "recommended_outcome": recommended_outcome,
        "outcome_scores": outcome_scores,
        "summary": summary,
    }

    _write_output(result)
    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    result  = run(dry_run=dry_run)
    status  = result.get("status", "UNKNOWN")
    print(f"\n[EXQ-111] EXIT status={status}", flush=True)
    sys.exit(0)
