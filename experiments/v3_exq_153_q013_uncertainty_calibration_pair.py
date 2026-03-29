#!/opt/local/bin/python3
"""
V3-EXQ-153 -- Q-013: Can deterministic JEPA plus derived dispersion match explicit
              stochastic uncertainty heads for REE precision routing?

Claim:    Q-013
Proposal: EXP-0100 (EVB-0077)

Q-013 asks:
  "Can deterministic JEPA plus derived dispersion match explicit stochastic uncertainty
  heads for REE precision routing?"

  The REE precision-routing machinery (ARC-005, ARC-004) relies on an uncertainty signal
  that gates E3 harm evaluation: when uncertainty is high, E3 should sharpen its harm
  gating; when uncertainty is low, it can relax. Two competing mechanisms for generating
  this uncertainty signal are:

    (A) DETERMINISTIC_DISPERSION -- JEPA-style position:
        z_world encoder is a standard deterministic encoder (no explicit uncertainty head).
        Uncertainty is derived post-hoc from the running variance of z_world across a
        short sliding window (window_size=10 steps). High variability in z_world =>
        high uncertainty. No reparametrization trick, no KL loss.
        Hypothesis: dispersion-derived uncertainty is a sufficient proxy for true epistemic
        uncertainty and achieves adequate harm-event prediction calibration.

    (B) STOCHASTIC_HEAD -- explicit uncertainty position:
        z_world encoder outputs both mu (mean) and log_var (log variance). The latent
        z_world is sampled via the reparametrization trick: z = mu + eps * sigma.
        Uncertainty = sigma (explicitly learned). A KL-regularisation term is added to
        training (KL divergence from N(0,1) to prevent posterior collapse).
        Hypothesis: explicitly learned sigma is a better-calibrated uncertainty signal
        that more reliably distinguishes harm-zone from safe-zone states.

  The discriminative question (REE-relevant operationalisation):
    Does the uncertainty signal (however derived) predict upcoming harm events?
    Measured by: AUC of uncertainty as a binary classifier for harm in the next K steps.
    This is the precision-routing criterion -- can the uncertainty signal route E3 attention
    toward harm-relevant states?

  The adjudication was "hybridize" (2026-02-25), meaning evidence is needed to determine
  whether deterministic dispersion is adequate or whether stochastic heads provide
  meaningfully better calibration. This experiment directly tests that question.

  Evidence interpretation:
    PASS: STOCHASTIC_HEAD calibration_auc substantially exceeds DETERMINISTIC_DISPERSION.
      => Q-013 resolved: stochastic uncertainty heads provide superior precision routing;
         hybridize recommendation is supported (stochastic heads needed for REE).
    PARTIAL_DET_ADEQUATE: Both conditions achieve acceptable AUC and STOCHASTIC_HEAD
      does not substantially exceed DETERMINISTIC_DISPERSION.
      => Q-013 partially resolved: deterministic dispersion is adequate at this scale;
         stochastic heads may be needed only at larger scale or with richer latent spaces.
    PARTIAL_BOTH_POOR: Both conditions fail to achieve calibration_auc >= 0.55.
      => Uncertainty signal not predictive of harm in either case; environment harm rate
         or uncertainty window too coarse; experiment inconclusive.
    FAIL: AUC invariant across conditions or n_harm_events too few to calibrate.
      => Implementation problem or insufficient harm encounters.

Pre-registered thresholds
--------------------------
C1: STOCHASTIC_HEAD achieves calibration_auc >= THRESH_CALIB_MIN = 0.58 (both seeds).
    (Stochastic head can predict upcoming harm above chance.)

C2: DETERMINISTIC_DISPERSION achieves calibration_auc >= THRESH_CALIB_MIN = 0.58 (both seeds).
    (Deterministic dispersion also reaches acceptable calibration level.)

C3: STOCHASTIC_HEAD calibration_auc exceeds DETERMINISTIC_DISPERSION by at least
    THRESH_STOCH_ADVANTAGE = 0.05 (both seeds).
    (Stochastic head provides meaningfully better calibration.)

C4: n_harm_events >= THRESH_MIN_HARM = 30 per condition per seed.
    (Data quality gate: sufficient harm encounters to calibrate AUC.)

C5: Seed consistency: direction of stochastic_auc vs det_auc is consistent across seeds.
    (Both seeds agree on which approach has higher AUC, or they are within 0.01.)

PASS: C1 + C3 + C4 + C5
  => Q-013 resolved in stochastic-heads direction. Hybridize recommendation supported.
     Explicit stochastic uncertainty heads provide substantially better precision routing.

PARTIAL (DET_ADEQUATE): C2 + C4 + C5 + NOT C3
  => Deterministic dispersion adequate at this scale. Stochastic heads do not offer
     a measurable advantage. REE can potentially defer stochastic uncertainty heads.

PARTIAL (BOTH_POOR): NOT C1 and NOT C2
  => Neither mechanism achieves calibrated uncertainty. Experiment inconclusive.
     May reflect insufficient harm rate, too-short window, or wrong env conditions.

FAIL: NOT C4 (too few harm events) or AUC values degenerate.
  => Implementation or configuration problem; not informative for Q-013.

Conditions
----------
DETERMINISTIC_DISPERSION:
  z_world encoder: standard Linear(world_obs_dim, world_dim) + LayerNorm.
  Uncertainty = std of z_world over last WINDOW_SIZE steps (running buffer).
  No KL loss. E1 prediction loss only.

STOCHASTIC_HEAD:
  z_world encoder: Linear(world_obs_dim, world_dim * 2) -> split into mu + log_var.
  z_world = mu + eps * exp(0.5 * log_var) (reparametrization).
  Uncertainty = exp(0.5 * log_var).mean() (sigma mean over latent dims).
  KL loss weight = KL_WEIGHT = 0.001 (light regularisation, prevents posterior collapse).
  E1 prediction loss + KL loss.

Both conditions:
  E1 deep predictor (LSTM) trained on world prediction error.
  No E3 harm training (avoids harm-supervision confound -- uncertainty must emerge
  from world-model training alone, not from explicit harm labels).
  Residue field disabled (avoids spatial-memory confound).
  Action policy: random (no RL signal -- pure world-model uncertainty calibration test).

Calibration metric:
  At each step, record (uncertainty_signal, harm_in_next_HORIZON steps).
  harm_in_next_HORIZON = 1 if any harm_signal > 0 in next HORIZON steps, else 0.
  AUC = area under ROC curve for (uncertainty_signal predicts harm_ahead).
  Computed over all eval-window steps with at least HORIZON lookahead.

Seeds:      [42, 123] (matched -- same env seed per condition)
Env:        CausalGridWorldV2 size=10, 4 hazards, 0 resources, hazard_harm=0.05,
            env_drift_interval=5, env_drift_prob=0.3
            (higher hazard density and frequent drift to produce ample harm events)
Protocol:   WARMUP_EPISODES=200 (train world model before calibration measurement)
            EVAL_EPISODES=50 (measure calibration)
            STEPS_PER_EPISODE=200
Estimated runtime: ~2 conditions x 2 seeds x 250 eps x 0.10 min/ep = ~100 min any machine
  (+20% overhead for uncertainty tracking) => ~120 min
"""

import sys
import random
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_153_q013_uncertainty_calibration_pair"
CLAIM_IDS = ["Q-013"]

# ---------------------------------------------------------------------------
# Pre-registered thresholds
# ---------------------------------------------------------------------------
THRESH_CALIB_MIN       = 0.58    # C1/C2: minimum AUC above chance for each condition
THRESH_STOCH_ADVANTAGE = 0.05    # C3: stochastic must exceed deterministic by this much
THRESH_MIN_HARM        = 30      # C4: minimum harm events per condition per seed for validity

# ---------------------------------------------------------------------------
# Protocol constants
# ---------------------------------------------------------------------------
WARMUP_EPISODES   = 200   # world-model training episodes before calibration measurement
EVAL_EPISODES     = 50    # episodes to measure calibration AUC
STEPS_PER_EPISODE = 200
LR                = 3e-4
KL_WEIGHT         = 0.001  # KL regularisation weight for STOCHASTIC_HEAD

WINDOW_SIZE = 10    # running variance window for DETERMINISTIC_DISPERSION
HORIZON     = 5     # lookahead steps for harm prediction in calibration

SEEDS      = [42, 123]
CONDITIONS = ["DETERMINISTIC_DISPERSION", "STOCHASTIC_HEAD"]

# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------
BODY_OBS_DIM  = 12
WORLD_OBS_DIM = 250   # CausalGridWorldV2 size=10
ACTION_DIM    = 5
WORLD_DIM     = 32


# ---------------------------------------------------------------------------
# Simple AUC computation (no sklearn dependency)
# ---------------------------------------------------------------------------

def _compute_auc(scores: List[float], labels: List[int]) -> float:
    """
    Compute AUC via the trapezoidal method.
    labels: 1 = positive (harm ahead), 0 = negative (no harm ahead).
    Returns AUC in [0, 1]. Returns 0.5 if no positive or no negative examples.
    """
    if len(scores) < 2:
        return 0.5
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    # Sort by descending score
    paired = sorted(zip(scores, labels), key=lambda x: -x[0])
    tp = 0
    fp = 0
    tpr_prev = 0.0
    fpr_prev = 0.0
    auc = 0.0

    for _score, label in paired:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tpr = tp / n_pos
        fpr = fp / n_neg
        # Trapezoidal area
        auc += (fpr - fpr_prev) * (tpr + tpr_prev) / 2.0
        tpr_prev = tpr
        fpr_prev = fpr

    return float(auc)


# ---------------------------------------------------------------------------
# Deterministic encoder (JEPA-style: plain linear + LayerNorm)
# ---------------------------------------------------------------------------

class DeterministicWorldEncoder(nn.Module):
    """
    Standard deterministic encoder.
    Maps world_obs -> z_world (no uncertainty head).
    Uncertainty derived post-hoc from running variance over z_world history.
    """
    def __init__(self, obs_dim: int, world_dim: int):
        super().__init__()
        self.fc = nn.Linear(obs_dim, world_dim)
        self.norm = nn.LayerNorm(world_dim)

    def forward(self, obs_world: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          z_world: [world_dim] encoded latent
          log_var: zeros (no uncertainty head -- will be derived externally)
        """
        z = self.norm(self.fc(obs_world))
        log_var = torch.zeros(z.shape, dtype=z.dtype)
        return z, log_var


# ---------------------------------------------------------------------------
# Stochastic encoder (reparametrization trick, mean + log_var head)
# ---------------------------------------------------------------------------

class StochasticWorldEncoder(nn.Module):
    """
    Stochastic encoder with explicit uncertainty head.
    Maps world_obs -> (mu, log_var); z_world sampled via reparametrization.
    """
    def __init__(self, obs_dim: int, world_dim: int):
        super().__init__()
        self.fc_shared = nn.Linear(obs_dim, world_dim * 2)
        self.fc_mu     = nn.Linear(world_dim * 2, world_dim)
        self.fc_logvar = nn.Linear(world_dim * 2, world_dim)

    def forward(self, obs_world: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          z_world: sampled [world_dim] latent (reparametrized)
          log_var: [world_dim] log variance
        """
        h       = F.relu(self.fc_shared(obs_world))
        mu      = self.fc_mu(h)
        log_var = self.fc_logvar(h).clamp(-4.0, 2.0)  # numerical stability
        eps     = torch.randn_like(mu)
        z_world = mu + eps * torch.exp(0.5 * log_var)
        return z_world, log_var


# ---------------------------------------------------------------------------
# Simple E1-style world predictor (MLP; no full LSTM to keep runtime feasible)
# ---------------------------------------------------------------------------

class SimpleWorldPredictor(nn.Module):
    """
    Lightweight world predictor: predicts next_z_world from (z_world, action).
    Used as E1 surrogate -- trains on world prediction error.
    """
    def __init__(self, world_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(world_dim + action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, world_dim),
        )

    def forward(self, z_world: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([z_world.view(1, -1), action.view(1, -1)], dim=-1)
        return self.net(inp).squeeze(0)


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed,
        size=10,
        num_hazards=4,     # higher hazard density: more harm events for calibration
        num_resources=0,
        hazard_harm=0.05,
        env_drift_interval=5,
        env_drift_prob=0.3,
    )


# ---------------------------------------------------------------------------
# Run one condition x seed
# ---------------------------------------------------------------------------

def _run_condition(
    seed: int,
    condition: str,
    warmup_episodes: int,
    eval_episodes: int,
    steps_per_episode: int,
    lr: float,
    dry_run: bool,
) -> Dict:
    """
    Run world-model training (warmup) then calibration measurement (eval).

    Returns dict with calibration_auc, n_harm_events, per-episode metrics.
    """
    if dry_run:
        warmup_episodes   = 4
        eval_episodes     = 4
        steps_per_episode = 20

    torch.manual_seed(seed)
    random.seed(seed)

    env = _make_env(seed)

    # Build encoder and predictor based on condition
    if condition == "DETERMINISTIC_DISPERSION":
        encoder   = DeterministicWorldEncoder(WORLD_OBS_DIM, WORLD_DIM)
        use_kl    = False
    else:  # STOCHASTIC_HEAD
        encoder   = StochasticWorldEncoder(WORLD_OBS_DIM, WORLD_DIM)
        use_kl    = True

    predictor = SimpleWorldPredictor(WORLD_DIM, ACTION_DIM)

    all_params = list(encoder.parameters()) + list(predictor.parameters())
    optimizer  = optim.Adam(all_params, lr=lr)

    # Running z_world buffer for deterministic dispersion
    z_world_buf: List[torch.Tensor] = []

    print(
        f"\n--- [{condition}] seed={seed}"
        f" warmup={warmup_episodes} eval={eval_episodes}"
        f" steps={steps_per_episode} kl={'ON' if use_kl else 'OFF'} ---",
        flush=True,
    )

    # ==========  WARMUP: train world predictor ====================
    print(f"  [{condition}] seed={seed} WARMUP phase ...", flush=True)

    total_warmup_loss = 0.0
    warmup_steps_total = 0

    _, obs_dict = env.reset()

    for ep in range(warmup_episodes):
        for _step in range(steps_per_episode):
            obs_world = torch.tensor(obs_dict["world_state"], dtype=torch.float32)

            z_world_cur, log_var_cur = encoder(obs_world)
            z_world_cur_det = z_world_cur.detach()

            # Keep buffer for deterministic dispersion
            z_world_buf.append(z_world_cur_det.clone())
            if len(z_world_buf) > WINDOW_SIZE:
                z_world_buf.pop(0)

            # Random action
            action_idx = random.randint(0, ACTION_DIM - 1)
            action = torch.zeros(ACTION_DIM)
            action[action_idx] = 1.0

            _, _, done, _, obs_dict_next = env.step(action.unsqueeze(0))

            # Prediction loss: predict z_world at next step
            obs_world_next = torch.tensor(obs_dict_next["world_state"], dtype=torch.float32)
            with torch.no_grad():
                z_world_next_target, _ = encoder(obs_world_next)

            z_world_pred = predictor(z_world_cur, action)
            pred_loss = F.mse_loss(z_world_pred, z_world_next_target.detach())

            # KL loss for stochastic head
            if use_kl:
                # KL(N(mu, sigma^2) || N(0,1)) = -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
                # Note: mu is implicit in z_world_cur before reparametrization;
                # approximate KL using log_var only (mean=0 approximation at start)
                kl_loss = -0.5 * torch.mean(1.0 + log_var_cur - log_var_cur.exp())
                total_loss = pred_loss + KL_WEIGHT * kl_loss
            else:
                total_loss = pred_loss

            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(all_params, 1.0)
            optimizer.step()

            total_warmup_loss += float(total_loss.item())
            warmup_steps_total += 1

            if done:
                _, obs_dict = env.reset()
                agent_reset = True
            else:
                obs_dict = obs_dict_next
                agent_reset = False

        if ep % 50 == 0:
            avg_loss = total_warmup_loss / max(warmup_steps_total, 1)
            print(
                f"  [{condition}] seed={seed} warmup ep={ep}/{warmup_episodes}"
                f" avg_loss={avg_loss:.5f}",
                flush=True,
            )

    mean_warmup_loss = total_warmup_loss / max(warmup_steps_total, 1)
    print(
        f"  [{condition}] seed={seed} WARMUP done mean_loss={mean_warmup_loss:.5f}",
        flush=True,
    )

    # ==========  EVAL: calibration measurement ====================
    print(f"  [{condition}] seed={seed} EVAL phase ...", flush=True)

    # Collect (uncertainty_signal, harm_signal) over entire eval window
    # Then compute calibration AUC with HORIZON lookahead
    uncertainty_log: List[float] = []
    harm_step_log:   List[float] = []   # harm_signal at each step

    _, obs_dict = env.reset()

    for ep in range(eval_episodes):
        for _step in range(steps_per_episode):
            obs_world = torch.tensor(obs_dict["world_state"], dtype=torch.float32)

            with torch.no_grad():
                z_world_cur, log_var_cur = encoder(obs_world)

            z_world_det = z_world_cur.detach()

            # Derive uncertainty signal
            if condition == "DETERMINISTIC_DISPERSION":
                z_world_buf.append(z_world_det.clone())
                if len(z_world_buf) > WINDOW_SIZE:
                    z_world_buf.pop(0)
                if len(z_world_buf) >= 2:
                    stacked = torch.stack(z_world_buf, dim=0)
                    # Mean per-dim std across dims -> scalar uncertainty
                    uncertainty = float(stacked.std(dim=0).mean().item())
                else:
                    uncertainty = 0.0
            else:  # STOCHASTIC_HEAD
                # sigma = exp(0.5 * log_var); mean over dims -> scalar
                uncertainty = float(torch.exp(0.5 * log_var_cur).mean().item())

            uncertainty_log.append(uncertainty)

            # Step
            action_idx = random.randint(0, ACTION_DIM - 1)
            action = torch.zeros(ACTION_DIM)
            action[action_idx] = 1.0
            _, reward, done, _, obs_dict = env.step(action.unsqueeze(0))

            harm_signal = float(min(0.0, reward))
            harm_step_log.append(abs(harm_signal))

            if done:
                _, obs_dict = env.reset()

    # Build calibration pairs: at each step t, label = 1 if harm in [t+1, t+HORIZON]
    n_steps = len(uncertainty_log)
    calib_scores: List[float] = []
    calib_labels: List[int]   = []

    for t in range(n_steps - HORIZON):
        harm_ahead = any(harm_step_log[t + 1 : t + 1 + HORIZON])
        calib_scores.append(uncertainty_log[t])
        calib_labels.append(1 if harm_ahead else 0)

    n_harm_events = int(sum(1 for h in harm_step_log if h > 0.0))
    calibration_auc = _compute_auc(calib_scores, calib_labels)

    mean_uncertainty = float(sum(uncertainty_log) / max(len(uncertainty_log), 1))

    print(
        f"  [{condition}] seed={seed} EVAL done"
        f" calibration_auc={calibration_auc:.4f}"
        f" n_harm={n_harm_events}"
        f" mean_uncertainty={mean_uncertainty:.5f}",
        flush=True,
    )

    return {
        "condition": condition,
        "seed": seed,
        "calibration_auc": calibration_auc,
        "n_harm_events": n_harm_events,
        "mean_uncertainty": mean_uncertainty,
        "mean_warmup_loss": mean_warmup_loss,
        "n_calib_steps": len(calib_scores),
        "n_positive_calib": int(sum(calib_labels)),
    }


# ---------------------------------------------------------------------------
# Criterion evaluation
# ---------------------------------------------------------------------------

def _evaluate_criteria(
    results_by_condition: Dict[str, List[Dict]],
) -> Dict[str, bool]:
    """Evaluate pre-registered criteria across both conditions and both seeds."""

    det   = results_by_condition["DETERMINISTIC_DISPERSION"]
    stoch = results_by_condition["STOCHASTIC_HEAD"]

    # C1: STOCHASTIC_HEAD calibration_auc >= THRESH_CALIB_MIN (both seeds)
    c1_seeds = [
        stoch[i]["calibration_auc"] >= THRESH_CALIB_MIN
        for i in range(len(stoch))
    ]
    c1 = all(c1_seeds)

    # C2: DETERMINISTIC_DISPERSION calibration_auc >= THRESH_CALIB_MIN (both seeds)
    c2_seeds = [
        det[i]["calibration_auc"] >= THRESH_CALIB_MIN
        for i in range(len(det))
    ]
    c2 = all(c2_seeds)

    # C3: STOCHASTIC_HEAD exceeds DETERMINISTIC by THRESH_STOCH_ADVANTAGE (both seeds)
    c3_seeds = [
        stoch[i]["calibration_auc"] - det[i]["calibration_auc"] >= THRESH_STOCH_ADVANTAGE
        for i in range(len(stoch))
    ]
    c3 = all(c3_seeds)

    # C4: n_harm_events >= THRESH_MIN_HARM for all conditions and seeds
    c4_cells = [
        r["n_harm_events"] >= THRESH_MIN_HARM
        for cond_results in results_by_condition.values()
        for r in cond_results
    ]
    c4 = all(c4_cells)

    # C5: seed consistency -- direction of stoch_auc vs det_auc consistent across seeds
    # (or within 0.01 -- effectively tied)
    c5_direction = [
        (stoch[i]["calibration_auc"] - det[i]["calibration_auc"]) > -0.01
        for i in range(len(stoch))
    ]
    c5 = all(c5_direction) or not any(c5_direction)

    return {
        "C1_stochastic_calibrated": c1,
        "C2_deterministic_calibrated": c2,
        "C3_stochastic_advantage": c3,
        "C4_sufficient_harm_events": c4,
        "C5_seed_consistent": c5,
    }


def _determine_outcome(criteria: Dict[str, bool]) -> str:
    c1 = criteria["C1_stochastic_calibrated"]
    c2 = criteria["C2_deterministic_calibrated"]
    c3 = criteria["C3_stochastic_advantage"]
    c4 = criteria["C4_sufficient_harm_events"]
    c5 = criteria["C5_seed_consistent"]

    # FAIL: insufficient harm events -- not enough data to calibrate
    if not c4:
        return "FAIL"

    # PASS: stochastic heads substantially better
    if c1 and c3 and c4 and c5:
        return "PASS"

    # PARTIAL (DET_ADEQUATE): deterministic dispersion adequate; stoch no advantage
    if c2 and c4 and c5 and not c3:
        return "PARTIAL_DET_ADEQUATE"

    # PARTIAL (BOTH_POOR): neither approach achieves calibration
    if not c1 and not c2:
        return "PARTIAL_BOTH_POOR"

    return "PARTIAL"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_experiment(dry_run: bool = False) -> Dict:
    """Run all conditions and compile the result pack."""
    print("=== V3-EXQ-153: Q-013 Uncertainty Calibration Pair ===", flush=True)
    print(f"Conditions: {CONDITIONS}  Seeds: {SEEDS}", flush=True)
    print("Pre-registered thresholds:", flush=True)
    print(f"  C1/C2 THRESH_CALIB_MIN       = {THRESH_CALIB_MIN}", flush=True)
    print(f"  C3    THRESH_STOCH_ADVANTAGE  = {THRESH_STOCH_ADVANTAGE}", flush=True)
    print(f"  C4    THRESH_MIN_HARM         = {THRESH_MIN_HARM}", flush=True)
    print(f"  WARMUP_EPISODES={WARMUP_EPISODES}  EVAL_EPISODES={EVAL_EPISODES}", flush=True)
    print(f"  WINDOW_SIZE={WINDOW_SIZE}  HORIZON={HORIZON}", flush=True)

    results_by_condition: Dict[str, List[Dict]] = {c: [] for c in CONDITIONS}

    for condition in CONDITIONS:
        print(f"\n=== Condition: {condition} ===", flush=True)
        for seed in SEEDS:
            result = _run_condition(
                seed=seed,
                condition=condition,
                warmup_episodes=WARMUP_EPISODES,
                eval_episodes=EVAL_EPISODES,
                steps_per_episode=STEPS_PER_EPISODE,
                lr=LR,
                dry_run=dry_run,
            )
            results_by_condition[condition].append(result)

    print("\n=== Evaluating criteria ===", flush=True)
    criteria = _evaluate_criteria(results_by_condition)
    outcome  = _determine_outcome(criteria)

    for k, v in criteria.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}", flush=True)
    print(f"Overall outcome: {outcome}", flush=True)

    # Summary metrics: mean over seeds per condition
    def _mean_seeds(cond: str, key: str) -> float:
        vals = [r[key] for r in results_by_condition[cond]]
        return float(sum(vals) / max(len(vals), 1))

    summary_metrics: Dict = {}
    for cond in CONDITIONS:
        prefix = cond.lower()
        summary_metrics[f"{prefix}_calibration_auc"]    = _mean_seeds(cond, "calibration_auc")
        summary_metrics[f"{prefix}_n_harm_events"]      = _mean_seeds(cond, "n_harm_events")
        summary_metrics[f"{prefix}_mean_uncertainty"]   = _mean_seeds(cond, "mean_uncertainty")
        summary_metrics[f"{prefix}_mean_warmup_loss"]   = _mean_seeds(cond, "mean_warmup_loss")

    # Pairwise delta (stochastic advantage)
    summary_metrics["delta_auc_stoch_vs_det"] = (
        summary_metrics["stochastic_head_calibration_auc"]
        - summary_metrics["deterministic_dispersion_calibration_auc"]
    )

    # Evidence direction
    if outcome == "PASS":
        evidence_direction = "supports"
        guidance = "stochastic_uncertainty_heads_superior_hybridize_confirmed"
    elif outcome == "PARTIAL_DET_ADEQUATE":
        evidence_direction = "mixed"
        guidance = "deterministic_dispersion_adequate_stochastic_not_needed_at_this_scale"
    elif outcome == "PARTIAL_BOTH_POOR":
        evidence_direction = "mixed"
        guidance = "neither_mechanism_calibrated_inconclusive"
    elif outcome == "PARTIAL":
        evidence_direction = "mixed"
        guidance = "partial_evidence_see_criteria"
    else:  # FAIL
        evidence_direction = "mixed"
        guidance = "insufficient_harm_events_or_degenerate_auc_implementation_problem"

    run_id = (
        "v3_exq_153_q013_uncertainty_calibration_"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "_v3"
    )

    pack = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "claim_ids": CLAIM_IDS,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "outcome": outcome,
        "evidence_direction": evidence_direction,
        "evidence_class": "discriminative_pair",
        "guidance": guidance,
        "criteria": criteria,
        "pre_registered_thresholds": {
            "THRESH_CALIB_MIN":       THRESH_CALIB_MIN,
            "THRESH_STOCH_ADVANTAGE": THRESH_STOCH_ADVANTAGE,
            "THRESH_MIN_HARM":        THRESH_MIN_HARM,
            "WINDOW_SIZE":            WINDOW_SIZE,
            "HORIZON":                HORIZON,
        },
        "summary_metrics": summary_metrics,
        "protocol": {
            "warmup_episodes":    WARMUP_EPISODES,
            "eval_episodes":      EVAL_EPISODES,
            "steps_per_episode":  STEPS_PER_EPISODE,
            "kl_weight":          KL_WEIGHT,
            "window_size":        WINDOW_SIZE,
            "horizon":            HORIZON,
        },
        "seeds": SEEDS,
        "scenario": (
            "Two-condition uncertainty calibration test:"
            " DETERMINISTIC_DISPERSION (plain linear encoder + running z_world std over"
            " last 10 steps as uncertainty signal),"
            " STOCHASTIC_HEAD (mean+log_var encoder, reparametrized z_world, sigma=exp(0.5*log_var))."
            " Both trained with E1 world prediction loss (no E3/harm supervision)."
            " KL_WEIGHT=0.001 for stochastic head."
            " Calibration metric: AUC of uncertainty predicting harm in next HORIZON=5 steps."
            " 2 seeds x 2 conditions = 4 cells."
            " CausalGridWorldV2 size=10 4 hazards 0 resources hazard_harm=0.05"
            " env_drift_interval=5 env_drift_prob=0.3 (high drift + hazard density)."
        ),
        "interpretation": (
            "PASS => Q-013 resolved: stochastic uncertainty heads provide substantially"
            " better calibration (AUC gap >= 0.05); hybridize recommendation supported."
            " REE precision routing requires explicit learned sigma, not just dispersion."
            " PARTIAL_DET_ADEQUATE => deterministic dispersion adequate at this scale;"
            " both achieve AUC >= 0.58; stochastic advantage not demonstrated."
            " REE may defer stochastic uncertainty heads in early V3 experiments."
            " PARTIAL_BOTH_POOR => neither mechanism achieves calibrated uncertainty;"
            " environment or window/horizon parameters need adjustment; inconclusive."
            " FAIL => too few harm events or degenerate AUC; implementation problem."
        ),
        "per_seed_results": {cond: results_by_condition[cond] for cond in CONDITIONS},
        "dry_run": dry_run,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    if not dry_run:
        out_dir = (
            Path(__file__).resolve().parents[2]
            / "REE_assembly"
            / "evidence"
            / "experiments"
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{run_id}.json"
        with open(out_path, "w") as f:
            json.dump(pack, f, indent=2)
        print(f"\nResult pack written to: {out_path}", flush=True)
    else:
        print("\n[dry_run] Result pack NOT written.", flush=True)
        print(json.dumps(pack, indent=2), flush=True)

    return pack


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    result = run_experiment(dry_run=dry_run)
    print(f"Done. Outcome: {result['outcome']}", flush=True)
