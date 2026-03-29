#!/opt/local/bin/python3
"""
V3-EXQ-154 -- Q-014: Do JEPA invariances hide ethically relevant distinctions
              in REE attribution contexts?

Claim:    Q-014
Proposal: EXP-0102 (EVB-0078)

Q-014 asks:
  "Do JEPA invariances hide ethically relevant distinctions in REE attribution
  contexts?"

  The REE attribution pipeline depends on z_world carrying information that
  discriminates between ethically distinct event types -- specifically:
    (A) agent-caused harm (the agent's action caused the harm contact)
    (B) environment-caused harm (hazard moved onto the agent, no causal role)
    (C) safe states (no harm)

  JEPA-style encoders are trained to predict the *invariant* features of future
  states by routing the prediction loss only through a predictor network (not
  the encoder), and optionally using a stop-gradient on the target encoder. The
  key property: the encoder is free to collapse features that are unpredictable
  or high-frequency, including exactly the features that distinguish agent-
  caused from environment-caused harm (which depend on the specific action taken
  vs. the agent's spatial relation to a hazard -- not on the invariant world
  structure).

  Two competing encoder training regimes are tested:

    (A) JEPA_INVARIANT -- JEPA-style invariance training:
        Encoder trained with stop-gradient on target, predictor-only loss
        gradient flow (encoder learns only invariant features). A collapse-
        prevention regularizer (VICReg-style variance term) is added to prevent
        degenerate collapse to zero but does NOT restore harm-source distinctions.
        The encoder learns world-state representations that are maximally
        predictable but invariant to the noise dimensions -- including action-
        conditioned harm attribution.
        Hypothesis: JEPA invariance suppresses the very signal needed for
        attribution (agent-caused vs env-caused harm), creating a structural
        blind spot.

    (B) HARM_SENSITIVE -- event-contrastive supervised encoder:
        Encoder trained with cross-entropy auxiliary loss on harm event type
        (SD-009 approach). The encoder must represent features that distinguish
        harm event types, including the action-conditioned attribution signal.
        Hypothesis: event-contrastive supervision preserves attribution-relevant
        distinctions that JEPA invariance would suppress.

  The discriminative question:
    Can a linear attribution probe decode harm SOURCE (agent-caused vs env-caused
    vs safe) from z_world?
    - This is the REE-relevant operationalisation: E3 harm evaluation depends on
      z_world carrying harm-source discriminative information for the counterfactual
      attribution pipeline (SD-003).
    - Measured by: AUC of a post-hoc linear probe trained on a held-out set.
    - A high AUC means z_world contains harm-source information.
    - A low AUC means z_world has collapsed away the attribution-relevant signal.

  The adjudication was "hybridize" (2026-02-25), meaning evidence is needed to
  determine whether JEPA invariances actually suppress ethically relevant distinctions
  or whether they are preserved incidentally. This experiment directly tests that.

  Evidence interpretation:
    PASS: JEPA_INVARIANT attribution_auc substantially lower than HARM_SENSITIVE.
      => Q-014 resolved: JEPA invariances do hide ethically relevant distinctions.
         Hybridize recommendation supported: REE cannot rely on pure JEPA encoders
         for attribution -- event-contrastive supervision is required.
    PARTIAL_BOTH_HIGH: Both conditions achieve high attribution_auc (>= THRESH_ATTR_MIN).
      => JEPA invariance does NOT hide attribution signal at this scale/environment.
         Blind-spot risk not confirmed. REE can use JEPA encoders provisionally.
    PARTIAL_BOTH_LOW: Both conditions fail to achieve attribution_auc >= THRESH_ATTR_MIN.
      => Neither encoder preserves attribution signal. Environment too simple or
         harm events too infrequent; experiment inconclusive.
    FAIL: Insufficient harm events or probe data for AUC computation.
      => Implementation or configuration problem; not informative for Q-014.

Pre-registered thresholds
--------------------------
C1: HARM_SENSITIVE achieves attribution_auc >= THRESH_ATTR_MIN = 0.65 (both seeds).
    (Event-contrastive encoder preserves attribution-relevant signal.)

C2: JEPA_INVARIANT achieves attribution_auc >= THRESH_ATTR_MIN = 0.65 (both seeds).
    (JEPA encoder ALSO preserves attribution signal -- blind spot NOT confirmed.)

C3: HARM_SENSITIVE attribution_auc exceeds JEPA_INVARIANT by at least
    THRESH_ATTR_ADVANTAGE = 0.08 (both seeds).
    (Event-contrastive supervision provides meaningfully better attribution probing.)

C4: n_harm_events_agent >= THRESH_MIN_AGENT_HARM = 20 AND
    n_harm_events_env   >= THRESH_MIN_ENV_HARM   = 20  per condition per seed.
    (Data quality gate: sufficient examples of BOTH harm types for probe training.)

C5: Seed consistency: direction of harm_sens_auc vs jepa_auc is consistent across seeds.
    (Both seeds agree on which encoder has higher attribution AUC, or within 0.01.)

PASS: C1 + C3 + C4 + C5
  => Q-014 resolved: JEPA invariances hide ethically relevant distinctions.
     Hybridize recommendation supported. REE requires event-contrastive supervision
     for attribution, not pure JEPA invariance.

PARTIAL (BOTH_HIGH): C2 + C4 + C5 + NOT C3
  => JEPA encoder preserves attribution signal at this scale. Blind-spot risk not
     demonstrated. May emerge at larger scale, deeper architectures, or with more
     complex harm event structures.

PARTIAL (BOTH_LOW): NOT C1 and NOT C2
  => Neither encoder preserves attribution signal. Environment or harm configuration
     insufficient; experiment inconclusive.

FAIL: NOT C4 (too few harm events) or degenerate probe (both AUCs ~ 0.5 with no
  data variance).
  => Not informative for Q-014.

Conditions
----------
JEPA_INVARIANT:
  Encoder: Linear(world_obs_dim, world_dim) + LayerNorm (online encoder).
  Target encoder: copy of online encoder with stop-gradient (EMA-updated, tau=0.99).
  Predictor: 2-layer MLP(world_dim, 64, world_dim).
  Prediction loss: MSE(predictor(z_online), z_target.detach()).
    Only the predictor and online encoder gradient flows from prediction loss.
    Target encoder is NOT trained by prediction loss.
  VICReg variance term: max(0, gamma - std(z_online)) to prevent collapse.
    gamma = 1.0, coeff = 1.0. This prevents zero-collapse but does NOT restore
    attribution signal -- it only prevents the most degenerate failure mode.
  No event classification loss.
  Probe: post-hoc linear probe (LogisticRegression equivalent, SGD, 200 steps)
    trained on held-out (z_world, harm_source_label) pairs.

HARM_SENSITIVE:
  Encoder: Linear(world_obs_dim, world_dim) + LayerNorm (same architecture).
  No target encoder, no predictor.
  Event classification loss: CrossEntropy(linear_head(z_world), event_type_label).
    Labels: 0=none, 1=env_caused_harm, 2=agent_caused_harm.
    weight = EVENT_CLASS_WEIGHT = 0.5 relative to world prediction MSE loss.
  World prediction loss: MSE(linear_forward(z_world, action), z_world_next.detach()).
    (Simplified E1 prediction -- same as JEPA target but both losses flow through encoder.)
  Probe: same as JEPA_INVARIANT.

Both conditions:
  Harm attribution label: derived at step time.
    - Agent takes action A. If harm_signal > 0 at next step:
      - Check if agent moved INTO hazard cell (agent-caused) or hazard is adjacent
        regardless of action (env-caused). Implementation: compare agent action direction
        vs hazard_contact_dir in obs_dict. Approximation: if agent action was TOWARD
        the hazard cell that caused contact, label = agent_caused; else env_caused.
    - Simple heuristic: actions 0-3 are directional (N/E/S/W). If the harm contact
      cell is in the direction of the action taken, label = 2 (agent_caused).
      Otherwise label = 1 (env_caused). Action 4 = stay -> always env_caused.
  Probe training on 80% of collected (z_world, harm_source_label) samples.
  Probe evaluation on held-out 20%.

Seeds:      [42, 123] (matched -- same env seed per condition)
Env:        CausalGridWorldV2 size=8, 5 hazards, 0 resources, hazard_harm=0.05,
            env_drift_interval=3, env_drift_prob=0.4
            (higher hazard density + frequent drift for more harm events of both types)
Protocol:   TRAIN_EPISODES=300 (train encoder before probe collection)
            PROBE_COLLECT_EPISODES=100 (collect probe data with frozen encoder)
            STEPS_PER_EPISODE=200
Estimated runtime:
  ~2 conditions x 2 seeds x 400 eps x 0.10 min/ep = ~160 min Mac
  (+20% event classification overhead) => ~192 min Mac
  (~5x slower on Daniel-PC => ~960 min; assign to Mac)
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


EXPERIMENT_TYPE = "v3_exq_154_q014_jepa_invariance_blind_spot_pair"
CLAIM_IDS = ["Q-014"]

# ---------------------------------------------------------------------------
# Pre-registered thresholds
# ---------------------------------------------------------------------------
THRESH_ATTR_MIN       = 0.65   # C1/C2: minimum attribution probe AUC
THRESH_ATTR_ADVANTAGE = 0.08   # C3: HARM_SENSITIVE must exceed JEPA_INVARIANT by this
THRESH_MIN_AGENT_HARM = 20     # C4: minimum agent-caused harm events for valid probe
THRESH_MIN_ENV_HARM   = 20     # C4: minimum env-caused harm events for valid probe

# ---------------------------------------------------------------------------
# Protocol constants
# ---------------------------------------------------------------------------
TRAIN_EPISODES          = 300   # encoder training episodes
PROBE_COLLECT_EPISODES  = 100   # probe data collection episodes (encoder frozen)
STEPS_PER_EPISODE       = 200
LR                      = 3e-4
EMA_TAU                 = 0.99  # EMA update rate for JEPA target encoder
VICREG_GAMMA            = 1.0   # VICReg variance target
VICREG_COEFF            = 1.0   # VICReg variance coefficient
EVENT_CLASS_WEIGHT      = 0.5   # event cross-entropy weight relative to prediction loss
PROBE_TRAIN_STEPS       = 200   # SGD steps for linear probe
PROBE_LR                = 1e-3
PROBE_TRAIN_FRAC        = 0.8   # fraction of probe data for training

SEEDS      = [42, 123]
CONDITIONS = ["JEPA_INVARIANT", "HARM_SENSITIVE"]

# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------
WORLD_OBS_DIM = 250   # CausalGridWorldV2 size=8 (actual flattened obs dim)
ACTION_DIM    = 5
WORLD_DIM     = 32
HIDDEN_DIM    = 64

# Harm source labels for classification
LABEL_SAFE        = 0
LABEL_ENV_HARM    = 1
LABEL_AGENT_HARM  = 2
N_HARM_CLASSES    = 3


# ---------------------------------------------------------------------------
# AUC computation (no sklearn dependency)
# ---------------------------------------------------------------------------

def _compute_auc(scores: List[float], labels: List[int]) -> float:
    """
    Compute binary AUC via trapezoidal method.
    labels: 1 = positive, 0 = negative.
    Returns AUC in [0, 1]. Returns 0.5 if no positive or no negative examples.
    """
    if len(scores) < 2:
        return 0.5
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

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
        auc += (fpr - fpr_prev) * (tpr + tpr_prev) / 2.0
        tpr_prev = tpr
        fpr_prev = fpr
    return float(auc)


def _multiclass_probe_auc(z_vecs: List[torch.Tensor], labels: List[int]) -> Tuple[float, int, int]:
    """
    Train a linear multinomial probe and evaluate one-vs-rest AUC for harm (classes 1+2)
    vs safe (class 0). Returns (mean_auc_harm_vs_safe, n_agent_harm, n_env_harm).

    Probe: SGD-trained linear classifier (LogReg equivalent, CrossEntropy loss).
    Binary AUC: label 1 = any harm (class 1 or 2), label 0 = safe (class 0).
    This tests: does z_world discriminate harm from safe, regardless of attribution?
    A secondary attribution AUC: agent_harm (class 2) vs env_harm (class 1) only.
    Primary metric used for criterion evaluation: attribution_auc = agent_harm vs env_harm.
    """
    if len(z_vecs) < 10:
        return 0.5, 0, 0

    n_agent = sum(1 for l in labels if l == LABEL_AGENT_HARM)
    n_env   = sum(1 for l in labels if l == LABEL_ENV_HARM)

    # Split train/eval
    n_train = max(1, int(len(z_vecs) * PROBE_TRAIN_FRAC))
    z_train  = torch.stack(z_vecs[:n_train])
    y_train  = torch.tensor(labels[:n_train], dtype=torch.long)
    z_eval   = torch.stack(z_vecs[n_train:])
    y_eval   = labels[n_train:]

    if len(z_eval) < 4:
        return 0.5, n_agent, n_env

    # Train linear probe
    probe = nn.Linear(WORLD_DIM, N_HARM_CLASSES)
    opt   = optim.SGD(probe.parameters(), lr=PROBE_LR)

    for _ in range(PROBE_TRAIN_STEPS):
        opt.zero_grad()
        logits = probe(z_train)
        loss   = F.cross_entropy(logits, y_train)
        loss.backward()
        opt.step()

    # Evaluate: attribution AUC = agent_harm (2) vs env_harm (1) only
    # Filter eval set to harm-only pairs
    harm_only_z = []
    harm_only_labels = []  # 1 = agent_caused, 0 = env_caused
    for i, label in enumerate(y_eval):
        if label == LABEL_AGENT_HARM:
            harm_only_z.append(z_eval[i])
            harm_only_labels.append(1)
        elif label == LABEL_ENV_HARM:
            harm_only_z.append(z_eval[i])
            harm_only_labels.append(0)

    if len(harm_only_z) < 4:
        # Not enough attribution examples -- return 0.5 (chance)
        return 0.5, n_agent, n_env

    with torch.no_grad():
        probe.eval()
        harm_z_t   = torch.stack(harm_only_z)
        logits_harm = probe(harm_z_t)
        # P(agent_caused) = softmax logit for class 2 - logit for class 1
        agent_logit = logits_harm[:, LABEL_AGENT_HARM].tolist()

    attribution_auc = _compute_auc(agent_logit, harm_only_labels)
    return attribution_auc, n_agent, n_env


# ---------------------------------------------------------------------------
# Encoder architectures
# ---------------------------------------------------------------------------

class WorldEncoder(nn.Module):
    """
    Shared encoder backbone: Linear(world_obs_dim, world_dim) + LayerNorm.
    Used by both conditions (same architecture, different training).
    """
    def __init__(self, obs_dim: int, world_dim: int):
        super().__init__()
        self.fc   = nn.Linear(obs_dim, world_dim)
        self.norm = nn.LayerNorm(world_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.norm(self.fc(obs))


class JEPAPredictor(nn.Module):
    """
    JEPA predictor: 2-layer MLP mapping z_online -> z_target prediction.
    Gradient flows through this to train the online encoder.
    """
    def __init__(self, world_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(world_dim + ACTION_DIM, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, world_dim),
        )

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([z.view(1, -1), action.view(1, -1)], dim=-1)
        return self.net(inp).squeeze(0)


class EventClassHead(nn.Module):
    """
    Linear classifier head for event type: safe / env_harm / agent_harm.
    Used by HARM_SENSITIVE condition only.
    """
    def __init__(self, world_dim: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(world_dim, n_classes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc(z.view(1, -1)).squeeze(0)


class WorldPredictor(nn.Module):
    """
    Simplified E1-style world predictor: (z_world, action) -> z_world_next.
    Used by HARM_SENSITIVE condition for supervised prediction.
    """
    def __init__(self, world_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(world_dim + action_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, world_dim),
        )

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([z.view(1, -1), action.view(1, -1)], dim=-1)
        return self.net(inp).squeeze(0)


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed,
        size=8,
        num_hazards=5,       # high hazard density for more harm events
        num_resources=0,
        hazard_harm=0.05,
        env_drift_interval=3,
        env_drift_prob=0.4,
    )


# ---------------------------------------------------------------------------
# Harm source inference from obs_dict and action
# ---------------------------------------------------------------------------

def _infer_harm_source(action_idx: int, obs_dict: Dict) -> int:
    """
    Infer whether harm contact was agent-caused or env-caused.

    Action directions: 0=up, 1=right, 2=down, 3=left, 4=stay.
    Heuristic: if harm_contact_direction in obs_dict is defined and aligns with
    the action taken, the agent moved into the hazard -> agent_caused.
    If action=stay or direction mismatch -> env_caused.

    Fallback: if harm_contact_direction is unavailable, use action alone:
    action=stay (4) -> env_caused; any directional action -> agent_caused
    (conservative: over-labels agent-caused but maintains separation).

    Returns: LABEL_ENV_HARM or LABEL_AGENT_HARM.
    """
    # Attempt to use harm_contact_dir from obs_dict if available
    harm_dir = obs_dict.get("harm_contact_dir", None)

    if action_idx == 4:
        # Stay action: agent did not move into hazard
        return LABEL_ENV_HARM

    if harm_dir is not None:
        # If harm_contact_dir matches action direction -> agent-caused
        # harm_contact_dir: 0=up,1=right,2=down,3=left (same encoding as actions 0-3)
        try:
            harm_dir_int = int(harm_dir)
            if harm_dir_int == action_idx:
                return LABEL_AGENT_HARM
            else:
                return LABEL_ENV_HARM
        except (TypeError, ValueError):
            pass

    # Fallback: any directional action while harm present -> agent-caused
    return LABEL_AGENT_HARM


# ---------------------------------------------------------------------------
# JEPA EMA update
# ---------------------------------------------------------------------------

def _ema_update(online: nn.Module, target: nn.Module, tau: float) -> None:
    """Update target encoder via exponential moving average of online encoder."""
    with torch.no_grad():
        for p_online, p_target in zip(online.parameters(), target.parameters()):
            p_target.data.mul_(tau).add_(p_online.data * (1.0 - tau))


# ---------------------------------------------------------------------------
# VICReg variance regularizer
# ---------------------------------------------------------------------------

def _vicreg_var_loss(z: torch.Tensor, gamma: float) -> torch.Tensor:
    """
    VICReg variance regularization: penalise dimensions with std < gamma.
    z: [1, world_dim] -- single sample; we accumulate over a batch tensor.
    For single-sample: penalise mean log-var across all dims.
    Note: with a single sample per step we cannot compute true batch std.
    Use squared norm as a proxy: penalise collapse to zero.
    """
    # Prevent collapse to zero by penalising small L2 norm
    norm = z.norm(dim=-1).mean()
    collapse_penalty = F.relu(gamma - norm)
    return collapse_penalty


# ---------------------------------------------------------------------------
# Run one condition x seed
# ---------------------------------------------------------------------------

def _run_condition(
    seed: int,
    condition: str,
    train_episodes: int,
    probe_collect_episodes: int,
    steps_per_episode: int,
    lr: float,
    dry_run: bool,
) -> Dict:
    """
    Train encoder under condition then collect probe data and evaluate.

    Returns dict with attribution_auc, n_agent_harm, n_env_harm, per-episode metrics.
    """
    if dry_run:
        train_episodes         = 4
        probe_collect_episodes = 4
        steps_per_episode      = 20

    torch.manual_seed(seed)
    random.seed(seed)

    env = _make_env(seed)

    # Build models based on condition
    online_encoder = WorldEncoder(WORLD_OBS_DIM, WORLD_DIM)

    if condition == "JEPA_INVARIANT":
        target_encoder = WorldEncoder(WORLD_OBS_DIM, WORLD_DIM)
        # Initialise target = online
        target_encoder.load_state_dict(online_encoder.state_dict())
        jepa_predictor = JEPAPredictor(WORLD_DIM, HIDDEN_DIM)
        event_class_head = None
        world_predictor  = None
        optimizer = optim.Adam(
            list(online_encoder.parameters()) + list(jepa_predictor.parameters()),
            lr=lr,
        )
    else:  # HARM_SENSITIVE
        target_encoder   = None
        jepa_predictor   = None
        event_class_head = EventClassHead(WORLD_DIM, N_HARM_CLASSES)
        world_predictor  = WorldPredictor(WORLD_DIM, ACTION_DIM)
        optimizer = optim.Adam(
            list(online_encoder.parameters())
            + list(event_class_head.parameters())
            + list(world_predictor.parameters()),
            lr=lr,
        )

    print(
        f"\n--- [{condition}] seed={seed}"
        f" train_eps={train_episodes} probe_eps={probe_collect_episodes}"
        f" steps={steps_per_episode} ---",
        flush=True,
    )

    # ======= TRAINING PHASE ============================================
    print(f"  [{condition}] seed={seed} TRAINING phase ...", flush=True)

    total_train_loss = 0.0
    total_train_steps = 0

    _, obs_dict = env.reset()
    prev_action_idx = 4  # stay (default start)

    for ep in range(train_episodes):
        for _step in range(steps_per_episode):
            obs_world = torch.tensor(obs_dict["world_state"], dtype=torch.float32)

            # Determine event label for this obs
            harm_signal = float(obs_dict.get("harm_signal", 0.0))
            if harm_signal > 0.0:
                event_label = _infer_harm_source(prev_action_idx, obs_dict)
            else:
                event_label = LABEL_SAFE

            # Random action
            action_idx = random.randint(0, ACTION_DIM - 1)
            action = torch.zeros(ACTION_DIM)
            action[action_idx] = 1.0

            # Step env
            _, _, done, _, obs_dict_next = env.step(action.unsqueeze(0))

            # Compute loss based on condition
            if condition == "JEPA_INVARIANT":
                z_online = online_encoder(obs_world)
                obs_world_next = torch.tensor(
                    obs_dict_next["world_state"], dtype=torch.float32
                )
                with torch.no_grad():
                    z_target = target_encoder(obs_world_next)  # stop-gradient target

                z_pred = jepa_predictor(z_online, action)
                pred_loss = F.mse_loss(z_pred, z_target)

                # VICReg collapse prevention (single-sample proxy)
                var_loss = _vicreg_var_loss(z_online.unsqueeze(0), VICREG_GAMMA)

                total_loss = pred_loss + VICREG_COEFF * var_loss

                optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(
                    list(online_encoder.parameters()) + list(jepa_predictor.parameters()),
                    1.0,
                )
                optimizer.step()

                # EMA update target encoder
                _ema_update(online_encoder, target_encoder, EMA_TAU)

            else:  # HARM_SENSITIVE
                z_online = online_encoder(obs_world)

                obs_world_next = torch.tensor(
                    obs_dict_next["world_state"], dtype=torch.float32
                )
                with torch.no_grad():
                    z_next_target = online_encoder(obs_world_next)

                z_pred    = world_predictor(z_online, action)
                pred_loss = F.mse_loss(z_pred, z_next_target.detach())

                event_logits = event_class_head(z_online)
                event_label_t = torch.tensor([event_label], dtype=torch.long)
                class_loss    = F.cross_entropy(event_logits.unsqueeze(0), event_label_t)

                total_loss = pred_loss + EVENT_CLASS_WEIGHT * class_loss

                optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(
                    list(online_encoder.parameters())
                    + list(event_class_head.parameters())
                    + list(world_predictor.parameters()),
                    1.0,
                )
                optimizer.step()

            total_train_loss  += float(total_loss.item())
            total_train_steps += 1

            prev_action_idx = action_idx
            if done:
                _, obs_dict = env.reset()
                prev_action_idx = 4
            else:
                obs_dict = obs_dict_next

        if ep % 100 == 0:
            avg_loss = total_train_loss / max(total_train_steps, 1)
            print(
                f"  [{condition}] seed={seed} train ep={ep}/{train_episodes}"
                f" avg_loss={avg_loss:.5f}",
                flush=True,
            )

    mean_train_loss = total_train_loss / max(total_train_steps, 1)
    print(
        f"  [{condition}] seed={seed} TRAINING done mean_loss={mean_train_loss:.5f}",
        flush=True,
    )

    # ======= PROBE COLLECTION PHASE ====================================
    print(f"  [{condition}] seed={seed} PROBE COLLECTION phase ...", flush=True)

    online_encoder.eval()

    probe_z:      List[torch.Tensor] = []
    probe_labels: List[int]          = []
    n_agent_harm = 0
    n_env_harm   = 0

    _, obs_dict = env.reset()
    prev_action_idx = 4

    for ep in range(probe_collect_episodes):
        for _step in range(steps_per_episode):
            obs_world = torch.tensor(obs_dict["world_state"], dtype=torch.float32)

            with torch.no_grad():
                z_world = online_encoder(obs_world)

            harm_signal = float(obs_dict.get("harm_signal", 0.0))
            if harm_signal > 0.0:
                src = _infer_harm_source(prev_action_idx, obs_dict)
                if src == LABEL_AGENT_HARM:
                    n_agent_harm += 1
                else:
                    n_env_harm += 1
                probe_z.append(z_world.detach().clone())
                probe_labels.append(src)
            else:
                # Include safe states occasionally to balance probe
                if random.random() < 0.05:
                    probe_z.append(z_world.detach().clone())
                    probe_labels.append(LABEL_SAFE)

            action_idx = random.randint(0, ACTION_DIM - 1)
            action = torch.zeros(ACTION_DIM)
            action[action_idx] = 1.0
            _, _, done, _, obs_dict = env.step(action.unsqueeze(0))

            prev_action_idx = action_idx
            if done:
                _, obs_dict = env.reset()
                prev_action_idx = 4

    print(
        f"  [{condition}] seed={seed} PROBE data: n_agent_harm={n_agent_harm}"
        f" n_env_harm={n_env_harm} total_probe={len(probe_z)}",
        flush=True,
    )

    # ======= ATTRIBUTION PROBE =========================================
    attribution_auc, _, _ = _multiclass_probe_auc(probe_z, probe_labels)

    print(
        f"  [{condition}] seed={seed} attribution_auc={attribution_auc:.4f}",
        flush=True,
    )

    return {
        "condition": condition,
        "seed": seed,
        "attribution_auc": attribution_auc,
        "n_agent_harm": n_agent_harm,
        "n_env_harm": n_env_harm,
        "n_probe_total": len(probe_z),
        "mean_train_loss": mean_train_loss,
    }


# ---------------------------------------------------------------------------
# Criterion evaluation
# ---------------------------------------------------------------------------

def _evaluate_criteria(
    results_by_condition: Dict[str, List[Dict]],
) -> Dict[str, bool]:
    """Evaluate pre-registered criteria across both conditions and both seeds."""

    jepa  = results_by_condition["JEPA_INVARIANT"]
    harm  = results_by_condition["HARM_SENSITIVE"]

    # C1: HARM_SENSITIVE attribution_auc >= THRESH_ATTR_MIN (both seeds)
    c1 = all(harm[i]["attribution_auc"] >= THRESH_ATTR_MIN for i in range(len(harm)))

    # C2: JEPA_INVARIANT attribution_auc >= THRESH_ATTR_MIN (both seeds)
    c2 = all(jepa[i]["attribution_auc"] >= THRESH_ATTR_MIN for i in range(len(jepa)))

    # C3: HARM_SENSITIVE exceeds JEPA_INVARIANT by THRESH_ATTR_ADVANTAGE (both seeds)
    c3 = all(
        harm[i]["attribution_auc"] - jepa[i]["attribution_auc"] >= THRESH_ATTR_ADVANTAGE
        for i in range(len(harm))
    )

    # C4: n_agent_harm and n_env_harm >= thresholds for all conditions and seeds
    c4 = all(
        r["n_agent_harm"] >= THRESH_MIN_AGENT_HARM
        and r["n_env_harm"] >= THRESH_MIN_ENV_HARM
        for cond_results in results_by_condition.values()
        for r in cond_results
    )

    # C5: seed consistency -- direction consistent across seeds (or within 0.01)
    c5_direction = [
        (harm[i]["attribution_auc"] - jepa[i]["attribution_auc"]) > -0.01
        for i in range(len(harm))
    ]
    c5 = all(c5_direction) or not any(c5_direction)

    return {
        "C1_harm_sensitive_calibrated": c1,
        "C2_jepa_invariant_calibrated": c2,
        "C3_harm_sensitive_advantage":  c3,
        "C4_sufficient_harm_events":    c4,
        "C5_seed_consistent":           c5,
    }


def _determine_outcome(criteria: Dict[str, bool]) -> str:
    c1 = criteria["C1_harm_sensitive_calibrated"]
    c2 = criteria["C2_jepa_invariant_calibrated"]
    c3 = criteria["C3_harm_sensitive_advantage"]
    c4 = criteria["C4_sufficient_harm_events"]
    c5 = criteria["C5_seed_consistent"]

    # FAIL: insufficient harm events
    if not c4:
        return "FAIL"

    # PASS: HARM_SENSITIVE substantially better than JEPA_INVARIANT
    if c1 and c3 and c4 and c5:
        return "PASS"

    # PARTIAL (BOTH_HIGH): both calibrated, no meaningful gap
    if c2 and c4 and c5 and not c3:
        return "PARTIAL_BOTH_HIGH"

    # PARTIAL (BOTH_LOW): neither achieves attribution calibration
    if not c1 and not c2:
        return "PARTIAL_BOTH_LOW"

    return "PARTIAL"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_experiment(dry_run: bool = False) -> Dict:
    """Run all conditions and compile the result pack."""
    print("=== V3-EXQ-154: Q-014 JEPA Invariance Blind Spot Pair ===", flush=True)
    print(f"Conditions: {CONDITIONS}  Seeds: {SEEDS}", flush=True)
    print("Pre-registered thresholds:", flush=True)
    print(f"  C1/C2 THRESH_ATTR_MIN       = {THRESH_ATTR_MIN}", flush=True)
    print(f"  C3    THRESH_ATTR_ADVANTAGE  = {THRESH_ATTR_ADVANTAGE}", flush=True)
    print(f"  C4    THRESH_MIN_AGENT_HARM  = {THRESH_MIN_AGENT_HARM}", flush=True)
    print(f"        THRESH_MIN_ENV_HARM    = {THRESH_MIN_ENV_HARM}", flush=True)
    print(f"  TRAIN_EPISODES={TRAIN_EPISODES}  PROBE_COLLECT_EPISODES={PROBE_COLLECT_EPISODES}", flush=True)

    results_by_condition: Dict[str, List[Dict]] = {c: [] for c in CONDITIONS}

    for condition in CONDITIONS:
        print(f"\n=== Condition: {condition} ===", flush=True)
        for seed in SEEDS:
            result = _run_condition(
                seed=seed,
                condition=condition,
                train_episodes=TRAIN_EPISODES,
                probe_collect_episodes=PROBE_COLLECT_EPISODES,
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
        summary_metrics[f"{prefix}_attribution_auc"]  = _mean_seeds(cond, "attribution_auc")
        summary_metrics[f"{prefix}_n_agent_harm"]     = _mean_seeds(cond, "n_agent_harm")
        summary_metrics[f"{prefix}_n_env_harm"]       = _mean_seeds(cond, "n_env_harm")
        summary_metrics[f"{prefix}_mean_train_loss"]  = _mean_seeds(cond, "mean_train_loss")

    # Pairwise delta (harm_sensitive advantage over jepa_invariant)
    summary_metrics["delta_auc_harm_sens_vs_jepa"] = (
        summary_metrics["harm_sensitive_attribution_auc"]
        - summary_metrics["jepa_invariant_attribution_auc"]
    )

    # Evidence direction
    if outcome == "PASS":
        evidence_direction = "supports"
        guidance = "jepa_invariance_hides_attribution_signal_hybridize_confirmed"
    elif outcome == "PARTIAL_BOTH_HIGH":
        evidence_direction = "mixed"
        guidance = "jepa_encoder_preserves_attribution_at_this_scale_blind_spot_not_confirmed"
    elif outcome == "PARTIAL_BOTH_LOW":
        evidence_direction = "mixed"
        guidance = "neither_encoder_achieves_attribution_calibration_inconclusive"
    elif outcome == "PARTIAL":
        evidence_direction = "mixed"
        guidance = "partial_evidence_see_criteria"
    else:  # FAIL
        evidence_direction = "mixed"
        guidance = "insufficient_harm_events_implementation_problem"

    run_id = (
        "v3_exq_154_q014_jepa_invariance_blind_spot_"
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
            "THRESH_ATTR_MIN":       THRESH_ATTR_MIN,
            "THRESH_ATTR_ADVANTAGE": THRESH_ATTR_ADVANTAGE,
            "THRESH_MIN_AGENT_HARM": THRESH_MIN_AGENT_HARM,
            "THRESH_MIN_ENV_HARM":   THRESH_MIN_ENV_HARM,
        },
        "summary_metrics": summary_metrics,
        "protocol": {
            "train_episodes":         TRAIN_EPISODES,
            "probe_collect_episodes": PROBE_COLLECT_EPISODES,
            "steps_per_episode":      STEPS_PER_EPISODE,
            "ema_tau":                EMA_TAU,
            "vicreg_gamma":           VICREG_GAMMA,
            "vicreg_coeff":           VICREG_COEFF,
            "event_class_weight":     EVENT_CLASS_WEIGHT,
            "probe_train_steps":      PROBE_TRAIN_STEPS,
            "probe_train_frac":       PROBE_TRAIN_FRAC,
        },
        "seeds": SEEDS,
        "scenario": (
            "Two-condition JEPA invariance blind-spot test:"
            " JEPA_INVARIANT (stop-gradient target encoder, predictor-only gradient flow,"
            " VICReg collapse prevention, no event supervision),"
            " HARM_SENSITIVE (event cross-entropy auxiliary loss + world prediction loss,"
            " full encoder gradient)."
            " Both conditions: random action policy, 300 train episodes, 100 probe episodes."
            " Attribution probe: post-hoc linear classifier trained on (z_world, harm_source)"
            " pairs; AUC for agent-caused vs env-caused harm separation."
            " 2 seeds x 2 conditions = 4 cells."
            " CausalGridWorldV2 size=8 5 hazards 0 resources hazard_harm=0.05"
            " env_drift_interval=3 env_drift_prob=0.4 (high hazard density + drift)."
        ),
        "interpretation": (
            "PASS => Q-014 resolved: JEPA invariance suppresses attribution signal."
            " attribution_auc gap >= 0.08 confirms that invariance training collapses"
            " agent-caused vs env-caused harm distinctions in z_world."
            " REE cannot use pure JEPA encoders for attribution;"
            " hybridize recommendation supported -- event-contrastive supervision required."
            " PARTIAL_BOTH_HIGH => JEPA encoder preserves attribution at this scale;"
            " blind-spot risk not confirmed. May emerge at larger scale or with richer"
            " action-conditional harm structures."
            " PARTIAL_BOTH_LOW => neither encoder achieves attribution calibration;"
            " environment too simple or harm events too infrequent; inconclusive."
            " FAIL => insufficient harm events; configuration or implementation problem."
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
