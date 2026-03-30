#!/opt/local/bin/python3
"""
V3-EXQ-085j -- SD-015 ResourceEncoder with Spatial Invariance

Claims: SD-015, SD-012, MECH-112

Supersedes: V3-EXQ-085i

=== ROOT CAUSE OF 085i C2 FAILURE ===

085i C3 PASSED (goal_resource_r_rfm=0.218 > 0.2): SD-015 representation hypothesis
confirmed. BUT C2 FAILED (benefit_ratio=1.03x). Per-seed breakdown:
  seed=42: goal_present=0.640, absent=0.790  (ratio=0.81x -- HARMFUL)
  seed=7:  goal_present=0.800, absent=0.560  (ratio=1.43x -- passes!)
  seed=13: goal_present=0.510, absent=0.550  (ratio=0.93x -- harmful)

Seeds 42 and 13 are actively hurt by goal guidance. Root cause:
  z_goal_resource = EMA of raw resource_field_view at contacts.
  Contact at cell (2,1) -> [0,0,...,1.0,...,0] (25-dim one-hot-ish)
  Contact at cell (7,8) -> [0,...,0,1.0,0,...] (completely different pattern)
  After resource respawn, z_goal still encodes the OLD contact pattern.
  cosine_sim(rf_current, z_goal) is near zero when resource is at a new location.
  => goal proximity is not tracking resource presence, just old position.

Seeds 42 and 13 experienced resource respawns to positions dissimilar to z_goal,
so goal proximity pointed the agent AWAY from resources. Seed 7 got lucky (respawns
close to the original contact position).

=== THE FIX: ResourceEncoder with Contact Prediction ===

ResourceEncoder (25 -> 16) trained to predict `is_contact`:
  - Positive: rf at contact events (ttype=='resource')
  - Negative: rf at non-contact steps
  - Training: binary cross-entropy on contact_head(encoder(rf))

Spatial invariance emerges: contact at (2,1), (7,8), (5,3), etc. all have
is_contact=1. The encoder must learn features that generalize across positions.
The natural solution: encode "proximity magnitude" (max/sum of field values) not
"proximity at specific cell". This gives a compact z_resource that is high at ALL
resource contact positions, not just the training position.

E2Resource (16+5 -> 64 -> 64 -> 16): forward model in ENCODED z_resource space.
  z_resource_next = E2Resource(z_resource_curr, action)
  Action selection: pick action maximising cosine_sim(E2Resource(z_r, a), z_goal).

z_goal seeded from enc.encode(rf) at contacts -- not raw rf.
goal_state_resource dim: ENCODER_DIM=16 (not 25).

=== PASS CRITERIA ===

C1: z_goal_norm_enc > 0.1         (encoder-based goal is non-trivially active)
C2: benefit_ratio >= 1.3x         (goal-guided nav beats random by 30%)
C3: goal_resource_r_enc > 0.2     (encoded goal proximity tracks resource proximity)
C4: enc_contact_acc > 0.7         (encoder has learned to predict contact)

Scientific interpretation:
  All PASS:    SD-015 full architecture confirmed; proceed to LatentStack integration
  C1+C3 PASS, C2 FAIL:  Goal representation works; navigation gap (action selection?)
  C1+C4 PASS, C3 FAIL:  Encoder classifies but goal not tracking resource nav
  C4 FAIL:    Encoder not learning; check buffer sizes, lr, warmup length
"""

import sys
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.goal import GoalState, GoalConfig
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_085j_sd015_resource_encoder"
CLAIM_IDS = ["SD-015", "SD-012", "MECH-112"]

RESOURCE_OBS_DIM = 25  # resource_field_view: 5x5 proximity grid
ENCODER_DIM = 16       # z_resource embedding dimension


# ------------------------------------------------------------------ #
# ResourceEncoder                                                      #
# ------------------------------------------------------------------ #

class ResourceEncoder(nn.Module):
    """
    Learns z_resource = f(resource_field_view) via contact prediction BCE.

    Spatial invariance: contact at any grid position maps to a similar
    z_resource because all contacts share is_contact=1 label.

    Trained with a contact_head (16 -> 1 sigmoid) using binary CE.
    Only the encoder() layers are used at goal-seeding time.
    """

    def __init__(self, resource_obs_dim: int = 25, encoder_dim: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(resource_obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, encoder_dim),
        )
        self.contact_head = nn.Linear(encoder_dim, 1)

    def encode(self, rf: torch.Tensor) -> torch.Tensor:
        """rf: [B, 25] -> z_resource: [B, ENCODER_DIM]"""
        return self.encoder(rf)

    def predict_contact(self, rf: torch.Tensor) -> torch.Tensor:
        """rf: [B, 25] -> contact_prob: [B, 1]"""
        return torch.sigmoid(self.contact_head(self.encoder(rf)))


# ------------------------------------------------------------------ #
# E2Resource (forward model in encoded z_resource space)              #
# ------------------------------------------------------------------ #

class E2Resource(nn.Module):
    """
    Predicts z_resource_next from (z_resource_curr, action).
    Operates in ENCODER_DIM space, not raw resource_field_view.

    Analog to E2.world_forward but for the resource proximity stream.
    Used for lookahead action selection: pick action maximising
    cosine_sim(E2Resource(z_resource, action), z_goal_resource).
    """

    def __init__(self, encoder_dim: int = 16, action_dim: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(encoder_dim + action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, encoder_dim),
        )

    def forward(self, z_resource: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """z_resource: [1, ENCODER_DIM], action: [1, action_dim] -> [1, ENCODER_DIM]"""
        return self.net(torch.cat([z_resource, action], dim=-1))


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


def _ensure_2d(t: torch.Tensor) -> torch.Tensor:
    """Ensure tensor is [1, dim] (add batch dim if needed)."""
    if t.dim() == 1:
        return t.unsqueeze(0)
    return t


def _place_resource_near_agent(env, max_dist: int = 3) -> bool:
    """Place a resource within max_dist Manhattan steps of the agent (curriculum)."""
    ax, ay = env.agent_x, env.agent_y
    candidates = []
    for dx in range(-max_dist, max_dist + 1):
        for dy in range(-max_dist, max_dist + 1):
            if dx == 0 and dy == 0:
                continue
            if abs(dx) + abs(dy) > max_dist:
                continue
            nx, ny = ax + dx, ay + dy
            if (0 < nx < env.size - 1 and 0 < ny < env.size - 1
                    and env.grid[nx, ny] == env.ENTITY_TYPES["empty"]):
                candidates.append((abs(dx) + abs(dy), nx, ny))
    if not candidates:
        return False
    candidates.sort()
    _, rx, ry = candidates[0]
    env.grid[rx, ry] = env.ENTITY_TYPES["resource"]
    env.resources.insert(0, [rx, ry])
    if env.use_proxy_fields:
        env._compute_proximity_fields()
    return True


def _e2r_guided_action(
    enc: ResourceEncoder,
    e2r: E2Resource,
    goal_state_enc: GoalState,
    rf_curr: torch.Tensor,
    num_actions: int,
    device,
) -> int:
    """
    Pick action maximising cosine_sim(E2Resource(z_resource, action), z_goal_resource).

    Uses encoded z_resource (16-dim) and E2Resource forward model.
    Falls back to random if goal is inactive.
    """
    if not goal_state_enc.is_active():
        return random.randint(0, num_actions - 1)
    with torch.no_grad():
        z_resource_curr = enc.encode(rf_curr)
        best_action = 0
        best_prox = -1.0
        for idx in range(num_actions):
            a = _action_to_onehot(idx, num_actions, device)
            z_resource_next = e2r(z_resource_curr, a)
            prox = goal_state_enc.goal_proximity(z_resource_next).mean().item()
            if prox > best_prox:
                best_prox = prox
                best_action = idx
    return best_action


def _resource_proximity(env) -> float:
    """1 / (1 + manhattan_dist_to_nearest_resource). 0.0 if no resources."""
    if not env.resources:
        return 0.0
    ax, ay = env.agent_x, env.agent_y
    min_dist = min(abs(ax - int(r[0])) + abs(ay - int(r[1])) for r in env.resources)
    return 1.0 / (1.0 + min_dist)


def _pearson_r(xs: List[float], ys: List[float]) -> float:
    """Pearson r from two equal-length float lists. Returns 0.0 if degenerate."""
    n = len(xs)
    if n < 4:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = (sum((x - mx) ** 2 for x in xs)) ** 0.5
    sy = (sum((y - my) ** 2 for y in ys)) ** 0.5
    if sx < 1e-9 or sy < 1e-9:
        return 0.0
    return num / (sx * sy)


def _contact_accuracy(
    enc: ResourceEncoder,
    pos_buf: List[torch.Tensor],
    neg_buf: List[torch.Tensor],
    k: int = 64,
) -> float:
    """Balanced accuracy of contact_head on pos/neg sample."""
    if len(pos_buf) < 4 or len(neg_buf) < 4:
        return 0.0
    k_pos = min(k, len(pos_buf))
    k_neg = min(k, len(neg_buf))
    with torch.no_grad():
        rf_pos = torch.cat(random.sample(pos_buf, k_pos), dim=0)
        rf_neg = torch.cat(random.sample(neg_buf, k_neg), dim=0)
        pred_pos = enc.predict_contact(rf_pos).squeeze(1)
        pred_neg = enc.predict_contact(rf_neg).squeeze(1)
        tp = float((pred_pos >= 0.5).float().mean().item())
        tn = float((pred_neg < 0.5).float().mean().item())
    return (tp + tn) / 2.0


# ------------------------------------------------------------------ #
# Main run function                                                     #
# ------------------------------------------------------------------ #

def _run_single(
    seed: int,
    goal_present: bool,
    warmup_episodes: int,
    eval_episodes: int,
    steps_per_episode: int,
    curriculum_episodes: int,
    self_dim: int,
    world_dim: int,
    lr: float,
    lr_enc: float,
    lr_e2r: float,
    alpha_world: float,
    alpha_self: float,
    harm_scale: float,
    proximity_harm_scale: float,
    novelty_bonus_weight: float,
    drive_weight: float,
) -> Dict:
    torch.manual_seed(seed)
    random.seed(seed)

    cond_label = "GOAL_PRESENT" if goal_present else "GOAL_ABSENT"

    env = CausalGridWorldV2(
        seed=seed,
        size=10,
        num_hazards=3,
        num_resources=4,
        hazard_harm=harm_scale,
        env_drift_interval=5,
        env_drift_prob=0.1,
        proximity_harm_scale=proximity_harm_scale,
        proximity_benefit_scale=proximity_harm_scale * 0.6,
        proximity_approach_threshold=0.15,
        hazard_field_decay=0.5,
        energy_decay=0.005,
        use_proxy_fields=True,
        resource_respawn_on_consume=True,
    )

    action_dim = env.action_dim

    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=action_dim,
        self_dim=self_dim,
        world_dim=world_dim,
        alpha_world=alpha_world,
        alpha_self=alpha_self,
        reafference_action_dim=0,
        novelty_bonus_weight=novelty_bonus_weight,
    )

    # z_world-seeded goal state (comparison baseline from 085g/085i)
    goal_config_world = GoalConfig(
        goal_dim=world_dim,
        alpha_goal=0.3,
        decay_goal=0.003,
        benefit_threshold=0.05,
        goal_weight=1.0,
        drive_weight=drive_weight,
        e1_goal_conditioned=True,
        z_goal_enabled=goal_present,
    )
    goal_state_world = GoalState(goal_config_world, device=torch.device("cpu"))

    # ResourceEncoder-seeded goal state (SD-015 -- contact-only, encoded)
    goal_config_enc = GoalConfig(
        goal_dim=ENCODER_DIM,  # 16, not 25
        alpha_goal=0.5,
        decay_goal=0.003,
        benefit_threshold=0.05,
        goal_weight=1.0,
        drive_weight=drive_weight,
        e1_goal_conditioned=False,
        z_goal_enabled=goal_present,
    )
    goal_state_enc = GoalState(goal_config_enc, device=torch.device("cpu"))

    agent = REEAgent(config)
    enc = ResourceEncoder(resource_obs_dim=RESOURCE_OBS_DIM, encoder_dim=ENCODER_DIM)
    e2r = E2Resource(encoder_dim=ENCODER_DIM, action_dim=action_dim)

    harm_buf_pos: List[torch.Tensor] = []
    harm_buf_neg: List[torch.Tensor] = []
    contact_buf_pos: List[torch.Tensor] = []  # rf at contact events
    contact_buf_neg: List[torch.Tensor] = []  # rf at non-contact steps
    MAX_BUF = 2000

    standard_params = [p for n, p in agent.named_parameters()
                       if "harm_eval_head" not in n]
    harm_eval_params = list(agent.e3.harm_eval_head.parameters())
    optimizer       = optim.Adam(standard_params, lr=lr)
    harm_eval_opt   = optim.Adam(harm_eval_params, lr=1e-4)
    enc_optimizer   = optim.Adam(enc.parameters(), lr=lr_enc)
    e2r_optimizer   = optim.Adam(e2r.parameters(), lr=lr_e2r)

    counts: Dict[str, int] = {
        "hazard_approach": 0, "env_caused_hazard": 0,
        "agent_caused_hazard": 0, "none": 0, "resource": 0,
        "benefit_approach": 0,
    }
    n_seedings_world = 0
    n_seedings_enc   = 0
    enc_train_losses: List[float] = []
    e2r_train_losses: List[float] = []

    # --- WARMUP TRAINING ---
    agent.train()
    enc.train()
    e2r.train()

    prev_z_resource: Optional[torch.Tensor] = None
    prev_action_oh:  Optional[torch.Tensor] = None

    for ep in range(warmup_episodes):
        _, obs_dict = env.reset()
        agent.reset()
        prev_z_resource = None
        prev_action_oh  = None

        if ep < curriculum_episodes:
            _place_resource_near_agent(env, max_dist=3)
            obs_dict = env._get_observation_dict()

        for _ in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            obs_harm  = obs_dict.get("harm_obs", None)
            rf_curr   = _ensure_2d(obs_dict["resource_field_view"].float())

            latent = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
            agent.clock.advance()

            z_world_curr = latent.z_world.detach()

            action_idx = random.randint(0, action_dim - 1)
            action_oh  = _action_to_onehot(action_idx, action_dim, agent.device)
            agent._last_action = action_oh

            _, harm_signal, done, info, obs_dict = env.step(action_oh)
            ttype = info.get("transition_type", "none")
            if ttype in counts:
                counts[ttype] += 1

            rf_next = _ensure_2d(obs_dict["resource_field_view"].float())

            # --- Encoder contact prediction training ---
            is_contact = (ttype == "resource")
            if is_contact:
                contact_buf_pos.append(rf_curr.detach())
                if len(contact_buf_pos) > MAX_BUF:
                    contact_buf_pos = contact_buf_pos[-MAX_BUF:]
            else:
                # Subsample to prevent buf_neg from drowning buf_pos
                if random.random() < 0.1:
                    contact_buf_neg.append(rf_curr.detach())
                    if len(contact_buf_neg) > MAX_BUF:
                        contact_buf_neg = contact_buf_neg[-MAX_BUF:]

            if len(contact_buf_pos) >= 8 and len(contact_buf_neg) >= 8:
                k_pos = min(16, len(contact_buf_pos))
                k_neg = min(16, len(contact_buf_neg))
                rf_pos = torch.cat(random.sample(contact_buf_pos, k_pos), dim=0)
                rf_neg = torch.cat(random.sample(contact_buf_neg, k_neg), dim=0)
                rf_b   = torch.cat([rf_pos, rf_neg], dim=0)
                lbl    = torch.cat([
                    torch.ones(k_pos, 1),
                    torch.zeros(k_neg, 1),
                ], dim=0)
                pred_c = enc.predict_contact(rf_b)
                enc_loss = F.binary_cross_entropy(pred_c, lbl)
                enc_optimizer.zero_grad()
                enc_loss.backward()
                torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
                enc_optimizer.step()
                enc_train_losses.append(enc_loss.item())

            # --- E2Resource forward model training ---
            # Use detached encoder targets to avoid interfering with enc's BCE training.
            with torch.no_grad():
                z_resource_curr_target = enc.encode(rf_curr.detach())

            if prev_z_resource is not None and prev_action_oh is not None:
                z_resource_pred = e2r(prev_z_resource.detach(),
                                      prev_action_oh.detach())
                e2r_loss = F.mse_loss(z_resource_pred,
                                      z_resource_curr_target.detach())
                e2r_optimizer.zero_grad()
                e2r_loss.backward()
                torch.nn.utils.clip_grad_norm_(e2r.parameters(), 1.0)
                e2r_optimizer.step()
                e2r_train_losses.append(e2r_loss.item())

            prev_z_resource = z_resource_curr_target.detach()
            prev_action_oh  = action_oh.detach()

            # --- Goal seeding (GOAL_PRESENT only, contact-only for z_goal_enc) ---
            if goal_present:
                if ttype == "resource":
                    # Primary: contact-gated seeding for both goal states
                    goal_state_world.update(z_world_curr, benefit_exposure=1.0,
                                            drive_level=1.0)
                    with torch.no_grad():
                        z_resource_contact = enc.encode(rf_curr.detach())
                    goal_state_enc.update(z_resource_contact, benefit_exposure=1.0,
                                          drive_level=1.0)
                    n_seedings_world += 1
                    n_seedings_enc   += 1
                elif obs_body.shape[-1] > 11:
                    # Secondary: proximity-based seeding for z_goal_world only.
                    # z_goal_enc: contact-only (no secondary seeding).
                    benefit_exposure = float(
                        obs_body[0, 11].item() if obs_body.dim() == 2
                        else obs_body[11].item()
                    )
                    energy = float(
                        obs_body[0, 3].item() if obs_body.dim() == 2
                        else obs_body[3].item()
                    )
                    drive_level = max(0.0, 1.0 - energy)
                    goal_state_world.update(z_world_curr, benefit_exposure,
                                            drive_level=drive_level)

            # --- Harm eval training ---
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
                zw_pos  = torch.cat([harm_buf_pos[i] for i in pos_idx], dim=0)
                zw_neg  = torch.cat([harm_buf_neg[i] for i in neg_idx], dim=0)
                zw_b    = torch.cat([zw_pos, zw_neg], dim=0)
                target  = torch.cat([
                    torch.ones(k_pos,  1, device=agent.device),
                    torch.zeros(k_neg, 1, device=agent.device),
                ], dim=0)
                pred_harm = agent.e3.harm_eval(zw_b)
                harm_loss = F.mse_loss(pred_harm, target)
                if harm_loss.requires_grad:
                    harm_eval_opt.zero_grad()
                    harm_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        agent.e3.harm_eval_head.parameters(), 0.5,
                    )
                    harm_eval_opt.step()

            if done:
                break

        if (ep + 1) % 100 == 0 or ep == warmup_episodes - 1:
            enc_str = ""
            if enc_train_losses:
                recent_enc = enc_train_losses[-100:]
                enc_avg    = sum(recent_enc) / max(1, len(recent_enc))
                enc_str    = f" enc_loss={enc_avg:.4f}"
            e2r_str = ""
            if e2r_train_losses:
                recent_e2r = e2r_train_losses[-100:]
                e2r_avg    = sum(recent_e2r) / max(1, len(recent_e2r))
                e2r_str    = f" e2r_loss={e2r_avg:.4f}"
            enc_acc = _contact_accuracy(enc, contact_buf_pos, contact_buf_neg)
            goal_norm_world = goal_state_world.goal_norm()
            goal_norm_enc   = goal_state_enc.goal_norm() if goal_present else 0.0
            curriculum_tag  = " [curriculum]" if ep < curriculum_episodes else ""
            print(
                f"  [train] seed={seed} cond={cond_label}"
                f" ep {ep+1}/{warmup_episodes}"
                f" resource={counts['resource']}"
                f" seedings_enc={n_seedings_enc}"
                f" z_goal_norm_enc={goal_norm_enc:.3f}"
                f" enc_acc={enc_acc:.3f}"
                f"{enc_str}{e2r_str}{curriculum_tag}",
                flush=True,
            )

    z_goal_norm_world_end = goal_state_world.goal_norm() if goal_present else 0.0
    z_goal_norm_enc_end   = goal_state_enc.goal_norm()   if goal_present else 0.0
    enc_final_loss = (
        float(sum(enc_train_losses[-500:]) / max(1, min(500, len(enc_train_losses))))
        if enc_train_losses else 0.0
    )
    e2r_final_loss = (
        float(sum(e2r_train_losses[-500:]) / max(1, min(500, len(e2r_train_losses))))
        if e2r_train_losses else 0.0
    )
    enc_final_acc = _contact_accuracy(enc, contact_buf_pos, contact_buf_neg)

    # --- EVAL ---
    agent.eval()
    enc.eval()
    e2r.eval()

    benefit_per_ep: List[float] = []
    harm_buf_eval_pos: List[torch.Tensor] = []
    harm_buf_eval_neg: List[torch.Tensor] = []

    goal_prox_world_vals: List[float] = []
    goal_prox_enc_vals:   List[float] = []
    resource_prox_vals:   List[float] = []
    harm_eval_vals:       List[float] = []
    goal_contrib_enc_vals: List[float] = []

    for _ in range(eval_episodes):
        _, obs_dict = env.reset()
        agent.reset()
        ep_benefit = 0.0

        for _ in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            obs_harm  = obs_dict.get("harm_obs", None)
            rf_curr   = _ensure_2d(obs_dict["resource_field_view"].float())

            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
                agent.clock.advance()
                z_world_curr = latent.z_world.detach()

            # E2Resource-guided action selection in GOAL_PRESENT; random in GOAL_ABSENT
            if goal_present and goal_state_enc.is_active():
                action_idx = _e2r_guided_action(
                    enc, e2r, goal_state_enc, rf_curr, action_dim, agent.device,
                )
            else:
                action_idx = random.randint(0, action_dim - 1)

            action_oh = _action_to_onehot(action_idx, action_dim, agent.device)
            agent._last_action = action_oh

            _, harm_signal, done, info, obs_dict = env.step(action_oh)
            ttype = info.get("transition_type", "none")

            if ttype == "resource":
                ep_benefit += 1.0
            if ttype == "benefit_approach" and obs_body.dim() == 2 and obs_body.shape[-1] > 11:
                ep_benefit += float(obs_body[0, 11].item()) * 0.1

            z_world_ev = z_world_curr
            if float(harm_signal) < 0:
                harm_buf_eval_pos.append(z_world_ev)
            else:
                harm_buf_eval_neg.append(z_world_ev)

            # Diagnostic metrics (GOAL_PRESENT only)
            if goal_present:
                with torch.no_grad():
                    gp_world = (
                        goal_state_world.goal_proximity(z_world_ev).mean().item()
                        if goal_state_world.is_active() else 0.0
                    )
                    z_resource_curr = enc.encode(rf_curr)
                    gp_enc = (
                        goal_state_enc.goal_proximity(z_resource_curr).mean().item()
                        if goal_state_enc.is_active() else 0.0
                    )
                    he = agent.e3.harm_eval(z_world_ev).mean().item()
                rp = _resource_proximity(env)
                goal_prox_world_vals.append(gp_world)
                goal_prox_enc_vals.append(gp_enc)
                resource_prox_vals.append(rp)
                harm_eval_vals.append(he)
                goal_contrib_enc_vals.append(goal_config_enc.goal_weight * gp_enc)

            if done:
                break

        benefit_per_ep.append(ep_benefit)

    avg_benefit = float(sum(benefit_per_ep) / max(1, len(benefit_per_ep)))

    # Harm calibration check
    calibration_gap = 0.0
    if len(harm_buf_eval_pos) >= 4 and len(harm_buf_eval_neg) >= 4:
        k = min(64, min(len(harm_buf_eval_pos), len(harm_buf_eval_neg)))
        zw_pos = torch.cat(harm_buf_eval_pos[-k:], dim=0)
        zw_neg = torch.cat(harm_buf_eval_neg[-k:], dim=0)
        with torch.no_grad():
            harm_pos = agent.e3.harm_eval(zw_pos).mean().item()
            harm_neg = agent.e3.harm_eval(zw_neg).mean().item()
        calibration_gap = harm_pos - harm_neg

    # Diagnostic correlations
    goal_resource_r_world = (
        _pearson_r(goal_prox_world_vals, resource_prox_vals)
        if goal_present else 0.0
    )
    goal_resource_r_enc = (
        _pearson_r(goal_prox_enc_vals, resource_prox_vals)
        if goal_present else 0.0
    )

    # MECH-124: goal_vs_harm_ratio
    mean_goal_contrib_enc = (
        float(sum(goal_contrib_enc_vals) / max(1, len(goal_contrib_enc_vals)))
        if goal_present else 0.0
    )
    mean_harm_eval = (
        float(sum(harm_eval_vals) / max(1, len(harm_eval_vals)))
        if goal_present else 0.0
    )
    goal_vs_harm_ratio = (
        mean_goal_contrib_enc / max(1e-6, mean_harm_eval)
        if goal_present else 0.0
    )

    print(
        f"  [eval] seed={seed} cond={cond_label}"
        f" avg_benefit/ep={avg_benefit:.3f}"
        f" z_goal_norm_enc={z_goal_norm_enc_end:.3f}"
        f" enc_loss={enc_final_loss:.4f}"
        f" e2r_loss={e2r_final_loss:.4f}"
        f" enc_acc={enc_final_acc:.3f}"
        f" cal_gap={calibration_gap:.4f}"
        f" goal_resource_r_enc={goal_resource_r_enc:.3f}"
        f" goal_vs_harm={goal_vs_harm_ratio:.3f}",
        flush=True,
    )

    return {
        "seed": seed,
        "condition": cond_label,
        "goal_present": goal_present,
        "avg_benefit_per_ep": float(avg_benefit),
        "z_goal_norm_world_end": float(z_goal_norm_world_end),
        "z_goal_norm_enc_end":   float(z_goal_norm_enc_end),
        "n_seedings_world": int(n_seedings_world),
        "n_seedings_enc":   int(n_seedings_enc),
        "calibration_gap":  float(calibration_gap),
        "goal_resource_r_world": float(goal_resource_r_world),
        "goal_resource_r_enc":   float(goal_resource_r_enc),
        "goal_vs_harm_ratio":    float(goal_vs_harm_ratio),
        "enc_final_loss":  float(enc_final_loss),
        "e2r_final_loss":  float(e2r_final_loss),
        "enc_final_acc":   float(enc_final_acc),
        "train_resource_events": int(counts["resource"]),
    }


def run(
    seeds: Tuple = (42, 7, 13),
    warmup_episodes: int = 800,
    eval_episodes: int = 100,
    steps_per_episode: int = 200,
    curriculum_episodes: int = 100,
    self_dim: int = 32,
    world_dim: int = 32,
    lr: float = 1e-3,
    lr_enc: float = 1e-3,
    lr_e2r: float = 5e-4,
    alpha_world: float = 0.9,
    alpha_self: float = 0.3,
    harm_scale: float = 0.02,
    proximity_harm_scale: float = 0.05,
    novelty_bonus_weight: float = 0.1,
    drive_weight: float = 2.0,
    **kwargs,
) -> dict:
    """GOAL_PRESENT (E2Resource-guided eval) vs GOAL_ABSENT (random eval)."""
    results_goal:    List[Dict] = []
    results_no_goal: List[Dict] = []

    for seed in seeds:
        for goal_present in [True, False]:
            label = "GOAL_PRESENT" if goal_present else "GOAL_ABSENT"
            print(
                f"\n[V3-EXQ-085j] {label} seed={seed}"
                f" warmup={warmup_episodes} eval={eval_episodes}"
                f" curriculum={curriculum_episodes}"
                f" drive_weight={drive_weight} alpha_world={alpha_world}"
                f" lr_enc={lr_enc} lr_e2r={lr_e2r}",
                flush=True,
            )
            r = _run_single(
                seed=seed,
                goal_present=goal_present,
                warmup_episodes=warmup_episodes,
                eval_episodes=eval_episodes,
                steps_per_episode=steps_per_episode,
                curriculum_episodes=curriculum_episodes,
                self_dim=self_dim,
                world_dim=world_dim,
                lr=lr,
                lr_enc=lr_enc,
                lr_e2r=lr_e2r,
                alpha_world=alpha_world,
                alpha_self=alpha_self,
                harm_scale=harm_scale,
                proximity_harm_scale=proximity_harm_scale,
                novelty_bonus_weight=novelty_bonus_weight,
                drive_weight=drive_weight,
            )
            if goal_present:
                results_goal.append(r)
            else:
                results_no_goal.append(r)

    def _avg(results: List[Dict], key: str) -> float:
        vals = [r[key] for r in results]
        return float(sum(vals) / max(1, len(vals)))

    z_goal_norm_world_avg  = _avg(results_goal, "z_goal_norm_world_end")
    z_goal_norm_enc_avg    = _avg(results_goal, "z_goal_norm_enc_end")
    benefit_goal_present   = _avg(results_goal,    "avg_benefit_per_ep")
    benefit_goal_absent    = _avg(results_no_goal, "avg_benefit_per_ep")
    cal_gap_goal_present   = _avg(results_goal, "calibration_gap")
    avg_goal_resource_r_w  = _avg(results_goal, "goal_resource_r_world")
    avg_goal_resource_r_e  = _avg(results_goal, "goal_resource_r_enc")
    avg_goal_vs_harm_ratio = _avg(results_goal, "goal_vs_harm_ratio")
    avg_enc_final_loss     = _avg(results_goal, "enc_final_loss")
    avg_e2r_final_loss     = _avg(results_goal, "e2r_final_loss")
    avg_enc_final_acc      = _avg(results_goal, "enc_final_acc")

    benefit_ratio = (
        benefit_goal_present / max(1e-6, benefit_goal_absent)
        if benefit_goal_absent > 1e-6 else 0.0
    )

    c1_pass = z_goal_norm_enc_avg > 0.1
    c2_pass = benefit_ratio >= 1.3
    c3_pass = avg_goal_resource_r_e > 0.2
    c4_pass = avg_enc_final_acc > 0.7

    all_pass = c1_pass and c2_pass and c3_pass and c4_pass
    criteria_met = sum([c1_pass, c2_pass, c3_pass, c4_pass])
    status = "PASS" if all_pass else "FAIL"

    if all_pass:
        decision = "retain_ree"
    elif c1_pass and c3_pass:
        decision = "hybridize"
    else:
        decision = "retire_ree_claim"

    mech124_flag_salience = avg_goal_vs_harm_ratio < 0.3

    print(f"\n[V3-EXQ-085j] Final results:", flush=True)
    print(
        f"  z_goal_norm_world={z_goal_norm_world_avg:.3f}"
        f"  z_goal_norm_enc={z_goal_norm_enc_avg:.3f}",
        flush=True,
    )
    print(
        f"  benefit_goal_present={benefit_goal_present:.3f}"
        f"  benefit_goal_absent={benefit_goal_absent:.3f}"
        f"  ratio={benefit_ratio:.2f}x",
        flush=True,
    )
    print(
        f"  goal_resource_r_world={avg_goal_resource_r_w:.3f}"
        f"  (comparison baseline)",
        flush=True,
    )
    print(
        f"  goal_resource_r_enc={avg_goal_resource_r_e:.3f}"
        f"  (SD-015 encoded: need > 0.2)",
        flush=True,
    )
    print(
        f"  enc_final_loss={avg_enc_final_loss:.4f}"
        f"  e2r_final_loss={avg_e2r_final_loss:.4f}"
        f"  enc_acc={avg_enc_final_acc:.3f}",
        flush=True,
    )
    print(
        f"  cal_gap={cal_gap_goal_present:.4f}"
        f"  goal_vs_harm={avg_goal_vs_harm_ratio:.3f}",
        flush=True,
    )
    print(
        f"  decision={decision}  status={status} ({criteria_met}/4)",
        flush=True,
    )

    failure_notes: List[str] = []
    if not c1_pass:
        failure_notes.append(
            f"C1 FAIL: z_goal_norm_enc={z_goal_norm_enc_avg:.3f} <= 0.1"
            " (encoder-based goal not active -- check n_seedings_enc > 0)"
        )
    if not c2_pass:
        if c3_pass:
            failure_notes.append(
                f"C2 FAIL: benefit_ratio={benefit_ratio:.2f}x < 1.3x BUT C3 PASS"
                f" (goal_resource_r_enc={avg_goal_resource_r_e:.3f} > 0.2)."
                " Goal representation works; navigation gap. Check action selection"
                " / nav_bias strength. Consider stronger lookahead or more eval eps."
            )
        else:
            failure_notes.append(
                f"C2 FAIL: benefit_ratio={benefit_ratio:.2f}x < 1.3x"
                f" (goal_present={benefit_goal_present:.3f}"
                f" vs goal_absent={benefit_goal_absent:.3f})"
            )
    if not c3_pass:
        failure_notes.append(
            f"C3 FAIL: goal_resource_r_enc={avg_goal_resource_r_e:.3f} < 0.2."
            f" Encoder acc={avg_enc_final_acc:.3f}."
            " If acc > 0.7: spatial invariance not generalising to goal tracking."
            " If acc < 0.7: encoder not converged (increase warmup / check buffers)."
        )
    if not c4_pass:
        failure_notes.append(
            f"C4 FAIL: enc_contact_acc={avg_enc_final_acc:.3f} <= 0.7."
            f" Encoder loss={avg_enc_final_loss:.4f}."
            " Contact buffer sizes: check n_seedings_enc and contact_buf_neg."
            " Try increasing warmup to 1000+ or lr_enc to 5e-3."
        )
    if mech124_flag_salience and c1_pass:
        failure_notes.append(
            f"MECH-124 V4 RISK: goal_vs_harm_ratio={avg_goal_vs_harm_ratio:.3f} < 0.3."
            " z_goal salience not competitive with harm salience."
        )

    for note in failure_notes:
        print(f"  {note}", flush=True)

    per_goal_rows = "\n".join(
        f"  seed={r['seed']}:"
        f" benefit/ep={r['avg_benefit_per_ep']:.3f}"
        f" z_goal_norm_enc={r['z_goal_norm_enc_end']:.3f}"
        f" r_enc={r['goal_resource_r_enc']:.3f}"
        f" r_world={r['goal_resource_r_world']:.3f}"
        f" enc_acc={r['enc_final_acc']:.3f}"
        f" enc_loss={r['enc_final_loss']:.4f}"
        f" e2r_loss={r['e2r_final_loss']:.4f}"
        for r in results_goal
    )
    per_nogoal_rows = "\n".join(
        f"  seed={r['seed']}: benefit/ep={r['avg_benefit_per_ep']:.3f}"
        for r in results_no_goal
    )

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(
            f"- {n}" for n in failure_notes
        )

    summary_markdown = (
        f"# V3-EXQ-085j -- SD-015 ResourceEncoder with Spatial Invariance\n\n"
        f"**Status:** {status}\n"
        f"**Claims:** SD-015, SD-012, MECH-112\n"
        f"**Decision:** {decision}\n"
        f"**Seeds:** {list(seeds)}\n"
        f"**Supersedes:** V3-EXQ-085i\n\n"
        f"## Architecture\n\n"
        f"ResourceEncoder (25->16) trained with contact prediction BCE for spatial"
        f" invariance. E2Resource (16+5->64->64->16) forward model in encoded space."
        f" z_goal seeded from enc.encode(rf) at contact events only."
        f" Action selection: E2Resource lookahead maximising cosine_sim to z_goal.\n\n"
        f"**SD-015 fix vs 085i:**"
        f" 085i used raw resource_field_view (25-dim) for z_goal -- spatially specific,"
        f" so resource respawn to new grid position broke goal tracking (2/3 seeds"
        f" goal harmful). ResourceEncoder learns position-invariant features via"
        f" contact prediction BCE.\n\n"
        f"**alpha_world:** {alpha_world}  (SD-008)\n"
        f"**SD-012:** drive_weight={drive_weight}, resource_respawn_on_consume=True\n"
        f"**lr_enc:** {lr_enc}  **lr_e2r:** {lr_e2r}\n"
        f"**enc_final_loss (avg):** {avg_enc_final_loss:.4f}\n"
        f"**e2r_final_loss (avg):** {avg_e2r_final_loss:.4f}\n"
        f"**enc_contact_acc (avg):** {avg_enc_final_acc:.3f}\n"
        f"**Warmup:** {warmup_episodes} eps"
        f" (curriculum={curriculum_episodes} eps)\n"
        f"**Eval:** {eval_episodes} eps"
        f" (GOAL_PRESENT: E2Resource-lookahead; GOAL_ABSENT: random)\n\n"
        f"## Key Diagnostic Comparison\n\n"
        f"| Representation | goal_resource_r | Note |\n"
        f"|---|---|---|\n"
        f"| z_world-seeded (085g) | {avg_goal_resource_r_w:.3f}"
        f" | baseline ~0.066 |\n"
        f"| enc-seeded (085i raw rf) | ~0.218 | 085i result |\n"
        f"| ResourceEncoder-seeded (085j) | {avg_goal_resource_r_e:.3f}"
        f" | need > 0.2 |\n\n"
        f"z_goal_norm_world: {z_goal_norm_world_avg:.3f}  "
        f"z_goal_norm_enc: {z_goal_norm_enc_avg:.3f}\n\n"
        f"## Navigation Results\n\n"
        f"| Condition | benefit/ep | z_goal_norm_enc | cal_gap | r_enc |\n"
        f"|---|---|---|---|---|\n"
        f"| GOAL_PRESENT | {benefit_goal_present:.3f} | {z_goal_norm_enc_avg:.3f}"
        f" | {cal_gap_goal_present:.4f} | {avg_goal_resource_r_e:.3f} |\n"
        f"| GOAL_ABSENT  | {benefit_goal_absent:.3f} | -- | -- | -- |\n\n"
        f"**Benefit ratio (goal/no-goal): {benefit_ratio:.2f}x**\n\n"
        f"## PASS Criteria\n\n"
        f"| Criterion | Result | Value |\n"
        f"|---|---|---|\n"
        f"| C1: z_goal_norm_enc > 0.1 | {'PASS' if c1_pass else 'FAIL'}"
        f" | {z_goal_norm_enc_avg:.3f} |\n"
        f"| C2: benefit ratio >= 1.3x | {'PASS' if c2_pass else 'FAIL'}"
        f" | {benefit_ratio:.2f}x |\n"
        f"| C3: goal_resource_r_enc > 0.2 | {'PASS' if c3_pass else 'FAIL'}"
        f" | {avg_goal_resource_r_e:.3f} |\n"
        f"| C4: enc_contact_acc > 0.7 | {'PASS' if c4_pass else 'FAIL'}"
        f" | {avg_enc_final_acc:.3f} |\n\n"
        f"Criteria met: {criteria_met}/4 -> **{status}**\n\n"
        f"## MECH-124 Diagnostics\n\n"
        f"goal_vs_harm_ratio: {avg_goal_vs_harm_ratio:.3f}"
        f" (< 0.3 = V4 risk)\n\n"
        f"## Per-Seed\n\n"
        f"GOAL_PRESENT:\n{per_goal_rows}\n\n"
        f"GOAL_ABSENT:\n{per_nogoal_rows}\n"
        f"{failure_section}\n"
    )

    metrics = {
        "z_goal_norm_world_avg":        float(z_goal_norm_world_avg),
        "z_goal_norm_enc_avg":          float(z_goal_norm_enc_avg),
        "benefit_per_ep_goal_present":  float(benefit_goal_present),
        "benefit_per_ep_goal_absent":   float(benefit_goal_absent),
        "benefit_ratio":                float(benefit_ratio),
        "calibration_gap_goal_present": float(cal_gap_goal_present),
        "goal_resource_r_world":        float(avg_goal_resource_r_w),
        "goal_resource_r_enc":          float(avg_goal_resource_r_e),
        "goal_vs_harm_ratio":           float(avg_goal_vs_harm_ratio),
        "enc_final_loss":               float(avg_enc_final_loss),
        "e2r_final_loss":               float(avg_e2r_final_loss),
        "enc_contact_acc":              float(avg_enc_final_acc),
        "mech124_flag_salience":        float(mech124_flag_salience),
        "drive_weight":                 float(drive_weight),
        "n_seeds":                      float(len(seeds)),
        "alpha_world":                  float(alpha_world),
        "lr_enc":                       float(lr_enc),
        "lr_e2r":                       float(lr_e2r),
        "curriculum_episodes":          float(curriculum_episodes),
        "crit1_pass": 1.0 if c1_pass else 0.0,
        "crit2_pass": 1.0 if c2_pass else 0.0,
        "crit3_pass": 1.0 if c3_pass else 0.0,
        "crit4_pass": 1.0 if c4_pass else 0.0,
        "criteria_met": float(criteria_met),
    }

    return {
        "status": status,
        "metrics": metrics,
        "summary_markdown": summary_markdown,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": (
            "supports" if all_pass
            else ("mixed" if criteria_met >= 2 else "weakens")
        ),
        "experiment_type": EXPERIMENT_TYPE,
        "fatal_error_count": 0,
    }


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",           type=int,   nargs="+", default=[42, 7, 13])
    parser.add_argument("--warmup",          type=int,   default=800)
    parser.add_argument("--eval-eps",        type=int,   default=100)
    parser.add_argument("--steps",           type=int,   default=200)
    parser.add_argument("--curriculum",      type=int,   default=100)
    parser.add_argument("--alpha-world",     type=float, default=0.9)
    parser.add_argument("--alpha-self",      type=float, default=0.3)
    parser.add_argument("--harm-scale",      type=float, default=0.02)
    parser.add_argument("--proximity-scale", type=float, default=0.05)
    parser.add_argument("--novelty-weight",  type=float, default=0.1)
    parser.add_argument("--drive-weight",    type=float, default=2.0)
    parser.add_argument("--lr-enc",          type=float, default=1e-3)
    parser.add_argument("--lr-e2r",          type=float, default=5e-4)
    parser.add_argument("--dry-run",         action="store_true",
                        help="Run 1 seed, 5 warmup eps, 5 eval eps for smoke test")
    args = parser.parse_args()

    if args.dry_run:
        args.seeds    = [42]
        args.warmup   = 5
        args.eval_eps = 5
        args.curriculum = 2
        print("[DRY-RUN] 1 seed, 5 warmup, 5 eval", flush=True)

    result = run(
        seeds=tuple(args.seeds),
        warmup_episodes=args.warmup,
        eval_episodes=args.eval_eps,
        steps_per_episode=args.steps,
        curriculum_episodes=args.curriculum,
        alpha_world=args.alpha_world,
        alpha_self=args.alpha_self,
        harm_scale=args.harm_scale,
        proximity_harm_scale=args.proximity_scale,
        novelty_bonus_weight=args.novelty_weight,
        drive_weight=args.drive_weight,
        lr_enc=args.lr_enc,
        lr_e2r=args.lr_e2r,
    )

    if args.dry_run:
        print("\n[DRY-RUN] Smoke test complete. Not writing output file.", flush=True)
        print(f"Status: {result['status']}", flush=True)
        for k, v in result["metrics"].items():
            print(f"  {k}: {v}", flush=True)
        sys.exit(0)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"] = ts
    result["claim"] = CLAIM_IDS[0]
    result["verdict"] = result["status"]
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
