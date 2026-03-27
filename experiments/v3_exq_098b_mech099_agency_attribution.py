#!/opt/local/bin/python3
"""
V3-EXQ-098b -- MECH-099 Three-Stream Agency Attribution Redesign

Claims: MECH-099, MECH-095

EXQ-098 FAIL diagnosis: both conditions were trained with BCE on harm_obs labels.
The lateral head learned a DUPLICATE harm signal, not a complementary agency signal.
With a smaller input (50 dims) than z_world (full world_obs_dim), it lost on AUC.
The adverse direction (auc_delta=-0.038) reflected information disadvantage, not
architectural failure.

Redesign rationale (MECH-099 biological basis):
  The lateral/third visual stream (MT -> MST/FST -> posterior STS -> TPJ) is
  specialised for DYNAMIC MOTION and AGENCY DETECTION, not harm detection per se.
  TPJ (MECH-095, active, EXQ-047k PASS) is the agency-detection comparator that
  distinguishes self-caused from other-caused change.

  Correct operationalisation: train the lateral head to predict CONTACT ATTRIBUTION
  (agent-caused vs env-caused hazard contact), not harm detection.

Discriminative pair:

  THREE_STREAM: standalone MotionLateralHead (motion-sensitive) + attribution_probe.
    Input: hazard_ch_t (25) + delta_hazard_ch (25) + contamination_view (25) = 75 dims.
    delta_hazard = hazard_ch_t - hazard_ch_{t-1} encodes hazard MOVEMENT (MT analog).
    contamination_view encodes the agent's prior footprint (agent_caused signal).
    Attribution probe: Linear(motion_dim -> 1), BCE on is_agent_caused on contact steps.
    Gradient: attribution_probe + MotionLateralHead ONLY. Agent encoder sees E1+E2 only.

  TWO_STREAM: no MotionLateralHead.
    Attribution probe: Linear(world_dim -> 1) on z_world.
    BCE on is_agent_caused. Gradient flows to z_world encoder (contaminates world rep).
    This is the two-stream constraint: all information, including attribution, must
    route through the shared world representation.

Both: harm_dim=0 (built-in stack.py lateral head disabled), alpha_world=0.9 (SD-008),
same env, same seeds.

Attribution labels (from CausalGridWorldV2 transition_type):
  "agent_caused_hazard" = agent walked into contaminated cell (agent's footprint) -> 1.0
  "env_caused_hazard"   = agent walked into an env-placed hazard cell             -> 0.0
  All other types: skip (no attribution gradient for those steps).

PASS criteria (ALL required):
  C1 (relative advantage): THREE_STREAM attribution_AUC >= TWO_STREAM attribution_AUC + 0.05
     AUC measured on hazard contact steps only (both agent_caused and env_caused).
  C2 (absolute learning): THREE_STREAM attribution_AUC >= 0.65
     The dedicated motion head must actually learn attribution, not just tie a bad baseline.
  C3 (behavioral): THREE_STREAM harm_avoidance_rate >= TWO_STREAM harm_avoidance_rate
     Attribution-guided flee (reverse action when predicted agent_caused) reduces contacts.

Flee policy: if attribution_sigmoid > FLEE_THRESHOLD -> reverse last action (predicted
agent-caused: I moved toward the hazard, so moving back helps). Else: random action.
"""

import math
import random
import sys
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


EXPERIMENT_TYPE = "v3_exq_098b_mech099_agency_attribution"
CLAIM_IDS = ["MECH-099", "MECH-095"]

# Hazard contact types that carry attribution labels
ATTRIBUTION_TYPES = {"agent_caused_hazard", "env_caused_hazard"}

# Flee threshold: if attribution_sigmoid > this, reverse last action
FLEE_THRESHOLD = 0.4

# Reverse-action map: CausalGridWorldV2.ACTIONS: {0:(-1,0), 1:(1,0), 2:(0,-1), 3:(0,1), 4:(0,0)}
REVERSE_ACTION = {0: 1, 1: 0, 2: 3, 3: 2, 4: 4}

# Hazard feature indices in world_obs (copied from ree_core/latent/stack.py SplitEncoder)
HAZARD_INDICES = list(range(3, 175, 7))   # [3, 10, ..., 171], length 25
CONT_SLICE = slice(175, 200)              # contamination_view, 25 dims


# ---------------------------------------------------------------------------
# MotionLateralHead: standalone motion-sensitive lateral head
# ---------------------------------------------------------------------------

class MotionLateralHead(nn.Module):
    """
    Motion-sensitive lateral head for agency attribution (MECH-099 redesign).

    Input: hazard_ch_t (25) + delta_hazard_ch (25) + contamination_view (25) = 75 dims.
      hazard_ch_t:       current hazard field at HAZARD_INDICES (position/intensity)
      delta_hazard_ch:   hazard_ch_t - hazard_ch_{t-1} (MOVEMENT of hazard objects)
      contamination_view: agent footprint field (encodes prior presence -> agent_caused)

    Biological analog: MT encodes motion, MST encodes optic flow + self-motion,
    FST/STS encode biological motion. The delta channel is the key MT-analog signal.
    contamination_view encodes the causal footprint of the agent -- needed to
    distinguish "I contaminated this cell" (agent_caused) from "hazard drifted here"
    (env_caused).

    Does NOT modify the REEAgent encoder. Gradient flows through this module only.
    """

    def __init__(self, out_dim: int = 16) -> None:
        super().__init__()
        # hazard_ch(25) + delta_hazard(25) + contamination_view(25) = 75 dims
        self.net = nn.Sequential(
            nn.Linear(75, 48),
            nn.ReLU(),
            nn.Linear(48, out_dim),
        )
        self.out_dim = out_dim

    def forward(
        self,
        world_obs: torch.Tensor,
        prev_hazard_ch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            world_obs:      [batch, world_obs_dim]
            prev_hazard_ch: [batch, 25] -- hazard features from previous step
        Returns:
            z_motion:           [batch, out_dim] -- motion-sensitive latent
            new_prev_hazard_ch: [batch, 25] detached -- for next step's delta
        """
        if world_obs.dim() == 1:
            world_obs = world_obs.unsqueeze(0)
        hazard_ch = world_obs[:, HAZARD_INDICES]          # [batch, 25]
        cont_view = world_obs[:, CONT_SLICE]              # [batch, 25]
        delta_hazard = hazard_ch - prev_hazard_ch         # [batch, 25] motion signal
        x = torch.cat([hazard_ch, delta_hazard, cont_view], dim=-1)  # [batch, 75]
        z_motion = self.net(x)
        return z_motion, hazard_ch.detach()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_2d(t: torch.Tensor) -> torch.Tensor:
    """Ensure tensor has a batch dimension (unsqueeze dim 0 if 1D)."""
    return t.unsqueeze(0) if t.dim() == 1 else t


def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


def _compute_auc(scores: List[float], labels: List[float]) -> float:
    """Rank-based (Wilcoxon-Mann-Whitney) AUC.
    Returns 0.5 if no positives or no negatives (degenerate case).
    """
    if not scores:
        return 0.5
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    pairs = sorted(zip(scores, labels), key=lambda x: x[0])
    auc = 0.0
    running_neg = 0.0
    for _, label in pairs:
        if label == 0.0:
            running_neg += 1.0
        else:
            auc += running_neg
    return auc / (n_pos * n_neg)


# ---------------------------------------------------------------------------
# Single cell runner
# ---------------------------------------------------------------------------

def _run_single(
    seed: int,
    three_stream: bool,
    warmup_episodes: int,
    eval_episodes: int,
    steps_per_episode: int,
    self_dim: int,
    world_dim: int,
    motion_dim: int,
    lr: float,
    alpha_world: float,
    alpha_self: float,
    harm_scale: float,
    proximity_harm_scale: float,
    env_drift_prob: float,
    env_drift_interval: int,
) -> Dict:
    """Run one (seed, condition) cell. Returns per-cell metrics dict."""
    torch.manual_seed(seed)
    random.seed(seed)

    cond = "THREE_STREAM" if three_stream else "TWO_STREAM"

    env = CausalGridWorldV2(
        seed=seed,
        size=12,
        num_hazards=4,
        hazard_harm=harm_scale,
        env_drift_interval=env_drift_interval,
        env_drift_prob=env_drift_prob,
        proximity_harm_scale=proximity_harm_scale,
        proximity_benefit_scale=proximity_harm_scale * 0.6,
        proximity_approach_threshold=0.15,
        hazard_field_decay=0.5,
    )

    # Both conditions: harm_dim=0 disables the built-in stack.py lateral head.
    # THREE_STREAM uses the standalone MotionLateralHead instead.
    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=self_dim,
        world_dim=world_dim,
        harm_dim=0,
        alpha_world=alpha_world,
        alpha_self=alpha_self,
    )
    config.latent.unified_latent_mode = False
    config.latent.use_harm_stream = False

    agent = REEAgent(config)
    device = agent.device

    # Module setup
    if three_stream:
        motion_head = MotionLateralHead(out_dim=motion_dim).to(device)
        attribution_probe = nn.Linear(motion_dim, 1).to(device)
        # Agent optimizer: E1+E2 only (no attribution gradient -> clean encoder)
        agent_opt = optim.Adam(agent.parameters(), lr=lr)
        # Attribution optimizer: motion_head + attribution_probe only
        attr_opt = optim.Adam(
            list(motion_head.parameters()) + list(attribution_probe.parameters()),
            lr=lr,
        )
    else:
        motion_head = None
        attribution_probe = nn.Linear(world_dim, 1).to(device)
        # Agent optimizer: E1+E2 (same lr)
        agent_opt = optim.Adam(agent.parameters(), lr=lr)
        # Attribution optimizer: agent params + probe (attribution contaminates z_world)
        # agent.parameters() appears in both opts; Adam handles shared params correctly.
        attr_opt = optim.Adam(
            list(agent.parameters()) + list(attribution_probe.parameters()),
            lr=lr,
        )

    # Training counters
    train_counts: Dict[str, int] = {
        "agent_caused": 0,
        "env_caused": 0,
        "total": 0,
    }

    # ---- TRAINING PHASE ----
    agent.train()
    attribution_probe.train()
    if motion_head is not None:
        motion_head.train()

    for ep in range(warmup_episodes):
        _, obs_dict = env.reset()
        agent.reset()
        prev_hazard_ch = torch.zeros(1, 25, device=device)

        for _ in range(steps_per_episode):
            obs_body = obs_dict["body_state"].to(device)
            obs_world = obs_dict["world_state"].to(device)

            latent = agent.sense(obs_body, obs_world)
            agent.clock.advance()

            # Compute motion latent (THREE_STREAM: standalone; updates prev_hazard_ch)
            if three_stream:
                z_motion, prev_hazard_ch = motion_head(obs_world, prev_hazard_ch)
            else:
                # Still update prev_hazard_ch for consistent step tracking
                hazard_ch = _ensure_2d(obs_world)[:, HAZARD_INDICES]
                prev_hazard_ch = hazard_ch.detach()

            action_idx = random.randint(0, env.action_dim - 1)
            action = _action_to_onehot(action_idx, env.action_dim, device)
            agent._last_action = action

            _, harm_signal, done, info, obs_dict = env.step(action)
            ttype = info.get("transition_type", "none")

            if ttype == "agent_caused_hazard":
                train_counts["agent_caused"] += 1
            elif ttype == "env_caused_hazard":
                train_counts["env_caused"] += 1
            train_counts["total"] += 1

            # E1 + E2 loss: backprop through agent only (both conditions)
            e1_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()
            total_agent_loss = e1_loss + e2_loss
            if total_agent_loss.requires_grad:
                agent_opt.zero_grad()
                total_agent_loss.backward()
                nn.utils.clip_grad_norm_(list(agent.parameters()), 1.0)
                agent_opt.step()

            # Attribution loss: SPARSE -- only on hazard contact steps
            if ttype in ATTRIBUTION_TYPES:
                attr_label_val = 1.0 if ttype == "agent_caused_hazard" else 0.0
                attr_label = torch.tensor([[attr_label_val]], device=device)
                if three_stream:
                    attr_pred = attribution_probe(z_motion)
                else:
                    attr_pred = attribution_probe(latent.z_world)
                attr_loss = F.binary_cross_entropy_with_logits(attr_pred, attr_label)
                attr_opt.zero_grad()
                attr_loss.backward()
                if three_stream:
                    nn.utils.clip_grad_norm_(
                        list(motion_head.parameters()) + list(attribution_probe.parameters()),
                        1.0,
                    )
                else:
                    nn.utils.clip_grad_norm_(
                        list(agent.parameters()) + list(attribution_probe.parameters()),
                        1.0,
                    )
                attr_opt.step()

            if done:
                break

        if (ep + 1) % 100 == 0 or ep == warmup_episodes - 1:
            print(
                f"  [train] seed={seed} cond={cond}"
                f" ep {ep + 1}/{warmup_episodes}"
                f" agent_caused={train_counts['agent_caused']}"
                f" env_caused={train_counts['env_caused']}"
                f" total={train_counts['total']}",
                flush=True,
            )

    # Warn if attribution signal was sparse
    n_min = min(train_counts["agent_caused"], train_counts["env_caused"])
    if n_min < 30:
        print(
            f"  [WARN] seed={seed} cond={cond}"
            f" sparse attribution events:"
            f" agent_caused={train_counts['agent_caused']}"
            f" env_caused={train_counts['env_caused']}"
            f" (min_class={n_min} < 30)"
            f" -- AUC may be unreliable",
            flush=True,
        )

    # ---- EVAL PHASE ----
    agent.eval()
    attribution_probe.eval()
    if motion_head is not None:
        motion_head.eval()

    attr_scores_contact: List[float] = []   # probe scores on hazard-contact steps only
    attr_labels_contact: List[float] = []   # 1.0=agent_caused, 0.0=env_caused
    harm_contacts = 0
    total_eval_steps = 0
    last_action_idx: Optional[int] = None

    for _ in range(eval_episodes):
        _, obs_dict = env.reset()
        agent.reset()
        prev_hazard_ch = torch.zeros(1, 25, device=device)
        last_action_idx = None

        for _ in range(steps_per_episode):
            obs_body = obs_dict["body_state"].to(device)
            obs_world = obs_dict["world_state"].to(device)

            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world)
                agent.clock.advance()

                if three_stream:
                    z_motion, prev_hazard_ch = motion_head(obs_world, prev_hazard_ch)
                    raw_score = attribution_probe(z_motion).item()
                else:
                    hazard_ch = _ensure_2d(obs_world)[:, HAZARD_INDICES]
                    prev_hazard_ch = hazard_ch.detach()
                    raw_score = attribution_probe(latent.z_world).item()

            # Attribution-guided flee policy:
            #   High score -> predicted agent_caused -> reverse last action
            #   Low score  -> predicted env_caused or none -> random action
            attr_sigmoid = 1.0 / (1.0 + math.exp(-raw_score))
            if attr_sigmoid > FLEE_THRESHOLD and last_action_idx is not None:
                action_idx = REVERSE_ACTION.get(
                    last_action_idx, random.randint(0, env.action_dim - 1)
                )
            else:
                action_idx = random.randint(0, env.action_dim - 1)

            action = _action_to_onehot(action_idx, env.action_dim, device)
            agent._last_action = action
            _, harm_signal, done, info, obs_dict = env.step(action)
            last_action_idx = action_idx

            ttype = info.get("transition_type", "none")
            total_eval_steps += 1

            # Attribution score collected ONLY on hazard contact steps
            if ttype in ATTRIBUTION_TYPES:
                lbl = 1.0 if ttype == "agent_caused_hazard" else 0.0
                attr_scores_contact.append(raw_score)
                attr_labels_contact.append(lbl)

            if float(harm_signal) < 0:
                harm_contacts += 1

            if done:
                break

    attribution_auc = _compute_auc(attr_scores_contact, attr_labels_contact)
    harm_avoidance_rate = 1.0 - harm_contacts / max(1, total_eval_steps)
    n_contact_eval = len(attr_scores_contact)
    n_agent_caused_eval = sum(1 for l in attr_labels_contact if l > 0.5)
    n_env_caused_eval = sum(1 for l in attr_labels_contact if l < 0.5)

    print(
        f"  [eval] seed={seed} cond={cond}"
        f" attr_auc={attribution_auc:.4f}"
        f" avoidance={harm_avoidance_rate:.4f}"
        f" harm_contacts={harm_contacts}/{total_eval_steps}"
        f" contact_steps={n_contact_eval}"
        f" (agent={n_agent_caused_eval} env={n_env_caused_eval})",
        flush=True,
    )

    return {
        "seed": seed,
        "condition": cond,
        "three_stream": three_stream,
        "attribution_auc": float(attribution_auc),
        "harm_avoidance_rate": float(harm_avoidance_rate),
        "harm_contacts": int(harm_contacts),
        "total_eval_steps": int(total_eval_steps),
        "n_contact_steps_eval": int(n_contact_eval),
        "n_agent_caused_eval": int(n_agent_caused_eval),
        "n_env_caused_eval": int(n_env_caused_eval),
        "n_agent_caused_train": int(train_counts["agent_caused"]),
        "n_env_caused_train": int(train_counts["env_caused"]),
        "train_total_steps": int(train_counts["total"]),
    }


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run(
    seeds: Tuple = (42, 7),
    warmup_episodes: int = 400,
    eval_episodes: int = 100,
    steps_per_episode: int = 200,
    self_dim: int = 32,
    world_dim: int = 32,
    motion_dim: int = 16,
    lr: float = 1e-3,
    alpha_world: float = 0.9,
    alpha_self: float = 0.3,
    harm_scale: float = 0.02,
    proximity_harm_scale: float = 0.01,
    env_drift_prob: float = 0.3,
    env_drift_interval: int = 3,
    **kwargs,
) -> dict:
    """Discriminative pair: THREE_STREAM (MotionLateralHead) vs TWO_STREAM (z_world)."""
    results_three: List[Dict] = []
    results_two: List[Dict] = []

    for seed in seeds:
        for three_stream in [True, False]:
            cond = "THREE_STREAM" if three_stream else "TWO_STREAM"
            print(
                f"\n[V3-EXQ-098b] {cond} seed={seed}"
                f" warmup={warmup_episodes} eval={eval_episodes}"
                f" alpha_world={alpha_world}"
                f" motion_dim={motion_dim if three_stream else 'N/A'}"
                f" drift_prob={env_drift_prob}",
                flush=True,
            )
            r = _run_single(
                seed=seed,
                three_stream=three_stream,
                warmup_episodes=warmup_episodes,
                eval_episodes=eval_episodes,
                steps_per_episode=steps_per_episode,
                self_dim=self_dim,
                world_dim=world_dim,
                motion_dim=motion_dim,
                lr=lr,
                alpha_world=alpha_world,
                alpha_self=alpha_self,
                harm_scale=harm_scale,
                proximity_harm_scale=proximity_harm_scale,
                env_drift_prob=env_drift_prob,
                env_drift_interval=env_drift_interval,
            )
            if three_stream:
                results_three.append(r)
            else:
                results_two.append(r)

    def _avg(results: List[Dict], key: str) -> float:
        vals = [r[key] for r in results]
        return float(sum(vals) / max(1, len(vals)))

    auc_three = _avg(results_three, "attribution_auc")
    auc_two   = _avg(results_two,   "attribution_auc")
    av_three  = _avg(results_three, "harm_avoidance_rate")
    av_two    = _avg(results_two,   "harm_avoidance_rate")

    auc_delta = auc_three - auc_two
    av_delta  = av_three - av_two

    # Pre-registered PASS criteria
    c1_pass = auc_delta  >= 0.05   # relative advantage: THREE beats TWO by 5pp
    c2_pass = auc_three  >= 0.65   # absolute: THREE actually learns attribution
    c3_pass = av_delta   >= 0.0    # behavioral: attribution-guided flee helps

    all_pass     = c1_pass and c2_pass and c3_pass
    criteria_met = sum([c1_pass, c2_pass, c3_pass])
    status       = "PASS" if all_pass else "FAIL"

    print(f"\n[V3-EXQ-098b] Final results:", flush=True)
    print(
        f"  attr_auc_THREE={auc_three:.4f}  attr_auc_TWO={auc_two:.4f}"
        f"  delta={auc_delta:+.4f}  (C1 thresh >=0.05)",
        flush=True,
    )
    print(
        f"  attr_auc_THREE={auc_three:.4f}  (C2 absolute thresh >=0.65)",
        flush=True,
    )
    print(
        f"  avoidance_THREE={av_three:.4f}  avoidance_TWO={av_two:.4f}"
        f"  delta={av_delta:+.4f}  (C3 thresh >=0)",
        flush=True,
    )
    print(f"  status={status} ({criteria_met}/3)", flush=True)

    failure_notes: List[str] = []
    if not c1_pass:
        failure_notes.append(
            f"C1 FAIL: attr_auc_THREE={auc_three:.4f} vs attr_auc_TWO={auc_two:.4f}"
            f" (delta={auc_delta:+.4f}, needs >=0.05)."
            " MotionLateralHead did not improve attribution AUC by 5pp over z_world."
        )
    if not c2_pass:
        failure_notes.append(
            f"C2 FAIL: attr_auc_THREE={auc_three:.4f} (needs >=0.65)."
            " THREE_STREAM did not reach absolute attribution learning threshold."
            " Possible causes: too few contact events (check n_agent_caused_train,"
            " n_env_caused_train); delta_hazard signal insufficient; motion_dim too small."
        )
    if not c3_pass:
        failure_notes.append(
            f"C3 FAIL: avoidance_THREE={av_three:.4f} vs avoidance_TWO={av_two:.4f}"
            f" (delta={av_delta:+.4f}, needs >=0)."
            " Attribution-guided flee did not reduce harm contacts."
        )

    for note in failure_notes:
        print(f"  {note}", flush=True)

    if all_pass:
        interpretation = (
            "MECH-099 SUPPORTED: dedicated motion-sensitive lateral head produces"
            f" better agency attribution (AUC delta={auc_delta:+.4f},"
            f" absolute AUC={auc_three:.4f} >= 0.65) and behavioral benefit"
            f" (avoidance delta={av_delta:+.4f})."
            " Three-stream architecture is superior to two-stream for agency attribution."
            " Supports MECH-095 (TPJ agency comparator) and MECH-099 (lateral stream)."
            " Operationalisation fix from EXQ-098: attribution not harm detection."
        )
    elif c1_pass and not c2_pass:
        interpretation = (
            "PARTIAL: C1 passes (relative advantage delta={:.4f}) but C2 fails"
            " (absolute AUC={:.4f} < 0.65). THREE_STREAM outperforms TWO_STREAM but"
            " neither learned reliable attribution. Data sparsity likely."
            " Check n_agent_caused_train and n_env_caused_train in per-seed results."
            " Consider increasing env_drift_prob or warmup_episodes."
        ).format(auc_delta, auc_three)
    elif c2_pass and not c1_pass:
        interpretation = (
            "PARTIAL: C2 passes (absolute AUC={:.4f} >= 0.65) but C1 fails"
            " (delta={:+.4f} < 0.05). THREE_STREAM learned attribution but did not"
            " outperform TWO_STREAM. z_world may encode agency attribution nearly as"
            " well as the dedicated motion head at this scale."
            " Consider: larger motion_dim, deeper lateral head, or longer training."
        ).format(auc_three, auc_delta)
    else:
        interpretation = (
            "MECH-099 NOT SUPPORTED at this operationalisation."
            f" attr_auc_THREE={auc_three:.4f}, delta={auc_delta:+.4f}."
            " Possible causes: (1) sparse attribution events (check train counts),"
            " (2) delta_hazard signal too noisy at current drift rate,"
            " (3) contamination dynamics do not produce discriminable agent_caused signal,"
            " (4) world_dim=32 already sufficient for attribution without lateral stream."
        )

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(
            f"- {n}" for n in failure_notes
        )

    per_three_rows = "\n".join(
        f"  seed={r['seed']}:"
        f" attr_auc={r['attribution_auc']:.4f}"
        f" avoidance={r['harm_avoidance_rate']:.4f}"
        f" agent_caused_train={r['n_agent_caused_train']}"
        f" env_caused_train={r['n_env_caused_train']}"
        f" contact_eval={r['n_contact_steps_eval']}"
        f" (a={r['n_agent_caused_eval']} e={r['n_env_caused_eval']})"
        for r in results_three
    )
    per_two_rows = "\n".join(
        f"  seed={r['seed']}:"
        f" attr_auc={r['attribution_auc']:.4f}"
        f" avoidance={r['harm_avoidance_rate']:.4f}"
        f" agent_caused_train={r['n_agent_caused_train']}"
        f" env_caused_train={r['n_env_caused_train']}"
        f" contact_eval={r['n_contact_steps_eval']}"
        f" (a={r['n_agent_caused_eval']} e={r['n_env_caused_eval']})"
        for r in results_two
    )

    summary_markdown = (
        f"# V3-EXQ-098b -- MECH-099 Agency Attribution Three-Stream Redesign\n\n"
        f"**Status:** {status}\n"
        f"**Claims:** MECH-099, MECH-095\n"
        f"**Seeds:** {list(seeds)}\n"
        f"**alpha_world:** {alpha_world}  (SD-008)\n"
        f"**Warmup:** {warmup_episodes} eps  **Eval:** {eval_episodes} eps\n"
        f"**env_drift_prob:** {env_drift_prob}  **env_drift_interval:** {env_drift_interval}\n\n"
        f"## Redesign from EXQ-098\n\n"
        f"EXQ-098 trained lateral head with BCE on harm_obs (duplicate harm signal)."
        f" Adverse direction (delta=-0.038) = information disadvantage, not architectural"
        f" failure. EXQ-098b trains lateral head with BCE on agency attribution labels"
        f" (agent_caused vs env_caused hazard contact). Motion-delta input (MT analog)"
        f" added to lateral head (delta_hazard = hazard_ch_t - hazard_ch_{{t-1}}).\n\n"
        f"## Pre-Registered Thresholds\n\n"
        f"C1: attr_AUC_THREE - attr_AUC_TWO >= 0.05  (relative attribution advantage)\n"
        f"C2: attr_AUC_THREE >= 0.65  (absolute attribution learning)\n"
        f"C3: avoidance_THREE >= avoidance_TWO  (behavioral: attribution-guided flee)\n\n"
        f"## Aggregate Results\n\n"
        f"| Metric | THREE_STREAM | TWO_STREAM | Delta | Pass |\n"
        f"|--------|-------------|-----------|-------|------|\n"
        f"| attribution_AUC (C1) | {auc_three:.4f} | {auc_two:.4f}"
        f" | {auc_delta:+.4f} | {'YES' if c1_pass else 'NO'} |\n"
        f"| attribution_AUC >= 0.65 (C2) | {auc_three:.4f} | -- | -- | {'YES' if c2_pass else 'NO'} |\n"
        f"| harm_avoidance (C3) | {av_three:.4f} | {av_two:.4f}"
        f" | {av_delta:+.4f} | {'YES' if c3_pass else 'NO'} |\n\n"
        f"## Interpretation\n\n"
        f"{interpretation}\n\n"
        f"## Per-Seed (THREE_STREAM)\n\n"
        f"{per_three_rows}\n\n"
        f"## Per-Seed (TWO_STREAM)\n\n"
        f"{per_two_rows}\n"
        f"{failure_section}\n"
    )

    return {
        "status": status,
        "metrics": {
            "attribution_auc_three":  float(auc_three),
            "attribution_auc_two":    float(auc_two),
            "auc_delta":              float(auc_delta),
            "harm_avoidance_three":   float(av_three),
            "harm_avoidance_two":     float(av_two),
            "avoidance_delta":        float(av_delta),
            "crit1_pass":             1.0 if c1_pass else 0.0,
            "crit2_pass":             1.0 if c2_pass else 0.0,
            "crit3_pass":             1.0 if c3_pass else 0.0,
            "criteria_met":           float(criteria_met),
            "n_seeds":                float(len(seeds)),
            "alpha_world":            float(alpha_world),
            "motion_dim":             float(motion_dim),
            "env_drift_prob":         float(env_drift_prob),
        },
        "summary_markdown": summary_markdown,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": (
            "supports" if all_pass
            else ("mixed" if (c1_pass or c2_pass) else "weakens")
        ),
        "experiment_type": EXPERIMENT_TYPE,
        "per_seed_three": results_three,
        "per_seed_two": results_two,
    }


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",           type=int,   nargs="+", default=[42, 7])
    parser.add_argument("--warmup",          type=int,   default=400)
    parser.add_argument("--eval-eps",        type=int,   default=100)
    parser.add_argument("--steps",           type=int,   default=200)
    parser.add_argument("--alpha-world",     type=float, default=0.9)
    parser.add_argument("--alpha-self",      type=float, default=0.3)
    parser.add_argument("--motion-dim",      type=int,   default=16)
    parser.add_argument("--harm-scale",      type=float, default=0.02)
    parser.add_argument("--proximity-scale", type=float, default=0.01)
    parser.add_argument("--drift-prob",      type=float, default=0.3)
    parser.add_argument("--drift-interval",  type=int,   default=3)
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Quick smoke test: 1 seed, 5 warmup, 3 eval, 50 steps. Writes JSON.",
    )
    args = parser.parse_args()

    if args.smoke_test:
        seeds    = (42,)
        warmup   = 5
        eval_eps = 3
        steps    = 50
        print("[V3-EXQ-098b] SMOKE TEST MODE", flush=True)
    else:
        seeds    = tuple(args.seeds)
        warmup   = args.warmup
        eval_eps = args.eval_eps
        steps    = args.steps

    result = run(
        seeds=seeds,
        warmup_episodes=warmup,
        eval_episodes=eval_eps,
        steps_per_episode=steps,
        alpha_world=args.alpha_world,
        alpha_self=args.alpha_self,
        motion_dim=args.motion_dim,
        harm_scale=args.harm_scale,
        proximity_harm_scale=args.proximity_scale,
        env_drift_prob=args.drift_prob,
        env_drift_interval=args.drift_interval,
    )

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
    if args.smoke_test:
        print("[SMOKE TEST] Key metrics:", flush=True)
        for k in [
            "attribution_auc_three", "attribution_auc_two", "auc_delta",
            "harm_avoidance_three", "harm_avoidance_two",
            "crit1_pass", "crit2_pass", "crit3_pass",
        ]:
            print(f"  {k}: {result['metrics'].get(k, 'N/A')}", flush=True)
        print("[SMOKE TEST] Per-seed attribution event counts:", flush=True)
        for r in result.get("per_seed_three", []):
            print(
                f"  THREE seed={r['seed']}:"
                f" agent_caused_train={r.get('n_agent_caused_train', '?')}"
                f" env_caused_train={r.get('n_env_caused_train', '?')}"
                f" contact_eval={r.get('n_contact_steps_eval', '?')}",
                flush=True,
            )
        for r in result.get("per_seed_two", []):
            print(
                f"  TWO  seed={r['seed']}:"
                f" agent_caused_train={r.get('n_agent_caused_train', '?')}"
                f" env_caused_train={r.get('n_env_caused_train', '?')}"
                f" contact_eval={r.get('n_contact_steps_eval', '?')}",
                flush=True,
            )
    else:
        for k, v in result["metrics"].items():
            print(f"  {k}: {v}", flush=True)
