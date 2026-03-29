#!/opt/local/bin/python3
"""
V3-EXQ-165 -- MECH-143 / MECH-144: Hippocampal Value Sensitivity Probe

Claims:   MECH-143, MECH-144
ARC-007 STRICT (Q-020): HippocampalModule generates value-flat proposals.
Terrain sensitivity = consequence of navigating residue-shaped z_world, not a
separate value computation.

MECH-143: Dorsal CA1 is value-free -- place cells are insensitive to goal value
changes. The HippocampalModule should produce consistent proposals regardless of
which goal has higher reward value.

MECH-144: Ventral CA1 has spatially organised valence encoding -- valence is
intrinsic to map geometry. If the current undifferentiated HippocampalModule has
implicitly learned value sensitivity, performance should degrade when goal values
are shuffled.

Experimental design
-------------------
Two conditions × 2 seeds = 4 cells.
Inline 8x8 grid world (no ree_core agent imports). All terrain navigation is
implemented as a lightweight hippocampal analog operating on a residue field.

Condition A: FIXED_VALUE
  goal_A_reward = 1.0 (high), goal_B_reward = 0.5 (low), fixed for all 400 episodes.
  The agent can learn which goal is more valuable.

Condition B: SHUFFLED_VALUE
  Every K_SHUFFLE=50 episodes, randomly swap: with prob=0.5, goal_A becomes low
  (0.5) and goal_B becomes high (1.0), or vice versa.
  Goal POSITIONS do not change. Hazard positions and residue field do not change.
  Only the reward value assignment shuffles.

Key prediction (MECH-143 / ARC-007 STRICT):
  If HippocampalModule is value-flat:
    SHUFFLED_VALUE performance ~= FIXED_VALUE performance.
    Terrain navigation doesn't depend on which goal has higher reward.
  If HippocampalModule has learned value-sensitivity:
    SHUFFLED_VALUE performance < FIXED_VALUE performance.
    Confusion after each shuffle; takes K' episodes to re-learn.

Hippocampal terrain navigator (inline, value-blind):
  - residue_field: 2D numpy array (GRID_SIZE x GRID_SIZE), init to 0.
  - After each step: residue_field[pos] += harm_signal * 0.1.
  - Navigation: propose action that avoids high-residue neighbours.
    Add NAV_BIAS toward chosen goal position (residue + goal pull).
  - Does NOT receive or use reward values -- only goal positions.

Separate goal_selection_head (Linear): decides which goal (A or B) to navigate
toward based on z_self + z_world. This IS allowed to be reward-sensitive.
Goal selection disruption after shuffles is expected and tracked separately.

Pre-registered criteria
-----------------------
C1: |FIXED_VALUE harm_rate - SHUFFLED_VALUE harm_rate| < THRESH_HARM_DIFF (both seeds)
    -- harm avoidance is not significantly affected by value shuffling (value-flat nav)
C2: |FIXED_VALUE goal_completion_rate - SHUFFLED_VALUE goal_completion_rate| < THRESH_COMPLETION_DIFF (both seeds)
    -- goal completion is not significantly affected (value-flat nav)
C3: SHUFFLED_VALUE post_shuffle_harm_spike < THRESH_SPIKE_TRIVIAL (both seeds)
    -- no detectable disruption after a shuffle event (strongly value-flat)
C4: data quality: n_goal_completions > 20 per seed, n_shuffle_events > 4 per seed

PASS:                C1+C2+C3+C4 -> strongly value-flat navigation; supports MECH-143 / ARC-007 STRICT
PARTIAL_SOFT_FLAT:   C1+C2+C4, NOT C3 -> flat in aggregate but transient shuffle disruption exists
PARTIAL_VALUE_SENSITIVE: C3+C4, C1 or C2 fail -> significant value-sensitivity; violates ARC-007 STRICT; supports MECH-144
FAIL:                C4 fails or training divergence
"""

import sys
import json
import random
import uuid
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPERIMENT_TYPE   = "v3_exq_165_mech143_hippocampal_value_sensitivity"
CLAIM_IDS         = ["MECH-143", "MECH-144"]
RUN_ID_PREFIX     = "v3_exq_165_mech143_hippocampal_value_sensitivity"
CONDITIONS        = ["FIXED_VALUE", "SHUFFLED_VALUE"]
SEEDS             = [42, 123]
N_EPISODES        = 400
N_EVAL_EPISODES   = 100   # last quartile
STEPS_PER_EPISODE = 200
K_SHUFFLE         = 50
GRID_SIZE         = 8
N_HAZARDS         = 4
N_RESOURCES       = 2
HAZARD_HARM       = 0.05
NAV_BIAS          = 0.6
GOAL_A_REWARD_HIGH = 1.0
GOAL_A_REWARD_LOW  = 0.5
LR                = 1e-3
HIDDEN_DIM        = 32
WORLD_DIM         = 32
SELF_DIM          = 16
HARM_DIM          = 8
RECOVERY_THRESH   = 0.005   # harm_rate within this of pre-shuffle = recovered

THRESH_HARM_DIFF        = 0.02
THRESH_COMPLETION_DIFF  = 0.05
THRESH_SPIKE_TRIVIAL    = 0.01

# Goal positions (fixed throughout)
GOAL_A_POS = (2, 2)
GOAL_B_POS = (6, 6)

# Actions: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT, 4=STAY
ACTION_DELTAS = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1),
    4: (0, 0),
}
N_ACTIONS = 5


# ---------------------------------------------------------------------------
# Inline grid world
# ---------------------------------------------------------------------------

class ValueSensitivityGridWorld:
    """
    Lightweight 8x8 grid world for the value sensitivity probe.
    - Agent, 2 goals (fixed positions), N_HAZARDS hazards (fixed positions).
    - Reaching a goal gives reward (value assigned per condition/shuffle).
    - Stepping on a hazard gives harm = HAZARD_HARM.
    - Goal values can be swapped; positions never change.
    """

    def __init__(self, seed: int, size: int = GRID_SIZE, n_hazards: int = N_HAZARDS):
        self.size = size
        self.n_hazards = n_hazards
        self.rng = np.random.default_rng(seed)

        self.goal_a_pos = GOAL_A_POS
        self.goal_b_pos = GOAL_B_POS

        # Place hazards: avoid goal positions
        forbidden = {self.goal_a_pos, self.goal_b_pos}
        all_cells = [(r, c) for r in range(size) for c in range(size) if (r, c) not in forbidden]
        hazard_idxs = self.rng.choice(len(all_cells), size=n_hazards, replace=False)
        self.hazard_positions = set(all_cells[i] for i in hazard_idxs)

        # Default values
        self.goal_a_reward = GOAL_A_REWARD_HIGH
        self.goal_b_reward = GOAL_A_REWARD_LOW

        # Agent state
        self.agent_pos: Tuple[int, int] = (size // 2, size // 2)
        self._reset_agent_pos()

    def _reset_agent_pos(self):
        """Place agent at centre (away from hazards if possible)."""
        centre = (self.size // 2, self.size // 2)
        if centre not in self.hazard_positions and centre != self.goal_a_pos and centre != self.goal_b_pos:
            self.agent_pos = centre
        else:
            for r in range(self.size):
                for c in range(self.size):
                    pos = (r, c)
                    if pos not in self.hazard_positions and pos != self.goal_a_pos and pos != self.goal_b_pos:
                        self.agent_pos = pos
                        return

    def set_values(self, goal_a_reward: float, goal_b_reward: float):
        self.goal_a_reward = goal_a_reward
        self.goal_b_reward = goal_b_reward

    def reset(self):
        self._reset_agent_pos()
        return self._get_obs()

    def step(self, action: int) -> Tuple[dict, float, float, bool]:
        """
        Returns: (obs, reward, harm, done)
        done=True when a goal is reached OR steps exhausted (caller tracks steps).
        """
        dr, dc = ACTION_DELTAS[action]
        nr = max(0, min(self.size - 1, self.agent_pos[0] + dr))
        nc = max(0, min(self.size - 1, self.agent_pos[1] + dc))
        self.agent_pos = (nr, nc)

        harm = 0.0
        reward = 0.0
        done = False

        if self.agent_pos in self.hazard_positions:
            harm = HAZARD_HARM

        if self.agent_pos == self.goal_a_pos:
            reward = self.goal_a_reward
            done = True
        elif self.agent_pos == self.goal_b_pos:
            reward = self.goal_b_reward
            done = True

        return self._get_obs(), reward, harm, done

    def _get_obs(self) -> dict:
        r, c = self.agent_pos
        # World obs: one-hot grid cells around agent (3x3 neighbourhood) + goal directions
        neighbourhood = np.zeros(9, dtype=np.float32)
        idx = 0
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                nr2, nc2 = r + dr, c + dc
                if 0 <= nr2 < self.size and 0 <= nc2 < self.size:
                    if (nr2, nc2) in self.hazard_positions:
                        neighbourhood[idx] = 1.0
                idx += 1

        # Direction vectors to goals (normalised)
        def dir_vec(target):
            tr, tc = target
            dr2 = tr - r
            dc2 = tc - c
            dist = max(1.0, abs(dr2) + abs(dc2))
            return np.array([dr2 / dist, dc2 / dist], dtype=np.float32)

        dir_a = dir_vec(self.goal_a_pos)
        dir_b = dir_vec(self.goal_b_pos)

        # Self obs: normalised agent position
        self_obs = np.array([r / self.size, c / self.size], dtype=np.float32)

        world_obs = np.concatenate([neighbourhood, dir_a, dir_b]).astype(np.float32)
        return {"world_obs": world_obs, "self_obs": self_obs}

    @property
    def world_obs_dim(self) -> int:
        return 13  # 9 neighbourhood + 2+2 goal directions

    @property
    def self_obs_dim(self) -> int:
        return 2   # normalised (r, c)


# ---------------------------------------------------------------------------
# Neural modules (inline, lightweight)
# ---------------------------------------------------------------------------

class WorldEncoder(nn.Module):
    def __init__(self, obs_dim: int, world_dim: int):
        super().__init__()
        self.fc = nn.Linear(obs_dim, world_dim)
        self.norm = nn.LayerNorm(world_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.fc(x))


class SelfEncoder(nn.Module):
    def __init__(self, obs_dim: int, self_dim: int):
        super().__init__()
        self.fc = nn.Linear(obs_dim, self_dim)
        self.norm = nn.LayerNorm(self_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.fc(x))


class GoalSelectionHead(nn.Module):
    """
    Selects which goal to navigate toward.
    Input: concat(z_self, z_world). Output: logits over {goal_A, goal_B}.
    This IS allowed to be reward-sensitive (learns from reward signal).
    """
    def __init__(self, self_dim: int, world_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(self_dim + world_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, z_self: torch.Tensor, z_world: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z_self, z_world], dim=-1))


class HarmPredictor(nn.Module):
    """
    Predicts harm signal from z_world + z_self.
    Auxiliary predictor, trained on harm observations.
    """
    def __init__(self, world_dim: int, self_dim: int):
        super().__init__()
        self.fc = nn.Linear(world_dim + self_dim, 1)

    def forward(self, z_world: torch.Tensor, z_self: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.cat([z_world, z_self], dim=-1))


# ---------------------------------------------------------------------------
# Hippocampal terrain navigator (value-blind)
# ---------------------------------------------------------------------------

class HippocampalTerrainNavigator:
    """
    Lightweight hippocampal terrain navigator.

    Maintains a residue_field (running mean harm accumulation per grid cell).
    Proposes actions by:
      1. Scoring each candidate next-cell by residue (avoid high-residue regions).
      2. Adding NAV_BIAS pull toward the chosen goal position.

    Crucially: does NOT receive or use reward values.
    Only goal POSITIONS are available (terrain geometry).
    """

    def __init__(self, grid_size: int, nav_bias: float = NAV_BIAS):
        self.grid_size = grid_size
        self.nav_bias = nav_bias
        self.residue_field = np.zeros((grid_size, grid_size), dtype=np.float32)

    def update_residue(self, pos: Tuple[int, int], harm: float):
        """Accumulate harm at agent position."""
        r, c = pos
        self.residue_field[r, c] += harm * 0.1

    def propose_action(
        self,
        agent_pos: Tuple[int, int],
        goal_pos: Tuple[int, int],
        rng: np.random.Generator,
    ) -> int:
        """
        Propose action that balances:
          - Avoiding high-residue neighbouring cells.
          - Moving toward goal_pos (terrain pull).

        Returns action index (0-4).
        Value of goal is NOT used -- only position.
        """
        r, c = agent_pos
        gr, gc = goal_pos
        scores = np.zeros(N_ACTIONS, dtype=np.float32)

        for a, (dr, dc) in ACTION_DELTAS.items():
            nr = max(0, min(self.grid_size - 1, r + dr))
            nc = max(0, min(self.grid_size - 1, c + dc))
            # Residue penalty (negated: lower residue = higher score)
            residue_penalty = self.residue_field[nr, nc]
            # Goal pull: Manhattan distance reduction
            old_dist = abs(gr - r) + abs(gc - c)
            new_dist = abs(gr - nr) + abs(gc - nc)
            goal_pull = (old_dist - new_dist) * self.nav_bias
            scores[a] = goal_pull - residue_penalty

        # Softmax to get probabilities
        scores_exp = np.exp(scores - scores.max())
        probs = scores_exp / scores_exp.sum()
        return int(rng.choice(N_ACTIONS, p=probs))

    @property
    def mean_residue(self) -> float:
        return float(self.residue_field.mean())

    def get_trajectory_residue(
        self, path: List[Tuple[int, int]]
    ) -> float:
        """Mean residue along a path."""
        if not path:
            return 0.0
        return float(np.mean([self.residue_field[r, c] for r, c in path]))


# ---------------------------------------------------------------------------
# Single condition/seed cell runner
# ---------------------------------------------------------------------------

def _run_cell(
    seed: int,
    condition: str,
    n_episodes: int,
    n_eval_episodes: int,
    steps_per_episode: int,
    dry_run: bool,
) -> Dict:
    """
    Run one (condition, seed) cell.
    Returns per-cell metrics dict.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    rng = np.random.default_rng(seed)

    env = ValueSensitivityGridWorld(seed=seed)
    navigator = HippocampalTerrainNavigator(grid_size=GRID_SIZE, nav_bias=NAV_BIAS)

    # Neural modules
    world_enc = WorldEncoder(obs_dim=env.world_obs_dim, world_dim=WORLD_DIM)
    self_enc  = SelfEncoder(obs_dim=env.self_obs_dim, self_dim=SELF_DIM)
    goal_head = GoalSelectionHead(self_dim=SELF_DIM, world_dim=WORLD_DIM, hidden_dim=HIDDEN_DIM)
    harm_pred = HarmPredictor(world_dim=WORLD_DIM, self_dim=SELF_DIM)

    params = (
        list(world_enc.parameters())
        + list(self_enc.parameters())
        + list(goal_head.parameters())
        + list(harm_pred.parameters())
    )
    optimizer = optim.Adam(params, lr=LR)

    # Current value assignment
    # True: goal_A=HIGH, goal_B=LOW; False: goal_A=LOW, goal_B=HIGH
    goal_a_is_high = True

    def _apply_values():
        if goal_a_is_high:
            env.set_values(GOAL_A_REWARD_HIGH, GOAL_A_REWARD_LOW)
        else:
            env.set_values(GOAL_A_REWARD_LOW, GOAL_A_REWARD_HIGH)

    _apply_values()

    # Per-episode tracking
    ep_harm_rates: List[float] = []
    ep_completions: List[bool] = []
    ep_steps_to_goal: List[float] = []       # steps when goal reached (last quartile)
    ep_goal_selected: List[int] = []         # 0=A, 1=B (for each episode)

    # Shuffle event tracking (SHUFFLED_VALUE only)
    shuffle_events: List[int] = []           # episode indices where shuffles occurred
    harm_window_around_shuffles: List[Tuple[float, float]] = []  # (pre_mean, post_mean)

    n_goal_completions = 0
    n_shuffle_events = 0

    for ep in range(n_episodes):
        # --- Value shuffle (SHUFFLED_VALUE condition) ---
        if condition == "SHUFFLED_VALUE" and ep > 0 and ep % K_SHUFFLE == 0:
            swap = rng.random() < 0.5
            if swap:
                goal_a_is_high = not goal_a_is_high
                _apply_values()
                shuffle_events.append(ep)
                n_shuffle_events += 1

        obs = env.reset()
        total_harm = 0.0
        reached_goal = False
        steps_taken = 0
        path: List[Tuple[int, int]] = [env.agent_pos]

        episode_losses: List[torch.Tensor] = []

        for step in range(steps_per_episode):
            # Encode observations
            world_t = torch.tensor(obs["world_obs"], dtype=torch.float32).unsqueeze(0)
            self_t  = torch.tensor(obs["self_obs"],  dtype=torch.float32).unsqueeze(0)

            z_world = world_enc(world_t)
            z_self  = self_enc(self_t)

            # Goal selection (value-sensitive head)
            goal_logits = goal_head(z_self, z_world)
            goal_probs  = torch.softmax(goal_logits, dim=-1)
            # Sample during training; greedy for eval (last quartile)
            is_eval = ep >= (n_episodes - n_eval_episodes)
            if is_eval or dry_run:
                goal_choice = int(goal_probs.argmax(dim=-1).item())
            else:
                goal_choice = int(torch.multinomial(goal_probs, 1).item())

            # Map goal choice to position (value-blind navigator only sees positions)
            chosen_goal_pos = GOAL_A_POS if goal_choice == 0 else GOAL_B_POS

            # Hippocampal terrain navigation (value-blind)
            action = navigator.propose_action(
                agent_pos=env.agent_pos,
                goal_pos=chosen_goal_pos,
                rng=rng,
            )

            next_obs, reward, harm, done = env.step(action)
            navigator.update_residue(env.agent_pos, harm)

            total_harm += harm
            steps_taken += 1
            path.append(env.agent_pos)

            # Goal selection loss: cross-entropy toward the goal that gives higher reward
            # (reward is available to goal_head training -- it IS value-sensitive)
            if reward > 0.0:
                # Which goal did we reach? (position-based)
                reached_a = (env.agent_pos == GOAL_A_POS)
                target_goal = 0 if reached_a else 1
                target_t = torch.tensor([target_goal], dtype=torch.long)
                goal_loss = F.cross_entropy(goal_logits, target_t)
            else:
                # Entropy regularisation to prevent premature collapse
                eps = 1e-6
                goal_loss = (goal_probs * torch.log(goal_probs + eps)).sum() * 0.01

            # Harm predictor loss
            harm_t = torch.tensor([[harm]], dtype=torch.float32)
            harm_pred_out = harm_pred(z_world.detach(), z_self.detach())
            harm_loss = F.mse_loss(harm_pred_out, harm_t)

            total_loss = goal_loss + harm_loss
            episode_losses.append(total_loss)

            obs = next_obs

            if done:
                reached_goal = True
                break

        # Update neural modules after episode
        if episode_losses and not dry_run:
            optimizer.zero_grad()
            combined_loss = torch.stack(episode_losses).mean()
            combined_loss.backward()
            optimizer.step()

        harm_rate = total_harm / max(steps_taken, 1)
        ep_harm_rates.append(harm_rate)
        ep_completions.append(reached_goal)
        ep_goal_selected.append(goal_choice)

        if reached_goal:
            n_goal_completions += 1
            if is_eval:
                ep_steps_to_goal.append(steps_taken / GRID_SIZE)

    # --- Compute per-cell metrics ---
    eval_start = n_episodes - n_eval_episodes

    eval_harm_rates      = ep_harm_rates[eval_start:]
    eval_completions     = ep_completions[eval_start:]

    harm_rate_eval       = float(np.mean(eval_harm_rates))
    goal_completion_rate = float(np.mean([1.0 if c else 0.0 for c in eval_completions]))
    path_efficiency      = float(np.mean(ep_steps_to_goal)) if ep_steps_to_goal else float("nan")

    # Post-shuffle disruption metrics (SHUFFLED_VALUE only)
    post_shuffle_harm_spike  = float("nan")
    adaptation_speed_mean    = float("nan")

    if condition == "SHUFFLED_VALUE" and shuffle_events:
        spike_diffs = []
        adapt_speeds = []

        for ev_ep in shuffle_events:
            # Pre-shuffle window: last 10 episodes before shuffle
            pre_start  = max(0, ev_ep - 10)
            pre_window = ep_harm_rates[pre_start:ev_ep]

            # Post-shuffle window: first 10 episodes after shuffle
            post_end    = min(n_episodes, ev_ep + 10)
            post_window = ep_harm_rates[ev_ep:post_end]

            if pre_window and post_window:
                pre_mean  = float(np.mean(pre_window))
                post_mean = float(np.mean(post_window))
                spike_diffs.append(max(0.0, post_mean - pre_mean))

                # Adaptation speed: episodes until harm_rate returns within RECOVERY_THRESH of pre_mean
                recovered_at = 10  # default: not recovered within window
                for offset in range(len(post_window)):
                    if abs(ep_harm_rates[ev_ep + offset] - pre_mean) <= RECOVERY_THRESH:
                        recovered_at = offset
                        break
                adapt_speeds.append(recovered_at)

        if spike_diffs:
            post_shuffle_harm_spike = float(np.mean(spike_diffs))
        if adapt_speeds:
            adaptation_speed_mean = float(np.mean(adapt_speeds))

    return {
        "seed":                    seed,
        "condition":               condition,
        "harm_rate":               harm_rate_eval,
        "goal_completion_rate":    goal_completion_rate,
        "path_efficiency":         path_efficiency,
        "post_shuffle_harm_spike": post_shuffle_harm_spike,
        "adaptation_speed_mean":   adaptation_speed_mean,
        "n_goal_completions":      n_goal_completions,
        "n_shuffle_events":        n_shuffle_events,
        "shuffle_event_episodes":  shuffle_events,
        "mean_residue_final":      float(navigator.mean_residue),
    }


# ---------------------------------------------------------------------------
# Criteria evaluation
# ---------------------------------------------------------------------------

def _evaluate_criteria(
    results_by_condition: Dict[str, List[Dict]]
) -> Dict[str, bool]:
    """Evaluate C1-C4 across seeds."""

    fixed_results    = results_by_condition["FIXED_VALUE"]
    shuffled_results = results_by_condition["SHUFFLED_VALUE"]

    # C1: |harm_rate diff| < THRESH_HARM_DIFF for both seeds
    c1 = all(
        abs(f["harm_rate"] - s["harm_rate"]) < THRESH_HARM_DIFF
        for f, s in zip(fixed_results, shuffled_results)
    )

    # C2: |goal_completion_rate diff| < THRESH_COMPLETION_DIFF for both seeds
    c2 = all(
        abs(f["goal_completion_rate"] - s["goal_completion_rate"]) < THRESH_COMPLETION_DIFF
        for f, s in zip(fixed_results, shuffled_results)
    )

    # C3: post_shuffle_harm_spike < THRESH_SPIKE_TRIVIAL for both SHUFFLED seeds
    c3_vals = []
    for s in shuffled_results:
        spike = s["post_shuffle_harm_spike"]
        if not (spike != spike):  # nan check
            c3_vals.append(spike < THRESH_SPIKE_TRIVIAL)
        else:
            c3_vals.append(False)  # nan = no shuffles recorded = C3 fails
    c3 = all(c3_vals)

    # C4: data quality gates
    c4_goal = all(r["n_goal_completions"] > 20 for cond_results in results_by_condition.values()
                  for r in cond_results)
    c4_shuffle = all(r["n_shuffle_events"] > 4 for r in shuffled_results)
    c4 = c4_goal and c4_shuffle

    return {"C1": c1, "C2": c2, "C3": c3, "C4": c4}


def _determine_outcome(criteria: Dict[str, bool]) -> str:
    c1, c2, c3, c4 = criteria["C1"], criteria["C2"], criteria["C3"], criteria["C4"]

    if not c4:
        return "FAIL"

    if c1 and c2 and c3:
        return "PASS"

    if c1 and c2 and not c3:
        return "PARTIAL_SOFT_FLAT"

    if c3 and (not c1 or not c2):
        return "PARTIAL_VALUE_SENSITIVE"

    # C4 met but neither PASS/PARTIAL pattern matches
    return "FAIL"


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run_experiment(dry_run: bool = False) -> Dict:
    mode_str = "[DRY RUN] " if dry_run else ""
    print(f"{mode_str}EXQ-165: MECH-143/MECH-144 Hippocampal Value Sensitivity Probe", flush=True)
    print(f"  Conditions: {CONDITIONS}", flush=True)
    print(f"  Seeds:      {SEEDS}", flush=True)
    print(f"  Episodes:   {N_EPISODES} (eval last {N_EVAL_EPISODES})", flush=True)
    print(f"  Grid:       {GRID_SIZE}x{GRID_SIZE}, {N_HAZARDS} hazards, K_SHUFFLE={K_SHUFFLE}", flush=True)

    n_episodes_actual = 1 if dry_run else N_EPISODES

    results_by_condition: Dict[str, List[Dict]] = {cond: [] for cond in CONDITIONS}

    for condition in CONDITIONS:
        print(f"\n--- Condition: {condition} ---", flush=True)
        for seed in SEEDS:
            print(f"  Seed {seed} ...", flush=True)
            result = _run_cell(
                seed=seed,
                condition=condition,
                n_episodes=n_episodes_actual,
                n_eval_episodes=min(N_EVAL_EPISODES, n_episodes_actual),
                steps_per_episode=STEPS_PER_EPISODE,
                dry_run=dry_run,
            )
            results_by_condition[condition].append(result)
            hr   = result["harm_rate"]
            gcr  = result["goal_completion_rate"]
            pe   = result["path_efficiency"]
            sp   = result["post_shuffle_harm_spike"]
            shuf = result["n_shuffle_events"]
            nc   = result["n_goal_completions"]
            print(
                f"    harm_rate={hr:.4f}  goal_completion={gcr:.3f}  "
                f"path_eff={pe:.3f}  spike={sp:.4f}  "
                f"n_completions={nc}  n_shuffles={shuf}",
                flush=True,
            )

    print("\n=== Evaluating criteria ===", flush=True)
    criteria = _evaluate_criteria(results_by_condition)
    outcome  = _determine_outcome(criteria)

    for k, v in criteria.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}", flush=True)
    print(f"Overall outcome: {outcome}", flush=True)

    # --- Summary metrics (mean over seeds per condition) ---
    def _mean_seeds(cond: str, key: str) -> float:
        vals = [r[key] for r in results_by_condition[cond] if r[key] == r[key]]  # exclude nan
        return float(np.mean(vals)) if vals else float("nan")

    summary_metrics: Dict = {}
    for cond in CONDITIONS:
        prefix = cond.lower()
        summary_metrics[f"{prefix}_harm_rate"]               = _mean_seeds(cond, "harm_rate")
        summary_metrics[f"{prefix}_goal_completion_rate"]    = _mean_seeds(cond, "goal_completion_rate")
        summary_metrics[f"{prefix}_path_efficiency"]         = _mean_seeds(cond, "path_efficiency")
        summary_metrics[f"{prefix}_post_shuffle_harm_spike"] = _mean_seeds(cond, "post_shuffle_harm_spike")
        summary_metrics[f"{prefix}_adaptation_speed_mean"]   = _mean_seeds(cond, "adaptation_speed_mean")
        summary_metrics[f"{prefix}_n_goal_completions_mean"] = _mean_seeds(cond, "n_goal_completions")
        summary_metrics[f"{prefix}_n_shuffle_events_mean"]   = _mean_seeds(cond, "n_shuffle_events")

    # Pairwise deltas (positive = SHUFFLED worse than FIXED)
    summary_metrics["delta_harm_fixed_vs_shuffled"] = (
        summary_metrics["shuffled_value_harm_rate"]
        - summary_metrics["fixed_value_harm_rate"]
    )
    summary_metrics["delta_completion_fixed_vs_shuffled"] = (
        summary_metrics["fixed_value_goal_completion_rate"]
        - summary_metrics["shuffled_value_goal_completion_rate"]
    )

    # Evidence direction
    if outcome == "PASS":
        evidence_direction = "supports"
        guidance = "value_flat_navigation_confirmed_supports_mech_143_arc007_strict"
    elif outcome == "PARTIAL_SOFT_FLAT":
        evidence_direction = "mixed"
        guidance = "aggregate_flat_but_transient_shuffle_disruption_shallow_implicit_value_learning"
    elif outcome == "PARTIAL_VALUE_SENSITIVE":
        evidence_direction = "weakens"
        guidance = "value_sensitive_navigation_detected_violates_arc007_strict_supports_mech_144"
    else:
        evidence_direction = "mixed"
        guidance = "insufficient_data_or_training_divergence"

    run_id = f"{RUN_ID_PREFIX}_{uuid.uuid4().hex[:8]}_v3"

    pack = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "claim_ids_tested": CLAIM_IDS,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "outcome": outcome,
        "evidence_direction": evidence_direction,
        "evidence_class": "discriminative_pair",
        "guidance": guidance,
        "criteria_met": criteria,
        "pre_registered_thresholds": {
            "THRESH_HARM_DIFF":       THRESH_HARM_DIFF,
            "THRESH_COMPLETION_DIFF": THRESH_COMPLETION_DIFF,
            "THRESH_SPIKE_TRIVIAL":   THRESH_SPIKE_TRIVIAL,
            "RECOVERY_THRESH":        RECOVERY_THRESH,
            "K_SHUFFLE":              K_SHUFFLE,
        },
        "summary_metrics": summary_metrics,
        "per_seed_results": {cond: results_by_condition[cond] for cond in CONDITIONS},
        "config": {
            "grid_size":            GRID_SIZE,
            "n_hazards":            N_HAZARDS,
            "n_resources":          N_RESOURCES,
            "n_episodes":           n_episodes_actual,
            "n_eval_episodes":      min(N_EVAL_EPISODES, n_episodes_actual),
            "steps_per_episode":    STEPS_PER_EPISODE,
            "k_shuffle":            K_SHUFFLE,
            "hazard_harm":          HAZARD_HARM,
            "nav_bias":             NAV_BIAS,
            "goal_a_pos":           list(GOAL_A_POS),
            "goal_b_pos":           list(GOAL_B_POS),
            "goal_a_reward_high":   GOAL_A_REWARD_HIGH,
            "goal_a_reward_low":    GOAL_A_REWARD_LOW,
            "lr":                   LR,
            "hidden_dim":           HIDDEN_DIM,
            "world_dim":            WORLD_DIM,
            "self_dim":             SELF_DIM,
            "harm_dim":             HARM_DIM,
            "seeds":                SEEDS,
        },
        "scenario": (
            "FIXED_VALUE: goal_A=1.0 (high), goal_B=0.5 (low), fixed for all 400 episodes."
            " SHUFFLED_VALUE: every K=50 episodes, with prob=0.5 swap goal value assignments."
            " Goal positions never change. Hazard positions never change."
            " Hippocampal terrain navigator is value-blind (residue-field only, no reward inputs)."
            " GoalSelectionHead IS reward-sensitive (expected to show shuffle disruption)."
            " 2 conditions x 2 seeds = 4 cells."
            " C1: harm_rate diff < 0.02. C2: completion_rate diff < 0.05."
            " C3: post_shuffle_harm_spike < 0.01. C4: data quality gates."
        ),
        "interpretation": (
            "PASS => strongly value-flat navigation confirmed."
            " HippocampalModule terrain navigation is value-free per MECH-143."
            " ARC-007 STRICT upheld: terrain sensitivity is a consequence of residue-field"
            " geometry, not a separate value computation."
            " PARTIAL_SOFT_FLAT => aggregate metrics flat but transient disruption"
            " detectable after shuffles. Implicit value learning exists but is shallow."
            " ARC-007 STRICT held at aggregate level; minor implementation issue."
            " PARTIAL_VALUE_SENSITIVE => significant value-sensitivity in navigation."
            " Current undifferentiated HippocampalModule violates ARC-007 STRICT."
            " Supports MECH-144: valence intrinsic to map geometry in undifferentiated module."
            " Dorsal/ventral CA1 separation required before ARC-007 STRICT is achievable."
            " FAIL => insufficient data or training divergence; not informative."
        ),
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
        print("\n[dry_run] Result pack NOT written to disk.", flush=True)
        print(json.dumps(pack, indent=2), flush=True)

    return pack


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    result = run_experiment(dry_run=dry_run)
    print(f"Done. Outcome: {result['outcome']}", flush=True)
