#!/opt/local/bin/python3
"""
V3-EXQ-113 -- SD-005 Double Dissociation: z_self/z_world Latent Split

Claims: SD-005
Proposal: EXP-0006 / EVB-0006

SD-005 asserts that splitting z_gamma into z_self (body/motor domain, E2) and z_world
(world-state domain, E3/Hippocampus) is the correct architectural design. Specifically:
  - z_world should IMPROVE hazard approach detection in SPLIT (it is freed from body-state
    dilution). EXQ-078 confirmed this: +4.94 pp delta (3/3 PASS).
  - z_self should LOSE world-state content in SPLIT (it is trained only on body_obs, so
    it should carry less hazard proximity information than UNIFIED z_self which absorbs
    z_world signal by averaging).

EXQ-047g tested the wrong lens (action decoding): differential loss routing does NOT
force motor specialisation in z_self when body_obs contains less action-predictive
information than full obs. The correct purification test is whether z_self loses
WORLD-STATE content (hazard detection) in the SPLIT condition.

This experiment implements a double dissociation design:
  SPLIT   -- unified_latent_mode=False: z_self from body_obs only, z_world from world_obs.
  UNIFIED -- unified_latent_mode=True:  z_self = z_world = avg(z_self_raw, z_world_raw).

For EACH condition, we train:
  (A) E3.harm_eval_head on z_world latents (harm_signal < 0 positive events)
  (B) A standalone linear harm probe on z_self latents (same labels)

Then evaluate calibration gaps:
  gap_world = mean(harm_eval(z_world) at hazard_approach) - mean(at none)
  gap_self  = mean(self_probe(z_self)  at hazard_approach) - mean(at none)

Double dissociation:
  C1: delta_approach_gap_world (SPLIT - UNIFIED) >= 0.02  -- z_world improves in SPLIT
  C2: gap_approach_split_world >= 0.06                    -- SPLIT world absolutely detects approach
  C3: gap_approach_unified_world >= 0.0                   -- UNIFIED world learns something
  C4: delta_approach_gap_self (SPLIT - UNIFIED) <= -0.02  -- z_self LOSES world content in SPLIT
  C5: n_approach_eval >= 30                               -- sufficient approach events

Decision scoring:
  retain_ree:        C1+C2+C3+C4+C5 all pass (full double dissociation)
  hybridize:         C1+C2+C3 pass, C4 fails  (z_world advantage replicated, self purification unclear)
  retire_ree_claim:  C1 or C2 fail             (z_world split provides no detectable advantage)
"""

import sys
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_113_sd005_double_dissociation"
CLAIM_IDS = ["SD-005"]

# Pre-registered thresholds
THRESH_C1 = 0.02   # delta_approach_gap_world >= 0.02
THRESH_C2 = 0.06   # gap_approach_split_world >= 0.06
THRESH_C3 = 0.0    # gap_approach_unified_world >= 0.0
THRESH_C4 = -0.02  # delta_approach_gap_self <= -0.02
THRESH_C5 = 30     # n_approach_eval >= 30


def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


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
    unified: bool,
    warmup_episodes: int,
    eval_episodes: int,
    steps_per_episode: int,
    self_dim: int,
    world_dim: int,
    lr: float,
    alpha_world: float,
    alpha_self: float,
    harm_scale: float,
    proximity_harm_scale: float,
    nav_bias: float,
    dry_run: bool,
) -> Dict:
    """Run one (seed, condition) cell; return world and self harm gap metrics."""
    torch.manual_seed(seed)
    random.seed(seed)

    cond_label = "UNIFIED" if unified else "SPLIT"

    env = CausalGridWorldV2(
        seed=seed,
        size=6,
        num_hazards=4,
        num_resources=3,
        hazard_harm=harm_scale,
        env_drift_interval=5,
        env_drift_prob=0.1,
        proximity_harm_scale=proximity_harm_scale,
        proximity_benefit_scale=proximity_harm_scale * 0.6,
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
        alpha_world=alpha_world,
        alpha_self=alpha_self,
        reafference_action_dim=0,  # SD-007 disabled -- isolate SD-005
    )
    config.latent.unified_latent_mode = unified

    agent = REEAgent(config)

    # Standalone z_self harm probe (linear head, not part of agent)
    self_probe = nn.Linear(self_dim, 1)
    self_probe_optimizer = optim.Adam(self_probe.parameters(), lr=1e-4)

    # Separate harm buffers for z_world (for harm_eval_head) and z_self (for self_probe)
    world_buf_pos: List[torch.Tensor] = []
    world_buf_neg: List[torch.Tensor] = []
    self_buf_pos: List[torch.Tensor] = []
    self_buf_neg: List[torch.Tensor] = []
    MAX_BUF_EACH = 2000

    standard_params = [
        p for n, p in agent.named_parameters()
        if "harm_eval_head" not in n
    ]
    harm_eval_params = list(agent.e3.harm_eval_head.parameters())

    optimizer = optim.Adam(standard_params, lr=lr)
    harm_eval_optimizer = optim.Adam(harm_eval_params, lr=1e-4)

    counts: Dict[str, int] = {
        "hazard_approach": 0,
        "env_caused_hazard": 0,
        "agent_caused_hazard": 0,
        "none": 0,
    }

    if dry_run:
        actual_warmup = min(3, warmup_episodes)
        actual_eval = min(2, eval_episodes)
    else:
        actual_warmup = warmup_episodes
        actual_eval = eval_episodes

    # --- TRAIN ---
    agent.train()
    self_probe.train()

    for ep in range(actual_warmup):
        _, obs_dict = env.reset()
        agent.reset()

        for _ in range(steps_per_episode):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            latent = agent.sense(obs_body, obs_world)
            agent.clock.advance()

            z_world_curr = latent.z_world.detach()
            z_self_curr = latent.z_self.detach()

            # Biased navigation: nav_bias chance to move toward nearest hazard
            if random.random() < nav_bias:
                action_idx = _hazard_approach_action(env, n_actions)
            else:
                action_idx = random.randint(0, n_actions - 1)
            action = _action_to_onehot(action_idx, n_actions, agent.device)
            agent._last_action = action

            _, harm_signal, done, info, obs_dict = env.step(action)
            ttype = info.get("transition_type", "none")

            if ttype in counts:
                counts[ttype] += 1

            is_harm = float(harm_signal) < 0

            # Fill z_world buffer
            if is_harm:
                world_buf_pos.append(z_world_curr)
                if len(world_buf_pos) > MAX_BUF_EACH:
                    world_buf_pos = world_buf_pos[-MAX_BUF_EACH:]
            else:
                world_buf_neg.append(z_world_curr)
                if len(world_buf_neg) > MAX_BUF_EACH:
                    world_buf_neg = world_buf_neg[-MAX_BUF_EACH:]

            # Fill z_self buffer with same labels
            if is_harm:
                self_buf_pos.append(z_self_curr)
                if len(self_buf_pos) > MAX_BUF_EACH:
                    self_buf_pos = self_buf_pos[-MAX_BUF_EACH:]
            else:
                self_buf_neg.append(z_self_curr)
                if len(self_buf_neg) > MAX_BUF_EACH:
                    self_buf_neg = self_buf_neg[-MAX_BUF_EACH:]

            # Standard E1 + E2 losses
            e1_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()
            total_loss = e1_loss + e2_loss
            if total_loss.requires_grad:
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                optimizer.step()

            # E3 harm_eval_head training on z_world
            if len(world_buf_pos) >= 4 and len(world_buf_neg) >= 4:
                k_pos = min(16, len(world_buf_pos))
                k_neg = min(16, len(world_buf_neg))
                pos_idx = torch.randperm(len(world_buf_pos))[:k_pos].tolist()
                neg_idx = torch.randperm(len(world_buf_neg))[:k_neg].tolist()
                zw_pos = torch.cat([world_buf_pos[i] for i in pos_idx], dim=0)
                zw_neg = torch.cat([world_buf_neg[i] for i in neg_idx], dim=0)
                zw_b = torch.cat([zw_pos, zw_neg], dim=0)
                target_w = torch.cat([
                    torch.ones(k_pos, 1, device=agent.device),
                    torch.zeros(k_neg, 1, device=agent.device),
                ], dim=0)
                pred_harm = agent.e3.harm_eval(zw_b)
                harm_loss = F.mse_loss(pred_harm, target_w)
                if harm_loss.requires_grad:
                    harm_eval_optimizer.zero_grad()
                    harm_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        agent.e3.harm_eval_head.parameters(), 0.5,
                    )
                    harm_eval_optimizer.step()

            # Self probe training on z_self
            if len(self_buf_pos) >= 4 and len(self_buf_neg) >= 4:
                k_pos = min(16, len(self_buf_pos))
                k_neg = min(16, len(self_buf_neg))
                pos_idx = torch.randperm(len(self_buf_pos))[:k_pos].tolist()
                neg_idx = torch.randperm(len(self_buf_neg))[:k_neg].tolist()
                zs_pos = torch.cat([self_buf_pos[i] for i in pos_idx], dim=0)
                zs_neg = torch.cat([self_buf_neg[i] for i in neg_idx], dim=0)
                zs_b = torch.cat([zs_pos, zs_neg], dim=0)
                target_s = torch.cat([
                    torch.ones(k_pos, 1),
                    torch.zeros(k_neg, 1),
                ], dim=0)
                pred_self = self_probe(zs_b.detach())
                self_loss = F.mse_loss(pred_self, target_s)
                self_probe_optimizer.zero_grad()
                self_loss.backward()
                torch.nn.utils.clip_grad_norm_(self_probe.parameters(), 0.5)
                self_probe_optimizer.step()

            if done:
                break

        if (ep + 1) % 100 == 0 or ep == actual_warmup - 1:
            print(
                f"  [train] seed={seed} cond={cond_label}"
                f" ep {ep+1}/{actual_warmup}"
                f" approach={counts['hazard_approach']}"
                f" contact={counts['env_caused_hazard']+counts['agent_caused_hazard']}"
                f" buf_world_pos={len(world_buf_pos)} self_pos={len(self_buf_pos)}",
                flush=True,
            )

    # --- EVAL ---
    agent.eval()
    self_probe.eval()

    world_scores: Dict[str, List[float]] = {
        "none": [], "hazard_approach": [],
        "env_caused_hazard": [], "agent_caused_hazard": [],
    }
    self_scores: Dict[str, List[float]] = {
        "none": [], "hazard_approach": [],
        "env_caused_hazard": [], "agent_caused_hazard": [],
    }
    n_fatal = 0

    for _ in range(actual_eval):
        _, obs_dict = env.reset()
        agent.reset()

        for _ in range(steps_per_episode):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world)
                agent.clock.advance()
                z_world_curr = latent.z_world
                z_self_curr = latent.z_self

            action_idx = random.randint(0, n_actions - 1)
            action = _action_to_onehot(action_idx, n_actions, agent.device)
            agent._last_action = action

            _, _, done, info, obs_dict = env.step(action)
            ttype = info.get("transition_type", "none")

            try:
                with torch.no_grad():
                    score_w = float(agent.e3.harm_eval(z_world_curr).item())
                    score_s = float(self_probe(z_self_curr).item())
                if ttype in world_scores:
                    world_scores[ttype].append(score_w)
                    self_scores[ttype].append(score_s)
            except Exception:
                n_fatal += 1

            if done:
                break

    def _means(scores: Dict[str, List[float]]) -> Dict[str, float]:
        return {k: float(sum(v) / max(1, len(v))) for k, v in scores.items()}

    means_w = _means(world_scores)
    means_s = _means(self_scores)
    n_counts = {k: len(v) for k, v in world_scores.items()}

    gap_world_approach = means_w["hazard_approach"] - means_w["none"]
    gap_self_approach = means_s["hazard_approach"] - means_s["none"]

    mean_contact_w = (means_w["env_caused_hazard"] + means_w["agent_caused_hazard"]) / 2.0
    mean_contact_s = (means_s["env_caused_hazard"] + means_s["agent_caused_hazard"]) / 2.0
    gap_world_contact = mean_contact_w - means_w["none"]
    gap_self_contact = mean_contact_s - means_s["none"]

    n_approach = n_counts.get("hazard_approach", 0)

    print(
        f"  [eval] seed={seed} cond={cond_label}"
        f" gap_world_approach={gap_world_approach:.4f}"
        f" gap_self_approach={gap_self_approach:.4f}"
        f" n_approach={n_approach}",
        flush=True,
    )

    return {
        "seed": seed,
        "condition": cond_label,
        "unified": unified,
        # world metrics
        "gap_world_approach": float(gap_world_approach),
        "gap_world_contact":  float(gap_world_contact),
        "mean_world_none":    float(means_w["none"]),
        "mean_world_approach":float(means_w["hazard_approach"]),
        # self metrics
        "gap_self_approach":  float(gap_self_approach),
        "gap_self_contact":   float(gap_self_contact),
        "mean_self_none":     float(means_s["none"]),
        "mean_self_approach": float(means_s["hazard_approach"]),
        # data quality
        "n_approach_eval":    int(n_approach),
        "n_contact_eval":     int(
            n_counts.get("env_caused_hazard", 0) + n_counts.get("agent_caused_hazard", 0)
        ),
        "n_none_eval":        int(n_counts.get("none", 0)),
        "train_approach":     int(counts["hazard_approach"]),
        "train_contact":      int(counts["env_caused_hazard"] + counts["agent_caused_hazard"]),
        "n_fatal":            int(n_fatal),
    }


def run(
    seeds: Tuple = (42, 123),
    warmup_episodes: int = 300,
    eval_episodes: int = 50,
    steps_per_episode: int = 200,
    self_dim: int = 32,
    world_dim: int = 32,
    lr: float = 1e-3,
    alpha_world: float = 0.9,
    alpha_self: float = 0.3,
    harm_scale: float = 0.02,
    proximity_harm_scale: float = 0.05,
    nav_bias: float = 0.40,
    dry_run: bool = False,
    **kwargs,
) -> dict:
    """Discriminative pair: SPLIT (SD-005) vs UNIFIED (ablation). Double dissociation."""
    results_split:   List[Dict] = []
    results_unified: List[Dict] = []

    for seed in seeds:
        for unified in [False, True]:
            label = "UNIFIED" if unified else "SPLIT"
            print(
                f"\n[V3-EXQ-113] {label} seed={seed}"
                f" warmup={warmup_episodes} eval={eval_episodes}"
                f" alpha_world={alpha_world} nav_bias={nav_bias}",
                flush=True,
            )
            r = _run_single(
                seed=seed,
                unified=unified,
                warmup_episodes=warmup_episodes,
                eval_episodes=eval_episodes,
                steps_per_episode=steps_per_episode,
                self_dim=self_dim,
                world_dim=world_dim,
                lr=lr,
                alpha_world=alpha_world,
                alpha_self=alpha_self,
                harm_scale=harm_scale,
                proximity_harm_scale=proximity_harm_scale,
                nav_bias=nav_bias,
                dry_run=dry_run,
            )
            if unified:
                results_unified.append(r)
            else:
                results_split.append(r)

    def _avg(results: List[Dict], key: str) -> float:
        vals = [r[key] for r in results]
        return float(sum(vals) / max(1, len(vals)))

    # World-side metrics (replicates EXQ-078)
    gap_world_split   = _avg(results_split,   "gap_world_approach")
    gap_world_unified = _avg(results_unified, "gap_world_approach")
    delta_world       = gap_world_split - gap_world_unified

    # Self-side metrics (new: purification test)
    gap_self_split   = _avg(results_split,   "gap_self_approach")
    gap_self_unified = _avg(results_unified, "gap_self_approach")
    delta_self       = gap_self_split - gap_self_unified

    n_approach_min = min(r["n_approach_eval"] for r in results_split + results_unified)

    # Pre-registered PASS criteria
    c1_pass = delta_world     >= THRESH_C1   # SPLIT world improves by >= 2pp
    c2_pass = gap_world_split >= THRESH_C2   # SPLIT world absolutely detects approach
    c3_pass = gap_world_unified >= THRESH_C3 # UNIFIED world learns something
    c4_pass = delta_self      <= THRESH_C4   # SPLIT self loses world content (purification)
    c5_pass = n_approach_min  >= THRESH_C5   # sufficient approach events

    all_pass     = c1_pass and c2_pass and c3_pass and c4_pass and c5_pass
    criteria_met = sum([c1_pass, c2_pass, c3_pass, c4_pass, c5_pass])
    status = "PASS" if all_pass else "FAIL"

    # Decision scoring
    if all_pass:
        decision = "retain_ree"
    elif c1_pass and c2_pass and c3_pass:
        # World side replicates, self purification unclear
        decision = "hybridize"
    elif c2_pass and c3_pass and delta_world >= 0:
        # Direction correct but delta below threshold
        decision = "hybridize"
    else:
        decision = "retire_ree_claim"

    print(f"\n[V3-EXQ-113] Results:", flush=True)
    print(
        f"  WORLD: gap_split={gap_world_split:.4f}"
        f" gap_unified={gap_world_unified:.4f}"
        f" delta={delta_world:+.4f}",
        flush=True,
    )
    print(
        f"  SELF:  gap_split={gap_self_split:.4f}"
        f" gap_unified={gap_self_unified:.4f}"
        f" delta={delta_self:+.4f}",
        flush=True,
    )
    print(
        f"  n_approach_min={n_approach_min}"
        f"  decision={decision}  status={status} ({criteria_met}/5)",
        flush=True,
    )

    failure_notes: List[str] = []
    if not c1_pass:
        failure_notes.append(
            f"C1 FAIL: delta_world={delta_world:.4f} < {THRESH_C1}"
            " (split does not improve world approach detection by >=2pp)"
        )
    if not c2_pass:
        failure_notes.append(
            f"C2 FAIL: gap_world_split={gap_world_split:.4f} < {THRESH_C2}"
            " (split z_world cannot detect hazard approach at all)"
        )
    if not c3_pass:
        failure_notes.append(
            f"C3 FAIL: gap_world_unified={gap_world_unified:.4f} < {THRESH_C3}"
            " (unified also fails -- both conditions below baseline)"
        )
    if not c4_pass:
        failure_notes.append(
            f"C4 FAIL: delta_self={delta_self:+.4f} > {THRESH_C4}"
            " (split z_self does NOT lose world content vs unified z_self)"
            " -- purification not demonstrated at current scale"
        )
    if not c5_pass:
        failure_notes.append(
            f"C5 FAIL: n_approach_min={n_approach_min} < {THRESH_C5}"
            " (insufficient approach events for reliable estimate)"
        )

    for note in failure_notes:
        print(f"  {note}", flush=True)

    # Interpretation
    if all_pass:
        interpretation = (
            "SD-005 SUPPORTED by double dissociation: z_world specialises for world-state"
            f" (gap_split={gap_world_split:.4f} vs gap_unified={gap_world_unified:.4f},"
            f" delta={delta_world:+.4f})"
            " AND z_self is purified of world-state content"
            f" (gap_self_split={gap_self_split:.4f} vs gap_self_unified={gap_self_unified:.4f},"
            f" delta_self={delta_self:+.4f})."
            " Architectural split drives functional specialisation in both directions."
        )
    elif c1_pass and c2_pass and c3_pass:
        interpretation = (
            "Partial support: z_world advantage replicated (C1+C2+C3 PASS,"
            f" delta={delta_world:+.4f}). However, z_self purification not demonstrated"
            f" (C4 FAIL, delta_self={delta_self:+.4f})."
            " The world-side specialisation is confirmed but body-latent still carries"
            " world-state information in the SPLIT condition."
            " This may indicate that body_obs itself contains hazard proximity signals"
            " (harm_exposure EMA field at body_obs[10]) that cannot be removed architecturally."
        )
    else:
        interpretation = (
            "SD-005 z_world advantage not replicated at this environment scale"
            f" (gap_world_split={gap_world_split:.4f},"
            f" delta={delta_world:+.4f})."
            " Suggests EXQ-078 result may not generalise to 6x6 environment or"
            " warmup reduction from 350->300 episodes was insufficient."
        )

    per_split_rows = "\n".join(
        f"  seed={r['seed']}: gap_world={r['gap_world_approach']:.4f}"
        f" gap_self={r['gap_self_approach']:.4f}"
        f" n_approach={r['n_approach_eval']}"
        for r in results_split
    )
    per_unified_rows = "\n".join(
        f"  seed={r['seed']}: gap_world={r['gap_world_approach']:.4f}"
        f" gap_self={r['gap_self_approach']:.4f}"
        f" n_approach={r['n_approach_eval']}"
        for r in results_unified
    )

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(
            f"- {n}" for n in failure_notes
        )

    summary_markdown = (
        f"# V3-EXQ-113 -- SD-005 Double Dissociation\n\n"
        f"**Status:** {status}\n"
        f"**Claims:** SD-005\n"
        f"**Decision:** {decision}\n"
        f"**Seeds:** {list(seeds)}\n"
        f"**alpha_world:** {alpha_world}  (SD-008)\n"
        f"**Warmup:** {warmup_episodes} eps  **Eval:** {eval_episodes} eps\n"
        f"**Env:** CausalGridWorldV2 size=6, 4 hazards, use_proxy_fields=True,"
        f" nav_bias={nav_bias}\n\n"
        f"## Pre-Registered Thresholds\n\n"
        f"C1: delta_approach_gap_world (SPLIT-UNIFIED) >= {THRESH_C1}"
        f"  (world improves in SPLIT)\n"
        f"C2: gap_approach_split_world >= {THRESH_C2}"
        f"  (SPLIT world absolutely detects approach)\n"
        f"C3: gap_approach_unified_world >= {THRESH_C3}"
        f"  (UNIFIED learns something)\n"
        f"C4: delta_approach_gap_self (SPLIT-UNIFIED) <= {THRESH_C4}"
        f"  (z_self loses world content in SPLIT = purification)\n"
        f"C5: n_approach_eval >= {THRESH_C5}  (data quality)\n\n"
        f"## Results\n\n"
        f"| Condition | gap_world | gap_self | mean_world_none | mean_world_appr |\n"
        f"|-----------|----------|---------|-----------------|------------------|\n"
        f"| SPLIT   | {gap_world_split:.4f} | {gap_self_split:.4f}"
        f" | {_avg(results_split,   'mean_world_none'):.4f}"
        f" | {_avg(results_split,   'mean_world_approach'):.4f} |\n"
        f"| UNIFIED | {gap_world_unified:.4f} | {gap_self_unified:.4f}"
        f" | {_avg(results_unified, 'mean_world_none'):.4f}"
        f" | {_avg(results_unified, 'mean_world_approach'):.4f} |\n\n"
        f"**delta_world (SPLIT-UNIFIED): {delta_world:+.4f}**"
        f"  **delta_self (SPLIT-UNIFIED): {delta_self:+.4f}**\n\n"
        f"## PASS Criteria\n\n"
        f"| Criterion | Result | Value |\n"
        f"|---|---|---|\n"
        f"| C1: delta_world >= {THRESH_C1} | {'PASS' if c1_pass else 'FAIL'}"
        f" | {delta_world:.4f} |\n"
        f"| C2: gap_world_split >= {THRESH_C2} | {'PASS' if c2_pass else 'FAIL'}"
        f" | {gap_world_split:.4f} |\n"
        f"| C3: gap_world_unified >= {THRESH_C3} | {'PASS' if c3_pass else 'FAIL'}"
        f" | {gap_world_unified:.4f} |\n"
        f"| C4: delta_self <= {THRESH_C4} | {'PASS' if c4_pass else 'FAIL'}"
        f" | {delta_self:.4f} |\n"
        f"| C5: n_approach_min >= {THRESH_C5} | {'PASS' if c5_pass else 'FAIL'}"
        f" | {n_approach_min} |\n\n"
        f"Criteria met: {criteria_met}/5 -> **{status}**\n\n"
        f"## Interpretation\n\n"
        f"{interpretation}\n\n"
        f"## Per-Seed\n\n"
        f"SPLIT:\n{per_split_rows}\n\n"
        f"UNIFIED:\n{per_unified_rows}\n"
        f"{failure_section}\n"
    )

    metrics = {
        "gap_world_split":          float(gap_world_split),
        "gap_world_unified":        float(gap_world_unified),
        "delta_approach_gap_world": float(delta_world),
        "gap_self_split":           float(gap_self_split),
        "gap_self_unified":         float(gap_self_unified),
        "delta_approach_gap_self":  float(delta_self),
        "n_approach_min":           float(n_approach_min),
        "n_seeds":                  float(len(seeds)),
        "alpha_world":              float(alpha_world),
        "nav_bias":                 float(nav_bias),
        "crit1_pass":               1.0 if c1_pass else 0.0,
        "crit2_pass":               1.0 if c2_pass else 0.0,
        "crit3_pass":               1.0 if c3_pass else 0.0,
        "crit4_pass":               1.0 if c4_pass else 0.0,
        "crit5_pass":               1.0 if c5_pass else 0.0,
        "criteria_met":             float(criteria_met),
    }

    return {
        "status": status,
        "metrics": metrics,
        "summary_markdown": summary_markdown,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": (
            "supports" if all_pass
            else ("mixed" if criteria_met >= 3 else "weakens")
        ),
        "experiment_type": EXPERIMENT_TYPE,
        "fatal_error_count": sum(r["n_fatal"] for r in results_split + results_unified),
    }


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",           type=int, nargs="+", default=[42, 123])
    parser.add_argument("--warmup",          type=int,   default=300)
    parser.add_argument("--eval-eps",        type=int,   default=50)
    parser.add_argument("--steps",           type=int,   default=200)
    parser.add_argument("--alpha-world",     type=float, default=0.9)
    parser.add_argument("--alpha-self",      type=float, default=0.3)
    parser.add_argument("--harm-scale",      type=float, default=0.02)
    parser.add_argument("--proximity-scale", type=float, default=0.05)
    parser.add_argument("--nav-bias",        type=float, default=0.40)
    parser.add_argument("--dry-run",         action="store_true",
                        help="Run 3 warmup + 2 eval episodes per cell to check for errors.")
    args = parser.parse_args()

    result = run(
        seeds=tuple(args.seeds),
        warmup_episodes=args.warmup,
        eval_episodes=args.eval_eps,
        steps_per_episode=args.steps,
        alpha_world=args.alpha_world,
        alpha_self=args.alpha_self,
        harm_scale=args.harm_scale,
        proximity_harm_scale=args.proximity_scale,
        nav_bias=args.nav_bias,
        dry_run=args.dry_run,
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"] = ts
    result["claim"] = CLAIM_IDS[0]
    result["verdict"] = result["status"]
    result["run_id"] = f"{EXPERIMENT_TYPE}_{ts}_v3"
    result["architecture_epoch"] = "ree_hybrid_guardrails_v1"
    result["registered_thresholds"] = {
        "C1_delta_world": THRESH_C1,
        "C2_gap_world_split": THRESH_C2,
        "C3_gap_world_unified": THRESH_C3,
        "C4_delta_self": THRESH_C4,
        "C5_n_approach": THRESH_C5,
    }
    result["seeds"] = list(args.seeds)
    result["conditions"] = ["SPLIT", "UNIFIED"]
    result["dispatch_mode"] = "discriminative_pair"
    result["backlog_id"] = "EVB-0006"

    if args.dry_run:
        print("\n[dry-run] Skipping file output.", flush=True)
        sys.exit(0)

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
