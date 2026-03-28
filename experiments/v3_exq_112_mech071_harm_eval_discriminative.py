"""
V3-EXQ-112 -- MECH-071 harm_eval Trained vs Random Discriminative Pair

Claims: MECH-071
Proposal: EXP-0005 / EVB-0005

MECH-071 asserts that E3.harm_eval learns a graded danger model from z_world:
  harm_eval_contact > harm_eval_approach > harm_eval_none.
E3 learns to assign higher harm scores to z_world states near hazards (approach)
and at contact, compared to safe locomotion (none).

Why prior pairs were insufficient (EVB-0005 still proposed after EXQ-079):
  EXQ-079 tested the SD-008 axis (alpha_world=0.9 vs 0.3). The SMOOTH_WORLD
  ablation showed calibration_gap=0.1123 -- unexpectedly high -- creating an
  ambiguous result (decision: hybridize). The axis being tested was "does
  alpha_world enable MECH-071?" not "does training E3.harm_eval produce MECH-071?".
  EXQ-026/029 confirmed the calibration gradient exists, but with single seeds and
  no matched control. EVB-0005 requires a clean discriminative pair.

Discriminative pair (within-run comparison at eval time):
  TRAINED  -- E3.harm_eval_head trained for full training phase (standard)
  RANDOM   -- fresh harm_eval_head, same architecture, random weights, never trained

Both heads receive the SAME z_world tensors from the SAME eval episodes.
This is a perfectly matched within-run comparison -- z_world representations
are identical; only the head differs. Any calibration gradient in TRAINED but
not RANDOM is attributable to training (not z_world structure alone).

Environment: CausalGridWorldV2 6x6, 4 hazards, use_proxy_fields=True (ARC-024).
Proxy fields provide z_world with proximity gradient structure. nav_bias=0.40
toward-hazard ensures sufficient approach and contact events.

Mechanism under test:
  Training E3.harm_eval_head on balanced (harm, no-harm) z_world samples teaches
  the head to discriminate hazard-proximate world states from safe states. The key
  prediction: calibration_gap_trained >> calibration_gap_random because z_world
  contains hazard proximity structure (ARC-024) but an untrained head cannot extract
  it. The graded model (none < approach < contact) emerges because proximity gradient
  is continuous in z_world (approach states intermediate between none and contact).

Pre-registered PASS criteria (ALL required, both seeds):
  C1: calibration_gap_trained    >= 0.08  (trained head detects hazard gradient)
  C2: calibration_gap_random     <= 0.04  (random head is near-flat)
  C3: delta_calibration_gap      >= 0.05  (training adds substantive signal)
  C4: harm_eval_contact > harm_eval_approach > harm_eval_none (trained, monotonic)
  C5: n_approach >= 10 AND n_agent_hazard >= 5 (sufficient events)

PASS = C1+C2+C3+C4+C5 all satisfied on >= pair_pass_count seeds.

Decision scoring:
  retain_ree:       PASS (C1+C2+C3+C4+C5, both seeds)
  hybridize:        C1/C4 pass but C2 fails (random also calibrated -- z_world
                    structure alone partially enables calibration; training amplifies)
  retire_ree_claim: C1 fails (trained head cannot learn harm gradient)
"""

import sys
import random
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_112_mech071_harm_eval_discriminative"
CLAIM_IDS = ["MECH-071"]


def _action_to_onehot(idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, idx] = 1.0
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


def _make_random_harm_head(world_dim: int, hidden_dim: int) -> nn.Sequential:
    """Create a randomly-initialised harm_eval head with the same architecture
    as E3.harm_eval_head. Never trained -- serves as the RANDOM baseline."""
    head = nn.Sequential(
        nn.Linear(world_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1),
        nn.Sigmoid(),
    )
    # Default PyTorch init (kaiming_uniform) -- intentionally never updated
    return head


def _run_single(
    seed: int,
    train_episodes: int,
    eval_episodes: int,
    steps_per_episode: int,
    self_dim: int,
    world_dim: int,
    hidden_dim: int,
    lr: float,
    alpha_world: float,
    alpha_self: float,
    nav_bias: float,
) -> Dict:
    """Train one agent with E3.harm_eval, then eval TRAINED vs RANDOM heads."""
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
    n_actions = env.action_dim

    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=n_actions,
        self_dim=self_dim,
        world_dim=world_dim,
        alpha_world=alpha_world,
        alpha_self=alpha_self,
        reafference_action_dim=0,  # isolate harm_eval mechanism; no SD-007
    )
    agent = REEAgent(config)

    # Separate optimizers: standard params + harm_eval_head
    standard_params = [
        p for n, p in agent.named_parameters()
        if "harm_eval_head" not in n
    ]
    harm_eval_params = list(agent.e3.harm_eval_head.parameters())
    optimizer = optim.Adam(standard_params, lr=lr)
    harm_eval_optimizer = optim.Adam(harm_eval_params, lr=1e-4)

    # Balanced harm_eval training buffers
    harm_buf_pos: List[torch.Tensor] = []
    harm_buf_neg: List[torch.Tensor] = []
    MAX_BUF_EACH = 2000

    train_counts: Dict[str, int] = {
        "hazard_approach": 0, "agent_caused_hazard": 0,
        "env_caused_hazard": 0, "none": 0,
    }

    # ---- TRAINING -----------------------------------------------------------
    agent.train()
    for ep in range(train_episodes):
        _, obs_dict = env.reset()
        agent.reset()

        for _ in range(steps_per_episode):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            latent = agent.sense(obs_body, obs_world)
            agent.clock.advance()

            z_world_curr = latent.z_world.detach()

            # Biased navigation: nav_bias toward-hazard, else random
            if random.random() < nav_bias:
                action_idx = _hazard_approach_action(env, n_actions)
            else:
                action_idx = random.randint(0, n_actions - 1)
            action = _action_to_onehot(action_idx, n_actions, agent.device)
            agent._last_action = action

            _, harm_signal, done, info, obs_dict = env.step(action)
            ttype = info.get("transition_type", "none")
            if ttype in train_counts:
                train_counts[ttype] += 1

            # Harm_eval buffer: harm_signal < 0 at hazard steps
            is_harm = float(harm_signal) < 0
            if is_harm:
                harm_buf_pos.append(z_world_curr)
                if len(harm_buf_pos) > MAX_BUF_EACH:
                    harm_buf_pos = harm_buf_pos[-MAX_BUF_EACH:]
            else:
                harm_buf_neg.append(z_world_curr)
                if len(harm_buf_neg) > MAX_BUF_EACH:
                    harm_buf_neg = harm_buf_neg[-MAX_BUF_EACH:]

            # Standard E1 + E2 losses
            e1_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()
            total_loss = e1_loss + e2_loss
            if total_loss.requires_grad:
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                optimizer.step()

            # E3 harm_eval balanced training (16 pos + 16 neg per step)
            if len(harm_buf_pos) >= 4 and len(harm_buf_neg) >= 4:
                k_pos = min(16, len(harm_buf_pos))
                k_neg = min(16, len(harm_buf_neg))
                pos_idx = torch.randperm(len(harm_buf_pos))[:k_pos].tolist()
                neg_idx = torch.randperm(len(harm_buf_neg))[:k_neg].tolist()
                zw_pos = torch.cat([harm_buf_pos[i] for i in pos_idx], dim=0)
                zw_neg = torch.cat([harm_buf_neg[i] for i in neg_idx], dim=0)
                zw_b = torch.cat([zw_pos, zw_neg], dim=0)
                target = torch.cat([
                    torch.ones(k_pos, 1, device=agent.device),
                    torch.zeros(k_neg, 1, device=agent.device),
                ], dim=0)
                pred = agent.e3.harm_eval(zw_b)
                harm_loss = F.mse_loss(pred, target)
                if harm_loss.requires_grad:
                    harm_eval_optimizer.zero_grad()
                    harm_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        agent.e3.harm_eval_head.parameters(), 0.5
                    )
                    harm_eval_optimizer.step()

            if done:
                break

        if (ep + 1) % 50 == 0 or ep == train_episodes - 1:
            print(
                f"  [train] seed={seed} ep {ep+1}/{train_episodes}"
                f" approach={train_counts['hazard_approach']}"
                f" contact={train_counts['agent_caused_hazard']}"
                f" buf_pos={len(harm_buf_pos)} buf_neg={len(harm_buf_neg)}",
                flush=True,
            )

    # ---- Create RANDOM head (same architecture, never trained) --------------
    random_head = _make_random_harm_head(world_dim, hidden_dim).to(agent.device)
    random_head.eval()

    # ---- EVAL: collect z_world states, score with TRAINED and RANDOM --------
    agent.eval()

    zw_by_type: Dict[str, List[torch.Tensor]] = {
        "none": [], "hazard_approach": [],
        "agent_caused_hazard": [], "env_caused_hazard": [],
    }
    n_fatal = 0

    for _ in range(eval_episodes):
        _, obs_dict = env.reset()
        agent.reset()

        for _ in range(steps_per_episode):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world)
                agent.clock.advance()
                z_world_curr = latent.z_world  # [1, world_dim]

            # Biased navigation during eval too (ensure enough events)
            if random.random() < nav_bias:
                action_idx = _hazard_approach_action(env, n_actions)
            else:
                action_idx = random.randint(0, n_actions - 1)
            action = _action_to_onehot(action_idx, n_actions, agent.device)
            agent._last_action = action

            _, _, done, info, obs_dict = env.step(action)
            ttype = info.get("transition_type", "none")

            try:
                if ttype in zw_by_type:
                    zw_by_type[ttype].append(z_world_curr.detach())
            except Exception:
                n_fatal += 1

            if done:
                break

    # Compute mean scores per type for TRAINED and RANDOM heads
    def _mean_score(head: nn.Module, zw_list: List[torch.Tensor]) -> float:
        if not zw_list:
            return 0.0
        batch = torch.cat(zw_list, dim=0)  # [N, world_dim]
        with torch.no_grad():
            scores = head(batch).squeeze(-1)  # [N]
        return float(scores.mean().item())

    trained_head = agent.e3.harm_eval_head

    scores_trained: Dict[str, float] = {
        k: _mean_score(trained_head, v) for k, v in zw_by_type.items()
    }
    scores_random: Dict[str, float] = {
        k: _mean_score(random_head, v) for k, v in zw_by_type.items()
    }
    n_counts = {k: len(v) for k, v in zw_by_type.items()}

    # All eval z_world for std calculation (trained head)
    all_zw = []
    for v in zw_by_type.values():
        all_zw.extend(v)
    if all_zw:
        all_batch = torch.cat(all_zw, dim=0)
        with torch.no_grad():
            all_scores_t = trained_head(all_batch).squeeze(-1)
        harm_pred_std_trained = float(all_scores_t.std().item())
        all_scores_r = random_head(all_batch).squeeze(-1)
        harm_pred_std_random = float(all_scores_r.std().item())
    else:
        harm_pred_std_trained = 0.0
        harm_pred_std_random = 0.0

    # calibration_gap = mean at agent_caused_hazard - mean at none
    calib_gap_trained = (
        scores_trained["agent_caused_hazard"] - scores_trained["none"]
    )
    calib_gap_random = (
        scores_random["agent_caused_hazard"] - scores_random["none"]
    )
    delta_calibration_gap = calib_gap_trained - calib_gap_random

    # Graded model check: contact > approach > none (trained)
    c4_graded = (
        scores_trained["agent_caused_hazard"] > scores_trained["hazard_approach"]
        and scores_trained["hazard_approach"] > scores_trained["none"]
    )

    print(
        f"  [eval] seed={seed} TRAINED:"
        f" none={scores_trained['none']:.4f}"
        f" approach={scores_trained['hazard_approach']:.4f}"
        f" agent_contact={scores_trained['agent_caused_hazard']:.4f}",
        flush=True,
    )
    print(
        f"         RANDOM:"
        f" none={scores_random['none']:.4f}"
        f" approach={scores_random['hazard_approach']:.4f}"
        f" agent_contact={scores_random['agent_caused_hazard']:.4f}",
        flush=True,
    )
    print(
        f"         calib_gap_trained={calib_gap_trained:.4f}"
        f" calib_gap_random={calib_gap_random:.4f}"
        f" delta={delta_calibration_gap:.4f}"
        f" graded={c4_graded}"
        f" n={n_counts}",
        flush=True,
    )

    return {
        "seed": seed,
        "calib_gap_trained": float(calib_gap_trained),
        "calib_gap_random": float(calib_gap_random),
        "delta_calibration_gap": float(delta_calibration_gap),
        "harm_pred_std_trained": float(harm_pred_std_trained),
        "harm_pred_std_random": float(harm_pred_std_random),
        "c4_graded": bool(c4_graded),
        "scores_trained_none": float(scores_trained["none"]),
        "scores_trained_approach": float(scores_trained["hazard_approach"]),
        "scores_trained_agent_contact": float(scores_trained["agent_caused_hazard"]),
        "scores_trained_env_contact": float(scores_trained["env_caused_hazard"]),
        "scores_random_none": float(scores_random["none"]),
        "scores_random_approach": float(scores_random["hazard_approach"]),
        "scores_random_agent_contact": float(scores_random["agent_caused_hazard"]),
        "n_none": int(n_counts["none"]),
        "n_approach": int(n_counts["hazard_approach"]),
        "n_agent_hazard": int(n_counts["agent_caused_hazard"]),
        "n_env_hazard": int(n_counts["env_caused_hazard"]),
        "train_approach": int(train_counts["hazard_approach"]),
        "train_agent_hazard": int(train_counts["agent_caused_hazard"]),
        "buf_pos": int(len(harm_buf_pos)),
        "buf_neg": int(len(harm_buf_neg)),
        "n_fatal": int(n_fatal),
    }


def run(
    seeds: Tuple = (42, 123),
    train_episodes: int = 200,
    eval_episodes: int = 60,
    steps_per_episode: int = 200,
    self_dim: int = 32,
    world_dim: int = 32,
    hidden_dim: int = 64,
    lr: float = 1e-3,
    alpha_world: float = 0.9,
    alpha_self: float = 0.3,
    nav_bias: float = 0.40,
    pair_pass_count: int = 2,
    dry_run: bool = False,
    **kwargs,
) -> dict:
    """
    Discriminative pair: TRAINED vs RANDOM E3.harm_eval_head.
    Runs each seed, aggregates PASS criteria.
    """
    if dry_run:
        print("[EXQ-112] Dry-run: checking environment + config...", flush=True)
        seed = seeds[0]
        torch.manual_seed(seed)
        env = CausalGridWorldV2(
            seed=seed, size=6, num_hazards=4, num_resources=2,
            hazard_harm=0.02, env_drift_interval=10, env_drift_prob=0.1,
            proximity_harm_scale=0.05, proximity_benefit_scale=0.03,
            proximity_approach_threshold=0.15, hazard_field_decay=0.5,
            use_proxy_fields=True,
        )
        config = REEConfig.from_dims(
            body_obs_dim=env.body_obs_dim, world_obs_dim=env.world_obs_dim,
            action_dim=env.action_dim, self_dim=self_dim, world_dim=world_dim,
            alpha_world=alpha_world, alpha_self=alpha_self, reafference_action_dim=0,
        )
        agent = REEAgent(config)
        random_head = _make_random_harm_head(world_dim, hidden_dim)
        _, obs_dict = env.reset()
        agent.reset()
        obs_body = obs_dict["body_state"]
        obs_world = obs_dict["world_state"]
        latent = agent.sense(obs_body, obs_world)
        agent.clock.advance()
        zw = latent.z_world.detach()
        with torch.no_grad():
            score_t = float(agent.e3.harm_eval(zw).item())
            score_r = float(random_head(zw).item())
        n_approach = 0
        n_contact = 0
        for _ in range(3):
            _, obs_dict = env.reset()
            for _ in range(50):
                obs_body = obs_dict["body_state"]
                obs_world = obs_dict["world_state"]
                agent.sense(obs_body, obs_world)
                agent.clock.advance()
                if random.random() < nav_bias:
                    ai = _hazard_approach_action(env, env.action_dim)
                else:
                    ai = random.randint(0, env.action_dim - 1)
                act = _action_to_onehot(ai, env.action_dim, agent.device)
                agent._last_action = act
                _, _, done, info, obs_dict = env.step(act)
                ttype = info.get("transition_type", "none")
                if ttype == "hazard_approach":
                    n_approach += 1
                elif ttype in ("agent_caused_hazard", "env_caused_hazard"):
                    n_contact += 1
                if done:
                    break
        print(
            f"[EXQ-112] Dry-run OK: trained_score={score_t:.4f}"
            f" random_score={score_r:.4f}"
            f" n_approach={n_approach} n_contact={n_contact}",
            flush=True,
        )
        print("[EXQ-112] Metrics: calib_gap_trained, calib_gap_random,"
              " delta_calibration_gap, c4_graded_pass, pair_pass_count,"
              " pair_seeds_passed", flush=True)
        return {}

    per_seed: List[Dict] = []
    for seed in seeds:
        print(
            f"\n[V3-EXQ-112] seed={seed} train={train_episodes} eps"
            f" eval={eval_episodes} eps nav_bias={nav_bias}",
            flush=True,
        )
        r = _run_single(
            seed=seed,
            train_episodes=train_episodes,
            eval_episodes=eval_episodes,
            steps_per_episode=steps_per_episode,
            self_dim=self_dim,
            world_dim=world_dim,
            hidden_dim=hidden_dim,
            lr=lr,
            alpha_world=alpha_world,
            alpha_self=alpha_self,
            nav_bias=nav_bias,
        )
        per_seed.append(r)

    def _avg(key: str) -> float:
        return float(sum(r[key] for r in per_seed) / max(1, len(per_seed)))

    calib_gap_trained = _avg("calib_gap_trained")
    calib_gap_random = _avg("calib_gap_random")
    delta_calibration_gap = _avg("delta_calibration_gap")
    harm_pred_std_trained = _avg("harm_pred_std_trained")
    n_approach_min = min(r["n_approach"] for r in per_seed)
    n_agent_min = min(r["n_agent_hazard"] for r in per_seed)

    # PASS criteria (pre-registered)
    c1_pass = calib_gap_trained >= 0.08
    c2_pass = calib_gap_random <= 0.04
    c3_pass = delta_calibration_gap >= 0.05
    c4_pass = all(r["c4_graded"] for r in per_seed)
    c5_pass = n_approach_min >= 10 and n_agent_min >= 5

    all_pass = c1_pass and c2_pass and c3_pass and c4_pass and c5_pass
    criteria_met = sum([c1_pass, c2_pass, c3_pass, c4_pass, c5_pass])

    # Per-seed pass/fail
    def _seed_pass(r: Dict) -> bool:
        return (
            r["calib_gap_trained"] >= 0.08
            and r["calib_gap_random"] <= 0.04
            and r["delta_calibration_gap"] >= 0.05
            and r["c4_graded"]
            and r["n_approach"] >= 10
            and r["n_agent_hazard"] >= 5
        )

    seeds_passed = sum(1 for r in per_seed if _seed_pass(r))
    status = "PASS" if seeds_passed >= pair_pass_count else "FAIL"

    # Decision scoring
    if status == "PASS":
        decision = "retain_ree"
    elif c1_pass and c4_pass and not c2_pass:
        decision = "hybridize"
    elif c1_pass and c4_pass:
        decision = "hybridize"
    else:
        decision = "retire_ree_claim"

    print(f"\n[V3-EXQ-112] Final results:", flush=True)
    print(
        f"  calib_gap_trained={calib_gap_trained:.4f}"
        f"  calib_gap_random={calib_gap_random:.4f}"
        f"  delta={delta_calibration_gap:.4f}",
        flush=True,
    )
    print(
        f"  c4_graded_pass={c4_pass}"
        f"  n_approach_min={n_approach_min}"
        f"  n_agent_min={n_agent_min}",
        flush=True,
    )
    print(
        f"  seeds_passed={seeds_passed}/{len(seeds)}"
        f"  criteria_met={criteria_met}/5"
        f"  status={status}  decision={decision}",
        flush=True,
    )

    failure_notes: List[str] = []
    if not c1_pass:
        failure_notes.append(
            f"C1 FAIL: calib_gap_trained={calib_gap_trained:.4f} < 0.08"
            " (trained head cannot detect harm gradient -- MECH-071 not supported)"
        )
    if not c2_pass:
        failure_notes.append(
            f"C2 FAIL: calib_gap_random={calib_gap_random:.4f} > 0.04"
            " (random head also calibrated -- z_world structure alone partially enables;"
            " training amplifies but is not necessary for basic discrimination)"
        )
    if not c3_pass:
        failure_notes.append(
            f"C3 FAIL: delta={delta_calibration_gap:.4f} < 0.05"
            " (training adds marginal signal above random baseline)"
        )
    if not c4_pass:
        failure_notes.append(
            "C4 FAIL: graded model not monotonic (contact > approach > none)"
            " in all seeds -- harm gradient not graded in z_world"
        )
    if not c5_pass:
        failure_notes.append(
            f"C5 FAIL: insufficient events"
            f" (n_approach_min={n_approach_min} < 10"
            f" or n_agent_min={n_agent_min} < 5)"
        )
    for note in failure_notes:
        print(f"  {note}", flush=True)

    per_seed_rows = "\n".join(
        f"  seed={r['seed']}:"
        f" calib_trained={r['calib_gap_trained']:.4f}"
        f" calib_random={r['calib_gap_random']:.4f}"
        f" delta={r['delta_calibration_gap']:.4f}"
        f" graded={r['c4_graded']}"
        f" n_approach={r['n_approach']}"
        f" n_agent={r['n_agent_hazard']}"
        for r in per_seed
    )

    if status == "PASS":
        interpretation = (
            "MECH-071 SUPPORTED: E3.harm_eval training produces a graded danger"
            " model (contact > approach > none) in z_world. The trained head"
            f" shows calib_gap={calib_gap_trained:.4f} vs random baseline"
            f" {calib_gap_random:.4f} (delta={delta_calibration_gap:.4f})."
            " z_world with ARC-024 proxy fields encodes proximity structure;"
            " supervised training extracts this to produce the MECH-071 gradient."
        )
    elif decision == "hybridize":
        interpretation = (
            "MECH-071 PARTIAL: Trained head shows calibration gradient"
            f" (calib_gap={calib_gap_trained:.4f}) but random head is also"
            f" above threshold ({calib_gap_random:.4f}). z_world structure"
            " (ARC-024 proxy fields) provides partial calibration signal without"
            " training. MECH-071 holds but the mechanism involves both z_world"
            " structure and learned harm_eval. Hybridize recommendation."
        )
    else:
        interpretation = (
            f"MECH-071 NOT SUPPORTED: Trained head calibration_gap={calib_gap_trained:.4f}"
            " fails to meet threshold. Either z_world does not encode sufficient"
            " proximity structure for E3 to learn harm gradients, or the training"
            " procedure is insufficient. Consider whether ARC-024 proxy fields"
            " are providing the expected gradient structure."
        )

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(
            f"- {n}" for n in failure_notes
        )

    summary_markdown = (
        f"# V3-EXQ-112 -- MECH-071 harm_eval Trained vs Random Discriminative Pair\n\n"
        f"**Status:** {status}\n"
        f"**Claims:** MECH-071\n"
        f"**Decision:** {decision}\n"
        f"**Seeds:** {list(seeds)}\n"
        f"**Grid:** 6x6, 4 hazards, use_proxy_fields=True (ARC-024)\n"
        f"**Train:** {train_episodes} eps  **Eval:** {eval_episodes} eps"
        f"  **nav_bias:** {nav_bias}\n\n"
        f"## Pre-Registered Thresholds\n\n"
        f"C1: calib_gap_trained >= 0.08\n"
        f"C2: calib_gap_random  <= 0.04\n"
        f"C3: delta_calibration_gap >= 0.05\n"
        f"C4: harm_eval_contact > harm_eval_approach > harm_eval_none (trained)\n"
        f"C5: n_approach >= 10 AND n_agent_hazard >= 5\n\n"
        f"## Results\n\n"
        f"| Head | calib_gap (agent-none) | harm_pred_std |\n"
        f"|------|------------------------|---------------|\n"
        f"| TRAINED | {calib_gap_trained:.4f} | {harm_pred_std_trained:.4f} |\n"
        f"| RANDOM  | {calib_gap_random:.4f} | {_avg('harm_pred_std_random'):.4f} |\n\n"
        f"**delta_calibration_gap: {delta_calibration_gap:+.4f}**\n\n"
        f"## PASS Criteria\n\n"
        f"| Criterion | Result | Value |\n"
        f"|---|---|---|\n"
        f"| C1: calib_gap_trained >= 0.08 | {'PASS' if c1_pass else 'FAIL'}"
        f" | {calib_gap_trained:.4f} |\n"
        f"| C2: calib_gap_random <= 0.04  | {'PASS' if c2_pass else 'FAIL'}"
        f" | {calib_gap_random:.4f} |\n"
        f"| C3: delta >= 0.05             | {'PASS' if c3_pass else 'FAIL'}"
        f" | {delta_calibration_gap:.4f} |\n"
        f"| C4: graded model (contact > approach > none) | {'PASS' if c4_pass else 'FAIL'}"
        f" | {c4_pass} |\n"
        f"| C5: n_approach>=10, n_agent>=5 | {'PASS' if c5_pass else 'FAIL'}"
        f" | approach={n_approach_min}, agent={n_agent_min} |\n\n"
        f"seeds passed: {seeds_passed}/{len(seeds)}  criteria met: {criteria_met}/5"
        f"  -> **{status}**\n\n"
        f"## Interpretation\n\n"
        f"{interpretation}\n\n"
        f"## Per-Seed\n\n"
        f"{per_seed_rows}\n"
        f"{failure_section}\n"
    )

    metrics = {
        "calib_gap_trained": float(calib_gap_trained),
        "calib_gap_random": float(calib_gap_random),
        "delta_calibration_gap": float(delta_calibration_gap),
        "harm_pred_std_trained": float(harm_pred_std_trained),
        "harm_pred_std_random": float(_avg("harm_pred_std_random")),
        "c4_graded_pass": 1.0 if c4_pass else 0.0,
        "n_approach_min": float(n_approach_min),
        "n_agent_min": float(n_agent_min),
        "seeds_passed": float(seeds_passed),
        "pair_pass_count": float(pair_pass_count),
        "n_seeds": float(len(seeds)),
        "crit1_pass": 1.0 if c1_pass else 0.0,
        "crit2_pass": 1.0 if c2_pass else 0.0,
        "crit3_pass": 1.0 if c3_pass else 0.0,
        "crit4_pass": 1.0 if c4_pass else 0.0,
        "crit5_pass": 1.0 if c5_pass else 0.0,
        "criteria_met": float(criteria_met),
    }

    return {
        "status": status,
        "metrics": metrics,
        "summary_markdown": summary_markdown,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": (
            "supports" if status == "PASS"
            else ("mixed" if criteria_met >= 3 else "weakens")
        ),
        "experiment_type": EXPERIMENT_TYPE,
        "fatal_error_count": sum(r["n_fatal"] for r in per_seed),
    }


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",       type=int,   nargs="+", default=[42, 123])
    parser.add_argument("--train",       type=int,   default=200)
    parser.add_argument("--eval-eps",    type=int,   default=60)
    parser.add_argument("--steps",       type=int,   default=200)
    parser.add_argument("--world-dim",   type=int,   default=32)
    parser.add_argument("--hidden-dim",  type=int,   default=64)
    parser.add_argument("--nav-bias",    type=float, default=0.40)
    parser.add_argument("--alpha-world", type=float, default=0.9)
    parser.add_argument("--dry-run",     action="store_true")
    args = parser.parse_args()

    result = run(
        seeds=tuple(args.seeds),
        train_episodes=args.train,
        eval_episodes=args.eval_eps,
        steps_per_episode=args.steps,
        world_dim=args.world_dim,
        hidden_dim=args.hidden_dim,
        nav_bias=args.nav_bias,
        alpha_world=args.alpha_world,
        dry_run=args.dry_run,
    )

    if args.dry_run:
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
