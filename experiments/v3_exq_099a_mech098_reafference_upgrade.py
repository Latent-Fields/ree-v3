"""
V3-EXQ-099a -- MECH-098 Reafference Upgrade Gate + Discriminative Pair (collection filter fix)

Claims: MECH-098
Supersedes: V3-EXQ-099

MECH-098 asserts that z_world corrected by reafference cancellation (SD-007) isolates
genuine external world-state changes from self-motion perspective shifts.

Root cause of prior FAILs (EXQ-069, EXQ-082):
  The current ReafferencePredictor (2-layer MLP, hidden_dim=64) achieves R2=0.339 on
  held-out data. At this quality the correction subtracts real signal along with noise,
  degrading z_world. This is an ENGINEERING failure, not a conceptual one.
  R2 target >= 0.70 before re-testing MECH-098.
  Upgrade path: 3-layer MLP, hidden_dim=128.

This experiment has TWO PHASES:

Phase 1 -- Predictor upgrade gate:
  Instantiate UpgradedReafferencePredictor (3-layer MLP, hidden_dim=128) inline.
  Collect (z_world_raw_prev, a_prev) -> delta_z_world_raw training data from all
  locomotion steps (transition_type NOT in genuine world-change events). This
  includes "none", "hazard_approach", and "benefit_approach" -- all are steps where
  the agent moved but the world did not genuinely change (no contact, no collection,
  no drift event). EXQ-099 bug: filter was ttype=="none", which with CausalGridWorldV2
  (use_proxy_fields=True) collected only ~8 samples in 2000 steps because proximity
  steps are reclassified to "hazard_approach"/"benefit_approach".
  Train supervised. Measure R2 on held-out set.
  If R2_test < 0.70: FAIL with note "predictor R2 insufficient (R2=X), engineering
  bottleneck remains." This DOES NOT weaken MECH-098 conceptually.

Phase 2 -- Discriminative pair (only if Phase 1 passes):
  Condition A (REAF_ON):  upgraded predictor active, correction applied
  Condition B (REAF_OFF): no correction (reafference_predictor disabled)
  Both: alpha_world=0.9 (SD-008), z_self/z_world split (SD-005), same seeds.

PASS criteria (ALL required):
  C1 (predictor gate): R2_test >= 0.70 on held-out delta_z_world.
     If C1 fails, status=FAIL. C2/C3 not evaluated.
  C2 (selectivity):    event_selectivity REAF_ON >= 1.1 * event_selectivity REAF_OFF.
     event_selectivity = mean|delta_z_world| on hazard_approach steps /
                         mean|delta_z_world| on locomotion-only (none) steps.
  C3 (harm benefit):   harm_avoidance_rate REAF_ON >= harm_avoidance_rate REAF_OFF.
     harm_avoidance_rate = fraction of hazard_approach steps where harm_eval > 0.5.

PASS if C1 AND C2 AND C3.
C1 pass + C2/C3 fail => genuine conceptual weakening of MECH-098.
"""

import sys
import copy
import random
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


EXPERIMENT_TYPE = "v3_exq_099a_mech098_reafference_upgrade"
CLAIM_IDS = ["MECH-098"]


# ---------------------------------------------------------------------------
# Upgraded predictor (inline -- do NOT modify stack.py)
# ---------------------------------------------------------------------------

class UpgradedReafferencePredictor(nn.Module):
    """
    3-layer MLP, hidden_dim=128. Upgrade of SD-007 ReafferencePredictor (2-layer, h=64).

    Same public interface as ReafferencePredictor in stack.py:
      forward(z_world_prev, a_prev) -> delta_z_world_loco
      correct_z_world(z_world_raw, z_world_prev, a_prev) -> z_world_corrected

    Biological basis: MSTd congruent neuron populations (Gu et al. 2008) decompose
    optic flow into self-motion (reafference) vs genuine world change (exafference)
    using efference copy from premotor cortex. This larger network better models the
    content-dependent perspective shift (cell content entering view during locomotion).

    MECH-101: input is z_world_raw_prev (NOT z_self_prev).
    """

    def __init__(self, world_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.world_dim = world_dim
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(world_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, world_dim),
        )

    def forward(
        self,
        z_world_prev: torch.Tensor,
        a_prev: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict z_world delta caused by self-motion (perspective shift).

        Args:
            z_world_prev: [batch, world_dim]
            a_prev:       [batch, action_dim]

        Returns:
            delta_z_world_loco: [batch, world_dim]
        """
        return self.net(torch.cat([z_world_prev, a_prev], dim=-1))

    def correct_z_world(
        self,
        z_world_raw: torch.Tensor,
        z_world_prev: torch.Tensor,
        a_prev: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply reafference correction.

        z_world_corrected = z_world_raw - delta_z_world_loco
        """
        return z_world_raw - self.forward(z_world_prev, a_prev)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


def _build_env(seed: int, harm_scale: float, proximity_harm_scale: float) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed,
        size=12,
        num_hazards=4,
        num_resources=5,
        hazard_harm=harm_scale,
        env_drift_interval=5,
        env_drift_prob=0.1,
        proximity_harm_scale=proximity_harm_scale,
        proximity_benefit_scale=proximity_harm_scale * 0.6,
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
# Phase 1: collect data + train upgraded predictor
# ---------------------------------------------------------------------------

def _collect_predictor_data(
    seed: int,
    n_episodes: int,
    steps_per_episode: int,
    self_dim: int,
    world_dim: int,
    alpha_world: float,
    alpha_self: float,
    harm_scale: float,
    proximity_harm_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collect (z_world_raw_prev + a_prev, delta_z_world_raw) training pairs
    from pure-locomotion steps (transition_type == "none").

    Uses reafference DISABLED so z_world_raw is truly the raw encoder output
    with no partial correction applied.
    """
    torch.manual_seed(seed)
    random.seed(seed)

    env = _build_env(seed, harm_scale, proximity_harm_scale)
    config = _build_config(env, self_dim, world_dim, alpha_world, alpha_self, reafference=False)
    agent = REEAgent(config)
    agent.eval()

    # Genuine world-change events: world actually changed, not just agent position.
    # These are excluded from reafference training data. "hazard_approach" and
    # "benefit_approach" are locomotion steps (perspective shift only) -- include them.
    GENUINE_WORLD_EVENTS = {
        "env_caused_hazard", "agent_caused_hazard",
        "resource", "sequence_complete", "waypoint",
    }

    inputs: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []

    for ep in range(n_episodes):
        _, obs_dict = env.reset()
        agent.reset()

        prev_z_world_raw: Optional[torch.Tensor] = None

        for step in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]

            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world)
            agent.clock.advance()

            z_world_raw_curr = latent.z_world_raw  # uncorrected encoder output
            if z_world_raw_curr is None:
                z_world_raw_curr = latent.z_world   # fallback (should not happen)

            action_idx = random.randint(0, env.action_dim - 1)
            action = _action_to_onehot(action_idx, env.action_dim, agent.device)
            agent._last_action = action

            _, _, done, info, obs_dict = env.step(action)
            ttype = info.get("transition_type", "none")

            # Collect all locomotion steps (no genuine world-change event).
            # Includes "none", "hazard_approach", "benefit_approach" -- all are
            # perspective-shift-only steps where reafference cancellation applies.
            if ttype not in GENUINE_WORLD_EVENTS and prev_z_world_raw is not None:
                delta = (z_world_raw_curr - prev_z_world_raw).detach()  # [1, world_dim]
                inp = torch.cat(
                    [prev_z_world_raw.detach(), action.detach()], dim=-1
                )  # [1, world_dim + action_dim]
                inputs.append(inp.squeeze(0))
                targets.append(delta.squeeze(0))

            prev_z_world_raw = z_world_raw_curr.detach()

            if done:
                break

    if not inputs:
        # Return empty tensors; caller handles insufficient data
        wdim = world_dim
        adim = env.action_dim
        return torch.empty(0, wdim + adim), torch.empty(0, wdim)

    return torch.stack(inputs), torch.stack(targets)


def _train_upgraded_predictor(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    world_dim: int,
    action_dim: int,
    n_epochs: int,
    lr: float,
    gate_threshold: float,
) -> Tuple["UpgradedReafferencePredictor", float]:
    """
    Train upgraded predictor (3-layer MLP, h=128) on collected data.
    80/20 train/test split. Returns (trained_predictor, r2_test).
    """
    n = len(inputs)
    print(
        f"  [Phase1] Training upgraded predictor on {n} samples"
        f" (80/20 split, {n_epochs} epochs)",
        flush=True,
    )

    if n < 20:
        print(
            f"  [Phase1] WARNING: only {n} samples -- insufficient for reliable R2."
            " R2 forced to 0.0.",
            flush=True,
        )
        pred = UpgradedReafferencePredictor(world_dim, action_dim)
        return pred, 0.0

    # Shuffle + split
    perm = torch.randperm(n)
    inputs  = inputs[perm]
    targets = targets[perm]
    n_train = int(0.8 * n)
    X_train, X_test = inputs[:n_train],  inputs[n_train:]
    Y_train, Y_test = targets[:n_train], targets[n_train:]

    pred = UpgradedReafferencePredictor(world_dim, action_dim)
    opt  = optim.Adam(pred.parameters(), lr=lr)

    pred.train()
    for epoch in range(n_epochs):
        pred_out = pred.net(X_train)
        loss = F.mse_loss(pred_out, Y_train)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if (epoch + 1) % 50 == 0 or epoch == n_epochs - 1:
            with torch.no_grad():
                test_pred = pred.net(X_test)
                ss_res = ((test_pred - Y_test) ** 2).sum().item()
                ss_tot = ((Y_test - Y_test.mean(dim=0)) ** 2).sum().item()
                r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-10 else 0.0
            print(
                f"  [Phase1] epoch {epoch+1}/{n_epochs}"
                f" train_loss={loss.item():.6f} R2_test={r2:.4f}",
                flush=True,
            )

    pred.eval()
    with torch.no_grad():
        test_pred = pred.net(X_test)
        ss_res = ((test_pred - Y_test) ** 2).sum().item()
        ss_tot = ((Y_test - Y_test.mean(dim=0)) ** 2).sum().item()
        r2_test = 1.0 - ss_res / ss_tot if ss_tot > 1e-10 else 0.0

    print(
        f"  [Phase1] Final R2_test={r2_test:.4f}"
        f" (gate threshold={gate_threshold:.2f})",
        flush=True,
    )
    return pred, r2_test


# ---------------------------------------------------------------------------
# Phase 2: single-condition run
# ---------------------------------------------------------------------------

def _run_single(
    seed: int,
    reafference: bool,
    upgraded_predictor: Optional["UpgradedReafferencePredictor"],
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
) -> Dict:
    """
    Run one condition (REAF_ON or REAF_OFF) and return evaluation metrics.

    Metrics:
      event_selectivity:   mean|delta_z_world| at hazard_approach /
                           mean|delta_z_world| at none (locomotion-only)
      harm_avoidance_rate: fraction of hazard_approach steps where harm_eval > 0.5
    """
    torch.manual_seed(seed)
    random.seed(seed)

    cond_label = "REAF_ON" if reafference else "REAF_OFF"

    env    = _build_env(seed, harm_scale, proximity_harm_scale)
    config = _build_config(env, self_dim, world_dim, alpha_world, alpha_self, reafference)
    agent  = REEAgent(config)

    # Swap in upgraded predictor for REAF_ON
    if reafference and upgraded_predictor is not None:
        pred_copy = copy.deepcopy(upgraded_predictor).to(agent.device)
        agent.latent_stack.reafference_predictor = pred_copy

    harm_buf_pos: List[torch.Tensor] = []
    harm_buf_neg: List[torch.Tensor] = []
    MAX_BUF_EACH = 2000

    standard_params = [
        p for n, p in agent.named_parameters()
        if "harm_eval_head" not in n
    ]
    harm_eval_params = list(agent.e3.harm_eval_head.parameters())
    optimizer          = optim.Adam(standard_params, lr=lr)
    harm_eval_optimizer = optim.Adam(harm_eval_params, lr=1e-4)

    train_counts: Dict[str, int] = {
        "hazard_approach": 0, "env_caused_hazard": 0,
        "agent_caused_hazard": 0, "none": 0,
    }

    # --- TRAIN ---
    agent.train()
    for ep in range(warmup_episodes):
        _, obs_dict = env.reset()
        agent.reset()

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
            if ttype in train_counts:
                train_counts[ttype] += 1

            # Harm replay buffer
            is_harm = float(harm_signal) < 0
            if is_harm:
                harm_buf_pos.append(z_world_curr)
                if len(harm_buf_pos) > MAX_BUF_EACH:
                    harm_buf_pos = harm_buf_pos[-MAX_BUF_EACH:]
            else:
                harm_buf_neg.append(z_world_curr)
                if len(harm_buf_neg) > MAX_BUF_EACH:
                    harm_buf_neg = harm_buf_neg[-MAX_BUF_EACH:]

            # E1 + E2 losses (fine-tunes reafference predictor when enabled)
            e1_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()
            total_loss = e1_loss + e2_loss
            if total_loss.requires_grad:
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                optimizer.step()

            # E3 harm_eval balanced training
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
                f" approach={train_counts['hazard_approach']}"
                f" contact={train_counts['env_caused_hazard']+train_counts['agent_caused_hazard']}"
                f" buf_pos={len(harm_buf_pos)} buf_neg={len(harm_buf_neg)}",
                flush=True,
            )

    # --- EVAL ---
    agent.eval()

    # Selectivity tracking: |delta_z_world| per transition type
    deltas_approach: List[float] = []
    deltas_none:     List[float] = []

    # Harm detection rate at approach steps
    approach_detected = 0
    approach_total    = 0

    n_fatal = 0

    for _ in range(eval_episodes):
        _, obs_dict = env.reset()
        agent.reset()
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

            # delta_z_world magnitude
            if prev_z_world is not None:
                delta_mag = float(
                    (z_world_curr - prev_z_world).abs().mean().item()
                )
                if ttype == "hazard_approach":
                    deltas_approach.append(delta_mag)
                elif ttype == "none":
                    deltas_none.append(delta_mag)

            # Harm detection at approach steps
            try:
                with torch.no_grad():
                    harm_score = float(agent.e3.harm_eval(z_world_curr).item())
                if ttype == "hazard_approach":
                    approach_total += 1
                    if harm_score > 0.5:
                        approach_detected += 1
            except Exception:
                n_fatal += 1

            prev_z_world = z_world_curr

            if done:
                break

    # Aggregate
    mean_delta_approach = float(
        sum(deltas_approach) / max(1, len(deltas_approach))
    )
    mean_delta_none = float(
        sum(deltas_none) / max(1, len(deltas_none))
    )
    event_selectivity = mean_delta_approach / max(mean_delta_none, 1e-8)
    harm_avoidance_rate = float(approach_detected) / max(approach_total, 1)

    print(
        f"  [eval] seed={seed} cond={cond_label}"
        f" selectivity={event_selectivity:.4f}"
        f" (approach={mean_delta_approach:.4f} / none={mean_delta_none:.4f})"
        f" harm_avoidance_rate={harm_avoidance_rate:.4f}"
        f" (detected={approach_detected}/{approach_total})",
        flush=True,
    )

    return {
        "seed": seed,
        "condition": cond_label,
        "reafference": reafference,
        "event_selectivity": float(event_selectivity),
        "harm_avoidance_rate": float(harm_avoidance_rate),
        "mean_delta_approach": float(mean_delta_approach),
        "mean_delta_none": float(mean_delta_none),
        "n_approach": int(approach_total),
        "n_approach_detected": int(approach_detected),
        "n_none_deltas": int(len(deltas_none)),
        "train_approach_events": int(train_counts["hazard_approach"]),
        "train_contact_events": int(
            train_counts["env_caused_hazard"] + train_counts["agent_caused_hazard"]
        ),
        "n_fatal": int(n_fatal),
    }


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------

def run(
    seed: int = 42,
    data_collection_episodes: int = 10,
    warmup_episodes: int = 150,
    eval_episodes: int = 20,
    steps_per_episode: int = 200,
    self_dim: int = 32,
    world_dim: int = 32,
    lr: float = 1e-3,
    alpha_world: float = 0.9,
    alpha_self: float = 0.3,
    harm_scale: float = 0.02,
    proximity_harm_scale: float = 0.05,
    predictor_epochs: int = 200,
    predictor_lr: float = 1e-3,
    gate_threshold: float = 0.70,
    **kwargs,
) -> dict:
    """
    Two-phase MECH-098 conflict-resolution experiment.

    Phase 1: train upgraded predictor, check R2_test >= gate_threshold.
    Phase 2: discriminative pair REAF_ON vs REAF_OFF (only if Phase 1 passes).
    """

    print(
        f"\n[V3-EXQ-099a] MECH-098 reafference upgrade experiment"
        f" seed={seed} alpha_world={alpha_world}"
        f" warmup={warmup_episodes} eval={eval_episodes}"
        f" steps={steps_per_episode}",
        flush=True,
    )
    env_dummy = _build_env(seed, harm_scale, proximity_harm_scale)

    # ----- Phase 1: predictor upgrade gate -----
    print("\n[V3-EXQ-099a] Phase 1 -- collecting predictor training data...", flush=True)
    inputs, targets = _collect_predictor_data(
        seed=seed,
        n_episodes=data_collection_episodes,
        steps_per_episode=steps_per_episode,
        self_dim=self_dim,
        world_dim=world_dim,
        alpha_world=alpha_world,
        alpha_self=alpha_self,
        harm_scale=harm_scale,
        proximity_harm_scale=proximity_harm_scale,
    )
    n_samples = len(inputs)
    print(f"  [Phase1] Collected {n_samples} locomotion samples (none+approach).", flush=True)

    upgraded_predictor, r2_test = _train_upgraded_predictor(
        inputs=inputs,
        targets=targets,
        world_dim=world_dim,
        action_dim=env_dummy.action_dim,
        n_epochs=predictor_epochs,
        lr=predictor_lr,
        gate_threshold=gate_threshold,
    )

    c1_pass = r2_test >= gate_threshold

    if not c1_pass:
        note = (
            f"predictor R2 insufficient (R2={r2_test:.4f},"
            f" threshold={gate_threshold:.2f}), MECH-098 cannot be"
            " meaningfully tested -- engineering bottleneck remains."
            " Upgrade path: increase hidden_dim further, add deeper architecture,"
            " or increase data collection episodes."
        )
        print(f"\n[V3-EXQ-099a] Phase 1 FAIL: {note}", flush=True)

        metrics = {
            "r2_test": float(r2_test),
            "gate_threshold": float(gate_threshold),
            "n_samples": int(n_samples),
            "c1_pass": 0.0,
            "c2_pass": 0.0,
            "c3_pass": 0.0,
            "phase1_gate_failed": 1.0,
        }
        summary_markdown = (
            f"# V3-EXQ-099 -- MECH-098 Reafference Upgrade: Phase 1 FAIL\n\n"
            f"**Status:** FAIL\n"
            f"**Claims:** MECH-098\n"
            f"**Reason:** engineering bottleneck -- predictor R2 insufficient\n\n"
            f"## Phase 1 Gate\n\n"
            f"R2_test={r2_test:.4f} (threshold={gate_threshold:.2f}) -- FAIL\n\n"
            f"## Note\n\n"
            f"{note}\n\n"
            f"MECH-098 is NOT weakened by this result. The claim is conceptually correct"
            f" but requires a higher-quality predictor before the discriminative test"
            f" is meaningful.\n"
        )
        return {
            "status": "FAIL",
            "metrics": metrics,
            "summary_markdown": summary_markdown,
            "claim_ids": CLAIM_IDS,
            "evidence_direction": "not_applicable",
            "experiment_type": EXPERIMENT_TYPE,
            "fatal_error_count": 0,
        }

    print(
        f"\n[V3-EXQ-099a] Phase 1 PASS: R2_test={r2_test:.4f} >= {gate_threshold:.2f}."
        " Proceeding to discriminative pair.",
        flush=True,
    )

    # ----- Phase 2: discriminative pair -----
    print("\n[V3-EXQ-099a] Phase 2 -- REAF_ON run...", flush=True)
    result_on = _run_single(
        seed=seed,
        reafference=True,
        upgraded_predictor=upgraded_predictor,
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
    )

    print("\n[V3-EXQ-099a] Phase 2 -- REAF_OFF run...", flush=True)
    result_off = _run_single(
        seed=seed,
        reafference=False,
        upgraded_predictor=None,
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
    )

    sel_on  = result_on["event_selectivity"]
    sel_off = result_off["event_selectivity"]
    har_on  = result_on["harm_avoidance_rate"]
    har_off = result_off["harm_avoidance_rate"]

    c2_pass = sel_on >= 1.1 * sel_off
    c3_pass = har_on >= har_off

    all_pass     = c1_pass and c2_pass and c3_pass
    criteria_met = sum([c1_pass, c2_pass, c3_pass])
    status       = "PASS" if all_pass else "FAIL"

    print(f"\n[V3-EXQ-099a] Final results:", flush=True)
    print(
        f"  event_selectivity REAF_ON={sel_on:.4f}"
        f"  REAF_OFF={sel_off:.4f}"
        f"  ratio={sel_on/max(sel_off, 1e-8):.4f}"
        f"  (C2 threshold: ON >= 1.1 * OFF)",
        flush=True,
    )
    print(
        f"  harm_avoidance_rate REAF_ON={har_on:.4f}"
        f"  REAF_OFF={har_off:.4f}"
        f"  (C3 threshold: ON >= OFF)",
        flush=True,
    )
    print(
        f"  R2_test={r2_test:.4f}  C1={'PASS' if c1_pass else 'FAIL'}"
        f"  C2={'PASS' if c2_pass else 'FAIL'}"
        f"  C3={'PASS' if c3_pass else 'FAIL'}"
        f"  -> {status} ({criteria_met}/3)",
        flush=True,
    )

    # Interpretation
    if all_pass:
        interpretation = (
            f"MECH-098 SUPPORTED: upgraded reafference predictor (R2={r2_test:.4f})"
            f" enables meaningful correction. REAF_ON event_selectivity={sel_on:.4f}"
            f" vs REAF_OFF={sel_off:.4f} (ratio={sel_on/max(sel_off,1e-8):.4f})."
            f" harm_avoidance_rate ON={har_on:.4f} >= OFF={har_off:.4f}."
            " Perspective-shift subtraction gives E3 a cleaner world-state signal."
            " Prior FAILs (EXQ-069, EXQ-082) confirmed as engineering bottleneck, not"
            " conceptual failure. Resolves conflict for MECH-098."
        )
    elif c1_pass:
        interpretation = (
            f"MECH-098 WEAKENED: upgraded predictor functional (R2={r2_test:.4f}),"
            " but reafference correction does not improve downstream metrics."
            f" C2 (selectivity): {'PASS' if c2_pass else 'FAIL'}"
            f" ({sel_on:.4f} vs 1.1*{sel_off:.4f}={1.1*sel_off:.4f})."
            f" C3 (harm avoidance): {'PASS' if c3_pass else 'FAIL'}"
            f" ({har_on:.4f} vs {har_off:.4f})."
            " The cancellation mechanism is not providing the expected benefit at this"
            " model/task scale. Genuine conceptual weakening of MECH-098."
        )
    else:
        interpretation = (
            f"MECH-098 cannot be evaluated: predictor R2={r2_test:.4f} below"
            f" gate {gate_threshold:.2f}. Engineering bottleneck persists."
        )

    failure_notes: List[str] = []
    if not c1_pass:
        failure_notes.append(
            f"C1 FAIL: R2_test={r2_test:.4f} < {gate_threshold:.2f}"
            " -- predictor upgrade insufficient, engineering bottleneck persists."
        )
    if not c2_pass:
        failure_notes.append(
            f"C2 FAIL: selectivity REAF_ON={sel_on:.4f} <"
            f" 1.1*REAF_OFF=1.1*{sel_off:.4f}={1.1*sel_off:.4f}"
            " -- reafference correction does not improve z_world event sensitivity."
        )
    if not c3_pass:
        failure_notes.append(
            f"C3 FAIL: harm_avoidance_rate REAF_ON={har_on:.4f} <"
            f" REAF_OFF={har_off:.4f}"
            " -- reafference correction does not improve E3 harm detection."
        )
    for note in failure_notes:
        print(f"  {note}", flush=True)

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(
            f"- {n}" for n in failure_notes
        )

    summary_markdown = (
        f"# V3-EXQ-099a -- MECH-098 Reafference Upgrade Gate + Discriminative Pair\n\n"
        f"**Status:** {status}\n"
        f"**Claims:** MECH-098\n"
        f"**Seed:** {seed}\n"
        f"**alpha_world:** {alpha_world}  (SD-008)\n"
        f"**Warmup:** {warmup_episodes} eps  **Eval:** {eval_episodes} eps\n\n"
        f"## Phase 1 -- Predictor Upgrade Gate\n\n"
        f"| Metric | Value |\n"
        f"|--------|-------|\n"
        f"| n_samples | {n_samples} |\n"
        f"| R2_test | {r2_test:.4f} |\n"
        f"| gate_threshold | {gate_threshold:.2f} |\n"
        f"| C1 | {'PASS' if c1_pass else 'FAIL'} |\n\n"
        f"## Phase 2 -- Discriminative Pair\n\n"
        f"| Condition | event_selectivity | harm_avoidance_rate | n_approach |\n"
        f"|-----------|------------------|---------------------|------------|\n"
        f"| REAF_ON  | {sel_on:.4f} | {har_on:.4f}"
        f" | {result_on['n_approach']} |\n"
        f"| REAF_OFF | {sel_off:.4f} | {har_off:.4f}"
        f" | {result_off['n_approach']} |\n\n"
        f"**Selectivity ratio (ON/OFF): {sel_on/max(sel_off,1e-8):.4f}"
        f" (C2 threshold: >= 1.1)**\n"
        f"**harm_avoidance_rate delta (ON - OFF): {har_on - har_off:+.4f}"
        f" (C3 threshold: >= 0)**\n\n"
        f"## PASS Criteria\n\n"
        f"| Criterion | Result | Value |\n"
        f"|---|---|---|\n"
        f"| C1: R2_test >= {gate_threshold:.2f} | {'PASS' if c1_pass else 'FAIL'}"
        f" | {r2_test:.4f} |\n"
        f"| C2: selectivity ON >= 1.1 x OFF | {'PASS' if c2_pass else 'FAIL'}"
        f" | ON={sel_on:.4f}, 1.1xOFF={1.1*sel_off:.4f} |\n"
        f"| C3: harm_avoidance_rate ON >= OFF | {'PASS' if c3_pass else 'FAIL'}"
        f" | ON={har_on:.4f}, OFF={har_off:.4f} |\n\n"
        f"Criteria met: {criteria_met}/3 -> **{status}**\n\n"
        f"## Interpretation\n\n"
        f"{interpretation}\n"
        f"{failure_section}\n"
    )

    metrics = {
        "r2_test":                  float(r2_test),
        "gate_threshold":           float(gate_threshold),
        "n_samples_phase1":         int(n_samples),
        "event_selectivity_on":     float(sel_on),
        "event_selectivity_off":    float(sel_off),
        "selectivity_ratio":        float(sel_on / max(sel_off, 1e-8)),
        "harm_avoidance_rate_on":   float(har_on),
        "harm_avoidance_rate_off":  float(har_off),
        "harm_avoidance_delta":     float(har_on - har_off),
        "mean_delta_approach_on":   float(result_on["mean_delta_approach"]),
        "mean_delta_none_on":       float(result_on["mean_delta_none"]),
        "mean_delta_approach_off":  float(result_off["mean_delta_approach"]),
        "mean_delta_none_off":      float(result_off["mean_delta_none"]),
        "n_approach_on":            int(result_on["n_approach"]),
        "n_approach_off":           int(result_off["n_approach"]),
        "alpha_world":              float(alpha_world),
        "warmup_episodes":          int(warmup_episodes),
        "c1_pass":                  1.0 if c1_pass else 0.0,
        "c2_pass":                  1.0 if c2_pass else 0.0,
        "c3_pass":                  1.0 if c3_pass else 0.0,
        "criteria_met":             float(criteria_met),
    }

    if all_pass:
        evidence_direction = "supports"
    elif criteria_met >= 2:
        evidence_direction = "mixed"
    else:
        evidence_direction = "weakens"

    return {
        "status": status,
        "metrics": metrics,
        "summary_markdown": summary_markdown,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": evidence_direction,
        "experiment_type": EXPERIMENT_TYPE,
        "fatal_error_count": int(
            result_on.get("n_fatal", 0) + result_off.get("n_fatal", 0)
        ),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser(
        description="V3-EXQ-099a MECH-098 reafference upgrade gate + discriminative pair"
    )
    parser.add_argument("--seed",                type=int,   default=42)
    parser.add_argument("--data-eps",            type=int,   default=10,
                        help="episodes for Phase 1 data collection")
    parser.add_argument("--warmup",              type=int,   default=150)
    parser.add_argument("--eval-eps",            type=int,   default=20)
    parser.add_argument("--steps",               type=int,   default=200)
    parser.add_argument("--alpha-world",         type=float, default=0.9)
    parser.add_argument("--alpha-self",          type=float, default=0.3)
    parser.add_argument("--harm-scale",          type=float, default=0.02)
    parser.add_argument("--proximity-scale",     type=float, default=0.05)
    parser.add_argument("--predictor-epochs",    type=int,   default=200)
    parser.add_argument("--gate-threshold",      type=float, default=0.70)
    parser.add_argument(
        "--smoke-test", action="store_true",
        help=(
            "Quick smoke test: 3 data eps, 5 warmup eps, 3 eval eps, 30 steps."
            " Gate threshold forced to -1.0 (always passes) to exercise full code path."
            " Writes output file."
        )
    )
    args = parser.parse_args()

    if args.smoke_test:
        data_eps   = 3
        warmup     = 5
        eval_eps   = 3
        steps      = 30
        gate_thr   = -1.0   # force Phase 1 pass so full code path is exercised
        print("[V3-EXQ-099a] SMOKE TEST MODE", flush=True)
    else:
        data_eps   = args.data_eps
        warmup     = args.warmup
        eval_eps   = args.eval_eps
        steps      = args.steps
        gate_thr   = args.gate_threshold

    result = run(
        seed=args.seed,
        data_collection_episodes=data_eps,
        warmup_episodes=warmup,
        eval_episodes=eval_eps,
        steps_per_episode=steps,
        alpha_world=args.alpha_world,
        alpha_self=args.alpha_self,
        harm_scale=args.harm_scale,
        proximity_harm_scale=args.proximity_scale,
        predictor_epochs=args.predictor_epochs,
        gate_threshold=gate_thr,
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"] = ts
    result["claim"]         = CLAIM_IDS[0]
    result["verdict"]       = result["status"]
    result["run_id"]        = f"{EXPERIMENT_TYPE}_{ts}_v3"
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
        print("[V3-EXQ-099a] SMOKE TEST COMPLETE", flush=True)
        for k in [
            "r2_test", "event_selectivity_on", "event_selectivity_off",
            "selectivity_ratio", "harm_avoidance_rate_on", "harm_avoidance_rate_off",
            "c1_pass", "c2_pass", "c3_pass",
        ]:
            v = result["metrics"].get(k, "N/A")
            print(f"  {k}: {v}", flush=True)
    else:
        for k, v in result["metrics"].items():
            print(f"  {k}: {v}", flush=True)
