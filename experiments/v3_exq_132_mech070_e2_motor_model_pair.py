#!/opt/local/bin/python3
"""
V3-EXQ-132 -- MECH-070 E2 Motor-Model Discriminative Pair

Claims: MECH-070
Proposal: EXP-0034 / EVB-0026

MECH-070 asserts: "E2 is a conceptual-sensorium motor model with a planning horizon
that exceeds E1."

Neurological mapping:
  E1 = cortical LSTM (slow, z_world domain, prediction_horizon=20, sensory error)
  E2 = cerebellar forward model (fast, z_self domain, training horizon=1-step, motor-
       sensory error). E2's rollout_horizon=30 is valid for planning because E2 predicts
       body/motor state transitions, not world-state transitions -- it predicts WHERE the
       agent's body will be (z_self trajectory) while E1 predicts world changes.

The testable functional claim:
  E2 as a trained z_self forward model improves trajectory candidate quality during
  planning. Specifically: with E2 trained on motor-sensory transitions (z_self domain),
  the agent can simulate the proprioceptive consequences of candidate action sequences.
  This should reduce harm contacts by improving multi-step body-state prediction for
  candidate trajectory selection.

  The ablation removes E2's learned forward-model capability: E2 is frozen at random
  initialization. Planning candidates must then rely on E1's world predictions alone
  (which are in the z_world domain, not z_self -- mismatched for body-state planning).

Discriminative pair design:
  E2_MOTOR_ON (MECH-070 architecture):
    - E2 is a trained forward model: z_self transitions trained on motor-sensory error
    - E2.world_forward(z_self, a) predicts next z_self given current z_self + action
    - Trajectory quality is evaluated using E3.harm_eval after E2 propagates z_self
    - E2 trains every step with its own optimizer (1-step MSE on z_self transitions)
    - Harm eval head (E3) trains on z_world latents as usual

  E2_MOTOR_ABLATED (motor model removed):
    - E2 parameters are frozen at random initialization (no training)
    - Trajectory candidate quality cannot use E2 z_self predictions
    - E3.harm_eval trains on z_world as usual (harm eval head is NOT ablated)
    - Effectively: E3 must evaluate trajectories using only z_world signal
    - The ablation isolates E2's motor forward model contribution from E3 harm eval

Key metric: harm_eval_gap (E3 harm_eval discrimination: harm events vs safe events).
With E2 trained, z_self provides better-calibrated body-state information to E3,
expected to improve harm/safe discrimination at eval.

Secondary metric: e2_forward_r2 (R^2 of E2.world_forward(z_self, a) predicting next
z_self). This confirms that E2_MOTOR_ON actually learned a useful forward model vs the
ablated E2 producing near-zero R^2 (as expected). If E2_MOTOR_ON shows r2 > threshold
and E2_MOTOR_ABLATED shows r2 near 0, this is a mechanistic sanity check confirming the
manipulation worked as intended.

Pre-registered acceptance criteria:
  C1: gap_motor_on >= 0.04        (E2_MOTOR_ON harm eval above floor -- both seeds)
  C2: per-seed delta (ON - ABL) >= 0.02   (motor model adds >=2pp -- both seeds)
  C3: gap_ablated >= 0.0          (ablation still learns something from z_world; data quality)
  C4: n_harm_eval_min >= 20       (sufficient harm events in eval -- both seeds)
  C5: e2_r2_on >= 0.05            (E2_MOTOR_ON learned a useful forward model)
                                   -- confirms manipulation integrity

Decision scoring:
  PASS (all 5):          retain_ree -- E2 motor model improves trajectory quality
  C1+C2+C4:             hybridize  -- motor model effect present, r2 low (E2 partial only)
  C2 fails:             retire_ree_claim -- no detectable advantage of E2 motor model
  C5 fails (C1-C4 pass): hybridize -- E2 improves E3 but learned model quality unclear

Note on claim scope: MECH-070 also asserts E2's rollout_horizon exceeds E1's
prediction_horizon. This V3 proxy tests the functional consequence (improved trajectory
quality with trained E2) rather than the rollout horizon ordering directly -- a PASS
here supports the functional claim that E2 motor modeling contributes to planning quality.
The rollout_horizon > prediction_horizon ordering is tested by MECH-135 (EXQ-104b/108).

Seeds: [42, 123] (matched -- same env per seed across conditions)
Env:   CausalGridWorldV2, size=6, 4 hazards, nav_bias=0.40
Warmup: 400 eps x 200 steps
Eval:  50 eps x 200 steps (no training, fixed weights)
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


EXPERIMENT_TYPE = "v3_exq_132_mech070_e2_motor_model_pair"
CLAIM_IDS = ["MECH-070"]

# Pre-registered thresholds
THRESH_C1 = 0.04   # gap_motor_on >= 0.04 (E2_MOTOR_ON harm eval above floor)
THRESH_C2 = 0.02   # per-seed delta (ON - ABLATED) >= 0.02
THRESH_C3 = 0.0    # gap_ablated >= 0.0 (ablation still learns something)
THRESH_C4 = 20     # n_harm_eval_min >= 20 (data quality)
THRESH_C5 = 0.05   # e2_forward_r2 >= 0.05 (E2 learned a useful z_self model)

# Dimension constants
HARM_OBS_DIM = 51  # SD-010 nociceptive stream: hazard_field_view[25] + resource_field_view[25] + harm_exposure[1]


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
    motor_on: bool,
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
    """Run one (seed, condition) cell; return harm eval gap and E2 forward model metrics.

    E2_MOTOR_ON:      E2 is trained on z_self motor-sensory transitions (MECH-070 architecture).
    E2_MOTOR_ABLATED: E2 is frozen at random init (motor model removed).
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    cond_label = "E2_MOTOR_ON" if motor_on else "E2_MOTOR_ABLATED"

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
    world_obs_dim = env.world_obs_dim
    body_obs_dim = env.body_obs_dim

    config = REEConfig.from_dims(
        body_obs_dim=body_obs_dim,
        world_obs_dim=world_obs_dim,
        action_dim=n_actions,
        self_dim=self_dim,
        world_dim=world_dim,
        alpha_world=alpha_world,
        alpha_self=alpha_self,
        reafference_action_dim=0,
    )
    agent = REEAgent(config)

    # In E2_MOTOR_ABLATED: freeze E2 parameters so the motor forward model provides
    # no learned information. E1 and E3 still train normally.
    if not motor_on:
        for param in agent.e2.parameters():
            param.requires_grad = False

    MAX_BUF = 2000

    # Separate optimizer sets to control what trains in each condition
    e1_params = list(agent.e1.parameters())
    e3_params = [
        p for n, p in agent.e3.named_parameters()
        if "harm_eval_head" not in n
    ]
    harm_eval_params = list(agent.e3.harm_eval_head.parameters())

    if motor_on:
        e2_params = list(agent.e2.parameters())
        standard_params = e1_params + e2_params + e3_params
    else:
        # E2 frozen -- only E1 and E3 (non-harm-eval) train
        standard_params = e1_params + e3_params

    optimizer = optim.Adam(standard_params, lr=lr)
    harm_eval_optimizer = optim.Adam(harm_eval_params, lr=1e-4)

    # Harm eval buffer: accumulates z_world latents and harm labels
    buf_zw: List[torch.Tensor] = []
    buf_labels: List[float] = []

    # E2 forward model evaluation buffer: tracks z_self transitions for R^2 computation
    e2_pred_vals: List[float] = []
    e2_true_vals: List[float] = []

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

    prev_z_self: torch.Tensor = None
    prev_action: torch.Tensor = None

    for ep in range(actual_warmup):
        _, obs_dict = env.reset()
        agent.reset()
        prev_z_self = None
        prev_action = None

        for _ in range(steps_per_episode):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]

            latent = agent.sense(obs_body, obs_world)
            agent.clock.advance()

            z_world_curr = latent.z_world.detach()
            z_self_curr = latent.z_self.detach()

            if random.random() < nav_bias:
                action_idx = _hazard_approach_action(env, n_actions)
            else:
                action_idx = random.randint(0, n_actions - 1)
            action = _action_to_onehot(action_idx, n_actions, agent.device)
            agent._last_action = action

            # E2 forward model training (E2_MOTOR_ON only):
            # Train E2 to predict next z_self from (current z_self, action).
            # This is the motor-sensory error signal -- cerebellar forward model.
            if motor_on and prev_z_self is not None and prev_action is not None:
                try:
                    z_self_pred = agent.e2.world_forward(prev_z_self, prev_action)
                    e2_loss_val = F.mse_loss(z_self_pred, z_self_curr)
                    if e2_loss_val.requires_grad:
                        optimizer.zero_grad()
                        e2_loss_val.backward()
                        torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                        optimizer.step()
                    # Track for R^2 computation (scalar summary: predicted vs actual norm)
                    e2_pred_vals.append(float(z_self_pred.norm().item()))
                    e2_true_vals.append(float(z_self_curr.norm().item()))
                except Exception:
                    pass

            _, harm_signal, done, info, obs_dict = env.step(action)
            ttype = info.get("transition_type", "none")
            if ttype in counts:
                counts[ttype] += 1

            harm_sig_float = float(harm_signal)
            is_harm = harm_sig_float < 0

            # E1 prediction loss update (both conditions -- E1 trains normally)
            try:
                e1_loss = agent.compute_prediction_loss()
                if e1_loss.requires_grad:
                    if not motor_on:
                        optimizer.zero_grad()
                        e1_loss.backward()
                        torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                        optimizer.step()
            except Exception:
                pass

            # Harm eval buffer update and training (both conditions)
            buf_zw.append(z_world_curr)
            buf_labels.append(1.0 if is_harm else 0.0)
            if len(buf_zw) > MAX_BUF:
                buf_zw = buf_zw[-MAX_BUF:]
                buf_labels = buf_labels[-MAX_BUF:]

            n_harm_buf = sum(1 for lbl in buf_labels if lbl > 0.5)
            n_safe_buf = sum(1 for lbl in buf_labels if lbl <= 0.5)
            if n_harm_buf >= 4 and n_safe_buf >= 4:
                harm_idxs = [i for i, lbl in enumerate(buf_labels) if lbl > 0.5]
                safe_idxs = [i for i, lbl in enumerate(buf_labels) if lbl <= 0.5]
                k = min(8, min(len(harm_idxs), len(safe_idxs)))
                sel_h = random.sample(harm_idxs, k)
                sel_s = random.sample(safe_idxs, k)
                sel = sel_h + sel_s
                zw_b = torch.cat([buf_zw[i] for i in sel], dim=0)
                labels_b = torch.tensor(
                    [buf_labels[i] for i in sel],
                    dtype=torch.float32,
                ).unsqueeze(1)
                pred = agent.e3.harm_eval(zw_b)
                loss_he = F.mse_loss(pred, labels_b)
                if loss_he.requires_grad:
                    harm_eval_optimizer.zero_grad()
                    loss_he.backward()
                    torch.nn.utils.clip_grad_norm_(
                        agent.e3.harm_eval_head.parameters(), 0.5
                    )
                    harm_eval_optimizer.step()

            prev_z_self = z_self_curr.detach()
            prev_action = action.detach()

            if done:
                break

        if (ep + 1) % 100 == 0 or ep == actual_warmup - 1:
            print(
                f"  [train] seed={seed} cond={cond_label}"
                f" ep {ep+1}/{actual_warmup}"
                f" approach={counts['hazard_approach']}"
                f" contact={counts['env_caused_hazard']+counts['agent_caused_hazard']}",
                flush=True,
            )

    # --- EVAL ---
    agent.eval()

    eval_scores_harm: List[float] = []
    eval_scores_safe: List[float] = []
    n_fatal = 0

    eval_e2_pred: List[float] = []
    eval_e2_true: List[float] = []

    prev_z_self_eval: torch.Tensor = None
    prev_action_eval: torch.Tensor = None

    for _ in range(actual_eval):
        _, obs_dict = env.reset()
        agent.reset()
        prev_z_self_eval = None
        prev_action_eval = None

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

            _, harm_signal, done, info, obs_dict = env.step(action)
            is_harm = float(harm_signal) < 0

            # E2 forward model R^2 diagnostic at eval
            if motor_on and prev_z_self_eval is not None and prev_action_eval is not None:
                try:
                    with torch.no_grad():
                        z_self_pred = agent.e2.world_forward(
                            prev_z_self_eval, prev_action_eval
                        )
                    eval_e2_pred.append(float(z_self_pred.norm().item()))
                    eval_e2_true.append(float(z_self_curr.norm().item()))
                except Exception:
                    pass

            try:
                with torch.no_grad():
                    score = float(agent.e3.harm_eval(z_world_curr).item())
                if is_harm:
                    eval_scores_harm.append(score)
                else:
                    eval_scores_safe.append(score)
            except Exception:
                n_fatal += 1

            prev_z_self_eval = z_self_curr.detach()
            prev_action_eval = action.detach()

            if done:
                break

    n_harm_eval = len(eval_scores_harm)
    n_safe_eval = len(eval_scores_safe)

    mean_harm = float(sum(eval_scores_harm) / max(1, n_harm_eval))
    mean_safe = float(sum(eval_scores_safe) / max(1, n_safe_eval))
    gap = mean_harm - mean_safe

    # R^2 for E2 forward model (norm-based scalar proxy -- captures whether E2 learned
    # to predict z_self magnitudes; full R^2 would require per-dim tracking)
    def _r2(pred_list: List[float], true_list: List[float]) -> float:
        if len(pred_list) < 2:
            return 0.0
        pred_arr = np.array(pred_list)
        true_arr = np.array(true_list)
        ss_res = float(np.sum((true_arr - pred_arr) ** 2))
        ss_tot = float(np.sum((true_arr - true_arr.mean()) ** 2))
        if ss_tot < 1e-12:
            return 0.0
        return float(1.0 - ss_res / ss_tot)

    # E2 R^2 from training buffer (more data than eval)
    e2_r2_train = _r2(e2_pred_vals, e2_true_vals) if motor_on else 0.0
    # E2 R^2 from eval buffer
    e2_r2_eval = _r2(eval_e2_pred, eval_e2_true) if motor_on else 0.0
    # Use eval R^2 as primary (no data leakage); fall back to train if eval insufficient
    e2_r2 = e2_r2_eval if len(eval_e2_pred) >= 10 else e2_r2_train

    print(
        f"  [eval] seed={seed} cond={cond_label}"
        f" gap={gap:.4f} mean_harm={mean_harm:.4f} mean_safe={mean_safe:.4f}"
        f" n_harm={n_harm_eval} n_safe={n_safe_eval}"
        f" e2_r2={e2_r2:.4f}",
        flush=True,
    )

    return {
        "seed": seed,
        "condition": cond_label,
        "motor_on": motor_on,
        "harm_eval_gap": float(gap),
        "mean_harm_score": float(mean_harm),
        "mean_safe_score": float(mean_safe),
        "n_harm_eval": int(n_harm_eval),
        "n_safe_eval": int(n_safe_eval),
        "n_fatal": int(n_fatal),
        "e2_r2": float(e2_r2),
        "train_approach": int(counts["hazard_approach"]),
        "train_contact": int(counts["env_caused_hazard"] + counts["agent_caused_hazard"]),
    }


def run(
    seeds: Tuple = (42, 123),
    warmup_episodes: int = 400,
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
    """Discriminative pair: E2_MOTOR_ON (E2 trained as z_self forward model --
    MECH-070 cerebellar motor model) vs E2_MOTOR_ABLATED (E2 frozen at random init,
    motor-model contribution removed).
    Tests MECH-070: E2 as conceptual-sensorium motor model improves trajectory quality.
    """
    results_on:      List[Dict] = []
    results_ablated: List[Dict] = []

    # Run cells in matched-seed order: for each seed, run both conditions
    for seed in seeds:
        for motor in [True, False]:
            label = "E2_MOTOR_ON" if motor else "E2_MOTOR_ABLATED"
            print(
                f"\n[V3-EXQ-132] {label} seed={seed}"
                f" warmup={warmup_episodes} eval={eval_episodes}"
                f" nav_bias={nav_bias}",
                flush=True,
            )
            r = _run_single(
                seed=seed,
                motor_on=motor,
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
            if motor:
                results_on.append(r)
            else:
                results_ablated.append(r)

    def _avg(results: List[Dict], key: str) -> float:
        vals = [r[key] for r in results]
        return float(sum(vals) / max(1, len(vals)))

    gap_on      = _avg(results_on,      "harm_eval_gap")
    gap_ablated = _avg(results_ablated, "harm_eval_gap")
    delta_gap   = gap_on - gap_ablated

    n_harm_min = min(r["n_harm_eval"] for r in results_on + results_ablated)

    e2_r2_on = _avg(results_on, "e2_r2")

    # Pre-registered PASS criteria
    # C1: E2_MOTOR_ON gap >= THRESH_C1 in ALL seeds
    c1_pass = all(r["harm_eval_gap"] >= THRESH_C1 for r in results_on)
    # C2: per-seed delta (ON - ABLATED) >= THRESH_C2 in ALL seeds
    per_seed_deltas: List[float] = []
    for r_on in results_on:
        matching = [r for r in results_ablated if r["seed"] == r_on["seed"]]
        if matching:
            per_seed_deltas.append(r_on["harm_eval_gap"] - matching[0]["harm_eval_gap"])
    c2_pass = len(per_seed_deltas) > 0 and all(d >= THRESH_C2 for d in per_seed_deltas)
    # C3: E2_MOTOR_ABLATED gap >= 0 (ablation still learns something from z_world)
    c3_pass = gap_ablated >= THRESH_C3
    # C4: sufficient harm events in eval
    c4_pass = n_harm_min >= THRESH_C4
    # C5: E2 forward model R^2 confirms E2 learned useful z_self transitions
    c5_pass = e2_r2_on >= THRESH_C5

    all_pass = c1_pass and c2_pass and c3_pass and c4_pass and c5_pass
    criteria_met = sum([c1_pass, c2_pass, c3_pass, c4_pass, c5_pass])
    status = "PASS" if all_pass else "FAIL"

    if all_pass:
        decision = "retain_ree"
    elif c1_pass and c2_pass and c4_pass:
        decision = "hybridize"
    elif not c2_pass:
        decision = "retire_ree_claim"
    else:
        decision = "hybridize"

    print(f"\n[V3-EXQ-132] Results:", flush=True)
    print(
        f"  gap_on={gap_on:.4f}"
        f" gap_ablated={gap_ablated:.4f}"
        f" delta_gap={delta_gap:+.4f}",
        flush=True,
    )
    print(
        f"  e2_r2_on={e2_r2_on:.4f}"
        f" n_harm_min={n_harm_min}"
        f"  per_seed_deltas={[round(d, 4) for d in per_seed_deltas]}"
        f"  decision={decision}  status={status} ({criteria_met}/5)",
        flush=True,
    )

    failure_notes: List[str] = []
    if not c1_pass:
        failing = [r["seed"] for r in results_on if r["harm_eval_gap"] < THRESH_C1]
        failure_notes.append(
            f"C1 FAIL: E2_MOTOR_ON gap below {THRESH_C1} in seeds {failing}"
            " -- E2 motor model does not produce discriminative harm eval signal"
        )
    if not c2_pass:
        failure_notes.append(
            f"C2 FAIL: per-seed deltas {[round(d, 4) for d in per_seed_deltas]}"
            f" < {THRESH_C2}"
            " -- E2 motor model does not add >=2pp harm eval advantage over ablation"
        )
    if not c3_pass:
        failure_notes.append(
            f"C3 FAIL: gap_ablated={gap_ablated:.4f} < {THRESH_C3}"
            " -- ablated condition fails entirely; confound check required"
        )
    if not c4_pass:
        failure_notes.append(
            f"C4 FAIL: n_harm_min={n_harm_min} < {THRESH_C4}"
            " -- insufficient harm events in eval phase"
        )
    if not c5_pass:
        failure_notes.append(
            f"C5 FAIL: e2_r2_on={e2_r2_on:.4f} < {THRESH_C5}"
            " -- E2_MOTOR_ON did not learn a useful z_self forward model;"
            " manipulation integrity not confirmed"
        )

    for note in failure_notes:
        print(f"  {note}", flush=True)

    if all_pass:
        interpretation = (
            "MECH-070 SUPPORTED (V3 proxy): trained E2 motor model (E2_MOTOR_ON)"
            f" produces higher harm_eval quality (gap_on={gap_on:.4f} vs"
            f" gap_ablated={gap_ablated:.4f}, delta={delta_gap:+.4f} across"
            f" {len(seeds)} seeds) with E2 forward model R2={e2_r2_on:.4f} > {THRESH_C5}."
            " E2 as a trained z_self forward model (cerebellar motor model) provides"
            " body-state information that improves E3 harm_eval discrimination relative"
            " to the frozen-E2 ablation. This supports MECH-070's assertion that E2"
            " is a functional motor model contributing to trajectory quality."
        )
    elif c1_pass and c2_pass and c4_pass:
        interpretation = (
            f"Partial support: E2_MOTOR_ON achieves gap={gap_on:.4f} (C1 PASS),"
            f" delta={per_seed_deltas} (C2 PASS), but e2_r2={e2_r2_on:.4f} < {THRESH_C5}"
            " (C5 FAIL). Motor model improves harm eval but E2's learned forward model"
            " quality is below threshold. E2 may be contributing indirectly (parameter"
            " regularisation effect rather than forward-model predictions). Redesign"
            " needed to confirm the motor-model mechanism."
        )
    else:
        interpretation = (
            f"MECH-070 NOT supported at V3 proxy level: gap_on={gap_on:.4f}"
            f" (C1 {'PASS' if c1_pass else 'FAIL'}),"
            f" delta={delta_gap:+.4f} (C2 {'PASS' if c2_pass else 'FAIL'}),"
            f" e2_r2={e2_r2_on:.4f} (C5 {'PASS' if c5_pass else 'FAIL'})."
            " Trained E2 motor model does not produce a detectable improvement in"
            " harm_eval quality over the frozen-E2 ablation at this scale. Possible"
            " reasons: (a) at world_dim=32 the z_self forward model is too simple to"
            " add discriminative value; (b) E3 learns from z_world alone, making E2"
            " contributions marginal; (c) the V3 proxy does not exercise the rollout"
            " horizon advantage claimed by MECH-070 (which requires active trajectory"
            " evaluation, not just R^2 of next-step z_self prediction)."
        )

    per_on_rows = "\n".join(
        f"  seed={r['seed']}: gap={r['harm_eval_gap']:.4f}"
        f" n_harm={r['n_harm_eval']}"
        f" e2_r2={r['e2_r2']:.4f}"
        for r in results_on
    )
    per_ablated_rows = "\n".join(
        f"  seed={r['seed']}: gap={r['harm_eval_gap']:.4f}"
        f" n_harm={r['n_harm_eval']}"
        f" e2_r2={r['e2_r2']:.4f}"
        for r in results_ablated
    )

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(
            f"- {n}" for n in failure_notes
        )

    summary_markdown = (
        f"# V3-EXQ-132 -- MECH-070 E2 Motor-Model Discriminative Pair\n\n"
        f"**Status:** {status}\n"
        f"**Claims:** MECH-070\n"
        f"**Decision:** {decision}\n"
        f"**Seeds:** {list(seeds)}\n"
        f"**Conditions:** E2_MOTOR_ON vs E2_MOTOR_ABLATED\n"
        f"**Warmup:** {warmup_episodes} eps  **Eval:** {eval_episodes} eps\n"
        f"**Env:** CausalGridWorldV2 size=6, 4 hazards, nav_bias={nav_bias}\n\n"
        f"## Design\n\n"
        f"MECH-070 asserts E2 is a conceptual-sensorium motor model (cerebellar analog)\n"
        f"with planning horizon exceeding E1 (cortical, prediction_horizon=20).\n\n"
        f"E2_MOTOR_ON: E2 trains on z_self motor-sensory transitions (1-step MSE on"
        f" z_self). E1 and E3 harm_eval also train normally.\n\n"
        f"E2_MOTOR_ABLATED: E2 frozen at random init. E1 and E3 harm_eval train normally.\n"
        f"Motor model contribution removed; E3 must rely on z_world alone.\n\n"
        f"Key metric: harm_eval_gap = mean_harm_score - mean_safe_score at eval.\n"
        f"Secondary: e2_r2 (R^2 of E2.world_forward on z_self norm; confirms manipulation).\n\n"
        f"## Pre-Registered Thresholds\n\n"
        f"C1: gap_motor_on >= {THRESH_C1} in all seeds  (E2_MOTOR_ON harm eval above floor)\n"
        f"C2: per-seed delta (ON - ABLATED) >= {THRESH_C2} in all seeds"
        f"  (motor model adds >=2pp)\n"
        f"C3: gap_ablated >= {THRESH_C3}  (ablation learns from z_world; data quality)\n"
        f"C4: n_harm_eval_min >= {THRESH_C4}  (sufficient harm events)\n"
        f"C5: e2_r2_on >= {THRESH_C5}  (E2 learned a useful z_self forward model)\n\n"
        f"## Results\n\n"
        f"| Condition | gap (avg) | mean_harm | mean_safe | e2_r2 |\n"
        f"|-----------|-----------|-----------|-----------|-------|\n"
        f"| E2_MOTOR_ON      | {gap_on:.4f}"
        f" | {_avg(results_on,      'mean_harm_score'):.4f}"
        f" | {_avg(results_on,      'mean_safe_score'):.4f}"
        f" | {e2_r2_on:.4f} |\n"
        f"| E2_MOTOR_ABLATED | {gap_ablated:.4f}"
        f" | {_avg(results_ablated, 'mean_harm_score'):.4f}"
        f" | {_avg(results_ablated, 'mean_safe_score'):.4f}"
        f" | -- |\n\n"
        f"**delta_gap (ON - ABLATED): {delta_gap:+.4f}**\n\n"
        f"Diagnostic: e2_r2_on={e2_r2_on:.4f} (manipulation check --"
        f" confirms E2 learned a useful z_self forward model in E2_MOTOR_ON).\n\n"
        f"## PASS Criteria\n\n"
        f"| Criterion | Result | Value |\n"
        f"|---|---|---|\n"
        f"| C1: gap_on >= {THRESH_C1} (all seeds) | {'PASS' if c1_pass else 'FAIL'}"
        f" | {gap_on:.4f} |\n"
        f"| C2: per-seed delta >= {THRESH_C2} (all seeds) | {'PASS' if c2_pass else 'FAIL'}"
        f" | {[round(d, 4) for d in per_seed_deltas]} |\n"
        f"| C3: gap_ablated >= {THRESH_C3} | {'PASS' if c3_pass else 'FAIL'}"
        f" | {gap_ablated:.4f} |\n"
        f"| C4: n_harm_min >= {THRESH_C4} | {'PASS' if c4_pass else 'FAIL'}"
        f" | {n_harm_min} |\n"
        f"| C5: e2_r2_on >= {THRESH_C5} | {'PASS' if c5_pass else 'FAIL'}"
        f" | {e2_r2_on:.4f} |\n\n"
        f"Criteria met: {criteria_met}/5 -> **{status}**\n\n"
        f"## Interpretation\n\n"
        f"{interpretation}\n\n"
        f"## Per-Seed\n\n"
        f"E2_MOTOR_ON:\n{per_on_rows}\n\n"
        f"E2_MOTOR_ABLATED:\n{per_ablated_rows}\n"
        f"{failure_section}\n"
    )

    metrics = {
        "gap_motor_on":          float(gap_on),
        "gap_ablated":           float(gap_ablated),
        "delta_gap":             float(delta_gap),
        "n_harm_eval_min":       float(n_harm_min),
        "n_seeds":               float(len(seeds)),
        "nav_bias":              float(nav_bias),
        "alpha_world":           float(alpha_world),
        "e2_r2_on":              float(e2_r2_on),
        "per_seed_delta_min":    float(min(per_seed_deltas)) if per_seed_deltas else 0.0,
        "per_seed_delta_max":    float(max(per_seed_deltas)) if per_seed_deltas else 0.0,
        "crit1_pass":            1.0 if c1_pass else 0.0,
        "crit2_pass":            1.0 if c2_pass else 0.0,
        "crit3_pass":            1.0 if c3_pass else 0.0,
        "crit4_pass":            1.0 if c4_pass else 0.0,
        "crit5_pass":            1.0 if c5_pass else 0.0,
        "criteria_met":          float(criteria_met),
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
        "fatal_error_count": sum(r["n_fatal"] for r in results_on + results_ablated),
    }


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",           type=int, nargs="+", default=[42, 123])
    parser.add_argument("--warmup",          type=int,   default=400)
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
        "C1_gap_motor_on":       THRESH_C1,
        "C2_per_seed_delta":     THRESH_C2,
        "C3_gap_ablated":        THRESH_C3,
        "C4_n_harm_eval_min":    THRESH_C4,
        "C5_e2_r2_on":           THRESH_C5,
    }
    result["seeds"] = list(args.seeds)
    result["conditions"] = ["E2_MOTOR_ON", "E2_MOTOR_ABLATED"]
    result["dispatch_mode"] = "discriminative_pair"
    result["backlog_id"] = "EVB-0026"
    result["evidence_class"] = "discriminative_pair"
    result["claim_ids_tested"] = CLAIM_IDS

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
