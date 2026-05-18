#!/opt/local/bin/python3
"""V3-EXQ-537c: SD-029/MECH-256 single-pass comparator -- forward-model CAPACITY LIFT.

Supersedes V3-EXQ-537b (FAIL non_contributory 2026-05-08T18:02Z; UNKNOWN-classified
because emit_outcome runner-conformance hook was missing). Transitively supersedes
V3-EXQ-537 / V3-EXQ-537a (both FAIL non_contributory 2026-05-08).

DIAGNOSIS OF V3-EXQ-537b (2026-05-08T18:02Z):
    537b applied the decouple-training-and-eval-curricula fix (P0/P1/P2 train
    on baseline substrate, P3 eval flips on scheduled_external_hazard). The
    decouple did NOT recover EXQ-195's 0.91 R^2 ceiling:
      seed 0: p2_r2=0.760  harm_forward_r2=0.619  p2_grad=False  p2_steps=48000
      seed 1: p2_r2=0.678  harm_forward_r2=0.656  p2_grad=False  p2_steps=48000
      seed 2: p2_r2=0.655  harm_forward_r2=0.588  p2_grad=False  p2_steps=48000
    All three seeds exhausted P2_MAX_WINDOWS=240 (48000 SGD steps) without
    reaching the C1 graduation gate of 0.85. Held-out R^2 on the curriculum-on
    P3 substrate dropped a further 5-10 points below training R^2, suggesting
    a distribution-shift cost on top of the capacity ceiling.

    Conclusion: the bottleneck is not the training curriculum -- it is the
    forward model's capacity. The (z_harm_dim=32, hidden_dim=128) ResidualHarmForward
    in E2HarmSConfig cannot fit the joint distribution of {z_harm_s_t,
    a_actual} -> z_harm_s_{t+1} on the SD-022 limb-damage + reef substrate even
    with 48000 steps and interventional contrastive loss.

DESIGN CHANGE (537c only): forward-model capacity lift.
    E2HarmSConfig(hidden_dim=256) -- 2x lift on the residual MLP hidden width.
    No other changes vs 537b (training curriculum stays decoupled, P3 eval
    keeps scheduled_external_hazard ON for balanced n_self/n_ext).

    If 537c reaches p2_grad=True with this lift, the 0.65 R^2 plateau on 537b
    was indeed a capacity ceiling and the test is now interpretable. If 537c
    still plateaus around 0.65 even at hidden_dim=256, the bottleneck is
    elsewhere (interventional margin loss fighting fit, observation
    representation lossy, label noise) and the question becomes whether to
    redesign more substantively (537d).

    Plus: runner-conformance hook (emit_outcome) added to fix 537b's UNKNOWN
    classification.

PER-CLAIM DIRECTION GRID (unchanged from 537/537b):
    C1 PASS + C2 PASS + C3 PASS + C4 PASS:
        SD-029 = supports, MECH-256 = supports.
    C1 FAIL:
        SD-029 = non_contributory, MECH-256 = non_contributory.
        (If 537c also FAILs C1 with capacity lift, the substrate or
         architecture is genuinely incapable of supporting the test;
         route to /diagnose-errors for 537d substantive redesign.)
    C1 PASS + C2 FAIL + C4 PASS:
        SD-029 = weakens, MECH-256 = weakens.
        (Clean refutation: forward model is fit, no efference-driven
         dissociation present.)
    C1 PASS + C2 PASS + C3 FAIL:
        MECH-256 = weakens. Dissociation present but not efference-driven.
        SD-029 = mixed.
    C4 FAIL (any seed): blanket non_contributory (substrate didn't deliver
        balanced events).

Per the SD-029 spec (claims.yaml):

    residual = z_harm_s_observed - E2_harm_s(z_harm_s_{t-1}, a_actual)

    "No counterfactual branch. No E2(a_cf) call. The forward prediction on
    the actually-taken action is the reference; the residual IS the agency
    signal."

And per MECH-256 falsifier: "one forward pass. No counterfactual".

ARM_1_intact (test):
    residual = ||z_harm_s_obs(t) - E2_harm_s(z_harm_s(t-1), a_actual(t-1))||
    Expected: residual_ext > residual_self (Shergill partial-attenuation:
    forward model predicts self-caused harm, fails on ext-caused).

ARM_2_scrambled (falsification):
    residual = ||z_harm_s_obs(t) - E2_harm_s(z_harm_s(t-1), a_random)||
    Expected: residual_self ~~ residual_ext (no efference-copy specificity
    => no dissociation).

PASS criteria (unchanged from 537/537b):
    C1 forward_r2 (held-out):       harm_forward_r2 >= 0.85 in ARM_1
    C2 dissociation (ARM_1):        mean_residual_ext > mean_residual_self * 1.3
                                    (Shergill ~30% partial attenuation margin)
    C3 falsification (ARM_2):       |mean_residual_ext - mean_residual_self| /
                                    max(mean_residual_self, 1e-6) < 0.15
                                    (dissociation collapses without efference)
    C4 event sufficiency:           n_self >= 20 AND n_ext >= 20 per seed
                                    in BOTH arms

claim_ids: ["SD-029", "MECH-256"]
experiment_purpose: evidence
supersedes: V3-EXQ-537b
"""

import json
import sys
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.agent import REEAgent
from ree_core.utils.config import REEConfig
from ree_core.predictors.e2_harm_s import E2HarmSConfig, E2HarmSForward
from experiment_protocol import emit_outcome

EXPERIMENT_TYPE = "v3_exq_537c_sd029_capacity_lift"
QUEUE_ID = "V3-EXQ-537c"
CLAIM_IDS = ["SD-029", "MECH-256"]
EXPERIMENT_PURPOSE = "evidence"

# 537c capacity lift: forward-model hidden dim raised 128 -> 256 to address the
# p2_grad=False ceiling at 0.65-0.76 R^2 across all seeds in 537/537a/537b.
E2_HARM_S_HIDDEN_DIM = 256

FLEE_THRESHOLD = 2
HARM_OBS_DIM = 51
HAZARD_FIELD_DIMS = 25

# Training graduation gates
P1_GRAD_THRESHOLD = 0.65   # P1 HarmEncoder (linear ceiling ~0.798)
P1_GRAD_WINDOW = 200
P1_GRAD_N_REQ = 5
P1_MAX_WINDOWS = 120
P2_GRAD_THRESHOLD = 0.85   # P2 forward model -- pushed up from EXQ-535a 0.65
P2_GRAD_WINDOW = 200
P2_GRAD_N_REQ = 5
P2_MAX_WINDOWS = 240       # doubled vs EXQ-535a to give 0.85 room
P0_EPSILON = 0.4

# P3 eval (event collection) parameters
P3_EXT_INTERVAL = 5
P3_EXT_PROB = 1.0
P3_TARGET_EXT = 50
P3_TARGET_SELF = 50
P3_MAX_STEPS = 12000
P3_HELDOUT_TARGET = 1500   # held-out transitions for forward_r2 eval

# C1-C4 acceptance thresholds
C1_R2_THRESHOLD = 0.85
C2_DISSOCIATION_RATIO = 1.30   # residual_ext > residual_self * 1.3
C3_FALSIFICATION_TOL = 0.15    # |ext - self| / self < 0.15 in scrambled arm
C4_MIN_EVENTS_PER_TYPE = 20

# Interventional training
USE_INTERVENTIONAL = True
INTERVENTIONAL_FRACTION = 0.3
INTERVENTIONAL_MARGIN = 0.1


def _move_toward(ax, ay, tx, ty):
    dx = tx - ax
    dy = ty - ay
    if dx == 0 and dy == 0:
        return 4
    if abs(dx) >= abs(dy):
        return 0 if dx < 0 else 1
    return 2 if dy < 0 else 3


def _heuristic_action(env, reef_cells, use_reef, rng):
    ax, ay = env.agent_x, env.agent_y
    nearest_hz = min(
        (abs(ax - h[0]) + abs(ay - h[1]) for h in env.hazards),
        default=999,
    )
    if use_reef and nearest_hz <= FLEE_THRESHOLD and reef_cells:
        if (ax, ay) in reef_cells:
            return 4
        target = min(reef_cells, key=lambda r: abs(ax - r[0]) + abs(ay - r[1]))
        return _move_toward(ax, ay, target[0], target[1])
    if env.resources:
        target = min(env.resources, key=lambda r: abs(ax - r[0]) + abs(ay - r[1]))
        return _move_toward(ax, ay, target[0], target[1])
    return int(rng.integers(0, env.action_dim))


def _moved_toward_hazard(ax, ay, action, hazards):
    DELTAS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1), 4: (0, 0)}
    if not hazards:
        return False
    dx, dy = DELTAS.get(action, (0, 0))
    new_x, new_y = ax + dx, ay + dy
    old_dist = min(abs(ax - h[0]) + abs(ay - h[1]) for h in hazards)
    new_dist = min(abs(new_x - h[0]) + abs(new_y - h[1]) for h in hazards)
    return new_dist < old_dist


def _action_onehot(action_int, action_dim):
    oh = torch.zeros(1, action_dim)
    oh[0, action_int] = 1.0
    return oh


def _p1_train_harm_encoder(agent, harm_head, harm_opt, transitions, seed,
                            window_size, n_req, max_windows, threshold):
    """Train HarmEncoder to predict max(hazard_field_view) from harm_obs."""
    rng_idx = np.random.default_rng(seed + 100)
    consecutive = 0
    total_steps = 0
    last_r2 = 0.0
    all_losses = []

    for _w in range(max_windows):
        preds, targets, losses = [], [], []
        for _ in range(window_size):
            idx = int(rng_idx.integers(0, len(transitions)))
            h_obs_t, _, _act, _ = transitions[idx]
            harm_in = h_obs_t.unsqueeze(0)
            z_hs = agent.latent_stack.harm_encoder(harm_in)
            pred = harm_head(z_hs).squeeze()
            tgt = h_obs_t[:HAZARD_FIELD_DIMS].max()
            loss = F.mse_loss(pred, tgt)
            harm_opt.zero_grad()
            loss.backward()
            harm_opt.step()
            preds.append(pred.detach())
            targets.append(tgt.detach())
            losses.append(loss.item())
            total_steps += 1

        preds_cat = torch.stack(preds)
        tgts_cat = torch.stack(targets)
        ss_res = ((preds_cat - tgts_cat) ** 2).sum().item()
        ss_tot = ((tgts_cat - tgts_cat.mean()) ** 2).sum().item()
        r2 = 1.0 - ss_res / max(ss_tot, 1e-8)
        last_r2 = r2
        all_losses.extend(losses)

        if r2 >= threshold:
            consecutive += 1
            if consecutive >= n_req:
                mean_loss = float(np.mean(all_losses[-window_size:]))
                return True, total_steps, r2, mean_loss
        else:
            consecutive = 0

    mean_loss = float(np.mean(all_losses[-window_size:])) if all_losses else 0.0
    return False, total_steps, last_r2, mean_loss


def _p2_train_forward_model(agent, e2_opt, transitions, action_dim, seed,
                             window_size, n_req, max_windows, threshold,
                             use_interventional, interventional_fraction):
    """Train E2_harm_s forward model. Adds interventional contrastive loss
    on `interventional_fraction` of batches when use_interventional=True
    (SD-013).
    """
    n_tr = len(transitions)
    rng_p2 = np.random.default_rng(seed + 200)
    consecutive = 0
    total_steps = 0
    last_r2 = 0.0
    all_losses = []

    for _w in range(max_windows):
        preds, targets, losses = [], [], []
        for _ in range(window_size):
            i = int(rng_p2.integers(0, max(1, n_tr - 1)))
            _, z_t, act_t, _ = transitions[i]
            _, z_t1, _, _ = transitions[i + 1]
            a_oh = _action_onehot(act_t, action_dim)
            z_pred = agent.e2_harm_s(z_t.detach(), a_oh)
            mse_loss = F.mse_loss(z_pred, z_t1.detach())

            total_loss = mse_loss
            if use_interventional and rng_p2.random() < interventional_fraction:
                a_cf_int = int(rng_p2.integers(0, action_dim))
                while a_cf_int == act_t:
                    a_cf_int = int(rng_p2.integers(0, action_dim))
                a_cf_oh = _action_onehot(a_cf_int, action_dim)
                int_loss = agent.e2_harm_s.compute_interventional_loss(
                    z_t.detach(), a_oh, a_cf_oh
                )
                total_loss = mse_loss + int_loss

            e2_opt.zero_grad()
            total_loss.backward()
            e2_opt.step()
            preds.append(z_pred.detach())
            targets.append(z_t1.detach())
            losses.append(mse_loss.item())
            total_steps += 1

        preds_cat = torch.cat(preds, dim=0)
        tgts_cat = torch.cat(targets, dim=0)
        ss_res = ((preds_cat - tgts_cat) ** 2).sum().item()
        ss_tot = ((tgts_cat - tgts_cat.mean(dim=0, keepdim=True)) ** 2).sum().item()
        r2 = 1.0 - ss_res / max(ss_tot, 1e-8)
        last_r2 = r2
        all_losses.extend(losses)

        if r2 >= threshold:
            consecutive += 1
            if consecutive >= n_req:
                mean_loss = float(np.mean(all_losses[-window_size:]))
                return True, total_steps, r2, mean_loss
        else:
            consecutive = 0

    mean_loss = float(np.mean(all_losses[-window_size:])) if all_losses else 0.0
    return False, total_steps, last_r2, mean_loss


def _p3_collect_events_and_heldout(agent, env_kwargs_base, seed, action_dim,
                                    rng, dry_run=False):
    """P3 evaluation: collect harm-event tuples + held-out transitions for R^2.

    Event tuple at step t (where harm event fires):
        z_prev, a_prev, z_curr  (numpy/tensor; z_prev/curr are encoded harm latents)

    Held-out transitions: every step in P3 contributes a (z_prev, a_prev, z_curr)
    triple to a separate held-out pool, used for forward_r2 evaluation.

    Returns (events_self, events_ext, heldout, n_steps).
    """
    p3_kwargs = dict(**env_kwargs_base)
    p3_kwargs["scheduled_external_hazard_enabled"] = True
    p3_kwargs["scheduled_external_hazard_interval"] = P3_EXT_INTERVAL
    p3_kwargs["scheduled_external_hazard_prob"] = P3_EXT_PROB
    p3_kwargs["scheduled_external_hazard_adjacent_only"] = False
    p3_kwargs["seed"] = seed + 9000

    p3_env = CausalGridWorldV2(**p3_kwargs)
    reef_cells = getattr(p3_env, "_reef_cells", set())

    events_self = []
    events_ext = []
    heldout = []
    n_self = 0
    n_ext = 0
    total_steps = 0

    p3_rng = np.random.default_rng(seed + 9100)

    if dry_run:
        max_steps = 200
        target_ext = 5
        target_self = 5
        heldout_target = 50
    else:
        max_steps = P3_MAX_STEPS
        target_ext = P3_TARGET_EXT
        target_self = P3_TARGET_SELF
        heldout_target = P3_HELDOUT_TARGET

    _, obs_dict = p3_env.reset()
    reef_cells = getattr(p3_env, "_reef_cells", set())

    h_obs_prev = None
    z_prev = None
    a_prev = None

    while total_steps < max_steps:
        ax, ay = p3_env.agent_x, p3_env.agent_y
        hazards_before = list(p3_env.hazards)
        h_obs = obs_dict["harm_obs"].float()

        with torch.no_grad():
            z_curr = agent.latent_stack.harm_encoder(h_obs.unsqueeze(0)).detach()

        if p3_rng.random() < P0_EPSILON:
            action = int(p3_rng.integers(0, action_dim))
        else:
            action = _heuristic_action(p3_env, reef_cells,
                                        p3_kwargs.get("reef_enabled", False), rng)

        moved_toward = _moved_toward_hazard(ax, ay, action, hazards_before)
        _, harm_signal, done, info, obs_dict_next = p3_env.step(action)

        external = bool(info.get("external_hazard_injected", False))
        is_self = (moved_toward and not external)
        is_ext = external

        # If we have a previous step, add held-out triple (every step, not just events)
        if z_prev is not None and a_prev is not None and len(heldout) < heldout_target:
            heldout.append({
                "z_prev": z_prev.clone(),
                "a_prev": a_prev,
                "z_curr": z_curr.clone(),
            })

        # Record event tuples (need previous z and action for residual)
        if z_prev is not None and a_prev is not None:
            ev_tuple = {
                "z_prev": z_prev.clone(),
                "a_prev": a_prev,
                "z_curr": z_curr.clone(),
            }
            if is_self:
                events_self.append(ev_tuple)
                n_self += 1
            if is_ext:
                events_ext.append(ev_tuple)
                n_ext += 1

        # Save current as prev for next iteration
        z_prev = z_curr.detach().clone()
        a_prev = action

        total_steps += 1
        obs_dict = obs_dict_next

        if done:
            _, obs_dict = p3_env.reset()
            reef_cells = getattr(p3_env, "_reef_cells", set())
            # Reset prev tracking on episode boundary
            h_obs_prev = None
            z_prev = None
            a_prev = None

        if (n_self >= target_self and n_ext >= target_ext
                and len(heldout) >= heldout_target):
            break

    return events_self, events_ext, heldout, total_steps


def _eval_heldout_r2(agent, heldout, action_dim):
    """Compute forward_r2 of E2(z_prev, a_prev) -> z_curr on held-out set."""
    if not heldout:
        return 0.0, 0
    preds, targets = [], []
    with torch.no_grad():
        for ev in heldout:
            z_prev = ev["z_prev"]
            a_prev_oh = _action_onehot(ev["a_prev"], action_dim)
            z_pred = agent.e2_harm_s(z_prev, a_prev_oh)
            preds.append(z_pred.detach())
            targets.append(ev["z_curr"].detach())
    preds_cat = torch.cat(preds, dim=0)
    tgts_cat = torch.cat(targets, dim=0)
    ss_res = ((preds_cat - tgts_cat) ** 2).sum().item()
    ss_tot = ((tgts_cat - tgts_cat.mean(dim=0, keepdim=True)) ** 2).sum().item()
    r2 = 1.0 - ss_res / max(ss_tot, 1e-8)
    return r2, len(heldout)


def _eval_single_pass_residuals(agent, events_self, events_ext, action_dim,
                                  scramble_actions=False, rng_scramble=None):
    """Compute single-pass comparator residuals at events.

    For each event (z_prev, a_prev, z_curr):
        prediction = E2(z_prev, a_used)
        residual   = ||z_curr - prediction||_2

    a_used = a_prev when scramble_actions=False (intact).
    a_used = random uniform action when scramble_actions=True (falsification).
    """
    res_self = []
    res_ext = []

    with torch.no_grad():
        for ev_list, res_list in [(events_self, res_self),
                                   (events_ext, res_ext)]:
            for ev in ev_list:
                z_prev = ev["z_prev"]
                z_curr = ev["z_curr"]
                if scramble_actions and rng_scramble is not None:
                    a_used = int(rng_scramble.integers(0, action_dim))
                else:
                    a_used = ev["a_prev"]
                a_oh = _action_onehot(a_used, action_dim)
                z_pred = agent.e2_harm_s(z_prev, a_oh)
                residual = (z_curr - z_pred).norm().item()
                res_list.append(residual)

    mean_self = float(np.mean(res_self)) if res_self else 0.0
    mean_ext = float(np.mean(res_ext)) if res_ext else 0.0
    std_self = float(np.std(res_self)) if res_self else 0.0
    std_ext = float(np.std(res_ext)) if res_ext else 0.0
    return mean_self, mean_ext, std_self, std_ext, len(res_self), len(res_ext)


def _run_seed(seed_idx, seed, env_kwargs, n_episodes, steps_per_ep, action_dim,
              rng, dry_run=False):
    """Run one seed: P0 + P1 + P2 + P3, return per-seed results dict."""
    use_reef = env_kwargs.get("reef_enabled", False)
    kw = dict(**env_kwargs, seed=seed)
    env = CausalGridWorldV2(**kw)

    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        use_harm_stream=True,
        harm_obs_dim=HARM_OBS_DIM,
        z_harm_dim=32,
        use_e2_harm_s_forward=True,
    )
    cfg.latent.alpha_world = 0.9  # SD-008 fix (EXQ-195 used 0.9)
    agent = REEAgent(cfg)

    # Replace agent.e2_harm_s with one configured for interventional training.
    if USE_INTERVENTIONAL:
        harm_s_cfg = E2HarmSConfig(
            use_e2_harm_s_forward=True,
            z_harm_dim=cfg.latent.z_harm_dim,
            action_dim=action_dim,
            hidden_dim=E2_HARM_S_HIDDEN_DIM,
            use_interventional=True,
            interventional_fraction=INTERVENTIONAL_FRACTION,
            interventional_margin=INTERVENTIONAL_MARGIN,
        )
        agent.e2_harm_s = E2HarmSForward(harm_s_cfg)

    # --- P0: Data collection ---
    transitions = []
    p0_rng = np.random.default_rng(seed + 50)

    for ep in range(n_episodes):
        _, obs_dict = env.reset()
        reef_cells = getattr(env, "_reef_cells", set())

        for _step in range(steps_per_ep):
            ax, ay = env.agent_x, env.agent_y
            h_obs = obs_dict["harm_obs"].float()
            harm_obs_t = h_obs.unsqueeze(0)

            if p0_rng.random() < P0_EPSILON:
                action = int(p0_rng.integers(0, env.action_dim))
            else:
                action = _heuristic_action(env, reef_cells, use_reef, rng)

            _, harm_signal, done, info, obs_dict_next = env.step(action)
            obs_dict = obs_dict_next

            with torch.no_grad():
                z_harm_s = agent.latent_stack.harm_encoder(harm_obs_t)

            transitions.append((
                h_obs.clone(),
                z_harm_s.detach().clone(),
                action,
                float(harm_signal),
            ))

            if done:
                break

    if dry_run:
        print(f"  seed={seed_idx} P0: transitions={len(transitions)}")

    # --- P1: HarmEncoder ---
    harm_head = nn.Linear(cfg.latent.z_harm_dim, 1)
    harm_opt = torch.optim.Adam(
        list(agent.latent_stack.harm_encoder.parameters())
        + list(harm_head.parameters()),
        lr=5e-4,
    )

    p1_grad, p1_steps, p1_r2, p1_loss = _p1_train_harm_encoder(
        agent, harm_head, harm_opt, transitions, seed,
        P1_GRAD_WINDOW, P1_GRAD_N_REQ, P1_MAX_WINDOWS, P1_GRAD_THRESHOLD,
    )

    if dry_run:
        print(f"  seed={seed_idx} P1: graduated={p1_grad} steps={p1_steps} r2={p1_r2:.3f}")

    if not p1_grad:
        return {
            "seed_idx": seed_idx, "seed": seed, "phase_failed": "P1",
            "p1_grad": False, "p1_steps": p1_steps, "p1_r2": p1_r2,
            "p2_steps": 0, "p2_r2": 0.0, "p2_grad": False,
            "harm_forward_r2": 0.0, "n_self": 0, "n_ext": 0,
            "n_heldout": 0, "p3_steps": 0,
            "intact_self": 0.0, "intact_ext": 0.0,
            "scrambled_self": 0.0, "scrambled_ext": 0.0,
        }

    # Re-encode transitions through trained HarmEncoder for P2 stop-gradient discipline.
    fresh_transitions = []
    with torch.no_grad():
        for (h_obs, _z_old, act, hs) in transitions:
            z_new = agent.latent_stack.harm_encoder(h_obs.unsqueeze(0))
            fresh_transitions.append((h_obs, z_new.detach().clone(), act, hs))

    # --- P2: Forward model (with interventional) ---
    e2_opt = torch.optim.Adam(
        agent.e2_harm_s.parameters(),
        lr=5e-4,
    )
    p2_grad, p2_steps, p2_r2, p2_loss = _p2_train_forward_model(
        agent, e2_opt, fresh_transitions, env.action_dim, seed,
        P2_GRAD_WINDOW, P2_GRAD_N_REQ, P2_MAX_WINDOWS, P2_GRAD_THRESHOLD,
        USE_INTERVENTIONAL, INTERVENTIONAL_FRACTION,
    )

    if dry_run:
        print(f"  seed={seed_idx} P2: graduated={p2_grad} steps={p2_steps} r2={p2_r2:.3f}")

    # --- P3: Event collection + held-out R^2 eval ---
    events_self, events_ext, heldout, p3_steps = _p3_collect_events_and_heldout(
        agent, env_kwargs, seed, env.action_dim, rng, dry_run=dry_run,
    )

    n_self = len(events_self)
    n_ext = len(events_ext)
    n_heldout = len(heldout)

    if dry_run:
        print(f"  seed={seed_idx} P3: n_self={n_self} n_ext={n_ext} "
              f"heldout={n_heldout} steps={p3_steps}")

    harm_forward_r2, _ = _eval_heldout_r2(agent, heldout, env.action_dim)

    # ARM_1 intact
    intact_self, intact_ext, intact_self_std, intact_ext_std, _, _ = (
        _eval_single_pass_residuals(
            agent, events_self, events_ext, env.action_dim,
            scramble_actions=False,
        )
    )

    # ARM_2 scrambled
    scramble_rng = np.random.default_rng(seed + 7777)
    scrambled_self, scrambled_ext, scrambled_self_std, scrambled_ext_std, _, _ = (
        _eval_single_pass_residuals(
            agent, events_self, events_ext, env.action_dim,
            scramble_actions=True, rng_scramble=scramble_rng,
        )
    )

    return {
        "seed_idx": seed_idx, "seed": seed,
        "phase_failed": None,
        "p1_grad": p1_grad, "p1_steps": p1_steps, "p1_r2": p1_r2,
        "p1_loss": p1_loss,
        "p2_grad": p2_grad, "p2_steps": p2_steps, "p2_r2": p2_r2,
        "p2_loss": p2_loss,
        "harm_forward_r2": harm_forward_r2,
        "n_self": n_self, "n_ext": n_ext, "n_heldout": n_heldout,
        "p3_steps": p3_steps,
        "intact_self": intact_self, "intact_ext": intact_ext,
        "intact_self_std": intact_self_std, "intact_ext_std": intact_ext_std,
        "scrambled_self": scrambled_self, "scrambled_ext": scrambled_ext,
        "scrambled_self_std": scrambled_self_std, "scrambled_ext_std": scrambled_ext_std,
    }


def _aggregate(seed_results, key):
    vals = [s[key] for s in seed_results if s.get(key) is not None]
    return float(np.mean(vals)) if vals else 0.0


def _evaluate_criteria(seed_results):
    """Compute C1-C4 from per-seed results."""
    c1_r2 = _aggregate(seed_results, "harm_forward_r2")
    c1 = c1_r2 >= C1_R2_THRESHOLD

    intact_self = _aggregate(seed_results, "intact_self")
    intact_ext = _aggregate(seed_results, "intact_ext")
    c2 = intact_ext > intact_self * C2_DISSOCIATION_RATIO

    scrambled_self = _aggregate(seed_results, "scrambled_self")
    scrambled_ext = _aggregate(seed_results, "scrambled_ext")
    if scrambled_self > 1e-6:
        c3_diff_ratio = abs(scrambled_ext - scrambled_self) / scrambled_self
    else:
        c3_diff_ratio = 0.0
    c3 = c3_diff_ratio < C3_FALSIFICATION_TOL

    n_self_per_seed = [s["n_self"] for s in seed_results]
    n_ext_per_seed = [s["n_ext"] for s in seed_results]
    c4 = (
        all(n >= C4_MIN_EVENTS_PER_TYPE for n in n_self_per_seed)
        and all(n >= C4_MIN_EVENTS_PER_TYPE for n in n_ext_per_seed)
    )

    return {
        "C1_forward_r2": c1, "C1_value": c1_r2,
        "C2_dissociation": c2, "C2_intact_self": intact_self,
        "C2_intact_ext": intact_ext, "C2_ratio": intact_ext / max(intact_self, 1e-6),
        "C3_falsification": c3, "C3_scrambled_self": scrambled_self,
        "C3_scrambled_ext": scrambled_ext, "C3_diff_ratio": c3_diff_ratio,
        "C4_event_sufficiency": c4,
        "C4_min_n_self": min(n_self_per_seed) if n_self_per_seed else 0,
        "C4_min_n_ext": min(n_ext_per_seed) if n_ext_per_seed else 0,
    }


def _evidence_direction_per_claim(criteria):
    """Apply per-claim direction matrix from the docstring."""
    c1 = criteria["C1_forward_r2"]
    c2 = criteria["C2_dissociation"]
    c3 = criteria["C3_falsification"]
    c4 = criteria["C4_event_sufficiency"]

    if not c4:
        return {"SD-029": "non_contributory", "MECH-256": "non_contributory"}, "non_contributory"
    if not c1:
        return {"SD-029": "non_contributory", "MECH-256": "non_contributory"}, "non_contributory"
    # C1 + C4 hold
    if c1 and c2 and c3:
        return {"SD-029": "supports", "MECH-256": "supports"}, "supports"
    if c1 and not c2:
        return {"SD-029": "weakens", "MECH-256": "weakens"}, "weakens"
    if c1 and c2 and not c3:
        return {"SD-029": "mixed", "MECH-256": "weakens"}, "mixed"
    return {"SD-029": "mixed", "MECH-256": "mixed"}, "mixed"


def main(dry_run=False):
    rng = np.random.default_rng(20260507)

    if dry_run:
        n_seeds = 1
        n_episodes = 8
        steps_per_ep = 50
        print("DRY_RUN")
    else:
        n_seeds = 3
        n_episodes = 200
        steps_per_ep = 200

    # 537b decouples training and eval curricula:
    #   P0 / P1 / P2 (training): scheduled_external_hazard OFF -- forward
    #   model trains on the predictable substrate so it can reach EXQ-195's
    #   ~0.91 R^2 capacity (537/537a hit a 0.71 ceiling because the
    #   training pool was contaminated with by-design-unpredictable ext
    #   events).
    #   P3 (eval): scheduled_external_hazard ON, override applied inside
    #   _p3_collect_events_and_heldout (interval=5, prob=1.0). This is what
    #   produces the balanced n_self / n_ext pool SD-029 C2/C3 require.
    env_kwargs = {
        "size": 12,
        "num_hazards": 3,
        "num_resources": 5,
        "use_proxy_fields": True,
        "env_drift_prob": 0.3,
        "env_drift_interval": 1,
        "limb_damage_enabled": True,
        "harm_history_len": 10,
        "reef_enabled": True,
        "n_reef_patches": 3,
        "reef_patch_radius": 2,
        "hazard_food_attraction": 0.7,
        # 537b: training-phase curriculum OFF (P3 eval re-enables it
        # via _p3_collect_events_and_heldout overrides).
        "scheduled_external_hazard_enabled": False,
    }

    seed_results = []
    for seed_idx in range(n_seeds):
        seed = int(rng.integers(0, 100000))
        sr = _run_seed(seed_idx, seed, env_kwargs, n_episodes, steps_per_ep,
                       action_dim=5, rng=rng, dry_run=dry_run)
        seed_results.append(sr)

    criteria = _evaluate_criteria(seed_results)
    per_claim, run_direction = _evidence_direction_per_claim(criteria)

    overall_pass = (criteria["C1_forward_r2"] and criteria["C2_dissociation"]
                    and criteria["C3_falsification"] and criteria["C4_event_sufficiency"])
    outcome = "PASS" if overall_pass else "FAIL"

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{timestamp}_v3"

    manifest = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": run_direction,
        "evidence_direction_per_claim": per_claim,
        "experiment_purpose": "evidence",
        "supersedes": "v3_exq_537b_sd029_decoupled_curricula",
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "outcome": outcome,
        "criteria": criteria,
        "metrics": {
            "per_seed": seed_results,
            "n_seeds": n_seeds,
            "n_episodes": n_episodes,
            "steps_per_ep": steps_per_ep,
            "p1_grad_threshold": P1_GRAD_THRESHOLD,
            "p2_grad_threshold": P2_GRAD_THRESHOLD,
            "p2_max_windows": P2_MAX_WINDOWS,
            "use_interventional": USE_INTERVENTIONAL,
            "interventional_fraction": INTERVENTIONAL_FRACTION,
            "c1_r2_threshold": C1_R2_THRESHOLD,
            "c2_dissociation_ratio_required": C2_DISSOCIATION_RATIO,
            "c3_falsification_tol": C3_FALSIFICATION_TOL,
            "c4_min_events_per_type": C4_MIN_EVENTS_PER_TYPE,
        },
    }

    if dry_run:
        print("DRY_RUN_COMPLETE")
        print(json.dumps({"outcome": outcome, "criteria": criteria,
                           "per_claim": per_claim}, indent=2))
        return manifest

    out_dir = Path(__file__).resolve().parent.parent.parent / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.json"
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote: {out_path}")
    print(f"Result written to: {out_path}", flush=True)
    print(f"Outcome: {outcome}")
    print(json.dumps(criteria, indent=2))
    # 537c: also return out_path so __main__ can pass it to emit_outcome.
    return manifest, out_path


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    result = main(dry_run=dry_run)
    if dry_run:
        sys.exit(0)
    _manifest, _out_path = result
    _outcome_raw = str(_manifest.get("outcome", "FAIL")).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=_out_path,
    )
    sys.exit(0)
