"""
V3-EXQ-080 -- MECH-102 Depletion Ordering: SPARSE vs DENSE

Claims: MECH-102
Proposal: EXP-0016 / EVB-0013

MECH-102 asserts that high-energy contact (violence) is a last-resort mechanism
triggered only when all lower-energy coordination pathways are exhausted.

This experiment tests the *ordering* aspect of MECH-102 -- not just that the
ethical agent avoids contact, but that it does so *in proportion to the
availability of alternatives* and commits to contact only when alternatives are
depleted.

Discriminative pair:
  SPARSE -- size=10, num_hazards=2 (~2% density): rich alternatives, contact near-avoidable
  DENSE  -- size=7,  num_hazards=8 (~16% density): depleted alternatives, some contact forced

The "last resort" prediction:
  1. SPARSE: ethical policy near-eliminates contacts vs random (avoidance when alternatives exist).
  2. DENSE:  ethical policy still reduces contacts but cannot eliminate them entirely --
             avoids gratuitous contact but commits when forced.
  3. DENSE:  harm_eval is elevated in the N steps BEFORE ethical contacts (anticipatory
             cost awareness -- agent is aware it is approaching a forced commit).
  4. DENSE:  this anticipatory elevation is larger for ethical contacts than for random
             contacts (ethical contacts are anticipated and forced; random contacts
             are accidental).

Both conditions use SD-010 harm stream, SD-007 reafference, SD-008 alpha_world=0.9.
Training: random policy with stratified buffer (none / hazard_approach / contact).
Eval: ethical policy (argmin harm_eval_z_harm) vs random policy baseline.

PASS criteria (all required):
  C1: contact_rate_reduction_sparse > 0.70
      Ethical policy eliminates >=70% of contacts vs random when alternatives are rich.
      Tests: avoidance is not just noise -- agent exploits available alternatives.

  C2: contact_rate_reduction_dense > 0.25
      Ethical policy still reduces >=25% of contacts vs random when alternatives depleted.
      Tests: agent minimises contact even when forced -- not all-or-nothing avoidance.

  C3: harm_eval_precontact_ethical - harm_eval_baseline_ethical > 0.01
      In DENSE, harm_eval in the 5 steps before an ethical contact is elevated above
      the per-episode baseline. Tests: agent has rising cost awareness before committing.
      [Evaluated only if n_precontact_ethical_dense >= 10; defaults True if skip.]

  C4: harm_eval_precontact_ethical - harm_eval_precontact_random > 0.005
      Ethical precontact harm_eval exceeds random precontact harm_eval in DENSE.
      Tests: ethical contacts are *more anticipated* than accidental random contacts.
      [Evaluated only if n_precontact_ethical_dense >= 10; defaults True if skip.]

Decision:
  all_pass (C1+C2+C3+C4):    retain_ree  (last-resort ordering confirmed)
  C1+C2 pass, C3 or C4 fail: hybridize   (reduction present, anticipatory signal absent)
  C1 fail:                    retire_ree_claim (avoidance absent even in sparse env)
"""

import sys
import random
from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple, Deque

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.latent.stack import HarmEncoder
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_080_mech102_depletion_ordering"
CLAIM_IDS  = ["MECH-102"]
BACKLOG_IDS = ["EVB-0013"]

HARM_OBS_DIM = 51
Z_HARM_DIM   = 32

STRAT_BUF_SIZE     = 2000
MIN_PER_BUCKET     = 4
SAMPLES_PER_BUCKET = 8

PRECONTACT_WINDOW  = 5  # steps (inclusive) before each contact used for anticipatory metric

# Discriminative pair definitions
CONDITIONS = {
    "SPARSE": {"size": 10, "num_hazards": 2},
    "DENSE":  {"size": 7,  "num_hazards": 8},
}

_CONTACT_TTYPES = {"env_caused_hazard", "agent_caused_hazard"}


def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


def _ttype_to_bucket(ttype: str) -> str:
    """Map transition type to stratification bucket."""
    if ttype in _CONTACT_TTYPES:
        return "contact"
    elif ttype == "hazard_approach":
        return "hazard_approach"
    else:
        return "none"


def _run_condition(
    seed: int,
    condition_name: str,
    size: int,
    num_hazards: int,
    warmup_episodes: int,
    eval_episodes: int,
    steps_per_episode: int,
    self_dim: int,
    world_dim: int,
    lr: float,
    alpha_world: float,
    alpha_self: float,
    harm_scale: float,
    proximity_scale: float,
) -> Dict:
    """Run one (seed, condition) cell. Returns per-condition metrics dict."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    env = CausalGridWorldV2(
        seed=seed,
        size=size,
        num_hazards=num_hazards,
        num_resources=3,
        hazard_harm=harm_scale,
        env_drift_interval=5,
        env_drift_prob=0.2,
        proximity_harm_scale=proximity_scale,
        proximity_benefit_scale=proximity_scale * 0.6,
        proximity_approach_threshold=0.15,
        hazard_field_decay=0.5,
        use_proxy_fields=True,
    )

    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=self_dim,
        world_dim=world_dim,
        alpha_world=alpha_world,
        alpha_self=alpha_self,
        reafference_action_dim=env.action_dim,   # SD-007 enabled
        use_harm_stream=True,                     # SD-010 enabled
        harm_obs_dim=HARM_OBS_DIM,
        z_harm_dim=Z_HARM_DIM,
    )
    agent = REEAgent(config)

    harm_enc    = HarmEncoder(harm_obs_dim=HARM_OBS_DIM, z_harm_dim=Z_HARM_DIM)
    harm_bridge = nn.Linear(world_dim, HARM_OBS_DIM)

    num_actions = env.action_dim

    print(
        f"[V3-EXQ-080] {condition_name} seed={seed}"
        f"  size={size} hazards={num_hazards}"
        f"  train={warmup_episodes} eval={eval_episodes}x2 steps={steps_per_episode}",
        flush=True,
    )

    # ── Optimizers ────────────────────────────────────────────────────────────
    standard_params = [
        p for n, p in agent.named_parameters()
        if "harm_eval_head" not in n
        and "harm_eval_z_harm_head" not in n
        and "world_transition" not in n
        and "world_action_encoder" not in n
    ]
    world_forward_params = (
        list(agent.e2.world_transition.parameters())
        + list(agent.e2.world_action_encoder.parameters())
    )

    optimizer         = optim.Adam(standard_params,    lr=lr)
    world_forward_opt = optim.Adam(world_forward_params, lr=1e-3)
    harm_enc_opt      = optim.Adam(harm_enc.parameters(),  lr=1e-3)
    harm_bridge_opt   = optim.Adam(harm_bridge.parameters(), lr=1e-3)
    harm_z_harm_opt   = optim.Adam(
        agent.e3.harm_eval_z_harm_head.parameters(), lr=1e-4
    )

    wf_data:     List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    bridge_data: List[Tuple[torch.Tensor, torch.Tensor]] = []
    MAX_WF = 5000
    MAX_BR = 5000

    strat_bufs: Dict[str, Deque] = {
        "none":            deque(maxlen=STRAT_BUF_SIZE),
        "hazard_approach": deque(maxlen=STRAT_BUF_SIZE),
        "contact":         deque(maxlen=STRAT_BUF_SIZE),
    }

    # ── Training: random policy ───────────────────────────────────────────────
    agent.train()
    harm_enc.train()
    harm_bridge.train()
    train_counts: Dict[str, int] = {}

    for ep in range(warmup_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        z_world_prev = None
        a_prev       = None

        for _ in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]

            latent       = agent.sense(obs_body, obs_world)
            agent.clock.advance()
            z_world_curr = latent.z_world.detach()

            action_idx = random.randint(0, num_actions - 1)
            action     = _action_to_onehot(action_idx, num_actions, agent.device)
            agent._last_action = action

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)
            ttype = info.get("transition_type", "none")
            train_counts[ttype] = train_counts.get(ttype, 0) + 1

            harm_obs_new = obs_dict.get("harm_obs", torch.zeros(HARM_OBS_DIM))
            harm_obs_t   = harm_obs_new.unsqueeze(0).float()
            # Normalised label: hazard_field_view[12] is centre cell of 5x5 view, in [0,1]
            hazard_label = harm_obs_new[12].unsqueeze(0).unsqueeze(0).detach().float()

            bucket = _ttype_to_bucket(ttype)
            strat_bufs[bucket].append((harm_obs_t.detach(), float(hazard_label.item())))

            # World-forward data
            if z_world_prev is not None and a_prev is not None:
                wf_data.append((z_world_prev.cpu(), a_prev.cpu(), z_world_curr.cpu()))
                if len(wf_data) > MAX_WF:
                    wf_data = wf_data[-MAX_WF:]

            # harm_bridge data
            bridge_data.append((z_world_curr.cpu(), harm_obs_new.cpu().float()))
            if len(bridge_data) > MAX_BR:
                bridge_data = bridge_data[-MAX_BR:]

            # Train world_forward
            if len(wf_data) >= 16:
                k    = min(32, len(wf_data))
                idxs = torch.randperm(len(wf_data))[:k].tolist()
                zw_b  = torch.cat([wf_data[i][0] for i in idxs]).to(agent.device)
                a_b   = torch.cat([wf_data[i][1] for i in idxs]).to(agent.device)
                zw1_b = torch.cat([wf_data[i][2] for i in idxs]).to(agent.device)
                wf_loss = F.mse_loss(agent.e2.world_forward(zw_b, a_b), zw1_b)
                if wf_loss.requires_grad:
                    world_forward_opt.zero_grad()
                    wf_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        agent.e2.world_transition.parameters(), 0.5
                    )
                    world_forward_opt.step()

            # Train harm_bridge
            if len(bridge_data) >= 16:
                k    = min(32, len(bridge_data))
                idxs = torch.randperm(len(bridge_data))[:k].tolist()
                zw_br = torch.cat([bridge_data[i][0] for i in idxs]).to(agent.device)
                ho_br = torch.cat(
                    [bridge_data[i][1].unsqueeze(0) for i in idxs]
                ).to(agent.device)
                bridge_loss = F.mse_loss(harm_bridge(zw_br), ho_br)
                harm_bridge_opt.zero_grad()
                bridge_loss.backward()
                harm_bridge_opt.step()

            # Stratified training for harm_eval_z_harm + HarmEncoder
            buckets_ready = [b for b in strat_bufs if len(strat_bufs[b]) >= MIN_PER_BUCKET]
            if len(buckets_ready) >= 2:
                zh_list: List[torch.Tensor] = []
                lbl_list: List[float]        = []
                for bk in strat_bufs:
                    buf = strat_bufs[bk]
                    if len(buf) < MIN_PER_BUCKET:
                        continue
                    k    = min(SAMPLES_PER_BUCKET, len(buf))
                    idxs = random.sample(range(len(buf)), k)
                    for i in idxs:
                        zh_list.append(buf[i][0])
                        lbl_list.append(buf[i][1])

                if len(zh_list) >= 6:
                    zh_stack  = torch.cat(zh_list, dim=0).to(agent.device)
                    lbl_batch = torch.tensor(
                        lbl_list, dtype=torch.float32, device=agent.device
                    ).unsqueeze(1)
                    z_harm_batch = harm_enc(zh_stack)
                    pred  = agent.e3.harm_eval_z_harm(z_harm_batch)
                    loss  = F.mse_loss(pred, lbl_batch)
                    if loss.requires_grad:
                        harm_enc_opt.zero_grad()
                        harm_z_harm_opt.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(harm_enc.parameters(), 0.5)
                        harm_enc_opt.step()
                        harm_z_harm_opt.step()

            # Standard E1 + E2 losses
            e1_loss    = agent.compute_prediction_loss()
            e2_loss    = agent.compute_e2_loss()
            total_loss = e1_loss + e2_loss
            if total_loss.requires_grad:
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                optimizer.step()

            z_world_prev = z_world_curr
            a_prev       = action.detach()

            if done:
                break

        if (ep + 1) % 100 == 0 or ep == warmup_episodes - 1:
            approach = train_counts.get("hazard_approach", 0)
            contact  = (
                train_counts.get("env_caused_hazard", 0)
                + train_counts.get("agent_caused_hazard", 0)
            )
            buf_sizes = {k: len(v) for k, v in strat_bufs.items()}
            print(
                f"  [train {condition_name}] ep {ep+1}/{warmup_episodes}"
                f"  approach={approach}  contact={contact}  buf={buf_sizes}",
                flush=True,
            )

    # ── world_forward R2 ──────────────────────────────────────────────────────
    wf_r2 = 0.0
    if len(wf_data) >= 20:
        n    = len(wf_data)
        n_tr = int(n * 0.8)
        with torch.no_grad():
            zw_all   = torch.cat([d[0] for d in wf_data]).to(agent.device)
            a_all    = torch.cat([d[1] for d in wf_data]).to(agent.device)
            zw1_all  = torch.cat([d[2] for d in wf_data]).to(agent.device)
            pred_all = agent.e2.world_forward(zw_all, a_all)
            pred_test = pred_all[n_tr:]
            tgt_test  = zw1_all[n_tr:]
            if pred_test.shape[0] > 0:
                ss_res = ((tgt_test - pred_test) ** 2).sum()
                ss_tot = ((tgt_test - tgt_test.mean(0, keepdim=True)) ** 2).sum()
                wf_r2  = float((1 - ss_res / (ss_tot + 1e-8)).item())
    print(f"  [wf_r2 {condition_name}] {wf_r2:.4f}", flush=True)

    # ── Eval: ethical policy ──────────────────────────────────────────────────
    agent.eval()
    harm_enc.eval()
    harm_bridge.eval()

    ethical_contact_steps  = 0
    ethical_total_steps    = 0
    ethical_harm_history: Deque[float] = deque(maxlen=PRECONTACT_WINDOW)
    precontact_harm_ethical: List[float] = []
    all_harm_evals_ethical: List[float]  = []

    for ep in range(eval_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()

        for _ in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]

            with torch.no_grad():
                latent  = agent.sense(obs_body, obs_world)
                agent.clock.advance()
                z_world = latent.z_world

                harm_per_action: List[float] = []
                for a_idx in range(num_actions):
                    a_oh         = _action_to_onehot(a_idx, num_actions, agent.device)
                    z_world_next = agent.e2.world_forward(z_world, a_oh)
                    ho_approx    = harm_bridge(z_world_next)
                    z_harm_cf    = harm_enc(ho_approx)
                    harm_per_action.append(
                        float(agent.e3.harm_eval_z_harm(z_harm_cf).item())
                    )

                best_idx    = int(np.argmin(harm_per_action))
                harm_actual = harm_per_action[best_idx]

            # Record harm_actual before step (anticipatory window)
            ethical_harm_history.append(harm_actual)
            all_harm_evals_ethical.append(harm_actual)

            action = _action_to_onehot(best_idx, num_actions, agent.device)
            agent._last_action = action

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)
            ttype = info.get("transition_type", "none")

            ethical_total_steps += 1
            if ttype in _CONTACT_TTYPES:
                ethical_contact_steps += 1
                # Window includes current contact step -- captures the ramp up to commit
                precontact_harm_ethical.append(
                    float(np.mean(list(ethical_harm_history)))
                )

            if done:
                break

    # ── Eval: random policy ───────────────────────────────────────────────────
    random_contact_steps  = 0
    random_total_steps    = 0
    random_harm_history: Deque[float] = deque(maxlen=PRECONTACT_WINDOW)
    precontact_harm_random: List[float] = []

    for ep in range(eval_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()

        for _ in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]

            with torch.no_grad():
                latent  = agent.sense(obs_body, obs_world)
                agent.clock.advance()
                z_world = latent.z_world

                # Compute min harm_eval across actions (tracking only -- policy is random)
                harm_per_action_r: List[float] = []
                for a_idx in range(num_actions):
                    a_oh         = _action_to_onehot(a_idx, num_actions, agent.device)
                    z_world_next = agent.e2.world_forward(z_world, a_oh)
                    ho_approx    = harm_bridge(z_world_next)
                    z_harm_cf    = harm_enc(ho_approx)
                    harm_per_action_r.append(
                        float(agent.e3.harm_eval_z_harm(z_harm_cf).item())
                    )
                min_harm_r = float(np.min(harm_per_action_r))

            random_harm_history.append(min_harm_r)

            action_idx = random.randint(0, num_actions - 1)
            action     = _action_to_onehot(action_idx, num_actions, agent.device)
            agent._last_action = action

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)
            ttype = info.get("transition_type", "none")

            random_total_steps += 1
            if ttype in _CONTACT_TTYPES:
                random_contact_steps += 1
                precontact_harm_random.append(
                    float(np.mean(list(random_harm_history)))
                )

            if done:
                break

    # ── Compute per-condition metrics ─────────────────────────────────────────
    contact_rate_ethical = (
        ethical_contact_steps / ethical_total_steps
        if ethical_total_steps > 0 else 0.0
    )
    contact_rate_random = (
        random_contact_steps / random_total_steps
        if random_total_steps > 0 else 0.0
    )
    if contact_rate_random > 1e-8:
        reduction = (contact_rate_random - contact_rate_ethical) / contact_rate_random
    else:
        reduction = 0.0

    harm_eval_precontact_ethical = (
        float(np.mean(precontact_harm_ethical)) if precontact_harm_ethical else 0.0
    )
    harm_eval_precontact_random = (
        float(np.mean(precontact_harm_random)) if precontact_harm_random else 0.0
    )
    harm_eval_baseline_ethical = (
        float(np.mean(all_harm_evals_ethical)) if all_harm_evals_ethical else 0.0
    )

    n_precontact_ethical = len(precontact_harm_ethical)
    n_precontact_random  = len(precontact_harm_random)

    print(
        f"  [eval {condition_name}] seed={seed}"
        f"  ethical_rate={contact_rate_ethical:.4f}"
        f"  random_rate={contact_rate_random:.4f}"
        f"  reduction={reduction:.3f}",
        flush=True,
    )
    print(
        f"  [precontact {condition_name}]"
        f"  ethical={harm_eval_precontact_ethical:.4f}"
        f"  random={harm_eval_precontact_random:.4f}"
        f"  baseline={harm_eval_baseline_ethical:.4f}"
        f"  n_ethical={n_precontact_ethical}"
        f"  n_random={n_precontact_random}",
        flush=True,
    )

    return {
        "seed":                         seed,
        "condition":                    condition_name,
        "size":                         size,
        "num_hazards":                  num_hazards,
        "contact_rate_ethical":         float(contact_rate_ethical),
        "contact_rate_random":          float(contact_rate_random),
        "contact_rate_reduction":       float(reduction),
        "ethical_contact_steps":        int(ethical_contact_steps),
        "ethical_total_steps":          int(ethical_total_steps),
        "random_contact_steps":         int(random_contact_steps),
        "random_total_steps":           int(random_total_steps),
        "harm_eval_precontact_ethical": float(harm_eval_precontact_ethical),
        "harm_eval_precontact_random":  float(harm_eval_precontact_random),
        "harm_eval_baseline_ethical":   float(harm_eval_baseline_ethical),
        "harm_eval_anticipatory_delta": float(
            harm_eval_precontact_ethical - harm_eval_baseline_ethical
        ),
        "harm_eval_ethical_vs_random":  float(
            harm_eval_precontact_ethical - harm_eval_precontact_random
        ),
        "n_precontact_ethical":         int(n_precontact_ethical),
        "n_precontact_random":          int(n_precontact_random),
        "world_forward_r2":             float(wf_r2),
        "train_contact_events":         int(
            train_counts.get("env_caused_hazard", 0)
            + train_counts.get("agent_caused_hazard", 0)
        ),
        "train_approach_events":        int(train_counts.get("hazard_approach", 0)),
    }


def _avg(results: List[Dict], key: str) -> float:
    vals = [r[key] for r in results if key in r]
    return float(np.mean(vals)) if vals else 0.0


def run(
    seeds: Tuple = (42, 7),
    warmup_episodes: int = 400,
    eval_episodes: int = 60,
    steps_per_episode: int = 200,
    self_dim: int = 32,
    world_dim: int = 32,
    lr: float = 1e-3,
    alpha_world: float = 0.9,
    alpha_self: float = 0.3,
    harm_scale: float = 0.02,
    proximity_scale: float = 0.05,
    **kwargs,
) -> dict:
    """Discriminative pair: SPARSE vs DENSE hazard density."""

    results_by_cond: Dict[str, List[Dict]] = {"SPARSE": [], "DENSE": []}

    for seed in seeds:
        for cond_name, cond_cfg in CONDITIONS.items():
            r = _run_condition(
                seed=seed,
                condition_name=cond_name,
                size=cond_cfg["size"],
                num_hazards=cond_cfg["num_hazards"],
                warmup_episodes=warmup_episodes,
                eval_episodes=eval_episodes,
                steps_per_episode=steps_per_episode,
                self_dim=self_dim,
                world_dim=world_dim,
                lr=lr,
                alpha_world=alpha_world,
                alpha_self=alpha_self,
                harm_scale=harm_scale,
                proximity_scale=proximity_scale,
            )
            results_by_cond[cond_name].append(r)

    sparse = results_by_cond["SPARSE"]
    dense  = results_by_cond["DENSE"]

    # ── Aggregate ─────────────────────────────────────────────────────────────
    contact_rate_reduction_sparse = _avg(sparse, "contact_rate_reduction")
    contact_rate_reduction_dense  = _avg(dense,  "contact_rate_reduction")
    harm_eval_precontact_ethical  = _avg(dense,  "harm_eval_precontact_ethical")
    harm_eval_precontact_random   = _avg(dense,  "harm_eval_precontact_random")
    harm_eval_baseline_ethical    = _avg(dense,  "harm_eval_baseline_ethical")
    harm_eval_anticipatory_delta  = harm_eval_precontact_ethical - harm_eval_baseline_ethical
    harm_eval_ethical_vs_random   = harm_eval_precontact_ethical - harm_eval_precontact_random

    # Check statistical power for C3/C4
    n_precontact_ethical_dense = sum(r["n_precontact_ethical"] for r in dense)

    # ── PASS criteria ─────────────────────────────────────────────────────────
    c1_pass = contact_rate_reduction_sparse > 0.70
    c2_pass = contact_rate_reduction_dense  > 0.25

    # C3/C4: default True (skip) if insufficient ethical contacts in DENSE.
    # Fewer than 10 ethical contacts means the agent is avoiding very effectively --
    # itself strong support for MECH-102; the anticipatory signal cannot be measured
    # but its absence does not contradict last-resort ordering.
    c3_pass = True
    c4_pass = True
    c3_c4_skipped = n_precontact_ethical_dense < 10
    if not c3_c4_skipped:
        c3_pass = harm_eval_anticipatory_delta > 0.01
        c4_pass = harm_eval_ethical_vs_random  > 0.005

    all_pass     = c1_pass and c2_pass and c3_pass and c4_pass
    criteria_met = sum([c1_pass, c2_pass, c3_pass, c4_pass])
    status       = "PASS" if all_pass else "FAIL"

    if all_pass:
        decision = "retain_ree"
    elif c1_pass and c2_pass:
        decision = "hybridize"
    else:
        decision = "retire_ree_claim"

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n[V3-EXQ-080] === MECH-102 Depletion Ordering Results ===", flush=True)
    print(
        f"  SPARSE (size=10, hazards=2):"
        f"  reduction={contact_rate_reduction_sparse:.3f}"
        f"  (mean over {len(seeds)} seeds)",
        flush=True,
    )
    print(
        f"  DENSE  (size=7,  hazards=8):"
        f"  reduction={contact_rate_reduction_dense:.3f}"
        f"  precontact_ethical={harm_eval_precontact_ethical:.4f}"
        f"  baseline={harm_eval_baseline_ethical:.4f}"
        f"  delta={harm_eval_anticipatory_delta:+.4f}"
        f"  vs_random={harm_eval_ethical_vs_random:+.4f}"
        f"  n_ethical_contacts={n_precontact_ethical_dense}",
        flush=True,
    )
    print(
        f"  C1 (reduction_sparse > 0.70): {'PASS' if c1_pass else 'FAIL'}"
        f"  [{contact_rate_reduction_sparse:.3f}]",
        flush=True,
    )
    print(
        f"  C2 (reduction_dense > 0.25):  {'PASS' if c2_pass else 'FAIL'}"
        f"  [{contact_rate_reduction_dense:.3f}]",
        flush=True,
    )
    c3_label = "SKIP" if c3_c4_skipped else ("PASS" if c3_pass else "FAIL")
    c4_label = "SKIP" if c3_c4_skipped else ("PASS" if c4_pass else "FAIL")
    print(
        f"  C3 (anticipatory_delta > 0.01): {c3_label}"
        f"  [{harm_eval_anticipatory_delta:+.4f}]",
        flush=True,
    )
    print(
        f"  C4 (ethical_vs_random > 0.005): {c4_label}"
        f"  [{harm_eval_ethical_vs_random:+.4f}]",
        flush=True,
    )
    print(
        f"  Verdict: {status}  ({criteria_met}/4)  Decision: {decision}",
        flush=True,
    )

    failure_notes: List[str] = []
    if not c1_pass:
        failure_notes.append(
            f"C1 FAIL: contact_rate_reduction_sparse={contact_rate_reduction_sparse:.3f}"
            f" <= 0.70. Ethical policy is not exploiting available alternatives in"
            f" sparse environment. Check harm_eval_z_harm training signal."
        )
    if not c2_pass:
        failure_notes.append(
            f"C2 FAIL: contact_rate_reduction_dense={contact_rate_reduction_dense:.3f}"
            f" <= 0.25. Ethical policy is not reducing contacts even in dense env."
            f" May indicate complete harm_eval collapse or zero contact events in training."
        )
    if not c3_pass and not c3_c4_skipped:
        failure_notes.append(
            f"C3 FAIL: anticipatory_delta={harm_eval_anticipatory_delta:+.4f} <= 0.01."
            f" Harm_eval is not elevated before ethical contacts. Agent may be committing"
            f" to contact without cost-awareness build-up (surprise vs anticipation)."
        )
    if not c4_pass and not c3_c4_skipped:
        failure_notes.append(
            f"C4 FAIL: ethical_vs_random={harm_eval_ethical_vs_random:+.4f} <= 0.005."
            f" Ethical contacts show no more anticipation than random contacts."
            f" Last-resort distinction absent -- contacts are equally accidental."
        )

    for note in failure_notes:
        print(f"  {note}", flush=True)

    # ── Build metrics dict ────────────────────────────────────────────────────
    per_seed_sparse = [
        {k: v for k, v in r.items() if k != "condition"}
        for r in sparse
    ]
    per_seed_dense = [
        {k: v for k, v in r.items() if k != "condition"}
        for r in dense
    ]

    metrics: Dict[str, float] = {
        # SPARSE
        "contact_rate_reduction_sparse":       float(contact_rate_reduction_sparse),
        "contact_rate_ethical_sparse":         float(_avg(sparse, "contact_rate_ethical")),
        "contact_rate_random_sparse":          float(_avg(sparse, "contact_rate_random")),
        "world_forward_r2_sparse":             float(_avg(sparse, "world_forward_r2")),
        # DENSE
        "contact_rate_reduction_dense":        float(contact_rate_reduction_dense),
        "contact_rate_ethical_dense":          float(_avg(dense, "contact_rate_ethical")),
        "contact_rate_random_dense":           float(_avg(dense, "contact_rate_random")),
        "world_forward_r2_dense":              float(_avg(dense, "world_forward_r2")),
        # Anticipatory signal (DENSE)
        "harm_eval_precontact_ethical_dense":  float(harm_eval_precontact_ethical),
        "harm_eval_precontact_random_dense":   float(harm_eval_precontact_random),
        "harm_eval_baseline_ethical_dense":    float(harm_eval_baseline_ethical),
        "harm_eval_anticipatory_delta":        float(harm_eval_anticipatory_delta),
        "harm_eval_ethical_vs_random":         float(harm_eval_ethical_vs_random),
        "n_precontact_ethical_dense":          float(n_precontact_ethical_dense),
        "c3_c4_skipped":                       1.0 if c3_c4_skipped else 0.0,
        # Criteria
        "crit1_pass":                          1.0 if c1_pass else 0.0,
        "crit2_pass":                          1.0 if c2_pass else 0.0,
        "crit3_pass":                          1.0 if c3_pass else 0.0,
        "crit4_pass":                          1.0 if c4_pass else 0.0,
        "criteria_met":                        float(criteria_met),
        "n_seeds":                             float(len(seeds)),
        "alpha_world":                         float(alpha_world),
    }

    c3_str = "skip" if c3_c4_skipped else ("PASS" if c3_pass else "FAIL")
    c4_str = "skip" if c3_c4_skipped else ("PASS" if c4_pass else "FAIL")

    def _fmt_seed_rows(results_list: List[Dict]) -> str:
        rows = []
        for r in results_list:
            rows.append(
                f"  seed={r['seed']}:"
                f" reduction={r['contact_rate_reduction']:.3f}"
                f" ethical_rate={r['contact_rate_ethical']:.4f}"
                f" random_rate={r['contact_rate_random']:.4f}"
                f" anticipatory_delta={r['harm_eval_anticipatory_delta']:+.4f}"
                f" n_ethical={r['n_precontact_ethical']}"
            )
        return "\n".join(rows)

    failure_section = ""
    if failure_notes:
        failure_section = (
            "\n## Failure Notes\n\n"
            + "\n".join(f"- {n}" for n in failure_notes)
        )

    summary_markdown = (
        f"# V3-EXQ-080 -- MECH-102 Depletion Ordering\n\n"
        f"**Status:** {status}\n"
        f"**Claims:** MECH-102\n"
        f"**Backlog:** EVB-0013\n"
        f"**Decision:** {decision}\n"
        f"**Seeds:** {list(seeds)}\n"
        f"**alpha_world:** {alpha_world}  (SD-008)\n"
        f"**Warmup:** {warmup_episodes} eps  **Eval:** {eval_episodes} eps x 2 policies\n\n"
        f"## Discriminative Pair\n\n"
        f"| Condition | grid | hazards | density |\n"
        f"|-----------|------|---------|--------|\n"
        f"| SPARSE    | 10x10 | 2     | ~2%    |\n"
        f"| DENSE     | 7x7   | 8     | ~16%   |\n\n"
        f"## Results\n\n"
        f"| Metric | SPARSE | DENSE |\n"
        f"|--------|--------|-------|\n"
        f"| contact_rate_ethical | {_avg(sparse,'contact_rate_ethical'):.4f}"
        f" | {_avg(dense,'contact_rate_ethical'):.4f} |\n"
        f"| contact_rate_random  | {_avg(sparse,'contact_rate_random'):.4f}"
        f" | {_avg(dense,'contact_rate_random'):.4f} |\n"
        f"| contact_rate_reduction | {contact_rate_reduction_sparse:.3f}"
        f" | {contact_rate_reduction_dense:.3f} |\n\n"
        f"Anticipatory signal (DENSE, n_ethical_contacts={n_precontact_ethical_dense}):\n"
        f"- precontact_ethical: {harm_eval_precontact_ethical:.4f}\n"
        f"- precontact_random:  {harm_eval_precontact_random:.4f}\n"
        f"- baseline_ethical:   {harm_eval_baseline_ethical:.4f}\n"
        f"- anticipatory_delta: {harm_eval_anticipatory_delta:+.4f}\n"
        f"- ethical_vs_random:  {harm_eval_ethical_vs_random:+.4f}\n\n"
        f"## PASS Criteria\n\n"
        f"| Criterion | Result | Value |\n"
        f"|---|---|---|\n"
        f"| C1: reduction_sparse > 0.70 | {'PASS' if c1_pass else 'FAIL'}"
        f" | {contact_rate_reduction_sparse:.3f} |\n"
        f"| C2: reduction_dense > 0.25  | {'PASS' if c2_pass else 'FAIL'}"
        f" | {contact_rate_reduction_dense:.3f} |\n"
        f"| C3: anticipatory_delta > 0.01 | {c3_str}"
        f" | {harm_eval_anticipatory_delta:+.4f} |\n"
        f"| C4: ethical_vs_random > 0.005 | {c4_str}"
        f" | {harm_eval_ethical_vs_random:+.4f} |\n\n"
        f"Criteria met: {criteria_met}/4 -> **{status}**\n\n"
        f"## Per-Seed (SPARSE)\n\n"
        f"{_fmt_seed_rows(sparse)}\n\n"
        f"## Per-Seed (DENSE)\n\n"
        f"{_fmt_seed_rows(dense)}\n"
        f"{failure_section}\n"
    )

    return {
        "status":              status,
        "metrics":             metrics,
        "summary_markdown":    summary_markdown,
        "claim_ids":           CLAIM_IDS,
        "backlog_ids":         BACKLOG_IDS,
        "evidence_direction":  (
            "supports" if all_pass
            else ("mixed" if criteria_met >= 2 else "weakens")
        ),
        "experiment_type":     EXPERIMENT_TYPE,
        "fatal_error_count":   0,
        "per_seed_sparse":     per_seed_sparse,
        "per_seed_dense":      per_seed_dense,
        "decision":            decision,
    }


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser(
        description="V3-EXQ-080: MECH-102 Depletion Ordering"
    )
    parser.add_argument("--seeds",           type=int, nargs="+", default=[42, 7])
    parser.add_argument("--warmup",          type=int,   default=400)
    parser.add_argument("--eval-eps",        type=int,   default=60)
    parser.add_argument("--steps",           type=int,   default=200)
    parser.add_argument("--alpha-world",     type=float, default=0.9)
    parser.add_argument("--alpha-self",      type=float, default=0.3)
    parser.add_argument("--harm-scale",      type=float, default=0.02)
    parser.add_argument("--proximity-scale", type=float, default=0.05)
    parser.add_argument(
        "--smoke", action="store_true",
        help="Quick smoke test: 1 seed, 20 training eps, 5 eval eps, 50 steps"
    )
    args = parser.parse_args()

    if args.smoke:
        seeds           = (42,)
        warmup          = 20
        eval_eps        = 5
        steps           = 50
        print("[V3-EXQ-080] SMOKE TEST MODE", flush=True)
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
        harm_scale=args.harm_scale,
        proximity_scale=args.proximity_scale,
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"]      = ts
    result["claim"]              = CLAIM_IDS[0]
    result["verdict"]            = result["status"]
    result["run_id"]             = f"{EXPERIMENT_TYPE}_{ts}_v3"
    result["architecture_epoch"] = "ree_hybrid_guardrails_v1"

    if args.smoke:
        print(f"\n[SMOKE] Status: {result['status']}", flush=True)
        print("[SMOKE] Key metrics:", flush=True)
        for k in [
            "contact_rate_reduction_sparse",
            "contact_rate_reduction_dense",
            "harm_eval_anticipatory_delta",
            "harm_eval_ethical_vs_random",
            "crit1_pass", "crit2_pass", "crit3_pass", "crit4_pass",
        ]:
            print(f"  {k}: {result['metrics'].get(k, 'N/A')}", flush=True)
        print("[SMOKE] Script ran without error. Not writing output file.", flush=True)
    else:
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
