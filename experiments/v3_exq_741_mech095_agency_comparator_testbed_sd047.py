#!/opt/local/bin/python3
"""
V3-EXQ-741 -- MECH-095 Agency-Comparator VALID TEST-BED on SD-047

Claims: MECH-095 (tpj.agency_detection_comparator)
Experiment purpose: evidence
Realizes: substrate_queue sd_id agency_comparator_testbed_sd047 (/implement-substrate)

Why this experiment exists (NOT a 3rd 047 letter)
-------------------------------------------------
MECH-095's two SD-047 retests both self-routed FAIL but were adjudicated
NON_CONTRIBUTORY (failure_autopsy_V3-EXQ-047l + _047m):
  - 047l: the env_events>0 additive fold saturated the EVAL probe partition
    (empty no-contact class) -> contact_recall pinned 0.0 in both arms.
  - 047m: the SAME fold, moved to the TRAINING is_world label, saturated it to
    ~const 1 at intensity 1.0 (2 drift sources fire env_events>0 on ~93%+ of
    steps) -> the routing head never saw a self-caused negative and homogenised
    z_world (ROUTED 0.492 < BASELINE 0.795, improvement -0.302). Its guard
    checked only the PROBE partition, not the training-label balance -> a FALSE
    non_degenerate=true.
Two non-valid reads on the same operationalisation implicate the OPERATIONALISATION,
not the substrate. The re-derive brake FIRED (2 MECH-095 non_contributory
autopsies). SD-047 the substrate is CORRECT; the missing piece is a VALID
test-bed. The sanctioned route is /implement-substrate -> this test-bed under a
NEW EXQ number (a different operationalisation + the pre-registered sweep is
explicitly brake-PERMITTED; a 3rd 047 letter is refused).

The two coupled gaps this test-bed closes
------------------------------------------
1. NON-SATURATING self/world label. The enabling ree_core change is landed:
   CausalGridWorldV2 flat kwarg tag_env_caused_multisource_ttype (default False,
   bit-identical OFF) fills a residual transition_type=="none" with
   "env_caused_multisource" when an SD-047 multi-source event fired but no
   agent-caused transition did. The label then uses transition_type directly
   (is_world = prev_ttype in WORLD_CAUSED) -- NO env_events>0 fold -- so at
   intensity 1.0 is_world sits at ~0.15 (a real self/world contrast), not the
   saturated ~0.93 the fold produced.
2. The pre-registered 4-arm SD-047 intensity sweep (ARM_0 OFF / ARM_1 0.25x /
   ARM_2 1.0x / ARM_3 4.0x, keyed ARM_0-vs-ARM_2), per
   docs/architecture/sd_047_multi_source_dynamics.md, so the Asai (2016)
   non-monotonicity (calibration-overshoot) and Woo/Spelke (2023) falsifier
   branches are both testable. A single binary point cannot distinguish
   "SD-047 works" from "SD-047 overshot calibration".

Two operationalisations, contrasted across the sweep (user chose BOTH)
----------------------------------------------------------------------
Per (arm x seed) cell, THREE conditions share the same eval partitions:
  BASELINE  -- agent E1+E2, no comparator. contact_recall_world on z_world
               (CONTACT_SET partition, 047k non-folded form -> BASELINE ~0.795).
  ROUTED_A  -- option A (gradient label): a BCE routing head trained on the
               NON-saturating label reshapes z_world by gradient. This is the
               047-lineage translation, now with a valid label. Can still corrupt
               z_world (it competes with E1+E2) -- that is the hypothesis under
               test.
  ROUTED_B  -- option B (read-out; biology-favoured per MECH-095 notes 6832-6849):
               an efference-copy forward model f_eff(z_self_t, a) predicts z_self
               change; agency_residual = ||observed dz_self - predicted dz_self||
               attributes divergence to the world. contact_recall_world_B is a
               probe on concat([z_world_baseline, agency_residual]). B NEVER
               backprops into z_world, so BASELINE and B share an IDENTICAL
               z_world -> recall_improvement_B cannot be corrupted negative
               (structurally impossible, unlike 047m's -0.302). It isolates the
               comparator's additive contribution.
B > A across arms confirms the read-out is the right functional translation
(the H1 operationalisation-fault reading); A > B or both flat routes elsewhere
(see interpretation grid below).

Guards (per arm x seed) -- the 047m false-clear fix
---------------------------------------------------
  PROBE-partition guard (KEEP from 047m): n_no_contact >= PROBE_NEG_FLOOR (5),
    else the balanced contact probe cannot run (spurious 0.0) -> that arm's read
    is not counted.
  SELF-WORLD BALANCE guard (NEW -- the training-label guard 047m LACKED):
    min(n_self_caused, n_world_caused) >= SW_BALANCE_FLOOR (5) on the
    SELF_CAUSED/WORLD_CAUSED transition_type partition. Arms failing this are
    label-degenerate for A / comparator-input-degenerate for B and are EXCLUDED
    from the discrimination read -- SURFACED in the manifest, never silently
    scored. This is the exact structure whose absence gave 047m a FALSE
    non_degenerate=true (it validated only the probe partition).

Acceptance (from the autopsy failure_record)
--------------------------------------------
A valid test-bed must (a) present a BALANCED self-vs-world label at >=1
informative arm AND (b) show recall_improvement > 0.04 (A or B) there;
BASELINE ~0.795 confirms the substrate carries the contact signal.

Interpretation grid (aggregate over the sweep)
----------------------------------------------
  Some arm passes BOTH guards AND (impr_A>0.04 or impr_B>0.04)
      -> PASS / supports. Re-gate MECH-095 (candidate/substrate_ceiling).
  All measured arms pass both guards but FLAT (no impr>0.04 anywhere)
      -> non_degenerate=true, evidence_direction=weakens, WOO_SPELKE note:
         a valid operationalisation that still cannot discriminate implicates
         motion-contingency/goal-directedness features that are V4-bound ->
         route MECH-095 substrate_ceiling -> substrate_conditional (V4-1).
  NO arm passes both guards
      -> non_degenerate=false, non_contributory (test-bed still not valid;
         should not recur given the guard, but self-reports honestly).
The inverted-U across arms (Asai) and whether ARM_2 (or ARM_1) peaks are
reported for calibration reading.

Run with:
  /opt/local/bin/python3 experiments/v3_exq_741_mech095_agency_comparator_testbed_sd047.py --dry-run
  /opt/local/bin/python3 experiments/v3_exq_741_mech095_agency_comparator_testbed_sd047.py
"""

import sys
import random
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig
from experiment_protocol import emit_outcome
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from _metrics import check_degeneracy
from experiments._lib.arm_fingerprint import arm_cell


EXPERIMENT_TYPE = "v3_exq_741_mech095_agency_comparator_testbed_sd047"
CLAIM_IDS = ["MECH-095"]
EXPERIMENT_PURPOSE = "evidence"

# Label / partition sets.
# WORLD_CAUSED: unambiguous env-driven, agent-quiet state change. env_caused_multisource
#   is the new tag from tag_env_caused_multisource_ttype (drift/transient/weather moved
#   while the agent caused nothing). NO hazard_approach here (that is agent-position-driven,
#   set from the agent's new cell) and NO env_events>0 fold (the 047 saturation bug).
WORLD_CAUSED = frozenset({"env_caused_hazard", "env_caused_multisource"})
# SELF_CAUSED: outcomes determined by where the AGENT moved / what it did.
SELF_CAUSED = frozenset({
    "agent_caused_hazard", "resource", "waypoint", "sequence_complete",
    "hazard_approach", "benefit_approach",
})
# CONTACT_SET: the 047k non-folded contact-probe partition (kept verbatim so BASELINE
#   is comparable to the 047k/047m 0.795). Distinct from the self/world label above.
CONTACT_SET = frozenset({"hazard_approach", "agent_caused_hazard", "env_caused_hazard"})

# Guards.
PROBE_NEG_FLOOR = 5      # balanced contact probe needs >= this many no-contact samples/cell.
SW_BALANCE_FLOOR = 5     # self/world label needs >= this many of BOTH classes/cell.

# SD-047 source defaults at intensity_scale=1.0 (identical to 047m / V3-EXQ-509).
WEATHER_SUPER_CELLS = 4
WEATHER_ALPHA_AR1 = 0.95
WEATHER_SIGMA = 0.10
TRANSIENT_P_APPEAR = 5e-3
TRANSIENT_P_DISAPPEAR = 0.10
N_DRIFT_SOURCES = 2
DRIFT_POLICY = "random_walk"

# The pre-registered 4-arm sweep. ms_enabled False = ARM_0 (thin-world OFF baseline).
ARMS: List[Dict] = [
    {"name": "ARM_0", "intensity": 0.0, "ms_enabled": False},
    {"name": "ARM_1", "intensity": 0.25, "ms_enabled": True},
    {"name": "ARM_2", "intensity": 1.0, "ms_enabled": True},
    {"name": "ARM_3", "intensity": 4.0, "ms_enabled": True},
]


def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


def _fit_balanced_probe(
    X_pos: torch.Tensor,
    X_neg: torch.Tensor,
    n_steps: int = 400,
    lr: float = 1e-2,
) -> float:
    """Balanced linear probe recall on the positive class (verbatim from 047k/047m)."""
    n = min(len(X_pos), len(X_neg))
    if n < 5:
        return 0.0
    idx_pos = torch.randperm(len(X_pos))[:n]
    idx_neg = torch.randperm(len(X_neg))[:n]
    X = torch.cat([X_pos[idx_pos], X_neg[idx_neg]], dim=0).float()
    y = torch.cat([torch.ones(n, dtype=torch.long), torch.zeros(n, dtype=torch.long)])
    probe = nn.Linear(X.shape[1], 2)
    opt = optim.Adam(probe.parameters(), lr=lr)
    for _ in range(n_steps):
        loss = F.cross_entropy(probe(X), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        preds = probe(X_pos[idx_pos]).argmax(dim=1)
        return float((preds == 1).float().mean().item())


def _fit_action_probe(
    X: torch.Tensor,
    y: torch.Tensor,
    n_classes: int,
    n_steps: int = 400,
    lr: float = 1e-2,
) -> float:
    X = X.detach().float()
    y = y.detach().long()
    probe = nn.Linear(X.shape[1], n_classes)
    opt = optim.Adam(probe.parameters(), lr=lr)
    for _ in range(n_steps):
        loss = F.cross_entropy(probe(X), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return float((probe(X).argmax(dim=1) == y).float().mean().item())


def _make_env(seed: int, intensity: float, ms_enabled: bool) -> CausalGridWorldV2:
    """047m's grid + all three SD-047 sources at intensity; non-saturating ttype tag ON."""
    return CausalGridWorldV2(
        seed=seed,
        size=12,
        num_hazards=4,
        num_resources=5,
        hazard_harm=0.02,
        env_drift_interval=5,
        env_drift_prob=0.1,
        proximity_harm_scale=0.05,
        proximity_benefit_scale=0.03,
        proximity_approach_threshold=0.15,
        hazard_field_decay=0.5,
        multi_source_dynamics_enabled=ms_enabled,
        multi_source_intensity_scale=(intensity if ms_enabled else 1.0),
        weather_field_enabled=ms_enabled,
        weather_super_cells=WEATHER_SUPER_CELLS,
        weather_alpha_ar1=WEATHER_ALPHA_AR1,
        weather_sigma=WEATHER_SIGMA,
        transient_events_enabled=ms_enabled,
        transient_p_appear=TRANSIENT_P_APPEAR,
        transient_p_disappear=TRANSIENT_P_DISAPPEAR,
        background_drift_enabled=ms_enabled,
        n_drift_sources=N_DRIFT_SOURCES,
        drift_policy=DRIFT_POLICY,
        # The non-saturating self/world label enabler (bit-identical OFF by default).
        tag_env_caused_multisource_ttype=True,
    )


def _build_agent(env: CausalGridWorldV2, self_dim: int, world_dim: int,
                 alpha_world: float, alpha_self: float, lr: float):
    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=self_dim,
        world_dim=world_dim,
        alpha_world=alpha_world,
        alpha_self=alpha_self,
        reafference_action_dim=0,
    )
    config.latent.unified_latent_mode = False
    agent = REEAgent(config)
    # Split the harm_eval head onto its own slow optimiser (047m pattern).
    harm_eval_params = list(agent.e3.harm_eval_head.parameters())
    return agent, harm_eval_params


def _harm_eval_step(agent, harm_buf_pos, harm_buf_neg, harm_eval_optimizer):
    if len(harm_buf_pos) >= 4 and len(harm_buf_neg) >= 4:
        k_pos = min(16, len(harm_buf_pos))
        k_neg = min(16, len(harm_buf_neg))
        pos_idx = torch.randperm(len(harm_buf_pos))[:k_pos].tolist()
        neg_idx = torch.randperm(len(harm_buf_neg))[:k_neg].tolist()
        zw_b = torch.cat(
            [harm_buf_pos[i] for i in pos_idx] + [harm_buf_neg[i] for i in neg_idx], dim=0)
        target = torch.cat([
            torch.ones(k_pos, 1, device=zw_b.device),
            torch.zeros(k_neg, 1, device=zw_b.device),
        ], dim=0)
        harm_loss = F.mse_loss(agent.e3.harm_eval(zw_b), target)
        if harm_loss.requires_grad:
            harm_eval_optimizer.zero_grad()
            harm_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.e3.harm_eval_head.parameters(), 0.5)
            harm_eval_optimizer.step()


def _train_agent(
    env, agent, harm_eval_params, *, use_routing, routing_head, f_eff,
    world_dim, self_dim, lr, lambda_route, warmup_episodes, steps_per_episode,
    ep_offset, ep_total, seed, arm_name, cond_label,
) -> Tuple[float, int]:
    """One agent training pass. Returns (mean_routing_loss, routing_steps).

    Trains E1+E2 always; optionally the BCE routing head (option A, reshapes z_world)
    and/or the efference-copy forward model f_eff (option B read-out, on DETACHED
    z_self -- never reshapes z_self). Prints [train] progress against ep_total.
    """
    standard_params = list(agent.parameters())
    if use_routing:
        standard_params = standard_params + list(routing_head.parameters())
    standard_params = [
        p for p in standard_params
        if not any(p is ph for ph in harm_eval_params)
    ]
    optimizer = optim.Adam(standard_params, lr=lr)
    harm_eval_optimizer = optim.Adam(harm_eval_params, lr=1e-4)
    f_eff_optimizer = optim.Adam(f_eff.parameters(), lr=1e-3) if f_eff is not None else None

    harm_buf_pos: List[torch.Tensor] = []
    harm_buf_neg: List[torch.Tensor] = []
    MAX_BUF = 2000
    total_routing_loss = 0.0
    routing_steps = 0

    agent.train()
    if routing_head is not None:
        routing_head.train()
    if f_eff is not None:
        f_eff.train()

    for ep in range(warmup_episodes):
        _, obs_dict = env.reset()
        agent.reset()
        prev_ttype = None
        prev_z_self = None
        prev_action_oh = None

        for _ in range(steps_per_episode):
            latent = agent.sense(obs_dict["body_state"], obs_dict["world_state"])
            agent.clock.advance()
            z_world_curr = latent.z_world
            z_self_curr = latent.z_self

            # Option A: BCE routing head on the NON-saturating transition_type label.
            routing_loss = torch.zeros(1)
            if use_routing and prev_ttype is not None:
                if prev_ttype in WORLD_CAUSED:
                    label_val = 1.0
                elif prev_ttype in SELF_CAUSED:
                    label_val = 0.0
                else:
                    label_val = None  # SKIP 'none' -- no loss on truly-quiet steps.
                if label_val is not None:
                    label = torch.tensor([[label_val]])
                    routing_loss = lambda_route * F.binary_cross_entropy_with_logits(
                        routing_head(z_world_curr), label)
                    total_routing_loss += routing_loss.item()
                    routing_steps += 1

            # Option B: efference-copy forward model on DETACHED z_self (never reshapes
            # z_self). Predict this step's z_self from the PREVIOUS z_self + action.
            f_eff_loss = torch.zeros(1)
            if f_eff is not None and prev_z_self is not None and prev_action_oh is not None:
                pred = f_eff(torch.cat([prev_z_self.detach(), prev_action_oh], dim=1))
                f_eff_loss = F.mse_loss(pred, z_self_curr.detach())

            z_world_track = z_world_curr.detach()

            action_idx = random.randint(0, env.action_dim - 1)
            action_oh = _action_to_onehot(action_idx, env.action_dim, agent.device)
            agent._last_action = action_oh

            _, harm_signal, done, info, obs_dict = env.step(action_oh)
            prev_ttype = info.get("transition_type", "none")
            prev_z_self = z_self_curr.detach()
            prev_action_oh = action_oh

            if float(harm_signal) < 0:
                harm_buf_pos.append(z_world_track)
                if len(harm_buf_pos) > MAX_BUF:
                    harm_buf_pos = harm_buf_pos[-MAX_BUF:]
            else:
                harm_buf_neg.append(z_world_track)
                if len(harm_buf_neg) > MAX_BUF:
                    harm_buf_neg = harm_buf_neg[-MAX_BUF:]

            total_loss = agent.compute_prediction_loss() + agent.compute_e2_loss() + routing_loss
            if total_loss.requires_grad:
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                optimizer.step()

            if f_eff is not None and f_eff_loss.requires_grad:
                f_eff_optimizer.zero_grad()
                f_eff_loss.backward()
                torch.nn.utils.clip_grad_norm_(f_eff.parameters(), 1.0)
                f_eff_optimizer.step()

            _harm_eval_step(agent, harm_buf_pos, harm_buf_neg, harm_eval_optimizer)

            if done:
                break

        cur = ep_offset + ep + 1
        if cur % 50 == 0 or ep == warmup_episodes - 1:
            mr = total_routing_loss / max(1, routing_steps) if use_routing else 0.0
            print(
                f"  [train] seed={seed} {arm_name} ep {cur}/{ep_total}"
                f" phase={cond_label} route_loss={mr:.4f}",
                flush=True,
            )

    mean_route = total_routing_loss / max(1, routing_steps) if use_routing else 0.0
    return mean_route, routing_steps


def _probe_agent(env, agent, f_eff, probe_episodes, steps_per_episode):
    """Frozen uniform-random probe. Collects z_self, z_world, [z_world|residual],
    action, and the SELF/WORLD + CONTACT partitions."""
    agent.eval()
    if f_eff is not None:
        f_eff.eval()

    world_contact: List[torch.Tensor] = []
    world_no_contact: List[torch.Tensor] = []
    wr_contact: List[torch.Tensor] = []       # [z_world | residual], contact
    wr_no_contact: List[torch.Tensor] = []
    self_all: List[torch.Tensor] = []
    world_all: List[torch.Tensor] = []
    wr_all: List[torch.Tensor] = []
    actions: List[int] = []
    residuals_self: List[float] = []          # residual on SELF_CAUSED steps
    residuals_world: List[float] = []         # residual on WORLD_CAUSED steps
    n_self_caused = 0
    n_world_caused = 0
    n_fatal = 0

    for _ in range(probe_episodes):
        _, obs_dict = env.reset()
        agent.reset()

        for _ in range(steps_per_episode):
            with torch.no_grad():
                latent = agent.sense(obs_dict["body_state"], obs_dict["world_state"])
                agent.clock.advance()
                zs_prev = latent.z_self.detach()

            action_idx = random.randint(0, env.action_dim - 1)
            action_oh = _action_to_onehot(action_idx, env.action_dim, agent.device)
            agent._last_action = action_oh

            _, _, done, info, obs_dict = env.step(action_oh)
            ttype = info.get("transition_type", "none")

            try:
                with torch.no_grad():
                    latent_next = agent.sense(
                        obs_dict["body_state"], obs_dict["world_state"])
                    agent.clock.advance()
                    zs = latent_next.z_self.detach()
                    zw = latent_next.z_world.detach()

                    # Read-out residual: observed dz_self vs efference prediction.
                    if f_eff is not None:
                        pred_next = f_eff(torch.cat([zs_prev, action_oh], dim=1))
                        residual = float(torch.norm(zs - pred_next).item())
                    else:
                        residual = 0.0
                    res_t = torch.tensor([[residual]], dtype=torch.float32)
                    zwr = torch.cat([zw, res_t], dim=1)

                    is_contact = ttype in CONTACT_SET
                    self_all.append(zs)
                    world_all.append(zw)
                    wr_all.append(zwr)
                    actions.append(action_idx)
                    if is_contact:
                        world_contact.append(zw)
                        wr_contact.append(zwr)
                    else:
                        world_no_contact.append(zw)
                        wr_no_contact.append(zwr)

                    # Self/world balance partition (for the guard + residual gap).
                    if ttype in WORLD_CAUSED:
                        n_world_caused += 1
                        residuals_world.append(residual)
                    elif ttype in SELF_CAUSED:
                        n_self_caused += 1
                        residuals_self.append(residual)

                    _, _, done2, _, obs_dict = env.step(
                        _action_to_onehot(
                            random.randint(0, env.action_dim - 1),
                            env.action_dim, agent.device))
                    if done2:
                        break
            except Exception:
                n_fatal += 1

            if done:
                break

    return {
        "world_contact": world_contact, "world_no_contact": world_no_contact,
        "wr_contact": wr_contact, "wr_no_contact": wr_no_contact,
        "self_all": self_all, "world_all": world_all, "wr_all": wr_all,
        "actions": actions,
        "residuals_self": residuals_self, "residuals_world": residuals_world,
        "n_self_caused": n_self_caused, "n_world_caused": n_world_caused,
        "n_fatal": n_fatal,
    }


def _recall_and_action(p, action_dim):
    """From a probe dict, compute contact recalls (world, world+residual) and action
    probe accuracies (self, world, world+residual)."""
    n_contact = len(p["world_contact"])
    n_no_contact = len(p["world_no_contact"])
    recall_world = 0.0
    recall_wr = 0.0
    if n_contact >= PROBE_NEG_FLOOR and n_no_contact >= PROBE_NEG_FLOOR:
        recall_world = _fit_balanced_probe(
            torch.cat(p["world_contact"], dim=0), torch.cat(p["world_no_contact"], dim=0))
        recall_wr = _fit_balanced_probe(
            torch.cat(p["wr_contact"], dim=0), torch.cat(p["wr_no_contact"], dim=0))
    acc_self = acc_world = acc_wr = 0.0
    if len(p["self_all"]) >= 20:
        y = torch.tensor(p["actions"], dtype=torch.long)
        acc_self = _fit_action_probe(torch.cat(p["self_all"], dim=0), y, action_dim)
        acc_world = _fit_action_probe(torch.cat(p["world_all"], dim=0), y, action_dim)
        acc_wr = _fit_action_probe(torch.cat(p["wr_all"], dim=0), y, action_dim)
    return {
        "n_contact": n_contact, "n_no_contact": n_no_contact,
        "recall_world": recall_world, "recall_wr": recall_wr,
        "acc_self": acc_self, "acc_world": acc_world, "acc_wr": acc_wr,
    }


def _run_cell(
    arm: Dict, seed: int, *, warmup_episodes, probe_episodes, steps_per_episode,
    self_dim, world_dim, lr, alpha_world, alpha_self, lambda_route, script_path,
) -> Dict:
    """One (arm x seed) cell: train baseline agent (+ efference f_eff) and routed_A
    agent, then probe all three conditions. Returns a per-cell row."""
    arm_name = arm["name"]
    intensity = float(arm["intensity"])
    ms_enabled = bool(arm["ms_enabled"])
    ep_total = 2 * warmup_episodes  # baseline pass + routed_A pass

    # config_slice for the arm fingerprint (full declared config for this cell).
    config_slice = {
        "env": {
            "size": 12, "num_hazards": 4, "num_resources": 5, "hazard_harm": 0.02,
            "env_drift_interval": 5, "env_drift_prob": 0.1, "proximity_harm_scale": 0.05,
            "proximity_benefit_scale": 0.03, "proximity_approach_threshold": 0.15,
            "hazard_field_decay": 0.5, "multi_source_dynamics_enabled": ms_enabled,
            "multi_source_intensity_scale": (intensity if ms_enabled else 1.0),
            "weather_field_enabled": ms_enabled, "weather_super_cells": WEATHER_SUPER_CELLS,
            "weather_alpha_ar1": WEATHER_ALPHA_AR1, "weather_sigma": WEATHER_SIGMA,
            "transient_events_enabled": ms_enabled, "transient_p_appear": TRANSIENT_P_APPEAR,
            "transient_p_disappear": TRANSIENT_P_DISAPPEAR, "background_drift_enabled": ms_enabled,
            "n_drift_sources": N_DRIFT_SOURCES, "drift_policy": DRIFT_POLICY,
            "tag_env_caused_multisource_ttype": True,
        },
        "agent": {
            "self_dim": self_dim, "world_dim": world_dim, "alpha_world": alpha_world,
            "alpha_self": alpha_self, "lr": lr, "lambda_route": lambda_route,
            "unified_latent_mode": False,
        },
        "schedule": {"warmup_episodes": warmup_episodes, "steps_per_episode": steps_per_episode},
        "arm": arm_name, "intensity_scale": intensity,
    }

    print(f"Seed {seed} Condition {arm_name}", flush=True)
    print(f"\n[V3-EXQ-741] {arm_name} seed={seed} intensity={intensity}"
          f" ms_enabled={ms_enabled} warmup={warmup_episodes}", flush=True)

    # A cell trains two independent agents (baseline+routed_A) with separate optimisers;
    # they share no mutable state, but the cell bundles a baseline WITH a routed pass and
    # this test-bed's operationalisation is deliberately in flux (its whole purpose is to
    # decide WHICH operationalisation is correct). Mark reuse-ineligible: a future consumer
    # must not silently reuse a bundled/in-flux baseline cell. See queue note (mint SKIP).
    with arm_cell(
        seed, config_slice=config_slice, script_path=script_path,
        config_slice_declared=True,
        extra_ineligible_reasons=[
            "testbed_operationalisation_in_flux",
            "bundled_baseline_plus_routed_cell",
        ],
    ) as cell:
        # --- BASELINE agent (routing OFF) + efference forward model f_eff (option B) ---
        env_b = _make_env(seed, intensity, ms_enabled)
        agent_b, harm_params_b = _build_agent(
            env_b, self_dim, world_dim, alpha_world, alpha_self, lr)
        f_eff = nn.Sequential(
            nn.Linear(self_dim + env_b.action_dim, 32), nn.ReLU(), nn.Linear(32, self_dim))
        _train_agent(
            env_b, agent_b, harm_params_b, use_routing=False, routing_head=None,
            f_eff=f_eff, world_dim=world_dim, self_dim=self_dim, lr=lr,
            lambda_route=lambda_route, warmup_episodes=warmup_episodes,
            steps_per_episode=steps_per_episode, ep_offset=0, ep_total=ep_total,
            seed=seed, arm_name=arm_name, cond_label="baseline")

        # --- ROUTED_A agent (BCE routing head, non-saturating label; option A) ---
        env_a = _make_env(seed, intensity, ms_enabled)
        agent_a, harm_params_a = _build_agent(
            env_a, self_dim, world_dim, alpha_world, alpha_self, lr)
        routing_head = nn.Sequential(
            nn.Linear(world_dim, 16), nn.ReLU(), nn.Linear(16, 1))
        mean_route_a, route_steps_a = _train_agent(
            env_a, agent_a, harm_params_a, use_routing=True, routing_head=routing_head,
            f_eff=None, world_dim=world_dim, self_dim=self_dim, lr=lr,
            lambda_route=lambda_route, warmup_episodes=warmup_episodes,
            steps_per_episode=steps_per_episode, ep_offset=warmup_episodes,
            ep_total=ep_total, seed=seed, arm_name=arm_name, cond_label="routedA")

        # --- PROBE (frozen) ---
        # BASELINE + ROUTED_B ride agent_b (identical z_world); ROUTED_A on agent_a.
        env_pb = _make_env(seed, intensity, ms_enabled)
        p_b = _probe_agent(env_pb, agent_b, f_eff, probe_episodes, steps_per_episode)
        env_pa = _make_env(seed, intensity, ms_enabled)
        p_a = _probe_agent(env_pa, agent_a, None, probe_episodes, steps_per_episode)

        m_b = _recall_and_action(p_b, env_b.action_dim)   # gives baseline + B (via wr)
        m_a = _recall_and_action(p_a, env_a.action_dim)   # gives A

        # Self/world balance measured on the baseline probe (identical env pressure).
        n_self = p_b["n_self_caused"]
        n_world = p_b["n_world_caused"]
        balance_ok = min(n_self, n_world) >= SW_BALANCE_FLOOR
        probe_ok = m_b["n_no_contact"] >= PROBE_NEG_FLOOR and m_b["n_contact"] >= PROBE_NEG_FLOOR

        res_self = p_b["residuals_self"]
        res_world = p_b["residuals_world"]
        mean_res = (sum(res_self + res_world) / max(1, len(res_self + res_world)))
        res_gap = (
            (sum(res_world) / max(1, len(res_world)))
            - (sum(res_self) / max(1, len(res_self)))
        ) if (res_self and res_world) else 0.0

        recall_baseline = m_b["recall_world"]      # probe on z_world
        recall_B = m_b["recall_wr"]                # probe on [z_world | residual]
        recall_A = m_a["recall_world"]             # probe on reshaped z_world_A
        dissoc_baseline = m_b["acc_self"] - m_b["acc_world"]
        dissoc_B = m_b["acc_self"] - m_b["acc_wr"]
        dissoc_A = m_a["acc_self"] - m_a["acc_world"]

        row = {
            "arm": arm_name, "seed": seed, "intensity": intensity, "ms_enabled": ms_enabled,
            "recall_baseline": float(recall_baseline),
            "recall_A": float(recall_A),
            "recall_B": float(recall_B),
            "recall_improvement_A": float(recall_A - recall_baseline),
            "recall_improvement_B": float(recall_B - recall_baseline),
            "action_dissoc_baseline": float(dissoc_baseline),
            "action_dissoc_A": float(dissoc_A),
            "action_dissoc_B": float(dissoc_B),
            "n_self_caused": int(n_self), "n_world_caused": int(n_world),
            "balance_ok": bool(balance_ok),
            "n_contact": int(m_b["n_contact"]), "n_no_contact": int(m_b["n_no_contact"]),
            "probe_ok": bool(probe_ok),
            "mean_residual": float(mean_res), "residual_world_minus_self": float(res_gap),
            "mean_routing_loss_A": float(mean_route_a),
            "routing_active_A": bool(mean_route_a > 0.001),
            "n_fatal": int(p_b["n_fatal"] + p_a["n_fatal"]),
        }
        cell.stamp(row)

    print(
        f"  [probe] seed={seed} {arm_name}"
        f" recall base={recall_baseline:.3f} A={recall_A:.3f} B={recall_B:.3f}"
        f" | impr_A={row['recall_improvement_A']:+.3f} impr_B={row['recall_improvement_B']:+.3f}"
        f" | n_self={n_self} n_world={n_world} balance={'OK' if balance_ok else 'DEGEN'}"
        f" n_no_contact={m_b['n_no_contact']} probe={'OK' if probe_ok else 'DEGEN'}"
        f" res_gap={res_gap:+.3f}",
        flush=True,
    )
    run_ok = (row["n_fatal"] == 0 and probe_ok and balance_ok)
    print(f"verdict: {'PASS' if run_ok else 'FAIL'}", flush=True)
    return row


def run(
    seeds: Tuple = (42, 7, 123, 99),
    warmup_episodes: int = 400,
    probe_episodes: int = 20,
    steps_per_episode: int = 200,
    self_dim: int = 32,
    world_dim: int = 32,
    lr: float = 1e-3,
    alpha_world: float = 0.9,
    alpha_self: float = 0.3,
    lambda_route: float = 0.1,
    **kwargs,
) -> dict:
    """4-arm SD-047 intensity sweep x seeds; three comparator conditions per cell."""
    script_path = Path(__file__)
    rows: List[Dict] = []
    for arm in ARMS:
        for seed in seeds:
            rows.append(_run_cell(
                arm, seed, warmup_episodes=warmup_episodes, probe_episodes=probe_episodes,
                steps_per_episode=steps_per_episode, self_dim=self_dim, world_dim=world_dim,
                lr=lr, alpha_world=alpha_world, alpha_self=alpha_self,
                lambda_route=lambda_route, script_path=script_path))

    # ---- Aggregate per arm ----
    def _avg(vals):
        vals = list(vals)
        return float(sum(vals) / max(1, len(vals)))

    per_arm: Dict[str, Dict] = {}
    for arm in ARMS:
        name = arm["name"]
        cells = [r for r in rows if r["arm"] == name]
        valid = [r for r in cells if r["probe_ok"] and r["balance_ok"]]
        per_arm[name] = {
            "intensity": arm["intensity"],
            "n_cells": len(cells),
            "n_valid_cells": len(valid),
            "recall_baseline": _avg(r["recall_baseline"] for r in cells),
            "recall_A": _avg(r["recall_A"] for r in cells),
            "recall_B": _avg(r["recall_B"] for r in cells),
            "recall_improvement_A": _avg(r["recall_improvement_A"] for r in cells),
            "recall_improvement_B": _avg(r["recall_improvement_B"] for r in cells),
            "action_dissoc_baseline": _avg(r["action_dissoc_baseline"] for r in cells),
            "action_dissoc_A": _avg(r["action_dissoc_A"] for r in cells),
            "action_dissoc_B": _avg(r["action_dissoc_B"] for r in cells),
            "n_self_caused_min": min((r["n_self_caused"] for r in cells), default=0),
            "n_world_caused_min": min((r["n_world_caused"] for r in cells), default=0),
            "n_no_contact_min": min((r["n_no_contact"] for r in cells), default=0),
            "residual_world_minus_self": _avg(r["residual_world_minus_self"] for r in cells),
            "mean_routing_loss_A": _avg(r["mean_routing_loss_A"] for r in cells),
            # Arm is a valid discrimination read iff a majority of its cells passed both guards.
            "arm_valid": len(valid) >= max(1, (len(cells) + 1) // 2),
            # Valid-cell improvements (the read that counts).
            "valid_improvement_A": _avg(r["recall_improvement_A"] for r in valid) if valid else 0.0,
            "valid_improvement_B": _avg(r["recall_improvement_B"] for r in valid) if valid else 0.0,
        }

    # ---- Decision logic / self-route ----
    valid_arms = [n for n, a in per_arm.items() if a["arm_valid"]]
    any_improvement = False
    best_arm = None
    best_impr = -1e9
    for n in valid_arms:
        a = per_arm[n]
        impr = max(a["valid_improvement_A"], a["valid_improvement_B"])
        if impr > best_impr:
            best_impr, best_arm = impr, n
        if impr > 0.04:
            any_improvement = True

    baseline_carries = max(
        (per_arm[n]["recall_baseline"] for n in valid_arms), default=0.0) > 0.55

    if not valid_arms:
        status = "FAIL"
        evidence_direction = "non_contributory"
        decision = "testbed_still_invalid_no_arm_passes_guards"
    elif any_improvement:
        status = "PASS"
        evidence_direction = "supports"
        decision = "valid_testbed_recall_improvement_regate_mech095"
    else:
        # Valid operationalisation(s), no discrimination anywhere -> Woo/Spelke branch.
        status = "FAIL"
        evidence_direction = "weakens"
        decision = "woo_spelke_route_substrate_conditional_v4_1"

    # B-vs-A read (which operationalisation, if any, discriminates -- the H1 signal).
    mean_valid_impr_A = _avg(per_arm[n]["valid_improvement_A"] for n in valid_arms) if valid_arms else 0.0
    mean_valid_impr_B = _avg(per_arm[n]["valid_improvement_B"] for n in valid_arms) if valid_arms else 0.0
    b_beats_a = mean_valid_impr_B > mean_valid_impr_A

    # Inverted-U (Asai) peak arm across the intensity axis (valid arms only).
    peak_arm = best_arm

    # ---- Non-degeneracy self-report ----
    # Whole run is non-degenerate iff at least one arm is a valid discrimination read.
    # Also feed the cross-arm recall spreads to check_degeneracy (all-cells-at-chance net).
    degeneracy = check_degeneracy({
        "recall_baseline": {
            "values": [r["recall_baseline"] for r in rows], "floor": 0.55},
    })
    if not valid_arms:
        degeneracy["non_degenerate"] = False
        reason = ("no arm passed BOTH the probe-partition and self-world-balance guards"
                  " -> comparator never validly exercised (test-bed not ready)")
        degeneracy["degeneracy_reason"] = (
            reason + " | " + degeneracy["degeneracy_reason"]
            if degeneracy.get("degeneracy_reason") else reason)
        dm = dict(degeneracy.get("degenerate_metrics") or {})
        dm["self_world_balance"] = reason
        degeneracy["degenerate_metrics"] = dm

    # ---- Report ----
    print(f"\n[V3-EXQ-741] Per-arm summary (4-arm SD-047 sweep, {len(seeds)} seeds):", flush=True)
    print("  arm    intens  base    A       B      impr_A  impr_B  n_self/world_min  valid", flush=True)
    for arm in ARMS:
        n = arm["name"]
        a = per_arm[n]
        print(
            f"  {n}  {a['intensity']:>5.2f}  {a['recall_baseline']:.3f}  {a['recall_A']:.3f}"
            f"  {a['recall_B']:.3f}  {a['recall_improvement_A']:+.3f}  {a['recall_improvement_B']:+.3f}"
            f"  {a['n_self_caused_min']}/{a['n_world_caused_min']}"
            f"  {'YES' if a['arm_valid'] else 'no'}",
            flush=True,
        )
    print(f"  valid_arms={valid_arms} peak_arm={peak_arm} best_impr={best_impr:+.3f}", flush=True)
    print(f"  B_beats_A={b_beats_a} (mean valid impr A={mean_valid_impr_A:+.3f}"
          f" B={mean_valid_impr_B:+.3f}) baseline_carries_contact={baseline_carries}", flush=True)
    print(f"  decision={decision} status={status} evidence_direction={evidence_direction}", flush=True)
    if not degeneracy["non_degenerate"]:
        print(f"  DEGENERATE: {degeneracy['degeneracy_reason']}"
              " -- scoring-excluded (non_contributory); route /failure-autopsy", flush=True)

    summary_markdown = (
        f"# V3-EXQ-741 -- MECH-095 Agency-Comparator VALID TEST-BED on SD-047\n\n"
        f"**Status:** {status}\n"
        f"**Claims:** MECH-095\n"
        f"**Decision:** {decision}\n"
        f"**evidence_direction:** {evidence_direction}\n"
        f"**Seeds:** {list(seeds)}\n"
        f"**Sweep:** ARM_0 OFF / ARM_1 0.25x / ARM_2 1.0x / ARM_3 4.0x"
        f" (key contrast ARM_0-vs-ARM_2)\n"
        f"**Valid arms (both guards):** {valid_arms}  peak={peak_arm}\n"
        f"**B_beats_A:** {b_beats_a} (mean valid impr A={mean_valid_impr_A:+.3f}"
        f" B={mean_valid_impr_B:+.3f})\n"
        f"**Baseline carries contact:** {baseline_carries}\n\n"
        f"## Per-arm results\n\n"
        f"| arm | intensity | recall_base | recall_A | recall_B | impr_A | impr_B |"
        f" n_self/world_min | n_no_contact_min | arm_valid |\n"
        f"|---|---|---|---|---|---|---|---|---|---|\n"
        + "".join(
            f"| {arm['name']} | {per_arm[arm['name']]['intensity']:.2f}"
            f" | {per_arm[arm['name']]['recall_baseline']:.3f}"
            f" | {per_arm[arm['name']]['recall_A']:.3f}"
            f" | {per_arm[arm['name']]['recall_B']:.3f}"
            f" | {per_arm[arm['name']]['recall_improvement_A']:+.3f}"
            f" | {per_arm[arm['name']]['recall_improvement_B']:+.3f}"
            f" | {per_arm[arm['name']]['n_self_caused_min']}"
            f"/{per_arm[arm['name']]['n_world_caused_min']}"
            f" | {per_arm[arm['name']]['n_no_contact_min']}"
            f" | {'YES' if per_arm[arm['name']]['arm_valid'] else 'no'} |\n"
            for arm in ARMS)
        + f"\n## Interpretation\n\n"
        f"- **{decision}**\n"
        f"- Read-out (B) vs gradient-label (A): "
        f"{'B > A (read-out is the better functional translation)' if b_beats_a else 'A >= B'}.\n"
        f"- If FLAT/all-valid-no-improvement -> Woo/Spelke: route MECH-095"
        f" substrate_ceiling -> substrate_conditional (V4-1 multi-agent ecology).\n"
    )

    metrics = {
        "n_valid_arms": float(len(valid_arms)),
        "best_improvement": float(best_impr if valid_arms else 0.0),
        "mean_valid_improvement_A": float(mean_valid_impr_A),
        "mean_valid_improvement_B": float(mean_valid_impr_B),
        "b_beats_a": 1.0 if b_beats_a else 0.0,
        "baseline_carries_contact": 1.0 if baseline_carries else 0.0,
        "arm2_recall_baseline": float(per_arm["ARM_2"]["recall_baseline"]),
        "arm2_recall_improvement_A": float(per_arm["ARM_2"]["recall_improvement_A"]),
        "arm2_recall_improvement_B": float(per_arm["ARM_2"]["recall_improvement_B"]),
        "arm0_recall_baseline": float(per_arm["ARM_0"]["recall_baseline"]),
    }
    for arm in ARMS:
        n = arm["name"]
        a = per_arm[n]
        for k in ("recall_baseline", "recall_A", "recall_B", "recall_improvement_A",
                  "recall_improvement_B", "action_dissoc_baseline", "action_dissoc_A",
                  "action_dissoc_B", "residual_world_minus_self", "n_valid_cells"):
            metrics[f"{n}_{k}"] = float(a[k])

    result_dict = {
        "status": status,
        "outcome": status,
        "metrics": metrics,
        "summary_markdown": summary_markdown,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "evidence_direction": evidence_direction,
        "experiment_type": EXPERIMENT_TYPE,
        "decision": decision,
        "per_arm": per_arm,
        "arm_results": rows,
        "valid_arms": valid_arms,
        "peak_arm": peak_arm,
        "b_beats_a": bool(b_beats_a),
        "fatal_error_count": sum(r["n_fatal"] for r in rows),
    }
    result_dict.update(degeneracy)
    return result_dict


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 123, 99])
    parser.add_argument("--warmup", type=int, default=400)
    parser.add_argument("--probe-eps", type=int, default=20)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--alpha-world", type=float, default=0.9)
    parser.add_argument("--alpha-self", type=float, default=0.3)
    parser.add_argument("--lambda-route", type=float, default=0.1)
    parser.add_argument("--dry-run", action="store_true",
                        help="Short smoke run (2 seeds, tiny episodes); relocates manifest.")
    args = parser.parse_args()

    if args.dry_run:
        seeds = tuple(args.seeds[:2]) if len(args.seeds) >= 2 else tuple(args.seeds)
        warmup, probe_eps, steps = 3, 3, 12
    else:
        seeds = tuple(args.seeds)
        warmup, probe_eps, steps = args.warmup, args.probe_eps, args.steps

    result = run(
        seeds=seeds,
        warmup_episodes=warmup,
        probe_episodes=probe_eps,
        steps_per_episode=steps,
        alpha_world=args.alpha_world,
        alpha_self=args.alpha_self,
        lambda_route=args.lambda_route,
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"] = ts
    result["timestamp_utc"] = ts
    result["claim"] = CLAIM_IDS[0]
    result["verdict"] = result["status"]
    result["run_id"] = f"{EXPERIMENT_TYPE}_{ts}_v3"
    result["architecture_epoch"] = "ree_hybrid_guardrails_v1"

    out_dir = (
        Path(__file__).resolve().parents[2]
        / "REE_assembly" / "evidence" / "experiments"
    )
    out_path = write_flat_manifest(
        result,
        out_dir,
        dry_run=args.dry_run,
        config=result.get("config"),
        seeds=None,
        script_path=Path(__file__),
    )

    print(f"\nResult written to: {out_path}", flush=True)
    print(f"Status: {result['status']}", flush=True)
    for k, v in result["metrics"].items():
        print(f"  {k}: {v}", flush=True)

    _outcome_raw = str(result.get("status", "FAIL")).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
        dry_run=args.dry_run,
    )
