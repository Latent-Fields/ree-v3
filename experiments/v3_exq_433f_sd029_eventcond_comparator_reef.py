#!/opt/local/bin/python3
"""
V3-EXQ-433f -- SD-029 Comparator Attenuation: Reef-Enriched Substrate.

Corrected rerun fixing BreathOscillator-disabled bug (Bug 1: breath_period=0 in all
prior runs) and _committed_step_idx saturation bug (Bug 2: counter saturated at
H-1=29 and never reset between consecutive E3 commits in non-bistable path).

--- Original 433e docstring below ---
V3-EXQ-433e -- SD-029 Comparator Attenuation: Reef-Enriched Substrate.

Claims: SD-029, MECH-256
Supersedes: V3-EXQ-433d (non_contributory due to monostrategy: agent adopts single
fixed route, preventing balanced agent-vs-env event distributions for C3/C4).

Reef enrichment (SD-050): reef_enabled=True adds coral-reef safe zones + food-attracted
hazard drift to CausalGridWorldV2. Creates two behavioral attractors ("flee to reef" vs
"forage") to break monostrategy. reef_patch_radius=1 for 8x8 grid (radius=2 would
cover too much of the small grid).

Root cause of EXQ-433c (2026-04-23):
  EXQ-433c used curriculum_interval=25, num_hazards=3, hazard_harm=0.04,
  adjacent_only=False. The EXQ-479 substrate validation (2026-04-24, ran AFTER
  433c) demonstrated those params do not produce balanced events reliably:
  agent dies before curriculum tick fires often enough; non-adjacent injections
  rarely become agent_caused via the scripted walker. EXQ-479 PASSed substrate
  validation with calibrated params (interval=10, num_hazards=2, hazard_harm=0.02,
  adjacent_only=True) -- that env config reliably injects events across all seeds.

Fix:
  Adopt EXQ-479's calibrated curriculum + env params. Keep the EXQ-433c
  three-phase pipeline (P0 encoder warmup, P1 interventional E2_harm_s training,
  balanced eval with scripted agent_caused elicitation). adjacent_only=True
  synergises with the scripted walker: hazards now reliably appear adjacent ->
  walker steps onto them -> agent_caused fires; proximity damage from the
  injected adjacent hazard fires env_caused on the same or next step.

  - num_hazards   3   -> 2     (479 calibrated)
  - hazard_harm   0.04 -> 0.02 (479 calibrated; agent survives past curriculum tick)
  - curriculum_interval  25 -> 10  (479 calibrated; ~20 opportunities/eval ep)
  - adjacent_only False -> True   (479 calibrated; pairs with scripted walker)
  - size          10 -> 8        (479 calibrated)
  - num_resources 4  -> 3        (479 calibrated)
  - proximity_harm_scale 0.12 -> 0.1 (479 calibrated)

Adds diagnostic metric C2b: per-event-type forward_r2 (from EXQ-433c).
Adds diagnostic metric C2c: per-event-type causal_sig (from EXQ-433c).

Protocol: 4 seeds x (200 P0 + 200 P1 + balanced_eval)
  Curriculum ON throughout all three phases.
  P0: HarmEncoder warmup (use_interventional=False).
  P1: Frozen-encoder E2_harm_s head training with interventional_fraction=0.7.
  Balanced eval: scripted agent-caused elicitation + organic env_caused events
                 from curriculum; target 20 trials per event type per seed.

PASS criteria (unchanged from EXQ-433c):
  C0 (gate): agent_caused_trials_sufficient AND env_caused_trials_sufficient
             in >=3/4 seeds.
  C1 (MECH-256): forward_r2_mean >= 0.9 in >=3/4 seeds (over event-balanced
                 pairs, not a monoculture of any one type).
  C2 (SD-029): attenuation_ratio in [0.2, 0.8] in >=3/4 seeds.
  C3: approach_snr > 5.0 in >=3/4 seeds.

If C0 fails, the run is FAIL but evidence_direction flips to
"inconclusive_insufficient_events" on SD-029 and MECH-256 -- governance does NOT
downgrade the claims from this run (the experiment did not produce interpretable
data). Same protective design as EXQ-433c.

PASS = C0 AND C1 AND C2 AND C3.

claim_ids: ["SD-029", "MECH-256"]
experiment_purpose: "evidence"
"""

import sys
import json
import argparse
import math
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig
from ree_core.predictors.e2_harm_s import E2HarmSConfig, E2HarmSForward


EXPERIMENT_TYPE = "v3_exq_433f_sd029_eventcond_comparator_reef"
CLAIM_IDS = ["SD-029", "MECH-256"]
EXPERIMENT_PURPOSE = "evidence"

SEEDS = [42, 7, 13, 99]
STEPS_PER_EP = 200
P0_EPS = 200
P1_EPS = 200
INTERVENTIONAL_FRACTION = 0.7
INTERVENTIONAL_MARGIN = 0.2

# Curriculum (SD-029) -- EXQ-479 calibrated params
CURRICULUM_INTERVAL = 10
CURRICULUM_PROB = 1.0
CURRICULUM_ADJACENT_ONLY = True

# Env -- EXQ-479 calibrated params
GRID_SIZE = 8
NUM_HAZARDS = 2
NUM_RESOURCES = 3
HAZARD_HARM = 0.02
PROXIMITY_HARM_SCALE = 0.1

TARGET_TRIALS_PER_TYPE = 20
EVAL_MAX_EPS = 400  # safety cap on eval episodes
EVENT_TYPES = ["env_caused_hazard", "agent_caused_hazard", "hazard_approach", "none"]


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed,
        size=GRID_SIZE,
        num_hazards=NUM_HAZARDS,
        num_resources=NUM_RESOURCES,
        hazard_harm=HAZARD_HARM,
        proximity_harm_scale=PROXIMITY_HARM_SCALE,
        proximity_benefit_scale=0.18,
        proximity_approach_threshold=0.2,
        hazard_field_decay=0.5,
        energy_decay=0.005,
        use_proxy_fields=True,
        resource_respawn_on_consume=True,
        harm_history_len=10,
        # SD-029 curriculum -- EXQ-479 calibrated params
        scheduled_external_hazard_enabled=True,
        scheduled_external_hazard_interval=CURRICULUM_INTERVAL,
        scheduled_external_hazard_prob=CURRICULUM_PROB,
        scheduled_external_hazard_adjacent_only=CURRICULUM_ADJACENT_ONLY,
        # SD-050 reef enrichment -- breaks monostrategy on 8x8 grid
        reef_enabled=True,
        n_reef_patches=3,
        reef_patch_radius=1,
        hazard_food_attraction=0.7,
    )


def _make_harm_fwd(agent: REEAgent) -> E2HarmSForward:
    z_dim = agent.config.latent.z_harm_dim
    a_dim = agent.config.e2.action_dim
    return E2HarmSForward(
        E2HarmSConfig(
            use_e2_harm_s_forward=True,
            z_harm_dim=z_dim,
            action_dim=a_dim,
            hidden_dim=128,
            action_enc_dim=16,
            learning_rate=5e-4,
            use_interventional=True,
            interventional_fraction=INTERVENTIONAL_FRACTION,
            interventional_margin=INTERVENTIONAL_MARGIN,
        )
    )


def _make_agent(env: CausalGridWorldV2, seed: int) -> REEAgent:
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=32,
        alpha_world=0.9,
        alpha_self=0.3,
        use_harm_stream=True,
        harm_obs_dim=51,
        use_affective_harm_stream=True,
        harm_obs_a_dim=50,
        harm_history_len=10,
        z_harm_a_dim=16,
    )
    return REEAgent(cfg)


def _obs_tensors(obs_dict):
    body = obs_dict["body_state"].float().unsqueeze(0)
    world = obs_dict["world_state"].float().unsqueeze(0)
    harm = obs_dict["harm_obs"].float().unsqueeze(0) if "harm_obs" in obs_dict else None
    harm_a = obs_dict["harm_obs_a"].float().unsqueeze(0) if "harm_obs_a" in obs_dict else None
    harm_hist = obs_dict["harm_history"].float().unsqueeze(0) if "harm_history" in obs_dict else None
    return body, world, harm, harm_a, harm_hist


def _r2(preds: torch.Tensor, targets: torch.Tensor) -> float:
    ss_res = float(((targets - preds) ** 2).sum().item())
    ss_tot = float(((targets - targets.mean(dim=0)) ** 2).sum().item())
    if ss_tot < 1e-8:
        return 0.0
    return 1.0 - ss_res / ss_tot


def _find_adjacent_hazard_action(env: CausalGridWorldV2) -> Optional[int]:
    """Return a move action that steps the agent onto an adjacent hazard cell.
    Returns None if no hazard is adjacent.
    Action encoding matches CausalGridWorldV2.ACTIONS:
      0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1), 4: NOOP.
    """
    ax, ay = env.agent_x, env.agent_y
    hazard_set = {(h[0], h[1]) for h in env.hazards}
    for action_idx, (dx, dy) in env.ACTIONS.items():
        if action_idx == 4:
            continue
        if env.toroidal:
            nx = (ax + dx) % env.size
            ny = (ay + dy) % env.size
        else:
            nx, ny = ax + dx, ay + dy
            if not (0 <= nx < env.size and 0 <= ny < env.size):
                continue
        if (nx, ny) in hazard_set:
            return action_idx
    return None


def _run_seed(seed: int, verbose: bool = True) -> Dict:
    torch.manual_seed(seed)
    random.seed(seed)

    env = _make_env(seed)
    agent = _make_agent(env, seed)
    harm_fwd = _make_harm_fwd(agent)
    optim = torch.optim.Adam(harm_fwd.parameters(), lr=5e-4)

    # Phase 0: encoder warmup (no interventional training)
    harm_fwd_p0 = E2HarmSForward(
        E2HarmSConfig(
            use_e2_harm_s_forward=True,
            z_harm_dim=harm_fwd.config.z_harm_dim,
            action_dim=harm_fwd.config.action_dim,
            hidden_dim=128,
            action_enc_dim=16,
            use_interventional=False,
        )
    )
    optim_p0 = torch.optim.Adam(harm_fwd_p0.parameters(), lr=5e-4)

    z_buf: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    MAX_BUF = 2000

    print(f"  P0 ({P0_EPS} eps) encoder warmup [curriculum ON, 479 params]...", flush=True)
    prev_z_harm: Optional[torch.Tensor] = None
    for ep in range(P0_EPS):
        agent.reset()
        _obs, obs_dict = env.reset()
        for _ in range(STEPS_PER_EP):
            body, world, harm, harm_a, harm_hist = _obs_tensors(obs_dict)
            latent = agent.sense(obs_body=body, obs_world=world, obs_harm=harm,
                                 obs_harm_a=harm_a, obs_harm_history=harm_hist)
            ticks = agent.clock.advance()
            wdim = latent.z_world.shape[-1]
            e1_prior = (
                agent._e1_tick(latent) if ticks.get("e1_tick", False)
                else torch.zeros(1, wdim, device=agent.device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            action = agent.select_action(candidates, ticks)
            z_harm_curr = latent.z_harm.detach() if latent.z_harm is not None else None
            if prev_z_harm is not None and z_harm_curr is not None:
                z_buf.append((prev_z_harm.cpu(), action.detach().cpu(), z_harm_curr.cpu()))
                if len(z_buf) > MAX_BUF:
                    z_buf = z_buf[-MAX_BUF:]
                if len(z_buf) >= 16:
                    k = min(32, len(z_buf))
                    idxs = torch.randperm(len(z_buf))[:k].tolist()
                    zb = torch.cat([z_buf[i][0] for i in idxs])
                    ab = torch.cat([z_buf[i][1] for i in idxs])
                    z1b = torch.cat([z_buf[i][2] for i in idxs])
                    z_pred = harm_fwd_p0(zb.detach(), ab.detach())
                    loss = harm_fwd_p0.compute_loss(z_pred, z1b.detach())
                    optim_p0.zero_grad()
                    loss.backward()
                    optim_p0.step()
            prev_z_harm = z_harm_curr
            _obs, _, done, _, obs_dict = env.step(action)
            if done:
                break

    harm_fwd.load_state_dict(harm_fwd_p0.state_dict(), strict=False)

    print(f"  P1 ({P1_EPS} eps) interventional [curriculum ON, 479 params]...", flush=True)
    prev_z_harm = None
    z_buf_p1: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for ep in range(P1_EPS):
        agent.reset()
        _obs, obs_dict = env.reset()
        for _ in range(STEPS_PER_EP):
            body, world, harm, harm_a, harm_hist = _obs_tensors(obs_dict)
            with torch.no_grad():
                latent = agent.sense(obs_body=body, obs_world=world, obs_harm=harm,
                                     obs_harm_a=harm_a, obs_harm_history=harm_hist)
                ticks = agent.clock.advance()
                wdim = latent.z_world.shape[-1]
                e1_prior = (
                    agent._e1_tick(latent) if ticks.get("e1_tick", False)
                    else torch.zeros(1, wdim, device=agent.device)
                )
                candidates = agent.generate_trajectories(latent, e1_prior, ticks)
                action = agent.select_action(candidates, ticks)
                z_harm_curr = latent.z_harm.detach() if latent.z_harm is not None else None
            if prev_z_harm is not None and z_harm_curr is not None:
                z_buf_p1.append((prev_z_harm.cpu(), action.cpu(), z_harm_curr.cpu()))
                if len(z_buf_p1) > MAX_BUF:
                    z_buf_p1 = z_buf_p1[-MAX_BUF:]
                if len(z_buf_p1) >= 16:
                    k = min(32, len(z_buf_p1))
                    idxs = torch.randperm(len(z_buf_p1))[:k].tolist()
                    zb = torch.cat([z_buf_p1[i][0] for i in idxs])
                    ab = torch.cat([z_buf_p1[i][1] for i in idxs])
                    z1b = torch.cat([z_buf_p1[i][2] for i in idxs])
                    z_pred = harm_fwd(zb.detach(), ab.detach())
                    loss = harm_fwd.compute_loss(z_pred, z1b.detach())
                    n_int = max(1, int(INTERVENTIONAL_FRACTION * k))
                    for _ in range(n_int):
                        idx_i = random.randint(0, k - 1)
                        a_cf_idx = random.randint(0, env.action_dim - 1)
                        a_cf = torch.zeros_like(ab[idx_i:idx_i+1])
                        a_cf[0, a_cf_idx] = 1.0
                        loss_int = harm_fwd.compute_interventional_loss(
                            zb[idx_i:idx_i+1].detach(),
                            ab[idx_i:idx_i+1].detach(),
                            a_cf.detach(),
                        )
                        loss = loss + loss_int / n_int
                    optim.zero_grad()
                    loss.backward()
                    optim.step()
            prev_z_harm = z_harm_curr
            _obs, _, done, _, obs_dict = env.step(action)
            if done:
                break

    # Balanced evaluation: scripted agent-caused elicitation + organic env_caused
    # events from the curriculum. 479 calibration: adjacent_only=True means the
    # curriculum injects hazards adjacent to the agent, which the scripted walker
    # then steps onto -- so a single curriculum tick reliably produces both
    # event types in close succession.
    print("  Balanced eval [curriculum ON, 479 params + scripted agent-caused]...", flush=True)
    event_pairs: Dict[str, List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = {
        et: [] for et in EVENT_TYPES
    }
    causal_sigs: Dict[str, List[float]] = {et: [] for et in EVENT_TYPES}

    action_dim = env.action_dim
    trials_done = {et: 0 for et in EVENT_TYPES}

    for ep in range(EVAL_MAX_EPS):
        if all(trials_done[et] >= TARGET_TRIALS_PER_TYPE for et in EVENT_TYPES):
            break
        agent.reset()
        _obs, obs_dict = env.reset()
        prev_z_harm = None
        prev_action = None
        prev_etype = "none"
        for _ in range(STEPS_PER_EP):
            body, world, harm, harm_a, harm_hist = _obs_tensors(obs_dict)
            with torch.no_grad():
                latent = agent.sense(obs_body=body, obs_world=world, obs_harm=harm,
                                     obs_harm_a=harm_a, obs_harm_history=harm_hist)
                ticks = agent.clock.advance()
                wdim = latent.z_world.shape[-1]
                e1_prior = (
                    agent._e1_tick(latent) if ticks.get("e1_tick", False)
                    else torch.zeros(1, wdim, device=agent.device)
                )
                candidates = agent.generate_trajectories(latent, e1_prior, ticks)
                # Scripted elicitation: if an adjacent hazard exists AND we
                # still need agent_caused samples, step onto it deterministically.
                scripted_action_idx = None
                if trials_done["agent_caused_hazard"] < TARGET_TRIALS_PER_TYPE:
                    scripted_action_idx = _find_adjacent_hazard_action(env)
                if scripted_action_idx is not None:
                    action = torch.zeros(1, action_dim)
                    action[0, scripted_action_idx] = 1.0
                else:
                    action = agent.select_action(candidates, ticks)
            _obs, harm_signal, done, info, obs_dict = env.step(action)
            etype = info.get("transition_type", "none")
            z_harm_curr = latent.z_harm.detach().cpu() if latent.z_harm is not None else None
            if prev_z_harm is not None and z_harm_curr is not None and prev_action is not None:
                if prev_etype in trials_done and trials_done[prev_etype] < TARGET_TRIALS_PER_TYPE:
                    event_pairs[prev_etype].append(
                        (prev_z_harm, prev_action.cpu(), z_harm_curr)
                    )
                    trials_done[prev_etype] += 1
                    with torch.no_grad():
                        z_prev_t = prev_z_harm.unsqueeze(0) if prev_z_harm.dim() == 1 else prev_z_harm
                        z_next_t = z_harm_curr.unsqueeze(0) if z_harm_curr.dim() == 1 else z_harm_curr
                        a_act_t = prev_action.cpu()
                        a_act_idx = int(a_act_t.argmax().item())
                        cf_candidates = [i for i in range(action_dim) if i != a_act_idx]
                        a_cf_idx = random.choice(cf_candidates) if cf_candidates else 0
                        a_cf_t = torch.zeros_like(a_act_t)
                        a_cf_t[0, a_cf_idx] = 1.0
                        z_pred_actual = harm_fwd(z_prev_t, a_act_t)
                        z_pred_cf = harm_fwd(z_prev_t, a_cf_t)
                        err_actual = float(((z_next_t - z_pred_actual) ** 2).mean().item())
                        err_cf = float(((z_next_t - z_pred_cf) ** 2).mean().item())
                        causal_sig = err_cf - err_actual
                        causal_sigs[prev_etype].append(causal_sig)
            prev_z_harm = z_harm_curr
            prev_action = action.detach()
            prev_etype = etype
            if done:
                break

    def _safe_r2_from_pairs(pairs):
        if not pairs:
            return 0.0
        with torch.no_grad():
            zb = torch.cat([p[0].unsqueeze(0) if p[0].dim() == 1 else p[0] for p in pairs])
            ab = torch.cat([p[1] for p in pairs])
            z1b = torch.cat([p[2].unsqueeze(0) if p[2].dim() == 1 else p[2] for p in pairs])
            zpred = harm_fwd(zb, ab)
            return _r2(zpred, z1b)

    r2_per_type = {et: _safe_r2_from_pairs(event_pairs[et]) for et in EVENT_TYPES}
    types_with_data = [et for et in EVENT_TYPES if len(event_pairs[et]) >= 5]
    if types_with_data:
        mean_r2 = sum(r2_per_type[et] for et in types_with_data) / len(types_with_data)
    else:
        mean_r2 = 0.0
    mean_causal = {
        et: (sum(causal_sigs[et]) / len(causal_sigs[et]) if causal_sigs[et] else 0.0)
        for et in EVENT_TYPES
    }

    def _mean_residual(pairs):
        if not pairs:
            return 0.0
        with torch.no_grad():
            zb = torch.cat([p[0].unsqueeze(0) if p[0].dim() == 1 else p[0] for p in pairs])
            ab = torch.cat([p[1] for p in pairs])
            z1b = torch.cat([p[2].unsqueeze(0) if p[2].dim() == 1 else p[2] for p in pairs])
            zpred = harm_fwd(zb, ab)
            return float(((z1b - zpred).abs()).mean().item())

    res_env = _mean_residual(event_pairs["env_caused_hazard"])
    res_agent = _mean_residual(event_pairs["agent_caused_hazard"])
    attenuation_ratio = res_env / res_agent if res_agent > 1e-8 else float("nan")

    res_approach = _mean_residual(event_pairs["hazard_approach"])
    res_none = _mean_residual(event_pairs["none"])
    approach_snr = res_approach / res_none if res_none > 1e-8 else float("nan")

    agent_caused_sufficient = trials_done["agent_caused_hazard"] >= TARGET_TRIALS_PER_TYPE
    env_caused_sufficient = trials_done["env_caused_hazard"] >= TARGET_TRIALS_PER_TYPE

    result = {
        "seed": seed,
        "forward_r2_mean": float(mean_r2),
        "forward_r2_per_type": {k: float(v) for k, v in r2_per_type.items()},
        "attenuation_ratio": float(attenuation_ratio) if not math.isnan(attenuation_ratio) else None,
        "approach_snr": float(approach_snr) if not math.isnan(approach_snr) else None,
        "causal_sig_per_type": {k: float(v) for k, v in mean_causal.items()},
        "residual_per_type": {
            "env_caused": float(res_env),
            "agent_caused": float(res_agent),
            "approach": float(res_approach),
            "none": float(res_none),
        },
        "trials_collected": {k: len(v) for k, v in event_pairs.items()},
        "agent_caused_trials_sufficient": bool(agent_caused_sufficient),
        "env_caused_trials_sufficient": bool(env_caused_sufficient),
    }

    if verbose:
        ar = f"{attenuation_ratio:.3f}" if not math.isnan(attenuation_ratio) else "nan"
        snr = f"{approach_snr:.1f}" if not math.isnan(approach_snr) else "nan"
        print(
            f"  [seed={seed}] r2={mean_r2:.3f} attn_ratio={ar} snr={snr} "
            f"n_agent={trials_done['agent_caused_hazard']} "
            f"n_env={trials_done['env_caused_hazard']}",
            flush=True,
        )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    if args.dry_run:
        print("Smoke: seed=42, tiny P0=2/P1=3 steps=15 target_trials=1")
        global P0_EPS, P1_EPS, TARGET_TRIALS_PER_TYPE, STEPS_PER_EP, EVAL_MAX_EPS
        P0_EPS, P1_EPS, TARGET_TRIALS_PER_TYPE, STEPS_PER_EP, EVAL_MAX_EPS = 2, 3, 1, 15, 5
        r = _run_seed(seed=42, verbose=True)
        assert r["forward_r2_mean"] is not None
        print("Smoke test PASSED")
        return

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        script_dir = Path(__file__).resolve().parents[1]
        out_dir = (
            script_dir.parent / "REE_assembly" / "evidence"
            / "experiments" / EXPERIMENT_TYPE
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for seed in SEEDS:
        print(f"\nSeed {seed}", flush=True)
        r = _run_seed(seed=seed, verbose=True)
        all_results.append(r)

    def safe(v, default=0.0):
        return v if v is not None else default

    c0_wins = sum(
        1 for r in all_results
        if r["agent_caused_trials_sufficient"] and r["env_caused_trials_sufficient"]
    )
    c0 = c0_wins >= 3

    c1_wins = sum(1 for r in all_results if safe(r["forward_r2_mean"]) >= 0.9)
    c1 = c1_wins >= 3

    c2_wins = sum(
        1 for r in all_results
        if r["attenuation_ratio"] is not None and 0.2 <= r["attenuation_ratio"] <= 0.8
    )
    c2 = c2_wins >= 3

    c3_wins = sum(1 for r in all_results if safe(r["approach_snr"]) > 5.0)
    c3 = c3_wins >= 3

    outcome = "PASS" if (c0 and c1 and c2 and c3) else "FAIL"

    summary = {
        "c0_trials_sufficient_gate": {
            "wins": c0_wins,
            "pass": c0,
            "desc": "agent_caused AND env_caused trials >= TARGET in >=3/4 seeds",
        },
        "c1_mech256_forward_r2": {
            "wins": c1_wins,
            "threshold_r2": 0.9,
            "pass": c1,
            "desc": "forward_r2_mean >= 0.9 in >=3/4 seeds (mean over event types with n>=5)",
        },
        "c2_sd029_attenuation_ratio": {
            "wins": c2_wins,
            "pass": c2,
            "desc": "attenuation_ratio in [0.2, 0.8] in >=3/4 seeds",
        },
        "c3_approach_snr": {
            "wins": c3_wins,
            "pass": c3,
            "desc": "approach_snr > 5.0 in >=3/4 seeds",
        },
        "c2b_per_type_r2_diagnostic": {
            "per_seed_r2": [r["forward_r2_per_type"] for r in all_results],
        },
        "c2c_causal_sig_diagnostic": {
            "per_seed": [r["causal_sig_per_type"] for r in all_results],
        },
    }

    if not c0:
        ed_overall = "inconclusive_insufficient_events"
        per_claim = {
            "SD-029": "inconclusive_insufficient_events",
            "MECH-256": "inconclusive_insufficient_events",
        }
    else:
        per_claim = {
            "SD-029": "supports" if (c1 and c2) else ("mixed" if c1 else "does_not_support"),
            "MECH-256": "supports" if c1 else "weakens",
        }
        ed_overall = "supports" if outcome == "PASS" else "weakens"

    print(f"\nOutcome: {outcome}", flush=True)
    for k, v in summary.items():
        if "diagnostic" not in k:
            print(f"  {k}: {v}", flush=True)

    output = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": ts,
        "outcome": outcome,
        "evidence_direction": ed_overall,
        "evidence_direction_per_claim": per_claim,
        "supersedes": "v3_exq_433e_sd029_eventcond_comparator_reef",
        "pass_criteria_summary": summary,
        "per_seed_results": all_results,
        "config": {
            "seeds": SEEDS,
            "p0_eps": P0_EPS,
            "p1_eps": P1_EPS,
            "steps_per_ep": STEPS_PER_EP,
            "interventional_fraction": INTERVENTIONAL_FRACTION,
            "interventional_margin": INTERVENTIONAL_MARGIN,
            "target_trials_per_type": TARGET_TRIALS_PER_TYPE,
            "curriculum_interval": CURRICULUM_INTERVAL,
            "curriculum_prob": CURRICULUM_PROB,
            "curriculum_adjacent_only": CURRICULUM_ADJACENT_ONLY,
            "grid_size": GRID_SIZE,
            "num_hazards": NUM_HAZARDS,
            "num_resources": NUM_RESOURCES,
            "hazard_harm": HAZARD_HARM,
            "proximity_harm_scale": PROXIMITY_HARM_SCALE,
        },
    }

    out_file = out_dir / f"{run_id}.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nWrote {out_file}", flush=True)


if __name__ == "__main__":
    main()
