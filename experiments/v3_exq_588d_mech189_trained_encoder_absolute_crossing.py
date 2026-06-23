"""V3-EXQ-588d -- MECH-189 super-ordinal goal-anchor TRAINED-ENCODER absolute
0.4 adult-z_goal-crossing READINESS DIAGNOSTIC.

infant_substrate:GAP-11b -- the trained-encoder evidence successor the closed
infant_substrate:GAP-11 (V3-EXQ-588c PASS) explicitly DEFERRED. 588c validated
the MECH-189 SuperOrdinalGoalMemory WRITE+READ substrate end to end via a
DISCRIMINATION criterion (ARM_ON adult median z_goal.norm 0.394 vs ARM_OFF 0.0,
3/3 seeds) on an UNTRAINED-encoder forced-feed harness. The DEV-NEED-006 0.4
ABSOLUTE crossing was a near-miss (0.394, 2/3 seeds) because the matured-anchor
norm ceilings at ~0.37 on the untrained-encoder readiness harness (the
forced-feed z_world-EMA asymptote ~0.9 * ||z_world||). 588c's manifest + the
GAP-11 closure prose routed that absolute gate to a SEPARATE trained-encoder
successor (this run): does a TRAINED z_world encoder (larger / properly-scaled
||z_world|| -> larger matured z_goal -> larger stored anchor) lift the adult
seeding readout across the 0.4 absolute gate?

BRAKE NOTE (Step 2.5b): MECH-189 carries 2 prior substrate_ceiling/non_contributory
autopsies (588_2026-05-19 substrate-unbuilt; 669a_2026-06-13 ecological-harness
goal_pipeline:GAP-2 starvation) and is epistemic_category=substrate_ceiling +
pending_retest_after_substrate with a 2026-06-19 ceiling_routing_note ("don't
re-queue the 669 line until a context-diversity substrate lands"). This run is
NOT the 669 line: it uses the 588c FORCED-FEED SINGLE-ANCHOR harness (which
forms exactly one anchor and bypasses both the 669a ecological-contact starvation
AND the 669b nursery-context anchor-store-saturation context-diversity ceiling).
It tests a DIFFERENT root -- whether the 0.37 ceiling is an UNTRAINED-ENCODER
artifact (z_world magnitude), an encoder-fidelity question orthogonal to
context-diversity. Run brake-exempt per user direction 2026-06-23 as a
DIAGNOSTIC (experiment_purpose=diagnostic, claim_ids=[] -> scoring-excluded), so
neither a PASS nor a FAIL weights MECH-189 governance; the result ROUTES the
absolute governance gate (see interpretation grid) without re-deriving the
ceiling on a starved test.

DESIGN -- 588c forced-feed harness + an added P0 ENCODER-TRAINING phase.
  P0 (encoder warmup): train the z_world encoder (SD-018 resource-proximity
    supervision on agent.latent_stack + E1 prediction) over N_p0 episodes in the
    env, so z_world is discriminative + properly scaled. The PREMISE of the
    trained-encoder successor. Measured: resource-proximity MSE start-window vs
    end-window (the encoder-trained readiness gate) + mean ||z_world|| at end.
  CHILD phase (write_enabled): forced high-salience benefit each step at visited
    z_world contexts -> the matured z_goal is written as the super-ordinal anchor
    (n_occupied=1, like 588c).
  FREEZE: agent.set_super_ordinal_write_enabled(False).
  ADULT phase (READ-only): fresh sub-floor z_goal each episode (goal_state.reset()),
    no benefit pulse -> only the MECH-189 seeding READ (cue_pull toward the stored
    anchor) lifts z_goal. Measure adult median peak z_goal.norm() and whether it
    crosses 0.4.

  ARM_ON  : use_super_ordinal_goal_anchors=True  (anchor store active)
  ARM_OFF : use_super_ordinal_goal_anchors=False (no store -> adult z_goal ~ 0)
  BOTH arms run the IDENTICAL P0 encoder warmup; the only difference is the store,
  so the absolute-crossing measurement is isolated to the trained-encoder seeding.

ACCEPTANCE -- diagnostic readiness/verdict grid.
  READINESS / non-vacuity preconditions (else substrate_not_ready_requeue):
    - encoder_trained: the P0 SD-018 resource-proximity MSE improves by
      >= RP_IMPROVE_MIN (the "trained-encoder" premise; below floor -> the
      encoder did not train, the crossing is uninterpretable, the V3-EXQ-642
      untrained-substrate confound) on >= 2/3 seeds.
    - arm_on child anchors form (n_occupied > 0).
    - arm_on adult seeding fires (n_seeds > 0).
    - arm_on adult seeding produces a non-zero z_goal (positive control, the SAME
      statistic the load-bearing crossing routes on).
  LOAD-BEARING criterion C_CROSS (the DEV-NEED-006 / 588 absolute gate): ARM_ON
    adult median peak z_goal.norm() > 0.4 on >= 2/3 seeds.

  Self-route (the routing the GAP-11b absolute governance gate consumes):
    readiness UNMET                 -> substrate_not_ready_requeue (re-queue with
                                       a longer / stronger P0; NOT a verdict).
    readiness MET + C_CROSS pass    -> trained_encoder_absolute_crossing_met
                                       (the 0.4 absolute gate clears with a trained
                                       encoder; the 0.37 ceiling WAS an
                                       untrained-encoder artifact -> route the
                                       MECH-189/DEV-NEED-006 absolute governance
                                       gate to a claim-tagged evidence successor).
    readiness MET + C_CROSS fail    -> absolute_crossing_ceiling_persists (the 0.37
                                       ceiling is NOT an untrained-encoder artifact;
                                       a genuine substrate ceiling survives a
                                       trained encoder -> route substrate
                                       enrichment + fire the MECH-189 re-derive
                                       brake at autopsy; NOT a falsification).
"""
from __future__ import annotations

import sys
import json
import random
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch
import torch.optim as optim
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ree_core.utils.config import REEConfig
from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from experiment_protocol import emit_outcome
from experiments._lib.arm_fingerprint import arm_cell

EXPERIMENT_TYPE = "v3_exq_588d_mech189_trained_encoder_absolute_crossing"
EXPERIMENT_PURPOSE = "diagnostic"

# Pre-registered thresholds (defined here, not inferred post-hoc).
ADULT_ZGOAL_GATE = 0.4          # LOAD-BEARING: the DEV-NEED-006 / 588 absolute gate
RP_IMPROVE_MIN = 0.20           # READINESS: min fractional drop in P0 resource-prox MSE
SEED_PASS_FRACTION = 2.0 / 3.0  # >= 2/3 seeds must pass the gated criterion
POSCTRL_FLOOR = 1e-3            # adult positive-control floor (same statistic as C_CROSS)
FORCED_BENEFIT = 0.5            # child forced-feed benefit (salience >> threshold)
FORCED_DRIVE = 0.9
GREEDY_FRAC = 0.4               # P0 mixed policy: 40% greedy-toward-resource

# P0 encoder-training learning rates.
LR_ENC = 1e-3                   # latent_stack (z_world) encoder via SD-018 proximity
LR_E1 = 1e-3                    # E1 world-model prediction


def _build_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        size=8,
        num_hazards=2,
        num_resources=3,
        use_proxy_fields=True,
        seed=seed,
    )


def _build_agent(env: CausalGridWorldV2, arm_on: bool) -> REEAgent:
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=4,
        z_goal_enabled=True,
        drive_weight=2.0,
        alpha_world=0.9,                       # SD-008
        use_resource_proximity_head=True,      # SD-018: trains z_world in P0
        use_super_ordinal_goal_anchors=arm_on,
        super_ordinal_salience_threshold=0.5,
        super_ordinal_complexity_mode="novelty",
        super_ordinal_complexity_threshold=0.2,
        super_ordinal_merge_similarity=0.8,
        super_ordinal_write_alpha=0.3,
        super_ordinal_seed_below_norm=ADULT_ZGOAL_GATE,
        super_ordinal_seed_match_threshold=0.3,
        super_ordinal_seed_strength=0.2,
    )
    return REEAgent(cfg)


def _onehot(idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, idx] = 1.0
    return v


def _greedy_toward_resource(env) -> int:
    """Greedy action: move toward nearest resource (Manhattan)."""
    ax, ay = env.agent_x, env.agent_y
    if not env.resources:
        return random.randint(0, env.action_dim - 1)
    best_d = float("inf")
    nearest = None
    for r in env.resources:
        rx, ry = int(r[0]), int(r[1])
        d = abs(ax - rx) + abs(ay - ry)
        if d < best_d:
            best_d = d
            nearest = (rx, ry)
    if nearest is None or best_d == 0:
        return random.randint(0, env.action_dim - 1)
    rx, ry = nearest
    dx, dy = rx - ax, ry - ay
    if abs(dx) >= abs(dy):
        return 1 if dx > 0 else 0
    return 3 if dy > 0 else 2


def _rp_target(obs_dict: Dict[str, Any]) -> Optional[float]:
    """Peak resource proximity from the 5x5 view (center index 12), SD-018 label."""
    rfv = obs_dict.get("resource_field_view", None)
    if rfv is None:
        return None
    try:
        return float(rfv[12].item())
    except Exception:
        return float(rfv.reshape(-1)[12])


def _step_world_dim(agent: REEAgent) -> int:
    return agent.config.latent.world_dim


def _p0_train_encoder(agent: REEAgent, env: CausalGridWorldV2, n_ep: int,
                      steps: int, seed: int, arm_label: str,
                      total_eps: int) -> Dict[str, float]:
    """P0: train the z_world encoder (SD-018 proximity + E1 prediction).

    Returns readiness diagnostics: resource-proximity MSE in the first vs last
    training window + mean ||z_world|| at the end of P0."""
    device = agent.device
    n_act = env.action_dim
    enc_opt = optim.Adam(agent.latent_stack.parameters(), lr=LR_ENC)
    e1_opt = optim.Adam(agent.e1.parameters(), lr=LR_E1)

    agent.train()
    rp_first: List[float] = []
    rp_last: List[float] = []
    zw_norms: List[float] = []
    window = max(1, (n_ep * steps) // 5)  # first/last fifth of P0
    tick_i = 0
    total_ticks = max(1, n_ep * steps)
    random.seed(seed * 1009 + 1)

    for ep in range(n_ep):
        _, obs_dict = env.reset()
        agent.reset()
        for _ in range(steps):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            obs_harm = obs_dict.get("harm_obs", None)
            rp_t = _rp_target(obs_dict)

            latent = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
            ticks = agent.clock.advance()
            if ticks.get("e1_tick", False):
                _ = agent._e1_tick(latent)

            zw_norm = float(latent.z_world.detach().norm().item())
            if tick_i >= total_ticks - window:
                zw_norms.append(zw_norm)

            # SD-018 resource-proximity encoder training (trains latent_stack).
            if rp_t is not None:
                rp_loss = agent.compute_resource_proximity_loss(rp_t, latent)
                if rp_loss.requires_grad:
                    rp_val = float(rp_loss.detach().item())
                    if tick_i < window:
                        rp_first.append(rp_val)
                    elif tick_i >= total_ticks - window:
                        rp_last.append(rp_val)
                    enc_opt.zero_grad()
                    rp_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        agent.latent_stack.parameters(), 1.0)
                    enc_opt.step()

            # E1 world-model prediction training.
            if len(agent._world_experience_buffer) >= 2:
                e1_loss = agent.compute_prediction_loss()
                if e1_loss.requires_grad:
                    e1_opt.zero_grad()
                    e1_loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.e1.parameters(), 1.0)
                    e1_opt.step()

            if random.random() < GREEDY_FRAC:
                action_idx = _greedy_toward_resource(env)
            else:
                action_idx = random.randint(0, n_act - 1)
            action_oh = _onehot(action_idx, n_act, device)
            agent._last_action = action_oh
            _, _harm, done, _info, obs_dict = env.step(action_oh)
            tick_i += 1
            if done:
                break
        if (ep + 1) % 5 == 0 or ep == n_ep - 1:
            print(f"  [train] p0 seed={seed} arm={arm_label} "
                  f"ep {ep + 1}/{total_eps}", flush=True)

    agent.eval()
    rp_start = sum(rp_first) / max(1, len(rp_first)) if rp_first else 0.0
    rp_end = sum(rp_last) / max(1, len(rp_last)) if rp_last else 0.0
    improve_frac = ((rp_start - rp_end) / rp_start) if rp_start > 1e-9 else 0.0
    zw_mean = sum(zw_norms) / max(1, len(zw_norms)) if zw_norms else 0.0
    return {
        "rp_mse_start": round(rp_start, 6),
        "rp_mse_end": round(rp_end, 6),
        "rp_improve_frac": round(float(improve_frac), 6),
        "zworld_norm_trained_mean": round(zw_mean, 6),
    }


def _child_episode(agent: REEAgent, env: CausalGridWorldV2, steps: int) -> None:
    """Forced-feed child episode: forms the super-ordinal anchor at visited
    contexts (trained encoder; writes enabled)."""
    _, obs_dict = env.reset()
    agent.reset()  # per-episode reset -- does NOT clear the super-ordinal store
    wd = _step_world_dim(agent)
    for _ in range(steps):
        obs_body = obs_dict["body_state"]
        obs_world = obs_dict["world_state"]
        latent = agent.sense(obs_body, obs_world)
        ticks = agent.clock.advance()
        e1_prior = (
            agent._e1_tick(latent) if ticks.get("e1_tick")
            else torch.zeros(1, wd, device=agent.device)
        )
        candidates = agent.generate_trajectories(latent, e1_prior, ticks)
        action = agent.select_action(candidates, ticks)
        action_idx = int(action.argmax(dim=-1).item())
        # FORCED high-salience benefit -> anchor write at this z_world context.
        agent.update_z_goal(benefit_exposure=FORCED_BENEFIT, drive_level=FORCED_DRIVE)
        _, _harm, done, _, obs_dict = env.step(action_idx)
        if done:
            break


def _adult_episode(agent: REEAgent, env: CausalGridWorldV2, steps: int) -> float:
    """Adult episode with fresh sub-floor z_goal: measures READ-only seeding.
    Returns the per-episode peak z_goal.norm()."""
    _, obs_dict = env.reset()
    agent.reset()
    if agent.goal_state is not None:
        agent.goal_state.reset()  # fresh sub-floor z_goal each adult episode
    wd = _step_world_dim(agent)
    peak = 0.0
    for _ in range(steps):
        obs_body = obs_dict["body_state"]
        obs_world = obs_dict["world_state"]
        with torch.no_grad():
            latent = agent.sense(obs_body, obs_world)
            ticks = agent.clock.advance()
            e1_prior = (
                agent._e1_tick(latent) if ticks.get("e1_tick")
                else torch.zeros(1, wd, device=agent.device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            action = agent.select_action(candidates, ticks)
            action_idx = int(action.argmax(dim=-1).item())
        # NO benefit pulse -> only the MECH-189 READ (super-ordinal seeding) can
        # lift z_goal. Writes are frozen.
        agent.update_z_goal(benefit_exposure=0.0, drive_level=0.0)
        cur = float(agent.goal_state.goal_norm())
        if cur > peak:
            peak = cur
        _, _harm, done, _, obs_dict = env.step(action_idx)
        if done:
            break
    return peak


def _run_seed_arm(arm_on: bool, seed: int, n_p0: int, n_child: int,
                  n_adult: int, steps: int) -> Dict[str, Any]:
    arm_label = "ON" if arm_on else "OFF"
    print(f"Seed {seed} Condition {arm_label}", flush=True)
    total_eps = n_p0 + n_child + n_adult
    full_config = {
        "arm_on": arm_on, "seed": seed, "n_p0": n_p0, "n_child": n_child,
        "n_adult": n_adult, "steps": steps,
        "forced_benefit": FORCED_BENEFIT, "forced_drive": FORCED_DRIVE,
        "seed_below_norm": ADULT_ZGOAL_GATE, "lr_enc": LR_ENC,
        "use_resource_proximity_head": True,
    }
    with arm_cell(seed, config_slice=full_config, script_path=Path(__file__)) as cell:
        torch.manual_seed(seed)
        env = _build_env(seed)
        agent = _build_agent(env, arm_on)
        som = agent.super_ordinal_goal_memory

        # P0: train the z_world encoder (the trained-encoder premise).
        readiness = _p0_train_encoder(
            agent, env, n_p0, steps, seed, arm_label, total_eps)

        # CHILD phase (write_enabled True by default; trained encoder).
        for ep in range(n_child):
            _child_episode(agent, env, steps)
            if (ep + 1) % 2 == 0 or ep == n_child - 1:
                print(f"  [train] child seed={seed} arm={arm_label} "
                      f"ep {n_p0 + ep + 1}/{total_eps}", flush=True)
        n_occupied = som.n_occupied() if som is not None else 0

        # FREEZE writes for the adult measurement phase.
        agent.set_super_ordinal_write_enabled(False)

        # ADULT phase (READ-only seeding).
        adult_peaks: List[float] = []
        for ep in range(n_adult):
            adult_peaks.append(_adult_episode(agent, env, steps))
            print(f"  [train] adult seed={seed} arm={arm_label} "
                  f"ep {n_p0 + n_child + ep + 1}/{total_eps}", flush=True)
        n_seeds = (som._n_seeds if som is not None else 0)

        adult_peaks_sorted = sorted(adult_peaks)
        m = len(adult_peaks_sorted)
        adult_median = (
            adult_peaks_sorted[m // 2] if m % 2 == 1
            else 0.5 * (adult_peaks_sorted[m // 2 - 1] + adult_peaks_sorted[m // 2])
        ) if m else 0.0

        # Per-cell progress verdict (rough signal; the load-bearing absolute
        # crossing is computed by pairing arms in run_experiment).
        cell_ok = (
            adult_median > ADULT_ZGOAL_GATE if arm_on
            else (adult_median < ADULT_ZGOAL_GATE)
        )
        print(f"verdict: {'PASS' if cell_ok else 'FAIL'}", flush=True)

        row = {
            "arm": arm_label,
            "arm_on": arm_on,
            "seed": seed,
            "rp_mse_start": readiness["rp_mse_start"],
            "rp_mse_end": readiness["rp_mse_end"],
            "rp_improve_frac": readiness["rp_improve_frac"],
            "zworld_norm_trained_mean": readiness["zworld_norm_trained_mean"],
            "child_n_occupied": int(n_occupied),
            "adult_n_seeds": int(n_seeds),
            "adult_peaks": [round(p, 6) for p in adult_peaks],
            "adult_median_zgoal_norm": round(float(adult_median), 6),
            "adult_crosses_gate": bool(adult_median > ADULT_ZGOAL_GATE),
        }
        cell.stamp(row)
    return row


def run_experiment(n_p0: int, n_child: int, n_adult: int, steps: int,
                   seeds: List[int], dry_run: bool) -> Dict[str, Any]:
    arm_results: List[Dict[str, Any]] = []
    for seed in seeds:
        for arm_on in (False, True):  # OFF baseline first, then ON
            arm_results.append(
                _run_seed_arm(arm_on, seed, n_p0, n_child, n_adult, steps)
            )

    on_rows = [r for r in arm_results if r["arm_on"]]
    off_rows = [r for r in arm_results if not r["arm_on"]]
    off_by_seed = {r["seed"]: r["adult_median_zgoal_norm"] for r in off_rows}

    # READINESS / non-vacuity preconditions.
    n_enc_trained = sum(1 for r in on_rows if r["rp_improve_frac"] >= RP_IMPROVE_MIN)
    frac_enc_trained = n_enc_trained / float(max(1, len(on_rows)))
    encoder_trained = frac_enc_trained >= SEED_PASS_FRACTION
    on_anchors_formed = all(r["child_n_occupied"] > 0 for r in on_rows)
    on_seeding_fired = all(r["adult_n_seeds"] > 0 for r in on_rows)
    on_mean_adult = (
        sum(r["adult_median_zgoal_norm"] for r in on_rows) / max(1, len(on_rows))
    )
    on_posctrl = on_mean_adult  # same statistic class C_CROSS routes on
    readiness_met = (
        encoder_trained and on_anchors_formed and on_seeding_fired
        and on_posctrl > POSCTRL_FLOOR
    )

    # LOAD-BEARING C_CROSS: ARM_ON adult median crosses the 0.4 absolute gate.
    n_on_cross = sum(
        1 for r in on_rows if r["adult_median_zgoal_norm"] > ADULT_ZGOAL_GATE
    )
    frac_on_cross = n_on_cross / float(max(1, len(on_rows)))
    c_cross_pass = frac_on_cross >= SEED_PASS_FRACTION

    # Non-degeneracy: ARM_ON adult z_goal must genuinely differ from ARM_OFF
    # (the seeding moved z_goal vs the no-store baseline ~0).
    off_max_adult = max(
        (r["adult_median_zgoal_norm"] for r in off_rows), default=0.0)
    c_cross_non_degenerate = (on_mean_adult - off_max_adult) > 0.1

    if not readiness_met:
        outcome = "FAIL"
        label = "substrate_not_ready_requeue"
        direction = "unknown"
    elif c_cross_pass:
        outcome = "PASS"
        label = "trained_encoder_absolute_crossing_met"
        direction = "supports"
    else:
        outcome = "FAIL"
        label = "absolute_crossing_ceiling_persists"
        direction = "weakens"

    on_min_improve = min((r["rp_improve_frac"] for r in on_rows), default=0.0)
    on_min_occupied = min((r["child_n_occupied"] for r in on_rows), default=0)
    on_min_seeds = min((r["adult_n_seeds"] for r in on_rows), default=0)

    result: Dict[str, Any] = {
        "run_id": f"{EXPERIMENT_TYPE}_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_v3",
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": [],
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "timestamp_utc": datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
        "outcome": outcome,
        "dry_run": dry_run,
        "config": {
            "n_p0": n_p0, "n_child": n_child, "n_adult": n_adult, "steps": steps,
            "seeds": seeds, "forced_benefit": FORCED_BENEFIT,
            "forced_drive": FORCED_DRIVE,
            "devneed006_gate_absolute": ADULT_ZGOAL_GATE,
            "rp_improve_min": RP_IMPROVE_MIN,
            "seed_pass_fraction": SEED_PASS_FRACTION,
            "lr_enc": LR_ENC, "lr_e1": LR_E1,
            "use_resource_proximity_head": True,
        },
        "metrics": {
            "frac_on_cross_absolute_gate": round(frac_on_cross, 4),
            "n_on_cross_absolute_gate": n_on_cross,
            "on_mean_adult_zgoal_norm": round(on_mean_adult, 6),
            "off_max_adult_zgoal_norm": round(off_max_adult, 6),
            "frac_encoder_trained": round(frac_enc_trained, 4),
            "n_encoder_trained": n_enc_trained,
            "on_anchors_formed": on_anchors_formed,
            "on_seeding_fired": on_seeding_fired,
            "devneed006_gate_threshold": ADULT_ZGOAL_GATE,
        },
        "arm_results": arm_results,
        "interpretation": {
            "label": label,
            "preconditions": [
                {
                    "name": "encoder_trained",
                    "description": "Fraction of ARM_ON seeds whose P0 SD-018 "
                                   "resource-proximity MSE improved by "
                                   ">= RP_IMPROVE_MIN (the trained-encoder premise; "
                                   "below the 2/3 floor is the V3-EXQ-642 "
                                   "untrained-substrate confound). Per-seed raw "
                                   "improve_frac is in arm_results / metrics "
                                   "(on_min=" + str(round(float(on_min_improve), 4))
                                   + ", rp_improve_min=" + str(RP_IMPROVE_MIN) + ").",
                    "measured": round(float(frac_enc_trained), 6),
                    "threshold": SEED_PASS_FRACTION,
                    "control": "P0 trains agent.latent_stack via the SD-018 "
                               "resource-proximity head; a seed counts as trained "
                               "when (rp_mse_start - rp_mse_end)/rp_mse_start "
                               ">= RP_IMPROVE_MIN",
                    "met": bool(encoder_trained),
                },
                {
                    "name": "arm_on_child_anchors_formed",
                    "description": "ARM_ON child phase wrote >=1 super-ordinal "
                                   "anchor (n_occupied>0).",
                    "measured": int(on_min_occupied),
                    "threshold": 1,
                    "met": bool(on_anchors_formed),
                },
                {
                    "name": "arm_on_adult_seeding_fired",
                    "description": "ARM_ON adult phase fired the READ seeding path "
                                   "(n_seeds>0).",
                    "measured": int(on_min_seeds),
                    "threshold": 1,
                    "met": bool(on_seeding_fired),
                },
                {
                    "name": "arm_on_adult_zgoal_positive_control",
                    "description": "Readiness: ARM_ON adult seeding produces a "
                                   "non-zero z_goal.norm (SAME statistic the "
                                   "load-bearing absolute crossing routes on).",
                    "measured": round(float(on_posctrl), 6),
                    "threshold": POSCTRL_FLOOR,
                    "control": "ARM_ON adult episodes seed z_goal from the "
                               "childhood anchor with no benefit pulse",
                    "met": bool(on_posctrl > POSCTRL_FLOOR),
                },
            ],
            "criteria_non_degenerate": {
                "C_CROSS": bool(c_cross_non_degenerate),
            },
            "criteria": [
                {
                    "name": "C_CROSS_arm_on_adult_zgoal_crosses_0p4_absolute",
                    "load_bearing": True,
                    "passed": bool(c_cross_pass),
                },
            ],
            "evidence_direction": direction,
            "routing_note": (
                "DIAGNOSTIC (claim_ids=[], scoring-excluded). "
                "trained_encoder_absolute_crossing_met -> the 0.37 ceiling was an "
                "untrained-encoder artifact; route the MECH-189/DEV-NEED-006 "
                "absolute governance gate to a claim-tagged evidence successor. "
                "absolute_crossing_ceiling_persists -> a genuine substrate ceiling "
                "survives a trained encoder; route substrate enrichment + fire the "
                "MECH-189 re-derive brake at autopsy (NOT a falsification). "
                "substrate_not_ready_requeue -> re-queue with a longer/stronger P0."
            ),
        },
        "evidence_direction": direction,
    }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.dry_run:
        n_p0, n_child, n_adult, steps, seeds = 2, 2, 2, 20, [42]
    else:
        n_p0, n_child, n_adult, steps, seeds = 30, 8, 6, 100, [42, 43, 44]

    result = run_experiment(n_p0, n_child, n_adult, steps, seeds, args.dry_run)

    out_dir = (
        Path(__file__).resolve().parents[2]
        / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{result['run_id']}.json"
    out_path.write_text(json.dumps(result, indent=2))

    print(f"outcome: {result['outcome']}", flush=True)
    print(f"label: {result['interpretation']['label']}", flush=True)
    print(f"frac_encoder_trained: {result['metrics']['frac_encoder_trained']}",
          flush=True)
    print(f"frac_on_cross_absolute_gate(0.4): "
          f"{result['metrics']['frac_on_cross_absolute_gate']}", flush=True)
    print(f"on_mean_adult_zgoal: {result['metrics']['on_mean_adult_zgoal_norm']} "
          f"off_max_adult_zgoal: {result['metrics']['off_max_adult_zgoal_norm']}",
          flush=True)
    print(f"wrote: {out_path}", flush=True)
    return result, out_path


if __name__ == "__main__":
    _result, _out_path = main()
    _outcome_raw = str(_result["outcome"]).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=_out_path,
        dry_run=_result["dry_run"],
    )
