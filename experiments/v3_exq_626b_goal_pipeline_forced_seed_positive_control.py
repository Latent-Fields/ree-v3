"""
V3-EXQ-626b: goal-pipeline FORCED-SEED L1 positive control (GAP-7 deliverable).

Lettered iteration of V3-EXQ-626 / 626a. Scientific question UNCHANGED at the
L1 level: can the experiment HARNESS see a forced supra-threshold seed produce a
non-zero, stable z_goal? The implementation of the prior iterations could not
answer this cleanly:

- 626 (Class-1 harness bug): the bespoke episode loop never called
  agent.update_z_goal(...), so z_goal stayed at zero-init across every arm.
- 626a (harness fix): wired agent.update_z_goal(...) every step BUT read the
  benefit signal from the env (body_state[11]), so the "positive control" arm
  ARM_A only formed z_goal on seeds that actually FORAGED. Seeds that never made
  resource contact (a goal_pipeline:GAP-2 foraging-competence leak) showed
  z_goal=0, and the harness could not separate "signal absent" from "signal
  present but inert". Per failure_autopsy_V3-EXQ-603e-626a-622_2026-06-03: 626a's
  P0 positive control formed z_goal on only 1/3 seeds (seed 44 = 0.19); the
  goal-pipeline WIRING layer is closed and verified, but the ecological positive
  control is gated on foraging competence (GAP-2), not on the harness fix.

THE FIX (this script): a genuine FORCED-SEED positive control, decoupled from
foraging. Every step, agent.update_z_goal(benefit_exposure=FORCED_BENEFIT,
drive_level=FORCED_DRIVE) is fed a FORCED supra-threshold benefit (1.0) and drive
(0.9) INDEPENDENT of what the env returns -- the run_stage0_nursery pattern from
experiments/scaffolded_sd054_onboarding.py (the agent is "fed" regardless of
resource contact). This makes z_goal formation depend ONLY on the harness fix +
the GoalState substrate gate (already validated at the unit level by
tests/contracts/test_goalstate_forced_seed_positive_control.py 6/6), NOT on the
GAP-2 foraging-competence substrate. The harness can now SEE a non-zero, stable
z_goal in a positive-control arm that is GAP-2-independent.

This is the L1 link of the goal-pipeline closure map (goal_pipeline_plan.md GAP-7,
Phase 7 / "L1 harness positive control"). It is NOT the L2-L3 object-binding
substrate (a later /implement-substrate task) and NOT the Stage-4 axis-dissociation
grid (which 626/626a tried to run; that is gated behind a trustworthy positive
control, which this script establishes).

Arms (3 arms x 3 seeds; each run is one forced-feed measurement phase)
---------------------------------------------------------------------
ARM_FORCED_SEED_ON  z_goal_enabled=True, forced_benefit=1.0, forced_drive=0.9.
                    POSITIVE CONTROL. z_goal must form and stay stable.
ARM_NO_BENEFIT      z_goal_enabled=True, forced_benefit=0.0, forced_drive=0.9.
                    NEGATIVE CONTROL. update_z_goal IS called every step but the
                    benefit pulse is sub-threshold, so z_goal must NOT form. Proves
                    the ARM_FORCED_SEED_ON signal is the forced seed, not a
                    loop/measurement artifact.
ARM_ZGOAL_OFF       z_goal_enabled=False, forced_benefit=1.0.
                    OFF-PARITY CONTROL. goal_state is None -> update_z_goal
                    early-returns -> z_goal_norm stays 0.0. Bit-identical OFF.

Pre-registered acceptance criteria
----------------------------------
C1 (positive control formation): ARM_FORCED_SEED_ON z_goal_norm_peak >= 0.4 on
   >= 2/3 seeds (matches the run_stage0_nursery G0 gate / V3-EXQ-634 Stage-0
   forced-feed which PASSed 3/3 at z_goal 0.40-0.45).
C2 (positive control stability): ARM_FORCED_SEED_ON z_goal_norm median over the
   last measurement window >= 0.1 on >= 2/3 seeds (the seed persists, not a
   one-step spike).
C3 (negative control -- no seed, no goal): ARM_NO_BENEFIT z_goal_norm_peak < 0.05
   on >= 2/3 seeds (z_goal does NOT form without the forced supra-threshold
   benefit -- the signal in ARM_FORCED_SEED_ON is provably the seed).
C4 (OFF parity): ARM_ZGOAL_OFF z_goal_norm_peak < 1e-9 on ALL seeds (goal pipeline
   disabled -> z_goal stays at zero-init; bit-identical OFF).

overall_pass = C1 AND C2 AND C3 AND C4.

A PASS re-establishes the L1 forced-seed positive control as a passing diagnostic
at the experiment-harness level, decoupled from GAP-2. It does NOT close GAP-7
(L7 consumer-readout audit + L2-L3 object-binding remain).

claim_ids = []  (harness positive control / substrate-readiness diagnostic; NOT
load-bearing evidence weighting per the CLAIM_IDS Accuracy Rule).
experiment_purpose = "diagnostic".
supersedes = V3-EXQ-626a (queue) / v3_exq_626a_goal_pipeline_developmental_window_diagnostic
(manifest experiment_type).

References:
- REE_assembly/evidence/planning/goal_pipeline_plan.md  (GAP-7, Phase 7, L0-L9 map)
- REE_assembly/evidence/planning/goal_stream_repair_diagnostic_ladder_2026-06-01.md (F1)
- REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-603e-626a-622_2026-06-03.md
- experiments/scaffolded_sd054_onboarding.py  (run_stage0_nursery forced-feed pattern)
- tests/contracts/test_goalstate_forced_seed_positive_control.py  (F0 unit positive control)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig

from experiment_protocol import emit_outcome
from experiments.pack_writer import write_flat_manifest  # noqa: E402

# -- Header constants --------------------------------------------------------
EXPERIMENT_TYPE = "v3_exq_626b_goal_pipeline_forced_seed_positive_control"
QUEUE_ID = "V3-EXQ-626b"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"
CLAIM_IDS: List[str] = []
EXPERIMENT_PURPOSE = "diagnostic"
SUPERSEDES = "v3_exq_626a_goal_pipeline_developmental_window_diagnostic"
SEEDS = [42, 43, 44]

# -- Forced-seed parameters (run_stage0_nursery pattern) ---------------------
FORCED_BENEFIT = 1.0   # supra-threshold benefit fed every step (>> benefit_threshold 0.1)
FORCED_DRIVE = 0.9     # forced drive (matches scaffold_stage0_forced_drive default)

# -- Pre-registered thresholds -----------------------------------------------
C1_FORCED_PEAK_FLOOR = 0.4        # positive-control formation floor (nursery G0 gate)
C2_FORCED_STABILITY_FLOOR = 0.1   # positive-control last-window median floor
C3_NO_BENEFIT_PEAK_CEILING = 0.05 # negative-control peak ceiling (no seed -> no goal)
C4_OFF_PARITY_EPS = 1e-9          # OFF-parity ceiling (z_goal stays at zero-init)
PASS_SEED_FRACTION = 2.0 / 3.0    # >= 2/3 seeds clears C1/C2/C3
LAST_WINDOW_EPISODES = 10         # window for the stability median

# -- Phase budget ------------------------------------------------------------
MEASURE_EPISODES_DEFAULT = 30
STEPS_PER_EPISODE_DEFAULT = 200

# -- Common training rates (mirrors v3_exq_626a) -----------------------------
LR_E1 = 1e-4
LR_E2_WF = 1e-3
BATCH_SIZE = 32
WF_BUF_MAX = 2000

# -- Env / agent dims --------------------------------------------------------
SELF_DIM = 32
WORLD_DIM = 32
HARM_DIM = 16
HARM_A_DIM = 7
HARM_HISTORY_LEN = 10

# -- dACC sub-weights (same shape as v3_exq_626a) ----------------------------
DACC_WEIGHT = 0.5
DACC_INTERACTION_WEIGHT = 0.5


# ---------------------------------------------------------------------------
# Arm definitions
# ---------------------------------------------------------------------------


@dataclass
class ArmConfig:
    """Per-arm configuration for the forced-seed positive control."""

    name: str
    z_goal_enabled: bool
    forced_benefit: float
    forced_drive: float


def arm_configs() -> List[ArmConfig]:
    return [
        ArmConfig(
            name="ARM_FORCED_SEED_ON",
            z_goal_enabled=True, forced_benefit=FORCED_BENEFIT, forced_drive=FORCED_DRIVE,
        ),
        ArmConfig(
            name="ARM_NO_BENEFIT",
            z_goal_enabled=True, forced_benefit=0.0, forced_drive=FORCED_DRIVE,
        ),
        ArmConfig(
            name="ARM_ZGOAL_OFF",
            z_goal_enabled=False, forced_benefit=FORCED_BENEFIT, forced_drive=FORCED_DRIVE,
        ),
    ]


# ---------------------------------------------------------------------------
# Env / agent builders
# ---------------------------------------------------------------------------


def _build_env(seed: int) -> CausalGridWorldV2:
    """
    Simple hazard-free-ish reef nursery env. Env contact is IRRELEVANT to the
    forced arm by construction (benefit is forced, not read from the env), so the
    env config only needs to be a valid stepping substrate for the agent loop.
    """
    return CausalGridWorldV2(
        size=12,
        num_hazards=0,
        num_resources=3,
        hazard_food_attraction=0.0,
        proximity_harm_scale=0.05,
        limb_damage_enabled=True,
        reef_enabled=True,
        reef_bipartite_layout=True,
        reef_bipartite_axis="horizontal",
        reef_bipartite_agent_band_radius=1,
        reef_bipartite_agent_spawn_in_reef_half=True,
        seed=seed,
    )


def build_agent(
    seed: int, device: torch.device, world_obs_dim: int, z_goal_enabled: bool
) -> REEAgent:
    """
    Build agent mirroring v3_exq_626a.build_agent, parameterised by
    z_goal_enabled so ARM_ZGOAL_OFF can exercise the OFF-parity path
    (goal_state is None -> update_z_goal early-returns).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    config = REEConfig.from_dims(
        body_obs_dim=17,
        world_obs_dim=world_obs_dim,
        action_dim=5,
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
        harm_dim=HARM_DIM,
        alpha_world=0.9,
        reafference_action_dim=5,
        use_harm_stream=True,
        z_harm_dim=HARM_DIM,
        use_affective_harm_stream=True,
        z_harm_a_dim=HARM_A_DIM,
        harm_history_len=HARM_HISTORY_LEN,
        use_resource_proximity_head=True,
        resource_proximity_weight=0.5,
        benefit_eval_enabled=True,
        benefit_weight=1.0,
        z_goal_enabled=z_goal_enabled,
        goal_weight=0.5,
        drive_weight=2.0,
        drive_ema_alpha=1.0,
        drive_floor=0.9,
        e1_goal_conditioned=True,
        limb_damage_enabled=True,
        damage_increment=0.15,
        failure_prob_scale=0.3,
        heal_rate=0.002,
        use_dacc=True,
        dacc_weight=DACC_WEIGHT,
        dacc_interaction_weight=DACC_INTERACTION_WEIGHT,
        use_e2_harm_a=True,
        use_mech295_liking_bridge=True,
        use_mech307_conjunction=True,
    )
    config.e3.commitment_threshold = 0.5
    config.heartbeat.beta_gate_bistable = True
    agent = REEAgent(config).to(device)
    return agent


# ---------------------------------------------------------------------------
# Training + measurement helpers (mirror v3_exq_626a)
# ---------------------------------------------------------------------------


def _train_step_e1_e2(
    agent: REEAgent, e1_opt, wf_opt, wf_buf: Deque, device: torch.device
) -> None:
    """E1 prediction + E2 world-forward updates; updates E3 running variance."""
    e1_opt.zero_grad()
    e1_loss = agent.compute_prediction_loss()
    if e1_loss.requires_grad:
        e1_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(agent.e1.parameters()), 1.0)
        e1_opt.step()

    if len(wf_buf) >= BATCH_SIZE:
        k = min(BATCH_SIZE, len(wf_buf))
        idxs = torch.randperm(len(wf_buf))[:k].tolist()
        zw_b = torch.cat([wf_buf[i][0] for i in idxs]).to(device)
        a_b = torch.cat([wf_buf[i][1] for i in idxs]).to(device)
        zw1_b = torch.cat([wf_buf[i][2] for i in idxs]).to(device)
        wf_pred = agent.e2.world_forward(zw_b, a_b)
        wf_loss = F.mse_loss(wf_pred, zw1_b)
        if wf_loss.requires_grad:
            wf_opt.zero_grad()
            wf_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(agent.e2.world_transition.parameters())
                + list(agent.e2.world_action_encoder.parameters()),
                1.0,
            )
            wf_opt.step()
        with torch.no_grad():
            agent.e3.update_running_variance((wf_pred.detach() - zw1_b).detach())


def _measure_z_goal_norm(agent: REEAgent) -> float:
    """Read z_goal_norm after update_z_goal. Returns 0.0 when goal_state is None."""
    goal_state = getattr(agent, "goal_state", None)
    if goal_state is None or not hasattr(goal_state, "goal_norm"):
        return 0.0
    try:
        return float(goal_state.goal_norm())
    except TypeError:
        return float(goal_state.goal_norm)


def _run_forced_episode(
    agent: REEAgent,
    env: CausalGridWorldV2,
    device: torch.device,
    steps_per_episode: int,
    forced_benefit: float,
    forced_drive: float,
    e1_opt,
    wf_opt,
    wf_buf: Deque,
    world_dim: int,
) -> Dict[str, Any]:
    """
    One forced-feed episode. After each env.step the goal pipeline is driven with
    a FORCED benefit + drive (NOT read from the env), decoupling z_goal formation
    from foraging. z_goal_norm is measured after the update.
    """
    _, obs_dict = env.reset()
    agent.reset()

    z_goal_peak = 0.0
    z_goal_sum = 0.0
    ep_len = 0
    z_world_prev: Optional[torch.Tensor] = None
    action_prev: Optional[torch.Tensor] = None

    agent.train()

    for step in range(steps_per_episode):
        obs_body = obs_dict["body_state"].to(device)
        obs_world = obs_dict["world_state"].to(device)

        latent = agent.sense(obs_body, obs_world)
        z_world_curr = latent.z_world.detach()

        if z_world_prev is not None and action_prev is not None:
            wf_buf.append((z_world_prev.cpu(), action_prev.cpu(), z_world_curr.cpu()))

        ticks = agent.clock.advance()
        e1_prior = (
            agent._e1_tick(latent)
            if ticks.get("e1_tick")
            else torch.zeros(1, world_dim, device=device)
        )
        candidates = agent.generate_trajectories(latent, e1_prior, ticks)
        action = agent.select_action(candidates, ticks)

        _train_step_e1_e2(agent, e1_opt, wf_opt, wf_buf, device)

        z_world_prev = z_world_curr
        action_prev = action.detach()
        action_idx = int(action.argmax(dim=-1).item())
        _, _harm_signal, done, _, obs_dict = env.step(action_idx)

        # FORCED SEED (the GAP-2-decoupling change vs 626a): drive the goal
        # pipeline with a FORCED benefit + drive, NOT the env-derived signal.
        # update_z_goal early-returns when goal_state is None (ARM_ZGOAL_OFF),
        # giving the bit-identical OFF-parity path.
        agent.update_z_goal(benefit_exposure=forced_benefit, drive_level=forced_drive)

        z_goal_now = _measure_z_goal_norm(agent)
        z_goal_sum += z_goal_now
        if z_goal_now > z_goal_peak:
            z_goal_peak = z_goal_now

        ep_len = step + 1
        if done:
            break

    return {
        "episode_length": ep_len,
        "z_goal_norm_peak": z_goal_peak,
        "z_goal_norm_mean": float(z_goal_sum / max(1, ep_len)),
    }


# ---------------------------------------------------------------------------
# Per-seed-arm runner
# ---------------------------------------------------------------------------


@dataclass
class CellSummary:
    seed: int
    arm: str
    z_goal_norm_peak_max: float
    z_goal_norm_median_last_window: float
    mean_episode_length: float
    per_episode_peaks: List[float] = field(default_factory=list)


def run_seed_arm(
    seed: int,
    arm: ArmConfig,
    device: torch.device,
    world_obs_dim: int,
    measure_eps: int,
    steps_per_episode: int,
) -> Dict[str, Any]:
    print(f"Seed {seed} Condition {arm.name}", flush=True)

    agent = build_agent(seed, device, world_obs_dim, z_goal_enabled=arm.z_goal_enabled)
    world_dim = agent.config.latent.world_dim

    e1_opt = optim.Adam(list(agent.e1.parameters()), lr=LR_E1)
    wf_opt = optim.Adam(
        list(agent.e2.world_transition.parameters())
        + list(agent.e2.world_action_encoder.parameters()),
        lr=LR_E2_WF,
    )
    wf_buf: Deque = deque(maxlen=WF_BUF_MAX)

    episodes: List[Dict[str, Any]] = []
    for ep in range(measure_eps):
        ep_metrics = _run_forced_episode(
            agent, _build_env(seed + ep), device, steps_per_episode,
            forced_benefit=arm.forced_benefit, forced_drive=arm.forced_drive,
            e1_opt=e1_opt, wf_opt=wf_opt, wf_buf=wf_buf, world_dim=world_dim,
        )
        episodes.append(ep_metrics)
        if ((ep + 1) % 5 == 0) or (ep + 1) == measure_eps:
            print(
                f"  [train] seed={seed} arm={arm.name} ep {ep + 1}/{measure_eps} "
                f"z_goal_peak={ep_metrics['z_goal_norm_peak']:.4f}",
                flush=True,
            )

    peaks = [float(e["z_goal_norm_peak"]) for e in episodes]
    means = [float(e["z_goal_norm_mean"]) for e in episodes]
    ep_lens = [int(e["episode_length"]) for e in episodes]
    last_window = means[-LAST_WINDOW_EPISODES:] if means else [0.0]
    summary = CellSummary(
        seed=seed,
        arm=arm.name,
        z_goal_norm_peak_max=float(max(peaks)) if peaks else 0.0,
        z_goal_norm_median_last_window=float(np.median(last_window)) if last_window else 0.0,
        mean_episode_length=float(sum(ep_lens)) / float(max(1, len(ep_lens))),
        per_episode_peaks=peaks,
    )
    print(
        f"verdict: seed={seed} arm={arm.name} "
        f"peak_max={summary.z_goal_norm_peak_max:.4f} "
        f"med_last={summary.z_goal_norm_median_last_window:.4f}",
        flush=True,
    )
    return summary.__dict__


# ---------------------------------------------------------------------------
# Acceptance evaluation
# ---------------------------------------------------------------------------


def _frac_above(values: List[float], floor: float) -> float:
    if not values:
        return 0.0
    return float(sum(1 for v in values if v >= floor)) / float(len(values))


def _frac_below(values: List[float], ceiling: float) -> float:
    if not values:
        return 0.0
    return float(sum(1 for v in values if v < ceiling)) / float(len(values))


def evaluate_acceptance(cells: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_arm: Dict[str, List[Dict[str, Any]]] = {}
    for c in cells:
        by_arm.setdefault(c["arm"], []).append(c)

    def peaks(arm: str) -> List[float]:
        return [float(c["z_goal_norm_peak_max"]) for c in by_arm.get(arm, [])]

    def medians(arm: str) -> List[float]:
        return [float(c["z_goal_norm_median_last_window"]) for c in by_arm.get(arm, [])]

    forced_peaks = peaks("ARM_FORCED_SEED_ON")
    forced_medians = medians("ARM_FORCED_SEED_ON")
    no_benefit_peaks = peaks("ARM_NO_BENEFIT")
    off_peaks = peaks("ARM_ZGOAL_OFF")

    c1_frac = _frac_above(forced_peaks, C1_FORCED_PEAK_FLOOR)
    c1_pass = c1_frac >= PASS_SEED_FRACTION

    c2_frac = _frac_above(forced_medians, C2_FORCED_STABILITY_FLOOR)
    c2_pass = c2_frac >= PASS_SEED_FRACTION

    c3_frac = _frac_below(no_benefit_peaks, C3_NO_BENEFIT_PEAK_CEILING)
    c3_pass = c3_frac >= PASS_SEED_FRACTION

    # C4: OFF parity must hold on ALL seeds (not 2/3).
    c4_pass = bool(off_peaks) and all(v < C4_OFF_PARITY_EPS for v in off_peaks)

    overall = bool(c1_pass and c2_pass and c3_pass and c4_pass)
    return {
        "C1_positive_control_formation": {
            "pass": c1_pass, "frac_seeds_clearing": c1_frac,
            "arm_forced_seed_on_peaks": forced_peaks, "floor": C1_FORCED_PEAK_FLOOR,
            "note": (
                "Forced supra-threshold benefit forms a non-zero z_goal that the "
                "harness can SEE, decoupled from GAP-2 foraging contact."
            ),
        },
        "C2_positive_control_stability": {
            "pass": c2_pass, "frac_seeds_clearing": c2_frac,
            "arm_forced_seed_on_last_window_medians": forced_medians,
            "floor": C2_FORCED_STABILITY_FLOOR,
        },
        "C3_negative_control_no_seed": {
            "pass": c3_pass, "frac_seeds_clearing": c3_frac,
            "arm_no_benefit_peaks": no_benefit_peaks, "ceiling": C3_NO_BENEFIT_PEAK_CEILING,
            "note": (
                "Without the forced supra-threshold benefit, z_goal does NOT form -- "
                "the positive-control signal is provably the seed, not a loop artifact."
            ),
        },
        "C4_off_parity": {
            "pass": c4_pass,
            "arm_zgoal_off_peaks": off_peaks, "eps": C4_OFF_PARITY_EPS,
            "note": "z_goal_enabled=False -> goal_state None -> z_goal stays zero-init.",
        },
        "overall_pass": overall,
    }


# ---------------------------------------------------------------------------
# Manifest + main
# ---------------------------------------------------------------------------


def emit_manifest(
    cells: List[Dict[str, Any]],
    acceptance: Dict[str, Any],
    out_dir: Path,
    cfg_phases: Dict[str, int],
) -> Path:
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"
    outcome = "PASS" if acceptance["overall_pass"] else "FAIL"
    manifest = {
        "run_id": run_id,
        "queue_id": QUEUE_ID,
        "experiment_type": EXPERIMENT_TYPE,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "supersedes": SUPERSEDES,
        "outcome": outcome,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "timestamp_utc": ts,
        "related_queue_ids": ["V3-EXQ-626", "V3-EXQ-626a", "V3-EXQ-634"],
        "design_doc": (
            "REE_assembly/evidence/planning/"
            "goal_stream_repair_diagnostic_ladder_2026-06-01.md"
        ),
        "autopsy": (
            "REE_assembly/evidence/planning/"
            "failure_autopsy_V3-EXQ-603e-626a-622_2026-06-03.md"
        ),
        "harness_fix_note": (
            "626a wired update_z_goal but read benefit from the env, so its positive "
            "control was gated on GAP-2 foraging competence (formed on 1/3 seeds). "
            "626b feeds a FORCED supra-threshold benefit every step (run_stage0_nursery "
            "pattern), so z_goal formation depends ONLY on the harness fix + GoalState "
            "gate, NOT on foraging. L1 positive control, GAP-2-independent."
        ),
        "forced_benefit": FORCED_BENEFIT,
        "forced_drive": FORCED_DRIVE,
        "acceptance": acceptance,
        "cells": cells,
        "seeds": SEEDS,
        "phase_budget": cfg_phases,
        "thresholds": {
            "C1_FORCED_PEAK_FLOOR": C1_FORCED_PEAK_FLOOR,
            "C2_FORCED_STABILITY_FLOOR": C2_FORCED_STABILITY_FLOOR,
            "C3_NO_BENEFIT_PEAK_CEILING": C3_NO_BENEFIT_PEAK_CEILING,
            "C4_OFF_PARITY_EPS": C4_OFF_PARITY_EPS,
            "PASS_SEED_FRACTION": PASS_SEED_FRACTION,
            "LAST_WINDOW_EPISODES": LAST_WINDOW_EPISODES,
        },
    }
    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=False,
        config=manifest.get("config"),
        seeds=SEEDS,
        script_path=Path(__file__),
    )
    print(f"manifest written: {out_path}")
    return out_path


def main(args: argparse.Namespace) -> Tuple[str, Optional[str]]:
    device = torch.device("cpu")
    if args.dry_run:
        measure_eps, steps_per_ep = 2, 20
    else:
        measure_eps = MEASURE_EPISODES_DEFAULT
        steps_per_ep = STEPS_PER_EPISODE_DEFAULT

    # Probe to read world_obs_dim under the reef + limb-damage env config.
    probe = _build_env(seed=0)
    probe.reset()
    world_obs_dim = probe.world_obs_dim

    arms = arm_configs()
    cells: List[Dict[str, Any]] = []
    for arm in arms:
        for seed in SEEDS:
            cell = run_seed_arm(
                seed=seed, arm=arm, device=device, world_obs_dim=world_obs_dim,
                measure_eps=measure_eps, steps_per_episode=steps_per_ep,
            )
            cells.append(cell)

    acceptance = evaluate_acceptance(cells)
    outcome = "PASS" if acceptance["overall_pass"] else "FAIL"
    print(
        f"Overall: {outcome} | "
        f"C1={acceptance['C1_positive_control_formation']['pass']} "
        f"C2={acceptance['C2_positive_control_stability']['pass']} "
        f"C3={acceptance['C3_negative_control_no_seed']['pass']} "
        f"C4={acceptance['C4_off_parity']['pass']}",
        flush=True,
    )

    if args.dry_run:
        print(f"verdict: dry_run overall={outcome}", flush=True)
        return outcome, None

    out_dir = Path(args.output_dir)
    cfg_phases = {"measure_eps": measure_eps, "steps_per_episode": steps_per_ep}
    out_path = emit_manifest(cells, acceptance, out_dir, cfg_phases=cfg_phases)
    return outcome, str(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(
            REPO_ROOT.parent
            / "REE_assembly"
            / "evidence"
            / "experiments"
            / EXPERIMENT_TYPE
        ),
    )
    args = parser.parse_args()
    outcome, manifest_path = main(args)
    if not args.dry_run and manifest_path:
        emit_outcome(outcome=outcome, manifest_path=manifest_path)
    sys.exit(0 if outcome == "PASS" else 1)
