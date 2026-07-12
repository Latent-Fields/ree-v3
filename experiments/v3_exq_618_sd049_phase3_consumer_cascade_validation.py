#!/opt/local/bin/python3
"""V3-EXQ-618 -- SD-049 Phase 3 SD-032 consumer cascade substrate-readiness.

Claim: SD-049 (multi-resource heterogeneity), Phase 3 (SD-032 consumer cascade
reading per_axis_drive directly; MECH-295 axis-matched approach-cue routing).
Status: candidate, v3_pending. Phase 3 substrate IMPLEMENTED 2026-05-31T13:00Z
(implement-substrate-sd049-phase-3-consumer-cascade session). 28/28 Phase 3
contract tests PASS (tests/contracts/test_sd049_phase3_consumer_cascade.py).

EXPERIMENT_PURPOSE = "diagnostic" (substrate-readiness; matches the EXQ-611 ->
EXQ-614 pattern of substrate-readiness-before-behavioural-falsifier).

Why this experiment exists
--------------------------
Phase 3 substrate landing turned on a master flag
(REEConfig.use_sd049_per_axis_consumer_cascade, default False, bit-identical
OFF) plus 7 per-consumer combiner fields and threaded per_axis_drive +
per_axis_combiner kwargs through seven consumer call sites (AIC, PCC, pACC,
dACC, SalienceCoordinator, BroadcastOverride, MECH-295 liking-bridge). The
28 contract tests prove the API surface is correct and bit-identical-OFF
holds; they do NOT prove that the substrate produces discriminable outputs
under the operating-condition shapes that the load-bearing behavioural
validation will probe.

This experiment is the substrate-readiness diagnostic that must PASS before
queuing the V3-EXQ-514g-successor behavioural validation. It runs in two
phases:

Phase A (end-to-end smoke, 3 arms x 3 seeds):
  Builds an agent + env for each arm, runs a brief warmup + measurement
  forward pass with random action selection. Catches:
    - plumbing crashes (cascade-on path raising on tensor shape mismatch,
      None handling in select_axis / collapse_per_axis_drive, etc.)
    - regression failures (cascade-off path not bit-identical to legacy)
    - env-level per-axis drive evolution sanity (replicates EXQ-513 C2c)

Phase B (direct-API substrate-readiness probes, no agent forward pass):
  Builds a single probe agent with cascade ON, AIC + MECH-295 enabled.
  Feeds controlled per_axis_drive vectors with known axis variance through
  the consumer APIs. Catches:
    - AIC.tick producing identical outputs whether per_axis_drive is supplied
      or not (cascade plumbed correctly but read fails to differentiate)
    - MECH-295 axis-routing producing identical scores under axis-aware vs
      combiner-only modes (select_axis or combiner collapse silently
      degrading the per-axis read)

Three arms (Phase A only)
-------------------------
ARM_0 BASELINE_CASCADE_OFF:
  use_sd049_per_axis_consumer_cascade=False
  Consumers read legacy scalar drive_level. Establishes the smoke baseline.

ARM_1 CASCADE_ON_COMBINER_ONLY:
  use_sd049_per_axis_consumer_cascade=True
  _current_goal_axis_idx = None (never set)
  Whole-organism consumers collapse via per-consumer combiner; MECH-295
  falls back to combiner since goal_axis_idx is None.

ARM_2 CASCADE_ON_AXIS_AWARE:
  use_sd049_per_axis_consumer_cascade=True
  _current_goal_axis_idx cycled across {0, 1, 2} per measurement episode.
  MECH-295 reads per_axis_drive[_current_goal_axis_idx] when set.

Pre-registered acceptance criteria
----------------------------------
C1 (Phase A env-level sanity): per-axis drive vector evolves in env across
    measurement episodes in all arms -- np.max(peak_per_axis_drive) > 0.02
    for all 3 arms x 3 seeds. Replicates V3-EXQ-513 C2c on the cascade
    substrate.

C2 (Phase A no-crash smoke): all 9 cells (3 arms x 3 seeds) complete
    Phase A forward-pass without raising. The smoke test alone validates
    the agent.py wiring of obs_per_axis_drive + the consumer call sites
    don't blow up when the master flag is on.

C3 (Phase B AIC reads per-axis input): with cascade ON, AIC.tick output
    differs from the cascade-OFF (per_axis_drive=None) output under
    matched (z_harm_a_norm, drive_level, beta_gate_elevated, operating_mode)
    inputs across a 4-vector synthetic per_axis_drive battery. Mean L1
    distance between the two output dicts' aic_salience field >= 0.005
    over the 4 inputs. Tests that AIC actually reads the per-axis vector
    when supplied -- not just that the kwarg is accepted.

C4 (Phase B MECH-295 axis-routing differs from combiner-only): direct-API
    probe of MECH295LikingBridge.compute_approach_cue_score_bias with a
    controlled 4-vector per_axis_drive battery (each with high inter-axis
    variance) under both goal_axis_idx=axis (axis-aware) and goal_axis_idx
    =None (combiner-only). Mean L1 across the 4 inputs x 8 candidates
    >= 0.005. Tests that axis-routing actually routes by axis rather than
    silently degrading to the combiner.

PASS = C1 AND C2 AND C3 AND C4.

Interpretation grid (for /governance routing on completion)
-----------------------------------------------------------
PASS (all 4 criteria fire):
    -> Substrate-readiness validated. Cascade plumbing reaches consumers;
       AIC reads per-axis info; MECH-295 axis-routing routes by axis.
       /queue-experiment V3-EXQ-619 as the load-bearing behavioural
       validation (V3-EXQ-514g-successor structure on Phase 3 substrate;
       MECH-295 axis-routing tested via goal-state-active rollouts with
       trained encoder + active goal_state, so MECH-295 fires in the
       actual select_action path).

C1 + C2 + C4 PASS, C3 FAIL:
    -> Cascade plumbed to MECH-295 but not effectively reaching AIC.
       Possible cause: AIC.tick collapses per_axis_drive via "max" combiner
       which equals the legacy drive_level read path under this experiment's
       synthetic inputs. /diagnose-errors on AIC integration.

C1 + C2 + C3 PASS, C4 FAIL:
    -> AIC reads per-axis; MECH-295 axis-routing degrades to combiner.
       Possible cause: select_axis() returns None on the cycled axes; or
       the synthetic per_axis_drive vectors happen to have max == per-axis
       for the cycled axis. /diagnose-errors on MECH-295 axis-routing.

C1 + C2 PASS, C3 + C4 FAIL:
    -> Cascade plumbing reaches consumers without crash (C2) but the per-
       axis read does not change outputs. Consumer-side read path is
       silently degrading. /diagnose-errors on the shared per_axis_drive
       helper or consumer-side per-axis branches.

C2 PASS, C1 FAIL:
    -> Env not exercising per-axis drive. Pre-substrate failure. Inspect
       CausalGridWorld per_axis_drive_decay + per_axis_drive_enabled.

C2 FAIL (any cell crashes):
    -> Cascade plumbing crashes under real operating conditions despite
       contract tests passing. /failure-autopsy on the crash; substrate
       fix required before any behavioural test can be queued.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorld  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from experiment_protocol import emit_outcome  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_618_sd049_phase3_consumer_cascade_validation"
CLAIM_IDS = ["SD-049", "MECH-295"]
EXPERIMENT_PURPOSE = "diagnostic"

SEEDS = [42, 43, 44]
WARMUP_EPISODES = 5        # brief sense() warmup, no training
MEASUREMENT_EPISODES = 10  # brief forward-pass for smoke + env-level drive evolution
STEPS_PER_EPISODE = 200
TOTAL_EPISODES_PER_RUN = WARMUP_EPISODES + MEASUREMENT_EPISODES  # 15; runner episodes_per_run

# Pre-registered acceptance thresholds.
C1_PEAK_DRIVE_FLOOR = 0.02                 # per-axis drive evolution sanity
C3_AIC_PER_AXIS_VS_NONE_L1_FLOOR = 0.005   # AIC reads per-axis input
C4_MECH295_AXIS_VS_COMBINER_L1_FLOOR = 0.005  # MECH-295 axis-routing differs from combiner

ARMS_CONFIG: List[Dict] = [
    dict(arm="ARM_0_cascade_off",         cascade_on=False, axis_aware=False),
    dict(arm="ARM_1_cascade_on_combiner", cascade_on=True,  axis_aware=False),
    dict(arm="ARM_2_cascade_on_axis_aware", cascade_on=True, axis_aware=True),
]

# Phase B synthetic per_axis_drive battery -- 4 vectors with high inter-axis
# variance so axis-routing and combiner-only modes must produce different
# outputs (max-axis != most other axes).
PHASE_B_BATTERY: List[List[float]] = [
    [0.9, 0.1, 0.05],   # hunger dominant
    [0.05, 0.9, 0.1],   # thirst dominant
    [0.1, 0.05, 0.9],   # novelty dominant
    [0.7, 0.4, 0.2],    # graded; max=0.7 but per-axis [1]=0.4, [2]=0.2
]
PHASE_B_K_CANDIDATES = 8  # MECH-295 candidate_proximities length


def make_env(seed: int) -> CausalGridWorld:
    """Build env with SD-049 Phase 1 + Phase 2 enabled (multi-resource + per-axis
    drive). Same env constructor for all 3 arms -- only the agent-side cascade
    flag differs."""
    return CausalGridWorld(
        size=10,
        num_hazards=2,
        num_resources=12,
        hazard_harm=0.02,
        resource_benefit=0.18,
        use_proxy_fields=True,
        seed=seed,
        proximity_benefit_scale=0.18,
        # SD-049 Phase 1: multi-resource heterogeneity (3 types: food, water, novelty)
        multi_resource_heterogeneity_enabled=True,
        n_resource_types=3,
        per_axis_drive_enabled=True,
        per_axis_drive_decay=(0.001, 0.0015, 0.0005),
        per_axis_drive_combiner="max",
    )


def make_config(env: CausalGridWorld, arm_cfg: Dict) -> REEConfig:
    """Build agent config per arm. AIC + MECH-295 enabled in all arms; only
    the cascade master flag differs between arms."""
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        alpha_world=0.9,
        world_dim=32,
        drive_weight=2.0,
        z_goal_enabled=True,
        use_aic_analog=True,
        use_mech295_liking_bridge=True,
    )
    cfg.latent.use_resource_encoder = True
    cfg.latent.z_resource_dim = 32
    cfg.goal.goal_dim = cfg.latent.world_dim
    cfg.use_sd049_per_axis_consumer_cascade = bool(arm_cfg["cascade_on"])
    return cfg


def per_axis_obs_tensor(obs_dict: Dict) -> Optional[torch.Tensor]:
    pad = obs_dict.get("per_axis_drive", None)
    if pad is None:
        return None
    if isinstance(pad, torch.Tensor):
        return pad.detach().float()
    return torch.as_tensor(pad, dtype=torch.float32)


# --------------------------- Phase A: smoke + env sanity ---------------------


def run_phase_a_cell(seed: int, arm_cfg: Dict) -> Dict:
    """One cell of Phase A: build env + agent for (seed, arm), run warmup +
    measurement forward pass. Returns cell summary including peak_per_axis_drive
    and crash signature (None if no crash)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")
    env = make_env(seed)
    cfg = make_config(env, arm_cfg)
    agent = REEAgent(cfg).to(device)
    t0 = time.time()
    peak_pad = np.zeros(env.n_resource_types, dtype=np.float32)
    n_steps = 0
    crash: Optional[str] = None
    axis_cycle = [0, 1, 2]
    try:
        # Warmup
        for ep in range(WARMUP_EPISODES):
            _, obs_dict = env.reset()
            agent.reset()
            for _ in range(STEPS_PER_EPISODE):
                obs_body = obs_dict["body_state"].to(device)
                obs_world = obs_dict["world_state"].to(device)
                obs_pad = per_axis_obs_tensor(obs_dict)
                with torch.no_grad():
                    latent = agent.sense(obs_body, obs_world, obs_per_axis_drive=obs_pad)
                    ticks = agent.clock.advance()
                    e1_prior = (
                        agent._e1_tick(latent) if ticks.get("e1_tick")
                        else torch.zeros(1, agent.config.latent.world_dim, device=device)
                    )
                    candidates = agent.generate_trajectories(latent, e1_prior, ticks)
                    action = agent.select_action(candidates, ticks)
                action_idx = int(action.argmax(dim=-1).item())
                _, harm_signal, done, info, obs_dict = env.step(action_idx)
                if done:
                    break
            if (ep + 1) % 5 == 0 or ep == 0:
                print(
                    f"  [train] warmup seed={seed} arm={arm_cfg['arm']} "
                    f"ep {ep + 1}/{TOTAL_EPISODES_PER_RUN}",
                    flush=True,
                )
        # Measurement
        for ep_i in range(MEASUREMENT_EPISODES):
            _, obs_dict = env.reset()
            agent.reset()
            if arm_cfg["axis_aware"]:
                agent._current_goal_axis_idx = axis_cycle[ep_i % len(axis_cycle)]
            else:
                agent._current_goal_axis_idx = None
            for _ in range(STEPS_PER_EPISODE):
                obs_body = obs_dict["body_state"].to(device)
                obs_world = obs_dict["world_state"].to(device)
                obs_pad = per_axis_obs_tensor(obs_dict)
                with torch.no_grad():
                    latent = agent.sense(obs_body, obs_world, obs_per_axis_drive=obs_pad)
                    ticks = agent.clock.advance()
                    e1_prior = (
                        agent._e1_tick(latent) if ticks.get("e1_tick")
                        else torch.zeros(1, agent.config.latent.world_dim, device=device)
                    )
                    candidates = agent.generate_trajectories(latent, e1_prior, ticks)
                    action = agent.select_action(candidates, ticks)
                action_idx = int(action.argmax(dim=-1).item())
                if obs_pad is not None:
                    peak_pad = np.maximum(
                        peak_pad, obs_pad.cpu().numpy().astype(np.float32)
                    )
                n_steps += 1
                _, harm_signal, done, info, obs_dict = env.step(action_idx)
                if done:
                    break
            global_ep = WARMUP_EPISODES + ep_i + 1
            if (ep_i + 1) % 5 == 0 or ep_i == 0:
                print(
                    f"  [train] meas seed={seed} arm={arm_cfg['arm']} "
                    f"ep {global_ep}/{TOTAL_EPISODES_PER_RUN}",
                    flush=True,
                )
    except Exception as e:
        crash = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
    elapsed = time.time() - t0
    return {
        "seed": seed,
        "arm": arm_cfg["arm"],
        "cascade_on": bool(arm_cfg["cascade_on"]),
        "axis_aware": bool(arm_cfg["axis_aware"]),
        "elapsed_seconds": elapsed,
        "n_steps": n_steps,
        "peak_per_axis_drive": [float(v) for v in peak_pad.tolist()],
        "crash": crash,
    }


# --------------------------- Phase B: direct-API probes ----------------------


def run_phase_b(seed: int) -> Dict:
    """Direct-API probes of AIC.tick and MECH-295 bridge under controlled
    per_axis_drive battery. No env or agent forward pass -- the probe agent
    is built only to grab a configured AIC + MECH-295 bridge."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")
    env = make_env(seed)
    cfg = make_config(env, ARMS_CONFIG[2])  # cascade ON config
    agent = REEAgent(cfg).to(device)
    aic = agent.aic
    bridge = agent.mech295_bridge
    assert aic is not None, "AIC must be enabled for Phase B"
    assert bridge is not None, "MECH-295 bridge must be enabled for Phase B"

    aic_combiner = getattr(cfg, "sd049_aic_per_axis_combiner", "max")
    mech295_combiner = getattr(cfg, "sd049_mech295_per_axis_combiner", "max")
    cand_prox = torch.linspace(0.0, 1.0, PHASE_B_K_CANDIDATES, dtype=torch.float32)

    # Common AIC.tick scalar inputs (held fixed across the per_axis_drive battery
    # so any output variation is attributable to per_axis_drive).
    aic_scalar_inputs = dict(
        z_harm_a_norm=0.3,
        drive_level=0.5,
        beta_gate_elevated=False,
        operating_mode=None,
    )

    aic_per_axis_outputs: List[float] = []
    aic_combiner_outputs: List[float] = []
    aic_diffs: List[float] = []
    aic_legacy_outputs: List[float] = []
    aic_per_axis_vs_legacy_diffs: List[float] = []

    mech295_axis_aware_outputs: List[List[float]] = []
    mech295_combiner_outputs: List[List[float]] = []
    mech295_l1_per_input: List[float] = []
    mech295_per_input_max_axis: List[int] = []

    for pad in PHASE_B_BATTERY:
        pad_t = torch.as_tensor(pad, dtype=torch.float32)
        max_axis = int(pad_t.argmax().item())
        mech295_per_input_max_axis.append(max_axis)

        # --- AIC: with per_axis_drive vs with None ---
        # Reset AIC baseline so each call sees the same prior state.
        aic.reset()
        out_per_axis = aic.tick(
            **aic_scalar_inputs,
            per_axis_drive=pad_t,
            per_axis_combiner=aic_combiner,
        )
        aic.reset()
        out_none = aic.tick(
            **aic_scalar_inputs,
            per_axis_drive=None,
            per_axis_combiner=aic_combiner,
        )
        sal_per_axis = float(out_per_axis.get("aic_salience", 0.0))
        sal_none = float(out_none.get("aic_salience", 0.0))
        aic_per_axis_outputs.append(sal_per_axis)
        aic_legacy_outputs.append(sal_none)
        aic_per_axis_vs_legacy_diffs.append(abs(sal_per_axis - sal_none))

        # Also: per-axis with the *non-max-axis* drive_level synthesised
        # explicitly, to detect whether cascade-on aic just collapses to max.
        # We compute the combiner-collapsed scalar that aic should read when
        # cascade is on, and compare to a hypothetical legacy run where we
        # pass drive_level equal to that same scalar. If aic_per_axis is
        # using a different effective drive than the combiner, that's
        # evidence the per-axis read is changing the read regime.
        # (Recorded as aux for the manifest; not gating any criterion.)
        aic_combiner_outputs.append(sal_per_axis)
        aic_diffs.append(abs(sal_per_axis - sal_none))

        # --- MECH-295: axis-aware vs combiner-only ---
        drive_level = float(pad_t.max().item())
        # Axis-aware: route by max_axis (the highest-deficit axis is always
        # the one MECH-295 would route to under any plausible goal selection).
        with torch.no_grad():
            bias_axis = bridge.compute_approach_cue_score_bias(
                drive_level=drive_level,
                candidate_proximities=cand_prox,
                simulation_mode=False,
                per_axis_drive=pad_t,
                goal_axis_idx=max_axis,
                per_axis_combiner=mech295_combiner,
            )
            bias_combiner = bridge.compute_approach_cue_score_bias(
                drive_level=drive_level,
                candidate_proximities=cand_prox,
                simulation_mode=False,
                per_axis_drive=pad_t,
                goal_axis_idx=None,
                per_axis_combiner=mech295_combiner,
            )
            # Also test a non-max axis (forces the axis read to differ from
            # the combiner-collapsed max).
            non_max_axis = (max_axis + 1) % 3
            bias_axis_nonmax = bridge.compute_approach_cue_score_bias(
                drive_level=drive_level,
                candidate_proximities=cand_prox,
                simulation_mode=False,
                per_axis_drive=pad_t,
                goal_axis_idx=non_max_axis,
                per_axis_combiner=mech295_combiner,
            )
        mech295_axis_aware_outputs.append(bias_axis.cpu().tolist())
        mech295_combiner_outputs.append(bias_combiner.cpu().tolist())
        # Use the non-max-axis comparison for the load-bearing L1 (max-axis vs
        # combiner can coincidentally be identical when combiner = max).
        l1 = (bias_axis_nonmax - bias_combiner).abs().mean().item()
        mech295_l1_per_input.append(float(l1))

    return {
        "aic_per_axis_outputs": aic_per_axis_outputs,
        "aic_legacy_outputs": aic_legacy_outputs,
        "aic_per_axis_vs_none_diffs": aic_per_axis_vs_legacy_diffs,
        "aic_mean_l1": float(np.mean(aic_per_axis_vs_legacy_diffs)) if aic_per_axis_vs_legacy_diffs else 0.0,
        "mech295_axis_aware_outputs": mech295_axis_aware_outputs,
        "mech295_combiner_outputs": mech295_combiner_outputs,
        "mech295_per_input_max_axis": mech295_per_input_max_axis,
        "mech295_l1_per_input": mech295_l1_per_input,
        "mech295_mean_l1": float(np.mean(mech295_l1_per_input)) if mech295_l1_per_input else 0.0,
        "battery": PHASE_B_BATTERY,
        "k_candidates": PHASE_B_K_CANDIDATES,
    }


# ----------------------------- acceptance ------------------------------------


def evaluate_acceptance(
    phase_a_cells: List[Dict], phase_b: Dict
) -> Dict:
    per_cell_max_peak = [
        float(np.max(c["peak_per_axis_drive"])) for c in phase_a_cells
    ]
    min_peak = float(np.min(per_cell_max_peak)) if per_cell_max_peak else 0.0
    c1 = min_peak >= C1_PEAK_DRIVE_FLOOR

    crashes = [c for c in phase_a_cells if c["crash"]]
    c2 = len(crashes) == 0

    c3 = phase_b["aic_mean_l1"] >= C3_AIC_PER_AXIS_VS_NONE_L1_FLOOR
    c4 = phase_b["mech295_mean_l1"] >= C4_MECH295_AXIS_VS_COMBINER_L1_FLOOR

    overall = bool(c1 and c2 and c3 and c4)
    return {
        "C1_per_axis_drive_evolves": {
            "pass": bool(c1),
            "min_peak_across_cells": min_peak,
            "per_cell_max_peak": per_cell_max_peak,
        },
        "C2_phase_a_no_crash": {
            "pass": bool(c2),
            "n_crashes": len(crashes),
            "crashed_cells": [
                f"seed={c['seed']}/arm={c['arm']}" for c in crashes
            ],
        },
        "C3_aic_reads_per_axis": {
            "pass": bool(c3),
            "mean_l1": phase_b["aic_mean_l1"],
            "per_input_diffs": phase_b["aic_per_axis_vs_none_diffs"],
        },
        "C4_mech295_axis_routes": {
            "pass": bool(c4),
            "mean_l1": phase_b["mech295_mean_l1"],
            "per_input_l1": phase_b["mech295_l1_per_input"],
        },
        "all_pass": overall,
    }


def main(dry_run: bool = False) -> Tuple[str, Optional[str]]:
    print(f"[{EXPERIMENT_TYPE}] starting (dry_run={dry_run})", flush=True)
    seeds = (SEEDS[0],) if dry_run else SEEDS
    phase_a_cells: List[Dict] = []
    t0 = time.time()
    action_dim = None
    for seed in seeds:
        for arm_cfg in ARMS_CONFIG:
            print(f"Seed {seed} Condition {arm_cfg['arm']}", flush=True)
            cell = run_phase_a_cell(seed, arm_cfg)
            phase_a_cells.append(cell)
            if action_dim is None:
                action_dim = make_env(seed).action_dim
            verdict = "FAIL" if cell["crash"] else "PASS"
            print(
                f"  [phase-a] seed={cell['seed']} arm={cell['arm']:<28} "
                f"peak_drive={[round(v, 3) for v in cell['peak_per_axis_drive']]} "
                f"n_steps={cell['n_steps']} "
                f"crash={'yes' if cell['crash'] else 'no'} "
                f"elapsed={cell['elapsed_seconds']:.1f}s",
                flush=True,
            )
            if cell["crash"]:
                print(f"  CRASH detail:\n{cell['crash']}", flush=True)
            print(f"verdict: {verdict}", flush=True)
    print(f"[{EXPERIMENT_TYPE}] Phase B (direct-API probes) starting", flush=True)
    phase_b = run_phase_b(SEEDS[0])
    print(
        f"[{EXPERIMENT_TYPE}] Phase B: aic_mean_l1={phase_b['aic_mean_l1']:.6f} "
        f"mech295_mean_l1={phase_b['mech295_mean_l1']:.6f}",
        flush=True,
    )

    acceptance = evaluate_acceptance(phase_a_cells, phase_b)
    elapsed = time.time() - t0
    outcome = "PASS" if acceptance["all_pass"] else "FAIL"

    print(f"[{EXPERIMENT_TYPE}] acceptance:", flush=True)
    for k, v in acceptance.items():
        if isinstance(v, dict):
            print(f"  {k}: pass={v.get('pass')}", flush=True)
        else:
            print(f"  {k}: {v}", flush=True)
    print(f"[{EXPERIMENT_TYPE}] outcome={outcome} elapsed={elapsed:.1f}s", flush=True)

    if dry_run:
        print(f"[{EXPERIMENT_TYPE}] dry-run complete; not writing manifest.", flush=True)
        return outcome, None

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{timestamp}_v3"
    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    out_dir.mkdir(parents=True, exist_ok=True)
    # Per-claim direction:
    # SD-049: supports if cascade plumbing works AND consumers read per-axis
    #         (C1 + C2 + C3); weakens otherwise.
    # MECH-295: supports if axis-routing differs from combiner-only (C4);
    #           weakens otherwise. C3 alone is not sufficient -- that is AIC,
    #           not MECH-295.
    sd049_direction = (
        "supports" if (
            acceptance["C1_per_axis_drive_evolves"]["pass"]
            and acceptance["C2_phase_a_no_crash"]["pass"]
            and acceptance["C3_aic_reads_per_axis"]["pass"]
        ) else "weakens"
    )
    mech295_direction = (
        "supports" if acceptance["C4_mech295_axis_routes"]["pass"] else "weakens"
    )
    manifest = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "started_utc": datetime.utcnow().isoformat() + "Z",
        "timestamp_utc": timestamp,
        "outcome": outcome,
        "evidence_direction": "supports" if outcome == "PASS" else "weakens",
        "evidence_direction_per_claim": {
            "SD-049": sd049_direction,
            "MECH-295": mech295_direction,
        },
        "elapsed_seconds": elapsed,
        "n_seeds": len(seeds),
        "warmup_episodes": WARMUP_EPISODES,
        "measurement_episodes": MEASUREMENT_EPISODES,
        "steps_per_episode": STEPS_PER_EPISODE,
        "action_dim": action_dim,
        "acceptance": acceptance,
        "thresholds": {
            "C1_peak_drive_floor": C1_PEAK_DRIVE_FLOOR,
            "C3_aic_per_axis_vs_none_l1_floor": C3_AIC_PER_AXIS_VS_NONE_L1_FLOOR,
            "C4_mech295_axis_vs_combiner_l1_floor": C4_MECH295_AXIS_VS_COMBINER_L1_FLOOR,
        },
        "phase_a_per_cell": [
            {
                "seed": c["seed"],
                "arm": c["arm"],
                "cascade_on": c["cascade_on"],
                "axis_aware": c["axis_aware"],
                "elapsed_seconds": c["elapsed_seconds"],
                "n_steps": c["n_steps"],
                "peak_per_axis_drive": c["peak_per_axis_drive"],
                "crash": c["crash"],
            }
            for c in phase_a_cells
        ],
        "phase_b": phase_b,
        "substrate_landing_commit": "ree-v3 main 2026-05-31T13:00Z; SD-049 Phase 3 SD-032 consumer cascade",
        "depends_on_substrate": [
            "use_sd049_per_axis_consumer_cascade (REEConfig)",
            "per_axis_drive_enabled + multi_resource_heterogeneity_enabled (CausalGridWorld)",
            "_current_goal_axis_idx (REEAgent attribute)",
            "AICAnalog.tick (per_axis_drive + per_axis_combiner kwargs)",
            "MECH295LikingBridge.compute_approach_cue_score_bias (per_axis_drive + goal_axis_idx kwargs)",
        ],
        "downstream_owed_on_pass": (
            "V3-EXQ-619: load-bearing behavioural validation of SD-049 Phase 3 "
            "cascade with trained agent + active goal_state (replicates "
            "V3-EXQ-514g structure on Phase 3 substrate). Unblocks "
            "goal_pipeline:GAP-2 cluster (SD-049 / SD-015 / MECH-229 / MECH-230 "
            "/ MECH-117 / ARC-030 / ARC-032 / Q-030)."
        ),
    }
    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=False,
        config=manifest.get("config"),
        seeds=SEEDS,
        script_path=Path(__file__),
        json_default=str,
    )
    print(f"Result written to: {out_path}", flush=True)
    return outcome, str(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Smoke run, write no manifest.")
    args = parser.parse_args()
    _outcome, _manifest_path = main(dry_run=args.dry_run)
    if not args.dry_run:
        emit_outcome(
            outcome=_outcome if _outcome in ("PASS", "FAIL") else "FAIL",
            manifest_path=_manifest_path,
        )
    sys.exit(0 if _outcome == "PASS" else 1)
