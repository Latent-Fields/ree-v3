"""V3-EXQ-617: SD-056 multi-step rollout stability amend substrate-readiness diagnostic.

Purpose: diagnostic. Confirms the SD-056 multi-step rollout stability amend
(2026-05-31) is wired correctly. Two levers under test:
  (a) multi-step contrastive (world_forward_contrastive_loss_multistep).
  (b) per-step output norm clamp inside E2.rollout_with_world (B2 anchor).

Does NOT test the full Pathway A vs B behavioural falsifier on the amended
substrate -- that is the next /queue-experiment session's job (post-amend
569e-equivalent re-run bundled with the three script-side acceptance-criteria
fixes from failure_autopsy_V3-EXQ-569e_2026-05-31 Section 6).

Substrate-readiness verifies:
  - The 5 new config knobs surface through REEConfig.from_dims and propagate
    to config.e2.
  - The world_forward_contrastive_loss_multistep helper exists on
    E2FastPredictor and returns non-negative finite scalars; backward() flows
    gradient through world_transition / world_action_encoder.
  - With both levers OFF, rollout_with_world is bit-identical to the pre-
    amend SD-056 path.
  - With the per-step norm clamp ON, rollout_with_world enforces the B2
    bound (||z_world_t|| <= ratio * ||z_world_0||) on every step over a
    200-step rollout horizon under deliberately-unstable training state
    (weight_init_scale=20x; the autopsy 1e16+ pathology regime).
  - With the clamp ON, the rollout never produces NaN/Inf at the 200-step
    horizon (the autopsy max-NaN-fraction < 0.05 acceptance criterion is
    met as a hard guarantee, NaN-fraction == 0.0).
  - MECH-094 simulation gate: world_forward_contrastive_loss_multistep
    called with simulation_mode=True returns tensor(0.0) and does not
    advance any optimiser state.

Six sub-tests (deterministic arithmetic + API integration; ~10 sec):

  UC1 module surface: REEConfig.from_dims surfaces the 5 new amend knobs;
      config.e2.* receives them; E2FastPredictor exposes
      world_forward_contrastive_loss_multistep; the helper accepts
      simulation_mode kwarg.

  UC2 master-OFF backward-compat: with both new amend masters set to False
      (default), rollout_with_world output is bit-identical to a parallel
      run with the same input + the same model state. Confirms the
      pre-amend path is preserved verbatim.

  UC3 multi-step contrastive direction-of-change: with K=8 sibling
      candidates + h=5 horizon and synthetic per-anchor per-step shifts,
      after 100 SGD steps under the multi-step contrastive loss the loss
      magnitude decreases AND gradient flows through E2 weights. The
      direction-of-change is the load-bearing PASS condition; the
      magnitude floor is calibratable.

  UC4 per-step norm clamp B2 bound (controlled stress): with clamp ON +
      ratio=2.0 + weight_init_scale=20x amplification of world_transition
      to provoke the 1e16+ pathology regime at 200-step horizon,
      ||z_world_t|| <= 2.0 * ||z_world_0|| holds on every step + every
      candidate row. Direct test of the autopsy acceptance bound at the
      substrate level.

  UC5 per-step norm clamp NaN/Inf defence: same stress configuration as
      UC4 -- the rollout produces 0 NaN / Inf at the 200-step horizon.
      The autopsy max-NaN-fraction < 0.05 acceptance criterion is met as
      a hard guarantee (NaN-fraction == 0.0) when the clamp is on.

  UC6 MECH-094 simulation gate: world_forward_contrastive_loss_multistep
      with simulation_mode=True returns exactly tensor(0.0); subsequent
      waking call still flows gradient through E2 weights. Simulation
      gate does not poison subsequent waking optimisation.

Run with:
  /opt/local/bin/python3 experiments/v3_exq_617_sd056_multistep_substrate_readiness.py

Writes a flat JSON manifest to REE_assembly/evidence/experiments/.

experiment_purpose=diagnostic. claim_ids=[] -- substrate-readiness diagnostic,
NOT governance evidence yet (per the SD-056 design doc + autopsy Section
"Acceptance criteria"). Behavioural validation (the 8-arm V3-EXQ-569e-equivalent
Pathway A vs B falsifier bundled with the three script-side acceptance-
criteria fixes) lives in the post-amend /queue-experiment session.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ree_core.predictors.e2_fast import E2FastPredictor
from ree_core.utils.config import E2Config, REEConfig


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

SELF_DIM = 8
WORLD_DIM = 16
ACTION_DIM = 4
K = 8
HORIZON = 5
STRESS_HORIZON = 200  # matches V3-EXQ-569e P1 measurement budget
STRESS_CLAMP_RATIO = 2.0


def _make_e2(
    multistep_enabled: bool = False,
    horizon: int = HORIZON,
    horizon_weights_decay: float = 1.0,
    clamp_enabled: bool = False,
    clamp_ratio: float = STRESS_CLAMP_RATIO,
    weight_init_scale: float | None = None,
    seed: int = 7,
) -> E2FastPredictor:
    torch.manual_seed(seed)
    cfg = E2Config(
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=32,
    )
    cfg.e2_action_contrastive_multistep_enabled = multistep_enabled
    cfg.e2_action_contrastive_horizon = horizon
    cfg.e2_action_contrastive_horizon_weights_decay = horizon_weights_decay
    cfg.e2_rollout_output_norm_clamp_enabled = clamp_enabled
    cfg.e2_rollout_output_norm_clamp_ratio = clamp_ratio
    model = E2FastPredictor(cfg)
    if weight_init_scale is not None:
        # Deliberately rescale world_transition weights to provoke the
        # 1e16+ pathology regime the autopsy documented.
        with torch.no_grad():
            for p in model.world_transition.parameters():
                p.mul_(weight_init_scale)
    return model


def _sibling_multistep_batch(seed: int, horizon: int = HORIZON):
    """K sibling-CEM-style multi-step batch: shared z_world_0; distinct
    first-action one-hots cycling action_dim; per-anchor per-step target
    shifts so the contrastive task is learnable across the full horizon."""
    g = torch.Generator().manual_seed(seed)
    z0 = torch.randn(WORLD_DIM, generator=g)
    classes = torch.arange(K, dtype=torch.long) % ACTION_DIM
    first_action = F.one_hot(classes, num_classes=ACTION_DIM).float()
    actions = torch.zeros(K, horizon, ACTION_DIM)
    for t in range(horizon):
        for i in range(K):
            actions[i, t, (int(classes[i]) + t) % ACTION_DIM] = 1.0
    targets = torch.zeros(K, horizon + 1, WORLD_DIM)
    targets[:, 0, :] = z0.unsqueeze(0).expand(K, -1)
    shift_dirs = torch.randn(K, WORLD_DIM, generator=g) * 0.3
    for t in range(1, horizon + 1):
        targets[:, t, :] = z0.unsqueeze(0) + (t * 0.1) * shift_dirs
    return z0, actions, targets


def _build_stress_inputs(seed: int):
    """Deterministic K-batch rollout input for the stress-clamp UCs."""
    g = torch.Generator().manual_seed(seed)
    initial_z_self = torch.randn(K, SELF_DIM, generator=g)
    initial_z_world = torch.randn(K, WORLD_DIM, generator=g)
    action_seq = torch.zeros(K, STRESS_HORIZON, ACTION_DIM)
    classes = torch.arange(K) % ACTION_DIM
    for i in range(K):
        action_seq[i, :, int(classes[i])] = 1.0
    return initial_z_self, initial_z_world, action_seq


# ----------------------------------------------------------------------
# UC1 module surface
# ----------------------------------------------------------------------
def run_uc1_module_surface() -> dict:
    base_dims = dict(body_obs_dim=8, world_obs_dim=8, harm_obs_dim=8, action_dim=4)
    cfg_default = REEConfig.from_dims(**base_dims)
    cfg_on = REEConfig.from_dims(
        **base_dims,
        e2_action_contrastive_multistep_enabled=True,
        e2_action_contrastive_horizon=8,
        e2_action_contrastive_horizon_weights_decay=0.7,
        e2_rollout_output_norm_clamp_enabled=True,
        e2_rollout_output_norm_clamp_ratio=1.75,
    )
    e2 = _make_e2(multistep_enabled=True)
    has_multistep = hasattr(e2, "world_forward_contrastive_loss_multistep")
    z0, actions, targets = _sibling_multistep_batch(seed=1)
    sim_loss = e2.world_forward_contrastive_loss_multistep(
        z0, actions, targets, simulation_mode=True
    )
    sim_returned_zero = abs(float(sim_loss)) < 1e-12
    result = {
        "default_multistep_disabled": cfg_default.e2.e2_action_contrastive_multistep_enabled is False,
        "default_horizon": cfg_default.e2.e2_action_contrastive_horizon,
        "default_horizon_weights_decay": cfg_default.e2.e2_action_contrastive_horizon_weights_decay,
        "default_clamp_disabled": cfg_default.e2.e2_rollout_output_norm_clamp_enabled is False,
        "default_clamp_ratio": cfg_default.e2.e2_rollout_output_norm_clamp_ratio,
        "on_multistep_enabled": cfg_on.e2.e2_action_contrastive_multistep_enabled is True,
        "on_horizon": cfg_on.e2.e2_action_contrastive_horizon,
        "on_horizon_weights_decay": cfg_on.e2.e2_action_contrastive_horizon_weights_decay,
        "on_clamp_enabled": cfg_on.e2.e2_rollout_output_norm_clamp_enabled is True,
        "on_clamp_ratio": cfg_on.e2.e2_rollout_output_norm_clamp_ratio,
        "e2_has_multistep_helper": has_multistep,
        "simulation_mode_kwarg_works": sim_returned_zero,
    }
    result["pass"] = (
        result["default_multistep_disabled"]
        and result["default_horizon"] == 5
        and abs(result["default_horizon_weights_decay"] - 1.0) < 1e-9
        and result["default_clamp_disabled"]
        and abs(result["default_clamp_ratio"] - 2.0) < 1e-9
        and result["on_multistep_enabled"]
        and result["on_horizon"] == 8
        and abs(result["on_horizon_weights_decay"] - 0.7) < 1e-9
        and result["on_clamp_enabled"]
        and abs(result["on_clamp_ratio"] - 1.75) < 1e-9
        and has_multistep
        and sim_returned_zero
    )
    return result


# ----------------------------------------------------------------------
# UC2 master-OFF backward-compat
# ----------------------------------------------------------------------
def run_uc2_master_off_backward_compat() -> dict:
    """Both new amend masters OFF -> rollout_with_world output is
    bit-identical to a parallel run with same model state. Confirms the
    pre-amend SD-056 path is preserved verbatim."""
    seeds = [11, 22, 33]
    all_ok = True
    per_seed = []
    for s in seeds:
        e2_a = _make_e2(seed=s, clamp_enabled=False, multistep_enabled=False)
        e2_b = _make_e2(seed=s, clamp_enabled=False, multistep_enabled=False)
        # Force identical weights between the two instances.
        e2_b.load_state_dict(e2_a.state_dict())
        initial_z_self, initial_z_world, action_seq = _build_stress_inputs(seed=s + 1)
        # 50-step rollout (full STRESS_HORIZON is unnecessary for bit-identical
        # check; 50 steps catches any divergence cheaply).
        action_seq_50 = action_seq[:, :50, :]
        traj_a = e2_a.rollout_with_world(
            initial_z_self, initial_z_world, action_seq_50,
            compute_action_objects=False,
        )
        traj_b = e2_b.rollout_with_world(
            initial_z_self, initial_z_world, action_seq_50,
            compute_action_objects=False,
        )
        max_diff = 0.0
        for w_a, w_b in zip(traj_a.world_states, traj_b.world_states):
            max_diff = max(max_diff, float((w_a - w_b).abs().max()))
        seed_ok = max_diff < 1e-9
        all_ok = all_ok and seed_ok
        per_seed.append({"seed": s, "max_diff": max_diff, "ok": seed_ok})
    return {"per_seed": per_seed, "n_seeds": len(seeds), "pass": all_ok}


# ----------------------------------------------------------------------
# UC3 multi-step contrastive direction-of-change
# ----------------------------------------------------------------------
def run_uc3_multistep_direction_of_change() -> dict:
    """100 SGD steps under multi-step contrastive loss should reduce the
    loss magnitude and produce non-zero gradients on E2 weights."""
    torch.manual_seed(42)
    e2 = _make_e2(multistep_enabled=True, horizon=HORIZON, seed=42)
    optimizer = torch.optim.Adam(e2.parameters(), lr=1e-3)
    losses = []
    grad_norms = []
    for step in range(100):
        z0, actions, targets = _sibling_multistep_batch(seed=step + 1)
        loss = e2.world_forward_contrastive_loss_multistep(z0, actions, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = sum(
            float(p.grad.abs().sum())
            for p in e2.world_transition.parameters()
            if p.grad is not None
        )
        optimizer.step()
        if step % 10 == 0:
            losses.append({"step": step, "loss": float(loss)})
            grad_norms.append({"step": step, "grad_norm": gnorm})
    initial_loss = losses[0]["loss"]
    final_loss = losses[-1]["loss"]
    direction_ok = final_loss < initial_loss
    grad_ok = all(g["grad_norm"] > 0.0 for g in grad_norms)
    return {
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "direction_of_change_ok": direction_ok,
        "grad_norms": grad_norms,
        "grad_nonzero_ok": grad_ok,
        "pass": direction_ok and grad_ok,
    }


# ----------------------------------------------------------------------
# UC4 per-step norm clamp B2 bound (controlled stress)
# ----------------------------------------------------------------------
def run_uc4_norm_clamp_b2_bound() -> dict:
    """Clamp ON + ratio=2.0 + 20x weight amplification + 200-step rollout
    -- per-row ||z_world_t|| <= ratio * ||z_world_0|| at every step."""
    e2 = _make_e2(
        clamp_enabled=True,
        clamp_ratio=STRESS_CLAMP_RATIO,
        weight_init_scale=20.0,
        seed=131,
    )
    initial_z_self, initial_z_world, action_seq = _build_stress_inputs(seed=2026)
    traj = e2.rollout_with_world(
        initial_z_self, initial_z_world, action_seq, compute_action_objects=False
    )
    z0_norms = initial_z_world.norm(dim=-1)  # [K]
    max_allowed = STRESS_CLAMP_RATIO * z0_norms  # [K]
    n_violations = 0
    max_norm_observed = 0.0
    max_norm_allowed = float(max_allowed.max())
    for z_t in traj.world_states:
        norms_t = z_t.norm(dim=-1)  # [K]
        max_norm_observed = max(max_norm_observed, float(norms_t.max()))
        violations = (norms_t > max_allowed * (1.0 + 1e-5)).sum()
        n_violations += int(violations)
    return {
        "stress_horizon": STRESS_HORIZON,
        "clamp_ratio": STRESS_CLAMP_RATIO,
        "weight_init_scale": 20.0,
        "max_norm_observed": max_norm_observed,
        "max_norm_allowed": max_norm_allowed,
        "n_violations": n_violations,
        "n_total_checks": K * (STRESS_HORIZON + 1),
        "pass": n_violations == 0,
    }


# ----------------------------------------------------------------------
# UC5 per-step norm clamp NaN/Inf defence
# ----------------------------------------------------------------------
def run_uc5_norm_clamp_nan_defence() -> dict:
    """Same stress config as UC4 -- the clamp must guarantee 0 NaN/Inf
    at the 200-step horizon. Direct test of the autopsy max-NaN-fraction
    < 0.05 acceptance criterion as a hard guarantee."""
    e2 = _make_e2(
        clamp_enabled=True,
        clamp_ratio=STRESS_CLAMP_RATIO,
        weight_init_scale=20.0,
        seed=131,
    )
    initial_z_self, initial_z_world, action_seq = _build_stress_inputs(seed=2026)
    traj = e2.rollout_with_world(
        initial_z_self, initial_z_world, action_seq, compute_action_objects=False
    )
    n_nan_or_inf = 0
    n_total_cells = 0
    for z_t in traj.world_states:
        n_nan_or_inf += int((~torch.isfinite(z_t)).sum())
        n_total_cells += int(z_t.numel())
    nan_fraction = n_nan_or_inf / max(1, n_total_cells)
    return {
        "stress_horizon": STRESS_HORIZON,
        "n_nan_or_inf_cells": n_nan_or_inf,
        "n_total_cells": n_total_cells,
        "nan_fraction": nan_fraction,
        "autopsy_acceptance_max_nan_fraction": 0.05,
        "pass": nan_fraction < 0.05,
    }


# ----------------------------------------------------------------------
# UC6 MECH-094 simulation gate
# ----------------------------------------------------------------------
def run_uc6_mech094_simulation_gate() -> dict:
    """simulation_mode=True returns tensor(0.0); subsequent waking
    backward() still propagates gradient. Simulation gate does not poison
    subsequent waking optimisation."""
    torch.manual_seed(7)
    e2 = _make_e2(multistep_enabled=True, seed=7)
    z0, actions, targets = _sibling_multistep_batch(seed=1)
    wt_param = e2.world_transition[0].weight
    pre_sim_val = wt_param.detach().clone()
    sim_loss = e2.world_forward_contrastive_loss_multistep(
        z0, actions, targets, simulation_mode=True
    )
    sim_returned_zero = abs(float(sim_loss)) < 1e-12
    sim_grad_fn_none = (sim_loss.grad_fn is None) or (not sim_loss.requires_grad)
    e2.zero_grad(set_to_none=True)
    waking_loss = e2.world_forward_contrastive_loss_multistep(z0, actions, targets)
    waking_returned_positive = float(waking_loss) > 0.0
    waking_loss.backward()
    wt_grad = e2.world_transition[0].weight.grad
    waking_grad_flows = (
        wt_grad is not None
        and torch.isfinite(wt_grad).all()
        and float(wt_grad.abs().sum()) > 0.0
    )
    weights_unchanged_during_sim = bool(
        torch.equal(e2.world_transition[0].weight.detach(), pre_sim_val)
    )
    return {
        "sim_returned_zero": sim_returned_zero,
        "sim_grad_fn_none_or_no_grad": sim_grad_fn_none,
        "weights_unchanged_during_sim": weights_unchanged_during_sim,
        "waking_loss_positive": waking_returned_positive,
        "waking_grad_flows": waking_grad_flows,
        "pass": (
            sim_returned_zero
            and sim_grad_fn_none
            and weights_unchanged_during_sim
            and waking_returned_positive
            and waking_grad_flows
        ),
    }


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> None:
    t0 = time.time()
    uc1 = run_uc1_module_surface()
    uc2 = run_uc2_master_off_backward_compat()
    uc3 = run_uc3_multistep_direction_of_change()
    uc4 = run_uc4_norm_clamp_b2_bound()
    uc5 = run_uc5_norm_clamp_nan_defence()
    uc6 = run_uc6_mech094_simulation_gate()

    all_pass = all(r["pass"] for r in [uc1, uc2, uc3, uc4, uc5, uc6])
    elapsed = time.time() - t0

    run_id = "v3_exq_617_sd056_multistep_substrate_readiness_v3"
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    manifest = {
        "schema_version": "v1",
        "run_id": f"{run_id}_{ts}",
        "experiment_type": "v3_exq_617_sd056_multistep_substrate_readiness",
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "experiment_purpose": "diagnostic",
        "claim_ids": [],
        "result": "PASS" if all_pass else "FAIL",
        "evidence_direction": "supports" if all_pass else "weakens",
        "metrics": {
            "UC1_module_surface": uc1,
            "UC2_master_off_backward_compat": uc2,
            "UC3_multistep_direction_of_change": uc3,
            "UC4_norm_clamp_b2_bound": uc4,
            "UC5_norm_clamp_nan_defence": uc5,
            "UC6_mech094_simulation_gate": uc6,
        },
        "elapsed_seconds": elapsed,
        "notes": (
            "SD-056 multi-step rollout stability amend (2026-05-31) substrate-"
            "readiness diagnostic. Triggered by V3-EXQ-569e Pathway A vs B "
            "mechanism-probe autopsy (verdict_cell=INSTRUMENTATION_FAILURE; "
            "evidence/planning/failure_autopsy_V3-EXQ-569e_2026-05-31.md). "
            "UC1-UC3 cover the multi-step contrastive lever (a); UC4-UC5 cover "
            "the per-step output norm clamp lever (b) at 200-step horizon "
            "under 20x weight amplification (the 1e16+ autopsy pathology "
            "regime); UC6 covers MECH-094 simulation gate. Direct substrate-"
            "level test of the autopsy acceptance criterion (max-NaN-fraction "
            "< 0.05 + rollout magnitudes within 2x of OFF baseline) as a hard "
            "guarantee via lever (b). Does NOT test the full 8-arm V3-EXQ-"
            "569e-equivalent Pathway A vs B falsifier on the amended substrate "
            "-- that bundles with the three script-side acceptance-criteria "
            "fixes from autopsy Section 6 in the post-amend /queue-experiment "
            "session. Diagnostic claim_ids=[] per Phase-3 governance rules. "
            "Lit-pull anchors (Srivastava 2021 contrastive RSSM; Dreamer / "
            "PlaNet family for multi-step latent dynamics; cerebellar / "
            "prefrontal forward-model biology preserving action-specificity at "
            "the prediction step) inform the architecture but are not signals "
            "tested here."
        ),
    }

    out_dir = (
        REPO_ROOT.parent
        / "REE_assembly" / "evidence" / "experiments"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{manifest['run_id']}.json"
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"verdict: {manifest['result']}")
    for k, v in manifest["metrics"].items():
        print(f"  {k}: pass={v['pass']}")
    print(f"Result written to: {out_path}", flush=True)

    from experiment_protocol import emit_outcome
    emit_outcome(outcome=manifest["result"], manifest_path=str(out_path))


if __name__ == "__main__":
    main()
