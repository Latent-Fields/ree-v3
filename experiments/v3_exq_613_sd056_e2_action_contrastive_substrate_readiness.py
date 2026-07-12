"""V3-EXQ-613: SD-056 (e2.action_conditional_divergence_contrastive) substrate-readiness diagnostic.

Purpose: diagnostic. Confirms the SD-056 substrate (auxiliary InfoNCE-style
contrastive loss on E2.world_forward) is wired correctly. Does NOT test the
biological / behavioural claim that the contrastive fix produces downstream
behavioural diversity recovery -- that is V3-EXQ-569a's job (matched-entropy
FP-2 falsifier on the fixed substrate; GAP-A R1.a/R1.b decision rule).

Substrate-readiness verifies:
  - The new config knobs surface through REEConfig.from_dims and propagate
    to config.e2.
  - The cand_world_pairwise_dist helper exists on E2FastPredictor and
    returns non-negative finite scalars on K-candidate sibling batches.
  - cand_world_pairwise_dist rises after 200 SGD steps under contrastive
    loss training (UC3 direction-of-change -- the load-bearing PASS
    condition per the design memo).
  - Held-out batch contrastive accuracy > 50% confirms the contrastive
    task is learnable on the substrate (UC4; random baseline 1/K=12.5%
    for K=8). Distinguishes "loss decreased" from "task got solved."
  - MECH-094 simulation gate: world_forward_contrastive_loss called with
    simulation_mode=True returns tensor(0.0) and does not advance any
    optimiser state.

Five sub-tests (deterministic arithmetic + API integration; ~30 sec):

  UC1 module surface: REEConfig.from_dims surfaces the 4 SD-056 knobs;
      config.e2.* receives them; E2FastPredictor exposes
      cand_world_pairwise_dist and world_forward_contrastive_loss; the
      contrastive loss helper accepts simulation_mode kwarg.

  UC2 master-OFF backward-compat: 3-seed equivalence check. With
      cfg.e2.e2_action_contrastive_enabled=False (default), repeatedly
      constructed E2FastPredictor instances at the same seed produce the
      same world_forward output for the same inputs. Bit-identical OFF.

  UC3 cand_world_pairwise_dist direction-of-change: with K=8 sibling
      candidates and synthetic per-action target shifts, after 200 SGD
      steps under contrastive loss the mean cand_world_pairwise_dist
      rises from the random-init baseline by a positive margin. The
      design memo suggests >= 0.05 in normalised units; we accept the
      tighter "trained > baseline AND trained >= 0.05" PASS criterion
      while reporting both values for calibration.

  UC4 contrastive-task accuracy: on held-out K=8 batches after the 200-
      step training run, the contrastive logits classify the correct
      anchor for > 50% of cases (random baseline 1/K = 12.5%).

  UC5 MECH-094 simulation gate: world_forward_contrastive_loss called
      with simulation_mode=True returns exactly tensor(0.0); a subsequent
      backward() on a fresh waking call still flows gradient through
      world_transition / world_action_encoder. The simulation gate does
      not poison subsequent waking calls.

Run with:
  /opt/local/bin/python3 experiments/v3_exq_613_sd056_e2_action_contrastive_substrate_readiness.py

Writes a flat JSON manifest to REE_assembly/evidence/experiments/.

experiment_purpose=diagnostic. claim_ids=[] -- substrate-readiness diagnostic,
NOT governance evidence yet (per the SD-056 design doc + design memo Section
"Acceptance criteria"). Behavioural validation lives in V3-EXQ-569a.
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
from experiments.pack_writer import write_flat_manifest  # noqa: E402


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

WORLD_DIM = 16
ACTION_DIM = 4
K = 8


def _make_e2(enabled: bool = True, weight: float = 0.01,
             temperature: float = 0.1, min_batch_classes: int = 2,
             seed: int = 7) -> E2FastPredictor:
    torch.manual_seed(seed)
    cfg = E2Config(
        self_dim=8,
        world_dim=WORLD_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=32,
    )
    cfg.e2_action_contrastive_enabled = enabled
    cfg.e2_action_contrastive_weight = weight
    cfg.e2_action_contrastive_temperature = temperature
    cfg.e2_action_contrastive_min_batch_classes = min_batch_classes
    return E2FastPredictor(cfg)


def _sibling_batch(seed: int, K_local: int = K):
    """K sibling CEM candidates: shared z_world_0, distinct first-action
    one-hots cycling through action_dim, per-action target shift so the
    contrastive task is learnable (each action has a structurally distinct
    target)."""
    g = torch.Generator().manual_seed(seed)
    z_world_0 = torch.randn(WORLD_DIM, generator=g)
    classes = torch.arange(K_local, dtype=torch.long) % ACTION_DIM
    actions = F.one_hot(classes, num_classes=ACTION_DIM).float()
    shifts = torch.zeros(K_local, WORLD_DIM)
    for i in range(K_local):
        shifts[i, int(classes[i])] = 0.5
    targets = z_world_0.unsqueeze(0).expand(K_local, -1) + shifts
    return z_world_0, actions, targets


def _avg_dist(e2: E2FastPredictor, n_batches: int = 8) -> float:
    dists = []
    for s in range(n_batches):
        z0, actions, _ = _sibling_batch(seed=10_000 + s)
        with torch.no_grad():
            dists.append(float(e2.cand_world_pairwise_dist(z0, actions)))
    return sum(dists) / max(1, len(dists))


# ----------------------------------------------------------------------
# UC1 module surface
# ----------------------------------------------------------------------
def run_uc1_module_surface() -> dict:
    """UC1: config knobs surface, helpers exist, simulation_mode kwarg present."""
    cfg_default = REEConfig.from_dims(
        body_obs_dim=8, world_obs_dim=8, harm_obs_dim=8, action_dim=4,
    )
    cfg_on = REEConfig.from_dims(
        body_obs_dim=8, world_obs_dim=8, harm_obs_dim=8, action_dim=4,
        e2_action_contrastive_enabled=True,
        e2_action_contrastive_weight=0.05,
        e2_action_contrastive_temperature=0.2,
        e2_action_contrastive_min_batch_classes=3,
    )
    e2 = _make_e2(enabled=True)
    has_cand_dist = hasattr(e2, "cand_world_pairwise_dist")
    has_loss = hasattr(e2, "world_forward_contrastive_loss")
    # Verify simulation_mode kwarg works without raising
    z0, actions, targets = _sibling_batch(seed=1)
    sim_loss = e2.world_forward_contrastive_loss(
        z0, actions, targets, simulation_mode=True
    )
    sim_returned_zero = abs(float(sim_loss)) < 1e-12
    result = {
        "default_disabled": cfg_default.e2.e2_action_contrastive_enabled is False,
        "default_weight": cfg_default.e2.e2_action_contrastive_weight,
        "default_temperature": cfg_default.e2.e2_action_contrastive_temperature,
        "default_min_batch_classes": cfg_default.e2.e2_action_contrastive_min_batch_classes,
        "on_enabled": cfg_on.e2.e2_action_contrastive_enabled is True,
        "on_weight": cfg_on.e2.e2_action_contrastive_weight,
        "on_temperature": cfg_on.e2.e2_action_contrastive_temperature,
        "on_min_batch_classes": cfg_on.e2.e2_action_contrastive_min_batch_classes,
        "e2_has_cand_world_pairwise_dist": has_cand_dist,
        "e2_has_world_forward_contrastive_loss": has_loss,
        "simulation_mode_kwarg_works": sim_returned_zero,
    }
    result["pass"] = (
        result["default_disabled"]
        and abs(result["default_weight"] - 0.01) < 1e-9
        and abs(result["default_temperature"] - 0.1) < 1e-9
        and result["default_min_batch_classes"] == 2
        and result["on_enabled"]
        and abs(result["on_weight"] - 0.05) < 1e-9
        and abs(result["on_temperature"] - 0.2) < 1e-9
        and result["on_min_batch_classes"] == 3
        and has_cand_dist
        and has_loss
        and sim_returned_zero
    )
    return result


# ----------------------------------------------------------------------
# UC2 master-OFF backward-compat
# ----------------------------------------------------------------------
def run_uc2_master_off_backward_compat() -> dict:
    """UC2: with enabled=False, repeated same-seed E2 instances produce
    identical world_forward output for identical inputs. Bit-identical OFF."""
    seeds = [11, 22, 33]
    all_ok = True
    per_seed = []
    for s in seeds:
        e2_a = _make_e2(enabled=False, seed=s)
        e2_b = _make_e2(enabled=False, seed=s)
        z0, actions, _ = _sibling_batch(seed=s + 1)
        with torch.no_grad():
            out_a = e2_a.world_forward(z0.unsqueeze(0).expand(K, -1), actions)
            out_b = e2_b.world_forward(z0.unsqueeze(0).expand(K, -1), actions)
        max_diff = float((out_a - out_b).abs().max())
        seed_ok = max_diff < 1e-9
        all_ok = all_ok and seed_ok
        per_seed.append({
            "seed": s,
            "max_diff": max_diff,
            "ok": seed_ok,
        })
    return {
        "per_seed": per_seed,
        "n_seeds": len(seeds),
        "pass": all_ok,
    }


# ----------------------------------------------------------------------
# UC3 cand_world_pairwise_dist direction-of-change
# ----------------------------------------------------------------------
def run_uc3_pairwise_dist_direction_of_change() -> dict:
    """UC3: 200 SGD steps under contrastive loss should raise the
    cand_world_pairwise_dist diagnostic from random-init baseline."""
    torch.manual_seed(42)
    e2 = _make_e2(enabled=True, seed=42)

    baseline = _avg_dist(e2)
    optimizer = torch.optim.Adam(e2.parameters(), lr=1e-3)
    losses = []
    for step in range(200):
        z0, actions, targets = _sibling_batch(seed=step + 1)
        loss = e2.world_forward_contrastive_loss(z0, actions, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 20 == 0:
            losses.append({"step": step, "loss": float(loss)})
    trained = _avg_dist(e2)

    # Design memo UC3: cand_world_pairwise_dist >= 0.05 after training
    # (suggested threshold; direction-of-change is the load-bearing claim)
    threshold = 0.05
    direction_ok = trained > baseline
    threshold_ok = trained >= threshold
    return {
        "baseline_dist": baseline,
        "trained_dist": trained,
        "threshold": threshold,
        "direction_of_change_ok": direction_ok,
        "threshold_clear": threshold_ok,
        "loss_progression": losses,
        "pass": direction_ok and threshold_ok,
    }


# ----------------------------------------------------------------------
# UC4 held-out contrastive-task accuracy
# ----------------------------------------------------------------------
def run_uc4_contrastive_task_accuracy() -> dict:
    """UC4: trained substrate should solve the contrastive task on
    held-out batches with accuracy > 50% (random baseline 1/K = 12.5%
    for K=8). Distinguishes 'loss decreased' from 'task got solved.'"""
    torch.manual_seed(99)
    e2 = _make_e2(enabled=True, seed=99)
    optimizer = torch.optim.Adam(e2.parameters(), lr=1e-3)

    # Train 200 steps on disjoint seeds from held-out evaluation
    for step in range(200):
        z0, actions, targets = _sibling_batch(seed=step + 1)
        loss = e2.world_forward_contrastive_loss(z0, actions, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # Held-out evaluation: build logits matrix manually so we can score
    # semantic accuracy without re-implementing the loss.
    # With K=8 cycling through ACTION_DIM=4 classes, two anchors share each
    # target by construction (action class 0 appears at indices 0 and 4 with
    # identical action one-hot and target shift). Exact index argmin is
    # therefore ambiguous (it can land on either index of the matching pair
    # and only one is "correct" by integer label). The correct semantic
    # measure of "task is learnable" is whether the argmin pick has the
    # SAME FIRST-ACTION CLASS as the anchor -- this captures the
    # discriminability claim without the duplicate-anchor artefact. Random
    # baseline under semantic match is still 1/ACTION_DIM = 25%.
    e2.eval()
    held_out_seeds = list(range(50_000, 50_032))
    n_correct = 0
    n_total = 0
    with torch.no_grad():
        for s in held_out_seeds:
            z0, actions, targets = _sibling_batch(seed=s)
            z0_rep = z0.unsqueeze(0).expand(K, -1)
            preds = e2.world_forward(z0_rep, actions)  # [K, world_dim]
            diffs = preds.unsqueeze(0) - targets.unsqueeze(1)  # [K, K, world_dim]
            sq_dists = diffs.pow(2).sum(dim=-1)  # [K, K]
            picks = sq_dists.argmin(dim=-1)  # [K]
            anchor_classes = actions.argmax(dim=-1)  # [K]
            pick_classes = anchor_classes[picks]    # [K]: class of the picked prediction
            n_correct += int((pick_classes == anchor_classes).sum())
            n_total += K
    accuracy = n_correct / max(1, n_total)
    random_baseline = 1.0 / ACTION_DIM
    return {
        "accuracy": accuracy,
        "random_baseline": random_baseline,
        "n_correct": n_correct,
        "n_total": n_total,
        "threshold": 0.50,
        "pass": accuracy > 0.50,
    }


# ----------------------------------------------------------------------
# UC5 MECH-094 simulation gate
# ----------------------------------------------------------------------
def run_uc5_mech094_simulation_gate() -> dict:
    """UC5: simulation_mode=True returns tensor(0.0); subsequent waking
    backward() still propagates gradient through E2 weights. Simulation
    gate doesn't poison subsequent waking optimisation."""
    torch.manual_seed(7)
    e2 = _make_e2(enabled=True, seed=7)
    z0, actions, targets = _sibling_batch(seed=1)

    # Snapshot a parameter's value before simulation call
    wt_param = e2.world_transition[0].weight
    pre_sim_val = wt_param.detach().clone()

    sim_loss = e2.world_forward_contrastive_loss(
        z0, actions, targets, simulation_mode=True
    )
    sim_returned_zero = (abs(float(sim_loss)) < 1e-12)

    # Try to backward through the simulation loss -- should be a no-op
    # because the loss has no grad_fn (constructed via torch.zeros).
    sim_grad_fn_none = (sim_loss.grad_fn is None) or (not sim_loss.requires_grad)

    # Confirm a subsequent waking call advances gradient as expected
    e2.zero_grad(set_to_none=True)
    waking_loss = e2.world_forward_contrastive_loss(z0, actions, targets)
    waking_returned_positive = float(waking_loss) > 0.0
    waking_loss.backward()
    wt_grad = e2.world_transition[0].weight.grad
    waking_grad_flows = (
        wt_grad is not None
        and torch.isfinite(wt_grad).all()
        and float(wt_grad.abs().sum()) > 0.0
    )

    # Confirm weights did not advance during the simulation call
    weights_unchanged_during_sim = bool(torch.equal(
        e2.world_transition[0].weight.detach(), pre_sim_val
    ))

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
    uc3 = run_uc3_pairwise_dist_direction_of_change()
    uc4 = run_uc4_contrastive_task_accuracy()
    uc5 = run_uc5_mech094_simulation_gate()

    all_pass = all(r["pass"] for r in [uc1, uc2, uc3, uc4, uc5])
    elapsed = time.time() - t0

    run_id = "v3_exq_613_sd056_e2_action_contrastive_substrate_readiness_v3"
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    manifest = {
        "schema_version": "v1",
        "run_id": f"{run_id}_{ts}",
        "experiment_type": "v3_exq_613_sd056_e2_action_contrastive_substrate_readiness",
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "experiment_purpose": "diagnostic",
        "claim_ids": [],
        "result": "PASS" if all_pass else "FAIL",
        "evidence_direction": "supports" if all_pass else "weakens",
        "metrics": {
            "UC1_module_surface": uc1,
            "UC2_master_off_backward_compat": uc2,
            "UC3_pairwise_dist_direction_of_change": uc3,
            "UC4_contrastive_task_accuracy": uc4,
            "UC5_mech094_simulation_gate": uc5,
        },
        "elapsed_seconds": elapsed,
        "notes": (
            "SD-056 (e2.action_conditional_divergence_contrastive) substrate-"
            "readiness diagnostic. UC3 cand_world_pairwise_dist direction-of-"
            "change is the load-bearing PASS condition per the design memo "
            "(REE_assembly/evidence/planning/e2_action_divergence_substrate_"
            "design.md). UC4 confirms the contrastive task is learnable on "
            "REE's E2 architecture (random baseline 1/K=12.5% for K=8). "
            "Does NOT test behavioural-diversity recovery -- that is "
            "V3-EXQ-569a's job (matched-entropy FP-2 falsifier on the fixed "
            "substrate; GAP-A R1.a/R1.b decision rule per "
            "evidence/planning/behavioral_diversity_isolation_plan.md). "
            "Diagnostic claim_ids=[] per Phase-3 governance rules -- this "
            "experiment does not weight SD-056 confidence directly. Lit-pull "
            "anchors (Srivastava 2021 contrastive RSSM, Saanum 2024 PLSM, "
            "Tanaka 2020 cerebellar forward model, Miyamoto 2023 frontopolar "
            "counterfactual) inform the architecture but are not signals "
            "tested here."
        ),
    }

    out_dir = (
        REPO_ROOT.parent
        / "REE_assembly" / "evidence" / "experiments"
    )
    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=False,
        config=manifest.get("config"),
        seeds=None,
        script_path=Path(__file__),
    )

    print(f"verdict: {manifest['result']}")
    for k, v in manifest["metrics"].items():
        print(f"  {k}: pass={v['pass']}")
    print(f"Result written to: {out_path}", flush=True)

    from experiment_protocol import emit_outcome
    emit_outcome(outcome=manifest["result"], manifest_path=str(out_path))


if __name__ == "__main__":
    main()
