"""V3-EXQ-466 (EXP-0162): SD-034 satisficing / residue-discharge landing.

Purpose: diagnostic. Validates that the SD-034 closure-operator's
rule-domain residue discharge (ResidueField.discharge_domain) attenuates
ONLY centers near the closure location (bounded, domain-scoped) AND
preserves the "residue cannot be erased" invariant (multiplicative
decay with hard floor, never literal zero).

This is the substrate landing test for the ocd4 satisficing row --
the ResidueField.discharge_domain() API is the substrate extension
listed in EXP-0162 prerequisites. Behavioural variants (tolerance-band
completion task; time-to-closure distributions; bimodal vs smooth
satisficing comparison) are deferred -- they require a new env variant
not yet on any roadmap. This landing validates the domain-scoped
discharge mechanism so those variants can be authored later.

Five sub-tests (all arithmetic + API):

  UC1 discharge_domain attenuates near-by centers: add residue at
      z=a, call discharge_domain(z=a), weight at a drops by factor.

  UC2 discharge_domain spares far centers: add residue at z=a and
      z=b (b far from a); discharge_domain(z=a) leaves b weight intact.

  UC3 discharge preserves invariant: multiplicative decay never erases
      (weight > 0 after many discharges).

  UC4 closure fire triggers domain discharge: closure fire with
      z_world near added residue attenuates the matching center's
      weight (integration with ClosureOperator._fire five-part signal).

  UC5 closure fire at distant z_world leaves residue untouched
      (domain scoping works end-to-end through ClosureOperator).

Run with:
  /opt/local/bin/python3 experiments/v3_exq_466_sd034_satisficing_residue_discharge.py

Writes a flat JSON manifest to REE_assembly/evidence/experiments/.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ree_core.agent import REEAgent
from ree_core.residue.field import ResidueField
from ree_core.utils.config import REEConfig, ResidueConfig


def _build_residue(world_dim: int = 32, bandwidth: float = 1.0) -> ResidueField:
    cfg = ResidueConfig()
    cfg.world_dim = world_dim
    cfg.bandwidth = bandwidth
    return ResidueField(cfg)


def run_uc1_nearby_attenuation() -> dict:
    torch.manual_seed(0)
    rf = _build_residue(world_dim=32, bandwidth=1.0)
    z_a = torch.zeros(1, 32)
    idx_a = rf.rbf_field.add_residue(z_a, intensity=1.0)
    w_before = float(rf.rbf_field.weights[idx_a].item())

    n_attenuated = rf.discharge_domain(z_a, factor=0.5, radius=1.5)
    w_after = float(rf.rbf_field.weights[idx_a].item())

    result = {
        "w_before": w_before,
        "w_after": w_after,
        "n_attenuated": n_attenuated,
        "ratio": w_after / (w_before + 1e-12),
    }
    result["pass"] = (
        w_before > 0.0
        and n_attenuated >= 1
        and 0.4 < result["ratio"] < 0.6
    )
    return result


def run_uc2_far_centers_spared() -> dict:
    torch.manual_seed(0)
    rf = _build_residue(world_dim=32, bandwidth=1.0)
    z_a = torch.zeros(1, 32)
    z_b = torch.zeros(1, 32)
    z_b[0, 0] = 10.0  # far from z_a relative to bandwidth=1.0 / radius=1.5
    idx_a = rf.rbf_field.add_residue(z_a, intensity=1.0)
    idx_b = rf.rbf_field.add_residue(z_b, intensity=1.0)
    wa_before = float(rf.rbf_field.weights[idx_a].item())
    wb_before = float(rf.rbf_field.weights[idx_b].item())

    rf.discharge_domain(z_a, factor=0.5, radius=1.5)
    wa_after = float(rf.rbf_field.weights[idx_a].item())
    wb_after = float(rf.rbf_field.weights[idx_b].item())

    result = {
        "wa_before": wa_before, "wa_after": wa_after,
        "wb_before": wb_before, "wb_after": wb_after,
        "b_preserved": abs(wb_after - wb_before) < 1e-6,
        "a_attenuated": wa_after < wa_before * 0.9,
    }
    result["pass"] = result["a_attenuated"] and result["b_preserved"]
    return result


def run_uc3_invariant_preserved() -> dict:
    """Repeated discharge -> weight trends to floor but never 0."""
    torch.manual_seed(0)
    rf = _build_residue(world_dim=32, bandwidth=1.0)
    z_a = torch.zeros(1, 32)
    idx_a = rf.rbf_field.add_residue(z_a, intensity=1.0)
    w_initial = float(rf.rbf_field.weights[idx_a].item())

    for _ in range(200):  # many halvings
        rf.discharge_domain(z_a, factor=0.5, radius=1.5)
    w_final = float(rf.rbf_field.weights[idx_a].item())

    result = {
        "w_initial": w_initial,
        "w_final": w_final,
        "w_final_nonzero": w_final > 0.0,
        "w_final_small": w_final < w_initial * 1e-3,
    }
    result["pass"] = result["w_final_nonzero"] and result["w_final_small"]
    return result


def run_uc4_closure_fire_triggers_discharge() -> dict:
    """End-to-end: ClosureOperator._fire attenuates matching residue."""
    torch.manual_seed(0)
    cfg = REEConfig.from_dims(
        body_obs_dim=12, world_obs_dim=250, action_dim=4,
        use_dacc=True,
        use_salience_coordinator=True,
        use_lateral_pfc_analog=True,
        use_closure_operator=True,
        closure_residue_discharge_factor=0.5,
        closure_residue_discharge_radius=1.5,
    )
    agent = REEAgent(cfg)
    world_dim = cfg.latent.world_dim
    z_a = torch.zeros(1, world_dim)
    idx_a = agent.residue_field.rbf_field.add_residue(z_a, intensity=1.0)
    w_before = float(agent.residue_field.rbf_field.weights[idx_a].item())

    agent.beta_gate.elevate()
    event = agent.closure_operator.emit_closure(
        action_class=0,
        z_world=z_a,
        bypass_mode_conditioning=True,
    )
    w_after = float(agent.residue_field.rbf_field.weights[idx_a].item())

    result = {
        "fired": event.fired,
        "w_before": w_before,
        "w_after": w_after,
        "ratio": w_after / (w_before + 1e-12),
        "residue_centers_discharged": event.residue_centers_discharged,
    }
    result["pass"] = (
        event.fired
        and 0.4 < result["ratio"] < 0.6
        and event.residue_centers_discharged >= 1
    )
    return result


def run_uc5_closure_fire_distant_spares_residue() -> dict:
    """Closure at z != residue location leaves residue untouched."""
    torch.manual_seed(0)
    cfg = REEConfig.from_dims(
        body_obs_dim=12, world_obs_dim=250, action_dim=4,
        use_dacc=True,
        use_salience_coordinator=True,
        use_lateral_pfc_analog=True,
        use_closure_operator=True,
        closure_residue_discharge_factor=0.5,
        closure_residue_discharge_radius=1.5,
    )
    agent = REEAgent(cfg)
    world_dim = cfg.latent.world_dim
    z_residue = torch.zeros(1, world_dim)
    z_residue[0, 0] = 0.0
    z_far = torch.zeros(1, world_dim)
    z_far[0, 0] = 20.0  # far from residue location
    idx_r = agent.residue_field.rbf_field.add_residue(z_residue, intensity=1.0)
    w_before = float(agent.residue_field.rbf_field.weights[idx_r].item())

    agent.beta_gate.elevate()
    event = agent.closure_operator.emit_closure(
        action_class=0,
        z_world=z_far,
        bypass_mode_conditioning=True,
    )
    w_after = float(agent.residue_field.rbf_field.weights[idx_r].item())

    result = {
        "fired": event.fired,
        "w_before": w_before,
        "w_after": w_after,
        "delta": abs(w_after - w_before),
        "residue_centers_discharged": event.residue_centers_discharged,
    }
    result["pass"] = (
        event.fired
        and result["delta"] < 1e-6
        # n_attenuated should be zero for distant-z case (domain-scoped)
    )
    return result


def main() -> None:
    t0 = time.time()
    uc1 = run_uc1_nearby_attenuation()
    uc2 = run_uc2_far_centers_spared()
    uc3 = run_uc3_invariant_preserved()
    uc4 = run_uc4_closure_fire_triggers_discharge()
    uc5 = run_uc5_closure_fire_distant_spares_residue()

    subtests = {
        "UC1_nearby_attenuation": uc1,
        "UC2_far_centers_spared": uc2,
        "UC3_invariant_preserved": uc3,
        "UC4_closure_fire_triggers_discharge": uc4,
        "UC5_closure_fire_distant_spares_residue": uc5,
    }
    all_pass = all(r["pass"] for r in subtests.values())
    elapsed = time.time() - t0

    run_id = "v3_exq_466_sd034_satisficing_residue_discharge_v3"
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    manifest = {
        "schema_version": "v1",
        "run_id": f"{run_id}_{ts}",
        "experiment_type": "v3_exq_466_sd034_satisficing_residue_discharge",
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "experiment_purpose": "diagnostic",
        "claim_ids": ["SD-034", "MECH-094"],
        "result": "PASS" if all_pass else "FAIL",
        "evidence_direction": "supports" if all_pass else "weakens",
        "evidence_direction_per_claim": {
            "SD-034": "supports" if all_pass else "weakens",
            "MECH-094": "supports" if all_pass else "weakens",
        },
        "metrics": subtests,
        "elapsed_seconds": elapsed,
        "notes": (
            "SD-034 satisficing landing diagnostic (ocd4 satisficing row). "
            "Tests ResidueField.discharge_domain() API: rule-domain "
            "multiplicative attenuation (UC1) that respects domain scoping "
            "(UC2) and preserves the 'residue cannot be erased' invariant "
            "via multiplicative-decay floor (UC3). End-to-end integration "
            "through ClosureOperator._fire confirmed (UC4/UC5). Behavioral "
            "variant (tolerance-band completion task -- bimodal vs smooth "
            "satisficing) is deferred to a future experiment that requires "
            "a new env variant. Anchor plan: REE_assembly/evidence/planning/"
            "sd033_governance_plan.md. Source: docs/thoughts/"
            "2026-04-20_ocd4.md row 'satisficing / No-Go thresholding'."
        ),
    }

    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{manifest['run_id']}.json"
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"result: {manifest['result']}")
    for k, v in manifest["metrics"].items():
        print(f"  {k}: pass={v['pass']}")
    print(f"Result written to: {out_path}", flush=True)


if __name__ == "__main__":
    main()
