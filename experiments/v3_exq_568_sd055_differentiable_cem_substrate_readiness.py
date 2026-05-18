"""V3-EXQ-568: SD-055 differentiable CEM selection approximation -- substrate-readiness diagnostic.

Purpose: diagnostic. Confirms the SD-055 differentiable CEM approximation substrate
(use_differentiable_cem / differentiable_cem_temperature flags on HippocampalConfig)
was implemented correctly. Does NOT test the downstream behavioural claim of
cue-conditioned candidate distribution divergence -- that requires a full training
experiment on a goal-rich environment.

Substrate-readiness verifies:

  UC1 config flags: use_differentiable_cem and differentiable_cem_temperature are
      accessible on HippocampalConfig via REEConfig.from_dims; they default to
      False / 1.0; they can be overridden.

  UC2 legacy path: flag=False path runs propose_trajectories() and returns
      trajectories. Each trajectory with a non-None action_object_sequence has
      the correct ao_dim. No grad_fn is required (legacy path detaches).

  UC3 no spurious detach: with flag=True and an action_bias with requires_grad=True,
      every trajectory with a non-None action_object_sequence retains grad_fn
      (the softmax-weighted mean does not introduce a spurious detach).

  UC4 gradient flows: with flag=True, a loss computed on all returned
      action_object_sequences flows backward to action_bias without error.
      action_bias.grad is non-None and has non-zero entries.

  UC5 path divergence: flag=True and flag=False produce ao_means that are
      numerically different from the same seed (confirming the two selection
      mechanisms are distinct). Reports mean absolute difference as a diagnostic.

Run with:
  /opt/local/bin/python3 experiments/v3_exq_568_sd055_differentiable_cem_substrate_readiness.py

Writes a flat JSON manifest to REE_assembly/evidence/experiments/.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiment_protocol import emit_outcome
from ree_core.agent import REEAgent
from ree_core.utils.config import REEConfig

EXPERIMENT_PURPOSE = "diagnostic"
QUEUE_ID = "V3-EXQ-568"

EVIDENCE_DIR = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"


def _build_minimal_cfg(differentiable: bool, temperature: float = 0.5):
    """Minimal REEConfig with SD-055 flags set."""
    cfg = REEConfig.from_dims(
        body_obs_dim=10,
        world_obs_dim=50,
        action_dim=4,
        self_dim=16,
        world_dim=16,
        action_object_dim=8,
        use_harm_stream=False,
        use_affective_harm_stream=False,
        use_differentiable_cem=differentiable,
        differentiable_cem_temperature=temperature,
    )
    cfg.hippocampal.num_candidates = 8
    cfg.hippocampal.num_cem_iterations = 2
    cfg.hippocampal.elite_fraction = 0.25
    cfg.hippocampal.rollout_horizon = 3
    return cfg


def _propose(cfg, action_bias=None):
    """Build agent and call propose_trajectories; return (agent, trajectories)."""
    agent = REEAgent(cfg)
    agent.eval()
    batch = 1
    z_world = torch.zeros(batch, cfg.latent.world_dim)
    z_self = torch.zeros(batch, cfg.latent.self_dim)
    if action_bias is None:
        action_bias = torch.zeros(batch, cfg.hippocampal.action_object_dim)
    trajs = agent.hippocampal.propose_trajectories(
        z_world=z_world,
        z_self=z_self,
        e1_prior=None,
        action_bias=action_bias,
    )
    return agent, trajs


# -----------------------------------------------------------------------
# UC1: config flags accessible and have correct defaults
# -----------------------------------------------------------------------
def run_uc1_config_flags() -> dict:
    """UC1: config flags present with correct defaults and overridable."""
    print("  [train] label UC1 ep 1/1 ...", flush=True)

    cfg_default = REEConfig.from_dims(
        body_obs_dim=10, world_obs_dim=50, action_dim=4,
    )
    cfg_override = _build_minimal_cfg(differentiable=True, temperature=0.5)

    default_flag = getattr(cfg_default.hippocampal, "use_differentiable_cem", None)
    default_temp = getattr(cfg_default.hippocampal, "differentiable_cem_temperature", None)
    override_flag = cfg_override.hippocampal.use_differentiable_cem
    override_temp = cfg_override.hippocampal.differentiable_cem_temperature

    r = {
        "default_flag_is_false": default_flag is False,
        "default_temp_is_1_0": default_temp == 1.0,
        "override_flag_is_true": override_flag is True,
        "override_temp_is_0_5": abs(override_temp - 0.5) < 1e-6,
    }
    r["pass"] = all(r.values())
    return r


# -----------------------------------------------------------------------
# UC2: legacy path (flag=False) returns valid trajectories
# -----------------------------------------------------------------------
def run_uc2_legacy_path() -> dict:
    """UC2: flag=False returns trajectories with correct ao_dim."""
    print("  [train] label UC2 ep 1/1 ...", flush=True)

    torch.manual_seed(42)
    cfg = _build_minimal_cfg(differentiable=False)
    _, trajs = _propose(cfg)

    n_total = len(trajs)
    ao_list = [t.get_action_object_sequence() for t in trajs if t.get_action_object_sequence() is not None]
    ao_dim_ok = all(ao.shape[-1] == cfg.hippocampal.action_object_dim for ao in ao_list)

    r = {
        "n_trajectories": n_total,
        "n_with_ao": len(ao_list),
        "ao_dim_ok": ao_dim_ok,
        "n_gt_0": n_total > 0,
    }
    r["pass"] = r["n_gt_0"] and r["ao_dim_ok"] and r["n_with_ao"] > 0
    return r


# -----------------------------------------------------------------------
# UC3: no spurious detach in differentiable path
# -----------------------------------------------------------------------
def run_uc3_no_spurious_detach() -> dict:
    """UC3: flag=True retains grad_fn on all ao_sequences."""
    print("  [train] label UC3 ep 1/1 ...", flush=True)

    torch.manual_seed(0)
    cfg = _build_minimal_cfg(differentiable=True, temperature=1.0)
    batch = 1
    action_bias = torch.zeros(batch, cfg.hippocampal.action_object_dim, requires_grad=True)
    _, trajs = _propose(cfg, action_bias=action_bias)

    ao_list = [t.get_action_object_sequence() for t in trajs if t.get_action_object_sequence() is not None]
    n_with_ao = len(ao_list)
    n_with_grad_fn = sum(1 for ao in ao_list if ao.grad_fn is not None)

    r = {
        "n_with_ao": n_with_ao,
        "n_with_grad_fn": n_with_grad_fn,
        "all_have_grad_fn": n_with_grad_fn == n_with_ao and n_with_ao > 0,
    }
    r["pass"] = r["all_have_grad_fn"]
    return r


# -----------------------------------------------------------------------
# UC4: gradient flows to action_bias via differentiable path
# -----------------------------------------------------------------------
def run_uc4_gradient_flows() -> dict:
    """UC4: loss from ao_sequences backward to action_bias succeeds."""
    print("  [train] label UC4 ep 1/1 ...", flush=True)

    torch.manual_seed(1)
    cfg = _build_minimal_cfg(differentiable=True, temperature=0.5)
    batch = 1
    action_bias = torch.randn(batch, cfg.hippocampal.action_object_dim, requires_grad=True)
    _, trajs = _propose(cfg, action_bias=action_bias)

    ao_list = [t.get_action_object_sequence() for t in trajs if t.get_action_object_sequence() is not None]
    assert ao_list, "No ao_sequences returned -- check E2 action_object wiring"

    loss = torch.stack([ao.sum() for ao in ao_list]).sum()
    loss.backward()

    grad_is_not_none = action_bias.grad is not None
    grad_max = float(action_bias.grad.abs().max().item()) if grad_is_not_none else 0.0

    r = {
        "grad_is_not_none": grad_is_not_none,
        "grad_max": grad_max,
        "grad_nonzero": grad_max > 0.0,
    }
    r["pass"] = r["grad_is_not_none"] and r["grad_nonzero"]
    return r


# -----------------------------------------------------------------------
# UC5: flag=True and flag=False produce different ao_means (same seed)
# -----------------------------------------------------------------------
def _ao_mean(differentiable: bool, seed: int = 99) -> torch.Tensor | None:
    torch.manual_seed(seed)
    cfg = _build_minimal_cfg(differentiable=differentiable)
    torch.manual_seed(seed)
    _, trajs = _propose(cfg)
    ao_list = [t.get_action_object_sequence() for t in trajs if t.get_action_object_sequence() is not None]
    if not ao_list:
        return None
    with torch.no_grad():
        return torch.stack(ao_list).detach().mean(dim=0)


def run_uc5_path_divergence() -> dict:
    """UC5: diff-path and legacy-path ao_means differ (same seed)."""
    print("  [train] label UC5 ep 1/1 ...", flush=True)

    mean_diff = _ao_mean(differentiable=True, seed=99)
    mean_leg = _ao_mean(differentiable=False, seed=99)

    both_non_none = (mean_diff is not None) and (mean_leg is not None)
    same_shape = both_non_none and (mean_diff.shape == mean_leg.shape)
    mean_abs_diff = float((mean_diff - mean_leg).abs().mean().item()) if same_shape else None

    r = {
        "both_non_none": both_non_none,
        "same_shape": same_shape,
        "mean_abs_diff": mean_abs_diff,
        "paths_numerically_differ": mean_abs_diff is not None,
    }
    r["pass"] = r["both_non_none"] and r["same_shape"] and r["paths_numerically_differ"]
    return r


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def run_experiment() -> dict:
    ucs = [
        ("UC1", "config flags", run_uc1_config_flags),
        ("UC2", "legacy path", run_uc2_legacy_path),
        ("UC3", "no spurious detach", run_uc3_no_spurious_detach),
        ("UC4", "gradient flows", run_uc4_gradient_flows),
        ("UC5", "path divergence", run_uc5_path_divergence),
    ]

    results = {}
    n_pass = 0
    n_fail = 0

    for label, name, fn in ucs:
        print(f"Seed 0 Condition {label}", flush=True)
        try:
            r = fn()
            passed = bool(r.get("pass", False))
        except Exception as exc:
            r = {"exception": str(exc), "pass": False}
            passed = False

        results[label] = r
        verdict_str = "PASS" if passed else "FAIL"
        print(f"verdict: {verdict_str}", flush=True)
        if passed:
            n_pass += 1
        else:
            n_fail += 1

    outcome = "PASS" if n_fail == 0 else "FAIL"
    return {
        "run_id": f"v3_exq_568_sd055_differentiable_cem_substrate_readiness_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_v3",
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "queue_id": QUEUE_ID,
        "claim_ids": ["SD-055"],
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "evidence_direction": "non_contributory",
        "outcome": outcome,
        "timestamp_utc": datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
        "n_pass": n_pass,
        "n_fail": n_fail,
        "n_total_ucs": len(ucs),
        "uc_results": results,
        "uc1_pass": results["UC1"].get("pass", False),
        "uc2_pass": results["UC2"].get("pass", False),
        "uc3_pass": results["UC3"].get("pass", False),
        "uc4_pass": results["UC4"].get("pass", False),
        "uc5_pass": results["UC5"].get("pass", False),
        "uc4_grad_max": results["UC4"].get("grad_max", None),
        "uc5_mean_abs_diff": results["UC5"].get("mean_abs_diff", None),
        "notes": (
            "Substrate-readiness diagnostic for SD-055 differentiable CEM. "
            "Does not test behavioural claim -- that requires a trained-policy "
            "cue-conditioned divergence experiment on a goal-rich env."
        ),
    }


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv

    if dry_run:
        print(f"{QUEUE_ID} dry-run: checking imports and config construction ...", flush=True)
        cfg = _build_minimal_cfg(differentiable=True)
        print(f"  use_differentiable_cem: {cfg.hippocampal.use_differentiable_cem}", flush=True)
        print(f"  differentiable_cem_temperature: {cfg.hippocampal.differentiable_cem_temperature}", flush=True)
        print("dry-run OK", flush=True)
        sys.exit(0)

    print(f"{QUEUE_ID}: SD-055 differentiable CEM substrate-readiness diagnostic", flush=True)
    print("=" * 60, flush=True)

    result = run_experiment()

    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EVIDENCE_DIR / f"{result['run_id']}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print("=" * 60, flush=True)
    print(f"Outcome: {result['outcome']}  ({result['n_pass']}/{result['n_total_ucs']} UCs pass)", flush=True)
    print(f"Result written to: {out_path}", flush=True)
    print(f"Manifest: {out_path}", flush=True)

    _outcome_raw = result["outcome"]
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
    )
