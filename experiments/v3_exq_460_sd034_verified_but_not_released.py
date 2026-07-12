"""V3-EXQ-460 (EXP-0156): SD-034 verified-but-not-released landing.

Purpose: diagnostic. Validates the SD-034 closure-operator five-part
signal (beta release, targeted No-Go via MECH-260, rule-domain residue
discharge, closure event to SD-032a, dACC pe cap/reset) and confirms
the ocd4 "verified-but-not-released" dissociation: with closure OFF,
beta stays latched and No-Go is not installed after completion; with
closure ON, release + No-Go fire.

This is the substrate landing test. Behavioural variants (full task
loop with E3 scoring) are deferred -- they depend on phased training
of rule_state + a task variant where "completion" has an observable
signature in score distributions. This landing validates the wiring
so those variants can be authored.

Six sub-tests (all arithmetic + API):

  UC1 agent-init backward compat: use_closure_operator=False -> agent
      has no closure_operator; existing experiments unaffected.

  UC2 agent-init wiring: use_closure_operator=True -> agent has
      closure_operator; salience.config.affinity_weights contains
      closure_event with internal_planning affinity.

  UC3 beta release on fire: with beta elevated and closure enabled,
      emit_closure(bypass_mode_conditioning=True) releases the latch.

  UC4 targeted No-Go installed: after fire, dacc._action_history
      contains nogo_injection_count copies of the completed action
      class (MECH-260 channel).

  UC5 dACC pe reset: _pe_ema is None after fire when reset_pe_ema=True.

  UC6 mode conditioning blocks fire: with current_mode not in
      allowed_closure_modes, closure does NOT fire (verified-but-
      not-released ARM A dissociation).

Run with:
  /opt/local/bin/python3 experiments/v3_exq_460_sd034_verified_but_not_released.py

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
from ree_core.utils.config import REEConfig
from experiments.pack_writer import write_flat_manifest  # noqa: E402


def _build_agent(use_closure: bool) -> REEAgent:
    cfg = REEConfig.from_dims(
        body_obs_dim=12, world_obs_dim=250, action_dim=4,
        use_dacc=True,
        use_salience_coordinator=True,
        use_lateral_pfc_analog=True,
        use_closure_operator=use_closure,
    )
    return REEAgent(cfg)


def run_uc1_backward_compat() -> dict:
    agent = _build_agent(use_closure=False)
    result = {
        "closure_operator_is_none": agent.closure_operator is None,
    }
    result["pass"] = result["closure_operator_is_none"]
    return result


def run_uc2_wiring() -> dict:
    agent = _build_agent(use_closure=True)
    aff = agent.salience.config.affinity_weights
    result = {
        "closure_operator_not_none": agent.closure_operator is not None,
        "closure_event_registered": "closure_event" in aff,
        "internal_planning_weight": aff.get("closure_event", {}).get(
            "internal_planning", None
        ),
    }
    result["pass"] = (
        result["closure_operator_not_none"]
        and result["closure_event_registered"]
        and result["internal_planning_weight"] is not None
        and result["internal_planning_weight"] > 0.0
    )
    return result


def run_uc3_beta_release_on_fire() -> dict:
    agent = _build_agent(use_closure=True)
    # Force beta elevated.
    agent.beta_gate.elevate()
    beta_pre = bool(agent.beta_gate.is_elevated)

    z_world = torch.zeros(1, agent.config.latent.world_dim)
    event = agent.closure_operator.emit_closure(
        action_class=2,
        z_world=z_world,
        bypass_mode_conditioning=True,
    )
    beta_post = bool(agent.beta_gate.is_elevated)
    result = {
        "beta_pre": beta_pre,
        "beta_post": beta_post,
        "event_fired": event.fired,
        "event_reason": event.reason,
    }
    result["pass"] = beta_pre and (not beta_post) and event.fired
    return result


def run_uc4_nogo_installed() -> dict:
    agent = _build_agent(use_closure=True)
    agent.beta_gate.elevate()
    pre_history = list(agent.dacc._action_history)
    expected_class = 1

    z_world = torch.zeros(1, agent.config.latent.world_dim)
    event = agent.closure_operator.emit_closure(
        action_class=expected_class,
        z_world=z_world,
        bypass_mode_conditioning=True,
    )
    post_history = list(agent.dacc._action_history)
    n_new = len(post_history) - len(pre_history)
    occurrences = post_history.count(expected_class) - pre_history.count(expected_class)

    result = {
        "fired": event.fired,
        "pre_history_len": len(pre_history),
        "post_history_len": len(post_history),
        "n_new_entries": n_new,
        "new_occurrences_of_target": occurrences,
        "expected_injection_count": agent.config.closure_nogo_injection_count,
    }
    result["pass"] = (
        event.fired
        and occurrences >= agent.config.closure_nogo_injection_count
    )
    return result


def run_uc5_pe_reset() -> dict:
    agent = _build_agent(use_closure=True)
    agent.beta_gate.elevate()
    # Seed _pe_ema to a non-None float so reset is observable.
    agent.dacc._pe_ema = 0.42
    z_world = torch.zeros(1, agent.config.latent.world_dim)
    event = agent.closure_operator.emit_closure(
        action_class=0,
        z_world=z_world,
        bypass_mode_conditioning=True,
    )
    result = {
        "fired": event.fired,
        "pe_ema_post": agent.dacc._pe_ema,
    }
    result["pass"] = event.fired and (agent.dacc._pe_ema is None)
    return result


def run_uc6_mode_conditioning_blocks_fire() -> dict:
    agent = _build_agent(use_closure=True)
    # Force beta elevated so the only blocker is mode conditioning.
    agent.beta_gate.elevate()
    z_world = torch.zeros(1, agent.config.latent.world_dim)

    # Mode not in allowed_closure_modes (default: external_task,
    # internal_planning). Pass a disallowed mode and do NOT bypass.
    event = agent.closure_operator.emit_closure(
        action_class=3,
        z_world=z_world,
        current_mode="offline_consolidation",
        sd033a_gate=1.0,
        bypass_mode_conditioning=False,
    )
    result = {
        "fired": event.fired,
        "reason": event.reason,
        "beta_still_elevated": bool(agent.beta_gate.is_elevated),
    }
    # Predicted: NOT fired; reason contains 'mode_disallowed'; beta stays up.
    result["pass"] = (
        (not event.fired)
        and "mode_disallowed" in (event.reason or "")
        and result["beta_still_elevated"]
    )
    return result


def main() -> None:
    t0 = time.time()
    uc1 = run_uc1_backward_compat()
    uc2 = run_uc2_wiring()
    uc3 = run_uc3_beta_release_on_fire()
    uc4 = run_uc4_nogo_installed()
    uc5 = run_uc5_pe_reset()
    uc6 = run_uc6_mode_conditioning_blocks_fire()

    subtests = {
        "UC1_backward_compat": uc1,
        "UC2_wiring": uc2,
        "UC3_beta_release_on_fire": uc3,
        "UC4_nogo_installed": uc4,
        "UC5_pe_reset": uc5,
        "UC6_mode_conditioning_blocks_fire": uc6,
    }
    all_pass = all(r["pass"] for r in subtests.values())
    elapsed = time.time() - t0

    run_id = "v3_exq_460_sd034_verified_but_not_released_v3"
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    manifest = {
        "schema_version": "v1",
        "run_id": f"{run_id}_{ts}",
        "experiment_type": "v3_exq_460_sd034_verified_but_not_released",
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "experiment_purpose": "diagnostic",
        "claim_ids": ["SD-034", "MECH-260", "MECH-261"],
        "result": "PASS" if all_pass else "FAIL",
        "evidence_direction": "supports" if all_pass else "weakens",
        "evidence_direction_per_claim": {
            "SD-034": "supports" if all_pass else "weakens",
            "MECH-260": "supports" if all_pass else "weakens",
            "MECH-261": "supports" if all_pass else "weakens",
        },
        "metrics": subtests,
        "elapsed_seconds": elapsed,
        "notes": (
            "SD-034 landing diagnostic (ocd4 verified-but-not-released row). "
            "Tests the five-part signal: beta release, targeted No-Go via "
            "MECH-260, rule-domain residue discharge (indirectly via emit "
            "path), closure event to SD-032a (UC2 affinity registration), "
            "dACC pe reset (UC5). Mode conditioning is confirmed (UC6) as "
            "the falsifiability guard: if MECH-090 + MECH-260 + MECH-094 "
            "tuning WITHOUT closure produces the signature in a follow-up "
            "behavioral variant, SD-034 is over-specification. Anchor plan: "
            "REE_assembly/evidence/planning/sd033_governance_plan.md. "
            "Source: docs/thoughts/2026-04-20_ocd4.md row 'verified-but-"
            "not-released'."
        ),
    }

    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=False,
        config=manifest.get("config"),
        seeds=None,
        script_path=Path(__file__),
    )

    print(f"result: {manifest['result']}")
    for k, v in manifest["metrics"].items():
        print(f"  {k}: pass={v['pass']}")
    print(f"Result written to: {out_path}", flush=True)


if __name__ == "__main__":
    main()
