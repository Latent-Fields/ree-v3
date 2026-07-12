#!/opt/local/bin/python3
"""V3-EXQ-461 (EXP-0157): MECH-090 + SD-033a delayed-reward persistence.

Purpose: diagnostic. Closes commitment_closure_plan.md GAP-2 by authoring the
substrate-readiness version of the ocd4 "delayed-reward persistence" row.

This is NOT the full behavioural delay-to-reward task. The full task requires
the Phase 3 CausalGridWorldV2 extension where reward is benefit-eligible only
after an N-step delay, with distractors present. This script validates the
substrate contract that the behavioural task will need:

  - MECH-090 BetaGate can hold policy propagation across a delay window.
  - A weakened / legacy no-Hold contrast remains distinguishable.
  - SD-033a rule_state persists under the MECH-261 low-write replay gate and
    is overwritten under full distractor writes.
  - A strengthened Hold threshold resists premature completion signals.
  - SD-034 terminal closure can release the held commitment and install No-Go.

Sub-tests:

  UC1 baseline_hold_window:
      Beta elevated for 12 synthetic delay ticks. propagate() returns None on
      every tick, hold_count increments, low completion does not release, high
      terminal completion releases and allows propagation.

  UC2 weakened_hold_passthrough:
      Legacy/no-bistable arm: beta is never elevated, so policy propagation
      passes through on every tick. This is the under-binding contrast that a
      behavioural successor should separate from baseline Hold.

  UC3 sd033a_replay_gate_preserves_rule_state:
      Install a rule_state with gate=1.0, then apply a distractor source for
      40 ticks. Under replay-like gate=0.05, cosine similarity to the installed
      rule remains high. Under full gate=1.0, the same distractor overwrites
      the rule_state. This is the SD-033a / MECH-261 delay persistence kernel.

  UC4 strengthened_hold_resists_premature_release:
      High release threshold arm: moderate completion does not release, near-
      certain completion does. This is the over-binding contrast.

  UC5 mech261_mode_gate_table:
      Directly verifies the sd_033a MECH-261 gate values used by UC3:
      external_task=1.0, internal_replay=0.05, offline_consolidation=0.3.

  UC6 terminal_closure_signal:
      With SD-034 enabled, explicit terminal closure releases MECH-090 beta,
      injects targeted MECH-260 No-Go, writes the closure_event signal, and
      resets dACC PE. This is the only SD-034 dependency in EXP-0157.

Run:
  /opt/local/bin/python3 experiments/v3_exq_461_mech090_sd033a_delayed_reward_persistence.py

Smoke:
  /opt/local/bin/python3 experiments/v3_exq_461_mech090_sd033a_delayed_reward_persistence.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiment_protocol import emit_outcome  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.cingulate.salience_coordinator import SalienceCoordinator  # noqa: E402
from ree_core.heartbeat.beta_gate import BetaGate  # noqa: E402
from ree_core.pfc.lateral_pfc_analog import (  # noqa: E402
    LateralPFCAnalog,
    LateralPFCConfig,
)
from ree_core.utils.config import REEConfig  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402


EXPERIMENT_TYPE = "v3_exq_461_mech090_sd033a_delayed_reward_persistence"
QUEUE_ID = "V3-EXQ-461"
CLAIM_IDS = ["MECH-090", "SD-033a", "SD-034"]
EXPERIMENT_PURPOSE = "diagnostic"
DELAY_TICKS = 12
RULE_INSTALL_TICKS = 30
DISTRACTOR_TICKS = 40


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_iso_now() -> str:
    return _utc_now().strftime("%Y-%m-%dT%H:%M:%SZ")


def _policy_state(value: float) -> torch.Tensor:
    return torch.full((1, 4), float(value), dtype=torch.float32)


def run_uc1_baseline_hold_window() -> dict:
    gate = BetaGate(initial_beta_elevated=False, completion_release_threshold=0.75)
    gate.elevate()
    none_count = 0
    for tick in range(DELAY_TICKS):
        out = gate.propagate(_policy_state(tick))
        if out is None:
            none_count += 1

    low_release = gate.receive_hippocampal_completion(0.50)
    beta_after_low = bool(gate.is_elevated)
    high_release = gate.receive_hippocampal_completion(0.90)
    beta_after_high = bool(gate.is_elevated)
    post = gate.propagate(_policy_state(99.0))
    state = gate.get_state()

    result = {
        "delay_ticks": DELAY_TICKS,
        "none_count": none_count,
        "hold_count": state["hold_count"],
        "propagation_count": state["propagation_count"],
        "low_completion_released": low_release,
        "beta_after_low_completion": beta_after_low,
        "high_completion_released": high_release,
        "beta_after_high_completion": beta_after_high,
        "post_release_propagated": post is not None,
    }
    result["pass"] = (
        none_count == DELAY_TICKS
        and state["hold_count"] == DELAY_TICKS
        and not low_release
        and beta_after_low
        and high_release
        and not beta_after_high
        and post is not None
        and state["propagation_count"] == 1
    )
    return result


def run_uc2_weakened_hold_passthrough() -> dict:
    gate = BetaGate(initial_beta_elevated=False, completion_release_threshold=0.75)
    passthrough_count = 0
    for tick in range(DELAY_TICKS):
        out = gate.propagate(_policy_state(tick))
        if out is not None:
            passthrough_count += 1
    state = gate.get_state()
    result = {
        "delay_ticks": DELAY_TICKS,
        "passthrough_count": passthrough_count,
        "hold_count": state["hold_count"],
        "propagation_count": state["propagation_count"],
        "beta_elevated": state["beta_elevated"],
    }
    result["pass"] = (
        passthrough_count == DELAY_TICKS
        and state["hold_count"] == 0
        and state["propagation_count"] == DELAY_TICKS
        and not state["beta_elevated"]
    )
    return result


def _make_lateral_pfc() -> LateralPFCAnalog:
    cfg = LateralPFCConfig(
        use_lateral_pfc_analog=True,
        update_eta=0.05,
        rule_dim=16,
        world_pool_weight=0.5,
    )
    return LateralPFCAnalog(delta_dim=8, world_dim=10, config=cfg)


def _install_rule_state(lpfc: LateralPFCAnalog) -> torch.Tensor:
    z_delta = torch.ones(1, 8)
    z_world = torch.ones(1, 10)
    for _ in range(RULE_INSTALL_TICKS):
        lpfc.update(z_delta, z_world, gate=1.0)
    return lpfc.rule_state.detach().clone()


def _apply_distractor(lpfc: LateralPFCAnalog, gate: float) -> None:
    z_delta = -torch.ones(1, 8)
    z_world = -torch.ones(1, 10)
    for _ in range(DISTRACTOR_TICKS):
        lpfc.update(z_delta, z_world, gate=gate)


def run_uc3_sd033a_replay_gate_preserves_rule_state() -> dict:
    torch.manual_seed(461)
    lpfc_low_gate = _make_lateral_pfc()
    installed_low = _install_rule_state(lpfc_low_gate)
    installed_norm = float(installed_low.norm().item())
    _apply_distractor(lpfc_low_gate, gate=0.05)
    low_cos = float(
        torch.nn.functional.cosine_similarity(
            installed_low, lpfc_low_gate.rule_state, dim=-1
        ).item()
    )
    low_norm = float(lpfc_low_gate.rule_state.norm().item())
    low_eta = float(lpfc_low_gate.get_state()["last_effective_eta"])

    torch.manual_seed(461)
    lpfc_full_gate = _make_lateral_pfc()
    installed_full = _install_rule_state(lpfc_full_gate)
    _apply_distractor(lpfc_full_gate, gate=1.0)
    full_cos = float(
        torch.nn.functional.cosine_similarity(
            installed_full, lpfc_full_gate.rule_state, dim=-1
        ).item()
    )
    full_norm = float(lpfc_full_gate.rule_state.norm().item())
    full_eta = float(lpfc_full_gate.get_state()["last_effective_eta"])

    result = {
        "installed_norm": installed_norm,
        "low_gate": 0.05,
        "low_gate_effective_eta": low_eta,
        "low_gate_post_norm": low_norm,
        "low_gate_cosine_to_installed": low_cos,
        "full_gate": 1.0,
        "full_gate_effective_eta": full_eta,
        "full_gate_post_norm": full_norm,
        "full_gate_cosine_to_installed": full_cos,
        "distractor_ticks": DISTRACTOR_TICKS,
    }
    result["pass"] = (
        installed_norm > 0.5
        and low_eta < 0.003
        and full_eta >= 0.049
        and low_cos >= 0.95
        and full_cos <= 0.0
    )
    return result


def run_uc4_strengthened_hold_resists_premature_release() -> dict:
    gate = BetaGate(initial_beta_elevated=False, completion_release_threshold=0.95)
    gate.elevate()
    moderate_release = gate.receive_hippocampal_completion(0.80)
    beta_after_moderate = bool(gate.is_elevated)
    high_release = gate.receive_hippocampal_completion(0.99)
    beta_after_high = bool(gate.is_elevated)
    result = {
        "completion_release_threshold": 0.95,
        "moderate_completion": 0.80,
        "moderate_released": moderate_release,
        "beta_after_moderate": beta_after_moderate,
        "high_completion": 0.99,
        "high_released": high_release,
        "beta_after_high": beta_after_high,
    }
    result["pass"] = (
        not moderate_release
        and beta_after_moderate
        and high_release
        and not beta_after_high
    )
    return result


def run_uc5_mech261_mode_gate_table() -> dict:
    coord = SalienceCoordinator()
    external_gate = coord.write_gate("sd_033a")

    coord._operating_mode = {
        "external_task": 0.0,
        "internal_planning": 0.0,
        "internal_replay": 1.0,
        "offline_consolidation": 0.0,
    }
    replay_gate = coord.write_gate("sd_033a")

    coord._operating_mode = {
        "external_task": 0.0,
        "internal_planning": 0.0,
        "internal_replay": 0.0,
        "offline_consolidation": 1.0,
    }
    offline_gate = coord.write_gate("sd_033a")

    result = {
        "external_task_gate": external_gate,
        "internal_replay_gate": replay_gate,
        "offline_consolidation_gate": offline_gate,
    }
    result["pass"] = (
        abs(external_gate - 1.0) < 1e-9
        and abs(replay_gate - 0.05) < 1e-9
        and abs(offline_gate - 0.3) < 1e-9
    )
    return result


def run_uc6_terminal_closure_signal() -> dict:
    cfg = REEConfig.from_dims(
        body_obs_dim=12,
        world_obs_dim=250,
        action_dim=4,
        use_dacc=True,
        use_salience_coordinator=True,
        use_lateral_pfc_analog=True,
        use_closure_operator=True,
    )
    agent = REEAgent(cfg)

    z_delta = torch.ones(1, cfg.latent.delta_dim)
    z_world = torch.zeros(1, cfg.latent.world_dim)
    agent.lateral_pfc.update(z_delta, z_world, gate=1.0)
    rule_norm_pre = float(agent.lateral_pfc.rule_state.norm().item())

    agent.beta_gate.elevate()
    agent.dacc._pe_ema = 0.42
    pre_history_len = len(agent.dacc._action_history)
    event = agent.closure_operator.emit_closure(
        action_class=2,
        z_world=z_world,
        current_mode="external_task",
        sd033a_gate=1.0,
        bypass_mode_conditioning=False,
    )
    post_history_len = len(agent.dacc._action_history)
    closure_signal_value = agent.salience._input_signals.get("closure_event")

    result = {
        "rule_norm_pre": rule_norm_pre,
        "event_fired": event.fired,
        "event_reason": event.reason,
        "beta_released": event.beta_released,
        "beta_post_elevated": bool(agent.beta_gate.is_elevated),
        "nogo_pushed": event.nogo_pushed,
        "pre_history_len": pre_history_len,
        "post_history_len": post_history_len,
        "salience_signal_written": event.salience_signal_written,
        "closure_signal_value": closure_signal_value,
        "pe_ema_reset": event.pe_ema_reset,
        "dacc_pe_ema_post": agent.dacc._pe_ema,
    }
    result["pass"] = (
        rule_norm_pre > 0.0
        and event.fired
        and event.reason == "explicit"
        and event.beta_released
        and not agent.beta_gate.is_elevated
        and event.nogo_pushed >= cfg.closure_nogo_injection_count
        and post_history_len >= pre_history_len + cfg.closure_nogo_injection_count
        and event.salience_signal_written
        and closure_signal_value == cfg.closure_signal_value
        and event.pe_ema_reset
        and agent.dacc._pe_ema is None
    )
    return result


def run_all_subtests() -> dict:
    return {
        "UC1_baseline_hold_window": run_uc1_baseline_hold_window(),
        "UC2_weakened_hold_passthrough": run_uc2_weakened_hold_passthrough(),
        "UC3_sd033a_replay_gate_preserves_rule_state": (
            run_uc3_sd033a_replay_gate_preserves_rule_state()
        ),
        "UC4_strengthened_hold_resists_premature_release": (
            run_uc4_strengthened_hold_resists_premature_release()
        ),
        "UC5_mech261_mode_gate_table": run_uc5_mech261_mode_gate_table(),
        "UC6_terminal_closure_signal": run_uc6_terminal_closure_signal(),
    }


def build_manifest(subtests: dict, elapsed: float) -> dict:
    all_pass = all(r.get("pass") for r in subtests.values())
    outcome = "PASS" if all_pass else "FAIL"
    ts = _utc_now().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"

    return {
        "schema_version": "v1",
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": _utc_iso_now(),
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": CLAIM_IDS,
        "result": outcome,
        "outcome": outcome,
        "evidence_direction": "supports" if all_pass else "weakens",
        "evidence_direction_per_claim": {
            claim_id: ("supports" if all_pass else "weakens")
            for claim_id in CLAIM_IDS
        },
        "metrics": subtests,
        "thresholds": {
            "delay_ticks": DELAY_TICKS,
            "rule_install_ticks": RULE_INSTALL_TICKS,
            "distractor_ticks": DISTRACTOR_TICKS,
            "uc3_low_gate_cosine_floor": 0.95,
            "uc3_full_gate_cosine_ceiling": 0.0,
        },
        "elapsed_seconds": elapsed,
        "notes": (
            "V3-EXQ-461 closes commitment_closure GAP-2 at substrate-readiness "
            "level for EXP-0157 delayed-reward persistence. It validates the "
            "Hold-axis ingredients without claiming the full behavioural "
            "delay-to-reward environment exists yet. Baseline Hold uses "
            "MECH-090 beta elevation to suppress policy propagation across a "
            "synthetic delay window; weakened Hold passes through immediately; "
            "SD-033a rule_state persists under the MECH-261 replay gate and is "
            "overwritten under full distractor writes; strengthened Hold resists "
            "premature completion; SD-034 terminal closure releases beta and "
            "installs No-Go. Full delayed-reward behavioural validation remains "
            "blocked on commitment_closure GAP-3 CausalGridWorldV2 env "
            "extensions."
        ),
    }


def main(dry_run: bool = False):
    t0 = time.time()
    subtests = run_all_subtests()
    elapsed = time.time() - t0
    manifest = build_manifest(subtests, elapsed)
    outcome = manifest["outcome"]

    print("=== V3-EXQ-461 delayed-reward persistence substrate diagnostic ===", flush=True)
    for name, result in subtests.items():
        verdict = "PASS" if result.get("pass") else "FAIL"
        print(f"{name}: {verdict}", flush=True)
    print(f"verdict: {outcome}", flush=True)

    if dry_run:
        return 0 if outcome == "PASS" else 1

    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=False,
        config=manifest.get("config"),
        seeds=None,
        script_path=Path(__file__),
    )
    print(f"Result written to: {out_path}", flush=True)
    return outcome, out_path, manifest["run_id"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Smoke run, no manifest.")
    args = parser.parse_args()
    result = main(dry_run=args.dry_run)
    if args.dry_run or result == 0:
        sys.exit(0)
    if result == 1:
        sys.exit(1)
    _outcome, _out_path, _run_id = result
    emit_outcome(
        outcome=_outcome if _outcome in ("PASS", "FAIL") else "FAIL",
        manifest_path=_out_path,
        run_id=_run_id,
        queue_id=QUEUE_ID,
        exit_reason="ok" if _outcome == "PASS" else "fail",
    )
    sys.exit(0)
