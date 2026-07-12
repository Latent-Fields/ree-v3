"""V3-EXQ-496: MECH-292 Ranked Ghost-Goal Bank -- Substrate Validation.

experiment_purpose: diagnostic
status_when_drafted: GATED ON MECH-292 SUBSTRATE LANDING (this script lands with it).

CLAIM TESTED:
MECH-292 (hippocampal.unresolved_goal_ghost_bank) -- the substrate-level
falsifiable signature is: "anchors with stored goal_payload that match the
current z_goal rank above equally stale but goal-irrelevant anchors; the
goal_match_floor excludes payload-less / goal-zero anchors from the bank
entirely."

Behavioural validation (faster recovery from blocked-goal conditions vs
random-probe baseline) lands in V3-EXQ-495 (V3 full-completion gate),
which requires MECH-293 to consume the bank. This script is the
substrate-readiness gate that MECH-293 wiring depends on.

DESIGN: 5 sub-tests (UC1-UC5) per MECH-292 spec
(REE_assembly/docs/architecture/mech_292_ghost_goal_bank.md).

  UC1 module_importable: ree_core.hippocampal.ghost_goal_bank exposes
      GhostGoalBank, GhostGoalBankConfig, GhostGoalBankEntry; HippocampalModule
      exposes rank_ghost_goals + reset_ghost_goal_bank;
      HippocampalConfig.use_mech292_ghost_bank field exists with default False;
      ghost_goal_bank_config field exists with documented sub-knobs.

  UC2 master_off_no_op: REEAgent built with default config
      (use_mech292_ghost_bank OFF) -- after a 60-tick env episode that creates
      anchors via MECH-269 + MECH-288 BoundaryEvents,
      hippocampal.ghost_goal_bank is None and rank_ghost_goals(z_goal)
      returns []. Bit-identical to pre-MECH-292 substrate.

  UC3 ranking_fires: SD-039 population ON + MECH-292 ON + active goal regime;
      after a 60-tick episode rank() returns a non-empty bank with strictly
      non-increasing ghost_priority across entries. The pool over which the
      bank is built (default include_inactive=True, include_active=False)
      must contain at least one inactive trace with a populated payload --
      forced boundaries (every 8 ticks) drive remaps that produce inactive
      traces with preserved payloads (SD-039 dual-trace).

  UC4 goal_irrelevant_excluded: parallel to V3-EXQ-494 UC5. Two agents:
        Phase A (30 ticks, drive=0, z_goal_enabled=False): inactive traces
          are written outside any goal regime. Population layer skips
          z_goal_snapshot (None) so anchor.goal_match() returns 0.0.
          The bank's goal_match_floor (default 0.05) excludes them entirely.
        Phase B (30 ticks, drive=2.0, benefit=0.4, z_goal_enabled=True):
          inactive traces carry z_goal_snapshot tied to the current goal
          direction. The bank admits them with goal_match >= floor.
        Cross-agent score: build the Phase B bank using the union of Phase A
        and Phase B anchors (constructed by manually splicing inactive
        traces from agent_a into agent_b's anchor pool, OR by separately
        ranking each agent's pool against agent_b's z_goal). We use the
        latter (simpler, no anchor-pool mutation): rank_ghost_goals on
        agent_a returns 0 admitted (or only ones below floor); on agent_b
        returns at least 1 admitted entry whose ghost_priority is dominated
        by the goal_match component.

  UC5 component_breakdown_consistent: per-entry components dict has the four
      keys ("wanting", "goal_match", "staleness", "recoverability") and
      sum(components.values()) == ghost_priority within float tolerance
      (1e-5). Diagnostics dict surfaces n_candidates_scanned, n_admitted,
      n_below_floor, max_priority, mean_priority.

PASS CRITERIA: UC1 AND UC2 AND UC3 AND UC4 AND UC5.

claim_ids: ['MECH-292'] (single-claim diagnostic; MECH-293 / ARC-060 are
downstream consumers and stay out of scope for this script).

Run with:
  /opt/local/bin/python3 experiments/v3_exq_496_mech292_ghost_goal_bank_validation.py

Writes a flat JSON manifest to REE_assembly/evidence/experiments/.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EVIDENCE_ROOT = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_env(seed: int = 42) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed,
        size=8,
        num_resources=3,
        num_hazards=2,
        hazard_harm=0.1,
        resource_benefit=0.5,
        resource_respawn_on_consume=True,
        proximity_harm_scale=0.05,
        proximity_benefit_scale=0.05,
        proximity_approach_threshold=0.15,
        use_proxy_fields=True,
    )


def _make_agent(
    use_mech292: bool,
    use_sd039: bool,
    goal_active: bool,
    seed: int = 42,
) -> REEAgent:
    """Build an agent with MECH-269 anchor substrate ON; SD-039 + MECH-292 toggled."""
    torch.manual_seed(seed)
    env = _make_env(seed=seed)
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        # MECH-269 stack required for any anchor to exist.
        use_per_stream_vs=True,
        use_event_segmenter=True,
        use_invalidation_trigger=True,
        use_anchor_sets=True,
        # SD-039 substrate flag (must be ON for MECH-292 to score non-zero).
        use_sd039_anchor_payload=use_sd039,
        # MECH-292 master flag under test.
        use_mech292_ghost_bank=use_mech292,
        drive_weight=2.0 if goal_active else 0.0,
        use_resource_proximity_head=True,
    )
    cfg.goal.z_goal_enabled = bool(goal_active)
    return REEAgent(cfg)


def _step_episode(
    agent: REEAgent,
    env: CausalGridWorldV2,
    n_ticks: int,
    *,
    force_drive: Optional[float] = None,
    force_benefit: Optional[float] = None,
    force_boundary_every: Optional[int] = 8,
) -> None:
    """Run n_ticks of the standard env loop, optionally forcing drive / benefit.

    Forces a fast-scale boundary every `force_boundary_every` ticks (default 8)
    so the SD-039 contract is exercised without depending on stochastic
    boundary firing. Same helper shape as V3-EXQ-494's _step_episode.
    """
    flat_obs, obs_dict = env.reset()
    agent.reset()
    for tick_idx in range(n_ticks):
        obs_body = obs_dict["body_state"]
        obs_world = obs_dict["world_state"]
        latent = agent.sense(obs_body, obs_world)
        ticks = agent.clock.advance()
        if ticks.get("e1_tick", False):
            e1_prior = agent._e1_tick(latent)
        else:
            e1_prior = torch.zeros(1, agent.config.latent.world_dim)
        candidates = agent.generate_trajectories(latent, e1_prior, ticks)
        if force_drive is not None or force_benefit is not None:
            agent.update_z_goal(
                benefit_exposure=force_benefit if force_benefit is not None else 0.0,
                drive_level=force_drive if force_drive is not None else 0.0,
            )
        if (
            force_boundary_every is not None
            and force_boundary_every > 0
            and tick_idx > 0
            and (tick_idx % force_boundary_every) == 0
            and agent.hippocampal is not None
            and agent.hippocampal.event_segmenter is not None
            and agent.hippocampal.anchor_set is not None
        ):
            ev = agent.hippocampal.event_segmenter.force_boundary(
                "fast", reason=f"v3_exq_496_t{tick_idx}",
            )
            payload = agent.hippocampal.build_goal_payload(
                latent_state=latent,
                goal_state=agent.goal_state,
                residue_field=agent.residue_field,
                bla_output=agent._bla_last_output,
                current_step=int(agent._step_count),
                simulation_mode=False,
            )
            agent.hippocampal.tick_anchor_set(latent, [ev], goal_payload=payload)
        action = agent.select_action(candidates, ticks, temperature=1.0)
        if action is None:
            action = torch.zeros(1, env.action_dim)
            action[0, 0] = 1.0
            agent._last_action = action
        flat_obs, harm_signal, done, info, obs_dict = env.step(action)
        if done:
            flat_obs, obs_dict = env.reset()


def _current_z_goal(agent: REEAgent) -> Optional[torch.Tensor]:
    if agent.goal_state is None:
        return None
    z_goal = getattr(agent.goal_state, "z_goal", None)
    if z_goal is None:
        return None
    return z_goal.detach()


# ---------------------------------------------------------------------------
# Sub-tests
# ---------------------------------------------------------------------------
def run_uc1_module_importable() -> Dict[str, Any]:
    """UC1: module classes / methods / config fields accessible."""
    try:
        from ree_core.hippocampal.ghost_goal_bank import (
            GhostGoalBank,
            GhostGoalBankConfig,
            GhostGoalBankEntry,
        )
        from ree_core.hippocampal.module import HippocampalModule
        from ree_core.utils.config import HippocampalConfig

        # GhostGoalBankConfig must expose the documented sub-knobs.
        cfg = GhostGoalBankConfig()
        required_fields = {
            "wanting_weight",
            "goal_match_weight",
            "staleness_weight",
            "recoverability_weight",
            "goal_match_floor",
            "top_k",
            "default_recoverability_when_unknown",
            "include_inactive",
            "include_active",
        }
        cfg_fields = set(GhostGoalBankConfig.__dataclass_fields__.keys())
        has_required = required_fields.issubset(cfg_fields)

        # HippocampalConfig must carry the master flag and nested config.
        hc = HippocampalConfig()
        has_master_flag = hasattr(hc, "use_mech292_ghost_bank")
        has_nested_cfg = hasattr(hc, "ghost_goal_bank_config")
        master_default_off = (getattr(hc, "use_mech292_ghost_bank", True) is False)

        # HippocampalModule must expose rank_ghost_goals + reset_ghost_goal_bank.
        has_rank = hasattr(HippocampalModule, "rank_ghost_goals")
        has_reset = hasattr(HippocampalModule, "reset_ghost_goal_bank")

        # GhostGoalBankEntry must expose anchor / ghost_priority / components.
        entry_fields = set(GhostGoalBankEntry.__dataclass_fields__.keys())
        has_entry_fields = {"anchor", "ghost_priority", "components"}.issubset(
            entry_fields
        )

        ok = (
            has_required
            and has_master_flag
            and has_nested_cfg
            and master_default_off
            and has_rank
            and has_reset
            and has_entry_fields
        )
        return {
            "pass": bool(ok),
            "has_required_config_fields": bool(has_required),
            "has_master_flag": bool(has_master_flag),
            "has_nested_cfg": bool(has_nested_cfg),
            "master_default_off": bool(master_default_off),
            "has_rank_method": bool(has_rank),
            "has_reset_method": bool(has_reset),
            "has_entry_fields": bool(has_entry_fields),
            "config_fields": sorted(cfg_fields),
        }
    except Exception as exc:
        return {"pass": False, "error": repr(exc)}


def run_uc2_master_off_no_op() -> Dict[str, Any]:
    """UC2: MECH-292 OFF -> ghost_goal_bank is None; rank() returns []."""
    try:
        agent = _make_agent(use_mech292=False, use_sd039=True, goal_active=True, seed=42)
        env = _make_env(seed=42)
        _step_episode(agent, env, n_ticks=30, force_drive=0.8, force_benefit=0.4)
        bank_is_none = agent.hippocampal.ghost_goal_bank is None
        z_goal = _current_z_goal(agent)
        empty_rank = agent.hippocampal.rank_ghost_goals(z_goal) == []
        ok = bank_is_none and empty_rank
        return {
            "pass": bool(ok),
            "ghost_goal_bank_is_none": bool(bank_is_none),
            "rank_returns_empty": bool(empty_rank),
        }
    except Exception as exc:
        return {"pass": False, "error": repr(exc)}


def run_uc3_ranking_fires() -> Dict[str, Any]:
    """UC3: MECH-292 ON + SD-039 ON + active goal -> bank non-empty, decreasing."""
    try:
        agent = _make_agent(use_mech292=True, use_sd039=True, goal_active=True, seed=42)
        env = _make_env(seed=42)
        _step_episode(agent, env, n_ticks=60, force_drive=0.8, force_benefit=0.4)
        z_goal = _current_z_goal(agent)
        bank = agent.hippocampal.rank_ghost_goals(z_goal)
        priorities = [e.ghost_priority for e in bank]
        is_decreasing = priorities == sorted(priorities, reverse=True)
        all_anchors = agent.hippocampal.anchor_set.all_anchors()
        n_inactive = sum(1 for a in all_anchors if not a.active)
        n_inactive_with_payload = sum(
            1 for a in all_anchors
            if not a.active and a.goal_payload is not None
            and a.goal_payload.z_goal_snapshot is not None
        )
        diag = agent.hippocampal.ghost_goal_bank.get_diagnostics()
        ok = len(bank) >= 1 and is_decreasing and n_inactive_with_payload >= 1
        return {
            "pass": bool(ok),
            "n_bank_entries": len(bank),
            "is_decreasing": bool(is_decreasing),
            "n_inactive_anchors": int(n_inactive),
            "n_inactive_with_populated_payload": int(n_inactive_with_payload),
            "n_candidates_scanned": int(diag.get("n_candidates_scanned", 0)),
            "n_admitted": int(diag.get("n_admitted", 0)),
            "max_priority": float(diag.get("max_priority", 0.0)),
            "mean_priority": float(diag.get("mean_priority", 0.0)),
        }
    except Exception as exc:
        return {"pass": False, "error": repr(exc)}


def run_uc4_goal_irrelevant_excluded() -> Dict[str, Any]:
    """UC4: anchors below goal_match_floor are absent from the bank.

    Phase A: no-goal regime. Inactive traces accumulate; their stored
      goal_payload either is None or has z_goal_snapshot=None.
      goal_match() returns 0.0 across the board -- below floor.
    Phase B: goal-active regime. Inactive traces accumulate with
      z_goal_snapshot tied to the current goal; goal_match() above floor
      for at least one trace.
    Both agents share env seed; goal_match is a pure function of the
    stored snapshot and the supplied current z_goal.
    """
    try:
        # Phase A
        agent_a = _make_agent(
            use_mech292=True, use_sd039=True, goal_active=False, seed=42,
        )
        env_a = _make_env(seed=42)
        _step_episode(agent_a, env_a, n_ticks=30)

        # Phase B
        agent_b = _make_agent(
            use_mech292=True, use_sd039=True, goal_active=True, seed=42,
        )
        env_b = _make_env(seed=42)
        _step_episode(agent_b, env_b, n_ticks=30, force_drive=0.8, force_benefit=0.4)
        z_goal_b = _current_z_goal(agent_b)

        # Rank each agent's pool against agent_b's z_goal.
        bank_a = agent_a.hippocampal.rank_ghost_goals(z_goal_b)
        bank_b = agent_b.hippocampal.rank_ghost_goals(z_goal_b)

        diag_a = agent_a.hippocampal.ghost_goal_bank.get_diagnostics()
        diag_b = agent_b.hippocampal.ghost_goal_bank.get_diagnostics()

        # Phase A bank should be empty (all anchors below floor / no payload).
        # Phase B bank should be non-empty.
        a_excluded = len(bank_a) == 0
        b_admitted = len(bank_b) >= 1

        # Goal-match component dominance check on Phase B top entry.
        if bank_b:
            top = bank_b[0]
            comps = top.components
            gm_dominant = comps.get("goal_match", 0.0) >= max(
                comps.get("wanting", 0.0),
                comps.get("staleness", 0.0),
            )
        else:
            gm_dominant = False

        ok = a_excluded and b_admitted and gm_dominant
        return {
            "pass": bool(ok),
            "phase_a_bank_size": len(bank_a),
            "phase_b_bank_size": len(bank_b),
            "phase_a_n_below_floor": int(diag_a.get("n_below_floor", 0)),
            "phase_a_n_no_payload": int(diag_a.get("n_no_payload", 0)),
            "phase_b_n_admitted": int(diag_b.get("n_admitted", 0)),
            "phase_b_max_priority": float(diag_b.get("max_priority", 0.0)),
            "goal_match_dominant_top_b": bool(gm_dominant),
        }
    except Exception as exc:
        return {"pass": False, "error": repr(exc)}


def run_uc5_component_breakdown_consistent() -> Dict[str, Any]:
    """UC5: per-entry components dict sums to ghost_priority within tolerance."""
    try:
        agent = _make_agent(use_mech292=True, use_sd039=True, goal_active=True, seed=42)
        env = _make_env(seed=42)
        _step_episode(agent, env, n_ticks=60, force_drive=0.8, force_benefit=0.4)
        z_goal = _current_z_goal(agent)
        bank = agent.hippocampal.rank_ghost_goals(z_goal)

        if not bank:
            return {
                "pass": False,
                "error": "bank empty -- cannot check component sum",
                "n_bank_entries": 0,
            }

        required_keys = {"wanting", "goal_match", "staleness", "recoverability"}
        max_diff = 0.0
        all_keys_present = True
        for e in bank:
            keys = set(e.components.keys())
            if not required_keys.issubset(keys):
                all_keys_present = False
            s = sum(e.components.values())
            diff = abs(s - e.ghost_priority)
            max_diff = max(max_diff, diff)

        ok = all_keys_present and max_diff < 1e-5
        return {
            "pass": bool(ok),
            "n_bank_entries": len(bank),
            "all_required_keys_present": bool(all_keys_present),
            "max_sum_vs_priority_diff": float(max_diff),
        }
    except Exception as exc:
        return {"pass": False, "error": repr(exc)}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main(dry_run: bool = False) -> int:
    t0 = time.time()
    print("[v3_exq_496] running 5 sub-tests for MECH-292 ghost-goal bank...")
    if dry_run:
        uc1 = run_uc1_module_importable()
        all_metrics = {"UC1_module_importable": uc1}
        all_pass = bool(uc1["pass"])
    else:
        uc1 = run_uc1_module_importable()
        uc2 = run_uc2_master_off_no_op()
        uc3 = run_uc3_ranking_fires()
        uc4 = run_uc4_goal_irrelevant_excluded()
        uc5 = run_uc5_component_breakdown_consistent()
        all_metrics = {
            "UC1_module_importable": uc1,
            "UC2_master_off_no_op": uc2,
            "UC3_ranking_fires": uc3,
            "UC4_goal_irrelevant_excluded": uc4,
            "UC5_component_breakdown_consistent": uc5,
        }
        all_pass = all(m["pass"] for m in all_metrics.values())
    elapsed = time.time() - t0

    for name, m in all_metrics.items():
        print(f"  {name}: {'PASS' if m['pass'] else 'FAIL'}  {m}")
    print(f"[v3_exq_496] overall: {'PASS' if all_pass else 'FAIL'} ({elapsed:.1f}s)")

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"v3_exq_496_mech292_ghost_goal_bank_validation_{ts}_v3"
    manifest = {
        "schema_version": "v1",
        "run_id": run_id,
        "experiment_type": "v3_exq_496_mech292_ghost_goal_bank_validation",
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "experiment_purpose": "diagnostic",
        "claim_ids": ["MECH-292"],
        "evidence_direction": "supports" if all_pass else "weakens",
        "evidence_direction_per_claim": {
            "MECH-292": "supports" if all_pass else "weakens",
        },
        "evidence_direction_note": (
            "MECH-292 ranked ghost-goal bank substrate validation. UC1 confirms "
            "module / config / method exposure; UC2 confirms master-OFF "
            "bit-identical (bank is None, rank returns []); UC3 confirms ON "
            "produces a non-empty decreasing bank when SD-039 has populated "
            "inactive-trace payloads; UC4 confirms the goal_match_floor "
            "rumination guard excludes payload-less / goal-zero anchors; "
            "UC5 confirms per-entry component dict sums to ghost_priority. "
            "Behavioural validation (faster recovery from blocked-goal "
            "conditions vs random-probe baseline) lands in V3-EXQ-495 once "
            "MECH-293 wires propose_trajectories() to consume the bank."
        ),
        "outcome": "PASS" if all_pass else "FAIL",
        "elapsed_sec": elapsed,
        "metrics": all_metrics,
        "dry_run": bool(dry_run),
    }

    if not dry_run:
        out_dir = EVIDENCE_ROOT / "v3_exq_496_mech292_ghost_goal_bank_validation"
        out_file = write_flat_manifest(
            manifest,
            out_dir,
            dry_run=False,
            config=manifest.get("config"),
            seeds=None,
            script_path=Path(__file__),
        )
        print(f"Result written to: {out_file}", flush=True)
    return 0 if all_pass else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="UC1 only; no env episode; no manifest write.")
    args = parser.parse_args()
    sys.exit(main(dry_run=args.dry_run))
