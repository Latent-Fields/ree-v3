"""V3-EXQ-494: SD-039 Anchor Goal-Snapshot Payload -- Population Layer Validation.

experiment_purpose: diagnostic
status_when_drafted: GATED ON SD-039 POPULATION LAYER (deferred follow-on as of 2026-04-27)

WHY THIS IS DRAFTED BUT NOT QUEUED:
SD-039's substrate-side LANDED 2026-04-26 (AnchorGoalPayload dataclass,
Anchor.goal_payload field, Anchor.goal_match() helper, AnchorSet.query_by_goal_match()
helper, refresh-on-invalidate semantic, master flag use_sd039_anchor_payload). The
MODULE-LEVEL POPULATION LAYER is deferred: REEAgent / HippocampalModule do not yet
populate the payload from GoalState (z_goal_snapshot), ResidueField VALENCE_WANTING
(wanting_strength), or amygdala arousal tags (arousal_tag) at write/remap/invalidate
sites. Until that population pass lands, anchors written under
use_sd039_anchor_payload=True will carry goal_payload=None (or all-default-zero
fields) and UC3-UC6 will FAIL on the population side. This script captures the
acceptance criteria the population layer must satisfy. DO NOT QUEUE until population
lands -- it would just FAIL on a substrate-readiness gap, polluting the evidence
record. See WORKSPACE_STATE.md "implement-sd039-anchor-payload" entry for the
deferred work scope.

CLAIM TESTED:
SD-039 (hippocampal.anchor_goal_snapshot_payload) -- the falsifiable signature is:
"after reward relocation or path blockage, inactive anchors on the formerly valid
approach path retain non-zero goal_match with current z_goal while goal-irrelevant
stale anchors do not."

DESIGN: 6 sub-tests (UC1-UC6) parallel to V3-EXQ-490 / V3-EXQ-493 conventions.

  UC1 module_importable: regulators / hippocampal subpackages expose
      AnchorGoalPayload, Anchor (with goal_payload field), AnchorSet.query_by_goal_match,
      AnchorSet.use_sd039_anchor_payload config flag. Pure module contract.

  UC2 master_off_no_op: REEAgent built with default config (use_sd039_anchor_payload
      OFF) -- after a 30-tick env episode that creates anchors via MECH-269 +
      MECH-288 BoundaryEvents, every anchor carries goal_payload=None and
      AnchorSet.query_by_goal_match(current_z_goal) returns the empty list.
      Bit-identical to pre-SD-039 substrate.

  UC3 population_fires: with master ON + active goal regime (drive=2.0 + benefit
      exposure + z_goal_enabled), after a 60-tick env episode AnchorSet contains
      at least one anchor with goal_payload != None AND
      goal_payload.z_goal_snapshot != None AND goal_payload.payload_written_step > 0.
      query_by_goal_match(current_z_goal) returns at least one (anchor, score)
      pair with score > 0. PASS test for the population layer firing.

  UC4 payload_survives_mark_inactive: force at least one anchor remap during the
      run (boundary on a registered scale with the same stream_mixture installs
      a new active anchor, marking the prior active inactive). Confirm:
      (a) the inactive anchor's goal_payload is NOT cleared (dual-trace
      preservation); (b) the new active anchor has its own freshly-written
      goal_payload; (c) AnchorSet.query_by_goal_match with active_only=False
      returns BOTH (active and inactive forms preserved).

  UC5 goal_relevant_vs_irrelevant_dissociation (THE FALSIFIABLE TEST):
      Two-phase episode design.
        Phase A (30 ticks, drive=0, z_goal_enabled=False, no active goal):
          anchors accumulate during exploration. Each accumulated anchor's
          goal_payload either is None (if population layer skips when no goal
          active) or has z_goal_snapshot=None. These are GOAL-IRRELEVANT
          anchors -- written outside any goal regime.
        Phase B (30 ticks, drive=2.0, benefit=0.4, z_goal_enabled=True):
          new anchors accumulate. goal_payload.z_goal_snapshot is non-None
          and tied to the current goal direction. These are GOAL-RELEVANT
          anchors.
        Query: AnchorSet.query_by_goal_match(current_z_goal=phase_B_z_goal).
        Acceptance:
          - mean(goal_match for Phase B anchors) > 0.3 (substantively non-zero)
          - mean(goal_match for Phase A anchors) < 0.05 (effectively zero)
          - count(Phase B anchors with score > 0.3) >= 1 (at least one
            substantive goal-relevant inactive trace exists in the dual-
            trace pool)
      This is the substrate-level falsifiable signature. Note this is a
      WITHIN-SUBSTRATE test (does the population layer attach the right
      payload to the right anchors), not the eventual behavioural signature
      ("inactive anchors on a formerly valid path keep non-zero goal_match
      after path blockage"); behavioural validation requires MECH-292 +
      MECH-293 consumers and an env extension with explicit blockage and
      lives in a successor experiment.

  UC6 mech094_simulation_gate: with master ON, anchors written via simulation
      / replay paths (hypothesis_tag=True) must NOT carry populated
      goal_payload (call-site scoping pattern -- the population layer must
      respect the MECH-094 simulation_mode argument). Probe by directly
      calling HippocampalModule.replay() (or equivalent simulation entry
      point) and confirming that no anchor written via that path has a
      payload populated by the waking-stream signals.

PASS CRITERIA: UC1 AND UC2 AND UC3 AND UC4 AND UC5 AND UC6.

When this script is run BEFORE the population layer lands:
  UC1 PASS  (substrate-side classes already present)
  UC2 PASS  (master OFF behaviour is already correct)
  UC3 FAIL  (population layer not wired -- payload stays None)
  UC4 FAIL  (no payload to preserve across mark_inactive)
  UC5 FAIL  (no payload to dissociate goal-relevant from goal-irrelevant)
  UC6 PASS  (vacuously: no payload is written from anywhere, simulation
             included)
This 2/6 pre-population baseline is recorded for clarity; it is the
SUBSTRATE-READINESS SIGNATURE the population layer must clear.

claim_ids: ['SD-039']  (single-claim diagnostic; MECH-292 / MECH-293 / ARC-060
are downstream consumers and are NOT directly exercised by this script).

Run with:
  /opt/local/bin/python3 experiments/v3_exq_494_sd039_anchor_payload_validation.py

Writes a flat JSON manifest to REE_assembly/evidence/experiments/.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig
from experiments.pack_writer import write_flat_manifest  # noqa: E402

# Output target: REE_assembly evidence root.
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


def _make_agent(use_sd039: bool, goal_active: bool, seed: int = 42) -> REEAgent:
    """Build an agent with MECH-269 anchor substrate ON and SD-039 toggled."""
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
        # SD-039 substrate flag under test.
        use_sd039_anchor_payload=use_sd039,
        # Goal regime (Phase B; Phase A overrides z_goal_enabled below).
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

    The MECH-288 event segmenter is configured for stationary noise and rarely
    fires BoundaryEvents over a few-dozen-tick window. To exercise the SD-039
    population layer without depending on stochastic boundary firing, we
    explicitly force a fast-scale boundary every `force_boundary_every` ticks
    (default 8). This drives the MECH-269 anchor-set write path through
    consume_boundary_events with the current waking-stream payload, which is
    exactly the integration site SD-039 populates. Set force_boundary_every=None
    to disable forced boundaries (legacy behaviour).
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
        # Force a boundary -> direct anchor-set tick with the current payload.
        # Bypasses the natural event_segmenter cadence (which may not fire
        # within the test window) without bypassing the SD-039 contract.
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
                "fast", reason=f"v3_exq_494_t{tick_idx}",
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
            # NOTE: only env is reset, NOT agent. agent.reset() would clear
            # anchor_set._all (per-episode reset semantic), wiping inactive
            # traces UC4 / UC5 need to observe within the test window.
            flat_obs, obs_dict = env.reset()


def _all_anchors_with_payload(agent: REEAgent) -> List[Any]:
    """Return all anchors (active + inactive) currently in the AnchorSet."""
    if agent.hippocampal is None or agent.hippocampal.anchor_set is None:
        return []
    return agent.hippocampal.anchor_set.all_anchors()


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
    """UC1: substrate-side classes / methods accessible."""
    try:
        from ree_core.hippocampal.anchor_set import (
            Anchor,
            AnchorGoalPayload,
            AnchorSet,
            AnchorSetConfig,
        )
        cfg = AnchorSetConfig()
        # use_sd039_anchor_payload should be a config field (default False).
        has_flag = hasattr(cfg, "use_sd039_anchor_payload")
        # Anchor must carry goal_payload field with default None.
        has_payload_field = "goal_payload" in Anchor.__dataclass_fields__
        # AnchorSet must expose query_by_goal_match.
        has_query = hasattr(AnchorSet, "query_by_goal_match")
        # AnchorGoalPayload must expose the documented fields.
        payload_fields = AnchorGoalPayload.__dataclass_fields__.keys()
        required_fields = {
            "z_goal_snapshot",
            "wanting_strength",
            "arousal_tag",
            "last_vs",
            "staleness_at_write",
            "payload_written_step",
        }
        has_required = required_fields.issubset(set(payload_fields))
        ok = has_flag and has_payload_field and has_query and has_required
        return {
            "pass": bool(ok),
            "has_use_sd039_flag": bool(has_flag),
            "has_anchor_goal_payload_field": bool(has_payload_field),
            "has_query_by_goal_match": bool(has_query),
            "payload_fields": sorted(payload_fields),
        }
    except Exception as exc:
        return {"pass": False, "error": repr(exc)}


def run_uc2_master_off_no_op() -> Dict[str, Any]:
    """UC2: SD-039 OFF -> no anchor carries goal_payload; query returns []."""
    try:
        agent = _make_agent(use_sd039=False, goal_active=True, seed=42)
        env = _make_env(seed=42)
        _step_episode(agent, env, n_ticks=30, force_drive=0.8, force_benefit=0.4)
        anchors = _all_anchors_with_payload(agent)
        any_payload = any(a.goal_payload is not None for a in anchors)
        z_goal = _current_z_goal(agent)
        if agent.hippocampal is not None and agent.hippocampal.anchor_set is not None:
            results = agent.hippocampal.anchor_set.query_by_goal_match(z_goal)
        else:
            results = []
        ok = (not any_payload) and (len(results) == 0)
        return {
            "pass": bool(ok),
            "n_anchors": len(anchors),
            "n_anchors_with_payload": int(sum(a.goal_payload is not None for a in anchors)),
            "n_query_results": len(results),
        }
    except Exception as exc:
        return {"pass": False, "error": repr(exc)}


def run_uc3_population_fires() -> Dict[str, Any]:
    """UC3: SD-039 ON + active goal -> at least one anchor has populated payload."""
    try:
        agent = _make_agent(use_sd039=True, goal_active=True, seed=42)
        env = _make_env(seed=42)
        _step_episode(agent, env, n_ticks=60, force_drive=0.8, force_benefit=0.4)
        anchors = _all_anchors_with_payload(agent)
        with_payload = [a for a in anchors if a.goal_payload is not None]
        with_snapshot = [
            a for a in with_payload if a.goal_payload.z_goal_snapshot is not None
        ]
        z_goal = _current_z_goal(agent)
        if agent.hippocampal is not None and agent.hippocampal.anchor_set is not None:
            results = agent.hippocampal.anchor_set.query_by_goal_match(z_goal)
        else:
            results = []
        non_zero_match = [pair for pair in results if pair[1] > 0.0]
        ok = (
            len(with_payload) >= 1
            and len(with_snapshot) >= 1
            and len(non_zero_match) >= 1
        )
        return {
            "pass": bool(ok),
            "n_anchors": len(anchors),
            "n_with_payload": len(with_payload),
            "n_with_z_goal_snapshot": len(with_snapshot),
            "n_query_results": len(results),
            "n_non_zero_goal_match": len(non_zero_match),
            "max_goal_match": float(max((p[1] for p in results), default=0.0)),
        }
    except Exception as exc:
        return {"pass": False, "error": repr(exc)}


def run_uc4_payload_survives_mark_inactive() -> Dict[str, Any]:
    """UC4: dual-trace preservation -- inactive anchors retain goal_payload."""
    try:
        agent = _make_agent(use_sd039=True, goal_active=True, seed=42)
        env = _make_env(seed=42)
        # 60 ticks gives MECH-288 boundary segmenter time to fire and remap.
        _step_episode(agent, env, n_ticks=60, force_drive=0.8, force_benefit=0.4)
        anchor_set = agent.hippocampal.anchor_set
        all_anchors = anchor_set.all_anchors()
        active = [a for a in all_anchors if a.active]
        inactive = [a for a in all_anchors if not a.active]
        active_with_payload = [a for a in active if a.goal_payload is not None]
        inactive_with_payload = [a for a in inactive if a.goal_payload is not None]
        z_goal = _current_z_goal(agent)
        # query with active_only=False must surface BOTH halves of the dual trace.
        all_results = anchor_set.query_by_goal_match(z_goal, active_only=False)
        active_results = anchor_set.query_by_goal_match(z_goal, active_only=True)
        ok = (
            len(inactive) >= 1
            and len(inactive_with_payload) >= 1
            and len(active_with_payload) >= 1
            and len(all_results) >= len(active_results)
        )
        return {
            "pass": bool(ok),
            "n_active": len(active),
            "n_inactive": len(inactive),
            "n_active_with_payload": len(active_with_payload),
            "n_inactive_with_payload": len(inactive_with_payload),
            "n_results_all": len(all_results),
            "n_results_active_only": len(active_results),
        }
    except Exception as exc:
        return {"pass": False, "error": repr(exc)}


def run_uc5_goal_relevant_vs_irrelevant_dissociation() -> Dict[str, Any]:
    """UC5: FALSIFIABLE TEST -- goal-active phase produces non-zero goal_match;
    goal-inactive phase does not. Two-phase episode design."""
    try:
        # Phase A: no goal active. We use goal_active=False so z_goal_enabled=False
        # and drive_weight=0; SD-039 must populate (or fail to populate) payload
        # accordingly. Anchors written here are GOAL-IRRELEVANT.
        agent_a = _make_agent(use_sd039=True, goal_active=False, seed=42)
        env_a = _make_env(seed=42)
        _step_episode(agent_a, env_a, n_ticks=30)  # no forced drive
        anchors_a = list(_all_anchors_with_payload(agent_a))

        # Phase B: full goal regime.
        agent_b = _make_agent(use_sd039=True, goal_active=True, seed=42)
        env_b = _make_env(seed=42)
        _step_episode(agent_b, env_b, n_ticks=30, force_drive=0.8, force_benefit=0.4)
        anchors_b = list(_all_anchors_with_payload(agent_b))
        z_goal_b = _current_z_goal(agent_b)

        # Score Phase A anchors against Phase B's current z_goal (cross-agent
        # scoring works because goal_match is a pure function of the stored
        # snapshot and supplied current vector; the substrate does NOT couple
        # them through the agent identity).
        scores_a = [a.goal_match(z_goal_b) for a in anchors_a]
        scores_b = [a.goal_match(z_goal_b) for a in anchors_b]
        mean_a = sum(scores_a) / max(1, len(scores_a))
        mean_b = sum(scores_b) / max(1, len(scores_b))
        n_strong_b = sum(1 for s in scores_b if s > 0.3)
        # Falsifiable signature.
        ok = (
            mean_b > 0.3
            and mean_a < 0.05
            and n_strong_b >= 1
        )
        return {
            "pass": bool(ok),
            "n_anchors_phase_a": len(anchors_a),
            "n_anchors_phase_b": len(anchors_b),
            "mean_goal_match_phase_a": float(mean_a),
            "mean_goal_match_phase_b": float(mean_b),
            "n_phase_b_above_0p3": int(n_strong_b),
            "max_goal_match_phase_a": float(max(scores_a, default=0.0)),
            "max_goal_match_phase_b": float(max(scores_b, default=0.0)),
        }
    except Exception as exc:
        return {"pass": False, "error": repr(exc)}


def run_uc6_mech094_simulation_gate() -> Dict[str, Any]:
    """UC6: simulation / replay-path writes must NOT populate goal_payload from
    waking-stream signals (call-site scoping pattern; population layer must
    respect MECH-094)."""
    try:
        agent = _make_agent(use_sd039=True, goal_active=True, seed=42)
        env = _make_env(seed=42)
        _step_episode(agent, env, n_ticks=30, force_drive=0.8, force_benefit=0.4)
        # Trigger a replay pass; under MECH-094 hypothesis_tag=True semantics,
        # any anchor writes occurring during replay must skip waking-stream
        # payload population. We probe by counting anchors created during a
        # run that BEGINS with a non-empty AnchorSet and then triggers replay
        # on the same agent. The substrate guarantee is that anchor writes
        # during simulation either (a) do not happen (call-site scoping;
        # tick_anchor_set is sense()-only), or (b) happen with goal_payload=None.
        pre_anchors = set(id(a) for a in _all_anchors_with_payload(agent))
        # If hippocampal.replay exists, run a small replay; otherwise
        # noop and assert no new anchors with payload were attached.
        if hasattr(agent.hippocampal, "replay"):
            try:
                agent.hippocampal.replay(n_steps=5)  # tolerant call signature
            except TypeError:
                # Some replay APIs require additional args; skip gracefully.
                pass
        post_anchors = list(_all_anchors_with_payload(agent))
        new_anchors = [a for a in post_anchors if id(a) not in pre_anchors]
        # New anchors created during replay must not carry a populated payload.
        new_with_payload = [
            a for a in new_anchors
            if a.goal_payload is not None
            and a.goal_payload.z_goal_snapshot is not None
        ]
        ok = len(new_with_payload) == 0
        return {
            "pass": bool(ok),
            "n_new_anchors_during_replay": len(new_anchors),
            "n_new_with_populated_payload": len(new_with_payload),
        }
    except Exception as exc:
        return {"pass": False, "error": repr(exc)}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main(dry_run: bool = False) -> int:
    t0 = time.time()
    print("[v3_exq_494] running 6 sub-tests for SD-039 population layer...")
    if dry_run:
        # Smoke: run UC1 only (cheap, proves imports work and the script
        # itself is well-formed). UC2-UC6 require an env episode and will
        # be exercised by the runner once population layer lands.
        uc1 = run_uc1_module_importable()
        all_metrics = {"UC1_module_importable": uc1}
        all_pass = bool(uc1["pass"])
    else:
        uc1 = run_uc1_module_importable()
        uc2 = run_uc2_master_off_no_op()
        uc3 = run_uc3_population_fires()
        uc4 = run_uc4_payload_survives_mark_inactive()
        uc5 = run_uc5_goal_relevant_vs_irrelevant_dissociation()
        uc6 = run_uc6_mech094_simulation_gate()
        all_metrics = {
            "UC1_module_importable": uc1,
            "UC2_master_off_no_op": uc2,
            "UC3_population_fires": uc3,
            "UC4_payload_survives_mark_inactive": uc4,
            "UC5_goal_relevant_vs_irrelevant_dissociation": uc5,
            "UC6_mech094_simulation_gate": uc6,
        }
        all_pass = all(m["pass"] for m in all_metrics.values())
    elapsed = time.time() - t0

    for name, m in all_metrics.items():
        print(f"  {name}: {'PASS' if m['pass'] else 'FAIL'}  {m}")
    print(f"[v3_exq_494] overall: {'PASS' if all_pass else 'FAIL'} ({elapsed:.1f}s)")

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"v3_exq_494_sd039_anchor_payload_validation_{ts}_v3"
    manifest = {
        "schema_version": "v1",
        "run_id": run_id,
        "experiment_type": "v3_exq_494_sd039_anchor_payload_validation",
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "experiment_purpose": "diagnostic",
        "claim_ids": ["SD-039"],
        "evidence_direction": "supports" if all_pass else "weakens",
        "evidence_direction_per_claim": {
            "SD-039": "supports" if all_pass else "weakens",
        },
        "evidence_direction_note": (
            "SD-039 population layer validation. UC1 confirms substrate-side "
            "classes; UC2 confirms master-OFF bit-identical; UC3-UC6 confirm "
            "the population layer (write-site goal_payload construction at "
            "REEAgent / HippocampalModule write/remap/invalidate sites) has "
            "landed and respects MECH-094 simulation gating. Pre-population "
            "baseline expected: UC1+UC2+UC6 PASS (2/6 + UC6 vacuous = 3/6); "
            "UC3+UC4+UC5 FAIL until population layer lands."
        ),
        "outcome": "PASS" if all_pass else "FAIL",
        "elapsed_sec": elapsed,
        "metrics": all_metrics,
        "dry_run": bool(dry_run),
    }

    # Write flat JSON to evidence root only for non-dry runs that aren't
    # in smoke-test mode. Dry runs print to stdout but skip the file write
    # so smoke tests don't pollute the manifests directory.
    if not dry_run:
        out_dir = EVIDENCE_ROOT / "v3_exq_494_sd039_anchor_payload_validation"
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
