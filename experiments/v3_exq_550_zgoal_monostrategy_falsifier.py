#!/opt/local/bin/python3
"""
V3-EXQ-550 -- z_goal-enabled MECH-269 monostrategy falsifier.

Claims: MECH-269 (diagnostic only; no claim weighting)

Purpose (diagnostic, not evidence)
----------------------------------
Falsifier for the hypothesis that the MECH-269 V_s "monostrategy"
diagnosis is a substrate-level finding, by ruling out a competing
config-default explanation.

Audit finding (2026-05-11):
    GoalConfig.z_goal_enabled defaults to False, and
    REEConfig.from_dims() also defaults z_goal_enabled to False. Every
    EXQ-476/476a/476b/476c run (and EXQ-433*, EXQ-470*) that fed the
    MECH-269 "monomodal policy" diagnosis was constructed without
    explicitly passing z_goal_enabled=True. The agent therefore had
    no structured approach drive in those runs -- only harm avoidance.
    The "agent collapses to a single action class" observation could
    plausibly reflect the absence of any competing drive state at the
    config layer, not a substrate-level monostrategy property.

This experiment is the cheap falsifier. It mirrors the EXQ-476c
entropy-probe structure (no training, V_s ON in both arms) and flips
z_goal_enabled between arms:

    ARM_OFF: z_goal_enabled=False
             (replicates EXQ-476c baseline; approach drive OFF)
    ARM_ON:  z_goal_enabled=True, drive_weight=2.0
             (SD-012 validated default goal pipeline activated; per-step
              update_z_goal call fed from info['benefit_exposure']
              so GoalState evolves rather than staying at zero)

Both arms hold the V_s invalidation circuit ON (per-stream_vs +
per_region_vs + event_segmenter + invalidation_trigger + anchor_sets)
and run on SD-029 scheduled-hazard-enabled CausalGridWorldV2. The
only manipulated variable is the z_goal master switch + its minimum
supporting hooks (drive_weight + per-step update_z_goal feed).

Metric and gate rule
--------------------
    action_class_entropy per arm per seed (Shannon entropy over
    executed action-class histogram across the run).

    PASS = ARM_ON - ARM_OFF >= 0.10 in >= 2/3 seeds.
        -> z_goal being wired alone meaningfully diversifies behaviour.
           Config-default hypothesis SUPPORTED. MECH-269 V_s
           cluster's "monomodal policy" hold on SD-029 must be
           revisited with z_goal-on runs before substrate claims
           can be sustained.

    FAIL = ARM_ON does not clear ARM_OFF by >= 0.10 in 2/3 seeds.
        -> config-default hypothesis WEAKENED. The wired-but-untrained
           goal pipeline does not break monostrategy on its own.
           Substrate diagnosis stands at this probe depth, BUT a
           negative result at no-training depth does not falsify the
           full strong reading -- a trained follow-up (matched P0
           warmup + ARM_ON vs ARM_OFF) is required before drawing
           a hard "substrate-level monostrategy" conclusion.

Three-row interpretation grid (per user 2026-05-11 message)
----------------------------------------------------------
    (i)   ARM_ON entropy >> ARM_OFF -> V_s monostrategy signature
          dissolves at the no-training depth; MECH-269 V_s cluster
          needs revisiting; SD-029 unblocked (or at least re-tested).
    (ii)  ARM_ON entropy ~= ARM_OFF -> V_s monostrategy signature
          persists despite z_goal being wired; config-default ruled
          OUT as the dominant cause at this probe depth; substrate
          diagnosis stands pending trained follow-up.
    (iii) ARM_ON arm crashes or produces incoherent agent (NaNs,
          all-zero action distribution, exception) -> reveals a
          separate substrate bug in the goal pipeline that itself
          needs addressing; treat as ERROR not FAIL.

experiment_purpose=diagnostic. No claim weighting. Result feeds
the audit decision in MEMORY.md SD-029 / MECH-269 entries.

Caveat (honest scoping)
-----------------------
No-training probe inherits EXQ-476c limitations:
  - resource_proximity_head is random, so update_z_goal seeds z_goal
    from a noisy benefit signal. z_goal will be non-zero and time-
    varying, but its semantic alignment with "approach the right
    cell" is no better than chance. The test is whether having
    *any* second scoring axis on E3 (even an untrained one) breaks
    the action-class collapse, not whether goal-directed behaviour
    is recovered.
  - PASS is a strong signal that z_goal-being-wired alone matters.
  - FAIL is a weaker signal -- it does not rule out that a TRAINED
    z_goal pipeline would diversify. A trained-arm follow-up is the
    natural successor experiment.

See REE_assembly/docs/architecture/v_s_invalidation_runtime.md
See ree-v3/CLAUDE.md MECH-269 / SD-012 / SD-029 sections.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from experiment_protocol import emit_outcome
from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_550_zgoal_monostrategy_falsifier"
QUEUE_ID = "V3-EXQ-550"
CLAIM_IDS = ["MECH-269"]
EXPERIMENT_PURPOSE = "diagnostic"

SEEDS = [42, 7, 17]
CONDITIONS = ["OFF", "ON"]
EPISODES = 6
STEPS_PER_EP = 200
ENTROPY_DELTA_THRESHOLD = 0.10
SEEDS_REQUIRED_TO_PASS = 2  # at least 2 of 3 seeds must clear the threshold


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed,
        size=10,
        num_hazards=2,
        num_resources=3,
        hazard_harm=0.04,
        proximity_harm_scale=0.12,
        proximity_benefit_scale=0.18,
        hazard_field_decay=0.5,
        energy_decay=0.005,
        use_proxy_fields=True,
        resource_respawn_on_consume=True,
        harm_history_len=10,
        # SD-029 scheduled-hazard curriculum (matches user spec)
        scheduled_external_hazard_enabled=True,
        scheduled_external_hazard_interval=50,
        scheduled_external_hazard_prob=0.5,
        scheduled_external_hazard_adjacent_only=True,
    )


def _make_agent(env: CausalGridWorldV2, condition: str) -> REEAgent:
    goal_on = condition == "ON"
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=32,
        alpha_world=0.9,
        alpha_self=0.3,
        use_harm_stream=True,
        harm_obs_dim=51,
        use_affective_harm_stream=True,
        harm_obs_a_dim=50,
        harm_history_len=10,
        z_harm_a_dim=16,
        # ---- The single manipulated variable ----
        # ARM_OFF: approach drive entirely absent (replicates EXQ-476c
        #          baseline + the EXQ-476* config-default condition).
        # ARM_ON:  z_goal master switch ON + SD-012 drive_weight=2.0.
        #          GoalState is constructed; goal_proximity term enters
        #          E3 trajectory scoring; per-step update_z_goal feed is
        #          activated in the probe loop below.
        z_goal_enabled=goal_on,
        drive_weight=(2.0 if goal_on else 0.0),
        e1_goal_conditioned=goal_on,
        goal_weight=(0.5 if goal_on else 0.0),
        # Matched V_s invalidation circuit (held ON in both arms,
        # mirrors EXQ-476c ARM_ON wiring + adds anchor_set wiring already
        # present in EXQ-476c). This is the "matched MECH-269 V_s
        # circuit" the user asked for.
        use_per_stream_vs=True,
        use_per_region_vs=True,
        use_event_segmenter=True,
        use_invalidation_trigger=True,
        use_anchor_sets=True,
    )
    return REEAgent(cfg)


def _shannon_entropy(counts: Dict[int, int]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    ent = 0.0
    for c in counts.values():
        if c <= 0:
            continue
        p = c / total
        ent -= p * math.log(p)
    return ent


def _run_condition(seed: int, condition: str) -> Dict:
    torch.manual_seed(seed)
    env = _make_env(seed)
    agent = _make_agent(env, condition)

    action_counts: Dict[int, int] = {}
    boundary_event_count = 0
    broadcast_event_count = 0
    anchor_active_peak = 0
    z_goal_norm_peak = 0.0
    z_goal_update_calls = 0
    n_ticks = 0
    n_nans = 0

    # iii: detect incoherent-agent path early. If select_action returns
    # NaN or non-finite values we bail out and surface as ERROR upstream.
    error_note = None

    for ep in range(EPISODES):
        _, obs_dict = env.reset()
        agent.reset()
        for _step in range(STEPS_PER_EP):
            body = obs_dict["body_state"].float().unsqueeze(0)
            world = obs_dict["world_state"].float().unsqueeze(0)
            harm = obs_dict.get("harm_obs")
            if harm is not None:
                harm = harm.float().unsqueeze(0)
            harm_a = obs_dict.get("harm_obs_a")
            if harm_a is not None:
                harm_a = harm_a.float().unsqueeze(0)
            harm_hist = obs_dict.get("harm_history")
            if harm_hist is not None:
                harm_hist = harm_hist.float().unsqueeze(0)
            latent = agent.sense(
                obs_body=body, obs_world=world,
                obs_harm=harm, obs_harm_a=harm_a, obs_harm_history=harm_hist,
            )
            ticks = agent.clock.advance()
            wdim = latent.z_world.shape[-1]
            e1_prior = (
                agent._e1_tick(latent) if ticks.get("e1_tick", False)
                else torch.zeros(1, wdim, device=agent.device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            action = agent.select_action(candidates, ticks)

            if not torch.isfinite(action).all():
                n_nans += 1
                if error_note is None:
                    error_note = (
                        f"non-finite action at seed={seed} cond={condition} "
                        f"ep={ep} step={_step}"
                    )
                break

            a_idx = int(action[0].argmax().item())
            action_counts[a_idx] = action_counts.get(a_idx, 0) + 1
            _, _harm_signal, done, info, obs_dict = env.step(action)
            n_ticks += 1

            # ARM_ON: feed update_z_goal so GoalState evolves rather than
            # staying at zero. drive_level proxy = 1 - energy (per
            # ree_core.agent.REEAgent.compute_drive_level convention,
            # body[3] is energy in proxy mode).
            if condition == "ON" and agent.goal_state is not None:
                benefit_exposure = float(info.get("benefit_exposure", 0.0))
                energy = float(body[0, 3].item())
                drive_level = max(0.0, 1.0 - energy)
                agent.update_z_goal(
                    benefit_exposure=benefit_exposure,
                    drive_level=drive_level,
                )
                z_goal_update_calls += 1
                z_goal_norm_peak = max(
                    z_goal_norm_peak, float(agent.goal_state.goal_norm())
                )

            # Diagnostic substrate counters (kept identical to EXQ-476c so
            # the two probes are directly comparable). Both arms have V_s
            # ON, so we always populate these.
            hc = agent.hippocampal
            be_q = getattr(hc, "_boundary_event_queue", None)
            if be_q is not None:
                boundary_event_count += len(be_q)
            br_q = getattr(hc, "_broadcast_event_queue", None)
            if br_q is not None:
                broadcast_event_count += len(br_q)
            anchor_set = getattr(hc, "anchor_set", None)
            if anchor_set is not None:
                anchor_active_peak = max(
                    anchor_active_peak, len(anchor_set.active_anchors())
                )

            if done:
                break

        if error_note is not None:
            break

    return {
        "condition": condition,
        "seed": seed,
        "action_class_entropy": _shannon_entropy(action_counts),
        "action_class_counts": action_counts,
        "n_actions": sum(action_counts.values()),
        "n_ticks": n_ticks,
        "n_nans": n_nans,
        "error_note": error_note,
        "boundary_event_count": boundary_event_count,
        "broadcast_event_count": broadcast_event_count,
        "anchor_active_peak": anchor_active_peak,
        "z_goal_update_calls": z_goal_update_calls,
        "z_goal_norm_peak": z_goal_norm_peak,
    }


def _print_plan() -> None:
    print(f"{EXPERIMENT_TYPE} -- z_goal monostrategy falsifier", flush=True)
    print(f"Arms: {CONDITIONS}", flush=True)
    print(f"Seeds: {SEEDS}", flush=True)
    print(f"Episodes x steps_per_ep: {EPISODES} x {STEPS_PER_EP}", flush=True)
    print(f"Env: CausalGridWorldV2 + SD-029 scheduled hazards ON", flush=True)
    print(f"Metric: action_class_entropy per arm per seed", flush=True)
    print(
        f"PASS = ON - OFF >= {ENTROPY_DELTA_THRESHOLD} in >= "
        f"{SEEDS_REQUIRED_TO_PASS}/{len(SEEDS)} seeds "
        "-> z_goal-wired alone breaks monostrategy",
        flush=True,
    )
    print(
        "FAIL -> config default ruled out at no-training depth; substrate "
        "diagnosis stands pending trained follow-up",
        flush=True,
    )
    print(f"experiment_purpose={EXPERIMENT_PURPOSE}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=f"{EXPERIMENT_TYPE} z_goal monostrategy falsifier"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print plan and exit 0; do not execute.",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="1 seed x 1 episode x 20 steps smoke test (no manifest written).",
    )
    args = parser.parse_args()

    if args.dry_run:
        _print_plan()
        print("DRY RUN OK", flush=True)
        # Dry-run is a developer hook; no manifest written, no emit_outcome.
        return (None, None)

    if args.smoke:
        global EPISODES, STEPS_PER_EP, SEEDS
        EPISODES = 1
        STEPS_PER_EP = 20
        SEEDS = [42]
        print("SMOKE MODE: 1 seed x 1 ep x 20 steps; no manifest write",
              flush=True)
        for cond in CONDITIONS:
            print(f"Seed {SEEDS[0]} Condition {cond}", flush=True)
            r = _run_condition(seed=SEEDS[0], condition=cond)
            print(
                f"  [train] label seed={SEEDS[0]} ep 1/1 "
                f"entropy={r['action_class_entropy']:.4f} "
                f"n_actions={r['n_actions']}",
                flush=True,
            )
            print(f"verdict: {'PASS' if r['error_note'] is None else 'FAIL'}",
                  flush=True)
        print("SMOKE OK", flush=True)
        return (None, None)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"
    out_dir = (
        Path(__file__).resolve().parents[2]
        / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results: List[Dict] = []
    any_error = False
    for seed in SEEDS:
        for cond in CONDITIONS:
            print(f"Seed {seed} Condition {cond}", flush=True)
            r = _run_condition(seed=seed, condition=cond)
            print(
                f"  [train] label seed={seed} ep {EPISODES}/{EPISODES} "
                f"entropy={r['action_class_entropy']:.4f} "
                f"n_actions={r['n_actions']} "
                f"boundaries={r['boundary_event_count']} "
                f"broadcasts={r['broadcast_event_count']} "
                f"anchor_peak={r['anchor_active_peak']} "
                f"z_goal_calls={r['z_goal_update_calls']} "
                f"z_goal_norm_peak={r['z_goal_norm_peak']:.4f}",
                flush=True,
            )
            if r["error_note"] is not None:
                any_error = True
                print(f"  ERROR: {r['error_note']}", flush=True)
                print(f"verdict: FAIL", flush=True)
            else:
                print(f"verdict: PASS", flush=True)
            all_results.append(r)

    off_by_seed = {r["seed"]: r for r in all_results if r["condition"] == "OFF"}
    on_by_seed = {r["seed"]: r for r in all_results if r["condition"] == "ON"}

    per_seed_delta = []
    seeds_passing = 0
    for seed in SEEDS:
        off_e = off_by_seed[seed]["action_class_entropy"]
        on_e = on_by_seed[seed]["action_class_entropy"]
        delta = on_e - off_e
        cleared = delta >= ENTROPY_DELTA_THRESHOLD
        per_seed_delta.append({
            "seed": seed,
            "off_entropy": off_e,
            "on_entropy": on_e,
            "delta": delta,
            "cleared": cleared,
        })
        if cleared:
            seeds_passing += 1

    if any_error:
        outcome = "FAIL"  # ERROR pathway -- runner classifies via exception only
        evidence_direction = "inconclusive"
    elif seeds_passing >= SEEDS_REQUIRED_TO_PASS:
        outcome = "PASS"
        evidence_direction = "weakens"   # falsifier PASS WEAKENS the substrate-level reading
    else:
        outcome = "FAIL"
        evidence_direction = "supports"  # falsifier FAIL SUPPORTS the substrate-level reading

    summary = {
        "gate_rule": (
            f"action_class_entropy(ON) - action_class_entropy(OFF) "
            f">= {ENTROPY_DELTA_THRESHOLD} in >= "
            f"{SEEDS_REQUIRED_TO_PASS}/{len(SEEDS)} seeds"
        ),
        "per_seed_delta": per_seed_delta,
        "seeds_passing": seeds_passing,
        "seeds_required": SEEDS_REQUIRED_TO_PASS,
        "pass": outcome == "PASS",
        "any_error_seed": any_error,
    }

    print(f"\nOutcome: {outcome}", flush=True)
    for row in per_seed_delta:
        print(
            f"  seed={row['seed']} off={row['off_entropy']:.4f} "
            f"on={row['on_entropy']:.4f} delta={row['delta']:.4f} "
            f"cleared={row['cleared']}",
            flush=True,
        )

    # Per-claim direction: single claim so per-claim collapses to the
    # overall direction. Diagnostic experiment -- result feeds an audit
    # decision, not direct claim weighting (see EXPERIMENT_PURPOSE).
    per_claim = {cid: evidence_direction for cid in CLAIM_IDS}

    output = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": ts,
        "outcome": outcome,
        "evidence_direction": evidence_direction,
        "evidence_direction_per_claim": per_claim,
        "evidence_direction_note": (
            "Diagnostic falsifier for the MECH-269 V_s monostrategy "
            "diagnosis. PASS means z_goal-being-wired alone shifts "
            "action-class entropy materially -- supports the "
            "config-default explanation and WEAKENS the substrate-level "
            "reading of MECH-269's hold on SD-029. FAIL means the "
            "wired-but-untrained goal pipeline does NOT break "
            "monostrategy at this probe depth -- the substrate-level "
            "reading SURVIVES at this depth (does NOT rule out that a "
            "trained z_goal pipeline would change the picture). ERROR-"
            "branch (non-finite action) reveals a separate substrate "
            "bug in the goal pipeline activation path and is reported "
            "via error_note + any_error_seed in pass_criteria_summary."
        ),
        "pass_criteria_summary": summary,
        "per_seed_results": all_results,
        "config": {
            "conditions": CONDITIONS,
            "seeds": SEEDS,
            "episodes": EPISODES,
            "steps_per_ep": STEPS_PER_EP,
            "entropy_delta_threshold": ENTROPY_DELTA_THRESHOLD,
            "seeds_required_to_pass": SEEDS_REQUIRED_TO_PASS,
            "sd029_scheduled_hazards": True,
            "v_s_circuit_in_both_arms": True,
        },
    }

    out_file = out_dir / f"{run_id}.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Output written to: {out_file}", flush=True)

    return (outcome, str(out_file))


if __name__ == "__main__":
    _outcome, _manifest_path = main()
    if _outcome is not None and _manifest_path is not None:
        emit_outcome(outcome=_outcome, manifest_path=_manifest_path)
    sys.exit(0)
