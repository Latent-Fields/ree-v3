#!/opt/local/bin/python3
"""
V3-EXQ-480 -- MECH-284 Phase 3 param-sweep diagnostic (post EXQ-478 FAIL).

Claims: MECH-284, MECH-269

Purpose (diagnostic)
--------------------
EXQ-478 Phase 3 landing diagnostic FAILED with a characteristic pattern:
  - ON arm anchor_resets FIRE (63 and 31 across seeds 42 and 7)
  - ON arm mean_staleness_peak reaches 0.94 / 0.83
  - But freeze_recommit_count is bit-identical (1 vs 1) in ON vs OFF
  - BOTH arms show action_class_entropy = 0.0 (agent is monostrategy --
    all 1200 actions same class on both OFF and ON)

Read: Phase 3 is wired and observable (accumulator fills, resets fire)
but behaviourally inert in this env. Two orthogonal suspicions:

  1) Parameter tuning is wrong for this env. Candidate knobs:
       leak_factor (default 0.995)        -- decay per tick
       attribution_mode ("equal" default) -- credit assignment
       reset_threshold (default 0.3)      -- AnchorSet hysteresis gate
  2) The OFF baseline is already monostrategy (entropy=0.0). There is
     essentially nothing for anchor resets to perturb into differently-
     shaped behaviour -- the agent has only one action in its executed
     repertoire, so anchor churn cannot differentiate policy output.

This sweep controls for BOTH by:
  (a) scanning A2 (stream_overlap attribution), A3 (leak 0.99, faster
      decay -> less accumulation), A4 (leak 0.999, slower decay -> more
      accumulation) against A1 (ON default replicate) and A0 (OFF
      baseline), and
  (b) turning SD-029 balanced hazard curriculum ON for ALL arms so the
      OFF arm has externally-caused hazard events to elicit
      behavioural differentiation. This moves the OFF floor off
      monostrategy so anchor resets have something to bite into.

Sweep arms (5 arms x 2 seeds = 10 runs)
---------------------------------------
  A0 OFF              : V_s circuit OFF; curriculum ON.
  A1 ON_default       : V_s ON; leak=0.995, attribution=equal,
                        reset_threshold=0.3; curriculum ON. Replicates
                        EXQ-478 ON under curriculum.
  A2 ON_stream_overlap: V_s ON; leak=0.995; attribution_mode=
                        "stream_overlap"; reset_threshold=0.3;
                        curriculum ON.
  A3 ON_leak_099      : V_s ON; leak=0.99 (FASTER decay, LESS staleness
                        accumulation); attribution=equal;
                        reset_threshold=0.3; curriculum ON.
  A4 ON_leak_0999     : V_s ON; leak=0.999 (SLOWER decay, MORE staleness
                        accumulation); attribution=equal;
                        reset_threshold=0.3; curriculum ON.

(Note: leak_factor multiplies staleness each tick, so a LOWER value
corresponds to faster decay, i.e. less steady-state staleness; a
HIGHER value closer to 1.0 corresponds to slower decay, i.e. higher
steady-state staleness. The task spec phrasing was ambiguous; this
script uses the standard "higher leak -> more accumulation" convention
matching StalenessAccumulator.tick_leak().)

Environment
-----------
CausalGridWorldV2 WITH SD-029 scheduled external hazard curriculum ON
(scheduled_external_hazard_enabled=True, interval=50, prob=0.5,
adjacent_only=True). Other env parameters match EXQ-478 so the ON/OFF
contrast is apples-to-apples.

Metrics per run
---------------
  action_class_entropy       -- detects monostrategy (target: > 0 on
                                all arms after curriculum ON).
  freeze_recommit_count      -- primary: identical-consecutive-action
                                runs >= 3 steps. Does MECH-284 reduce it?
  anchor_reset_count         -- active->inactive transitions observed
                                via anchor_set.active_anchors() delta.
  mean_staleness_peak        -- per-episode max staleness, averaged.
  time_to_first_reset        -- step index of first observed
                                anchor_reset (None if none fired).
  staleness_at_first_reset   -- max staleness accumulator value at the
                                tick of the first reset (None if none).

Pass / fail rule
----------------
PASS iff at least one V_s-ON arm (A1 .. A4) produces
  freeze_recommit_count(arm) < freeze_recommit_count(A0 OFF)
in >= 2/2 seeds AND entropy(A0 OFF) > 0 (curriculum actually moved the
baseline off monostrategy).

If entropy(A0 OFF) == 0, the curriculum did not break monostrategy and
the diagnostic is INCONCLUSIVE regardless of the freeze comparison --
we cannot distinguish "MECH-284 is behaviourally ineffective" from
"there is nothing for it to perturb".

FAIL means Phase 3 is wired-but-coupled-wrong at the behaviour level
across all sweeps -- not a tuning issue but an architectural one;
revisit the anchor-reset -> action-selection pathway before adding
more sweeps.

experiment_purpose=diagnostic. Substrate-readiness gate. Gates Phase B.2
stub buildout for V3-EXQ-445d / 449c / 455a.

See:
  REE_assembly/docs/architecture/v_s_invalidation_runtime.md
  ree-v3/CLAUDE.md MECH-284 / MECH-269 Phase 3 section.
  REE_assembly/evidence/experiments/v3_exq_478_mech284_phase3_diagnostic/
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig
from experiments.pack_writer import write_flat_manifest  # noqa: E402


EXPERIMENT_TYPE = "v3_exq_480_mech284_phase3_param_sweep"
CLAIM_IDS = ["MECH-284", "MECH-269"]
EXPERIMENT_PURPOSE = "diagnostic"

SEEDS = [42, 7]
ARMS = [
    "A0_OFF",
    "A1_ON_default",
    "A2_ON_stream_overlap",
    "A3_ON_leak_099",
    "A4_ON_leak_0999",
]
EPISODES = 6
STEPS_PER_EP = 200
FREEZE_RUN_LEN = 3


# ------------------------------------------------------------------ #
# Arm -> param overrides                                             #
# ------------------------------------------------------------------ #
def _arm_params(arm: str) -> Dict:
    """Per-arm config overrides. Curriculum is ON for ALL arms."""
    vs_on = arm != "A0_OFF"
    base = {
        "vs_on": vs_on,
        "curriculum_on": True,
        "leak_factor": 0.995,
        "attribution_mode": "equal",
        "reset_threshold": 0.3,
        "hysteresis_k": 5,
    }
    if arm == "A0_OFF":
        return base
    if arm == "A1_ON_default":
        return base
    if arm == "A2_ON_stream_overlap":
        base["attribution_mode"] = "stream_overlap"
        return base
    if arm == "A3_ON_leak_099":
        base["leak_factor"] = 0.99
        return base
    if arm == "A4_ON_leak_0999":
        base["leak_factor"] = 0.999
        return base
    raise ValueError(f"Unknown arm: {arm}")


# ------------------------------------------------------------------ #
# Env + agent builders                                               #
# ------------------------------------------------------------------ #
def _make_env(seed: int, curriculum_on: bool) -> CausalGridWorldV2:
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
        scheduled_external_hazard_enabled=curriculum_on,
        scheduled_external_hazard_interval=50,
        scheduled_external_hazard_prob=0.5,
        scheduled_external_hazard_adjacent_only=True,
    )


def _make_agent(env: CausalGridWorldV2, params: Dict) -> REEAgent:
    vs_on = bool(params["vs_on"])
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
        # Full V_s invalidation circuit -- matched to EXQ-478 ON arm
        use_per_stream_vs=vs_on,
        use_event_segmenter=vs_on,
        use_invalidation_trigger=vs_on,
        use_anchor_sets=vs_on,
        use_per_region_vs=vs_on,
        use_staleness_accumulator=vs_on,
        use_mech284_hysteresis=vs_on,
    )

    # Apply sweep overrides directly to the HippocampalConfig substructures.
    # from_dims() does not expose these knobs; set them before REEAgent
    # construction so StalenessAccumulator / AnchorSet pick them up.
    if vs_on:
        cfg.hippocampal.staleness_accumulator.leak_factor = float(
            params["leak_factor"]
        )
        cfg.hippocampal.staleness_accumulator.attribution_mode = str(
            params["attribution_mode"]
        )
        cfg.hippocampal.anchor_set.reset_threshold = float(
            params["reset_threshold"]
        )
        cfg.hippocampal.anchor_set.hysteresis_k = int(params["hysteresis_k"])

    return REEAgent(cfg)


# ------------------------------------------------------------------ #
# Metric helpers                                                     #
# ------------------------------------------------------------------ #
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


def _count_freeze_runs(action_seq: List[int], run_len: int) -> int:
    if not action_seq:
        return 0
    runs = 0
    cur_val = action_seq[0]
    cur_len = 1
    for a in action_seq[1:]:
        if a == cur_val:
            cur_len += 1
        else:
            if cur_len >= run_len:
                runs += 1
            cur_val = a
            cur_len = 1
    if cur_len >= run_len:
        runs += 1
    return runs


# ------------------------------------------------------------------ #
# Per-condition run                                                  #
# ------------------------------------------------------------------ #
def _run_arm(seed: int, arm: str) -> Dict:
    params = _arm_params(arm)
    torch.manual_seed(seed)
    env = _make_env(seed, params["curriculum_on"])
    agent = _make_agent(env, params)

    action_counts: Dict[int, int] = {}
    action_seq: List[int] = []
    n_ticks = 0
    anchor_reset_count = 0
    prev_active_keys: set = set()
    staleness_peaks: List[float] = []

    time_to_first_reset: Optional[int] = None
    staleness_at_first_reset: Optional[float] = None

    for _ep in range(EPISODES):
        obs, _info = env.reset()
        ep_peak = 0.0
        prev_active_keys = set()
        for _step in range(STEPS_PER_EP):
            action = agent.act(obs)
            a_idx = int(action[0].argmax().item())
            action_counts[a_idx] = action_counts.get(a_idx, 0) + 1
            action_seq.append(a_idx)
            obs, _harm, done, _info, _obs_dict = env.step(a_idx)
            n_ticks += 1

            if params["vs_on"]:
                hc = agent.hippocampal
                anchor_set = getattr(hc, "anchor_set", None)
                sa = getattr(hc, "staleness_accumulator", None)
                if anchor_set is not None:
                    active_now = {a.key for a in anchor_set.active_anchors()}
                    gone = prev_active_keys - active_now
                    n_gone = len(gone)
                    if n_gone > 0 and time_to_first_reset is None:
                        time_to_first_reset = n_ticks
                        if sa is not None:
                            snap_now = sa.snapshot()
                            staleness_at_first_reset = (
                                float(max(snap_now.values()))
                                if snap_now else 0.0
                            )
                    anchor_reset_count += n_gone
                    prev_active_keys = active_now
                if sa is not None:
                    snap = sa.snapshot()
                    if snap:
                        ep_peak = max(ep_peak, max(snap.values()))

            if done:
                break

        if params["vs_on"]:
            staleness_peaks.append(ep_peak)

    mean_staleness_peak = (
        sum(staleness_peaks) / len(staleness_peaks)
        if staleness_peaks else 0.0
    )

    return {
        "arm": arm,
        "seed": seed,
        "params": {
            k: v for k, v in params.items() if k not in ("vs_on",)
        },
        "action_class_entropy": _shannon_entropy(action_counts),
        "action_class_counts": action_counts,
        "n_actions": sum(action_counts.values()),
        "n_ticks": n_ticks,
        "freeze_recommit_count": _count_freeze_runs(action_seq, FREEZE_RUN_LEN),
        "anchor_reset_count": anchor_reset_count,
        "mean_staleness_peak": mean_staleness_peak,
        "time_to_first_reset": time_to_first_reset,
        "staleness_at_first_reset": staleness_at_first_reset,
    }


# ------------------------------------------------------------------ #
# Plan / smoke                                                        #
# ------------------------------------------------------------------ #
def _print_plan() -> None:
    print(
        f"{EXPERIMENT_TYPE} -- MECH-284 Phase 3 param-sweep diagnostic",
        flush=True,
    )
    print(f"Arms: {ARMS}", flush=True)
    print(f"Seeds: {SEEDS}", flush=True)
    print(f"Episodes x steps_per_ep: {EPISODES} x {STEPS_PER_EP}", flush=True)
    print(f"Total runs: {len(ARMS) * len(SEEDS)}", flush=True)
    print(
        "Metrics: freeze_recommit_count, anchor_reset_count, "
        "mean_staleness_peak, action_class_entropy, "
        "time_to_first_reset, staleness_at_first_reset",
        flush=True,
    )
    print(
        "PASS iff entropy(A0_OFF) > 0 AND at least one ON arm produces "
        "freeze_recommit_count(arm) < freeze_recommit_count(A0_OFF) in "
        ">= 2/2 seeds. Otherwise FAIL or INCONCLUSIVE.",
        flush=True,
    )
    print(f"experiment_purpose={EXPERIMENT_PURPOSE}", flush=True)


# ------------------------------------------------------------------ #
# Main                                                                #
# ------------------------------------------------------------------ #
def main() -> int:
    parser = argparse.ArgumentParser(
        description=f"{EXPERIMENT_TYPE} MECH-284 Phase 3 param sweep"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print plan and exit 0; do not execute.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run one arm (A0_OFF) one seed one episode 20 steps; exit 0.",
    )
    args = parser.parse_args()

    if args.dry_run:
        _print_plan()
        print("DRY RUN OK", flush=True)
        return 0

    if args.smoke:
        global EPISODES, STEPS_PER_EP
        EPISODES = 1
        STEPS_PER_EP = 20
        print("SMOKE: running A0_OFF seed 42 1 ep 20 steps", flush=True)
        r = _run_arm(seed=42, arm="A0_OFF")
        print(
            f"  -> entropy={r['action_class_entropy']:.4f} "
            f"freeze_runs={r['freeze_recommit_count']} "
            f"resets={r['anchor_reset_count']} "
            f"staleness_peak={r['mean_staleness_peak']:.4f} "
            f"n_ticks={r['n_ticks']}",
            flush=True,
        )
        print("SMOKE OK", flush=True)
        return 0

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"
    out_dir = (
        Path(__file__).resolve().parents[2]
        / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results: List[Dict] = []
    for seed in SEEDS:
        for arm in ARMS:
            print(f"Seed {seed} Arm {arm}", flush=True)
            r = _run_arm(seed=seed, arm=arm)
            print(
                f"  -> entropy={r['action_class_entropy']:.4f} "
                f"freeze={r['freeze_recommit_count']} "
                f"resets={r['anchor_reset_count']} "
                f"peak={r['mean_staleness_peak']:.4f} "
                f"first_reset={r['time_to_first_reset']} "
                f"stale@reset={r['staleness_at_first_reset']} "
                f"n_ticks={r['n_ticks']}",
                flush=True,
            )
            all_results.append(r)

    # Build per-seed, per-arm lookup for the gate.
    by_seed: Dict[int, Dict[str, Dict]] = {s: {} for s in SEEDS}
    for r in all_results:
        by_seed[int(r["seed"])][str(r["arm"])] = r

    # OFF-baseline entropy gate.
    off_entropies = [by_seed[s]["A0_OFF"]["action_class_entropy"] for s in SEEDS]
    off_entropy_gate_ok = all(e > 0.0 for e in off_entropies)

    # Per ON-arm: does it beat OFF freeze in >= 2/2 seeds?
    on_arms = [a for a in ARMS if a != "A0_OFF"]
    per_arm_gate: List[Dict] = []
    winning_arms: List[str] = []
    for arm in on_arms:
        seed_beats: List[Tuple[int, int, int, bool]] = []
        for s in SEEDS:
            off_fr = by_seed[s]["A0_OFF"]["freeze_recommit_count"]
            on_fr = by_seed[s][arm]["freeze_recommit_count"]
            seed_beats.append((s, off_fr, on_fr, on_fr < off_fr))
        seeds_beating = sum(1 for _, _, _, b in seed_beats if b)
        arm_wins = seeds_beating >= len(SEEDS)
        if arm_wins:
            winning_arms.append(arm)
        per_arm_gate.append(
            {
                "arm": arm,
                "per_seed": [
                    {
                        "seed": s,
                        "off_freeze": off_fr,
                        "on_freeze": on_fr,
                        "beats_off": b,
                    }
                    for (s, off_fr, on_fr, b) in seed_beats
                ],
                "seeds_beating": seeds_beating,
                "arm_wins_gate": arm_wins,
            }
        )

    if not off_entropy_gate_ok:
        outcome = "FAIL"
        outcome_note = (
            "OFF baseline entropy is 0 across seeds -- curriculum did not "
            "break monostrategy; MECH-284 sweep cannot be scored because "
            "there is nothing for anchor resets to perturb. Diagnostic "
            "inconclusive at the behavioural layer."
        )
    elif winning_arms:
        outcome = "PASS"
        outcome_note = (
            f"Arms beating OFF freeze in 2/2 seeds: {winning_arms}. "
            "Phase 3 is behaviourally effective at the winning arm's "
            "tuning; use that tuning for downstream V3-EXQ-445d / 449c / "
            "455a stub buildouts."
        )
    else:
        outcome = "FAIL"
        outcome_note = (
            "Curriculum broke OFF monostrategy but NO V_s-ON arm beat the "
            "OFF freeze_recommit_count in 2/2 seeds across the "
            "attribution/leak sweep. Suggests Phase 3 is wired-but-"
            "coupled-wrong at the behaviour level, not a tuning issue. "
            "Next step: instrument anchor-reset -> action-selection "
            "pathway rather than continuing to sweep params."
        )

    summary = {
        "off_entropy_gate_ok": off_entropy_gate_ok,
        "off_entropies_by_seed": dict(zip([str(s) for s in SEEDS], off_entropies)),
        "per_arm_gate": per_arm_gate,
        "winning_arms": winning_arms,
        "outcome": outcome,
        "outcome_note": outcome_note,
    }

    print(f"\nOutcome: {outcome}", flush=True)
    print(f"Note: {outcome_note}", flush=True)
    for row in per_arm_gate:
        print(
            f"  arm={row['arm']} wins={row['arm_wins_gate']} "
            f"seeds_beating={row['seeds_beating']}/{len(SEEDS)}",
            flush=True,
        )

    # Build per-claim, per-arm evidence direction breakdown.
    # Whole-experiment direction: supports iff PASS, else inconclusive
    # (FAIL with null result is still "inconclusive" for the claims --
    # we learned the substrate is wired-but-behaviourally-inert, not
    # that the claim is wrong).
    exp_dir = "supports" if outcome == "PASS" else "inconclusive"
    per_claim = {cid: exp_dir for cid in CLAIM_IDS}
    per_arm_per_claim: Dict[str, Dict[str, str]] = {}
    for row in per_arm_gate:
        arm = row["arm"]
        dir_ = "supports" if row["arm_wins_gate"] else "inconclusive"
        per_arm_per_claim[arm] = {cid: dir_ for cid in CLAIM_IDS}

    output = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": ts,
        "supersedes": None,
        "outcome": outcome,
        "evidence_direction": exp_dir,
        "evidence_direction_per_claim": per_claim,
        "evidence_direction_per_arm_per_claim": per_arm_per_claim,
        "evidence_direction_note": (
            "MECH-284 Phase 3 parameter sweep after EXQ-478 FAIL "
            "(accumulator populates, resets fire, freeze unchanged, "
            "entropy=0 on both arms). This sweep adds SD-029 curriculum "
            "ON for all arms (to break OFF monostrategy) and varies "
            "attribution_mode + leak_factor. Substrate-readiness "
            "diagnostic; not governance evidence. Successor to EXQ-478 "
            "for Phase 3 behavioural validation. Gates V3-EXQ-445d / "
            "449c / 455a Phase B.2 stub buildouts."
        ),
        "pass_criteria_summary": summary,
        "per_arm_results": all_results,
        "config": {
            "arms": ARMS,
            "seeds": SEEDS,
            "episodes": EPISODES,
            "steps_per_ep": STEPS_PER_EP,
            "freeze_run_len": FREEZE_RUN_LEN,
            "curriculum_on": True,
            "curriculum_interval": 50,
            "curriculum_prob": 0.5,
        },
    }

    out_file = write_flat_manifest(
        output,
        out_dir,
        dry_run=False,
        config=output.get("config"),
        seeds=SEEDS,
        script_path=Path(__file__),
    )
    print(f"Result written to: {out_file}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
