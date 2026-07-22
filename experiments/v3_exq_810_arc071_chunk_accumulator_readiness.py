"""V3-EXQ-810 -- ARC-071 / MECH-323 substrate-readiness: does the chunking accumulator fire?

PURPOSE (diagnostic / substrate-readiness validation, NOT governance evidence).
ARC-071 policy_composition_via_repeated_grounding was implemented 2026-07-22 as
MECH-323 formation (ChunkAccumulator) + MECH-324 maintenance (ChunkLibrary) +
the MECH-322 sleep-replay carve-out, all default OFF. Nothing has ever run against
it. This is the first question in the registered validation order, and deliberately
the SMALLEST one: does the accumulator fire AT ALL when driven by the whole agent?

WHAT THIS DOES *NOT* CLAIM. Behavioural latency and rollout-cost reduction are
ARC-071's headline predictions, and they are NOT measured here. Those need chunks in
the proposal pool (use_chunk_proposal_injection) plus the ARM_1-vs-ARM_2 contrast
that isolates MECH-324, per MECH-324's own registered three-arm design. Proposal
injection is OFF in every arm of this run, so E3 never sees a chunk and the action
stream is unaffected by chunk formation. That is intentional: a readiness probe must
not also perturb the behaviour it would later measure.

ARMS (3, same seeds), all with use_chunk_proposal_injection=False:
  ARM_OFF  use_policy_chunking=False -- the operators are never instantiated.
  ARM_FORM use_policy_chunking=True, use_chunk_maintenance=False -- MECH-323 alone.
           The registered ARM_1 dissociation: chunks must FORM but never crystallise.
  ARM_FULL use_policy_chunking=True, use_chunk_maintenance=True -- MECH-323+324.

THE TASK MUST SUPPLY OUTCOME CONTRAST, NOT MERELY REPETITION. MECH-323's evaluative
gate is RELATIVE -- a sub-sequence must beat the agent's RUNNING BASELINE by a margin.
A regime in which every trial scores the same forms nothing however often a sequence
repeats (pinned by contract test_c6_uniform_outcomes_form_nothing). So the outcome
signal here is per-episode resource collection on a resource-bearing grid, which
genuinely varies episode to episode. A run that formed nothing because every episode
scored identically would be a TASK defect, not a substrate ceiling -- which is exactly
why the outcome spread is a measured precondition below rather than an assumption.

FAILURE RECORD: none. ARC-071 has no prior experiment; this is the first. The
acceptance criteria therefore come from the registered MECH-323 / MECH-324 designs
rather than from a FAIL to be moved.

GOV-REUSE-1 (Step 2.4): the decisive readout is chunk_acc_n_formed (and
chunk_lib_n_crystallised) under use_policy_chunking=True. That config knob and those
readouts did not exist before 2026-07-22, so no recorded manifest on any
substrate_hash can carry them -- there is no prior ARC-071/MECH-323/MECH-324 run in
evidence/experiments/ at all. Not recoverable -> run.

MECH-094 (SAFETY-CRITICAL). This run also serves as the live-agent check on the
strict gate: the waking path is the only writer, so chunk_acc_n_replay_formed must be
0 in every arm, and use_chunk_replay_origin_path is False everywhere. C4 pins it. The
bench contracts pin the same property at module level; this pins it through agent.py.

DV-SYMMETRY (Step 3 mandatory declaration, per arm):
  ARM_OFF DV = chunk_acc_n_formed, structurally 0 (no operator exists). Not a
    measurement and not claimed as one -- it is the inertness control for C3.
  ARM_FORM / ARM_FULL DV = chunk_acc_n_formed and chunk_lib_n_crystallised, which are
    COUNTS over the minted chunk set. Symmetry group of a count: permutation of the
    tallied sub-sequences and any relabelling of action-class indices (a count is a
    symmetric function of its set). The manipulation is NOT invariant under it: turning
    the accumulator on changes the CARDINALITY of the minted set from 0, and cardinality
    is not preserved by any permutation of set members. The ARM_FORM-vs-ARM_FULL
    manipulation (maintenance on/off) likewise changes the cardinality of the
    CRYSTALLISED subset, not the labelling of its members.
  Note the manipulation is NOT a broadcast scalar on a rank readout -- the 604c hazard
  does not apply, because no arm's DV is derived from an argmax or any order statistic
  over candidates. With injection OFF the chunking machinery does not touch selection
  at all, so there is no selection-level DV in this run to be invariant under.

SLEEP DRIVER: not applicable -- no sleep phase is entered (the MECH-322 carve-out is
OFF in every arm, and this run makes no sleep call).
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ree_core.utils.config import REEConfig  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from experiment_protocol import emit_outcome  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_810_arc071_chunk_accumulator_readiness"
EXPERIMENT_PURPOSE = "diagnostic"
ANCHOR_REACHABILITY_EXEMPT = (
    "The readiness predicate IS the gate's own definition, not a narrower "
    "hand-written proxy for it. MECH-323's evaluative gate is DEFINED as "
    "'outcome mean exceeds the running baseline by a margin', which is "
    "structurally unsatisfiable exactly when the outcome stream has no spread; "
    "the precondition asserts that same statistic (pstdev of the per-episode "
    "outcome) against a 0.05 floor. It is reachable by construction on any task "
    "whose episodes differ at all, and was measured far clear of the gate on this "
    "nursery in a pre-queue bench run (spread 0.33 at 120 episodes, seed 101, "
    "ARM_FULL -- 7 chunks formed, 6 crystallised). There is no separate control "
    "whose reproducibility could fail independently of the asserted quantity."
)

CLAIM_IDS = ["ARC-071", "MECH-323", "MECH-324"]

SEEDS = [101, 202, 303]
ARMS = ["ARM_OFF", "ARM_FORM", "ARM_FULL"]
N_EPISODES = 120
STEPS_PER_EPISODE = 24

# Chunking parameters. Scaled DOWN from the registered defaults (R_min 20 / W 100 /
# C_min 5) to fit a 120-episode probe: the registered values are calibrated for a
# long-lived agent, and at 120 trials a 20-repetition requirement inside a 100-trial
# window would leave the gate structurally near-unsatisfiable. The RATIOS and every
# structural property (hysteresis gap, evaluative gate, depth cap, size budget) are
# preserved; only the absolute counts are reduced. This is a readiness probe, not a
# parameter-calibration run.
CHUNK_MIN_REPETITIONS = 5
CHUNK_WINDOW_TRIALS = 60
CHUNK_CRYSTALLISATION_MIN = 2

# Pre-registered thresholds (defined HERE, never inferred post-hoc).
MIN_CHUNKS_FORMED = 1        # C1: ARM_FORM/ARM_FULL must mint at least this many
MIN_CRYSTALLISED = 1         # C2: ARM_FULL must crystallise at least this many
SEED_PASS_FRACTION = 2.0 / 3.0

# READINESS PRECONDITION. The evaluative gate is RELATIVE, so an outcome stream with
# no spread makes formation structurally impossible regardless of substrate health.
# The routed statistic is the outcome SPREAD (stdev), so the readiness check asserts
# the SAME statistic -- not a mean or a magnitude proxy for it (the V3-EXQ-643
# same-statistic rule). Below the floor the run self-routes substrate_not_ready_requeue,
# never a substrate-verdict label: a flat task cannot falsify MECH-323.
OUTCOME_SPREAD_FLOOR = 0.05


def _build_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(size=8, num_hazards=0, num_resources=6,
                             use_proxy_fields=True, seed=seed)


def _arm_flags(arm: str) -> Dict[str, Any]:
    return {
        "use_policy_chunking": arm in ("ARM_FORM", "ARM_FULL"),
        "use_chunk_maintenance": arm == "ARM_FULL",
        # Injection OFF in every arm: E3 never sees a chunk, so the action stream is
        # untouched by chunk formation and this probe cannot perturb what a later
        # behavioural run measures.
        "use_chunk_proposal_injection": False,
        # MECH-322 carve-out OFF in every arm. The strict MECH-094 gate is the
        # property under test here.
        "use_chunk_replay_origin_path": False,
    }


def _build_agent(env: CausalGridWorldV2, arm: str) -> REEAgent:
    flags = _arm_flags(arm)
    return REEAgent(REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=4,
        alpha_world=0.9,  # SD-008: z_world fidelity
        chunk_min_repetitions=CHUNK_MIN_REPETITIONS,
        chunk_window_trials=CHUNK_WINDOW_TRIALS,
        chunk_crystallisation_min=CHUNK_CRYSTALLISATION_MIN,
        **flags,
    ))


def _config_slice(arm: str) -> Dict[str, Any]:
    slice_ = {
        "env": {"size": 8, "num_hazards": 0, "num_resources": 6,
                "use_proxy_fields": True},
        "schedule": {"n_episodes": N_EPISODES, "steps": STEPS_PER_EPISODE},
        "alpha_world": 0.9,
        "chunk_min_repetitions": CHUNK_MIN_REPETITIONS,
        "chunk_window_trials": CHUNK_WINDOW_TRIALS,
        "chunk_crystallisation_min": CHUNK_CRYSTALLISATION_MIN,
    }
    slice_.update(_arm_flags(arm))
    return slice_


def _run_cell(arm: str, seed: int) -> Dict[str, Any]:
    print(f"Seed {seed} Condition {arm}", flush=True)
    with arm_cell(seed, config_slice=_config_slice(arm),
                  script_path=Path(__file__)) as cell:
        env = _build_env(seed)
        agent = _build_agent(env, arm)
        wd = agent.config.latent.world_dim
        episode_outcomes: List[float] = []

        for ep in range(N_EPISODES):
            _, obs = env.reset()
            agent.reset()
            ep_reward = 0.0
            for _ in range(STEPS_PER_EPISODE):
                latent = agent.sense(obs["body_state"], obs["world_state"])
                ticks = agent.clock.advance()
                e1 = (agent._e1_tick(latent) if ticks.get("e1_tick")
                      else torch.zeros(1, wd, device=agent.device))
                cands = agent.generate_trajectories(latent, e1, ticks)
                action = agent.select_action(cands, ticks)
                # env.step -> (flat_obs, harm_signal, done, info, obs_dict).
                # harm_signal is the per-step scalar (negative = harm, positive =
                # benefit); its episode sum is the outcome the evaluative gate reads.
                _flat, harm_signal, done, _info, obs = env.step(
                    int(action.argmax(dim=-1).item()))
                ep_reward += float(harm_signal)
                if done:
                    break
            # The trial-boundary outcome report. No-op in ARM_OFF (returns []).
            agent.note_chunk_outcome(ep_reward)
            episode_outcomes.append(ep_reward)
            if (ep + 1) % 30 == 0:
                st = agent.get_chunking_state()
                print(f"  [train] chunk seed={seed} arm={arm} ep {ep+1}/{N_EPISODES} "
                      f"formed={st.get('chunk_acc_n_formed', 0)} "
                      f"cryst={st.get('chunk_lib_n_crystallised', 0)}", flush=True)

        st = agent.get_chunking_state()
        outcome_spread = (statistics.pstdev(episode_outcomes)
                          if len(episode_outcomes) > 1 else 0.0)
        row = {
            "arm": arm,
            "seed": seed,
            "chunk_acc_n_formed": int(st.get("chunk_acc_n_formed", 0)),
            "chunk_acc_n_replay_formed": int(st.get("chunk_acc_n_replay_formed", 0)),
            "chunk_acc_n_simulation_skips": int(st.get("chunk_acc_n_simulation_skips", 0)),
            "chunk_acc_n_steps": int(st.get("chunk_acc_n_steps", 0)),
            "chunk_acc_n_outcomes": int(st.get("chunk_acc_n_outcomes", 0)),
            "chunk_acc_n_tracked_sequences": int(st.get("chunk_acc_n_tracked_sequences", 0)),
            "chunk_lib_size": int(st.get("chunk_lib_size", 0)),
            "chunk_lib_n_crystallised": int(st.get("chunk_lib_n_crystallised", 0)),
            "chunk_lib_n_dissolved": int(st.get("chunk_lib_n_dissolved", 0)),
            "chunk_lib_n_selectable": int(st.get("chunk_lib_n_selectable", 0)),
            "chunk_lib_n_replay_origin": int(st.get("chunk_lib_n_replay_origin", 0)),
            "chunk_lib_by_state": st.get("chunk_lib_by_state", {}),
            "chunking_instantiated": bool(agent.policy_chunking is not None),
            # Task-side readouts (recorded generously; the outcome stream is the
            # substrate for the evaluative gate, so its distribution is load-bearing).
            "episode_outcome_mean": (statistics.fmean(episode_outcomes)
                                     if episode_outcomes else 0.0),
            "episode_outcome_spread": outcome_spread,
            "episode_outcome_min": min(episode_outcomes) if episode_outcomes else 0.0,
            "episode_outcome_max": max(episode_outcomes) if episode_outcomes else 0.0,
            "per_episode_outcomes": episode_outcomes,
            "n_episodes": len(episode_outcomes),
        }
        cell.stamp(row)
    # One verdict line per seed x condition cell (runner progress contract: the
    # number of verdict lines must equal seeds x conditions).
    cell_ok = (row["chunk_acc_n_formed"] >= MIN_CHUNKS_FORMED
               if arm != "ARM_OFF" else row["chunk_acc_n_formed"] == 0)
    print(f"verdict: {'PASS' if cell_ok else 'FAIL'}", flush=True)
    return row


def _worst_cell(rows: List[Dict[str, Any]], key: str):
    """The minimum value of `key` across rows, plus the offending cell id.

    Returns the EXTREMUM, not the mean, so a readiness `met` that quantifies over
    cells (all(...)) recomputes exactly from the reported number, and names the
    culprit.
    """
    if not rows:
        return 0.0, None
    worst = min(rows, key=lambda r: r[key])
    return float(worst[key]), f"{worst['arm']}/seed{worst['seed']}"


def run_experiment() -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for arm in ARMS:
        for seed in SEEDS:
            rows.append(_run_cell(arm, seed))

    by_arm = {a: [r for r in rows if r["arm"] == a] for a in ARMS}

    def frac(arm: str, pred) -> float:
        cells = by_arm[arm]
        return (sum(1 for r in cells if pred(r)) / len(cells)) if cells else 0.0

    # --- Readiness precondition: the SAME statistic the evaluative gate routes on.
    spread_worst, spread_cell = _worst_cell(
        [r for r in rows if r["arm"] != "ARM_OFF"], "episode_outcome_spread")
    spread_ready = spread_worst >= OUTCOME_SPREAD_FLOOR

    # --- Pre-registered criteria.
    # C1 (LOAD-BEARING): the accumulator fires at all in the chunking-on arms.
    c1_form = frac("ARM_FORM", lambda r: r["chunk_acc_n_formed"] >= MIN_CHUNKS_FORMED)
    c1_full = frac("ARM_FULL", lambda r: r["chunk_acc_n_formed"] >= MIN_CHUNKS_FORMED)
    c1_pass = (c1_form >= SEED_PASS_FRACTION) and (c1_full >= SEED_PASS_FRACTION)

    # C2: MECH-324 crystallises in ARM_FULL.
    c2_pass = frac("ARM_FULL",
                   lambda r: r["chunk_lib_n_crystallised"] >= MIN_CRYSTALLISED) >= SEED_PASS_FRACTION

    # C3: ARM_OFF is inert -- the operators are not instantiated and nothing forms.
    c3_pass = all((not r["chunking_instantiated"]) and r["chunk_acc_n_formed"] == 0
                  for r in by_arm["ARM_OFF"])

    # C4 (SAFETY): no chunk from replayed/imagined content, in any arm.
    c4_pass = all(r["chunk_acc_n_replay_formed"] == 0 and r["chunk_lib_n_replay_origin"] == 0
                  for r in rows)

    # C5: the ARM_FORM dissociation -- formation without maintenance must NOT
    # crystallise and must expose nothing selectable.
    c5_pass = all(r["chunk_lib_n_crystallised"] == 0 and r["chunk_lib_n_selectable"] == 0
                  for r in by_arm["ARM_FORM"])

    overall_pass = spread_ready and c1_pass and c2_pass and c3_pass and c4_pass and c5_pass

    # --- Non-degeneracy. C1/C2 are counts; they discriminate only if the chunking
    # arms could have differed from the OFF arm at all.
    form_counts = [r["chunk_acc_n_formed"] for r in rows]
    non_degenerate = spread_ready and (max(form_counts) > min(form_counts))

    if not spread_ready:
        label = "substrate_not_ready_requeue"
    elif overall_pass:
        label = "chunk_accumulator_fires"
    elif not c1_pass:
        label = "chunk_accumulator_silent"
    elif not c2_pass:
        label = "chunk_formation_without_crystallisation"
    else:
        label = "chunk_accumulator_partial"

    metrics = {
        "c1_pass": c1_pass, "c2_pass": c2_pass, "c3_pass": c3_pass,
        "c4_pass": c4_pass, "c5_pass": c5_pass,
        "c1_form_seed_frac": c1_form, "c1_full_seed_frac": c1_full,
        "form_n_formed_mean": statistics.fmean(
            [r["chunk_acc_n_formed"] for r in by_arm["ARM_FORM"]]),
        "full_n_formed_mean": statistics.fmean(
            [r["chunk_acc_n_formed"] for r in by_arm["ARM_FULL"]]),
        "full_n_crystallised_mean": statistics.fmean(
            [r["chunk_lib_n_crystallised"] for r in by_arm["ARM_FULL"]]),
        "form_n_crystallised_mean": statistics.fmean(
            [r["chunk_lib_n_crystallised"] for r in by_arm["ARM_FORM"]]),
        "outcome_spread_worst": spread_worst,
        "outcome_spread_worst_cell": spread_cell,
        "n_replay_formed_total": sum(r["chunk_acc_n_replay_formed"] for r in rows),
    }

    interpretation = {
        "label": label,
        "preconditions": [
            {
                "name": "episode_outcome_spread_supra_floor",
                "description": ("MECH-323's evaluative gate is RELATIVE to a running "
                                "baseline, so formation is structurally impossible on a "
                                "flat outcome stream. Asserts the SAME statistic the "
                                "gate routes on (spread), on the chunking-on arms."),
                "measured": spread_worst,
                "threshold": OUTCOME_SPREAD_FLOOR,
                "direction": "lower",
                "control": ("worst cell across all chunking-on cells; the task is a "
                            "resource-bearing grid whose per-episode collection varies"),
                "offending_cell": spread_cell,
                "met": spread_ready,
            },
        ],
        "criteria": [
            {"name": "C1_accumulator_fires", "load_bearing": True, "passed": c1_pass},
            {"name": "C2_crystallisation", "load_bearing": False, "passed": c2_pass},
            {"name": "C3_off_arm_inert", "load_bearing": False, "passed": c3_pass},
            {"name": "C4_no_replay_origin_chunks", "load_bearing": False, "passed": c4_pass},
            {"name": "C5_formation_only_dissociation", "load_bearing": False, "passed": c5_pass},
        ],
        "criteria_non_degenerate": {
            # C1/C2 discriminate only if some arm differs from ARM_OFF's structural 0.
            "C1": non_degenerate,
            "C2": non_degenerate,
            # C3/C4/C5 are structural inertness/safety assertions: they are meaningful
            # regardless of whether formation occurred, because they assert the ABSENCE
            # of something that must never appear.
            "C3": True,
            "C4": True,
            "C5": True,
        },
    }

    return {
        "outcome": "PASS" if overall_pass else "FAIL",
        "metrics": metrics,
        "per_seed_rows": rows,
        "arm_results": rows,
        "interpretation": interpretation,
        "non_degenerate": non_degenerate,
        "degeneracy_reason": (
            None if non_degenerate
            else ("outcome stream had no spread, so the relative evaluative gate could "
                  "not fire for any arm -- the task, not the substrate, is flat")
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    t0 = time.perf_counter()

    global SEEDS, N_EPISODES
    if args.dry_run:
        SEEDS = [101]
        N_EPISODES = 6

    result = run_experiment()
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    full_config = {
        "seeds": SEEDS, "arms": ARMS, "n_episodes": N_EPISODES,
        "steps_per_episode": STEPS_PER_EPISODE,
        "chunk_min_repetitions": CHUNK_MIN_REPETITIONS,
        "chunk_window_trials": CHUNK_WINDOW_TRIALS,
        "chunk_crystallisation_min": CHUNK_CRYSTALLISATION_MIN,
        "min_chunks_formed": MIN_CHUNKS_FORMED,
        "min_crystallised": MIN_CRYSTALLISED,
        "outcome_spread_floor": OUTCOME_SPREAD_FLOOR,
        "seed_pass_fraction": SEED_PASS_FRACTION,
        "arm_config_slices": {a: _config_slice(a) for a in ARMS},
    }
    manifest = {
        "run_id": f"{EXPERIMENT_TYPE}_{ts}_v3",
        "experiment_type": EXPERIMENT_TYPE,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "evidence_direction": "supports" if result["outcome"] == "PASS" else "unknown",
        "evidence_direction_per_claim": {
            "ARC-071": "supports" if result["outcome"] == "PASS" else "unknown",
            "MECH-323": "supports" if result["metrics"]["c1_pass"] else "unknown",
            "MECH-324": "supports" if result["metrics"]["c2_pass"] else "unknown",
        },
        "outcome": result["outcome"],
        "timestamp_utc": ts,
        "metrics": result["metrics"],
        "per_seed_rows": result["per_seed_rows"],
        "arm_results": result["arm_results"],
        "interpretation": result["interpretation"],
        "non_degenerate": result["non_degenerate"],
        "degeneracy_reason": result["degeneracy_reason"],
    }
    out_path = write_flat_manifest(
        manifest,
        Path(__file__).resolve().parents[2] / "REE_assembly" / "evidence" / "experiments",
        dry_run=args.dry_run,
        config=full_config,
        seeds=SEEDS,
        script_path=Path(__file__),
        started_at=t0,
    )
    m = result["metrics"]
    print(f"outcome: {result['outcome']}", flush=True)
    print(f"label: {result['interpretation']['label']}", flush=True)
    print(f"formed: FORM={m['form_n_formed_mean']:.2f} FULL={m['full_n_formed_mean']:.2f} "
          f"| cryst FULL={m['full_n_crystallised_mean']:.2f} "
          f"FORM={m['form_n_crystallised_mean']:.2f}", flush=True)
    print(f"C1={m['c1_pass']} C2={m['c2_pass']} C3={m['c3_pass']} C4={m['c4_pass']} "
          f"C5={m['c5_pass']} spread_worst={m['outcome_spread_worst']:.4f} "
          f"({m['outcome_spread_worst_cell']})", flush=True)
    print(f"wrote: {out_path}", flush=True)
    return result, out_path, args.dry_run


if __name__ == "__main__":
    _result, _out_path, _dry_run = main()
    _outcome_raw = str(_result["outcome"]).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=str(_out_path),
        dry_run=_dry_run,
    )
