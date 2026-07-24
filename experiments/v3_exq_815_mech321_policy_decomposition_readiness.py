"""V3-EXQ-815 -- ARC-070 / MECH-321 substrate-readiness: does decomposition fire?

PURPOSE (diagnostic / substrate-readiness validation, NOT governance evidence).
ARC-070 policy_decomposition_via_event_segmenter was implemented 2026-07-24 as
MECH-321 (ree_core/policy/policy_decomposition.py), wired into HippocampalModule's
propose_trajectories (pre-commit, R4 first phase) and REEAgent's beta-gate release
handling (mid-execution, R4 second phase). Nothing has ever run against it. This is
the first question in the registered validation order, deliberately the SMALLEST
one: does MECH-321.evaluate() actually get called by the live agent, and when it
fires, does the chunk-candidate pool actually change (withhold-and-replace)? Mirrors
how V3-EXQ-810 was ARC-071's first question ("does the accumulator fire at all").

WHAT THIS DOES *NOT* CLAIM. The full ARM_0/ARM_1/ARM_2 discriminative-pair design
from MECH-321's claims.yaml functional_restatement (V_s-drop primary vs
bottleneck-state primary vs OFF baseline, measuring execution-time
prediction-failure rate) is NOT run here -- that needs a task with genuine
region-conditioned predictability heterogeneity (SD-054 reef or similar) and is a
later experiment, exactly as V3-EXQ-810 deferred ARC-071's ARM_1-vs-ARM_2
behavioural-latency measurement past its own readiness probe.

REQUIRED SUBSTRATE STACK. MECH-321 depends_on ARC-070/MECH-288/MECH-269/MECH-094,
NOT MECH-323/324 -- but the ONLY configuration in which MECH-321 has anything to
decompose is when ARC-071 chunk injection is also active (a chunk must be in the
CEM pool before MECH-321 can withhold or replace it). So both arms here run
use_policy_chunking=True + use_chunk_proposal_injection=True + use_event_segmenter=True
(the loud precondition on the latter is enforced by REEAgent itself) +
use_per_stream_vs=True (so the R1 V_s-drop half of the OR trigger is exercisable at
all -- without it, HippocampalModule._region_vs() falls back to a constant 1.0 and
only the MECH-288 boundary half of R1 can ever fire). The single manipulation is
use_policy_decomposition (ARM_OFF / ARM_ON).

FAILURE RECORD: none. ARC-070/MECH-321 has no prior experiment; this is the first.
Acceptance criteria come from the registered MECH-321 design (functional_restatement
R1-R4), not from a FAIL to be moved.

GOV-REUSE-1 (Step 2.4): the decisive readouts are decomp_n_evaluated_precommit,
decomp_n_decomposed_precommit / decomp_n_marked_unreliable, and
mech321_leaf_tiles_added. None of these fields, nor the use_policy_decomposition
knob, existed before 2026-07-24 -- there is no prior ARC-070/MECH-321 run in
evidence/experiments/ at all, and no recorded manifest on any substrate_hash can
carry them. Not recoverable -> run.

MECH-094 note (not safety-critical here, unlike ARC-071). Unlike MECH-323's strict
write-refusal, PolicyDecomposition.evaluate() has no residue-write side effect to
gate (see policy_decomposition.py's "asymmetry with ARC-071" docstring) -- there is
nothing analogous to C4/chunk_acc_n_replay_formed to pin. This run's C4 instead pins
the CHUNK-FORMATION side's existing MECH-094 guarantee (n_replay_formed == 0),
carried over unchanged from V3-EXQ-810 because ARC-071 chunking is still active in
both arms here.

DV-SYMMETRY (Step 3 mandatory declaration, per arm):
  ARM_OFF DV = decomp_n_evaluated_precommit, structurally 0 (agent.policy_decomposition
    is None; HippocampalModule never registers a decomposition source, so
    _apply_policy_decomposition's early-return makes it a no-op). Not a measurement
    and not claimed as one -- it is the inertness control for C3.
  ARM_ON DV = decomp_n_evaluated_precommit / decomp_n_decomposed_precommit /
    decomp_n_marked_unreliable / mech321_leaf_tiles_added, which are COUNTS over
    live evaluate() calls and pool mutations. Symmetry group of a count: permutation
    of which candidate/tick triggered and relabelling of action-class indices (a
    count is a symmetric function of its set). The manipulation is NOT invariant
    under it: turning MECH-321 on changes the CARDINALITY of evaluate() calls from
    0, and cardinality is not preserved by any permutation of call members.
  Not a broadcast-scalar-on-rank-readout case (the 604c hazard) -- none of these DVs
  are derived from an argmax or order statistic over the CEM candidate pool; they
  are direct counts of MECH-321's own internal decisions and pool-mutation side
  effects, read via agent.get_policy_decomposition_state() and
  agent.hippocampal._last_propose_diagnostics.

SLEEP DRIVER: not applicable -- no sleep phase is entered in this run.
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
from ree_core.policy import ChunkedPrimitive, ChunkState  # noqa: E402
from experiment_protocol import emit_outcome  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_815_mech321_policy_decomposition_readiness"
EXPERIMENT_PURPOSE = "diagnostic"
ANCHOR_REACHABILITY_EXEMPT = (
    "The readiness predicate IS a direct wiring check, not a proxy for a "
    "behavioural effect. MECH-321 fires whenever an ARC-071 chunk candidate is in "
    "the CEM pool AND either (a) region V_s drops below decomposition_vs_threshold "
    "or (b) MECH-288's boundary_on(stream=rollout,...) fires on that candidate's "
    "own rolled-out states. A crystallised chunk is seeded directly into the "
    "library at cell setup (rather than relying on ARC-071 formation to fire "
    "within the probe's short horizon, which is a separate substrate question "
    "V3-EXQ-810 already answers), so the precondition -- 'a chunk candidate "
    "exists for MECH-321 to evaluate' -- is satisfied by construction. There is no "
    "separate control whose reproducibility could fail independently of this."
)

CLAIM_IDS = ["ARC-070", "MECH-321"]

SEEDS = [11, 23, 47]
ARMS = ["ARM_OFF", "ARM_ON"]
N_EPISODES = 20
STEPS_PER_EPISODE = 24

# Decomposition parameters (registered defaults -- this probe does not need to
# tune them, since the seeded chunk + real rollout dynamics are enough to
# exercise the trigger; see the readiness precondition below).
DECOMPOSITION_VS_THRESHOLD = 0.5
DECOMPOSITION_DEPTH_CAP = 3

# Seeded chunk: depth 1 (the common ARC-071 formation shape -- also the shape
# this session's own activation smoke test caught a regression on, see
# policy_decomposition.py history), 3 raw actions long.
SEEDED_CHUNK_SEQUENCE = (0, 1, 2)

# Pre-registered thresholds (defined HERE, never inferred post-hoc).
MIN_EVALUATED = 1            # C1 (LOAD-BEARING): evaluate() must be called at all
SEED_PASS_FRACTION = 2.0 / 3.0


def _build_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(size=8, num_hazards=0, num_resources=6,
                             use_proxy_fields=True, seed=seed)


def _arm_flags(arm: str) -> Dict[str, Any]:
    return {
        # Constant baseline in both arms (see module docstring "required
        # substrate stack"): without these, there is nothing in the CEM pool
        # for MECH-321 to ever evaluate, regardless of use_policy_decomposition.
        "use_event_segmenter": True,
        "use_policy_chunking": True,
        "use_chunk_proposal_injection": True,
        "use_per_stream_vs": True,
        # The single manipulation.
        "use_policy_decomposition": arm == "ARM_ON",
        "decomposition_vs_threshold": DECOMPOSITION_VS_THRESHOLD,
        "decomposition_depth_cap": DECOMPOSITION_DEPTH_CAP,
    }


def _build_agent(env: CausalGridWorldV2, arm: str) -> REEAgent:
    flags = _arm_flags(arm)
    agent = REEAgent(REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=4,
        alpha_world=0.9,  # SD-008: z_world fidelity
        **flags,
    ))
    # Seed a crystallised chunk directly -- ARC-071 formation readiness is a
    # separate, already-answered question (V3-EXQ-810); this probe isolates
    # MECH-321's own wiring given a chunk candidate exists.
    chunk = ChunkedPrimitive(
        sequence=SEEDED_CHUNK_SEQUENCE, depth=1,
        state=ChunkState.CRYSTALLISED, selection_weight=1.0,
    )
    agent.policy_chunking.library.register(chunk)
    return agent


def _config_slice(arm: str) -> Dict[str, Any]:
    slice_ = {
        "env": {"size": 8, "num_hazards": 0, "num_resources": 6,
                "use_proxy_fields": True},
        "schedule": {"n_episodes": N_EPISODES, "steps": STEPS_PER_EPISODE},
        "alpha_world": 0.9,
        "seeded_chunk_sequence": list(SEEDED_CHUNK_SEQUENCE),
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

        for ep in range(N_EPISODES):
            _, obs = env.reset()
            agent.reset()
            # ARC-071 formation state is preserved across agent.reset() by
            # design (end_episode(), not reset() -- see agent.py), so the
            # seeded chunk survives episode boundaries.
            for _ in range(STEPS_PER_EPISODE):
                latent = agent.sense(obs["body_state"], obs["world_state"])
                ticks = agent.clock.advance()
                e1 = (agent._e1_tick(latent) if ticks.get("e1_tick")
                      else torch.zeros(1, wd, device=agent.device))
                cands = agent.generate_trajectories(latent, e1, ticks)
                action = agent.select_action(cands, ticks)
                # env.step -> (flat_obs, harm_signal, done, info, obs_dict).
                _flat, _harm_signal, done, _info, obs = env.step(
                    int(action.argmax(dim=-1).item()))
                if done:
                    break
            if (ep + 1) % 5 == 0:
                st = agent.get_policy_decomposition_state()
                print(f"  [train] decomp seed={seed} arm={arm} ep {ep+1}/{N_EPISODES} "
                      f"evaluated={st.get('decomp_n_evaluated_precommit', 0)} "
                      f"decomposed={st.get('decomp_n_decomposed_precommit', 0)}",
                      flush=True)

        st = agent.get_policy_decomposition_state()
        chunk_st = agent.get_chunking_state()
        propose_diag = dict(getattr(agent.hippocampal, "_last_propose_diagnostics", {}) or {})
        row = {
            "arm": arm,
            "seed": seed,
            "decomp_n_evaluated_precommit": int(st.get("decomp_n_evaluated_precommit", 0)),
            "decomp_n_evaluated_midexec": int(st.get("decomp_n_evaluated_midexec", 0)),
            "decomp_n_decomposed_precommit": int(st.get("decomp_n_decomposed_precommit", 0)),
            "decomp_n_decomposed_midexec": int(st.get("decomp_n_decomposed_midexec", 0)),
            "decomp_n_marked_unreliable": int(st.get("decomp_n_marked_unreliable", 0)),
            "decomp_n_vs_trigger": int(st.get("decomp_n_vs_trigger", 0)),
            "decomp_n_boundary_fires": int(st.get("decomp_n_boundary_fires", 0)),
            "decomposition_instantiated": bool(agent.policy_decomposition is not None),
            "mech321_chunks_withheld": int(propose_diag.get("mech321_chunks_withheld", 0)),
            "mech321_chunks_decomposed": int(propose_diag.get("mech321_chunks_decomposed", 0)),
            "mech321_chunks_marked_unreliable": int(
                propose_diag.get("mech321_chunks_marked_unreliable", 0)
            ),
            "mech321_leaf_tiles_added": int(propose_diag.get("mech321_leaf_tiles_added", 0)),
            "arc071_chunk_candidates_added": int(
                propose_diag.get("arc071_chunk_candidates_added", 0)
            ),
            # MECH-094 carryover from V3-EXQ-810 (ARC-071 chunking is active in
            # both arms here) -- unrelated to MECH-321's own trigger, but the
            # strict formation guarantee must hold regardless.
            "chunk_acc_n_replay_formed": int(chunk_st.get("chunk_acc_n_replay_formed", 0)),
        }
        cell.stamp(row)
    # One verdict line per seed x condition cell (runner progress contract).
    cell_ok = (row["decomp_n_evaluated_precommit"] >= MIN_EVALUATED
               if arm == "ARM_ON" else row["decomp_n_evaluated_precommit"] == 0)
    print(f"verdict: {'PASS' if cell_ok else 'FAIL'}", flush=True)
    return row


def run_experiment() -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for arm in ARMS:
        for seed in SEEDS:
            rows.append(_run_cell(arm, seed))

    by_arm = {a: [r for r in rows if r["arm"] == a] for a in ARMS}

    def frac(arm: str, pred) -> float:
        cells = by_arm[arm]
        return (sum(1 for r in cells if pred(r)) / len(cells)) if cells else 0.0

    # --- READINESS PRECONDITION: a chunk candidate must actually have reached
    # the CEM pool for MECH-321 to have anything to evaluate. Asserts the SAME
    # statistic C1 depends on existing (arc071_chunk_candidates_added > 0),
    # not a proxy for it. The seeded chunk (ANCHOR_REACHABILITY_EXEMPT above)
    # makes this reachable by construction; this precondition still measures
    # it rather than assuming it, in case chunk injection itself is silently
    # broken.
    cand_worst = min((r["arc071_chunk_candidates_added"] for r in rows), default=0)
    cand_ready = cand_worst >= 1

    # --- Pre-registered criteria.
    # C1 (LOAD-BEARING): MECH-321.evaluate() is actually called in ARM_ON.
    c1_pass = frac("ARM_ON", lambda r: r["decomp_n_evaluated_precommit"] >= MIN_EVALUATED) >= SEED_PASS_FRACTION

    # C2: the trigger fires at least occasionally against real network
    # activity (decompose OR marked_unreliable -- either is evidence R1 fired).
    c2_pass = frac(
        "ARM_ON",
        lambda r: (r["decomp_n_decomposed_precommit"] + r["decomp_n_marked_unreliable"]) >= 1,
    ) >= SEED_PASS_FRACTION

    # C3: when a decomposition occurs, the pool-replacement side effect is
    # real (leaf tiles actually added), not just a decision computed and
    # discarded.
    c3_pass = all(
        (r["mech321_leaf_tiles_added"] >= 1) if r["mech321_chunks_decomposed"] >= 1 else True
        for r in by_arm["ARM_ON"]
    )

    # C4: ARM_OFF is inert -- no decomposition source is ever registered, and
    # ARC-071 chunk candidates pass through the pool exactly as they did
    # before MECH-321 existed (bit-identical-when-off).
    c4_pass = all(
        (not r["decomposition_instantiated"])
        and r["decomp_n_evaluated_precommit"] == 0
        and r["mech321_chunks_withheld"] == 0
        and r["arc071_chunk_candidates_added"] >= 1
        for r in by_arm["ARM_OFF"]
    )

    # C5 (MECH-094 carryover): no chunk minted from replayed/imagined content,
    # in any arm -- the ARC-071 guarantee is untouched by MECH-321 landing.
    c5_pass = all(r["chunk_acc_n_replay_formed"] == 0 for r in rows)

    overall_pass = cand_ready and c1_pass and c2_pass and c3_pass and c4_pass and c5_pass

    # --- Non-degeneracy. C1/C2 discriminate only if evaluate()-call counts
    # differ between arms at all.
    eval_counts = [r["decomp_n_evaluated_precommit"] for r in rows]
    non_degenerate = cand_ready and (max(eval_counts) > min(eval_counts))

    if not cand_ready:
        label = "substrate_not_ready_requeue"
    elif overall_pass:
        label = "policy_decomposition_fires"
    elif not c1_pass:
        label = "policy_decomposition_silent"
    elif not c2_pass:
        label = "policy_decomposition_trigger_never_fires"
    elif not c3_pass:
        label = "policy_decomposition_decision_not_applied"
    else:
        label = "policy_decomposition_partial"

    metrics = {
        "c1_pass": c1_pass, "c2_pass": c2_pass, "c3_pass": c3_pass,
        "c4_pass": c4_pass, "c5_pass": c5_pass,
        "on_n_evaluated_mean": statistics.fmean(
            [r["decomp_n_evaluated_precommit"] for r in by_arm["ARM_ON"]]),
        "on_n_decomposed_mean": statistics.fmean(
            [r["decomp_n_decomposed_precommit"] for r in by_arm["ARM_ON"]]),
        "on_n_marked_unreliable_mean": statistics.fmean(
            [r["decomp_n_marked_unreliable"] for r in by_arm["ARM_ON"]]),
        "on_n_leaf_tiles_added_mean": statistics.fmean(
            [r["mech321_leaf_tiles_added"] for r in by_arm["ARM_ON"]]),
        "on_n_vs_trigger_total": sum(r["decomp_n_vs_trigger"] for r in by_arm["ARM_ON"]),
        "on_n_boundary_fires_total": sum(r["decomp_n_boundary_fires"] for r in by_arm["ARM_ON"]),
        "off_n_evaluated_total": sum(r["decomp_n_evaluated_precommit"] for r in by_arm["ARM_OFF"]),
        "candidate_pool_worst": cand_worst,
        "n_replay_formed_total": sum(r["chunk_acc_n_replay_formed"] for r in rows),
    }

    interpretation = {
        "label": label,
        "preconditions": [
            {
                "name": "chunk_candidate_reaches_pool",
                "description": ("MECH-321 has nothing to evaluate unless an ARC-071 "
                                "chunk candidate actually reaches the CEM pool. Asserts "
                                "the same statistic C1/C4 depend on "
                                "(arc071_chunk_candidates_added), not a proxy for it."),
                "measured": float(cand_worst),
                "threshold": 1.0,
                "direction": "lower",
                "control": ("worst cell across ALL cells (both arms); a crystallised "
                            "chunk is seeded directly at cell setup"),
                "met": cand_ready,
            },
        ],
        "criteria": [
            {"name": "C1_evaluate_called", "load_bearing": True, "passed": c1_pass},
            {"name": "C2_trigger_fires", "load_bearing": True, "passed": c2_pass},
            {"name": "C3_pool_mutation_applied", "load_bearing": False, "passed": c3_pass},
            {"name": "C4_off_arm_inert", "load_bearing": False, "passed": c4_pass},
            {"name": "C5_no_replay_origin_chunks", "load_bearing": False, "passed": c5_pass},
        ],
        "criteria_non_degenerate": {
            # C1/C2/C3 discriminate only if ARM_ON differs from ARM_OFF's
            # structural 0 at all.
            "C1": non_degenerate,
            "C2": non_degenerate,
            "C3": non_degenerate,
            # C4/C5 are structural inertness/safety assertions: meaningful
            # regardless of whether MECH-321 fired, because they assert the
            # ABSENCE of something that must never appear.
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
            else ("no ARM_ON cell ever called evaluate() more than ARM_OFF's "
                  "structural 0, so the substrate, not merely the task, is silent")
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    t0 = time.perf_counter()

    global SEEDS, N_EPISODES
    if args.dry_run:
        SEEDS = [11]
        N_EPISODES = 3

    result = run_experiment()
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    full_config = {
        "seeds": SEEDS, "arms": ARMS, "n_episodes": N_EPISODES,
        "steps_per_episode": STEPS_PER_EPISODE,
        "decomposition_vs_threshold": DECOMPOSITION_VS_THRESHOLD,
        "decomposition_depth_cap": DECOMPOSITION_DEPTH_CAP,
        "seeded_chunk_sequence": list(SEEDED_CHUNK_SEQUENCE),
        "min_evaluated": MIN_EVALUATED,
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
            "ARC-070": "supports" if result["outcome"] == "PASS" else "unknown",
            "MECH-321": "supports" if result["metrics"]["c1_pass"] and result["metrics"]["c2_pass"] else "unknown",
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
    print(f"ON evaluated={m['on_n_evaluated_mean']:.2f} decomposed={m['on_n_decomposed_mean']:.2f} "
          f"marked_unreliable={m['on_n_marked_unreliable_mean']:.2f} "
          f"leaf_tiles={m['on_n_leaf_tiles_added_mean']:.2f}", flush=True)
    print(f"C1={m['c1_pass']} C2={m['c2_pass']} C3={m['c3_pass']} C4={m['c4_pass']} "
          f"C5={m['c5_pass']} candidate_pool_worst={m['candidate_pool_worst']}", flush=True)
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
