"""V3-EXQ-820 -- ARC-070 / MECH-321 R1-vs-R5-vs-OFF three-arm dissociation (adds ARM_2).

PURPOSE (evidence; governance-weighting). Completes the discriminative validation from
MECH-321's claims.yaml functional_restatement by adding ARM_2 (the R5 bottleneck-state
trigger) on top of the ARM_0/ARM_1 design of V3-EXQ-816. V3-EXQ-816 is the ARM_0(OFF) /
ARM_1(V_s-drop) behavioural half; ARM_2 was deferred there because the bottleneck-state
trigger was not built. The R5 bottleneck trigger mode landed 2026-07-25 (ree-v3 2422632:
REEConfig.decomposition_trigger_mode = "vs_boundary" | "bottleneck"), so the full
R1-vs-R5-vs-OFF dissociation is now buildable. This run cites V3-EXQ-816 as the ARM_0/ARM_1
half and reuses its canonical baseline module + reuse-eligible ARM_0.

    ARM_0 (use_policy_decomposition=False): the full MECH-321 substrate stack is ON
        (event_segmenter + policy_chunking + chunk injection + per_stream_vs) but no
        decomposition ever withholds/replaces a chunk. Identical to 816's ARM_0; reused via
        the mech321_policy_decomposition baseline module when a compatible mint is banked,
        else run fresh (and re-minted reuse-eligible here).
    ARM_1 (use_policy_decomposition=True, decomposition_trigger_mode="vs_boundary"): the R1
        trigger -- a chunk whose region shows low V_s (or on which MECH-288 fires) is
        decomposed pre-commitment. Identical trigger to 816's ARM_1.
    ARM_2 (use_policy_decomposition=True, decomposition_trigger_mode="bottleneck"): the R5
        alternative -- a chunk is decomposed ONLY at bottleneck states, REGARDLESS OF V_s.
        Bottleneck = an online incremental McGovern & Barto 2001 diverse-density accumulator
        (a region recurred >= min_visits times AND is a funnel entered-from/exited-to >=
        min_distinct_neighbors distinct regions) over a coarse quantised z_world region key.

THE DISCRIMINATIVE PREDICTION (functional_restatement). R1 (ARM_1) gates decomposition on
V_s; R5 (ARM_2) gates on bottleneck topology, IGNORING V_s. So the two triggers must fire at
DIFFERENT places: ARM_1's decompositions cluster at LOW V_s (V_s is its trigger); ARM_2's do
NOT (V_s is not in its trigger at all), so a substantial fraction of ARM_2's decompositions
occur at states whose V_s is ABOVE the decomposition threshold -- states ARM_1's V_s half
could never trigger on. That V_s-independence is the load-bearing, direct measurement of the
R1-vs-R5 mechanistic dissociation and is what the R5 verdict ("distinct mechanisms") asserts.
On this repeated-structure environment (60 episodes over one 8x8 grid), the R5 verdict also
predicts ARM_1 and ARM_2 CONVERGE on reducing execution-time forward PE (overlapping outputs
on repeated structure) -- carried here as a non-load-bearing corroboration DV.

LOAD-BEARING DV -- V_s-independence of ARM_2's decomposition locus.
  frac_decomp_at_high_vs[arm] = fraction of that arm's pre-commit decomposition EVENTS that
  occurred at a step whose region V_s >= decomposition_vs_threshold. A decomposition EVENT is
  a per-measuring-step positive delta in decomp_n_decomposed_precommit (attributed to the
  region V_s read on the same step, which is the region context propose_trajectories acted
  on). Pre-registered: ARM_2 must decompose at high-V_s states at a rate the V_s trigger
  cannot explain -- frac_decomp_at_high_vs[ARM_2] >= HIGH_VS_DECOMP_FLOOR AND it exceeds
  ARM_1's by DISSOCIATION_MARGIN. ARM_1 CAN reach high V_s only via the MECH-288 boundary
  half of its OR trigger, so a low ARM_1 fraction with a high ARM_2 fraction is exactly the
  V_s-vs-bottleneck dissociation. This is NOT an arithmetic identity: nothing forces ARM_2's
  bottleneck states to have any particular V_s -- their V_s distribution is an empirical
  property of where recurring funnels sit in this env, measured, not constructed.

DV-SYMMETRY (Step 3 mandatory per-arm declaration).
  ARM_0 DV = frac_decomp_at_high_vs is structurally 0 (no decomposition events at all;
    use_policy_decomposition=False). Not a measurement -- the inertness control.
  ARM_1 / ARM_2 load-bearing DV = frac_decomp_at_high_vs, a ratio of counts over
    decomposition EVENTS. Symmetry group of a count-ratio: permutation of which
    event/step/seed contributed and relabelling of action-class indices (a count is a
    symmetric function of its set). The manipulation (decomposition_trigger_mode) is NOT
    invariant under it: switching vs_boundary -> bottleneck changes WHICH steps decompose
    (V_s-gated set vs bottleneck-gated set), so the region-V_s values AT the decomposition
    events differ in content and cardinality between arms -- the ratio is a genuine measured
    difference, not fixed before the run. Verified, not assumed: non_degenerate requires
    ARM_2's executed action sequence to differ from ARM_0's on >= 1 seed AND ARM_2's
    bottleneck trigger to have actually fired (decomp_n_bottleneck_fires > 0); a bottleneck
    trigger that never fired (env lacked recurring funnel structure at these params) is
    substrate-not-ready, NOT a falsification.
  NOT a broadcast-scalar-on-argmax case (the 604c hazard): frac_decomp_at_high_vs is a
    direct count over MECH-321's own internal decomposition decisions and the region V_s read
    at those steps, not a scalar derived from an argmax/order-statistic over the CEM pool.
  Corroboration DV = low-V_s execution-time forward PE reduction vs ARM_0 (same DV as 816),
    NON-load-bearing: a null (e.g. ARM_2 does not reduce forward PE) does not sink the run --
    the load-bearing claim is the V_s-INDEPENDENCE of ARM_2's trigger, not that it helps.

READINESS (P0). The dissociation is meaningful only if (P1) V_s actually drops below
threshold somewhere (so "high vs low V_s" is a real partition and ARM_1's V_s half can fire),
(P2a) ARM_1 decomposition fires, (P2b) ARM_2's BOTTLENECK trigger fires and decomposes
(decomp_n_bottleneck_fires > 0 AND decomp_n_decomposed_precommit > 0 -- the env has recurring
funnel structure at these bottleneck params), and (P3) BOTH ARM_1 and ARM_2 produce enough
decomposition events to estimate frac_decomp_at_high_vs (>= MIN_DECOMP_EVENTS each). Any
failure self-routes substrate_not_ready_requeue with non_degenerate=False (scoring-excluded),
NEVER a falsification of the claim.

GOV-REUSE-1 (Step 2.4). The decisive readout is frac_decomp_at_high_vs under
decomposition_trigger_mode="bottleneck". That config value did not exist before 2026-07-25
(ree-v3 2422632); no recorded manifest on any substrate_hash can carry an ARM_2 bottleneck
readout, and V3-EXQ-816 (the only other ARC-070/MECH-321 behavioural run) has not run and has
no ARM_2. Not recoverable -> run. Checked: no v3_exq_816* / v3_exq_820* manifest exists in
evidence/experiments/ (816 queued-not-run; 820 is this run; 819 was taken by a concurrent
mech457/inv088 session, so this ARM_2 successor took the next free number 820).

MECH-094 note. Like 816, PolicyDecomposition.evaluate() has no residue-write side effect in
either trigger mode (see policy_decomposition.py "asymmetry with ARC-071"); hypothesis_tag is
a diagnostics label, not a gate. C_MECH094 carries over the ARC-071 chunk-formation guarantee
(chunk_acc_n_replay_formed == 0), unchanged, since chunking is active in all three arms.

SLEEP DRIVER: not applicable -- no sleep phase is entered in this run.
"""
from __future__ import annotations

import argparse
import math
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ree_core.utils.config import REEConfig  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.policy import ChunkedPrimitive, ChunkState  # noqa: E402
from experiment_protocol import emit_outcome  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._lib.arm_reuse import try_reuse_cell  # noqa: E402
from experiments._lib.baselines import mech321_policy_decomposition as base  # noqa: E402
from experiments._lib.manifest_core import stamp_recording_core  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_820_mech321_policy_decomposition_bottleneck_arm2"
EXPERIMENT_PURPOSE = "evidence"
CLAIM_IDS = ["ARC-070", "MECH-321"]

# The validator's hold-weighted-E3-readout heuristic flags the per-env-step read of
# select_action()'s return. Triage (per the gate's own SAFE/AT-RISK/DISQUALIFYING rubric):
# the LOAD-BEARING DV (frac_decomp_at_high_vs) is NOT hold-weighted -- it counts pre-commit
# decomposition EVENTS (positive deltas in decomp_n_decomposed_precommit), and those advance
# ONLY on a fresh propose: generate_trajectories returns CACHED candidates on a held tick, so
# _apply_policy_decomposition never runs and the decomposed count does not increment on holds.
# The flagged select_action() return feeds only measure_action_seq, used for a
# THRESHOLD-INVARIANT binary "sequences differ" non-degeneracy check (SAFE per the rubric).
# The per-step forward-PE corroboration DV (from update_residue, which trains + measures every
# executed step) is per-executed-step BY DESIGN -- it is "execution-time forward prediction
# failure", the identical per-step readout V3-EXQ-816 carries as its own load-bearing DV --
# and it is NON-load-bearing here. Denominators are auditable: decomp_events_total (per arm)
# is emitted for the load-bearing frac.
E3_HOLD_WEIGHTED_READOUT_EXEMPT = (
    "Load-bearing DV counts fresh-propose-only decomposition events (held ticks return cached "
    "candidates -> no decompose, no count advance); flagged select_action() read feeds only a "
    "threshold-invariant action-sequence divergence check; per-step forward-PE is a "
    "non-load-bearing corroboration DV identical to V3-EXQ-816's per-executed-step readout."
)

SEEDS = [11, 23, 47, 71, 97]
ARMS = ["ARM_0", "ARM_1", "ARM_2"]

# Schedule / env from the canonical baseline module so ARM_0 matches V3-EXQ-816's ARM_0 BY
# CONSTRUCTION (same off_path_config_slice -> same arm fingerprint -> reuse HITs).
WARMUP_EPISODES = base.WARMUP_EPISODES
MEASURE_EPISODES = base.MEASURE_EPISODES
STEPS_PER_EPISODE = base.STEPS_PER_EPISODE
SEEDED_CHUNK_SEQUENCE = base.SEEDED_CHUNK_SEQUENCE

# Decomposition parameters. Threshold 0.5 matches V3-EXQ-816 / V3-EXQ-815 (a realistic chance
# for the V_s half to fire on this env's stability distribution), and it is the partition
# boundary for "high vs low V_s" the load-bearing DV reads.
DECOMPOSITION_VS_THRESHOLD = 0.5
DECOMPOSITION_DEPTH_CAP = 3          # mirrors ARC-071 chunk_max_depth (substrate default)

# R5 bottleneck-trigger params (ARM_2 only; substrate defaults). Deliberately loose so the
# accumulator CAN fire on a repeated-structure env within the schedule; P2b self-routes to
# substrate_not_ready_requeue if the env lacks recurring funnel structure at these params
# (never a falsification).
BOTTLENECK_MIN_VISITS = 3
BOTTLENECK_MIN_DISTINCT_NEIGHBORS = 2
BOTTLENECK_REGION_QUANT = 1.0
BOTTLENECK_REGION_DIMS = 8

REUSE_BASELINE_FROM = None  # cite a V3-EXQ-816 ARM_0 mint run_id here once 816 runs; None =
                            # reuse ANY compatible banked mint, else run fresh (self-mint).

# --- Pre-registered thresholds (defined HERE, never inferred post-hoc). ---
# Load-bearing: ARM_2 must decompose at high-V_s (>= threshold) states -- the V_s-independence
# signature -- at a rate ARM_1's V_s trigger cannot produce.
HIGH_VS_DECOMP_FLOOR = 0.20          # >= 20% of ARM_2's decompositions at region_vs >= thresh
DISSOCIATION_MARGIN = 0.10           # ARM_2 high-V_s frac exceeds ARM_1's by >= this
# Readiness floors.
MIN_LOW_VS_STEPS = 5                 # P1: low-V_s steps must exist (per worst ARM_1 cell)
MIN_DECOMP_EVENTS = 5                # P3: min decomposition events per arm to estimate a frac
SEED_FIRE_FRACTION = 3.0 / 5.0       # decomposition must fire in a majority of ARM_1/ARM_2 cells


# ---------------------------------------------------------------------------
def _build_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(**base.env_kwargs(seed))


def _arm_flags(arm: str) -> Dict[str, Any]:
    if arm == "ARM_0":
        return dict(base.off_arm_flags())
    flags = dict(base.substrate_stack_flags())
    flags.update({
        "use_policy_decomposition": True,
        "decomposition_vs_threshold": DECOMPOSITION_VS_THRESHOLD,
        "decomposition_depth_cap": DECOMPOSITION_DEPTH_CAP,
    })
    if arm == "ARM_1":
        flags["decomposition_trigger_mode"] = "vs_boundary"
    elif arm == "ARM_2":
        flags.update({
            "decomposition_trigger_mode": "bottleneck",
            "decomposition_bottleneck_min_visits": BOTTLENECK_MIN_VISITS,
            "decomposition_bottleneck_min_distinct_neighbors": BOTTLENECK_MIN_DISTINCT_NEIGHBORS,
            "decomposition_bottleneck_region_quant": BOTTLENECK_REGION_QUANT,
            "decomposition_bottleneck_region_dims": BOTTLENECK_REGION_DIMS,
        })
    return flags


def _config_slice(arm: str) -> Dict[str, Any]:
    if arm == "ARM_0":
        # Single source of truth -- byte-identical to V3-EXQ-816's ARM_0 emit, so the
        # fingerprint matches and reuse HITs (or self-mints for a later consumer).
        return base.off_path_config_slice()
    slice_: Dict[str, Any] = {
        "env": {"size": base.ENV_SIZE, "num_hazards": base.ENV_NUM_HAZARDS,
                "num_resources": base.ENV_NUM_RESOURCES,
                "use_proxy_fields": base.ENV_USE_PROXY_FIELDS},
        "schedule": {"warmup_episodes": WARMUP_EPISODES,
                     "measure_episodes": MEASURE_EPISODES, "steps": STEPS_PER_EPISODE},
        "alpha_world": base.ALPHA_WORLD,
        "seeded_chunk_sequence": list(SEEDED_CHUNK_SEQUENCE),
    }
    slice_.update(_arm_flags(arm))
    return slice_


def _build_agent(env: CausalGridWorldV2, arm: str) -> REEAgent:
    agent = REEAgent(REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=4,
        alpha_world=base.ALPHA_WORLD,
        **_arm_flags(arm),
    ))
    # Seed the identical crystallised chunk in ALL arms (only ARM_1/ARM_2 can decompose it).
    # ARC-071 formation state survives agent.reset() by design.
    chunk = ChunkedPrimitive(
        sequence=SEEDED_CHUNK_SEQUENCE, depth=1,
        state=ChunkState.CRYSTALLISED, selection_weight=1.0,
    )
    agent.policy_chunking.library.register(chunk)
    return agent


def _measure_cell(arm: str, seed: int) -> Dict[str, Any]:
    """Run one seed x arm cell and collect the DVs. Shared by the fresh-run path and the
    ARM_0 reuse-miss fallback."""
    env = _build_env(seed)
    agent = _build_agent(env, arm)
    wd = agent.config.latent.world_dim
    total_eps = WARMUP_EPISODES + MEASURE_EPISODES

    fwd_pe_low: List[float] = []
    action_seq: List[int] = []
    n_measure_steps = 0
    low_vs_steps = 0
    # Load-bearing DV inputs: decomposition EVENTS partitioned by region V_s at the step.
    decomp_events_high_vs = 0
    decomp_events_low_vs = 0
    prev_decomposed = 0

    for ep in range(total_eps):
        _, obs = env.reset()
        agent.reset()
        measuring = ep >= WARMUP_EPISODES
        for _ in range(STEPS_PER_EPISODE):
            latent = agent.sense(obs["body_state"], obs["world_state"])
            ticks = agent.clock.advance()
            e1 = (agent._e1_tick(latent) if ticks.get("e1_tick")
                  else torch.zeros(1, wd, device=agent.device))
            cands = agent.generate_trajectories(latent, e1, ticks)
            action = agent.select_action(cands, ticks)
            a_idx = int(action.argmax(dim=-1).item())
            _flat, harm_signal, done, _info, obs = env.step(a_idx)
            metrics = agent.update_residue(harm_signal)

            if measuring:
                region_vs = float(agent.hippocampal._region_vs())
                is_low = region_vs < DECOMPOSITION_VS_THRESHOLD
                # Detect a pre-commit decomposition EVENT this step: a positive delta in the
                # cumulative decomposed count (decomposition happens during
                # generate_trajectories above, so the region_vs read on this same step is the
                # region context it acted on). Partition the event by high/low V_s.
                st = agent.get_policy_decomposition_state()
                decomposed_now = int(st.get("decomp_n_decomposed_precommit", 0))
                if decomposed_now > prev_decomposed:
                    if is_low:
                        decomp_events_low_vs += 1
                    else:
                        decomp_events_high_vs += 1
                prev_decomposed = decomposed_now

                pe_raw = metrics.get("e3_prediction_error")
                if pe_raw is not None:
                    pe = float(pe_raw.detach()) if torch.is_tensor(pe_raw) else float(pe_raw)
                    if math.isfinite(pe) and is_low:
                        fwd_pe_low.append(pe)
                n_measure_steps += 1
                if is_low:
                    low_vs_steps += 1
                action_seq.append(a_idx)
            else:
                # Keep prev_decomposed current through warmup so the first measuring-window
                # event is a true delta, not the whole warmup accumulation.
                st = agent.get_policy_decomposition_state()
                prev_decomposed = int(st.get("decomp_n_decomposed_precommit", 0))
            if done:
                break
        if (ep + 1) % 10 == 0:
            st = agent.get_policy_decomposition_state()
            print(f"  [train] decomp seed={seed} arm={arm} ep {ep+1}/{total_eps} "
                  f"decomposed={st.get('decomp_n_decomposed_precommit', 0)} "
                  f"bottleneck_fires={st.get('decomp_n_bottleneck_fires', 0)} "
                  f"low_vs_steps={low_vs_steps}", flush=True)

    st = agent.get_policy_decomposition_state()
    chunk_st = agent.get_chunking_state()
    decomp_events_total = decomp_events_high_vs + decomp_events_low_vs
    row = {
        "arm": arm,
        "seed": seed,
        "n_measure_steps": int(n_measure_steps),
        "low_vs_steps": int(low_vs_steps),
        "low_vs_step_frac": (low_vs_steps / n_measure_steps) if n_measure_steps else 0.0,
        # Load-bearing DV inputs.
        "decomp_events_total": int(decomp_events_total),
        "decomp_events_high_vs": int(decomp_events_high_vs),
        "decomp_events_low_vs": int(decomp_events_low_vs),
        "frac_decomp_at_high_vs": (
            (decomp_events_high_vs / decomp_events_total) if decomp_events_total else None
        ),
        # Corroboration DV (non-load-bearing): low-V_s execution-time forward PE.
        "fwd_pe_lowvs_mean": (statistics.fmean(fwd_pe_low) if fwd_pe_low else None),
        "fwd_pe_lowvs_n": len(fwd_pe_low),
        # Mechanism firing (readiness).
        "decomp_n_evaluated_precommit": int(st.get("decomp_n_evaluated_precommit", 0)),
        "decomp_n_decomposed_precommit": int(st.get("decomp_n_decomposed_precommit", 0)),
        "decomp_n_marked_unreliable": int(st.get("decomp_n_marked_unreliable", 0)),
        "decomp_n_vs_trigger": int(st.get("decomp_n_vs_trigger", 0)),
        "decomp_n_boundary_fires": int(st.get("decomp_n_boundary_fires", 0)),
        "decomp_n_bottleneck_fires": int(st.get("decomp_n_bottleneck_fires", 0)),
        "decomp_trigger_mode": st.get("decomp_trigger_mode", "vs_boundary"),
        "decomp_n_bottleneck_regions_tracked": int(
            st.get("decomp_n_bottleneck_regions_tracked", 0)),
        "decomposition_instantiated": bool(agent.policy_decomposition is not None),
        # Executed action sequence (measurement window) -- non-degeneracy check.
        "measure_action_seq": list(action_seq),
        # MECH-094 carryover (ARC-071 chunking active in all arms).
        "chunk_acc_n_replay_formed": int(chunk_st.get("chunk_acc_n_replay_formed", 0)),
    }
    return row


def _run_cell(arm: str, seed: int) -> Dict[str, Any]:
    print(f"Seed {seed} Condition {arm}", flush=True)
    # ARM_0 is minted reuse-eligible (driver excluded) so it matches V3-EXQ-816's ARM_0 and
    # any later consumer; ARM_1/ARM_2 are treatment arms (never reused as-is).
    fold_driver = arm != "ARM_0"

    if arm == "ARM_0":
        # OPT-IN reuse: reuse a banked, compatible ARM_0 mint if one exists (816's or a prior
        # 819's), else run fresh. include_driver_script_in_hash=False matches 816's emit.
        needed = ["fwd_pe_lowvs_mean", "low_vs_steps", "measure_action_seq", "n_measure_steps"]
        cached = try_reuse_cell(
            config_slice=_config_slice("ARM_0"),
            seed=seed,
            script_path=Path(__file__),
            needed_keys=needed,
            cite_run_id=REUSE_BASELINE_FROM,
            include_driver_script_in_hash=False,
        )
        if cached is not None:
            row = dict(cached)
            row.setdefault("arm", "ARM_0")
            row.setdefault("seed", seed)
            # OFF arm never decomposes; make the load-bearing fields explicit for the reused cell.
            row.setdefault("decomp_events_total", 0)
            row.setdefault("decomp_events_high_vs", 0)
            row.setdefault("decomp_events_low_vs", 0)
            row.setdefault("frac_decomp_at_high_vs", None)
            row.setdefault("decomp_n_bottleneck_fires", 0)
            print(f"verdict: {'PASS' if row.get('decomp_events_total', 0) == 0 else 'FAIL'}",
                  flush=True)
            return row

    with arm_cell(seed, config_slice=_config_slice(arm), script_path=Path(__file__),
                  include_driver_script_in_hash=fold_driver) as cell:
        row = _measure_cell(arm, seed)
        cell.stamp(row)

    if arm == "ARM_0":
        cell_ok = row["decomp_n_evaluated_precommit"] == 0
    elif arm == "ARM_1":
        cell_ok = (row["decomp_n_decomposed_precommit"] + row["decomp_n_marked_unreliable"]) >= 1
    else:  # ARM_2
        cell_ok = row["decomp_n_bottleneck_fires"] >= 1 and row["decomp_n_decomposed_precommit"] >= 1
    print(f"verdict: {'PASS' if cell_ok else 'FAIL'}", flush=True)
    return row


# ---------------------------------------------------------------------------
def run_experiment() -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for arm in ARMS:
        for seed in SEEDS:
            rows.append(_run_cell(arm, seed))
    by_arm = {a: [r for r in rows if r["arm"] == a] for a in ARMS}

    def _mean_opt(xs: List[Optional[float]]) -> Optional[float]:
        vals = [x for x in xs if x is not None]
        return statistics.fmean(vals) if vals else None

    # --- READINESS (P0). ---
    # P1: V_s heterogeneity -- the high/low-V_s partition is real (worst ARM_1 cell).
    lowvs_worst = min((r["low_vs_steps"] for r in by_arm["ARM_1"]), default=0)
    p1_vs_heterogeneity = lowvs_worst >= MIN_LOW_VS_STEPS
    # P2a: ARM_1 decomposition fires in a majority of cells.
    arm1_fired_frac = (sum(1 for r in by_arm["ARM_1"]
                           if (r["decomp_n_decomposed_precommit"] + r["decomp_n_marked_unreliable"]) >= 1)
                       / len(by_arm["ARM_1"])) if by_arm["ARM_1"] else 0.0
    p2a_arm1_fires = arm1_fired_frac >= SEED_FIRE_FRACTION
    # P2b: ARM_2 BOTTLENECK trigger fires AND decomposes in a majority of cells.
    arm2_fired_frac = (sum(1 for r in by_arm["ARM_2"]
                           if r["decomp_n_bottleneck_fires"] >= 1 and r["decomp_n_decomposed_precommit"] >= 1)
                       / len(by_arm["ARM_2"])) if by_arm["ARM_2"] else 0.0
    p2b_arm2_bottleneck_fires = arm2_fired_frac >= SEED_FIRE_FRACTION
    # P3: enough decomposition events in BOTH treatment arms to estimate a fraction.
    arm1_events_worst = min((r["decomp_events_total"] for r in by_arm["ARM_1"]), default=0)
    arm2_events_worst = min((r["decomp_events_total"] for r in by_arm["ARM_2"]), default=0)
    arm1_events_total = sum(r["decomp_events_total"] for r in by_arm["ARM_1"])
    arm2_events_total = sum(r["decomp_events_total"] for r in by_arm["ARM_2"])
    p3_enough_events = (arm1_events_total >= MIN_DECOMP_EVENTS
                        and arm2_events_total >= MIN_DECOMP_EVENTS)

    readiness_met = (p1_vs_heterogeneity and p2a_arm1_fires
                     and p2b_arm2_bottleneck_fires and p3_enough_events)

    # --- Load-bearing evidence criterion (V_s-independence of ARM_2's trigger). ---
    # Pool decomposition events across seeds within each arm (a fraction over the pooled
    # event set; per-cell fractions are noisy at low event counts).
    def _pooled_high_vs_frac(arm: str) -> Optional[float]:
        hi = sum(r["decomp_events_high_vs"] for r in by_arm[arm])
        tot = sum(r["decomp_events_total"] for r in by_arm[arm])
        return (hi / tot) if tot else None

    arm1_high_vs_frac = _pooled_high_vs_frac("ARM_1")
    arm2_high_vs_frac = _pooled_high_vs_frac("ARM_2")
    dissociation = (
        (arm2_high_vs_frac - arm1_high_vs_frac)
        if (arm1_high_vs_frac is not None and arm2_high_vs_frac is not None) else None
    )
    arm2_high_vs_ok = (arm2_high_vs_frac is not None
                       and arm2_high_vs_frac >= HIGH_VS_DECOMP_FLOOR)
    dissociation_ok = (dissociation is not None and dissociation >= DISSOCIATION_MARGIN)
    c_main_supports = readiness_met and arm2_high_vs_ok and dissociation_ok

    # --- Corroboration (non-load-bearing): convergence on repeated structure --
    # do BOTH ARM_1 and ARM_2 reduce low-V_s forward PE vs ARM_0 (paired-ish, arm means)?
    off_fwd = _mean_opt([r["fwd_pe_lowvs_mean"] for r in by_arm["ARM_0"]])
    arm1_fwd = _mean_opt([r["fwd_pe_lowvs_mean"] for r in by_arm["ARM_1"]])
    arm2_fwd = _mean_opt([r["fwd_pe_lowvs_mean"] for r in by_arm["ARM_2"]])
    c_converge = (off_fwd is not None and arm1_fwd is not None and arm2_fwd is not None
                  and arm1_fwd <= off_fwd and arm2_fwd <= off_fwd)

    # --- Non-degeneracy: ARM_2's manipulation must change the executed trajectory on >= 1
    # seed AND its bottleneck trigger must have actually fired. Else scoring-EXCLUDED.
    def _action_divergence(treatment: str) -> bool:
        off = {r["seed"]: r for r in by_arm["ARM_0"]}
        return any(
            r["seed"] in off and off[r["seed"]].get("measure_action_seq") != r.get("measure_action_seq")
            for r in by_arm[treatment]
        )
    arm2_action_divergence = _action_divergence("ARM_2")
    non_degenerate = bool(readiness_met and arm2_action_divergence)

    if not readiness_met:
        if not p1_vs_heterogeneity:
            degeneracy_reason = (f"V_s never dropped below threshold sufficiently (worst ARM_1 "
                                 f"cell had {lowvs_worst} low-V_s steps < {MIN_LOW_VS_STEPS}); "
                                 "the high/low-V_s partition is not exercisable")
        elif not p2a_arm1_fires:
            degeneracy_reason = (f"ARM_1 (V_s-drop) decomposition fired in only "
                                 f"{arm1_fired_frac:.2f} of cells (< {SEED_FIRE_FRACTION})")
        elif not p2b_arm2_bottleneck_fires:
            degeneracy_reason = (f"ARM_2 bottleneck trigger fired+decomposed in only "
                                 f"{arm2_fired_frac:.2f} of cells (< {SEED_FIRE_FRACTION}); this "
                                 "env lacks recurring funnel structure at these bottleneck "
                                 "params -- substrate/params not ready, NOT a falsification")
        else:
            degeneracy_reason = (f"too few decomposition events to estimate frac_decomp_at_high_vs "
                                 f"(ARM_1 total {arm1_events_total}, ARM_2 total {arm2_events_total}, "
                                 f"need >= {MIN_DECOMP_EVENTS} each)")
        label = "substrate_not_ready_requeue"
    elif not non_degenerate:
        degeneracy_reason = ("readiness met but ARM_2 produced action sequences identical to "
                             "ARM_0 on every seed -- the bottleneck manipulation was "
                             "behaviourally inert")
        label = "manipulation_inert"
    else:
        degeneracy_reason = None
        if c_main_supports:
            label = "bottleneck_trigger_is_vs_independent"
        elif dissociation is not None and dissociation <= 0:
            label = "bottleneck_trigger_tracks_vs_like_r1"   # R1 and R5 do NOT dissociate
        else:
            label = "bottleneck_dissociation_below_threshold"

    # evidence_direction: only a non-degenerate, readiness-met run weighs the claim.
    if not non_degenerate:
        direction = "unknown"
    elif c_main_supports:
        direction = "supports"
    else:
        # ARM_2 gated on V_s just like ARM_1 (no dissociation) with readiness met would weaken
        # the R5 "distinct mechanism" claim.
        direction = "weakens"

    outcome = "PASS" if c_main_supports else "FAIL"

    metrics = {
        "readiness_met": readiness_met,
        "p1_vs_heterogeneity": p1_vs_heterogeneity,
        "p2a_arm1_fires": p2a_arm1_fires,
        "p2b_arm2_bottleneck_fires": p2b_arm2_bottleneck_fires,
        "p3_enough_events": p3_enough_events,
        "lowvs_worst_arm1": lowvs_worst,
        "arm1_decomp_fired_frac": arm1_fired_frac,
        "arm2_bottleneck_fired_frac": arm2_fired_frac,
        "arm1_events_total": arm1_events_total,
        "arm2_events_total": arm2_events_total,
        "arm1_events_worst": arm1_events_worst,
        "arm2_events_worst": arm2_events_worst,
        # Load-bearing.
        "arm1_high_vs_decomp_frac": arm1_high_vs_frac,
        "arm2_high_vs_decomp_frac": arm2_high_vs_frac,
        "dissociation_high_vs_frac": dissociation,
        "arm2_high_vs_ok": arm2_high_vs_ok,
        "dissociation_ok": dissociation_ok,
        "c_main_supports": c_main_supports,
        # Corroboration.
        "off_lowvs_fwd_pe_mean": off_fwd,
        "arm1_lowvs_fwd_pe_mean": arm1_fwd,
        "arm2_lowvs_fwd_pe_mean": arm2_fwd,
        "c_converge_on_repeated_structure": c_converge,
        # Firing audit (both treatment arms).
        "arm1_vs_trigger_total": sum(r["decomp_n_vs_trigger"] for r in by_arm["ARM_1"]),
        "arm2_vs_trigger_total": sum(r["decomp_n_vs_trigger"] for r in by_arm["ARM_2"]),
        "arm2_bottleneck_fires_total": sum(r["decomp_n_bottleneck_fires"] for r in by_arm["ARM_2"]),
        "arm2_action_divergence": arm2_action_divergence,
        "n_replay_formed_total": sum(r["chunk_acc_n_replay_formed"] for r in rows),
    }

    interpretation = {
        "label": label,
        "preconditions": [
            {"name": "vs_heterogeneity_low_vs_steps_present",
             "description": ("The high/low-V_s partition (and ARM_1's V_s trigger) require "
                             "region V_s < threshold to occur. Worst ARM_1 cell low-V_s steps."),
             "measured": float(lowvs_worst), "threshold": float(MIN_LOW_VS_STEPS),
             "direction": "lower", "control": "worst ARM_1 cell", "met": p1_vs_heterogeneity},
            {"name": "arm1_vs_drop_decomposition_fires",
             "description": "Fraction of ARM_1 cells where V_s-drop decomposition fired.",
             "measured": float(arm1_fired_frac), "threshold": float(SEED_FIRE_FRACTION),
             "direction": "lower", "control": "ARM_1 cells", "met": p2a_arm1_fires},
            {"name": "arm2_bottleneck_decomposition_fires",
             "description": ("READINESS: ARM_2's bottleneck trigger must fire AND decompose "
                             "(env has recurring funnel structure at these params). The SAME "
                             "statistic the load-bearing frac_decomp_at_high_vs is computed "
                             "over -- if ARM_2 never decomposes there is no locus to measure."),
             "measured": float(arm2_fired_frac), "threshold": float(SEED_FIRE_FRACTION),
             "direction": "lower", "control": "ARM_2 cells (positive control for the R5 trigger)",
             "met": p2b_arm2_bottleneck_fires},
            {"name": "enough_decomposition_events_both_arms",
             "description": ("frac_decomp_at_high_vs is estimable only with enough events. "
                             "Worst of the two arms' pooled event totals."),
             "measured": float(min(arm1_events_total, arm2_events_total)),
             "threshold": float(MIN_DECOMP_EVENTS),
             "direction": "lower", "control": "min over ARM_1/ARM_2 pooled events",
             "met": p3_enough_events},
        ],
        "criteria": [
            {"name": "C_MAIN_arm2_decomposes_at_high_vs", "load_bearing": True, "passed": c_main_supports},
            {"name": "C_CONVERGE_both_reduce_lowvs_forward_pe", "load_bearing": False, "passed": c_converge},
        ],
        "criteria_non_degenerate": {
            "C_MAIN": non_degenerate,
            "C_CONVERGE": non_degenerate,
        },
    }

    return {
        "outcome": outcome,
        "evidence_direction": direction,
        "metrics": metrics,
        "per_seed_rows": rows,
        "arm_results": rows,
        "interpretation": interpretation,
        "non_degenerate": non_degenerate,
        "degeneracy_reason": degeneracy_reason,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    t0 = time.perf_counter()

    global SEEDS, WARMUP_EPISODES, MEASURE_EPISODES
    if args.dry_run:
        SEEDS = [11, 23]
        WARMUP_EPISODES = 3
        MEASURE_EPISODES = 4

    result = run_experiment()
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    full_config = {
        "seeds": SEEDS, "arms": ARMS,
        "warmup_episodes": WARMUP_EPISODES, "measure_episodes": MEASURE_EPISODES,
        "steps_per_episode": STEPS_PER_EPISODE,
        "decomposition_vs_threshold": DECOMPOSITION_VS_THRESHOLD,
        "decomposition_depth_cap": DECOMPOSITION_DEPTH_CAP,
        "bottleneck_min_visits": BOTTLENECK_MIN_VISITS,
        "bottleneck_min_distinct_neighbors": BOTTLENECK_MIN_DISTINCT_NEIGHBORS,
        "bottleneck_region_quant": BOTTLENECK_REGION_QUANT,
        "bottleneck_region_dims": BOTTLENECK_REGION_DIMS,
        "seeded_chunk_sequence": list(SEEDED_CHUNK_SEQUENCE),
        "high_vs_decomp_floor": HIGH_VS_DECOMP_FLOOR,
        "dissociation_margin": DISSOCIATION_MARGIN,
        "min_low_vs_steps": MIN_LOW_VS_STEPS,
        "min_decomp_events": MIN_DECOMP_EVENTS,
        "seed_fire_fraction": SEED_FIRE_FRACTION,
        "reuse_baseline_from": REUSE_BASELINE_FROM,
        "arm_config_slices": {a: _config_slice(a) for a in ARMS},
    }
    manifest = {
        "run_id": f"{EXPERIMENT_TYPE}_{ts}_v3",
        "experiment_type": EXPERIMENT_TYPE,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "evidence_direction": result["evidence_direction"],
        "evidence_direction_per_claim": {
            "ARC-070": result["evidence_direction"],
            "MECH-321": result["evidence_direction"],
        },
        "outcome": result["outcome"],
        "timestamp_utc": ts,
        "metrics": result["metrics"],
        "per_seed_rows": result["per_seed_rows"],
        "arm_results": result["arm_results"],
        "interpretation": result["interpretation"],
        "non_degenerate": result["non_degenerate"],
        "degeneracy_reason": result["degeneracy_reason"],
        "sleep_driver_pattern": None,
        "reuse_baseline_from": REUSE_BASELINE_FROM,
        "cites": {"arm_0_arm_1_half": "V3-EXQ-816"},
    }
    # Multi-arm: stamp AFTER arm_results is assembled so substrate_hash HOISTS from per-cell
    # fingerprints (ARM_1/ARM_2 folded-driver, ARM_0 excluded-driver -- consistent per cell).
    stamp_recording_core(manifest, config=full_config, seeds=SEEDS,
                         script_path=Path(__file__), started_at=t0)

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
    print(f"direction: {result['evidence_direction']} non_degenerate: {result['non_degenerate']}", flush=True)
    print(f"readiness_met={m['readiness_met']} (P1={m['p1_vs_heterogeneity']} "
          f"P2a={m['p2a_arm1_fires']} P2b={m['p2b_arm2_bottleneck_fires']} P3={m['p3_enough_events']})",
          flush=True)
    print(f"arm2_high_vs_frac={m['arm2_high_vs_decomp_frac']} "
          f"arm1_high_vs_frac={m['arm1_high_vs_decomp_frac']} "
          f"dissociation={m['dissociation_high_vs_frac']}", flush=True)
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
