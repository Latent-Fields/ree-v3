"""V3-EXQ-816 -- ARC-070 / MECH-321 discriminative validation (ARM_0 vs ARM_1).

PURPOSE (evidence; governance-weighting). V3-EXQ-815 answered the readiness
question -- MECH-321.evaluate() is called and the CEM pool actually mutates. This
run answers the BEHAVIOURAL question from MECH-321's claims.yaml
functional_restatement falsifiable-prediction block: does V_s-drop-triggered
decomposition change what the agent commits to, and does that reduce
execution-time forward-prediction failure relative to the OFF baseline? This is
the evidence ARC-070 / MECH-321 need to clear v3_pending.

    ARM_0 (use_policy_decomposition=False): the full MECH-321 substrate stack is
        ON (event_segmenter + policy_chunking + chunk injection + per_stream_vs),
        so ARC-071 chunks form and are injected into the CEM pool and can be
        committed and executed BLINDLY, paying prediction-failure cost at
        execution. No decomposition ever withholds/replaces a chunk.
    ARM_1 (use_policy_decomposition=True, R1 V_s-drop OR MECH-288-boundary
        trigger -- the ONLY trigger the substrate implements): a chunk whose
        region shows low V_s (or on which MECH-288 fires on the rolled-out
        latent trajectory) is withheld pre-commitment and replaced with
        finer-grain leaf tiles, or marked unreliable at the depth cap.

WHAT IS DELIBERATELY NOT HERE -- ARM_2. MECH-321's functional_restatement
specifies a third arm (ARM_2: bottleneck-state PRIMARY trigger, R5 alternative,
"decompose only at bottleneck states regardless of V_s"). The substrate
(ree_core/policy/policy_decomposition.py evaluate()) implements ONLY the R1
trigger `vs_trigger or boundary_fired` -- there is NO bottleneck-state trigger
and no config to select one. R5 is flagged OPTIONAL / deferred in the
functional_restatement itself ("Defer until ARC-071 child-MECH design lands").
ARM_2 is therefore a substrate build routed to /implement-substrate (chip spawned
2026-07-25); a successor experiment adds ARM_2 on top of this ARM_0/ARM_1 design
once that trigger lands. ARM_0 here is minted reuse-eligible (see
experiments/_lib/baselines/mech321_policy_decomposition.py) so the successor can
reuse it rather than re-train it.

GOV-REUSE-1 (Step 2.4). The decisive readouts are the per-arm, low-V_s-region
mean execution-time forward-prediction error (e3_prediction_error) and rollout
deliberation cost under use_policy_decomposition ON vs OFF. use_policy_decomposition
did not exist before 2026-07-24; V3-EXQ-815 is the only prior ARC-070/MECH-321 run
and it recorded FIRING COUNTS on a NON-LEARNING loop (no update_residue call), not
a cross-arm behavioural forward-PE contrast on a trained forward model. No recorded
manifest on any substrate_hash carries this readout -> not recoverable -> run.

WHY V_s HETEROGENEITY IS AN EMPIRICAL, NOT A RIGGED, PREMISE. The implemented
region V_s (HippocampalModule._region_vs -> mean per_stream_vs) is a
`1 - relative-tick-to-tick-latent-change` stability proxy, NOT forward-model PE.
So "a low-V_s region is one where a committed chunk's forward prediction fails"
is exactly the hypothesis under test, not an identity. The hazard + native
env-drift environment produces regions where the encoded latent moves fast
(low V_s); whether decomposing chunks there reduces EXECUTION-time forward PE is
what ARM_1-vs-ARM_0 measures. ARM_1 == ARM_0 falsifies the primary V_s-drop
trigger (functional_restatement), i.e. weakens the claim.

DV-SYMMETRY (Step 3 mandatory per-arm declaration).
  Load-bearing DV = mean execution-time forward-prediction error
    (e3_prediction_error, from agent.update_residue) over the measurement window,
    restricted to low-V_s steps, contrasted ARM_1 vs ARM_0 as a per-seed PAIRED
    delta. Symmetry group of the executed-trajectory forward PE: it is a sum of
    per-step squared-error terms; the group is permutation of steps and
    relabelling of action-class indices. The manipulation (use_policy_decomposition)
    is NOT invariant under it: turning decomposition ON withholds/replaces/aborts
    a COMMITTED chunk in triggering regions, changing WHICH action the agent
    executes -- so the executed action sequence, and hence the per-step forward-PE
    terms, differ in cardinality and content. Verified, not assumed: non_degenerate
    requires the ARM_0 and ARM_1 measurement-window executed action sequences to
    actually differ on at least one seed (else the manipulation is inert and this
    IS a symmetry trap -- the run is scoring-excluded, not scored as weakens).
  NOT a broadcast-scalar-on-argmax case (the 604c hazard): forward PE is measured
    on the E2 first-step rollout of the COMMITTED trajectory vs the observed
    z_world, not derived from an argmax/order-statistic over the CEM candidate
    pool. Deliberation cost (candidate pool size) is a genuine cardinality, not a
    rank readout.
  Secondary corroboration DV = rollout deliberation cost (candidate_samples_collected
    per e3-tick). Predicted higher locally in ARM_1 (a withheld chunk is replaced by
    N leaf tiles). NON-load-bearing: a null here (e.g. the CEM pool is budget-capped)
    does not sink the run; the forward-PE contrast is load-bearing.

READINESS (P0). A cross-arm forward-PE contrast is meaningful only if (P1) V_s
actually drops below threshold somewhere (the trigger CAN fire), (P2) decomposition
actually fires in ARM_1, and (P3) the committed-trajectory forward PE is a real,
bounded, VARYING quantity on the OFF positive control (world_forward learned
something -- an untrained forward model gives z_world deltas that do not track
moves, per e3_selector/agent source, so the DV would be noise). If any fails the
run self-routes substrate_not_ready_requeue with non_degenerate=False (scoring-
excluded), NEVER a falsification of the claim.

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
from experiments._lib.baselines import mech321_policy_decomposition as base  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_816_mech321_policy_decomposition_discriminative"
EXPERIMENT_PURPOSE = "evidence"
CLAIM_IDS = ["ARC-070", "MECH-321"]

SEEDS = [11, 23, 47, 71, 97]
ARMS = ["ARM_0", "ARM_1"]

# Schedule / env come from the canonical baseline module so ARM_0 matches the
# successor ARM_2 run BY CONSTRUCTION (see baseline module docstring).
WARMUP_EPISODES = base.WARMUP_EPISODES
MEASURE_EPISODES = base.MEASURE_EPISODES
STEPS_PER_EPISODE = base.STEPS_PER_EPISODE
SEEDED_CHUNK_SEQUENCE = base.SEEDED_CHUNK_SEQUENCE

# Decomposition parameters (ARM_1 only; registered defaults). Threshold 0.5 (as in
# V3-EXQ-815) rather than the substrate default 0.4, to give the V_s-drop half of
# the OR trigger a realistic chance of firing on this env's stability distribution.
DECOMPOSITION_VS_THRESHOLD = 0.5
DECOMPOSITION_DEPTH_CAP = 3          # mirrors ARC-071 chunk_max_depth (substrate default)

# --- Pre-registered thresholds (defined HERE, never inferred post-hoc). ---
# Load-bearing: ARM_1 must reduce low-V_s-region execution-time forward PE vs ARM_0
# by a RELATIVE margin AND clear an effect-size gate on the SD of the PAIRED delta.
REL_IMPROVEMENT_FLOOR = 0.05        # >=5% relative reduction in low-V_s forward PE
EFFECT_SIZE_K_SIGMA = 2.0           # paired-delta mean must exceed K * SE(delta)
# Readiness floors.
PE_SANITY_CEIL = 1.0e3              # forward PE explosion / NaN guard (upper bound)
MIN_LOW_VS_STEPS = 5               # a cell must have >= this many low-V_s steps to
                                    # contribute a low-V_s mean (else its delta is NaN)
SEED_FIRE_FRACTION = 3.0 / 5.0      # decomposition must fire in a majority of ARM_1 cells


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
    return flags


def _config_slice(arm: str) -> Dict[str, Any]:
    if arm == "ARM_0":
        # Single source of truth -- MUST match the successor's reuse call.
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
    # Seed the identical crystallised chunk in BOTH arms (only ARM_1 can decompose
    # it). ARC-071 formation state survives agent.reset() by design.
    chunk = ChunkedPrimitive(
        sequence=SEEDED_CHUNK_SEQUENCE, depth=1,
        state=ChunkState.CRYSTALLISED, selection_weight=1.0,
    )
    agent.policy_chunking.library.register(chunk)
    return agent


def _run_cell(arm: str, seed: int) -> Dict[str, Any]:
    print(f"Seed {seed} Condition {arm}", flush=True)
    total_eps = WARMUP_EPISODES + MEASURE_EPISODES
    # ARM_0 is minted reuse-eligible (driver excluded from the hash) so the ARM_2
    # successor can reuse it; ARM_1 is a treatment arm (never reused as-is).
    fold_driver = arm != "ARM_0"
    with arm_cell(seed, config_slice=_config_slice(arm), script_path=Path(__file__),
                  include_driver_script_in_hash=fold_driver) as cell:
        env = _build_env(seed)
        agent = _build_agent(env, arm)
        wd = agent.config.latent.world_dim

        fwd_pe_low: List[float] = []
        fwd_pe_all: List[float] = []
        delib_low: List[int] = []
        delib_all: List[int] = []
        action_seq: List[int] = []
        net_harm = 0.0
        n_measure_steps = 0
        low_vs_steps = 0

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
                # Full waking loop: update_residue trains the forward model AND
                # returns the committed-trajectory execution-time forward PE.
                metrics = agent.update_residue(harm_signal)

                if measuring:
                    region_vs = float(agent.hippocampal._region_vs())
                    is_low = region_vs < DECOMPOSITION_VS_THRESHOLD
                    pe_raw = metrics.get("e3_prediction_error")
                    if pe_raw is not None:
                        pe = float(pe_raw.detach()) if torch.is_tensor(pe_raw) else float(pe_raw)
                        if math.isfinite(pe):
                            fwd_pe_all.append(pe)
                            if is_low:
                                fwd_pe_low.append(pe)
                    # Deliberation cost only on a fresh proposal (e3-tick); between
                    # e3-ticks generate_trajectories returns cached candidates and
                    # _last_propose_diagnostics is stale.
                    if ticks.get("e3_tick"):
                        pd = agent.hippocampal.get_last_propose_diagnostics() or {}
                        pool = int(pd.get("candidate_samples_collected", len(cands)))
                        delib_all.append(pool)
                        if is_low:
                            delib_low.append(pool)
                    net_harm += float(harm_signal)
                    n_measure_steps += 1
                    if is_low:
                        low_vs_steps += 1
                    action_seq.append(a_idx)
                if done:
                    break
            if (ep + 1) % 10 == 0:
                st = agent.get_policy_decomposition_state()
                print(f"  [train] decomp seed={seed} arm={arm} ep {ep+1}/{total_eps} "
                      f"decomposed={st.get('decomp_n_decomposed_precommit', 0)} "
                      f"marked={st.get('decomp_n_marked_unreliable', 0)} "
                      f"low_vs_steps={low_vs_steps}", flush=True)

        st = agent.get_policy_decomposition_state()
        chunk_st = agent.get_chunking_state()

        def _mean(xs: List[float]) -> Optional[float]:
            return statistics.fmean(xs) if xs else None

        row = {
            "arm": arm,
            "seed": seed,
            "n_measure_steps": int(n_measure_steps),
            "low_vs_steps": int(low_vs_steps),
            "low_vs_step_frac": (low_vs_steps / n_measure_steps) if n_measure_steps else 0.0,
            # Load-bearing DV inputs.
            "fwd_pe_lowvs_mean": _mean(fwd_pe_low),
            "fwd_pe_all_mean": _mean(fwd_pe_all),
            "fwd_pe_lowvs_n": len(fwd_pe_low),
            "fwd_pe_all_n": len(fwd_pe_all),
            "fwd_pe_all_var": (statistics.pvariance(fwd_pe_all) if len(fwd_pe_all) > 1 else 0.0),
            # Secondary corroboration DV.
            "delib_cost_lowvs_mean": _mean([float(x) for x in delib_low]),
            "delib_cost_all_mean": _mean([float(x) for x in delib_all]),
            # Task-performance context.
            "net_harm_per_step": (net_harm / n_measure_steps) if n_measure_steps else 0.0,
            # Mechanism firing (readiness).
            "decomp_n_evaluated_precommit": int(st.get("decomp_n_evaluated_precommit", 0)),
            "decomp_n_evaluated_midexec": int(st.get("decomp_n_evaluated_midexec", 0)),
            "decomp_n_decomposed_precommit": int(st.get("decomp_n_decomposed_precommit", 0)),
            "decomp_n_marked_unreliable": int(st.get("decomp_n_marked_unreliable", 0)),
            "decomp_n_vs_trigger": int(st.get("decomp_n_vs_trigger", 0)),
            "decomp_n_boundary_fires": int(st.get("decomp_n_boundary_fires", 0)),
            "decomposition_instantiated": bool(agent.policy_decomposition is not None),
            # Executed action sequence (measurement window) -- non-degeneracy check.
            "measure_action_seq": list(action_seq),
            # MECH-094 carryover (ARC-071 chunking active in both arms).
            "chunk_acc_n_replay_formed": int(chunk_st.get("chunk_acc_n_replay_formed", 0)),
        }
        cell.stamp(row)

    fired = row["decomp_n_decomposed_precommit"] + row["decomp_n_marked_unreliable"]
    cell_ok = (fired >= 1) if arm == "ARM_1" else (row["decomp_n_evaluated_precommit"] == 0)
    print(f"verdict: {'PASS' if cell_ok else 'FAIL'}", flush=True)
    return row


# ---------------------------------------------------------------------------
def _paired_deltas(by_arm: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Per-seed paired ARM_0-minus-ARM_1 low-V_s forward-PE deltas.

    A positive delta = ARM_1 has LOWER execution-time forward PE (improvement).
    Only seeds where BOTH arms produced >= MIN_LOW_VS_STEPS low-V_s observations
    contribute (else the low-V_s mean is unreliable / NaN).
    """
    off = {r["seed"]: r for r in by_arm["ARM_0"]}
    on = {r["seed"]: r for r in by_arm["ARM_1"]}
    out: List[Dict[str, Any]] = []
    for seed in sorted(set(off) & set(on)):
        ro, rn = off[seed], on[seed]
        if (ro["fwd_pe_lowvs_mean"] is None or rn["fwd_pe_lowvs_mean"] is None
                or ro["low_vs_steps"] < MIN_LOW_VS_STEPS
                or rn["low_vs_steps"] < MIN_LOW_VS_STEPS):
            continue
        out.append({
            "seed": seed,
            "off_lowvs": float(ro["fwd_pe_lowvs_mean"]),
            "on_lowvs": float(rn["fwd_pe_lowvs_mean"]),
            "delta": float(ro["fwd_pe_lowvs_mean"]) - float(rn["fwd_pe_lowvs_mean"]),
            "action_seq_differs": ro["measure_action_seq"] != rn["measure_action_seq"],
        })
    return out


def run_experiment() -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for arm in ARMS:
        for seed in SEEDS:
            rows.append(_run_cell(arm, seed))
    by_arm = {a: [r for r in rows if r["arm"] == a] for a in ARMS}

    # --- READINESS (P0). ---
    # P1: V_s heterogeneity -- the trigger CAN fire (worst ARM_1 cell has low-V_s steps).
    lowvs_worst = min((r["low_vs_steps"] for r in by_arm["ARM_1"]), default=0)
    p1_vs_heterogeneity = lowvs_worst >= MIN_LOW_VS_STEPS
    # P2: decomposition actually fires in a majority of ARM_1 cells.
    fired_frac = (sum(1 for r in by_arm["ARM_1"]
                      if (r["decomp_n_decomposed_precommit"] + r["decomp_n_marked_unreliable"]) >= 1)
                  / len(by_arm["ARM_1"])) if by_arm["ARM_1"] else 0.0
    p2_decomp_fires = fired_frac >= SEED_FIRE_FRACTION
    # P3: forward PE is a real, bounded, VARYING quantity on the OFF positive control.
    off_pe_var_worst = min((r["fwd_pe_all_var"] for r in by_arm["ARM_0"]), default=0.0)
    off_pe_mean_worst = max((r["fwd_pe_all_mean"] or float("inf")) for r in by_arm["ARM_0"]) \
        if by_arm["ARM_0"] else float("inf")
    p3_pe_meaningful = (off_pe_var_worst > 1e-12) and (off_pe_mean_worst < PE_SANITY_CEIL)

    readiness_met = p1_vs_heterogeneity and p2_decomp_fires and p3_pe_meaningful

    # --- Load-bearing evidence criterion (only meaningful once readiness holds). ---
    deltas = _paired_deltas(by_arm)
    delta_vals = [d["delta"] for d in deltas]
    n_pairs = len(delta_vals)
    delta_mean = statistics.fmean(delta_vals) if delta_vals else 0.0
    delta_sd = statistics.stdev(delta_vals) if n_pairs > 1 else 0.0
    delta_se = (delta_sd / math.sqrt(n_pairs)) if n_pairs > 0 else 0.0
    off_lowvs_ref = statistics.fmean([d["off_lowvs"] for d in deltas]) if deltas else 0.0
    rel_improvement = (delta_mean / off_lowvs_ref) if off_lowvs_ref > 0 else 0.0

    effect_size_ok = (n_pairs >= 2) and (delta_mean > EFFECT_SIZE_K_SIGMA * delta_se)
    rel_floor_ok = rel_improvement >= REL_IMPROVEMENT_FLOOR
    c_main_supports = readiness_met and effect_size_ok and rel_floor_ok

    # --- Secondary corroboration (non-load-bearing). ---
    on_delib = statistics.fmean([r["delib_cost_lowvs_mean"] for r in by_arm["ARM_1"]
                                 if r["delib_cost_lowvs_mean"] is not None]) \
        if any(r["delib_cost_lowvs_mean"] is not None for r in by_arm["ARM_1"]) else None
    off_delib = statistics.fmean([r["delib_cost_lowvs_mean"] for r in by_arm["ARM_0"]
                                  if r["delib_cost_lowvs_mean"] is not None]) \
        if any(r["delib_cost_lowvs_mean"] is not None for r in by_arm["ARM_0"]) else None
    c_delib_corroborates = (on_delib is not None and off_delib is not None
                            and on_delib >= off_delib)

    # --- Non-degeneracy: the manipulation must actually change the executed
    # trajectory on >= 1 seed, AND decomposition must have fired. Else this is a
    # DV-symmetry trap (forward PE identical by construction) or a silent
    # substrate, and the run is scoring-EXCLUDED rather than scored as weakens.
    any_action_divergence = any(d["action_seq_differs"] for d in deltas) or any(
        by_arm["ARM_0"][i]["measure_action_seq"] != by_arm["ARM_1"][i]["measure_action_seq"]
        for i in range(min(len(by_arm["ARM_0"]), len(by_arm["ARM_1"])))
    )
    non_degenerate = bool(readiness_met and any_action_divergence)

    if not readiness_met:
        if not p1_vs_heterogeneity:
            degeneracy_reason = ("V_s never dropped below decomposition_vs_threshold in "
                                 "a majority-sufficient way (worst ARM_1 cell had "
                                 f"{lowvs_worst} low-V_s steps < {MIN_LOW_VS_STEPS}); the "
                                 "R1 trigger's V_s half could not be exercised")
        elif not p2_decomp_fires:
            degeneracy_reason = (f"decomposition fired in only {fired_frac:.2f} of ARM_1 "
                                 f"cells (< {SEED_FIRE_FRACTION}); no behavioural contrast "
                                 "to measure")
        else:
            degeneracy_reason = ("OFF-arm committed-trajectory forward PE was constant / "
                                 "unbounded (world_forward not action-conditional at this "
                                 "horizon); the forward-PE DV is not meaningful")
        label = "substrate_not_ready_requeue"
    elif not non_degenerate:
        degeneracy_reason = ("readiness met but ARM_0 and ARM_1 produced identical executed "
                             "action sequences on every seed -- the decomposition manipulation "
                             "was behaviourally inert (DV-symmetry: forward PE identical by "
                             "construction)")
        label = "manipulation_inert"
    else:
        degeneracy_reason = None
        if c_main_supports:
            label = "decomposition_reduces_execution_prediction_failure"
        elif delta_mean <= 0:
            label = "decomposition_does_not_reduce_prediction_failure"  # ARM_1 ~= or worse than ARM_0
        else:
            label = "decomposition_effect_below_threshold"

    # evidence_direction: only a run that both is non-degenerate AND has readiness
    # met can weigh the claim; otherwise scoring-excluded (unknown).
    if not non_degenerate:
        direction = "unknown"
    elif c_main_supports:
        direction = "supports"
    else:
        # ARM_1 == ARM_0 (or worse) with readiness met falsifies the primary
        # V_s-drop trigger, per the functional_restatement.
        direction = "weakens"

    outcome = "PASS" if c_main_supports else "FAIL"

    metrics = {
        "readiness_met": readiness_met,
        "p1_vs_heterogeneity": p1_vs_heterogeneity,
        "p2_decomp_fires": p2_decomp_fires,
        "p3_pe_meaningful": p3_pe_meaningful,
        "lowvs_worst_arm1": lowvs_worst,
        "decomp_fired_frac_arm1": fired_frac,
        "off_pe_var_worst": off_pe_var_worst,
        "off_pe_mean_worst": off_pe_mean_worst,
        "n_paired_seeds": n_pairs,
        "delta_mean_lowvs_fwd_pe": delta_mean,
        "delta_sd": delta_sd,
        "delta_se": delta_se,
        "rel_improvement": rel_improvement,
        "effect_size_ok": effect_size_ok,
        "rel_floor_ok": rel_floor_ok,
        "c_main_supports": c_main_supports,
        "c_delib_corroborates": c_delib_corroborates,
        "on_lowvs_delib_mean": on_delib,
        "off_lowvs_delib_mean": off_delib,
        "any_action_divergence": any_action_divergence,
        "off_net_harm_per_step_mean": statistics.fmean([r["net_harm_per_step"] for r in by_arm["ARM_0"]]),
        "on_net_harm_per_step_mean": statistics.fmean([r["net_harm_per_step"] for r in by_arm["ARM_1"]]),
        "n_replay_formed_total": sum(r["chunk_acc_n_replay_formed"] for r in rows),
    }

    interpretation = {
        "label": label,
        "preconditions": [
            {"name": "vs_heterogeneity_low_vs_steps_present",
             "description": ("The R1 V_s-drop trigger can only fire where region V_s < "
                             "threshold. Asserts the worst ARM_1 cell has >= MIN_LOW_VS_STEPS "
                             "low-V_s measurement steps -- the same quantity the low-V_s "
                             "forward-PE mean is computed over, not a proxy."),
             "measured": float(lowvs_worst), "threshold": float(MIN_LOW_VS_STEPS),
             "direction": "lower", "control": "worst ARM_1 cell", "met": p1_vs_heterogeneity},
            {"name": "decomposition_fires_in_arm1",
             "description": "Fraction of ARM_1 cells where decomposition actually fired.",
             "measured": float(fired_frac), "threshold": float(SEED_FIRE_FRACTION),
             "direction": "lower", "control": "ARM_1 cells", "met": p2_decomp_fires},
            {"name": "off_forward_pe_varies",
             "description": ("OFF positive control: committed-trajectory forward PE must have "
                             "non-zero variance (world_forward learned something). Worst-cell "
                             "variance."),
             "measured": float(off_pe_var_worst), "threshold": 1e-12,
             "direction": "lower", "control": "worst ARM_0 cell", "met": (off_pe_var_worst > 1e-12)},
            {"name": "off_forward_pe_bounded",
             "description": "OFF positive control: forward PE bounded (no explosion / NaN).",
             "measured": float(off_pe_mean_worst), "threshold": float(PE_SANITY_CEIL),
             "direction": "upper", "control": "worst ARM_0 cell",
             "met": (off_pe_mean_worst < PE_SANITY_CEIL)},
        ],
        "criteria": [
            {"name": "C_MAIN_lowvs_forward_pe_reduced", "load_bearing": True, "passed": c_main_supports},
            {"name": "C_DELIB_deliberation_cost_rises", "load_bearing": False, "passed": c_delib_corroborates},
        ],
        "criteria_non_degenerate": {
            "C_MAIN": non_degenerate,
            "C_DELIB": non_degenerate,
        },
    }

    return {
        "outcome": outcome,
        "evidence_direction": direction,
        "metrics": metrics,
        "per_seed_rows": rows,
        "arm_results": rows,
        "paired_deltas": deltas,
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
        WARMUP_EPISODES = 2
        MEASURE_EPISODES = 2

    result = run_experiment()
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    full_config = {
        "seeds": SEEDS, "arms": ARMS,
        "warmup_episodes": WARMUP_EPISODES, "measure_episodes": MEASURE_EPISODES,
        "steps_per_episode": STEPS_PER_EPISODE,
        "decomposition_vs_threshold": DECOMPOSITION_VS_THRESHOLD,
        "decomposition_depth_cap": DECOMPOSITION_DEPTH_CAP,
        "seeded_chunk_sequence": list(SEEDED_CHUNK_SEQUENCE),
        "rel_improvement_floor": REL_IMPROVEMENT_FLOOR,
        "effect_size_k_sigma": EFFECT_SIZE_K_SIGMA,
        "min_low_vs_steps": MIN_LOW_VS_STEPS,
        "seed_fire_fraction": SEED_FIRE_FRACTION,
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
        "paired_deltas": result["paired_deltas"],
        "interpretation": result["interpretation"],
        "non_degenerate": result["non_degenerate"],
        "degeneracy_reason": result["degeneracy_reason"],
        "sleep_driver_pattern": None,
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
    print(f"direction: {result['evidence_direction']} non_degenerate: {result['non_degenerate']}", flush=True)
    print(f"readiness_met={m['readiness_met']} (P1={m['p1_vs_heterogeneity']} "
          f"P2={m['p2_decomp_fires']} P3={m['p3_pe_meaningful']})", flush=True)
    print(f"n_pairs={m['n_paired_seeds']} delta_mean={m['delta_mean_lowvs_fwd_pe']:.5g} "
          f"rel_improvement={m['rel_improvement']:.4f} effect_size_ok={m['effect_size_ok']}", flush=True)
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
