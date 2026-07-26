"""V3-EXQ-816d -- GOV-FANOUT-1 leg P-A (ENVIRONMENT axis), DOSE ESCALATION.

DIAGNOSTIC. Continues the fan-out probe for hypothesis H-env-underdrives-uncertainty
in the frozen ledger question `policy_decomposition_discrimination` (claims
ARC-070 / MECH-321), spun out of the V3-EXQ-816 + V3-EXQ-820 cluster autopsy
(failure_autopsy_816-820-policy-decomposition-cluster_2026-07-26) and continued
from V3-EXQ-816b's dose (failure_autopsy_V3-EXQ-816b_2026-07-26). This leg does
NOT weight either claim -- it BEARS ON them (experiment_purpose=diagnostic,
claim_ids=[]).

WHY THIS RUN (816b was informative but non-eliminating). 816b harshened the env
(env_drift_interval=3, world_rule_shift_interval=24/depth=1) and moved
off_pe_mean_worst from the 816/820 baseline (~0.005) to 0.0086 -- 86% of the
PE_ELEVATED_FLOOR=0.01 discrimination floor, but still short of it, so zero
low-V_s steps resulted (same failure shape as 816/820). The autopsy's
routing_note (user-confirmed 2026-07-26) recommends "a next letter (env axis,
stronger dose -- e.g. world_rule_shift_depth 1->2 and/or env_drift_interval->1)"
continuing the SAME hypothesis BEFORE running V3-EXQ-816c (P-B, already run --
its measurement-comparator design presupposes PE is actually elevated, which
816b's dose did not achieve).

DOSE ESCALATION (all three autopsy-named levers moved one step further; see
experiments/_lib/baselines/mech321_policy_decomposition_harshened_v2.py for the
full table + rationale):
    env_drift_interval:        3  -> 1  (fastest possible hazard drift)
    world_rule_shift_interval: 24 -> 12 (twice-per-episode shifts)
    world_rule_shift_depth:    1  -> 2  (two transpositions per shift)
Everything else (grid shape, hazards, resources, proxy fields, substrate
stack, schedule, alpha_world, seeded crystallised chunk) is IDENTICAL to 816b,
so the ONLY change from 816b is dose magnitude on the same three axes -- a
clean escalation, not a redesign, and pre-registered thresholds (MIN_LOW_VS_STEPS,
PE_ELEVATED_FLOOR) are UNCHANGED from 816/816b so the dose-response is
comparable across the ladder.

H-env-underdrives-uncertainty (the hypothesis this leg attacks). A harsher /
non-stationary env produces BOTH forward-PE heterogeneity AND low-V_s regions, so
the R1 trigger can fire and the R1-vs-OFF forward-PE contrast becomes measurable.

  NULL (declared, per GOV-FANOUT-1). Even under this escalated-dose env, the
  worst ARM_1 cell's low-V_s steps remain < MIN_LOW_VS_STEPS (region-V_s does
  not drop) -> at THIS dose, H-env-underdrives-uncertainty is still not
  confirmed. What that null MEANS then DEPENDS on the co-recorded forward-PE
  distribution (recorded identically to 816b):
    * low-V_s absent WHILE forward-PE is ELEVATED (>= PE_ELEVATED_FLOOR) -- the
      escalated dose crossed the PE floor but region-V_s stayed decoupled from
      it -> favours the sibling hypothesis H-vs-proxy-saturation (already
      adjudicated head-on by V3-EXQ-816c / P-B; this would mean 816c's premise
      was in fact met at THIS dose, retroactively informative for reading 816c).
    * low-V_s absent AND forward-PE STILL not elevated -- even this escalation
      was insufficient; the env-underdrives-uncertainty campaign has now made
      two dose steps without crossing the floor (816b: 0.0086 vs 0.01;
      diminishing-returns signal for whether a THIRD dose step is worth
      queuing before conceding the axis to H-vs-proxy-saturation).
  A null here is therefore INFORMATIVE, not wasted -- it routes between the
  live hypotheses and calibrates whether the environment axis is worth a
  further dose step at all.

RECORDING (preserves the 816b distributional recording unchanged). Per cell:
the full region-V_s distribution (min/mean/max/std/10-bin histogram over
[0,1] + raw samples) alongside the per-cell forward-PE distribution
(min/mean/max/var/histogram + raw samples). Recorded for BOTH arms (OFF
internals as richly as ON) -- identical shape to 816b so the two runs are
directly comparable point-by-point.

DV-SYMMETRY (Step 3 mandatory per-arm declaration). Identical reasoning to
816b (same design, only the env dose differs):
  PRIMARY load-bearing readout = worst-cell low-V_s step count under the
    escalated-dose env (an ENV+encoder property read on ARM_1's cells). Both
    arms see the identical env and both populate per_stream_vs the same way,
    so there is no manipulation for any DV symmetry to annihilate.
  ARM_0 = OFF observation / positive control (use_policy_decomposition=False).
    No manipulation, hence no symmetry trap; anchors the forward-PE positive
    control and the region-V_s distribution of the untreated agent.
  ARM_1 SECONDARY (non-load-bearing) DV = low-V_s execution-time forward-PE
    contrasted ARM_1-vs-ARM_0 (only evaluable if low-V_s appears). Symmetry
    group = permutation of steps + relabelling of action-class indices; the
    manipulation (use_policy_decomposition) is NOT invariant under it (it
    withholds/replaces a COMMITTED chunk in triggering regions, changing which
    action executes -> different per-step PE terms).
  NOT a broadcast-scalar-on-argmax case (604c hazard): forward PE is measured
    on the E2 first-step rollout of the committed trajectory vs observed
    z_world, not derived from an argmax/order-statistic over the CEM
    candidate pool.

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
from experiments._lib.baselines import mech321_policy_decomposition_harshened_v2 as base  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_816d_mech321_policy_decomposition_harshened_env_v2"
EXPERIMENT_PURPOSE = "diagnostic"
CLAIM_IDS: List[str] = []                       # diagnostic: bears on, does not weight
BEARS_ON = ["ARC-070", "MECH-321"]

# Identical hold-weighted-E3-readout triage to V3-EXQ-816b (same design, same
# readout sourcing -- only the env dose differs).
E3_HOLD_WEIGHTED_READOUT_EXEMPT = (
    "Load-bearing DV is low_vs_steps (region_vs count, updated every step via "
    "update_per_stream_vs, not select_action); flagged a_idx read feeds only a threshold-invariant "
    "action-sequence divergence check; per-step forward-PE is a non-load-bearing corroboration DV "
    "identical to V3-EXQ-816b's per-executed-step readout."
)

# Identical anchor-reachability triage to V3-EXQ-816b: single-scalar
# reachable-by-construction positive-control floors / degeneracy definitions,
# each empirically MET by the --dry-run smoke.
ANCHOR_REACHABILITY_EXEMPT = (
    "Readiness preconditions are single-scalar reachable-by-construction positive-control floors / "
    "degeneracy definitions (PE var > 1e-9, PE mean < 1e3, per_stream_vs non-empty), each met by "
    "the dry-run smoke; none is a composite/narrow predicate that could be sub-maximal by "
    "construction (the 778d hazard)."
)

SEEDS = [11, 23, 47, 71, 97]
ARMS = ["ARM_0", "ARM_1"]

# Schedule / env come from the DOSE-ESCALATED HARSHENED baseline module
# (single source of truth so ARM_0 is minted for THIS dose's own sub-lineage).
WARMUP_EPISODES = base.WARMUP_EPISODES
MEASURE_EPISODES = base.MEASURE_EPISODES
STEPS_PER_EPISODE = base.STEPS_PER_EPISODE
SEEDED_CHUNK_SEQUENCE = base.SEEDED_CHUNK_SEQUENCE

# Decomposition parameters (ARM_1 only). Threshold 0.5 matches V3-EXQ-816/816b
# (the partition boundary for "low V_s" and the same value the whole ladder uses).
DECOMPOSITION_VS_THRESHOLD = 0.5
DECOMPOSITION_DEPTH_CAP = 3          # mirrors ARC-071 chunk_max_depth (substrate default)

VS_HIST_BINS = 10                    # region-V_s histogram bins over [0, 1]
PE_HIST_BINS = 12                    # forward-PE histogram bins over [0, pe_hist_max]

# --- Pre-registered thresholds (defined HERE, never inferred post-hoc). ---
# UNCHANGED from V3-EXQ-816/816b so the dose-response ladder is comparable.
# PRIMARY (load-bearing, H-env): the harshened env must produce low-V_s steps.
MIN_LOW_VS_STEPS = 5                 # worst ARM_1 cell must clear this (== 816/816b's floor)
# Disambiguation of a low-V_s-absent null: was forward-PE nonetheless elevated?
# 816's default-env baseline forward-PE was ~0.005 (autopsy); 816b reached
# 0.0086 at dose 1. "Elevated" = clearing a floor set at 2x the original
# reference -- UNCHANGED so 816d's crossing (or not) of the SAME floor is
# directly comparable to 816b's 0.0086 reading.
PE_ELEVATED_FLOOR = 0.01
PE_VAR_FLOOR = 1e-9                  # forward-PE positive-control variance floor
PE_SANITY_CEIL = 1.0e3              # forward-PE explosion / NaN guard (upper bound)
# SECONDARY (non-load-bearing) forward-PE reduction contrast (only if low-V_s appears).
REL_IMPROVEMENT_FLOOR = 0.05        # >=5% relative reduction in low-V_s forward PE
EFFECT_SIZE_K_SIGMA = 2.0           # paired-delta mean must exceed K * SE(delta)
SEED_FIRE_FRACTION = 3.0 / 5.0      # decomposition must fire in a majority of ARM_1 cells


# ---------------------------------------------------------------------------
def _histogram(xs: List[float], n_bins: int, lo: float, hi: float) -> List[int]:
    """Fixed-bin histogram over [lo, hi] (values clamped into the range)."""
    hist = [0] * n_bins
    if hi <= lo or not xs:
        return hist
    span = hi - lo
    for x in xs:
        if not math.isfinite(x):
            continue
        frac = (x - lo) / span
        idx = int(frac * n_bins)
        if idx < 0:
            idx = 0
        elif idx >= n_bins:
            idx = n_bins - 1
        hist[idx] += 1
    return hist


def _dist_stats(xs: List[float]) -> Dict[str, Any]:
    """Summary stats for a distribution (None-safe)."""
    if not xs:
        return {"n": 0, "min": None, "mean": None, "max": None, "std": None, "var": None}
    return {
        "n": len(xs),
        "min": float(min(xs)),
        "mean": float(statistics.fmean(xs)),
        "max": float(max(xs)),
        "std": float(statistics.pstdev(xs)) if len(xs) > 1 else 0.0,
        "var": float(statistics.pvariance(xs)) if len(xs) > 1 else 0.0,
    }


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
        # Single source of truth -- this dose's own minted OFF closure.
        return base.off_path_config_slice()
    slice_: Dict[str, Any] = {
        "env": {"size": base.ENV_SIZE, "num_hazards": base.ENV_NUM_HAZARDS,
                "num_resources": base.ENV_NUM_RESOURCES,
                "use_proxy_fields": base.ENV_USE_PROXY_FIELDS,
                "env_drift_interval": base.HARSH_ENV_DRIFT_INTERVAL,
                "world_rule_shift_enabled": base.HARSH_WORLD_RULE_SHIFT_ENABLED,
                "world_rule_shift_interval": base.HARSH_WORLD_RULE_SHIFT_INTERVAL,
                "world_rule_shift_depth": base.HARSH_WORLD_RULE_SHIFT_DEPTH},
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
    # Seed the identical crystallised chunk in BOTH arms (only ARM_1 can decompose it).
    chunk = ChunkedPrimitive(
        sequence=SEEDED_CHUNK_SEQUENCE, depth=1,
        state=ChunkState.CRYSTALLISED, selection_weight=1.0,
    )
    agent.policy_chunking.library.register(chunk)
    return agent


def _run_cell(arm: str, seed: int) -> Dict[str, Any]:
    print(f"Seed {seed} Condition {arm}", flush=True)
    total_eps = WARMUP_EPISODES + MEASURE_EPISODES
    # ARM_0 is minted reuse-eligible (driver excluded) so a future same-dose
    # consumer can reuse it; ARM_1 is a treatment arm (never reused as-is).
    fold_driver = arm != "ARM_0"
    with arm_cell(seed, config_slice=_config_slice(arm), script_path=Path(__file__),
                  include_driver_script_in_hash=fold_driver) as cell:
        env = _build_env(seed)
        agent = _build_agent(env, arm)
        wd = agent.config.latent.world_dim

        region_vs_samples: List[float] = []
        fwd_pe_low: List[float] = []
        fwd_pe_all: List[float] = []
        action_seq: List[int] = []
        net_harm = 0.0
        n_measure_steps = 0
        low_vs_steps = 0
        max_n_streams = 0                       # V_s-tracking-live guard (degenerate-fallback)

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
                    region_vs_samples.append(region_vs)
                    n_streams = len(agent.hippocampal.per_stream_vs)
                    if n_streams > max_n_streams:
                        max_n_streams = n_streams
                    is_low = region_vs < DECOMPOSITION_VS_THRESHOLD
                    pe_raw = metrics.get("e3_prediction_error")
                    if pe_raw is not None:
                        pe = float(pe_raw.detach()) if torch.is_tensor(pe_raw) else float(pe_raw)
                        if math.isfinite(pe):
                            fwd_pe_all.append(pe)
                            if is_low:
                                fwd_pe_low.append(pe)
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
                      f"vs_trigger={st.get('decomp_n_vs_trigger', 0)} "
                      f"low_vs_steps={low_vs_steps}", flush=True)

        st = agent.get_policy_decomposition_state()
        chunk_st = agent.get_chunking_state()
        vs_stats = _dist_stats(region_vs_samples)
        pe_stats = _dist_stats(fwd_pe_all)
        pe_hist_max = max(pe_stats["max"] or 0.0, PE_ELEVATED_FLOOR)

        def _mean(xs: List[float]) -> Optional[float]:
            return statistics.fmean(xs) if xs else None

        row = {
            "arm": arm,
            "seed": seed,
            "n_measure_steps": int(n_measure_steps),
            "low_vs_steps": int(low_vs_steps),
            "low_vs_step_frac": (low_vs_steps / n_measure_steps) if n_measure_steps else 0.0,
            "max_n_streams_tracked": int(max_n_streams),   # >0 => region_vs is live, not fallback 1.0
            # --- Region-V_s distribution (preserved from 816b's recording gap fix). ---
            "region_vs_min": vs_stats["min"],
            "region_vs_mean": vs_stats["mean"],
            "region_vs_max": vs_stats["max"],
            "region_vs_std": vs_stats["std"],
            "region_vs_n": vs_stats["n"],
            "region_vs_hist": _histogram(region_vs_samples, VS_HIST_BINS, 0.0, 1.0),
            "region_vs_hist_edges": [i / VS_HIST_BINS for i in range(VS_HIST_BINS + 1)],
            "region_vs_samples": [round(x, 6) for x in region_vs_samples],
            # --- Per-cell forward-PE distribution (recorded alongside V_s). ---
            "fwd_pe_all_mean": pe_stats["mean"],
            "fwd_pe_all_min": pe_stats["min"],
            "fwd_pe_all_max": pe_stats["max"],
            "fwd_pe_all_std": pe_stats["std"],
            "fwd_pe_all_var": pe_stats["var"] or 0.0,
            "fwd_pe_all_n": pe_stats["n"],
            "fwd_pe_hist": _histogram(fwd_pe_all, PE_HIST_BINS, 0.0, pe_hist_max),
            "fwd_pe_hist_max": float(pe_hist_max),
            "fwd_pe_samples": [round(x, 8) for x in fwd_pe_all],
            "fwd_pe_lowvs_mean": _mean(fwd_pe_low),
            "fwd_pe_lowvs_n": len(fwd_pe_low),
            # Task-performance context.
            "net_harm_per_step": (net_harm / n_measure_steps) if n_measure_steps else 0.0,
            # Mechanism firing (readiness / audit).
            "decomp_n_evaluated_precommit": int(st.get("decomp_n_evaluated_precommit", 0)),
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

    # A cell "verdict" here is a local readiness proxy for the runner progress bar,
    # not a claim verdict: did this cell exercise the substrate as intended?
    if arm == "ARM_1":
        fired = row["decomp_n_decomposed_precommit"] + row["decomp_n_marked_unreliable"]
        cell_ok = fired >= 1
    else:
        cell_ok = row["decomp_n_evaluated_precommit"] == 0
    print(f"verdict: {'PASS' if cell_ok else 'FAIL'}", flush=True)
    return row


# ---------------------------------------------------------------------------
def _paired_deltas(by_arm: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Per-seed paired ARM_0-minus-ARM_1 low-V_s forward-PE deltas (secondary DV)."""
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

    # --- READINESS: V_s tracking must be LIVE (per_stream_vs populated), else region_vs is the
    # degenerate constant-1.0 fallback and a low_vs_steps=0 would be a substrate artefact, NOT
    # an "env too easy" finding. Same underlying quantity (region_vs) the primary criterion routes
    # on. The autopsy ruled the fallback out for 816/816b/820; this makes it self-checking. ---
    vs_tracking_worst = min((r["max_n_streams_tracked"] for r in rows), default=0)
    vs_tracking_live = vs_tracking_worst >= 1

    # --- PRIMARY readout (H-env): did the escalated-dose env produce low-V_s steps? ---
    lowvs_worst = min((r["low_vs_steps"] for r in by_arm["ARM_1"]), default=0)
    low_vs_produced = lowvs_worst >= MIN_LOW_VS_STEPS

    # --- Forward-PE positive control + elevation (disambiguates a low-V_s-absent null). ---
    off_pe_var_worst = min((r["fwd_pe_all_var"] for r in by_arm["ARM_0"]), default=0.0)
    off_pe_mean_worst = max((r["fwd_pe_all_mean"] or float("inf")) for r in by_arm["ARM_0"]) \
        if by_arm["ARM_0"] else float("inf")
    pe_control_ok = (off_pe_var_worst > PE_VAR_FLOOR) and (off_pe_mean_worst < PE_SANITY_CEIL)
    # "Elevated" = the escalated-dose env's forward-PE cleared the elevation floor on any arm
    # (best-case: the env DID drive prediction uncertainty somewhere).
    pe_mean_best = max((r["fwd_pe_all_mean"] or 0.0) for r in rows) if rows else 0.0
    pe_elevated = pe_mean_best >= PE_ELEVATED_FLOOR

    # --- SECONDARY contrast (non-load-bearing): forward-PE reduction, only if low-V_s. ---
    deltas = _paired_deltas(by_arm)
    delta_vals = [d["delta"] for d in deltas]
    n_pairs = len(delta_vals)
    delta_mean = statistics.fmean(delta_vals) if delta_vals else 0.0
    delta_sd = statistics.stdev(delta_vals) if n_pairs > 1 else 0.0
    delta_se = (delta_sd / math.sqrt(n_pairs)) if n_pairs > 0 else 0.0
    off_lowvs_ref = statistics.fmean([d["off_lowvs"] for d in deltas]) if deltas else 0.0
    rel_improvement = (delta_mean / off_lowvs_ref) if off_lowvs_ref > 0 else 0.0
    fired_frac = (sum(1 for r in by_arm["ARM_1"]
                      if (r["decomp_n_decomposed_precommit"] + r["decomp_n_marked_unreliable"]) >= 1)
                  / len(by_arm["ARM_1"])) if by_arm["ARM_1"] else 0.0
    secondary_supports = (low_vs_produced and n_pairs >= 2
                          and delta_mean > EFFECT_SIZE_K_SIGMA * delta_se
                          and rel_improvement >= REL_IMPROVEMENT_FLOOR)

    any_action_divergence = any(d["action_seq_differs"] for d in deltas) or any(
        by_arm["ARM_0"][i]["measure_action_seq"] != by_arm["ARM_1"][i]["measure_action_seq"]
        for i in range(min(len(by_arm["ARM_0"]), len(by_arm["ARM_1"])))
    )

    # C_LOWVS (load-bearing) is non-degenerate iff V_s tracking was live, the forward-PE
    # positive control held (a real, bounded, varying PE signal), and a measurement window existed.
    n_measure_ok = all(r["n_measure_steps"] > 0 for r in rows)
    c_lowvs_non_degenerate = bool(vs_tracking_live and pe_control_ok and n_measure_ok)
    c_secondary_non_degenerate = bool(low_vs_produced and any_action_divergence)

    # --- Self-route (diagnostic hypothesis; adjudicated by /failure-autopsy or /governance). ---
    if not vs_tracking_live:
        label = "substrate_not_ready_requeue"
        degeneracy_reason = ("region-V_s tracking was not live (per_stream_vs empty -> region_vs "
                             "degenerate constant-1.0 fallback); a low_vs_steps=0 here is a "
                             "substrate artefact, not an env-too-easy finding -- re-queue with "
                             "V_s tracking confirmed")
    elif not pe_control_ok:
        label = "substrate_not_ready_requeue"
        degeneracy_reason = ("OFF-arm forward PE was constant / unbounded on the escalated-dose "
                             "env (world_forward not action-conditional, or PE exploded); the "
                             "forward-PE positive control failed, so neither low-V_s nor the "
                             "decoupling question is answerable")
    elif low_vs_produced:
        label = "env_harshen_produced_low_vs"     # H-env SUPPORTED at this dose
        degeneracy_reason = None
    elif pe_elevated:
        # escalated env drove PE past the floor but V_s did not drop -> decoupling signature.
        label = "low_vs_absent_despite_elevated_pe"   # H-env refuted at this dose; favours H-vs-proxy-saturation
        degeneracy_reason = None
    else:
        label = "env_still_underdrives_uncertainty"   # even this escalation insufficient
        degeneracy_reason = None

    # Diagnostic: bears on the claims, never weights them.
    direction = "non_contributory"
    outcome = "PASS" if low_vs_produced else "FAIL"

    metrics = {
        # READINESS.
        "vs_tracking_live": vs_tracking_live,
        "vs_tracking_worst_n_streams": vs_tracking_worst,
        # PRIMARY.
        "lowvs_worst_arm1": lowvs_worst,
        "low_vs_produced": low_vs_produced,
        "min_low_vs_steps_floor": MIN_LOW_VS_STEPS,
        # Forward-PE positive control + elevation.
        "off_pe_var_worst": off_pe_var_worst,
        "off_pe_mean_worst": off_pe_mean_worst,
        "pe_control_ok": pe_control_ok,
        "pe_mean_best_arm": pe_mean_best,
        "pe_elevated": pe_elevated,
        "pe_elevated_floor": PE_ELEVATED_FLOOR,
        # Dose-response context vs 816b (0.0086 at dose 1) for direct comparison.
        "predecessor_816b_off_pe_mean_worst": 0.0086,
        # Region-V_s distribution summary (per-arm means of the per-cell stats).
        "region_vs_mean_over_cells_arm0": statistics.fmean(
            [r["region_vs_mean"] for r in by_arm["ARM_0"] if r["region_vs_mean"] is not None] or [0.0]),
        "region_vs_min_over_cells_arm1": min(
            [r["region_vs_min"] for r in by_arm["ARM_1"] if r["region_vs_min"] is not None] or [1.0]),
        # SECONDARY forward-PE reduction contrast.
        "n_paired_seeds": n_pairs,
        "delta_mean_lowvs_fwd_pe": delta_mean,
        "delta_sd": delta_sd,
        "delta_se": delta_se,
        "rel_improvement": rel_improvement,
        "secondary_forward_pe_reduced": secondary_supports,
        "decomp_fired_frac_arm1": fired_frac,
        "arm1_vs_trigger_total": sum(r["decomp_n_vs_trigger"] for r in by_arm["ARM_1"]),
        "any_action_divergence": any_action_divergence,
        "off_net_harm_per_step_mean": statistics.fmean([r["net_harm_per_step"] for r in by_arm["ARM_0"]]),
        "on_net_harm_per_step_mean": statistics.fmean([r["net_harm_per_step"] for r in by_arm["ARM_1"]]),
        "n_replay_formed_total": sum(r["chunk_acc_n_replay_formed"] for r in rows),
    }

    interpretation = {
        "label": label,
        "preconditions": [
            {"name": "vs_tracking_live",
             "description": ("READINESS: region-V_s tracking must be live (per_stream_vs "
                             "populated), else region_vs is the degenerate constant-1.0 fallback "
                             "and low_vs_steps=0 is a substrate artefact not an env finding. "
                             "Worst-cell max streams tracked during measurement."),
             "measured": float(vs_tracking_worst), "threshold": 1.0,
             "direction": "lower", "control": "worst cell (min over cells of max streams tracked)",
             "met": vs_tracking_live},
            {"name": "off_forward_pe_varies",
             "description": ("READINESS positive control: OFF-arm committed-trajectory forward "
                             "PE must have non-zero variance (world_forward learned something on "
                             "the escalated-dose env). SAME statistic the elevation disambiguation "
                             "reads. Worst ARM_0 cell variance."),
             "measured": float(off_pe_var_worst), "threshold": float(PE_VAR_FLOOR),
             "direction": "lower", "control": "worst ARM_0 cell (positive control)",
             "met": (off_pe_var_worst > PE_VAR_FLOOR)},
            {"name": "off_forward_pe_bounded",
             "description": "READINESS: forward PE bounded (no explosion / NaN). Worst ARM_0 cell mean.",
             "measured": float(off_pe_mean_worst), "threshold": float(PE_SANITY_CEIL),
             "direction": "upper", "control": "worst ARM_0 cell",
             "met": (off_pe_mean_worst < PE_SANITY_CEIL)},
        ],
        "criteria": [
            {"name": "C_LOWVS_escalated_env_produces_low_vs", "load_bearing": True,
             "passed": low_vs_produced},
            {"name": "C_SECONDARY_lowvs_forward_pe_reduced", "load_bearing": False,
             "passed": secondary_supports},
        ],
        "criteria_non_degenerate": {
            "C_LOWVS": c_lowvs_non_degenerate,
            "C_SECONDARY": c_secondary_non_degenerate,
        },
        "null_reading_guide": {
            "env_harshen_produced_low_vs": "H-env-underdrives-uncertainty SUPPORTED at this dose; R1 trigger now exercisable",
            "low_vs_absent_despite_elevated_pe": "H-env refuted at this dose; forward-PE elevated but V_s flat -> favours H-vs-proxy-saturation (see V3-EXQ-816c)",
            "env_still_underdrives_uncertainty": "even this escalation insufficient (PE not elevated); consider a third dose step or conceding the env axis",
            "substrate_not_ready_requeue": "forward-PE positive control failed; re-queue with a corrected forward model / horizon",
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
        "non_degenerate": c_lowvs_non_degenerate,
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
        "harsh_env_drift_interval": base.HARSH_ENV_DRIFT_INTERVAL,
        "harsh_world_rule_shift_enabled": base.HARSH_WORLD_RULE_SHIFT_ENABLED,
        "harsh_world_rule_shift_interval": base.HARSH_WORLD_RULE_SHIFT_INTERVAL,
        "harsh_world_rule_shift_depth": base.HARSH_WORLD_RULE_SHIFT_DEPTH,
        "seeded_chunk_sequence": list(SEEDED_CHUNK_SEQUENCE),
        "min_low_vs_steps": MIN_LOW_VS_STEPS,
        "pe_elevated_floor": PE_ELEVATED_FLOOR,
        "pe_var_floor": PE_VAR_FLOOR,
        "rel_improvement_floor": REL_IMPROVEMENT_FLOOR,
        "effect_size_k_sigma": EFFECT_SIZE_K_SIGMA,
        "seed_fire_fraction": SEED_FIRE_FRACTION,
        "vs_hist_bins": VS_HIST_BINS,
        "pe_hist_bins": PE_HIST_BINS,
        "arm_config_slices": {a: _config_slice(a) for a in ARMS},
    }
    manifest = {
        "run_id": f"{EXPERIMENT_TYPE}_{ts}_v3",
        "experiment_type": EXPERIMENT_TYPE,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "bears_on": BEARS_ON,
        "fanout_question": "policy_decomposition_discrimination",
        "fanout_hypothesis": "H-env-underdrives-uncertainty",
        "evidence_direction": result["evidence_direction"],
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
        "cites": {"predecessor_easy_env_runs": ["V3-EXQ-816", "V3-EXQ-820"],
                  "predecessor_dose_1_run": "V3-EXQ-816b",
                  "cluster_autopsy": "failure_autopsy_816-820-policy-decomposition-cluster_2026-07-26",
                  "predecessor_autopsy": "failure_autopsy_V3-EXQ-816b_2026-07-26"},
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
    print(f"low_vs_produced={m['low_vs_produced']} lowvs_worst_arm1={m['lowvs_worst_arm1']} "
          f"pe_control_ok={m['pe_control_ok']} pe_elevated={m['pe_elevated']} "
          f"pe_mean_best={m['pe_mean_best_arm']:.5g}", flush=True)
    print(f"n_pairs={m['n_paired_seeds']} delta_mean={m['delta_mean_lowvs_fwd_pe']:.5g} "
          f"secondary_reduced={m['secondary_forward_pe_reduced']}", flush=True)
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
