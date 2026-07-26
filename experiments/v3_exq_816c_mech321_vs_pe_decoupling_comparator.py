"""V3-EXQ-816c -- GOV-FANOUT-1 leg P-B (MEASUREMENT axis).

DIAGNOSTIC. Fan-out probe for hypothesis H-vs-proxy-saturation in the frozen
ledger question `policy_decomposition_discrimination` (claims ARC-070 / MECH-321),
spun out of the V3-EXQ-816 + V3-EXQ-820 cluster autopsy
(failure_autopsy_816-820-policy-decomposition-cluster_2026-07-26). This leg does
NOT weight either claim -- it BEARS ON them (experiment_purpose=diagnostic,
claim_ids=[]) and makes the decoupling DIRECTLY OBSERVABLE.

THE OPEN OPERATIONALIZATION QUESTION. MECH-321's R1 trigger reads region V_s =
HippocampalModule._region_vs() = mean per_stream_vs, a '1 - relative-tick-to-tick-
latent-change' STABILITY proxy. The claim's own gloss reads V_s as "cannot reliably
predict outcomes" -- i.e. forward-model PE. Whether latent-stability-V_s and
forward-model PE co-occur is UNRESOLVED, and 816/820 produced neither (no low-V_s
AND near-zero PE), so they could not discriminate them. MECH-321's
functional_restatement itself lists "E2 forward-model disagreement" as an
ALTERNATIVE trigger -- so this is not a measurement nit; it bears on which trigger
the claim should commit to.

H-vs-proxy-saturation (the hypothesis this leg attacks). region-V_s (latent
stability) saturates near 1.0 in a trained encoder and is DECOUPLED from forward-
model PE, so no env manipulation lowers it and the R1 V_s-drop trigger cannot fire
in a competent agent.

  NULL (declared, per GOV-FANOUT-1). On the SAME harshened env P-A uses, forward-PE
  heterogeneity is PRESENT (the PE signal varies -- the env drives prediction
  uncertainty) while V_s heterogeneity is ABSENT (region-V_s stays flat / >=
  threshold; no low-V_s steps) -> region-V_s is DECOUPLED from forward-PE ->
  H-vs-proxy-saturation CONFIRMED, and MECH-321's R1 trigger must be reframed
  toward its own forward-model-disagreement alternative (routes to a candidate
  /claim-synthesis on MECH-321 trigger operationalization). The CONVERSE null (V_s
  DOES drop -- low-V_s steps present) REFUTES H-vs-proxy-saturation: region-V_s is
  not saturated, and the env-axis reading (V3-EXQ-816b / P-A) governs instead.

HOW IT DISCRIMINATES (and closes the portfolio's worst aliasing gap). "V_s flat"
alone is AMBIGUOUS: it could mean the env under-drove uncertainty (H-env not yet
refuted -- needs a harsher env) OR region-V_s is structurally saturated / decoupled
(H-vs-proxy). This leg breaks the aliasing by recording forward-PE on the SAME
harshened env: if V_s is flat WHILE forward-PE is heterogeneous, the env DID drive
uncertainty (PE moved) but V_s failed to track it -> DECOUPLED, not env-underdrive.
If BOTH V_s and PE are flat, the env under-drove uncertainty entirely -> the
forward-PE positive control fails and this leg self-routes substrate_not_ready
(pointing back to a harsher env, i.e. P-A's territory), NEVER a false decoupling
verdict. The forward-PE positive control is exactly what makes the decoupling
verdict falsifiable.

WHAT IT MEASURES. A single OBSERVATION arm (the harshened OFF path -- substrate
stack ON, decomposition OFF; identical to P-A's ARM_0, constructed from the same
harshened baseline module) run to a trained forward model, recording per measuring
step the ALIGNED pair (region_vs, e3_prediction_error). Per cell it reports: the
region-V_s distribution (min/mean/max/std/histogram/samples), the forward-PE
distribution (same), the Spearman rank correlation between the uncertainty proxy
(1 - region_vs) and forward-PE (positive = V_s tracks PE; ~0 = decoupled), and the
firing loci of a V_s-trigger (region_vs < threshold) vs a forward-PE-trigger
(forward_pe >= a fixed pre-registered marker) with their co-fire overlap -- the
"record where each fires" the autopsy asked for.

DV-SYMMETRY (Step 3 mandatory declaration). There is exactly ONE arm and NO
manipulation: this is pure observation of two co-recorded substrate readouts on the
untreated agent. With no manipulation there is no manipulation-DV for any symmetry
to annihilate, so no DV-symmetry trap is possible. The load-bearing readout is the
JOINT relationship between region_vs and forward_pe (heterogeneity of each + their
rank correlation), measured, not a difference-of-arms. Non-degeneracy requires only
that the forward-PE positive control holds (PE actually varies) so the relationship
is estimable.

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
from typing import Any, Dict, List, Optional, Tuple

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ree_core.utils.config import REEConfig  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.policy import ChunkedPrimitive, ChunkState  # noqa: E402
from experiment_protocol import emit_outcome  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._lib.baselines import mech321_policy_decomposition_harshened as base  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_816c_mech321_vs_pe_decoupling_comparator"
EXPERIMENT_PURPOSE = "diagnostic"
CLAIM_IDS: List[str] = []                       # diagnostic: bears on, does not weight
BEARS_ON = ["ARC-070", "MECH-321"]

# The readiness preconditions here are simple reachable-by-construction positive-control floors /
# degeneracy definitions -- forward_pe_varies (var > 1e-9), forward_pe_bounded (mean < 1e3),
# enough_paired_steps (>= 30), vs_tracking_live (per_stream_vs non-empty). Each is a single scalar
# floor/ceiling, not a composite multi-rail score that could be structurally sub-maximal (the
# V3-EXQ-778d hazard the anchor-reachability gate guards), and each was empirically MET by the
# --dry-run smoke (pe_control_ok=True, vs_tracking_live=True, enough pairs). No instrument-
# specification gap for assert_anchor_reachable to guard.
ANCHOR_REACHABILITY_EXEMPT = (
    "Readiness preconditions are single-scalar reachable-by-construction positive-control floors / "
    "degeneracy definitions (PE var > 1e-9, PE mean < 1e3, paired >= 30, per_stream_vs non-empty), "
    "each met by the dry-run smoke; none is a composite/narrow predicate that could be sub-maximal "
    "by construction (the 778d hazard)."
)

SEEDS = [11, 23, 47, 71, 97]
OBS_ARM = "OBS"                                  # single observation arm (harshened OFF path)

# Schedule / env from the HARSHENED baseline module (same source of truth as P-A, so the
# observation arm matches P-A's ARM_0 fingerprint BY CONSTRUCTION).
WARMUP_EPISODES = base.WARMUP_EPISODES
MEASURE_EPISODES = base.MEASURE_EPISODES
STEPS_PER_EPISODE = base.STEPS_PER_EPISODE
SEEDED_CHUNK_SEQUENCE = base.SEEDED_CHUNK_SEQUENCE

# Partition boundary for the V_s-trigger (matches 816/816b/820: region_vs < 0.5 = low).
DECOMPOSITION_VS_THRESHOLD = 0.5

VS_HIST_BINS = 10                    # region-V_s histogram bins over [0, 1]
PE_HIST_BINS = 12                    # forward-PE histogram bins over [0, pe_hist_max]

# --- Pre-registered thresholds (defined HERE, never inferred post-hoc). ---
# LOAD-BEARING (threshold-INVARIANT) decoupling readout uses heterogeneity + rank
# correlation, so it does not depend on any absolute PE cut:
MIN_LOW_VS_STEPS = 5                 # pooled: V_s heterogeneity present iff >= this many low-V_s steps
PE_VAR_FLOOR = 1e-9                  # forward-PE positive-control variance floor (matched statistic)
PE_SANITY_CEIL = 1.0e3              # forward-PE explosion / NaN guard (upper bound)
SPEARMAN_COUPLED_FLOOR = 0.2        # |rho| >= this (positive) = V_s tracks PE (corroborating, not gating)
MIN_PAIRED_STEPS = 30               # min aligned (V_s, PE) pairs to estimate a correlation
# OBSERVATIONAL ONLY (a fixed marker for the "PE-trigger fires" count; NOT the load-bearing
# gate). 816's default-env forward-PE was ~0.005; a marker at 2x that flags "notably high PE".
PE_TRIGGER_MARKER = 0.01


# ---------------------------------------------------------------------------
def _histogram(xs: List[float], n_bins: int, lo: float, hi: float) -> List[int]:
    hist = [0] * n_bins
    if hi <= lo or not xs:
        return hist
    span = hi - lo
    for x in xs:
        if not math.isfinite(x):
            continue
        idx = int((x - lo) / span * n_bins)
        if idx < 0:
            idx = 0
        elif idx >= n_bins:
            idx = n_bins - 1
        hist[idx] += 1
    return hist


def _dist_stats(xs: List[float]) -> Dict[str, Any]:
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


def _rankdata(xs: List[float]) -> List[float]:
    """Average-rank of each element (ties share the mean of their rank span)."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0   # 1-indexed average rank
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _spearman(a: List[float], b: List[float]) -> Optional[float]:
    """Spearman rho = Pearson correlation of the ranks (None if undefined)."""
    if len(a) != len(b) or len(a) < 3:
        return None
    ra, rb = _rankdata(a), _rankdata(b)
    n = len(ra)
    ma, mb = statistics.fmean(ra), statistics.fmean(rb)
    cov = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    va = sum((ra[i] - ma) ** 2 for i in range(n))
    vb = sum((rb[i] - mb) ** 2 for i in range(n))
    if va <= 0 or vb <= 0:
        return None
    return float(cov / math.sqrt(va * vb))


def _build_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(**base.env_kwargs(seed))


def _config_slice() -> Dict[str, Any]:
    # The observation arm IS the harshened OFF path -- single source of truth (matches P-A's ARM_0).
    return base.off_path_config_slice()


def _build_agent(env: CausalGridWorldV2) -> REEAgent:
    agent = REEAgent(REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=4,
        alpha_world=base.ALPHA_WORLD,
        **base.off_arm_flags(),
    ))
    chunk = ChunkedPrimitive(
        sequence=SEEDED_CHUNK_SEQUENCE, depth=1,
        state=ChunkState.CRYSTALLISED, selection_weight=1.0,
    )
    agent.policy_chunking.library.register(chunk)
    return agent


def _run_cell(seed: int) -> Dict[str, Any]:
    print(f"Seed {seed} Condition {OBS_ARM}", flush=True)
    total_eps = WARMUP_EPISODES + MEASURE_EPISODES
    # Observation arm minted reuse-eligible (driver excluded) so it matches P-A's harshened ARM_0.
    with arm_cell(seed, config_slice=_config_slice(), script_path=Path(__file__),
                  include_driver_script_in_hash=False) as cell:
        env = _build_env(seed)
        agent = _build_agent(env)
        wd = agent.config.latent.world_dim

        region_vs_all: List[float] = []                     # region_vs on every measuring step
        paired: List[Tuple[float, float]] = []              # aligned (region_vs, forward_pe)
        n_measure_steps = 0
        low_vs_steps = 0
        max_n_streams = 0                                    # V_s-tracking-live guard
        pe_trigger_fires = 0                                # forward_pe >= PE_TRIGGER_MARKER
        vs_trigger_fires = 0                                # region_vs < threshold (== low_vs_steps)
        cofire = 0                                          # both fired on the same step

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
                    region_vs_all.append(region_vs)
                    n_streams = len(agent.hippocampal.per_stream_vs)
                    if n_streams > max_n_streams:
                        max_n_streams = n_streams
                    is_low = region_vs < DECOMPOSITION_VS_THRESHOLD
                    n_measure_steps += 1
                    if is_low:
                        low_vs_steps += 1
                        vs_trigger_fires += 1
                    pe_raw = metrics.get("e3_prediction_error")
                    if pe_raw is not None:
                        pe = float(pe_raw.detach()) if torch.is_tensor(pe_raw) else float(pe_raw)
                        if math.isfinite(pe):
                            paired.append((region_vs, pe))
                            pe_fires = pe >= PE_TRIGGER_MARKER
                            if pe_fires:
                                pe_trigger_fires += 1
                            if pe_fires and is_low:
                                cofire += 1
                if done:
                    break
            if (ep + 1) % 10 == 0:
                print(f"  [train] obs seed={seed} arm={OBS_ARM} ep {ep+1}/{total_eps} "
                      f"low_vs_steps={low_vs_steps} pe_fires={pe_trigger_fires} "
                      f"paired={len(paired)}", flush=True)

        vs_vals = region_vs_all
        pe_vals = [p[1] for p in paired]
        unc_vals = [1.0 - p[0] for p in paired]             # uncertainty proxy = 1 - region_vs
        vs_stats = _dist_stats(vs_vals)
        pe_stats = _dist_stats(pe_vals)
        pe_hist_max = max(pe_stats["max"] or 0.0, PE_TRIGGER_MARKER)
        rho = _spearman(unc_vals, pe_vals)

        row = {
            "arm": OBS_ARM,
            "seed": seed,
            "n_measure_steps": int(n_measure_steps),
            "n_paired_steps": len(paired),
            "low_vs_steps": int(low_vs_steps),
            "low_vs_step_frac": (low_vs_steps / n_measure_steps) if n_measure_steps else 0.0,
            "max_n_streams_tracked": int(max_n_streams),   # >0 => region_vs live, not fallback 1.0
            # --- Region-V_s distribution. ---
            "region_vs_min": vs_stats["min"],
            "region_vs_mean": vs_stats["mean"],
            "region_vs_max": vs_stats["max"],
            "region_vs_std": vs_stats["std"],
            "region_vs_var": vs_stats["var"] or 0.0,
            "region_vs_n": vs_stats["n"],
            "region_vs_hist": _histogram(vs_vals, VS_HIST_BINS, 0.0, 1.0),
            "region_vs_hist_edges": [i / VS_HIST_BINS for i in range(VS_HIST_BINS + 1)],
            "region_vs_samples": [round(x, 6) for x in vs_vals],
            # --- Forward-PE distribution. ---
            "fwd_pe_mean": pe_stats["mean"],
            "fwd_pe_min": pe_stats["min"],
            "fwd_pe_max": pe_stats["max"],
            "fwd_pe_std": pe_stats["std"],
            "fwd_pe_var": pe_stats["var"] or 0.0,
            "fwd_pe_n": pe_stats["n"],
            "fwd_pe_hist": _histogram(pe_vals, PE_HIST_BINS, 0.0, pe_hist_max),
            "fwd_pe_hist_max": float(pe_hist_max),
            "fwd_pe_samples": [round(x, 8) for x in pe_vals],
            # --- Joint decoupling readouts. ---
            "spearman_unc_vs_pe": rho,     # (1 - region_vs) vs forward_pe; + = V_s tracks PE
            "paired_region_vs_samples": [round(p[0], 6) for p in paired],
            # --- Where each trigger fires (observational). ---
            "vs_trigger_fires": int(vs_trigger_fires),
            "pe_trigger_fires": int(pe_trigger_fires),
            "cofire_vs_and_pe": int(cofire),
            "pe_trigger_marker": PE_TRIGGER_MARKER,
        }
        cell.stamp(row)

    # Runner progress verdict (local readiness proxy, not a claim verdict): did this cell
    # produce a usable paired sample?
    cell_ok = row["n_paired_steps"] >= MIN_PAIRED_STEPS
    print(f"verdict: {'PASS' if cell_ok else 'FAIL'}", flush=True)
    return row


# ---------------------------------------------------------------------------
def run_experiment() -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = [_run_cell(seed) for seed in SEEDS]

    # --- Pool across seeds. ---
    total_low_vs = sum(r["low_vs_steps"] for r in rows)
    total_paired = sum(r["n_paired_steps"] for r in rows)
    total_measure = sum(r["n_measure_steps"] for r in rows)
    # Pooled forward-PE variance (positive control) and boundedness.
    pe_var_worst = min((r["fwd_pe_var"] for r in rows), default=0.0)
    pe_var_best = max((r["fwd_pe_var"] for r in rows), default=0.0)
    pe_mean_worst = max((r["fwd_pe_mean"] or float("inf")) for r in rows) if rows else float("inf")
    pe_mean_best = max((r["fwd_pe_mean"] or 0.0) for r in rows) if rows else 0.0
    # Region-V_s heterogeneity.
    vs_min_over_cells = min((r["region_vs_min"] for r in rows if r["region_vs_min"] is not None),
                            default=1.0)
    vs_var_best = max((r["region_vs_var"] for r in rows), default=0.0)
    # V_s-tracking-live guard (per_stream_vs populated => region_vs is not the fallback 1.0).
    vs_tracking_worst = min((r["max_n_streams_tracked"] for r in rows), default=0)
    vs_tracking_live = vs_tracking_worst >= 1
    # Per-cell Spearman (mean over cells that produced one).
    rhos = [r["spearman_unc_vs_pe"] for r in rows if r["spearman_unc_vs_pe"] is not None]
    rho_mean = statistics.fmean(rhos) if rhos else None

    # --- READINESS (P0). Matched statistic: the decoupling reads PE heterogeneity, so
    # readiness asserts the forward-PE positive control (PE varies + bounded + enough pairs). ---
    pe_varies = pe_var_best > PE_VAR_FLOOR
    pe_bounded = pe_mean_worst < PE_SANITY_CEIL
    enough_pairs = total_paired >= MIN_PAIRED_STEPS
    pe_control_ok = pe_varies and pe_bounded and enough_pairs

    # --- Heterogeneity of each signal. ---
    vs_heterogeneous = total_low_vs >= MIN_LOW_VS_STEPS      # V_s actually drops somewhere
    pe_heterogeneous = pe_var_best > PE_VAR_FLOOR            # forward-PE actually varies

    # --- Load-bearing decoupling verdict (threshold-invariant). ---
    decoupled = bool(pe_control_ok and pe_heterogeneous and not vs_heterogeneous)
    not_saturated = bool(pe_control_ok and vs_heterogeneous)   # V_s DOES drop -> refutes saturation

    n_measure_ok = total_measure > 0
    # C_DECOUPLING is testable (non-degenerate) iff V_s tracking is live (so "V_s flat" is a real
    # reading not the fallback 1.0), PE is a real varying bounded signal with enough paired
    # samples (the SAME statistic the verdict routes on), and a measurement window existed.
    c_decoupling_non_degenerate = bool(vs_tracking_live and pe_control_ok and n_measure_ok)

    if not vs_tracking_live:
        # V_s not live -> region_vs is the degenerate constant-1.0 fallback -> "V_s flat" would be
        # an artefact, and a decoupling verdict would be false. Not testable.
        label = "substrate_not_ready_requeue"
        degeneracy_reason = ("region-V_s tracking was not live (per_stream_vs empty -> region_vs "
                             "degenerate constant-1.0 fallback); the decoupling is untestable "
                             "because 'V_s flat' cannot be distinguished from 'V_s tracking dead'")
    elif not pe_control_ok:
        label = "substrate_not_ready_requeue"
        if not pe_varies:
            degeneracy_reason = ("forward-PE positive control failed: PE did not vary on the "
                                 "harshened env (env under-drove uncertainty entirely) -- the "
                                 "decoupling is untestable; needs a harsher env (P-A territory)")
        elif not enough_pairs:
            degeneracy_reason = (f"too few aligned (V_s, PE) pairs ({total_paired} < "
                                 f"{MIN_PAIRED_STEPS}) to estimate the relationship")
        else:
            degeneracy_reason = "forward-PE unbounded / NaN on the harshened env"
    elif vs_heterogeneous:
        label = "vs_tracks_uncertainty_not_saturated"   # H-vs-proxy-saturation REFUTED
        degeneracy_reason = None
    elif pe_heterogeneous:
        label = "vs_pe_decoupled_proxy_saturation"      # H-vs-proxy-saturation CONFIRMED
        degeneracy_reason = None
    else:
        # pe_control_ok but neither heterogeneity present -- should be unreachable
        # (pe_control_ok implies pe_heterogeneous); kept for total coverage.
        label = "substrate_not_ready_requeue"
        degeneracy_reason = "no heterogeneity in either signal despite positive control"

    direction = "non_contributory"     # diagnostic: bears on claims, never weights them
    # PASS = the decoupling question was answerable (V_s live AND forward-PE positive control
    # held); the label carries WHICH answer. FAIL = substrate_not_ready (couldn't test).
    outcome = "PASS" if (vs_tracking_live and pe_control_ok) else "FAIL"

    metrics = {
        "vs_tracking_live": vs_tracking_live,
        "vs_tracking_worst_n_streams": vs_tracking_worst,
        "pe_control_ok": pe_control_ok,
        "pe_varies": pe_varies,
        "pe_bounded": pe_bounded,
        "enough_pairs": enough_pairs,
        "vs_heterogeneous": vs_heterogeneous,
        "pe_heterogeneous": pe_heterogeneous,
        "decoupled": decoupled,
        "not_saturated": not_saturated,
        "total_low_vs_steps": total_low_vs,
        "total_paired_steps": total_paired,
        "total_measure_steps": total_measure,
        "pe_var_worst": pe_var_worst,
        "pe_var_best": pe_var_best,
        "pe_mean_worst": pe_mean_worst,
        "pe_mean_best": pe_mean_best,
        "region_vs_min_over_cells": vs_min_over_cells,
        "region_vs_var_best": vs_var_best,
        "spearman_unc_vs_pe_mean_over_cells": rho_mean,
        "spearman_coupled_floor": SPEARMAN_COUPLED_FLOOR,
        "vs_tracks_pe_by_correlation": (rho_mean is not None and rho_mean >= SPEARMAN_COUPLED_FLOOR),
        "vs_trigger_fires_total": sum(r["vs_trigger_fires"] for r in rows),
        "pe_trigger_fires_total": sum(r["pe_trigger_fires"] for r in rows),
        "cofire_total": sum(r["cofire_vs_and_pe"] for r in rows),
    }

    interpretation = {
        "label": label,
        "preconditions": [
            {"name": "vs_tracking_live",
             "description": ("READINESS: region-V_s tracking must be live (per_stream_vs "
                             "populated), else region_vs is the degenerate constant-1.0 fallback "
                             "and a 'V_s flat' reading is an artefact -- a false decoupling. "
                             "Worst-cell max streams tracked during measurement."),
             "measured": float(vs_tracking_worst), "threshold": 1.0,
             "direction": "lower", "control": "worst cell (min over cells of max streams tracked)",
             "met": vs_tracking_live},
            {"name": "forward_pe_varies",
             "description": ("READINESS positive control: forward PE must actually vary on the "
                             "harshened env (best-cell variance). SAME statistic the decoupling "
                             "verdict routes on (pe_heterogeneous). If flat, the env under-drove "
                             "uncertainty and the decoupling is untestable."),
             "measured": float(pe_var_best), "threshold": float(PE_VAR_FLOOR),
             "direction": "lower", "control": "best cell (positive control for PE heterogeneity)",
             "met": pe_varies},
            {"name": "forward_pe_bounded",
             "description": "READINESS: forward PE bounded (no explosion / NaN). Worst cell mean.",
             "measured": float(pe_mean_worst), "threshold": float(PE_SANITY_CEIL),
             "direction": "upper", "control": "worst cell", "met": pe_bounded},
            {"name": "enough_paired_steps",
             "description": ("READINESS: enough aligned (region_vs, forward_pe) pairs pooled "
                             "across seeds to estimate the relationship."),
             "measured": float(total_paired), "threshold": float(MIN_PAIRED_STEPS),
             "direction": "lower", "control": "pooled over seeds", "met": enough_pairs},
        ],
        "criteria": [
            {"name": "C_DECOUPLING_testable_and_resolved", "load_bearing": True,
             "passed": pe_control_ok},
        ],
        "criteria_non_degenerate": {
            "C_DECOUPLING": c_decoupling_non_degenerate,
        },
        "null_reading_guide": {
            "vs_pe_decoupled_proxy_saturation": "H-vs-proxy-saturation CONFIRMED: PE varies while V_s stays flat -> region-V_s decoupled from forward-PE -> reframe MECH-321 R1 trigger toward forward-model disagreement (candidate /claim-synthesis)",
            "vs_tracks_uncertainty_not_saturated": "H-vs-proxy-saturation REFUTED: region-V_s DOES drop -> not saturated -> env-axis reading (V3-EXQ-816b) governs",
            "substrate_not_ready_requeue": "forward-PE positive control failed (PE flat / too few pairs) -> env under-drove uncertainty -> needs a harsher env (P-A territory), NOT a decoupling verdict",
        },
    }

    return {
        "outcome": outcome,
        "evidence_direction": direction,
        "metrics": metrics,
        "per_seed_rows": rows,
        "arm_results": rows,
        "interpretation": interpretation,
        "non_degenerate": c_decoupling_non_degenerate,
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
        "seeds": SEEDS, "obs_arm": OBS_ARM,
        "warmup_episodes": WARMUP_EPISODES, "measure_episodes": MEASURE_EPISODES,
        "steps_per_episode": STEPS_PER_EPISODE,
        "decomposition_vs_threshold": DECOMPOSITION_VS_THRESHOLD,
        "harsh_env_drift_interval": base.HARSH_ENV_DRIFT_INTERVAL,
        "harsh_world_rule_shift_enabled": base.HARSH_WORLD_RULE_SHIFT_ENABLED,
        "harsh_world_rule_shift_interval": base.HARSH_WORLD_RULE_SHIFT_INTERVAL,
        "harsh_world_rule_shift_depth": base.HARSH_WORLD_RULE_SHIFT_DEPTH,
        "seeded_chunk_sequence": list(SEEDED_CHUNK_SEQUENCE),
        "min_low_vs_steps": MIN_LOW_VS_STEPS,
        "pe_var_floor": PE_VAR_FLOOR,
        "pe_sanity_ceil": PE_SANITY_CEIL,
        "spearman_coupled_floor": SPEARMAN_COUPLED_FLOOR,
        "min_paired_steps": MIN_PAIRED_STEPS,
        "pe_trigger_marker": PE_TRIGGER_MARKER,
        "vs_hist_bins": VS_HIST_BINS,
        "pe_hist_bins": PE_HIST_BINS,
        "obs_arm_config_slice": _config_slice(),
    }
    manifest = {
        "run_id": f"{EXPERIMENT_TYPE}_{ts}_v3",
        "experiment_type": EXPERIMENT_TYPE,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "bears_on": BEARS_ON,
        "fanout_question": "policy_decomposition_discrimination",
        "fanout_hypothesis": "H-vs-proxy-saturation",
        "evidence_direction": result["evidence_direction"],
        "outcome": result["outcome"],
        "timestamp_utc": ts,
        "metrics": result["metrics"],
        "per_seed_rows": result["per_seed_rows"],
        "arm_results": result["arm_results"],
        "interpretation": result["interpretation"],
        "non_degenerate": result["non_degenerate"],
        "degeneracy_reason": result["degeneracy_reason"],
        "sleep_driver_pattern": None,
        "cites": {"env_axis_sibling_leg": "V3-EXQ-816b",
                  "predecessor_easy_env_runs": ["V3-EXQ-816", "V3-EXQ-820"],
                  "autopsy": "failure_autopsy_816-820-policy-decomposition-cluster_2026-07-26"},
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
    print(f"pe_control_ok={m['pe_control_ok']} vs_heterogeneous={m['vs_heterogeneous']} "
          f"pe_heterogeneous={m['pe_heterogeneous']} decoupled={m['decoupled']}", flush=True)
    print(f"rho_mean={m['spearman_unc_vs_pe_mean_over_cells']} "
          f"vs_fires={m['vs_trigger_fires_total']} pe_fires={m['pe_trigger_fires_total']} "
          f"cofire={m['cofire_total']}", flush=True)
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
