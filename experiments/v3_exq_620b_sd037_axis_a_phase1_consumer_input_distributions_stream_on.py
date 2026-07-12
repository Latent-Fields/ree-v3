#!/opt/local/bin/python3
"""
V3-EXQ-620b: SD-037 axis (a) Phase 1 substrate-readiness diagnostic, STREAM ON
(SUPERSEDES V3-EXQ-620 -- affective-harm-stream-disabled measurement artifact).

ROOT-CAUSE FIX (diagnosed 2026-06-01, /diagnose-errors via V3-EXQ-625):
  V3-EXQ-620 returned pooled identically-zero distributions across all six
  consumer-input quantities and was read as "axis (a) empirically unmeetable",
  which justified the entire axis-(b) sustained-threat env-curriculum plan. That
  zero reading was a CONFIG artifact, not a substrate finding: the measurement
  reads agent._current_latent.z_harm_a, which is None unless
  cfg.latent.use_affective_harm_stream=True. ARM_PHASE1_BASELINE sets
  gap4_operating=True -> build_config routed to REEConfig.goal_stream(), which
  never forwarded the SD-011 flags, so the AffectiveHarmEncoder was never
  instantiated and z_harm_a stayed None on every latent (every measured quantity
  read exactly 0.0). The env emits harm_obs_a[50] and StepHarness threads it;
  only the config flag was off.

  Fix: build_config(env, arm, enable_affective_harm_stream=True) enables the
  SD-011 stream on the gap4 path (guarded library opt-in added 2026-06-01;
  default False keeps every other gap4 caller bit-identical). This re-run is the
  axis-(a) BASELINE env (default ENV_FISHTANK_KWARGS, NO scheduled-hazard
  curriculum) with the stream actually live. Its purpose is to answer whether
  axis (b) was even necessary: if z_harm_a is now non-zero on the baseline env,
  the axis-(a) p70 recalibration rule becomes measurable and the axis-(b) env
  overlay (V3-EXQ-625b) may be unnecessary. Outcome is tied to the axis-(a)
  acceptance gate (z_harm_a non-zero in >=2/3 seeds) instead of the vacuous
  "ran cleanly" gate.

Logs raw per-step distributions of every quantity that feeds a consumer-module
input gate, so the Phase 2 p70 recalibration rule can read them. Implements
Phase 1 of evidence/planning/sd_037_axis_a_consumer_input_recalibration_plan.md
(REE_assembly).

Background: V3-EXQ-483e FAILed substrate-ceiling at the consumer-input-threshold
layer. SD-037 broadcast saturates (override_signal_nonzero_steps == total
across all ON arms) and the 4-channel MECH-281 cascade is fully wired
(2026-05-30 amend), but BLAConfig.arousal_threshold_on=0.4 / CeAConfig.fast_
route_threshold=0.5 / PAGFreezeGateConfig.theta_freeze=2.0 + duration_input_
threshold=0.4 / dACC PE floors all sit above fishtank baseline signal
magnitudes, so output * (1 + gain * override_signal) = 0 * anything = 0
across every consumer except lateral_pfc.

This experiment does NOT validate any scientific hypothesis. It is a pure
measurement pass on the OFF baseline (SD-037 master OFF, all cascade gains 0;
SD-036+MECH-279+SalienceCoordinator+dACC all ON so the substrate surface
matches 483e ARM_0 minus the broadcast). claim_ids=[].

Acceptance: manifest emitted with all six quantity distributions populated
per seed and pooled (no PASS/FAIL on metric values).

The six measured quantities:

  z_harm_a_norm                     -> BLA arousal_threshold_on gate input
  cea_low_freq_magnitude            -> CeA fast_route_threshold gate input
                                       (read from cea._last_low_freq_mag,
                                        the same scalar CeA compares against
                                        fast_route_threshold at line 337)
  z_harm_a_instant_val              -> PAG duration_input_threshold gate input
                                       (per-tick scalar z_harm_a magnitude
                                        the PAG compares against
                                        duration_input_threshold)
  pag_sustained_product             -> PAG theta_freeze gate input
                                       (z_harm_a * pag._duration_above_threshold;
                                        same product compared against
                                        theta_freeze at freeze_gate.py line 244)
  bla_pe_magnitude                  -> BLA PE channel (read from bla.pe_magnitude;
                                        z_harm_a - E2_harm_a residual scalar)
  dacc_pe                           -> dACC PE input (read from dacc._last_bundle['pe'])

Per quantity, per seed: min / max / mean / std / p10 / p25 / p50 / p70 /
p80 / p90 / p95 / p99, 20-bin histogram bin edges + counts, and
zero_fraction (fraction of ticks exactly zero).
Also pooled across seeds.

The Phase 2 p70 recalibration rule reads p70 of each distribution and assigns
the corresponding consumer-module config knob to that value (with floors and
the current default as ceiling). Phase 3 verification re-runs the OFF baseline
with the recalibrated knobs and checks that BLA/CeA/PAG/dACC outputs lift
above zero in >= 2/3 seeds. Phase 4 (V3-EXQ-483f, deferred) re-runs the 483e
4-arm shape on the recalibrated substrate.

claim_ids=[]
experiment_purpose=diagnostic
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

import numpy as np

from experiment_protocol import emit_outcome  # noqa: E402
from _harness import StepHarness, StepHooks  # noqa: E402
from _lib.goal_pipeline_tier1 import (  # noqa: E402
    ArmSpec,
    ENV_FISHTANK_KWARGS,
    EVAL_EPISODES_DEFAULT,
    SEEDS_DEFAULT,
    STEPS_PER_EPISODE_DEFAULT,
    WARMUP_EPISODES_DEFAULT,
    build_config,
    make_env,
    warmup_train,
)
from ree_core.agent import REEAgent
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_620b_sd037_axis_a_phase1_consumer_input_distributions_stream_on"
QUEUE_ID = "V3-EXQ-620b"
SUPERSEDES = "V3-EXQ-620"
CLAIM_IDS: List[str] = []
EXPERIMENT_PURPOSE = "diagnostic"

# Axis-(a) substrate-readiness gate. The axis-(a) question is simply whether the
# affective harm stream produces non-zero z_harm_a on the BASELINE env (no
# curriculum). >=2/3 seeds non-zero clears it. Pooled p70 is reported for the
# Phase 2 recalibration rule but is informational here.
ZHARMA_NONZERO_SEEDS_MIN_FRAC = 2 / 3

SEEDS = SEEDS_DEFAULT  # [42, 7, 19]
WARMUP_EPISODES = WARMUP_EPISODES_DEFAULT  # 50
EVAL_EPISODES = EVAL_EPISODES_DEFAULT  # 10
STEPS_PER_EPISODE = STEPS_PER_EPISODE_DEFAULT  # 200

# Substrate matches 483e ARM_0 OFF_OFF (cascade gains > 0 but broadcast pinned at 0
# via raised recruitment_threshold). For Phase 1 we want SD-037 master OFF AND all
# cascade gains 0 so the measurement is of the agent's pure natural baseline -- the
# broadcast neither fires nor amplifies anything. PAG / SalienceCoordinator / dACC /
# amygdala all ON so the substrate surface matches the 483e ARM_0 configuration
# minus the broadcast (so the distributions we measure ARE what the consumer modules
# see at baseline in the validation env).
COMMON_CONSUMER_FLAGS: Dict[str, Any] = {
    "use_salience_coordinator": True,
    "use_lateral_pfc_analog": True,
    "use_amygdala_analog": True,
    "use_bla_analog": True,
    "use_cea_analog": True,
}

ARM_PHASE1_BASELINE = ArmSpec(
    "PHASE1_BASELINE",
    gap4_operating=True,
    use_gabaergic_decay=True,
    use_pag_freeze_gate=True,
    use_broadcast_override=False,
    extra_config={
        **COMMON_CONSUMER_FLAGS,
        "use_dacc": True,
        # Cascade gains all 0.0 -- structurally inert even if some upstream
        # caller turned the broadcast on; explicit for clarity.
        "override_pfc_eta_gain": 0.0,
        "override_bla_encoding_gain": 0.0,
        "override_cea_amplitude_gain": 0.0,
        "override_beta_interrupt_gain": 0.0,
        # dACC bias is clipped by dacc_bias_max_abs (default 0.0 -- structurally
        # silent output). Non-negotiable for this diagnostic (per the plan doc
        # Phase 1 spec): set to a non-trivial value so dACC PE distribution is
        # OBSERVABLE. Value chosen to match the bias_scale of lateral_pfc / curiosity /
        # tonic_vigor / mech295 sibling score-bias modules (0.1) so the dACC bias
        # contribution is commensurate with siblings when the PE input clears the
        # internal PE floor.
        "dacc_bias_max_abs": 0.1,
        # Non-zero weight so the dACC bundle actually composes a non-trivial signal
        # (separately from the bias clip; the precision_scale + effort_cost defaults
        # produce a bundle even at weight=0 but the bundle's downstream effect needs
        # weight > 0 to be observable). 0.1 mirrors sibling weights.
        "dacc_weight": 0.1,
    },
)

# Quantities measured per step. Each entry: (key, description).
MEASURED_QUANTITIES = [
    ("z_harm_a_norm", "BLA arousal_threshold_on input"),
    ("cea_low_freq_magnitude", "CeA fast_route_threshold input"),
    ("z_harm_a_instant_val", "PAG duration_input_threshold input"),
    ("pag_sustained_product", "PAG theta_freeze input"),
    ("bla_pe_magnitude", "BLA PE channel"),
    ("dacc_pe", "dACC PE input"),
]

PERCENTILES = [10, 25, 50, 70, 80, 90, 95, 99]
HISTOGRAM_BINS = 20
ZERO_FLOOR = 1e-9  # values below this counted as exactly zero in zero_fraction


def _z_harm_a_norm(agent: REEAgent) -> float:
    lat = getattr(agent, "_current_latent", None)
    if lat is None:
        return 0.0
    za = getattr(lat, "z_harm_a", None)
    if za is None:
        return 0.0
    return float(za.norm().item())


def _cea_low_freq_magnitude(agent: REEAgent) -> float:
    cea = getattr(agent, "cea", None)
    if cea is None:
        return 0.0
    return float(getattr(cea, "_last_low_freq_mag", 0.0))


def _pag_duration_above(agent: REEAgent) -> int:
    pg = getattr(agent, "pag_freeze_gate", None)
    if pg is None:
        return 0
    return int(getattr(pg, "_duration_above_threshold", 0))


def _bla_pe_magnitude(agent: REEAgent) -> float:
    bla = getattr(agent, "bla", None)
    if bla is None:
        return 0.0
    return float(getattr(bla, "_last_pe_magnitude", 0.0))


def _dacc_pe(agent: REEAgent) -> float:
    da = getattr(agent, "dacc", None)
    if da is None:
        return 0.0
    bundle = getattr(da, "_last_bundle", None)
    if bundle is None or not isinstance(bundle, dict):
        return 0.0
    pe = bundle.get("pe", 0.0)
    try:
        return float(pe)
    except (TypeError, ValueError):
        return 0.0


def measure_step(agent: REEAgent) -> Dict[str, float]:
    """One per-tick sample across the six measured quantities."""
    z_norm = _z_harm_a_norm(agent)
    duration = _pag_duration_above(agent)
    return {
        "z_harm_a_norm": z_norm,
        "cea_low_freq_magnitude": _cea_low_freq_magnitude(agent),
        # PAG duration_input_threshold compares the PER-TICK z_harm_a norm
        # against duration_input_threshold (line 226 of freeze_gate.py). The
        # measured quantity is the same as z_harm_a_norm THIS tick.
        "z_harm_a_instant_val": z_norm,
        # PAG theta_freeze input = z * duration (line 244 of freeze_gate.py).
        # We compute it on the post-step state so duration reflects the count
        # AFTER PAG's own tick this step.
        "pag_sustained_product": z_norm * float(duration),
        "bla_pe_magnitude": _bla_pe_magnitude(agent),
        "dacc_pe": _dacc_pe(agent),
    }


def summarise_distribution(values: List[float]) -> Dict[str, Any]:
    """Convert a per-tick sample list into the canonical summary dict."""
    if not values:
        return {
            "n": 0,
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "percentiles": {f"p{p}": 0.0 for p in PERCENTILES},
            "histogram": {"bin_edges": [], "counts": []},
            "zero_fraction": 0.0,
        }
    arr = np.asarray(values, dtype=np.float64)
    n = int(arr.size)
    pct_vals = np.percentile(arr, PERCENTILES)
    counts, edges = np.histogram(arr, bins=HISTOGRAM_BINS)
    n_zero = int(np.sum(arr < ZERO_FLOOR))
    return {
        "n": n,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "percentiles": {
            f"p{p}": float(pct_vals[i]) for i, p in enumerate(PERCENTILES)
        },
        "histogram": {
            "bin_edges": [float(x) for x in edges.tolist()],
            "counts": [int(c) for c in counts.tolist()],
        },
        "zero_fraction": float(n_zero) / float(n),
    }


def eval_phase1_diagnostic(
    agent: REEAgent,
    env,
    *,
    num_episodes: int,
    steps_per_episode: int,
    seed: int,
    arm_label: str,
) -> Dict[str, Any]:
    """Eval loop measuring the six consumer-input quantities every tick."""
    # Per-quantity sample arrays for this seed
    samples: Dict[str, List[float]] = {q: [] for q, _ in MEASURED_QUANTITIES}
    total_steps = [0]
    action_counts: Dict[int, int] = {}

    def on_post_step(*, agent, latent, action, obs_dict, ticks, step, **kwargs) -> None:
        total_steps[0] += 1
        meas = measure_step(agent)
        for k, v in meas.items():
            samples[k].append(v)
        aidx = int(action.argmax(dim=-1).item())
        action_counts[aidx] = action_counts.get(aidx, 0) + 1

    hooks = StepHooks(on_post_step=on_post_step)
    harness = StepHarness(agent, env, train_mode=False, hooks=hooks, seed=seed)
    agent.eval()

    for ep in range(num_episodes):
        _, obs_dict = env.reset()
        agent.reset()
        harness.reset()
        for _ in range(steps_per_episode):
            result = harness.step(obs_dict)
            obs_dict = result.next_obs_dict
            if result.done:
                break

    return {
        "arm": arm_label,
        "seed": int(seed),
        "total_eval_steps": int(total_steps[0]),
        "action_counts": {str(k): int(v) for k, v in action_counts.items()},
        "per_quantity_samples": samples,
    }


def render_markdown_summary(
    cohort_summary: Dict[str, Any],
    pooled_summary: Dict[str, Any],
    *,
    timestamp_utc: str,
    queue_id: str,
    run_id: str,
) -> str:
    lines: List[str] = []
    lines.append(f"# SD-037 Axis (a) Phase 1 -- Consumer-Input Distributions")
    lines.append("")
    lines.append(f"- Queue id: `{queue_id}`")
    lines.append(f"- Run id: `{run_id}`")
    lines.append(f"- Timestamp UTC: `{timestamp_utc}`")
    lines.append(f"- Manifest: `evidence/experiments/{run_id}.json`")
    lines.append(f"- Plan: [sd_037_axis_a_consumer_input_recalibration_plan.md](sd_037_axis_a_consumer_input_recalibration_plan.md)")
    lines.append("")
    lines.append("Phase 1 substrate-readiness diagnostic. Substrate matches 483e ARM_0 OFF_OFF (PAG-engaging env via SD-036 + MECH-279, SalienceCoordinator + dACC + amygdala all enabled) but with `use_broadcast_override=False` and all four MECH-281 cascade gains 0.0. Pure baseline: this is the natural fishtank distribution every consumer-module input gate sees when the broadcast is silent.")
    lines.append("")
    lines.append("## Phase 2 recalibration table (p70 candidates)")
    lines.append("")
    lines.append("| Consumer-input quantity | Knob | Current default | Measured p70 (pooled) | Phase 2 candidate |")
    lines.append("|---|---|---|---|---|")
    knob_map = [
        ("z_harm_a_norm", "BLAConfig.arousal_threshold_on", 0.4),
        ("cea_low_freq_magnitude", "CeAConfig.fast_route_threshold", 0.5),
        ("z_harm_a_instant_val", "PAGFreezeGateConfig.duration_input_threshold", 0.4),
        ("pag_sustained_product", "PAGFreezeGateConfig.theta_freeze", 2.0),
        ("bla_pe_magnitude", "(BLA PE channel; informational)", None),
        ("dacc_pe", "DACCConfig.dacc_precision_scale (informational; rescale not threshold)", None),
    ]
    pooled = pooled_summary.get("per_quantity", {})
    for qkey, knob, default in knob_map:
        p70 = pooled.get(qkey, {}).get("percentiles", {}).get("p70", 0.0)
        if default is None:
            candidate_str = "(see plan Phase 2 dACC rescale rule)"
            default_str = "n/a"
        else:
            floor = 0.05 if qkey != "pag_sustained_product" else 0.1
            candidate = max(floor, min(default, p70))
            candidate_str = f"{candidate:.4f}"
            default_str = f"{default:.2f}"
        lines.append(f"| `{qkey}` | `{knob}` | {default_str} | {p70:.4f} | {candidate_str} |")
    lines.append("")
    lines.append("Floor / ceiling per Phase 2 plan: floor 0.05 (theta_freeze 0.1); ceiling current default (so a high-p70 cannot raise a threshold above its current value -- Phase 2 NEVER raises a default, only lowers).")
    lines.append("")
    lines.append("## Per-seed distributions")
    lines.append("")
    rows = cohort_summary.get("rows", [])
    for row in rows:
        seed = row.get("seed")
        lines.append(f"### Seed {seed}")
        lines.append("")
        lines.append("| Quantity | min | max | mean | std | p70 | p90 | zero_frac |")
        lines.append("|---|---|---|---|---|---|---|---|")
        per_q = row.get("per_quantity", {})
        for qkey, _ in MEASURED_QUANTITIES:
            d = per_q.get(qkey, {})
            pc = d.get("percentiles", {})
            lines.append(
                f"| `{qkey}` | {d.get('min', 0.0):.4f} | {d.get('max', 0.0):.4f} | "
                f"{d.get('mean', 0.0):.4f} | {d.get('std', 0.0):.4f} | "
                f"{pc.get('p70', 0.0):.4f} | {pc.get('p90', 0.0):.4f} | "
                f"{d.get('zero_fraction', 0.0):.3f} |"
            )
        lines.append("")
    lines.append("## Pooled distributions (across seeds)")
    lines.append("")
    lines.append("| Quantity | n | min | max | mean | std | p10 | p25 | p50 | p70 | p80 | p90 | p95 | p99 | zero_frac |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for qkey, _ in MEASURED_QUANTITIES:
        d = pooled.get(qkey, {})
        pc = d.get("percentiles", {})
        lines.append(
            f"| `{qkey}` | {d.get('n', 0)} | {d.get('min', 0.0):.4f} | "
            f"{d.get('max', 0.0):.4f} | {d.get('mean', 0.0):.4f} | "
            f"{d.get('std', 0.0):.4f} | "
            f"{pc.get('p10', 0.0):.4f} | {pc.get('p25', 0.0):.4f} | "
            f"{pc.get('p50', 0.0):.4f} | {pc.get('p70', 0.0):.4f} | "
            f"{pc.get('p80', 0.0):.4f} | {pc.get('p90', 0.0):.4f} | "
            f"{pc.get('p95', 0.0):.4f} | {pc.get('p99', 0.0):.4f} | "
            f"{d.get('zero_fraction', 0.0):.3f} |"
        )
    lines.append("")
    lines.append("## Routing")
    lines.append("")
    lines.append("PASS -> Phase 2 deterministic recalibration rule (read p70 from the pooled distribution; apply per-experiment overrides; queue Phase 3 verification diagnostic). FAIL with all-zero z_harm_a_norm + bla_pe (would mean even the affective stream itself is silent at baseline) -> route to axis (b) SD-029-style sustained-threat env curriculum without waiting on a p60 / p80 sweep, because static threshold lowering cannot help when the upstream signal is absent.")
    lines.append("")
    return "\n".join(lines)


def run_experiment(
    *,
    seeds: List[int],
    warmup_episodes: int,
    eval_episodes: int,
    steps_per_episode: int,
    dry_run: bool,
) -> Dict[str, Any]:
    arm = ARM_PHASE1_BASELINE
    rows: List[Dict[str, Any]] = []
    pooled_samples: Dict[str, List[float]] = {q: [] for q, _ in MEASURED_QUANTITIES}

    total_runs = len(seeds)  # 1 condition

    if dry_run:
        # Same shape, drastically reduced budget.
        warmup_episodes = 2
        eval_episodes = 2
        steps_per_episode = 30
        seeds = seeds[:1]
        total_runs = len(seeds)

    for run_idx, seed in enumerate(seeds):
        print(f"Seed {seed} Condition {arm.arm_id}", flush=True)
        env = make_env(seed, env_kwargs=ENV_FISHTANK_KWARGS)
        env._exq_env_kwargs = dict(ENV_FISHTANK_KWARGS)
        # enable_affective_harm_stream=True is the V3-EXQ-620 root-cause fix:
        # the gap4_operating path otherwise leaves the SD-011 affective harm
        # stream off, so z_harm_a is None and every measured quantity reads 0.0.
        cfg = build_config(env, arm, enable_affective_harm_stream=True)
        import random as _random
        _random.seed(seed)
        np.random.seed(seed)
        import torch as _torch
        _torch.manual_seed(seed)
        agent = REEAgent(cfg)

        # Warmup: train E1/E2/E3 the same way 483e does. Random-policy agents
        # would underestimate consumer-input magnitudes because the affective
        # stream needs a representative-policy trajectory profile.
        warmup_train(
            agent,
            env,
            num_episodes=warmup_episodes,
            steps_per_episode=steps_per_episode,
            label=f"phase1_seed_{seed}",
            progress_total_episodes=warmup_episodes + eval_episodes,
        )

        # Eval pass with the measurement hook.
        eval_row = eval_phase1_diagnostic(
            agent,
            env,
            num_episodes=eval_episodes,
            steps_per_episode=steps_per_episode,
            seed=seed,
            arm_label=arm.arm_id,
        )

        # Summarise per-quantity distributions for this seed.
        per_quantity: Dict[str, Any] = {}
        for qkey, _ in MEASURED_QUANTITIES:
            seed_samples = eval_row["per_quantity_samples"][qkey]
            per_quantity[qkey] = summarise_distribution(seed_samples)
            pooled_samples[qkey].extend(seed_samples)
        eval_row.pop("per_quantity_samples", None)
        eval_row["per_quantity"] = per_quantity
        # Progress instrumentation: one verdict line per seed.
        # Verdict is always PASS at the per-seed level because this is a measurement
        # pass -- the overall outcome is set on the run-level manifest after the
        # pooled distribution check.
        n_steps = int(eval_row.get("total_eval_steps", 0))
        print(
            f"verdict: PASS  seed={seed}  total_eval_steps={n_steps}",
            flush=True,
        )

        # Final progress line at run boundary for the runner's bar.
        print(
            f"  [train] phase1_seed_{seed} ep {warmup_episodes + eval_episodes}"
            f"/{warmup_episodes + eval_episodes}",
            flush=True,
        )

        rows.append(eval_row)

    cohort_summary = {
        "rows": rows,
    }

    # Pooled summary.
    pooled_per_quantity: Dict[str, Any] = {}
    for qkey, _ in MEASURED_QUANTITIES:
        pooled_per_quantity[qkey] = summarise_distribution(pooled_samples[qkey])
    pooled_summary = {"per_quantity": pooled_per_quantity}

    # V3-EXQ-620b: outcome TIED to the axis-(a) substrate-readiness gate, not the
    # vacuous "ran cleanly" gate. The gate is: z_harm_a_norm zero_fraction < 1.0
    # (i.e. the SD-011 affective stream produced a non-zero signal at least once)
    # in >= 2/3 seeds on the baseline env. data_populated is retained as a
    # precondition so a crashed/empty run still FAILs.
    n_seeds = len(rows)
    nonzero_seeds = 0
    per_seed_zharma: List[Dict[str, Any]] = []
    for row in rows:
        zf = float(row.get("per_quantity", {}).get("z_harm_a_norm", {}).get("zero_fraction", 1.0))
        ok = zf < 1.0
        if ok:
            nonzero_seeds += 1
        per_seed_zharma.append({"seed": row.get("seed"), "zero_fraction": zf, "ok": ok})
    required_seeds = max(2, math.ceil(ZHARMA_NONZERO_SEEDS_MIN_FRAC * n_seeds)) if n_seeds else 0
    zharma_nonzero_pass = n_seeds > 0 and nonzero_seeds >= required_seeds
    acceptance = {
        "C1_zharma_nonzero": zharma_nonzero_pass,
        "C1_detail": {
            "required": ">=2/3 seeds with z_harm_a_norm zero_fraction < 1.0",
            "achieved": f"{nonzero_seeds}/{n_seeds}",
            "per_seed": per_seed_zharma,
        },
        "pooled_z_harm_a_norm_p70": float(
            pooled_per_quantity.get("z_harm_a_norm", {}).get("percentiles", {}).get("p70", 0.0)
        ),
        "acceptance_pass": bool(zharma_nonzero_pass),
    }

    data_populated = all(
        pooled_per_quantity[qkey].get("n", 0) > 0 for qkey, _ in MEASURED_QUANTITIES
    )
    outcome = "PASS" if (data_populated and acceptance["acceptance_pass"]) else "FAIL"

    return {
        "cohort_summary": cohort_summary,
        "pooled_summary": pooled_summary,
        "outcome": outcome,
        "outcome_basis": "z_harm_a_norm non-zero (zero_fraction < 1.0) in >=2/3 seeds AND all six pooled distributions populated",
        "data_populated": data_populated,
        "acceptance": acceptance,
        "total_runs": total_runs,
    }


def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description=EXPERIMENT_TYPE)
    parser.add_argument("--dry-run", action="store_true", help="Reduced-budget smoke")
    args = parser.parse_args(argv)
    _run_started = datetime.now(timezone.utc)

    timestamp_utc = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{timestamp_utc}_v3"
    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.json"

    result = run_experiment(
        seeds=SEEDS,
        warmup_episodes=WARMUP_EPISODES,
        eval_episodes=EVAL_EPISODES,
        steps_per_episode=STEPS_PER_EPISODE,
        dry_run=args.dry_run,
    )

    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "outcome": result["outcome"],
        "outcome_basis": result.get("outcome_basis", ""),
        "timestamp_utc": timestamp_utc,
        "evidence_direction": "non_contributory",
        "supersedes": SUPERSEDES,
        "supersedes_run_type": "v3_exq_620_sd037_axis_a_phase1_consumer_input_distributions",
        "affective_harm_stream_enabled": True,
        "acceptance_gate_axis_a": result["acceptance"],
        "seeds": SEEDS,
        "warmup_episodes": WARMUP_EPISODES,
        "eval_episodes": EVAL_EPISODES,
        "steps_per_episode": STEPS_PER_EPISODE,
        "env_kwargs": dict(ENV_FISHTANK_KWARGS),
        "arm": {
            "arm_id": ARM_PHASE1_BASELINE.arm_id,
            "gap4_operating": ARM_PHASE1_BASELINE.gap4_operating,
            "use_gabaergic_decay": ARM_PHASE1_BASELINE.use_gabaergic_decay,
            "use_pag_freeze_gate": ARM_PHASE1_BASELINE.use_pag_freeze_gate,
            "use_broadcast_override": ARM_PHASE1_BASELINE.use_broadcast_override,
            "extra_config": ARM_PHASE1_BASELINE.extra_config,
        },
        "measured_quantities": [
            {"key": k, "description": d} for k, d in MEASURED_QUANTITIES
        ],
        "percentiles_reported": PERCENTILES,
        "histogram_bins": HISTOGRAM_BINS,
        "cohort_summary": result["cohort_summary"],
        "pooled_summary": result["pooled_summary"],
    }

    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=False,
        config=manifest.get("config"),
        seeds=SEEDS,
        script_path=Path(__file__),
        elapsed_seconds=(datetime.now(timezone.utc) - _run_started).total_seconds(),
    )
    print(f"manifest written: {out_path}", flush=True)

    # Plan-side markdown summary (under evidence/planning/).
    plan_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "planning"
    plan_dir.mkdir(parents=True, exist_ok=True)
    md_path = plan_dir / f"sd037_consumer_input_distributions_{timestamp_utc}.md"
    md_text = render_markdown_summary(
        result["cohort_summary"],
        result["pooled_summary"],
        timestamp_utc=timestamp_utc,
        queue_id=QUEUE_ID,
        run_id=run_id,
    )
    md_path.write_text(md_text, encoding="utf-8")
    print(f"plan summary written: {md_path}", flush=True)

    outcome_raw = str(manifest["outcome"]).upper()
    return outcome_raw, str(out_path)


if __name__ == "__main__":
    _outcome_raw, _manifest_path = main()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=_manifest_path,
    )
