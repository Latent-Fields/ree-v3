"""V3-EXQ-924: E3 score variance decomposition, remeasured post-scorer-fix.

DIAGNOSTIC, NOT a claim-adjudicating falsifier (EXPERIMENT_PURPOSE="diagnostic",
claim_ids=[], matching V3-EXQ-571/609/858 precedent). Excluded from governance
confidence/conflict scoring by construction.

WHY THIS RUN EXISTS (session mech357-pressure-scoping-11e9c9, F-dominance causal-
localisation scoping, 2026-08-12): every existing MECH-439/ARC-062 F-dominance
measurement -- V3-EXQ-571 (the historical "F explains 88-89% of E3 temporal score
variance" figure), 689d, 689i, 699/699b, 705b, 707/707a/707b/707c, 709/711/713,
719a, 852, and 858 (the ARC-062 GOV-FANOUT-1 Leg P-B f_weight attenuation ladder) --
ran BEFORE commit 193bbec (SD-E3-SCORER-COMPLETION, 2026-08-09). Verified directly
against that commit's diff: prior to it, `compute_reality_cost()` and
`compute_harm_cost_fallback()` UNCONDITIONALLY subtracted the output of two
`nn.Sequential` heads (`reality_scorer`, `harm_cost_fallback_scorer`) touched by no
loss anywhere in ree_core -- pure random-init noise added into literal F (and the
harm fallback term) on every trajectory score, in every condition, on the live
selection path, in EVERY run in that lineage. There was no flag to disable it; the
flag (`E3Config.e3_include_untrained_fallback_scorers`, default False = the fix)
did not exist until this commit. So no prior F-dominance measurement reflects the
literal F that exists today (coherence-only: pure z_world transition smoothness).

This experiment answers, cheaply, whether that matters before any larger frozen-
state/frozen-candidate causal-replay diagnostic is built: does literal F's share of
E3 committed-selection score variance change materially once the untrained-scorer
noise is removed? It reuses V3-EXQ-571's exact `temporal_fraction` formula
(var(component over time) / var(sum-of-components over time)) for direct numeric
comparability to the historical 88-89% figure, PLUS V3-EXQ-609's per-candidate
inter-candidate-spread state classification (state_A_zero_emission /
state_B_methodology_mismatch / state_C_per_candidate_signal), which V3-EXQ-571's
own root-cause doc (v3_exq_571_root_cause_2026-05-25.md) found the original
mean-collapsed bias_fraction metric was structurally blind to.

NEW vs both predecessors: a `scorer_state` dimension -- every (arm, seed) cell
runs BOTH `fixed` (e3_include_untrained_fallback_scorers=False, the current
default/live path) AND `legacy` (=True, reproduces the pre-2026-08-09 behaviour
bit-for-bit per the fix commit's own docstring) so the delta the scorer fix makes
is measured directly in the SAME run, on the SAME substrate revision, rather than
inferred by comparing against a stale historical manifest with no substrate_hash.

No training occurs (frozen random-init weights; a fixed policy stepping the env
for N_STEPS ticks per cell, matching 571/609's own design) -- this is architectural
score-decomposition instrumentation, not a behavioural or learning experiment.
Cheap by construction: no P0/P1 phases, no arm-reuse-fingerprint baseline to bank
(nothing here is a checkpoint worth banking), runtime is seconds per cell.

GOV-REUSE-1 check performed before authoring (REE_assembly/scripts/reanalysis_query.py
query --readout temporal_fraction_f --require-readout): only V3-EXQ-571's own two
manifests carry a comparable readout, both UNVERIFIABLE (no recoverable
substrate_hash -- both predate the 2026-07-12 recording standard). Not recoverable;
this run is necessary.

Substrate-path overlap gate (2.5c) checked against open substrate_queue entries
with substrate_paths touching ree_core/predictors/e3_selector.py: only
SD-E3-SCORER-COMPLETION (severity=degrading, status=implemented_pending_validation,
i.e. the very fix this experiment measures) -- filtered as closed by the gate's own
"implemented" substring match; no corrupting blocker.

Re-derive brake (2.5b): not applicable -- claim_ids=[] means no claim is being
re-tested; this is instrumentation, not adjudication.

Arms (matching V3-EXQ-571/609 for direct comparability):
  ARM_0_baseline:        SP-CEM ON, diversity stack OFF
  ARM_1_diversity_stack: SP-CEM ON, MECH-313 noise_floor + MECH-314 curiosity +
                          MECH-320 tonic_vigor ON

Scorer states:
  fixed:  e3_include_untrained_fallback_scorers=False (current default/live path)
  legacy: e3_include_untrained_fallback_scorers=True  (pre-2026-08-09 behaviour,
          reproduced bit-identically per the fix commit's own docstring)

f_weight is left at its SD-085 default (1.0, unattenuated) in every cell --
this experiment isolates the scorer-contamination question, not the
f_weight-attenuation question (that is V3-EXQ-858's separate, still-pending-a-
gate-fix-rerun question).

Headline outputs:
  temporal_fraction["f"] per (arm, scorer_state) -- directly comparable to the
      historical 88-89% figure.
  delta_fixed_minus_legacy per component -- how much the scorer fix moves each
      component's variance share, isolating the fix's effect from arm/seed noise.
  substrate_state_per_channel per (arm, scorer_state) -- does removing the noise
      change which channels read as state_A/B/C.

Outcome: PASS if decomp data was collected in every one of the 2 arms x 2 scorer
states (independent of what the numbers turn out to say -- this is a measurement
run, not a hypothesis test with a pass/fail bar).
"""

import sys
import time
import argparse
import numpy as np
import torch
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiment_protocol import emit_outcome
from ree_core.environment.causal_grid_world import CausalGridWorld
from ree_core.agent import REEAgent
from ree_core.utils.config import REEConfig
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from experiments._lib.manifest_core import stamp_recording_core
from experiments._lib.arm_fingerprint import arm_cell
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator
from experiments._metrics import check_degeneracy

EXPERIMENT_PURPOSE = "diagnostic"
EXPERIMENT_TYPE = "v3_exq_924_e3_score_variance_decomp_scorer_fix_remeasure"
SUPERSEDES = None  # Not a supersede: 571/609 stand as the pre-fix historical record.

SEEDS = [42, 123, 456]
# E3 fires only every heartbeat.e3_steps_per_tick env steps (default 10) -- confirmed
# empirically during authoring: agent.e3.last_score_decomp is BIT-IDENTICAL across 7
# consecutive act_with_split_obs() calls before changing (a genuine latch, not sampling
# noise). N_STEPS=600 env steps therefore yields ~60 GENUINE fresh E3 selections per
# cell (not 600) -- the latch is cleared and non-fresh ticks are skipped below, per the
# fresh-selection-only measurement safeguard, so this is the true sample size, not a
# pseudo-replicated one.
N_STEPS = 600
GRID_SIZE = 8
NUM_HAZARDS = 1

WARN_LOW_FRACTION = 0.01
WARN_HIGH_FRACTION = 0.90

SPREAD_ZERO_EPS = 1e-9
MEAN_ZERO_EPS = 1e-9

ARM_NAMES = ["ARM_0_baseline", "ARM_1_diversity_stack"]
SCORER_STATES = ["fixed", "legacy"]  # fixed=False (current), legacy=True (pre-08-09)

PER_CHANNEL_BIAS_COMPONENTS = [
    "dacc", "lateral_pfc", "ofc", "gated_policy",
    "mech295_liking", "curiosity", "tonic_vigor", "forced",
]
SCALAR_BIAS_COMPONENTS = ["noise_floor_temp", "total_bias"]

# Score components from e3_selector's _last_traj_components (unchanged schema
# from V3-EXQ-571/609; verified present at ree_core/predictors/e3_selector.py:1285).
SCORE_COMPONENTS = ["f", "f_weighted", "harm_weighted", "residue_weighted",
                    "benefit_weighted", "novelty_weighted", "goal_weighted"]
# temporal_fraction denominator uses the raw (unweighted) additive components,
# matching V3-EXQ-571 exactly: f + harm_weighted + residue_weighted + ... .
# "f_weighted" is tracked alongside for completeness but excluded from the sum
# used as the denominator (it would double-count "f" since f_weight=1.0 here).
TEMPORAL_FRACTION_COMPONENTS = ["f", "harm_weighted", "residue_weighted",
                                 "benefit_weighted", "novelty_weighted", "goal_weighted"]

_ZG = ZGoalStreamAccumulator()


def make_env_and_agent(seed, arm_name, scorer_state):
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = CausalGridWorld(
        seed=None,  # OS entropy at construction; matches 571/609 (no env-seed knob needed for a diagnostic)
        size=GRID_SIZE,
        num_hazards=NUM_HAZARDS,
        use_proxy_fields=True,
    )
    _, obs_dict = env.reset()

    body_obs_dim = obs_dict["body_state"].shape[0]
    world_obs_dim = obs_dict["world_state"].shape[0]
    action_dim = env.action_dim

    config = REEConfig().from_dims(
        body_obs_dim=body_obs_dim,
        world_obs_dim=world_obs_dim,
        action_dim=action_dim,
        harm_obs_dim=51,
        use_support_preserving_cem=True,
        support_preserving_min_first_action_classes=2,
        support_preserving_stratified_elites=True,
        support_preserving_ao_std_floor=0.2,
    )
    config.latent.alpha_world = 0.9  # SD-008: high-fidelity z_world

    if arm_name == "ARM_1_diversity_stack":
        config.use_noise_floor = True
        config.noise_floor_alpha = 0.1
        config.noise_floor_min_temperature = 1.0
        config.use_structured_curiosity = True
        config.use_curiosity_novelty = True
        config.use_curiosity_uncertainty = True
        config.use_curiosity_learning_progress = True
        config.curiosity_novelty_weight = 0.05
        config.curiosity_uncertainty_weight = 0.05
        config.curiosity_learning_progress_weight = 0.05
        config.curiosity_bias_scale = 0.1
        config.use_tonic_vigor = True
        config.tonic_vigor_half_life = 100.0
        config.tonic_vigor_w_action = 0.1
        config.tonic_vigor_w_passive = 0.1
        config.tonic_vigor_bias_scale = 0.1

    # THE experimental manipulation: fixed (current default) vs legacy (pre-fix).
    config.e3.e3_include_untrained_fallback_scorers = (scorer_state == "legacy")
    # Left at SD-085 default -- not the axis under test here.
    config.e3.f_weight = 1.0

    agent = REEAgent(config=config)
    agent.eval()
    agent.e3.e3_score_decomp_enabled = True

    return agent, env, obs_dict


def _decomp_value(last_bias, key):
    if last_bias is None:
        return 0.0
    return float(last_bias.get(key, 0.0))


def _classify_state(mean_val, spread_val):
    if spread_val > SPREAD_ZERO_EPS:
        return "state_C_per_candidate_signal"
    if abs(mean_val) <= MEAN_ZERO_EPS:
        return "state_A_zero_emission"
    return "state_B_methodology_mismatch"


def run_cell(arm_name, scorer_state, seed, dry_run=False):
    agent, env, obs_dict = make_env_and_agent(seed, arm_name, scorer_state)
    # 15, not 5: E3 only fires every ~10 env steps (heartbeat.e3_steps_per_tick),
    # so a 5-step dry-run would latch through its ENTIRE window and smoke-test
    # nothing about the fresh-selection reading path. 15 guarantees >=1 fresh tick.
    n_steps = 15 if dry_run else N_STEPS

    per_step_decomp = []
    per_step_bias_mean = []
    per_step_bias_std = []
    per_step_bias_range = []
    per_step_bias_scalar = {k: [] for k in SCALAR_BIAS_COMPONENTS}
    per_step_scores = []
    n_latched_ticks = 0
    n_fresh_ticks = 0

    for step_i in range(n_steps):
        obs_body = torch.tensor(obs_dict["body_state"]).float()
        obs_world = torch.tensor(obs_dict["world_state"]).float()

        # Clear the E3 diagnostics latch IMMEDIATELY before the call. E3 only
        # fires every heartbeat.e3_steps_per_tick env steps (default 10);
        # act_with_split_obs()'s internal select_action() returns the HELD
        # action on a non-e3 tick without calling e3.select(), so these
        # trackers otherwise hold the PREVIOUS fresh selection's values
        # verbatim -- confirmed empirically (bit-identical across 7
        # consecutive calls) during authoring. Recording every tick
        # unconditionally would re-record one selection as ~10 fake
        # observations. See CLAUDE.md "Sample-size integrity (latched
        # substrate diagnostics)".
        agent.e3.last_score_decomp = {}
        agent._last_score_bias_decomp = {}
        agent.e3.last_scores = None

        with torch.no_grad():
            _action = agent.act_with_split_obs(obs_body, obs_world)

        last_decomp = agent.e3.last_score_decomp
        last_bias = agent._last_score_bias_decomp
        last_scores = agent.e3.last_scores

        is_fresh = bool(last_decomp and last_decomp.get("per_candidate"))
        if is_fresh:
            n_fresh_ticks += 1
            cands = last_decomp["per_candidate"]
            step_comps = {}
            for comp in SCORE_COMPONENTS:
                vals = [c.get(comp, 0.0) for c in cands]
                step_comps[comp] = float(np.mean(vals))
            per_step_decomp.append(step_comps)

            if isinstance(last_bias, dict) and last_bias:
                mean_step, std_step, range_step = {}, {}, {}
                for ch in PER_CHANNEL_BIAS_COMPONENTS:
                    mean_step[ch] = _decomp_value(last_bias, ch)
                    std_step[ch] = _decomp_value(last_bias, f"{ch}_std_across_K")
                    range_step[ch] = _decomp_value(last_bias, f"{ch}_bias_range_mean")
                per_step_bias_mean.append(mean_step)
                per_step_bias_std.append(std_step)
                per_step_bias_range.append(range_step)
                for ch in SCALAR_BIAS_COMPONENTS:
                    per_step_bias_scalar[ch].append(_decomp_value(last_bias, ch))

            if last_scores is not None and last_scores.numel() > 1:
                per_step_scores.append(last_scores.cpu().tolist())
        else:
            n_latched_ticks += 1
            # record nothing this tick -- not a fresh E3 selection

        if isinstance(_action, torch.Tensor):
            action_idx = int(_action.argmax().item())
        else:
            action_idx = int(_action)
        _, _, terminated, _, obs_dict = env.step(action_idx % env.action_dim)
        if terminated:
            agent.reset()
            _, obs_dict = env.reset()

        if (step_i + 1) % 100 == 0 or (step_i + 1) == n_steps:
            print(f"    [decomp] {arm_name}/{scorer_state} seed={seed} "
                  f"ep {step_i + 1}/{n_steps} fresh={n_fresh_ticks} "
                  f"latched={n_latched_ticks}", flush=True)

    _ZG.observe(agent)

    result = compute_decomp(
        per_step_decomp, per_step_bias_mean, per_step_bias_std,
        per_step_bias_range, per_step_bias_scalar, per_step_scores,
    )
    result["arm"] = arm_name
    result["scorer_state"] = scorer_state
    result["seed"] = seed
    result["n_env_steps"] = n_steps
    result["n_fresh_ticks"] = n_fresh_ticks
    result["n_latched_ticks"] = n_latched_ticks
    result["n_steps_collected"] = len(per_step_decomp)  # == n_fresh_ticks
    return result


def compute_decomp(per_step_decomp, per_step_bias_mean, per_step_bias_std,
                    per_step_bias_range, per_step_bias_scalar, per_step_scores):
    result = {
        "temporal_variance": {}, "temporal_std": {}, "temporal_fraction": {},
        "bias_inter_candidate_spread_mean": {}, "bias_inter_candidate_range_mean": {},
        "bias_temporal_mean_mean": {}, "substrate_state_per_channel": {},
        "inter_candidate_score_std_mean": 0.0,
        "noise_floor_temp_mean": 0.0, "total_bias_mean_over_time": 0.0,
        "warnings": [],
    }

    if not per_step_decomp:
        result["warnings"].append("no decomp data collected -- e3_score_decomp_enabled not wired")
        return result

    # -- Temporal variance fraction (V3-EXQ-571 formula, exact reuse) --
    comp_arrays = {c: np.array([s.get(c, 0.0) for s in per_step_decomp], dtype=np.float64)
                   for c in SCORE_COMPONENTS}
    total_per_step = np.zeros(len(per_step_decomp), dtype=np.float64)
    for c in TEMPORAL_FRACTION_COMPONENTS:
        total_per_step += comp_arrays[c]
    total_var = float(np.var(total_per_step)) + 1e-12

    for c in SCORE_COMPONENTS:
        v = float(np.var(comp_arrays[c]))
        result["temporal_variance"][c] = v
        result["temporal_std"][c] = float(np.std(comp_arrays[c]))
        result["temporal_fraction"][c] = v / total_var
        if result["temporal_fraction"][c] > WARN_HIGH_FRACTION:
            result["warnings"].append(f"temporal: {c} fraction={result['temporal_fraction'][c]:.3f} > {WARN_HIGH_FRACTION} (dominant)")

    # -- Per-channel inter-candidate spread + state classification (V3-EXQ-609) --
    if per_step_bias_mean:
        for ch in PER_CHANNEL_BIAS_COMPONENTS:
            means = np.array([s.get(ch, 0.0) for s in per_step_bias_mean], dtype=np.float64)
            stds = np.array([s.get(ch, 0.0) for s in per_step_bias_std], dtype=np.float64)
            ranges = np.array([s.get(ch, 0.0) for s in per_step_bias_range], dtype=np.float64)

            mean_over_time = float(np.mean(means))
            spread_over_time = float(np.mean(stds))
            result["bias_temporal_mean_mean"][ch] = mean_over_time
            result["bias_inter_candidate_spread_mean"][ch] = spread_over_time
            result["bias_inter_candidate_range_mean"][ch] = float(np.mean(ranges))
            result["substrate_state_per_channel"][ch] = _classify_state(mean_over_time, spread_over_time)

        if per_step_bias_scalar.get("noise_floor_temp"):
            result["noise_floor_temp_mean"] = float(np.mean(per_step_bias_scalar["noise_floor_temp"]))
        if per_step_bias_scalar.get("total_bias"):
            result["total_bias_mean_over_time"] = float(np.mean(per_step_bias_scalar["total_bias"]))

    if per_step_scores:
        stds = [float(np.std(s)) for s in per_step_scores if len(s) > 1]
        if stds:
            result["inter_candidate_score_std_mean"] = float(np.mean(stds))

    return result


def run_experiment(dry_run=False, started_at=None):
    all_results = {}
    cell_fingerprints = []
    n_cells = len(ARM_NAMES) * len(SCORER_STATES) * len(SEEDS)
    cell_i = 0

    for arm_name in ARM_NAMES:
        for scorer_state in SCORER_STATES:
            key = f"{arm_name}__{scorer_state}"
            all_results[key] = []
            print(f"\n== {key} ==", flush=True)
            print(f"Seed - Condition {key}", flush=True)
            for seed in SEEDS:
                cell_i += 1
                config_slice = {
                    "arm_name": arm_name,
                    "scorer_state": scorer_state,
                    "e3_include_untrained_fallback_scorers": (scorer_state == "legacy"),
                    "f_weight": 1.0,
                    "grid_size": GRID_SIZE,
                    "num_hazards": NUM_HAZARDS,
                    "n_steps": N_STEPS,
                }
                with arm_cell(seed, config_slice=config_slice, script_path=Path(__file__)) as cell:
                    r = run_cell(arm_name, scorer_state, seed, dry_run=dry_run)
                    cell.stamp(r)
                cell_fingerprints.append(r.get("arm_fingerprint"))
                all_results[key].append(r)
                # PASS requires a genuine sample of FRESH E3 selections, not merely
                # that the env loop completed -- n_steps_collected == n_fresh_ticks
                # by construction (see run_cell), so this also guards against the
                # latch degenerating to near-zero fresh ticks (e.g. a cadence
                # misconfiguration).
                verdict = "PASS" if r["n_fresh_ticks"] >= 10 else "FAIL"
                print(f"  [decomp] {key} seed={seed} ep {cell_i}/{n_cells} "
                      f"env_steps={r['n_env_steps']} fresh={r['n_fresh_ticks']} "
                      f"latched={r['n_latched_ticks']}", flush=True)
                print(f"verdict: {verdict}", flush=True)

    # -- Per (arm, scorer_state) summaries --
    cell_summaries = {}
    for arm_name in ARM_NAMES:
        for scorer_state in SCORER_STATES:
            key = f"{arm_name}__{scorer_state}"
            results = all_results[key]
            summary = {"arm": arm_name, "scorer_state": scorer_state, "n_seeds": len(results)}
            summary["n_fresh_ticks_total"] = sum(r["n_fresh_ticks"] for r in results)
            summary["n_latched_ticks_total"] = sum(r["n_latched_ticks"] for r in results)
            summary["n_env_steps_total"] = sum(r["n_env_steps"] for r in results)
            for c in SCORE_COMPONENTS:
                fracs = [r["temporal_fraction"].get(c, 0.0) for r in results]
                summary[f"temporal_fraction_{c}"] = float(np.mean(fracs))
            for ch in PER_CHANNEL_BIAS_COMPONENTS:
                spreads = [r["bias_inter_candidate_spread_mean"].get(ch, 0.0) for r in results]
                summary[f"bias_inter_candidate_spread_mean_{ch}"] = float(np.mean(spreads))
                seed_states = [r["substrate_state_per_channel"].get(ch, "state_unknown") for r in results]
                counts = {}
                for s in seed_states:
                    counts[s] = counts.get(s, 0) + 1
                summary[f"substrate_state_modal_{ch}"] = max(counts.items(), key=lambda kv: kv[1])[0]
            ic_stds = [r.get("inter_candidate_score_std_mean", 0.0) for r in results]
            summary["inter_candidate_score_std_mean"] = float(np.mean(ic_stds))
            all_warnings = []
            for r in results:
                all_warnings.extend(r.get("warnings", []))
            summary["all_warnings"] = sorted(set(all_warnings))
            cell_summaries[key] = summary

    # -- The headline comparison: fixed vs legacy, per arm --
    scorer_fix_deltas = {}
    for arm_name in ARM_NAMES:
        fixed_key = f"{arm_name}__fixed"
        legacy_key = f"{arm_name}__legacy"
        fixed_s, legacy_s = cell_summaries[fixed_key], cell_summaries[legacy_key]
        deltas = {"arm": arm_name}
        for c in SCORE_COMPONENTS:
            k = f"temporal_fraction_{c}"
            deltas[f"delta_{k}_fixed_minus_legacy"] = fixed_s[k] - legacy_s[k]
        deltas["fixed_temporal_fraction_f"] = fixed_s["temporal_fraction_f"]
        deltas["legacy_temporal_fraction_f"] = legacy_s["temporal_fraction_f"]
        state_flips = []
        for ch in PER_CHANNEL_BIAS_COMPONENTS:
            fk = f"substrate_state_modal_{ch}"
            if fixed_s[fk] != legacy_s[fk]:
                state_flips.append({"channel": ch, "fixed": fixed_s[fk], "legacy": legacy_s[fk]})
        deltas["substrate_state_flips_fixed_vs_legacy"] = state_flips
        scorer_fix_deltas[arm_name] = deltas

    all_cells_collected = all(
        r["n_fresh_ticks"] >= 10
        for results in all_results.values() for r in results
    )
    outcome = "PASS" if all_cells_collected else "FAIL"
    total_fresh = sum(r["n_fresh_ticks"] for results in all_results.values() for r in results)
    total_latched = sum(r["n_latched_ticks"] for results in all_results.values() for r in results)

    # Non-degeneracy self-report (validate_experiments.py: a diagnostic reporting
    # evidence_direction must not self-route PASS on a structurally pinned readout
    # -- e.g. every cell reading identical temporal_fraction[f] would make the
    # scorer-fix comparison meaningless even though "steps collected" is fine).
    f_frac_values = [
        r["temporal_fraction"].get("f", 0.0)
        for results in all_results.values() for r in results
    ]
    degeneracy = check_degeneracy({
        "temporal_fraction_f_across_cells": {"values": f_frac_values},
    })
    fixed_f_arm0 = scorer_fix_deltas["ARM_0_baseline"]["fixed_temporal_fraction_f"]
    legacy_f_arm0 = scorer_fix_deltas["ARM_0_baseline"]["legacy_temporal_fraction_f"]
    outcome_note = (
        f"decomp collected in all {n_cells} cells ({len(ARM_NAMES)} arms x "
        f"{len(SCORER_STATES)} scorer states x {len(SEEDS)} seeds); "
        f"{total_fresh} genuine fresh E3 selections recorded, {total_latched} "
        f"latched (non-fresh) ticks correctly skipped -- see CLAUDE.md "
        f"Sample-size integrity for why this distinction is load-bearing. "
        f"ARM_0_baseline temporal_fraction[f]: fixed={fixed_f_arm0:.4f} vs "
        f"legacy={legacy_f_arm0:.4f} (delta={fixed_f_arm0 - legacy_f_arm0:.4f}). "
        f"Historical V3-EXQ-571 reference figure (ARM_0, pre-fix substrate, "
        f"different revision, no substrate_hash for direct comparison): ~0.886. "
        f"See scorer_fix_deltas for the full per-arm, per-component comparison."
    )

    return {
        "outcome": outcome,
        "outcome_note": outcome_note,
        "cell_summaries": cell_summaries,
        "scorer_fix_deltas": scorer_fix_deltas,
        "per_seed_results": all_results,
        "cell_fingerprints_n": len(cell_fingerprints),
        "non_degenerate": degeneracy["non_degenerate"],
        "degeneracy_reason": degeneracy["degeneracy_reason"],
        "degenerate_metrics": degeneracy["degenerate_metrics"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V3-EXQ-924: E3 score variance decomp, post-scorer-fix remeasurement")
    parser.add_argument("--dry-run", action="store_true", help="Short run for smoke testing")
    args = parser.parse_args()

    t0 = time.perf_counter()
    result = run_experiment(dry_run=args.dry_run, started_at=t0)

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"v3_exq_924_e3_score_variance_decomp_scorer_fix_remeasure_{timestamp}_v3"

    out_dir = Path(__file__).parent.parent.parent / "REE_assembly" / "evidence" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": [],
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "evidence_direction": "diagnostic",
        "outcome": result["outcome"],
        "outcome_note": result["outcome_note"],
        "non_degenerate": result["non_degenerate"],
        "degeneracy_reason": result["degeneracy_reason"],
        "degenerate_metrics": result["degenerate_metrics"],
        "timestamp_utc": timestamp,
        "n_steps": N_STEPS,
        "arm_names": ARM_NAMES,
        "scorer_states": SCORER_STATES,
        "score_components": SCORE_COMPONENTS,
        "temporal_fraction_components": TEMPORAL_FRACTION_COMPONENTS,
        "per_channel_bias_components": PER_CHANNEL_BIAS_COMPONENTS,
        "cell_summaries": result["cell_summaries"],
        "scorer_fix_deltas": result["scorer_fix_deltas"],
        "per_seed_results": result["per_seed_results"],
        "supersedes": SUPERSEDES,
        "related_finding_docs": [
            "REE_assembly/evidence/planning/v3_exq_571_root_cause_2026-05-25.md",
        ],
        "related_commits": [
            "ree-v3 193bbec: SD-E3-SCORER-COMPLETION, gates untrained scorer heads out by default (2026-08-09)",
        ],
        "methodology_note": (
            "Reuses V3-EXQ-571's temporal_fraction formula verbatim for direct "
            "numeric comparability, and V3-EXQ-609's per-candidate spread state "
            "classification. Adds a scorer_state (fixed vs legacy) dimension not "
            "present in either predecessor, to isolate the SD-E3-SCORER-COMPLETION "
            "fix's effect on the SAME substrate revision rather than comparing "
            "against a stale historical manifest with no substrate_hash."
        ),
    }

    config_for_recording = {
        "seeds": SEEDS,
        "n_steps": N_STEPS,
        "grid_size": GRID_SIZE,
        "num_hazards": NUM_HAZARDS,
        "arm_names": ARM_NAMES,
        "scorer_states": SCORER_STATES,
        "f_weight": 1.0,
    }
    stamp_recording_core(
        manifest,
        config=config_for_recording,
        seeds=SEEDS,
        script_path=Path(__file__),
        started_at=t0,
        z_goal_stream_stats=_ZG.stats(),
    )

    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=bool(args.dry_run),
        config=manifest.get("config"),
        seeds=SEEDS,
        script_path=Path(__file__),
        json_default=str,
    )
    print(f"Manifest written: {out_path}")
    print(f"Result written to: {out_path}")

    if args.dry_run:
        print("DRY RUN complete.")

    _outcome_raw = str(result["outcome"]).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
        dry_run=bool(args.dry_run),
    )
