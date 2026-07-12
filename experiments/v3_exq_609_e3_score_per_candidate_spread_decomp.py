"""V3-EXQ-609: E3 score per-candidate spread decomposition diagnostic.

DIFFERENT SCIENTIFIC QUESTION FROM V3-EXQ-571 (per CLAUDE.md EXQ Versioning):
this is a diagnostic methodology change that records DIFFERENT measurements,
so it gets a new EXQ number rather than an a/b/c iteration.

Per REE_assembly/evidence/planning/v3_exq_571_root_cause_2026-05-25.md
(commit a79915151b on master), V3-EXQ-571's mean-collapsed
'bias_fraction_curiosity = 0.0' headline was right for the wrong reason:
the diagnostic at ree_core/agent.py recorded only the per-channel MEAN
across the [K] bias vector. For per-candidate signals (the only kind that
can shift argmin), mean-across-K is roughly stationary in time even when
per-candidate spread is non-zero within each step. The temporal-variance
test then read zero for both the right and the wrong reason simultaneously,
masking the actual question: "does this channel produce a per-candidate
vector with non-zero spread within each step?"

V3-EXQ-609 reads the sibling-session-extended per-channel std_across_K
and bias_range_mean keys (sibling: exq571-decomp-per-channel-spread-
20260526T070807Z, agent.py edit) and distinguishes three substrate
states per channel:

  state A "zero-emission":
      mean == 0 AND std_across_K == 0
      -> channel emits identically zero per-candidate vector;
         upstream input is empty / inactive / not arriving.
         Canonical example: MECH-314a curiosity in an untrained agent
         with empty ResidueField (residue.accumulate fires only on
         harm_occurred AND committed_trajectory; random-policy probes
         satisfy neither).

  state B "methodology-mismatch":
      mean != 0 AND std_across_K == 0
      -> channel emits a non-zero scalar broadcast across all K
         candidates; argmin-invariant by construction. EXQ-571's
         mean-collapsed metric would still register non-zero bias_fraction,
         but the bias cannot shift selection. Canonical example: the
         deleted MECH-111 broadcast novelty branch (e3_selector.py:606-613,
         removed 2026-05-25 by sibling session
         mech314a-propagation-root-cause-20260525T205723Z).

  state C "per-candidate signal":
      std_across_K > 0
      -> channel produces genuine per-candidate variation that CAN
         shift argmin. This is the only state under which a non-zero
         mean is a behaviourally-meaningful contribution.

The state C/B distinction is what EXQ-571 could not detect. Currently
(2026-05-25 substrate state) every bias channel reading first-step
z_world summaries fires in state A or B: MECH-314a is structurally zero
in untrained runs (state A); manually seeding ResidueField with active
centers produces saturating mean -1.0 but std_across_K = 0.0 across
K=32 (state B), because E2 world-forward compresses K diverse first-
action candidates to identical first-step z_world (same root cause as
the 2026-05-17 ARC-062 GAP-B autopsy; cand_world_pairwise_dist=0.0000
across K=32). See finding doc for the empirical drivers.

Arms (matching V3-EXQ-571 for direct comparability):
  ARM_0: Baseline (SP-CEM ON, diversity stack OFF)
  ARM_1: Diversity stack ON (MECH-313 noise_floor + MECH-314 curiosity +
         MECH-320 tonic_vigor)

Metrics added vs V3-EXQ-571:
  bias_inter_candidate_spread_mean[channel] -- mean over time of
      <channel>_std_across_K. THE primary new signal.
  bias_inter_candidate_range_mean[channel] -- mean over time of
      <channel>_bias_range_mean (max - min across K).
  per-channel substrate_state classification in {state_A_zero_emission,
      state_B_methodology_mismatch, state_C_per_candidate_signal,
      state_unknown_no_substrate} -- the latter fires when the sibling
      agent.py extension has not yet landed and the new keys are absent
      (current substrate state on master pre-sibling-commit).

Per-channel warnings emitted (only when sibling keys are present):
  state_B for any of {curiosity, tonic_vigor, dacc, lateral_pfc, ofc,
      gated_policy, mech295_liking}
      -> "<channel> emits non-zero mean ({mean:.3f}) with zero spread
         across K candidates -- methodology-mismatch (argmin-invariant
         broadcast). EXQ-571 mean-collapse metric would have mis-read
         this as 'channel propagating'; it is not."
  state_A for curiosity
      -> "curiosity emits zero per-candidate vector -- check upstream
         input (ResidueField active_mask.sum, MECH-314a
         _compute_novelty input pipeline; see
         v3_exq_571_root_cause_2026-05-25.md for the canonical
         E2-world-forward-compression case)."

Outcome:
  PASS if both arms collected decomp data and ARM_1 per-channel spread
  measurements landed in any state (including state_unknown_no_substrate,
  which is the expected pre-sibling state). This is a diagnostic-purpose
  experiment (excluded from governance confidence scoring per
  evidence_direction=diagnostic / non_contributory).

EXPERIMENT_PURPOSE = "diagnostic" -- excluded from governance.
"""

import sys
import json
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

EXPERIMENT_PURPOSE = "diagnostic"
EXPERIMENT_TYPE = "v3_exq_609_e3_score_per_candidate_spread_decomp"
SUPERSEDES = None  # NOT a supersede of EXQ-571: different scientific question.

SEEDS = [42, 123, 456]
N_STEPS = 200
GRID_SIZE = 8
NUM_HAZARDS = 1

# Spread thresholds for substrate-state classification.
SPREAD_ZERO_EPS = 1e-9       # std_across_K below this is treated as zero
MEAN_ZERO_EPS = 1e-9         # |mean| below this is treated as zero

ARM_NAMES = ["ARM_0_baseline", "ARM_1_diversity_stack"]

# Per-channel bias components recorded in agent._last_score_bias_decomp.
# (noise_floor_temp and total_bias are recorded as scalars not per-K vectors
# in the sibling agent.py extension, so they are excluded from per-candidate
# spread classification but still read for completeness.)
PER_CHANNEL_BIAS_COMPONENTS = ["dacc", "lateral_pfc", "ofc", "gated_policy",
                                "mech295_liking", "curiosity", "tonic_vigor",
                                "forced"]
SCALAR_BIAS_COMPONENTS = ["noise_floor_temp", "total_bias"]

# Score components from e3_selector instrumentation (unchanged from EXQ-571).
SCORE_COMPONENTS = ["f", "harm_weighted", "residue_weighted",
                    "benefit_weighted", "novelty_weighted", "goal_weighted"]


def make_env_and_agent(seed, arm_name):
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = CausalGridWorld(
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
        # SP-CEM for candidate diversity (matches EXQ-571 setup)
        use_support_preserving_cem=True,
        support_preserving_min_first_action_classes=2,
        support_preserving_stratified_elites=True,
        support_preserving_ao_std_floor=0.2,
    )
    config.latent.alpha_world = 0.9  # SD-008: high-fidelity z_world

    # ARM_1: diversity stack
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

    agent = REEAgent(config=config)
    agent.eval()

    # Enable E3 score decomposition instrumentation
    agent.e3.e3_score_decomp_enabled = True

    return agent, env, obs_dict


def _decomp_value(last_bias, key):
    """Read a value from agent._last_score_bias_decomp; default 0.0 on miss.

    Used for both legacy mean keys (e.g. 'curiosity') and sibling-extension
    spread keys (e.g. 'curiosity_std_across_K', 'curiosity_bias_range_mean').
    Returns 0.0 when the key is absent so the script works against the
    pre-sibling substrate state (every spread reads as 0; substrate_state
    classified as state_unknown_no_substrate).
    """
    if last_bias is None:
        return 0.0
    return float(last_bias.get(key, 0.0))


def run_arm(arm_name, seed, dry_run=False):
    agent, env, obs_dict = make_env_and_agent(seed, arm_name)
    n_steps = 5 if dry_run else N_STEPS

    per_step_decomp = []   # score components per step (mean across candidates)
    per_step_bias_mean = []      # per-channel mean across K
    per_step_bias_std = []       # per-channel std across K (sibling key)
    per_step_bias_range = []     # per-channel range across K (sibling key)
    per_step_bias_scalar = {k: [] for k in SCALAR_BIAS_COMPONENTS}
    per_step_scores = []   # K-vector of final scores

    keys_present_sample = None  # for diagnostic: which decomp keys exist on first tick

    for step_i in range(n_steps):
        obs_body = torch.tensor(obs_dict["body_state"]).float()
        obs_world = torch.tensor(obs_dict["world_state"]).float()

        with torch.no_grad():
            _action = agent.act_with_split_obs(obs_body, obs_world)

        last_decomp = agent.e3.last_score_decomp
        last_bias = agent._last_score_bias_decomp
        last_scores = agent.e3.last_scores

        if step_i == 0 and isinstance(last_bias, dict):
            keys_present_sample = sorted(last_bias.keys())

        if last_decomp and last_decomp.get("per_candidate"):
            cands = last_decomp["per_candidate"]
            step_comps = {}
            for comp in SCORE_COMPONENTS:
                vals = [c.get(comp, 0.0) for c in cands]
                step_comps[comp] = float(np.mean(vals))
            per_step_decomp.append(step_comps)

        if isinstance(last_bias, dict):
            # Per-channel: legacy mean (unsuffixed) + sibling extension keys.
            mean_step = {}
            std_step = {}
            range_step = {}
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

        if isinstance(_action, torch.Tensor):
            action_idx = int(_action.argmax().item())
        else:
            action_idx = int(_action)
        _, _, terminated, _, obs_dict = env.step(action_idx % env.action_dim)
        if terminated:
            agent.reset()
            _, obs_dict = env.reset()

    result = compute_spread_decomp(
        per_step_decomp,
        per_step_bias_mean,
        per_step_bias_std,
        per_step_bias_range,
        per_step_bias_scalar,
        per_step_scores,
        keys_present_sample,
        arm_name,
    )
    result["arm"] = arm_name
    result["seed"] = seed
    result["n_steps_collected"] = len(per_step_decomp)
    return result


def _classify_state(mean_val, spread_val, substrate_present):
    """Classify a per-channel (mean_over_time, spread_over_time) pair.

    Returns one of:
      state_unknown_no_substrate  -- sibling agent.py extension absent
      state_A_zero_emission        -- mean ~ 0 AND spread ~ 0
      state_B_methodology_mismatch -- mean != 0 AND spread ~ 0
      state_C_per_candidate_signal -- spread > 0
    """
    if not substrate_present:
        return "state_unknown_no_substrate"
    if spread_val > SPREAD_ZERO_EPS:
        return "state_C_per_candidate_signal"
    if abs(mean_val) <= MEAN_ZERO_EPS:
        return "state_A_zero_emission"
    return "state_B_methodology_mismatch"


def compute_spread_decomp(per_step_decomp,
                          per_step_bias_mean,
                          per_step_bias_std,
                          per_step_bias_range,
                          per_step_bias_scalar,
                          per_step_scores,
                          keys_present_sample,
                          arm_name):
    """Compute per-channel inter-candidate spread + substrate-state classification."""
    result = {
        "bias_inter_candidate_spread_mean": {},   # NEW: mean over time of std_across_K
        "bias_inter_candidate_range_mean": {},    # NEW: mean over time of bias_range_mean
        "bias_temporal_mean_mean": {},            # legacy mean-over-time of mean-over-K
        "substrate_state_per_channel": {},        # NEW: state A/B/C/unknown per channel
        "inter_candidate_score_std_mean": 0.0,
        "noise_floor_temp_mean": 0.0,
        "total_bias_mean_over_time": 0.0,
        "decomp_keys_present_sample": keys_present_sample or [],
        "sibling_substrate_present": False,
        "warnings": [],
        "scaling_recommendations": [],
    }

    if not per_step_bias_mean:
        result["warnings"].append("no bias decomp data collected")
        return result

    # Detect whether the sibling agent.py extension is live by checking for
    # at least one of the new spread keys on the sampled tick. Pre-sibling
    # substrate produces only the legacy unsuffixed mean keys.
    sibling_keys = [k for k in (keys_present_sample or [])
                    if k.endswith("_std_across_K") or k.endswith("_bias_range_mean")]
    sibling_substrate_present = len(sibling_keys) > 0
    result["sibling_substrate_present"] = sibling_substrate_present

    # Per-channel aggregation.
    for ch in PER_CHANNEL_BIAS_COMPONENTS:
        means = np.array([s.get(ch, 0.0) for s in per_step_bias_mean], dtype=np.float64)
        stds = np.array([s.get(ch, 0.0) for s in per_step_bias_std], dtype=np.float64)
        ranges = np.array([s.get(ch, 0.0) for s in per_step_bias_range], dtype=np.float64)

        mean_over_time_of_mean = float(np.mean(means))
        spread_over_time = float(np.mean(stds))
        range_over_time = float(np.mean(ranges))

        result["bias_temporal_mean_mean"][ch] = mean_over_time_of_mean
        result["bias_inter_candidate_spread_mean"][ch] = spread_over_time
        result["bias_inter_candidate_range_mean"][ch] = range_over_time

        state = _classify_state(mean_over_time_of_mean, spread_over_time,
                                sibling_substrate_present)
        result["substrate_state_per_channel"][ch] = state

        # Emit per-channel warnings ONLY when the sibling substrate is live.
        if sibling_substrate_present:
            if state == "state_B_methodology_mismatch":
                result["warnings"].append(
                    f"{ch} emits non-zero mean ({mean_over_time_of_mean:.3f}) "
                    f"with zero spread across K candidates -- methodology-mismatch "
                    f"(argmin-invariant broadcast). EXQ-571 mean-collapse metric "
                    f"would have mis-read this as 'channel propagating'; it is not."
                )
            elif state == "state_A_zero_emission" and ch == "curiosity":
                result["warnings"].append(
                    "curiosity emits zero per-candidate vector -- check upstream "
                    "input (ResidueField active_mask.sum, MECH-314a _compute_novelty "
                    "input pipeline; see v3_exq_571_root_cause_2026-05-25.md for the "
                    "canonical E2-world-forward-compression case)."
                )

    # Scalar bias channels.
    if per_step_bias_scalar.get("noise_floor_temp"):
        result["noise_floor_temp_mean"] = float(
            np.mean(per_step_bias_scalar["noise_floor_temp"])
        )
    if per_step_bias_scalar.get("total_bias"):
        result["total_bias_mean_over_time"] = float(
            np.mean(per_step_bias_scalar["total_bias"])
        )

    # Inter-candidate raw-score spread (independent of the bias channel readout).
    if per_step_scores:
        stds = [float(np.std(s)) for s in per_step_scores if len(s) > 1]
        if stds:
            result["inter_candidate_score_std_mean"] = float(np.mean(stds))

    # Scaling recommendations.
    recs = []
    if not sibling_substrate_present:
        recs.append(
            "sibling agent.py extension not detected -- decomp record carries "
            "only legacy mean keys; spread classification falls back to "
            "state_unknown_no_substrate for every channel. Re-run after the "
            "sibling commit (exq571-decomp-per-channel-spread-20260526T070807Z) "
            "lands to obtain state A/B/C classification."
        )
    else:
        # Count per-channel states for a one-line summary recommendation.
        states = list(result["substrate_state_per_channel"].values())
        n_C = sum(s == "state_C_per_candidate_signal" for s in states)
        n_B = sum(s == "state_B_methodology_mismatch" for s in states)
        n_A = sum(s == "state_A_zero_emission" for s in states)
        recs.append(
            f"per-channel substrate-state census across {len(PER_CHANNEL_BIAS_COMPONENTS)} "
            f"channels: state_C (per-candidate signal) = {n_C}, "
            f"state_B (methodology-mismatch broadcast) = {n_B}, "
            f"state_A (zero emission) = {n_A}. "
            f"Only state_C channels can shift argmin; state_B / state_A "
            f"channels cannot, regardless of their non-zero mean."
        )

    result["scaling_recommendations"] = recs
    return result


def run_experiment(dry_run=False):
    all_results = {}
    arm_summaries = {}

    for arm_name in ARM_NAMES:
        all_results[arm_name] = []
        print(f"\n== {arm_name} ==", flush=True)
        for seed in SEEDS:
            print(f"  seed={seed}", flush=True)
            r = run_arm(arm_name, seed, dry_run=dry_run)
            all_results[arm_name].append(r)

    # Summarise per arm (mean across seeds).
    for arm_name in ARM_NAMES:
        arm_results = all_results[arm_name]
        summary = {"arm": arm_name, "n_seeds": len(arm_results)}

        for ch in PER_CHANNEL_BIAS_COMPONENTS:
            spreads = [r["bias_inter_candidate_spread_mean"].get(ch, 0.0)
                       for r in arm_results]
            ranges = [r["bias_inter_candidate_range_mean"].get(ch, 0.0)
                      for r in arm_results]
            means = [r["bias_temporal_mean_mean"].get(ch, 0.0)
                     for r in arm_results]
            summary[f"bias_inter_candidate_spread_mean_{ch}"] = float(np.mean(spreads))
            summary[f"bias_inter_candidate_range_mean_{ch}"] = float(np.mean(ranges))
            summary[f"bias_temporal_mean_mean_{ch}"] = float(np.mean(means))

            # Modal state across seeds (most frequent classification).
            seed_states = [r["substrate_state_per_channel"].get(ch, "state_unknown_no_substrate")
                           for r in arm_results]
            state_counts = {}
            for s in seed_states:
                state_counts[s] = state_counts.get(s, 0) + 1
            modal_state = max(state_counts.items(), key=lambda kv: kv[1])[0]
            summary[f"substrate_state_modal_{ch}"] = modal_state

        ic_stds = [r.get("inter_candidate_score_std_mean", 0.0) for r in arm_results]
        summary["inter_candidate_score_std_mean"] = float(np.mean(ic_stds))

        sibs = [r.get("sibling_substrate_present", False) for r in arm_results]
        summary["sibling_substrate_present_all_seeds"] = all(sibs)
        summary["sibling_substrate_present_any_seed"] = any(sibs)

        all_warnings = []
        all_recs = []
        for r in arm_results:
            all_warnings.extend(r.get("warnings", []))
            all_recs.extend(r.get("scaling_recommendations", []))
        summary["all_warnings"] = sorted(set(all_warnings))
        summary["scaling_recommendations"] = sorted(set(all_recs))

        arm_summaries[arm_name] = summary

    # Outcome determination.
    arm0_steps = float(np.mean([r["n_steps_collected"]
                                for r in all_results["ARM_0_baseline"]]))
    arm1_steps = float(np.mean([r["n_steps_collected"]
                                for r in all_results["ARM_1_diversity_stack"]]))

    if arm0_steps < 1 or arm1_steps < 1:
        outcome = "FAIL"
        outcome_note = (
            f"decomp data collection failed: ARM_0 steps={arm0_steps:.0f}, "
            f"ARM_1 steps={arm1_steps:.0f}; e3_score_decomp_enabled may not be wired"
        )
    else:
        sibling_live = arm_summaries["ARM_1_diversity_stack"].get(
            "sibling_substrate_present_all_seeds", False
        )
        arm1_curi_spread = arm_summaries["ARM_1_diversity_stack"].get(
            "bias_inter_candidate_spread_mean_curiosity", 0.0
        )
        outcome = "PASS"
        outcome_note = (
            f"decomp data collected (ARM_0 {arm0_steps:.0f} steps, "
            f"ARM_1 {arm1_steps:.0f} steps); "
            f"sibling agent.py extension {'LIVE' if sibling_live else 'NOT DETECTED'}; "
            f"ARM_1 curiosity inter-candidate spread (mean over time of std_across_K) "
            f"= {arm1_curi_spread:.6f}. See arm_summaries for per-channel substrate-"
            f"state classification (state_A/B/C/unknown)."
        )

    return {
        "outcome": outcome,
        "outcome_note": outcome_note,
        "arm_summaries": arm_summaries,
        "per_seed_results": {
            arm: [
                {k: v for k, v in r.items()
                 if k not in ("bias_temporal_mean_mean",)}
                for r in results
            ]
            for arm, results in all_results.items()
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V3-EXQ-609: E3 score per-candidate spread decomp")
    parser.add_argument("--dry-run", action="store_true", help="Short run for smoke testing")
    args = parser.parse_args()

    result = run_experiment(dry_run=args.dry_run)

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"v3_exq_609_e3_score_per_candidate_spread_decomp_{timestamp}_v3"

    out_dir = (
        Path(__file__).parent.parent.parent
        / "REE_assembly"
        / "evidence"
        / "experiments"
    )
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
        "timestamp_utc": timestamp,
        "seeds": SEEDS,
        "n_steps": N_STEPS,
        "spread_zero_eps": SPREAD_ZERO_EPS,
        "mean_zero_eps": MEAN_ZERO_EPS,
        "arm_names": ARM_NAMES,
        "per_channel_bias_components": PER_CHANNEL_BIAS_COMPONENTS,
        "scalar_bias_components": SCALAR_BIAS_COMPONENTS,
        "score_components": SCORE_COMPONENTS,
        "arm_summaries": result["arm_summaries"],
        "per_seed_results": result["per_seed_results"],
        "supersedes": SUPERSEDES,
        "related_finding_doc": (
            "REE_assembly/evidence/planning/v3_exq_571_root_cause_2026-05-25.md"
        ),
        "methodology_note": (
            "V3-EXQ-609 records DIFFERENT measurements from V3-EXQ-571 "
            "(per-channel std_across_K + bias_range_mean instead of mean-only). "
            "Per CLAUDE.md EXQ Versioning, diagnostic methodology change asking a "
            "different scientific question gets a new EXQ number, not an a/b/c "
            "iteration of EXQ-571. NOT a supersede."
        ),
    }

    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=False,
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
    )
