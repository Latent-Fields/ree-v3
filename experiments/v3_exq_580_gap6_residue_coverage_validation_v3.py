"""
V3-EXQ-580: GAP-6 residue_coverage_pct / harm_benefit_ratio metric substrate validation.

Validates the new read-only ResidueField.get_coverage_telemetry() added at
ree-v3/ree_core/residue/field.py for infant_substrate:GAP-6. The metric is
field-side (not env-info like GAP-5): residue is accumulated into a
ResidueField as the agent takes harm in CausalGridWorldV2, then the telemetry
is read at episode end.

ARM_0_grad_off: harm_gradient_enabled=False. Harm fires only on direct hazard
                contact, so few residue accumulate() calls -> low coverage.
ARM_1_grad_on:  harm_gradient_enabled=True (GAP-1, owner V3-EXQ-576, done).
                Graded harm fires every tick within the gradient band around a
                hazard, so many more accumulate() calls -> the harm-residue RBF
                field is populated over far more of its centers -> higher
                residue_coverage_pct. This is the DEV-NEED-004 residue-geography
                formation signal the metric must be able to discriminate.

Benefit terrain is disabled in both arms, so harm_benefit_ratio must return the
-1.0 "undefined" sentinel and benefit_total must be -1.0 (terrain off, distinct
from terrain-on-zero-benefit which would return benefit_total 0.0).

PASS = C0 AND C1 AND C2 AND C3 AND C4 across all seeds:
  C0  Both arms numerically well-formed: residue_coverage_pct in [0, 1], no
      NaN/inf, residue_coverage_threshold == 0.02 * RESIDUE_SCALE_FACTOR > 0,
      residue_active_centers <= residue_n_centers.
  C1  ARM_1 residue_coverage_pct strictly > 0 AND >= ARM_0 +
      THRESH_ARM1_MIN_MARGIN (per seed) -- the harm-gradient substrate must
      lay residue over strictly more of the field than binary contact.
  C2  harm_benefit_ratio == -1.0 AND benefit_total == -1.0 in BOTH arms
      (benefit terrain off sentinel).
  C3  Non-invasive contract (protects EXQ-575): get_statistics() still returns
      exactly {total_residue, num_harm_events, active_centers, mean_weight};
      two consecutive get_coverage_telemetry() calls return identical dicts;
      rbf_field.weights unchanged across the telemetry call.
  C4  ARM_0 lays at least some residue (harm contact happened) so the
      ARM_1 > ARM_0 comparison is meaningful, not 0-vs-0.

experiment_purpose: diagnostic (substrate readiness test, not a claim
hypothesis test). Unblocks DEV-NEED-004 (residue-geography blocking gate) and
DEV-NEED-008 per REE_assembly/evidence/planning/infant_substrate_plan.md
(GAP-6). claim_ids: [] (telemetry metric validation only; no governance
weighting; mirrors GAP-1 V3-EXQ-576 / GAP-5 V3-EXQ-579 precedent).
"""

import argparse
import json
import math
import os
import random
import sys
from datetime import datetime

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experiment_protocol import emit_outcome
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.residue.field import ResidueField
from ree_core.utils.config import ResidueConfig

EXPERIMENT_PURPOSE = "diagnostic"
QUEUE_ID = "V3-EXQ-580"

SEEDS = [0, 1, 2]
N_EPISODES = 40
N_STEPS = 120

# ARC-046 infant hazard-protection scale (developmental_curriculum.md).
RESIDUE_SCALE_FACTOR = 0.1
EXPECTED_THRESHOLD = 0.02 * RESIDUE_SCALE_FACTOR  # 0.002

# Wide RBF field so the every-tick-in-band ARM_1 signal does not collapse to
# the same saturated coverage as the sparse on-contact ARM_0 (the C1
# discriminator depends on ARM_1 not saturating to the ARM_0 value).
N_BASIS_FUNCTIONS = 512

# Pre-registered acceptance thresholds (constants, not derived post-hoc).
# C1 = user spec: ARM_1 residue_coverage_pct strictly > 0 AND strictly
# greater than ARM_0, with a small fixed margin so a noise-level tie does
# not pass. The structural every-tick-in-band vs on-contact gap clears this
# by a wide margin (dry-run: weakest-seed margin ~0.036).
THRESH_ARM1_MIN_MARGIN = 0.01
SENTINEL = -1.0                   # benefit-terrain-off / undefined ratio

CONDITIONS = [
    ("ARM_0_grad_off", False),
    ("ARM_1_grad_on", True),
]


def _z_world(env, world_dim):
    """Deterministic non-degenerate z_world stand-in from agent grid position.

    GAP-6 is a metric over the RBF field; add_residue() assigns centers by an
    internal cycling index regardless of this vector, so the value does not
    affect coverage -- it is derived from position only to avoid degenerate
    all-identical centers.
    """
    v = torch.zeros(1, world_dim)
    v[0, 0] = float(env.agent_x) / max(1, env.size)
    v[0, 1] = float(env.agent_y) / max(1, env.size)
    return v


def _finite(x):
    return isinstance(x, (int, float)) and math.isfinite(x)


def run_experiment(n_episodes, dry_run=False):
    results_by_seed = {}
    c0_by_seed = []
    c1_by_seed = []
    c2_by_seed = []
    c3_by_seed = []
    c4_by_seed = []

    for seed in SEEDS:
        results_by_seed[seed] = {}
        cov_by_arm = {}

        for cond_label, grad_on in CONDITIONS:
            print(f"Seed {seed} Condition {cond_label}", flush=True)

            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

            env = CausalGridWorldV2(
                size=12, seed=seed, num_hazards=2, num_resources=3,
                use_proxy_fields=False,
                harm_gradient_enabled=grad_on,
                harm_gradient_outer_radius=3.0,
                harm_gradient_scale=1.0,
            )
            rcfg = ResidueConfig(num_basis_functions=N_BASIS_FUNCTIONS)
            # benefit_terrain_enabled=False (default)
            field = ResidueField(rcfg)
            world_dim = rcfg.world_dim

            harm_ticks = 0
            for ep in range(n_episodes):
                env.reset()
                done = False
                for _ in range(N_STEPS):
                    if done:
                        break
                    action = np.random.randint(0, 5)
                    _, harm_signal, done, _info, _ = env.step(action)
                    harm_mag = max(0.0, -float(harm_signal))
                    if harm_mag > 0.0:
                        field.accumulate(
                            _z_world(env, world_dim),
                            harm_magnitude=harm_mag,
                            hypothesis_tag=False,
                        )
                        harm_ticks += 1

                print_interval = max(1, n_episodes // 5)
                if (ep + 1) % print_interval == 0:
                    print(
                        f"  [train] seed={seed} cond={cond_label} "
                        f"ep {ep + 1}/{n_episodes}",
                        flush=True,
                    )

            # --- non-invasive contract probe (C3) ---
            stats_keys = set(field.get_statistics().keys())
            w_before = field.rbf_field.weights.detach().clone()
            tel = field.get_coverage_telemetry(
                residue_scale_factor=RESIDUE_SCALE_FACTOR
            )
            tel2 = field.get_coverage_telemetry(
                residue_scale_factor=RESIDUE_SCALE_FACTOR
            )
            weights_unchanged = torch.equal(
                field.rbf_field.weights.detach(), w_before
            )
            stats_unchanged = stats_keys == {
                "total_residue",
                "num_harm_events",
                "active_centers",
                "mean_weight",
            }
            telemetry_idempotent = tel == tel2

            cov = tel["residue_coverage_pct"]
            cov_by_arm[cond_label] = cov

            # C0 well-formedness
            c0 = (
                _finite(cov) and 0.0 <= cov <= 1.0
                and _finite(tel["residue_coverage_threshold"])
                and abs(tel["residue_coverage_threshold"] - EXPECTED_THRESHOLD)
                < 1e-12
                and tel["residue_active_centers"] <= tel["residue_n_centers"]
                and _finite(tel["harm_total"])
            )
            # C2 benefit-terrain-off sentinel
            c2 = (
                tel["harm_benefit_ratio"] == SENTINEL
                and tel["benefit_total"] == SENTINEL
            )
            # C3 non-invasive
            c3 = (
                stats_unchanged
                and telemetry_idempotent
                and weights_unchanged
            )

            results_by_seed[seed][cond_label] = {
                "grad_on": grad_on,
                "harm_ticks": harm_ticks,
                "telemetry": tel,
                "c0_wellformed": bool(c0),
                "c2_sentinel": bool(c2),
                "c3_noninvasive": bool(c3),
                "stats_unchanged": bool(stats_unchanged),
                "telemetry_idempotent": bool(telemetry_idempotent),
                "weights_unchanged": bool(weights_unchanged),
            }

            passed_arm = c0 and c2 and c3
            print(f"verdict: {'PASS' if passed_arm else 'FAIL'}", flush=True)

        arm0 = results_by_seed[seed]["ARM_0_grad_off"]
        arm1 = results_by_seed[seed]["ARM_1_grad_on"]

        c0_seed = arm0["c0_wellformed"] and arm1["c0_wellformed"]
        c2_seed = arm0["c2_sentinel"] and arm1["c2_sentinel"]
        c3_seed = arm0["c3_noninvasive"] and arm1["c3_noninvasive"]
        c1_seed = (
            cov_by_arm["ARM_1_grad_on"] > 0.0
            and cov_by_arm["ARM_1_grad_on"]
            >= cov_by_arm["ARM_0_grad_off"] + THRESH_ARM1_MIN_MARGIN
        )
        c4_seed = arm0["harm_ticks"] > 0

        c0_by_seed.append(c0_seed)
        c1_by_seed.append(c1_seed)
        c2_by_seed.append(c2_seed)
        c3_by_seed.append(c3_seed)
        c4_by_seed.append(c4_seed)

        results_by_seed[seed]["seed_summary"] = {
            "arm0_coverage": cov_by_arm["ARM_0_grad_off"],
            "arm1_coverage": cov_by_arm["ARM_1_grad_on"],
            "c0": bool(c0_seed),
            "c1": bool(c1_seed),
            "c2": bool(c2_seed),
            "c3": bool(c3_seed),
            "c4": bool(c4_seed),
        }

    c0_pass = all(c0_by_seed) if c0_by_seed else False
    c1_pass = all(c1_by_seed) if c1_by_seed else False
    c2_pass = all(c2_by_seed) if c2_by_seed else False
    c3_pass = all(c3_by_seed) if c3_by_seed else False
    c4_pass = all(c4_by_seed) if c4_by_seed else False
    overall_pass = c0_pass and c1_pass and c2_pass and c3_pass and c4_pass

    return {
        "outcome": "PASS" if overall_pass else "FAIL",
        "c0_pass": c0_pass,
        "c1_pass": c1_pass,
        "c2_pass": c2_pass,
        "c3_pass": c3_pass,
        "c4_pass": c4_pass,
        "c0_by_seed": c0_by_seed,
        "c1_by_seed": c1_by_seed,
        "c2_by_seed": c2_by_seed,
        "c3_by_seed": c3_by_seed,
        "c4_by_seed": c4_by_seed,
        "results_by_seed": results_by_seed,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    dry_run = args.dry_run
    n_episodes = 5 if dry_run else N_EPISODES

    print(
        "V3-EXQ-580 GAP-6 residue_coverage_pct / harm_benefit_ratio validation",
        flush=True,
    )
    print(
        f"  dry_run={dry_run} n_episodes={n_episodes} seeds={SEEDS} "
        f"residue_scale_factor={RESIDUE_SCALE_FACTOR} "
        f"threshold={EXPECTED_THRESHOLD}",
        flush=True,
    )

    result = run_experiment(n_episodes=n_episodes, dry_run=dry_run)

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"v3_exq_580_gap6_residue_coverage_validation_{timestamp}_v3"

    manifest = {
        "run_id": run_id,
        "queue_id": QUEUE_ID,
        "experiment_type": "gap6_residue_coverage_validation",
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": [],
        "evidence_direction": "non_contributory",
        "timestamp_utc": timestamp,
        "outcome": result["outcome"],
        "dry_run": dry_run,
        "config": {
            "seeds": SEEDS,
            "n_episodes": n_episodes,
            "n_steps": N_STEPS,
            "residue_scale_factor": RESIDUE_SCALE_FACTOR,
            "expected_threshold": EXPECTED_THRESHOLD,
            "n_basis_functions": N_BASIS_FUNCTIONS,
            "thresh_arm1_min_margin": THRESH_ARM1_MIN_MARGIN,
        },
        "acceptance_checks": {
            "C0_both_arms_wellformed": result["c0_pass"],
            "C1_arm1_coverage_strictly_gt_arm0": result["c1_pass"],
            "C2_benefit_terrain_off_sentinel": result["c2_pass"],
            "C3_noninvasive_get_statistics_intact": result["c3_pass"],
            "C4_arm0_has_harm_residue": result["c4_pass"],
        },
        "c0_by_seed": result["c0_by_seed"],
        "c1_by_seed": result["c1_by_seed"],
        "c2_by_seed": result["c2_by_seed"],
        "c3_by_seed": result["c3_by_seed"],
        "c4_by_seed": result["c4_by_seed"],
        "results_by_seed": result["results_by_seed"],
    }

    out_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "REE_assembly", "evidence", "experiments"
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{run_id}.json")

    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Manifest written: {out_path}", flush=True)
    print(f"Outcome: {result['outcome']}", flush=True)
    print(f"  C0 (both arms well-formed):              {result['c0_pass']}", flush=True)
    print(f"  C1 (ARM_1 coverage strictly > ARM_0):    {result['c1_pass']}", flush=True)
    print(f"  C2 (benefit-terrain-off sentinel):       {result['c2_pass']}", flush=True)
    print(f"  C3 (non-invasive; get_statistics intact):{result['c3_pass']}", flush=True)
    print(f"  C4 (ARM_0 has harm residue):             {result['c4_pass']}", flush=True)

    _outcome_raw = str(result["outcome"]).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
    )
