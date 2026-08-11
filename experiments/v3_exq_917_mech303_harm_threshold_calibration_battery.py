"""V3-EXQ-917: MECH-303 contextual_safety_harm_threshold reachability/discrimination
calibration battery.

SLEEP DRIVER: n/a (no sleep loop used)

PURPOSE (diagnostic, config-default calibration -- NOT a MECH-303 validity test).
Follow-on to the 2026-08-11 finding documented in
REE_assembly/evidence/planning/mech303_contextual_safety_threshold_reachability.md:
REEConfig.contextual_safety_harm_threshold defaults to 0.05, but agent.py sense()'s live
gate (new_latent.z_harm_a.norm() < contextual_safety_harm_threshold -> accumulate_safety)
is unreachable at that default in every prior experiment that exercised it (V3-EXQ-520
had to override to 999; V3-EXQ-764 measured the real z_harm_a norm baseline at ~0.547
safe / ~0.542 unsafe, both ~11x above 0.05 and barely different from each other -- an
SD-011 affective-harm-encoder limitation). This script runs the discriminating probe that
finding routed to /queue-experiment: does ANY fixed threshold value exist that is both
(a) reachable (fires on a meaningful fraction of ticks) and (b) discriminates safe from
hazardous contexts with a real margin, across a wider battery than the two points 764
measured -- or is the tension structural?

MECHANISTIC REFRAME FOUND DURING DESIGN (Step 2.5a empirical probe, this session).
SD-022 (`ree_core/environment/causal_grid_world.py` ~3816-3838) re-sources harm_obs_a
from the agent's own BODY-DAMAGE state (`limb_damage`) whenever `limb_damage_enabled=True`
-- which is what every production driver that has exercised this gate (764, 520, 916) sets.
Under that sourcing, z_harm_a tracks accumulated bodily harm, not hazard proximity --
its dependence on `num_hazards` is only INDIRECT, mediated by whether the agent's random
walk actually collided with a hazard. A one-tick probe under this sourcing (5 conditions,
1 seed, 150 ticks) reproduced 764's cited numbers almost exactly (h0 mean 0.547) and showed
a NON-monotonic, weakly-discriminating relationship with num_hazards (0.547 / 0.524 / 0.440
/ 0.380 / 0.429 for num_hazards in 0/1/2/4/8) -- consistent with the existing finding.
The SAME probe with `limb_damage_enabled=False` (the pre-SD-022 legacy path: `harm_obs_a`
is a slow EMA of the hazard/resource PROXIMITY FIELD at the agent's location, i.e. a direct
proximity signal) showed a much more monotonic, better-discriminating pattern (h0 0.559 /
h1 0.347 / h2 0.560 / h4 0.732 / h8 0.716 -- clearly higher at h4/h8 than at h0). So this
battery crosses num_hazards with SOURCING_MODE ("damage_sourced" = production default,
"proximity_ema_sourced" = legacy direct-proximity path) to test whether the tension is an
intrinsic SD-011 encoder limitation (both modes fail to discriminate) or specific to the
SD-022 damage-sourcing choice (only damage_sourced fails).

DESIGN. For each of 2 sourcing modes x 5 num_hazards levels (10 arms), N_SEEDS seeds each
random-walk the agent through EPISODE_TICKS ticks (agent.sense() ticks the full internal
pipeline every step per the 764/520 precedent's `_act` helper, but the ENVIRONMENT is
stepped with a random action -- decouples "internal computation" from "which way the walk
goes", matching precedent and avoiding a learned-policy confound). z_harm_a.norm() is
recorded every tick (live agent path, real z_harm_a -- NOT V3-EXQ-760's ground-truth
hazard_field_view proxy). Per-cell raw norms are retained (generous recording).

AGGREGATION. For each sourcing mode independently and each threshold in SWEEP_THRESHOLDS:
  reachability(tau)  = mean per-seed fire-rate (frac of ticks with norm < tau) in the
                        SAFE_GROUP (num_hazards in {0,1})
  AUC(tau)           = rank-biserial AUC of per-seed fire-rate as a safe(SAFE_GROUP)-vs-
                        unsafe(UNSAFE_GROUP={4,8}) classifier (num_hazards=2 excluded as
                        an ambiguous middle condition, reported separately as context)
C1 (load-bearing, damage_sourced -- the production default's own question): does any tau
  clear reachability(tau) >= REACH_FLOOR AND AUC(tau) >= AUC_BAR? If so, recommend that tau
  as the new default and route to /implement-substrate.
C2 (load-bearing, proximity_ema_sourced -- the mechanistic-reframe question): same test
  under the legacy sourcing. Distinguishes "SD-011 encoder cannot discriminate at all" from
  "SD-022 damage-sourcing is why the production default cannot discriminate".
Overall PASS iff C1 or C2 holds (found something actionable); FAIL iff neither does (the
tension is structural under both sourcings actually exercised here -- a genuine SD-011
falsifier data point, routed to governance/claim-synthesis for the dual-nociceptive-split
work rather than /implement-substrate, since no threshold recalibration alone would fix a
result that is null under both sourcing modes).

READINESS (non-vacuity, self-route substrate_not_ready_requeue -- NEVER a false structural
verdict): the positive control is that z_harm_a itself is reachable and non-degenerate at
all (finite, > MIN_APPARATUS_NORM on a large fraction of ticks, across the WHOLE battery).
This is a distinct failure mode from "does not discriminate" -- it would mean the gate is
broken/wired-off, not that the encoder is merely uninformative.

claim_ids=[SD-011] (the affective-harm encoder's discrimination properties are what this
battery characterises). MECH-303 itself is NOT tagged: its own claim and V3-EXQ-760's
representation-level PASS are unaffected by this run -- this is a downstream default-config
usability question for OTHER drivers, per the mech303_contextual_safety_threshold_
reachability.md finding this experiment follows on from.
experiment_purpose=diagnostic (calibrating a default config value, not testing MECH-303's
mechanism validity).
"""
import argparse
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402

from _lib.arm_fingerprint import arm_cell  # noqa: E402
from _lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402
from pack_writer import write_flat_manifest  # noqa: E402
from experiment_protocol import emit_outcome  # noqa: E402

_ZG = ZGoalStreamAccumulator()

EXPERIMENT_PURPOSE = "diagnostic"
EXPERIMENT_TYPE = "v3_exq_917_mech303_harm_threshold_calibration_battery"
ARCH_EPOCH = "ree_hybrid_guardrails_v1"
CLAIM_IDS = ["SD-011"]

DEVICE = torch.device("cpu")

# The z_harm_a_apparatus_reachable precondition is reachable by construction, not a
# narrower-than-state predicate: z_harm_a is an unconditional every-tick MLP forward pass
# over harm_obs_a (SD-011), never gated by training/convergence, so a finite nonzero norm
# is the ONLY way sense() can return in practice -- confirmed empirically in this session's
# Step 2.5a probe and in the --dry-run smoke (n_valid=20/20 on every one of 20 cells, both
# sourcing modes, all 5 density levels). A below-floor reading would mean z_harm_a itself is
# wired off/None (a genuine substrate defect), which self-routes substrate_not_ready_requeue
# and never a substrate verdict -- so no instrument-gap-as-verdict mislabel is possible.
ANCHOR_REACHABILITY_EXEMPT = (
    "z_harm_a_apparatus_reachable is reachable by construction: z_harm_a is an unconditional "
    "every-tick forward pass (no training/convergence gate), so a finite nonzero norm is the "
    "only way sense() returns in practice (confirmed: 20/20 valid ticks on every dry-run cell, "
    "both sourcing modes x all 5 density levels); below-floor self-routes substrate_not_ready_"
    "requeue, never a substrate verdict"
)

# --- pre-registered constants -------------------------------------------------
N_SEEDS = 10                 # per (mode, num_hazards) arm
SEED_BASE = 3000
EPISODE_TICKS = 150          # progress denominator M (per seed x arm)
GRID_SIZE = 10
NUM_RESOURCES = 3

DENSITY_LEVELS = [0, 1, 2, 4, 8]
SAFE_GROUP = [0, 1]
UNSAFE_GROUP = [4, 8]
MIDDLE_GROUP = [2]           # reported as context only, excluded from the binary AUC

# SD-022 damage-sourcing (production default: every prior driver that exercised this gate
# -- 764, 520, 916 -- sets limb_damage_enabled=True) vs the pre-SD-022 legacy direct-
# proximity EMA path (limb_damage_enabled=False). See module docstring "MECHANISTIC REFRAME".
SOURCING_MODES = {
    "damage_sourced": dict(limb_damage_enabled=True, heal_rate=0.4, harm_obs_a_dim=7),
    "proximity_ema_sourced": dict(limb_damage_enabled=False, harm_obs_a_dim=50),
}
PRIMARY_MODE = "damage_sourced"       # the production default -- C1's own question
SECONDARY_MODE = "proximity_ema_sourced"  # C2's mechanistic-reframe question

SWEEP_THRESHOLDS = [
    0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4,
    0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8,
]
REACH_FLOOR = 0.15    # SAFE-group mean fire-rate must clear this to count as "reachable"
AUC_BAR = 0.75         # pre-registered discrimination bar (below V3-EXQ-760's 0.884 --
                       # that was a ground-truth-gated representation-level ceiling; this
                       # is a workable-default bar for the noisier live agent-path signal)
MIN_APPARATUS_NORM = 1e-6   # readiness floor: z_harm_a norm must be measurably nonzero
APPARATUS_VALID_FRAC_FLOOR = 0.99  # readiness floor: fraction of ticks with a valid reading

_SUBSTRATE_COMMON = dict(use_harm_stream=True, harm_obs_dim=51, use_affective_harm_stream=True)


def _env_kwargs(mode, num_hazards):
    mode_kw = dict(SOURCING_MODES[mode])
    mode_kw.pop("harm_obs_a_dim", None)
    kw = dict(size=GRID_SIZE, num_hazards=num_hazards, num_resources=NUM_RESOURCES,
              use_proxy_fields=True)
    kw.update(mode_kw)
    return kw


def _cfg_kwargs(env, mode):
    kw = dict(body_obs_dim=env.body_obs_dim, world_obs_dim=env.world_obs_dim, action_dim=5)
    kw.update(_SUBSTRATE_COMMON)
    kw["harm_obs_a_dim"] = SOURCING_MODES[mode]["harm_obs_a_dim"]
    return kw


def _sense(agent, obs):
    return agent.sense(
        obs["body_state"], obs["world_state"],
        obs_harm=obs.get("harm_obs"), obs_harm_a=obs.get("harm_obs_a"),
    )


def _act(agent, latent):
    """Ticks the full internal pipeline (mirrors the V3-EXQ-764/520 precedent's `_act`
    helper) so the live path is exercised exactly as it is in a real driver. The returned
    action is discarded by the caller -- the environment is stepped randomly (see module
    docstring) so this only advances agent-internal state (clock, E1/E3 ticking)."""
    ticks = agent.clock.advance()
    world_dim = agent.config.latent.world_dim
    e1_prior = agent._e1_tick(latent) if ticks.get("e1_tick") else torch.zeros(1, world_dim, device=DEVICE)
    candidates = agent.generate_trajectories(latent, e1_prior, ticks)
    action = agent.select_action(candidates, ticks)
    return int(action.argmax(dim=-1).item())


def _run_cell(mode, num_hazards, seed, episode_ticks):
    env = CausalGridWorldV2(seed=SEED_BASE + seed, **_env_kwargs(mode, num_hazards))
    cfg = REEConfig.from_dims(**_cfg_kwargs(env, mode))
    agent = REEAgent(cfg)
    agent.reset()
    _, obs = env.reset()
    norms = []
    n_valid = 0
    for step in range(episode_ticks):
        latent = _sense(agent, obs)
        if latent.z_harm_a is not None:
            n = float(latent.z_harm_a.detach().norm().item())
            norms.append(n)
            if np.isfinite(n) and n > MIN_APPARATUS_NORM:
                n_valid += 1
        _ = _act(agent, latent)
        _flat, _harm, done, _info, obs = env.step(random.randint(0, 4))
        if done:
            _, obs = env.reset()
        if (step + 1) % 50 == 0:
            print(f"  [train] arm={mode}_h{num_hazards} seed={seed} "
                  f"ep {step + 1}/{episode_ticks}", flush=True)
    _ZG.observe(agent)   # AFTER stepping -- no goal use case here, so this is a genuine
                         # goal-OFF measured-zero (never leave it as "unmeasured")
    n_total = len(norms)
    return {
        "mode": mode,
        "num_hazards": num_hazards,
        "seed": seed,
        "n_ticks": n_total,
        "n_valid_apparatus": n_valid,
        "mean_norm": float(np.mean(norms)) if norms else 0.0,
        "std_norm": float(np.std(norms)) if norms else 0.0,
        "min_norm": float(np.min(norms)) if norms else 0.0,
        "max_norm": float(np.max(norms)) if norms else 0.0,
        "norms": norms,
        "fire_rate_by_threshold": {
            f"{tau:g}": (float(np.mean([1.0 if n < tau else 0.0 for n in norms])) if norms else 0.0)
            for tau in SWEEP_THRESHOLDS
        },
    }


def _run_arm(mode, num_hazards, seeds, episode_ticks):
    rows = []
    for seed in seeds:
        print(f"Seed {seed} Condition {mode}_h{num_hazards}", flush=True)
        with arm_cell(
            seed,
            config_slice={
                "mode": mode, "num_hazards": num_hazards,
                "env_kwargs": _env_kwargs(mode, num_hazards),
                "substrate_common": _SUBSTRATE_COMMON,
                "harm_obs_a_dim": SOURCING_MODES[mode]["harm_obs_a_dim"],
                "episode_ticks": episode_ticks,
            },
            script_path=Path(__file__),
            config_slice_declared=True,
        ) as cell:
            row = _run_cell(mode, num_hazards, seed, episode_ticks)
            cell.stamp(row)
        print(f"  [{mode}_h{num_hazards} seed={seed}] mean_norm={row['mean_norm']:.4f} "
              f"n_valid={row['n_valid_apparatus']}/{row['n_ticks']}", flush=True)
        print("verdict: PASS", flush=True)   # per-cell run completion marker
        rows.append(row)
    return rows


def _auc(pos_scores, neg_scores):
    """Rank-biserial AUC (Mann-Whitney U / (n_pos*n_neg)) -- P(pos_score > neg_score),
    ties counted as 0.5. Higher score = more "SAFE"-predicting (fires more often)."""
    n_pos, n_neg = len(pos_scores), len(neg_scores)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    wins = 0.0
    for p in pos_scores:
        for q in neg_scores:
            if p > q:
                wins += 1.0
            elif p == q:
                wins += 0.5
    return wins / (n_pos * n_neg)


def _mode_analysis(rows_by_hazard):
    """rows_by_hazard: {num_hazards: [row, ...]}. Returns per-threshold reachability + AUC
    (SAFE_GROUP vs UNSAFE_GROUP) plus the recommended tau (max AUC among qualifying)."""
    per_threshold = []
    best = None
    for tau in SWEEP_THRESHOLDS:
        key = f"{tau:g}"
        safe_scores = [r["fire_rate_by_threshold"][key] for h in SAFE_GROUP for r in rows_by_hazard[h]]
        unsafe_scores = [r["fire_rate_by_threshold"][key] for h in UNSAFE_GROUP for r in rows_by_hazard[h]]
        reach = float(np.mean(safe_scores)) if safe_scores else 0.0
        auc = _auc(safe_scores, unsafe_scores)
        qualifies = bool(reach >= REACH_FLOOR and auc >= AUC_BAR)
        entry = {
            "threshold": tau, "reachability_safe_group": reach, "auc_safe_vs_unsafe": auc,
            "qualifies": qualifies,
            "safe_scores_nondegenerate": bool(float(np.std(safe_scores)) > 1e-9) if len(safe_scores) > 1 else False,
            "unsafe_scores_nondegenerate": bool(float(np.std(unsafe_scores)) > 1e-9) if len(unsafe_scores) > 1 else False,
        }
        per_threshold.append(entry)
        if qualifies and (best is None or auc > best["auc_safe_vs_unsafe"]):
            best = entry
    middle_mean = float(np.mean([r["mean_norm"] for h in MIDDLE_GROUP for r in rows_by_hazard[h]]))
    density_means = {h: float(np.mean([r["mean_norm"] for r in rows_by_hazard[h]])) for h in DENSITY_LEVELS}
    return {
        "per_threshold": per_threshold,
        "recommended": best,
        "density_condition_means": density_means,
        "middle_condition_mean_norm": middle_mean,
        "density_spread_std": float(np.std(list(density_means.values()))),
    }


def run_experiment(seeds, episode_ticks):
    per_cell = []
    analysis_by_mode = {}
    for mode in SOURCING_MODES:
        rows_by_hazard = {}
        for num_hazards in DENSITY_LEVELS:
            rows = _run_arm(mode, num_hazards, seeds, episode_ticks)
            rows_by_hazard[num_hazards] = rows
            per_cell.extend(rows)
        analysis_by_mode[mode] = _mode_analysis(rows_by_hazard)

    # Readiness: apparatus reachable across the WHOLE battery (all modes, all arms).
    total_ticks = sum(r["n_ticks"] for r in per_cell)
    total_valid = sum(r["n_valid_apparatus"] for r in per_cell)
    apparatus_valid_frac = float(total_valid) / float(total_ticks) if total_ticks else 0.0
    readiness_met = apparatus_valid_frac >= APPARATUS_VALID_FRAC_FLOOR

    c1 = analysis_by_mode[PRIMARY_MODE]["recommended"] is not None
    c2 = analysis_by_mode[SECONDARY_MODE]["recommended"] is not None

    criteria_non_degenerate = {
        f"density_battery_nondegenerate_{mode}": bool(analysis_by_mode[mode]["density_spread_std"] > 1e-6)
        for mode in SOURCING_MODES
    }
    # AUC computation non-degeneracy: at least one threshold's safe/unsafe score sets
    # actually varied (else every AUC is a tie-only 0.5, not a real measurement).
    for mode in SOURCING_MODES:
        any_nondegenerate = any(
            e["safe_scores_nondegenerate"] or e["unsafe_scores_nondegenerate"]
            for e in analysis_by_mode[mode]["per_threshold"]
        )
        criteria_non_degenerate[f"auc_computation_nondegenerate_{mode}"] = bool(any_nondegenerate)

    if not readiness_met:
        outcome = "FAIL"
        evidence_direction = "non_contributory"
        label = "substrate_not_ready_requeue"
        overall_pass = False
    else:
        overall_pass = bool(c1 or c2)
        outcome = "PASS" if overall_pass else "FAIL"
        if c1:
            label = "mech303_damage_sourced_threshold_found"
        elif c2:
            label = "mech303_tension_sourcing_mode_dependent"
        else:
            label = "mech303_tension_structural_sd011_gated"
        evidence_direction = "supports" if overall_pass else "weakens"

    non_degenerate = readiness_met and all(criteria_non_degenerate.values())

    manifest = {
        "run_id": None,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": ARCH_EPOCH,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "outcome": outcome,
        "evidence_direction": evidence_direction,
        "non_degenerate": non_degenerate,
        "degeneracy_reason": None if non_degenerate else (
            "substrate_not_ready" if not readiness_met else "zero_battery_or_auc_spread"),
        "timestamp_utc": datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
        "arm_results": per_cell,
        "analysis_by_sourcing_mode": analysis_by_mode,
        "readiness": {
            "apparatus_valid_frac": apparatus_valid_frac,
            "apparatus_valid_frac_floor": APPARATUS_VALID_FRAC_FLOOR,
            "readiness_met": bool(readiness_met),
        },
        "interpretation": {
            "label": label,
            "preconditions": [
                {"name": "z_harm_a_apparatus_reachable",
                 "description": "fraction of ALL battery ticks where z_harm_a.norm() is finite and > floor -- the gate's own measurement, not its discrimination",
                 "measured": apparatus_valid_frac, "threshold": APPARATUS_VALID_FRAC_FLOOR,
                 "direction": "lower",
                 "control": "every arm in the battery (both sourcing modes, all 5 density levels)",
                 "met": bool(readiness_met)},
            ],
            "criteria_non_degenerate": criteria_non_degenerate,
        },
        "criteria": [
            {"name": "C1_damage_sourced_threshold_found (production default)",
             "load_bearing": True, "passed": bool(c1)},
            {"name": "C2_proximity_ema_sourced_threshold_found (mechanistic reframe)",
             "load_bearing": True, "passed": bool(c2)},
        ],
        "combination_rule": "overall PASS iff C1 OR C2 (found ANY workable config); "
                             "both fail -> structural SD-011 falsifier, neither routes to "
                             "/implement-substrate as a scalar default change.",
        "ethics_preflight": {
            "involves_negative_valence": False,
            "involves_suffering_like_state": False,
            "involves_self_model": False,
            "involves_inescapability_or_helplessness": False,
            "involves_offline_replay_over_harm": False,
            "involves_social_mind_or_language": False,
            "involves_human_data_or_clinical_context": False,
            "decision": "allow",
        },
        "notes": (
            "MECH-303 contextual_safety_harm_threshold calibration battery, follow-on to "
            "the 2026-08-11 reachability finding (mech303_contextual_safety_threshold_"
            "reachability.md). GOV-REUSE-1 check: the decisive readout (per-threshold "
            "reachability + AUC across a 5-level num_hazards battery under 2 sourcing "
            "modes) is not recorded anywhere -- V3-EXQ-764 measured only 2 ad-hoc points "
            "(num_hazards 0 and 8, single inline probe, no manifest -- 764 itself is HELD, "
            "never queued) and V3-EXQ-520 only forced accumulation via a threshold=999 "
            "positive control. Not recoverable -> ran fresh. Crosses num_hazards x "
            "SOURCING_MODE (damage_sourced = production default via SD-022 body-damage "
            "re-sourcing; proximity_ema_sourced = pre-SD-022 legacy direct-proximity EMA) "
            "-- a mechanistic reframe found during this session's Step 2.5a empirical "
            "probe (see module docstring), not anticipated by the original finding doc. "
            "C1/C2 are independent load-bearing criteria (see combination_rule); this is "
            "an OR-gated PASS, not a plain AND of all criteria."
        ),
    }
    return manifest, overall_pass


def build_and_run():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    args = ap.parse_args()

    t0 = time.perf_counter()
    episode_ticks = 20 if args.dry_run else EPISODE_TICKS
    seeds = list(range(2 if args.dry_run else args.seeds))

    manifest, _overall_pass = run_experiment(seeds, episode_ticks)

    ts = manifest["timestamp_utc"]
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"
    manifest["run_id"] = run_id

    full_config = {
        "sourcing_modes": SOURCING_MODES,
        "density_levels": DENSITY_LEVELS,
        "safe_group": SAFE_GROUP,
        "unsafe_group": UNSAFE_GROUP,
        "middle_group": MIDDLE_GROUP,
        "sweep_thresholds": SWEEP_THRESHOLDS,
        "reach_floor": REACH_FLOOR,
        "auc_bar": AUC_BAR,
        "grid_size": GRID_SIZE,
        "num_resources": NUM_RESOURCES,
        "substrate_common": _SUBSTRATE_COMMON,
        "episode_ticks": episode_ticks,
    }

    out_path = write_flat_manifest(
        manifest, None, dry_run=args.dry_run,
        config=full_config, seeds=seeds, script_path=Path(__file__), started_at=t0,
        z_goal_stream_stats=_ZG.stats(),
    )
    print(f"outcome={manifest['outcome']} evidence_direction={manifest['evidence_direction']} "
          f"label={manifest['interpretation']['label']}", flush=True)
    print(f"manifest={out_path}", flush=True)
    return manifest, out_path, run_id, args.dry_run


if __name__ == "__main__":
    _manifest, _out_path, _run_id, _dry = build_and_run()
    emit_outcome(
        outcome=_manifest["outcome"] if _manifest["outcome"] in ("PASS", "FAIL") else "FAIL",
        manifest_path=_out_path,
        run_id=_run_id,
        dry_run=_dry,
    )
