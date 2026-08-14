"""V3-EXQ-930: SD-MECH303-THRESHOLD-SOURCING dedicated-signal validation.

SLEEP DRIVER: n/a (no sleep loop used)

PURPOSE (diagnostic -- substrate-readiness validation of a landed substrate fix, NOT a
MECH-303 mechanism test). Validates SD-MECH303-THRESHOLD-SOURCING (landed 2026-08-14,
ree-v3 b257e7ad14): the DEDICATED anticipatory hazard-proximity signal added for MECH-303's
contextual-safety gate, decoupled from SD-022's damage-sourced z_harm_a.

BACKGROUND. V3-EXQ-917 (harm-threshold calibration battery) found that under the production
scenario (limb_damage_enabled=True, SD-022 body-damage re-sourcing) MECH-303's live gate
signal z_harm_a.norm() cannot discriminate safe-vs-unsafe contexts at ANY of 18 swept
thresholds (AUC capped at 0.52, chance), because SD-022 deliberately DECOUPLES z_harm_a from
current spatial location. The dedicated proximity signal was built precisely so MECH-303's
gate has an anticipatory, spatially-contextual signal that discriminates EVEN when SD-022
body-damage sourcing is active.

DESIGN. Under the exact production scenario that defeats z_harm_a -- limb_damage_enabled=True
(z_harm_a is damage-sourced) AND safety_proximity_signal_enabled=True (the env emits the
dedicated obs_dict["safety_proximity_harm"] channel) -- run 5 num_hazards density levels x
N_SEEDS seeds. Each cell random-walks the agent through EPISODE_TICKS ticks (agent.sense()
ticks the full internal pipeline every step per the 917/764/520 precedent; the ENVIRONMENT is
stepped with a random action to decouple internal computation from walk direction). Per tick,
record BOTH candidate gate signals from the SAME live path:
  - dedicated_proximity  = obs_dict["safety_proximity_harm"] (the new signal MECH-303's gate
                           reads when contextual_safety_gate_source="proximity_signal")
  - damage_sourced_zharma = z_harm_a.norm()                  (the old signal, damage-sourced,
                           the control 917 already showed fails)
For each signal independently and each threshold in SWEEP_THRESHOLDS, compute:
  reachability(tau) = mean per-seed fire-rate in the SAFE_GROUP (frac of ticks where the
                      "harm absent" gate would fire = signal < tau)
  AUC(tau)          = rank-biserial AUC of per-seed fire-rate as a safe(SAFE_GROUP)-vs-
                      unsafe(UNSAFE_GROUP) classifier

CRITERIA.
  C1 (LOAD-BEARING) -- dedicated_proximity: does ANY tau clear reachability >= REACH_FLOOR
     AND AUC >= AUC_BAR? This is the acceptance gate for the landed substrate: the dedicated
     signal must find a reachable+discriminating threshold that the damage-sourced signal
     cannot. Expected strong PASS (safe(nh=0) ~0.0 vs unsafe(nh=8) ~0.87 by construction ->
     AUC ~0.95+, well above the AUC_BAR=0.75 pre-registered bar and the ~0.84 acceptance
     target from the failure record).
  C2 (REPORTED CONTEXT, NOT load-bearing) -- damage_sourced_zharma: does z_harm_a find a
     qualifying tau under the SAME limb_damage_enabled=True scenario? Expected NO (reproduces
     917's chance-level C1 finding), confirming the decoupling was necessary. A surprising C2
     PASS would be a context note, not a run failure -- the run validates the dedicated
     signal, not the absence of the old one.
overall PASS iff C1 (the dedicated signal discriminates). C2 is recorded for the reader.

READINESS (P0). The load-bearing C1 routes on a DISCRIMINATION statistic (safe-vs-unsafe AUC,
a cross-density range), so the readiness precondition asserts the SAME statistic on the
dedicated signal: its per-density means must SPREAD across the battery (density_spread_std >
DENSITY_SPREAD_FLOOR) -- i.e. the signal actually varies with hazard density. A flat signal
(spread below floor) means the dedicated channel is inert/misconfigured -> self-route
substrate_not_ready_requeue, NEVER a substrate verdict. This matches the routed statistic
(range/spread), not a magnitude proxy (the V3-EXQ-643 mismatch trap).

GOV-REUSE-1. The decisive readout -- per-threshold reachability + safe-vs-unsafe AUC for the
DEDICATED proximity signal (obs_dict["safety_proximity_harm"]) under limb_damage_enabled=True
-- is recorded in NO prior manifest: the dedicated signal did not exist before ree-v3
b257e7ad14 (2026-08-14), so no run on any prior substrate_hash carries it. V3-EXQ-917 measured
only z_harm_a.norm() (the control here), never the new channel. Not recoverable -> ran fresh.
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
EXPERIMENT_TYPE = "v3_exq_930_mech303_dedicated_proximity_signal_validation"
ARCH_EPOCH = "ree_hybrid_guardrails_v1"
CLAIM_IDS = ["MECH-303"]

DEVICE = torch.device("cpu")

# The apparatus-reachable precondition (both signals produce a finite reading every tick) is
# reachable by construction: z_harm_a is an unconditional every-tick MLP forward pass (SD-011,
# no training/convergence gate) and safety_proximity_harm is a deterministic env EMA emitted
# every step when safety_proximity_signal_enabled=True. A below-floor reading means a signal
# is wired off/None (a genuine substrate defect), which self-routes substrate_not_ready_requeue
# and never a substrate verdict -- so no instrument-gap-as-verdict mislabel is possible.
ANCHOR_REACHABILITY_EXEMPT = (
    "both gate signals are reachable by construction: z_harm_a is an unconditional every-tick "
    "forward pass and safety_proximity_harm is a deterministic every-tick env EMA, so a finite "
    "reading is the only way sense()/step() returns in practice; below-floor self-routes "
    "substrate_not_ready_requeue, never a substrate verdict"
)

# --- pre-registered constants -------------------------------------------------
N_SEEDS = 10                 # per num_hazards arm
SEED_BASE = 3000
EPISODE_TICKS = 150          # progress denominator M (per seed x arm)
GRID_SIZE = 10
NUM_RESOURCES = 3

DENSITY_LEVELS = [0, 1, 2, 4, 8]
SAFE_GROUP = [0, 1]
UNSAFE_GROUP = [4, 8]
MIDDLE_GROUP = [2]           # reported as context only, excluded from the binary AUC

# The two candidate gate signals recorded per tick from the SAME live path, under the
# production scenario (limb_damage_enabled=True) that defeats z_harm_a in V3-EXQ-917.
DEDICATED_SIGNAL = "dedicated_proximity"          # obs_dict["safety_proximity_harm"] -- C1
CONTROL_SIGNAL = "damage_sourced_zharma"          # z_harm_a.norm() -- C2 (reproduces 917)
SIGNALS = [DEDICATED_SIGNAL, CONTROL_SIGNAL]

SWEEP_THRESHOLDS = [
    0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4,
    0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8,
]
REACH_FLOOR = 0.15    # SAFE-group mean fire-rate must clear this to count as "reachable"
AUC_BAR = 0.75         # pre-registered discrimination bar (below V3-EXQ-760's 0.884 ground-
                       # truth ceiling; the ~0.84 acceptance target from the failure record is
                       # reported separately as meets_acceptance_target)
ACCEPTANCE_AUC_TARGET = 0.84  # the failure-record acceptance target (proximity_ema band 0.84-0.97)
MIN_APPARATUS_VALUE = 1e-9    # readiness floor: a signal reading is measurably present
APPARATUS_VALID_FRAC_FLOOR = 0.99   # fraction of ticks with a valid reading for BOTH signals
DENSITY_SPREAD_FLOOR = 1e-3   # dedicated signal per-density means must spread (routed statistic)

_SUBSTRATE_COMMON = dict(use_harm_stream=True, harm_obs_dim=51, use_affective_harm_stream=True)
# harm_obs_a is 7-dim under limb_damage_enabled=True (SD-022 body-damage vector).
HARM_OBS_A_DIM = 7


def _env_kwargs(num_hazards):
    return dict(
        size=GRID_SIZE, num_hazards=num_hazards, num_resources=NUM_RESOURCES,
        use_proxy_fields=True,
        limb_damage_enabled=True, heal_rate=0.4,            # production scenario (damage-sourced z_harm_a)
        safety_proximity_signal_enabled=True,               # emit the dedicated channel
    )


def _cfg_kwargs(env):
    kw = dict(body_obs_dim=env.body_obs_dim, world_obs_dim=env.world_obs_dim, action_dim=5)
    kw.update(_SUBSTRATE_COMMON)
    kw["harm_obs_a_dim"] = HARM_OBS_A_DIM
    return kw


def _sense(agent, obs):
    sp = obs.get("safety_proximity_harm")
    sp_val = float(sp.item()) if sp is not None else None
    return agent.sense(
        obs["body_state"], obs["world_state"],
        obs_harm=obs.get("harm_obs"), obs_harm_a=obs.get("harm_obs_a"),
        obs_safety_proximity=sp_val,
    )


def _act(agent, latent):
    """Ticks the full internal pipeline (mirrors the V3-EXQ-917/764/520 precedent's `_act`
    helper) so the live path is exercised exactly as it is in a real driver. The returned
    action is discarded by the caller -- the environment is stepped randomly (module docstring)
    so this only advances agent-internal state (clock, E1/E3 ticking)."""
    ticks = agent.clock.advance()
    world_dim = agent.config.latent.world_dim
    e1_prior = agent._e1_tick(latent) if ticks.get("e1_tick") else torch.zeros(1, world_dim, device=DEVICE)
    candidates = agent.generate_trajectories(latent, e1_prior, ticks)
    action = agent.select_action(candidates, ticks)
    return int(action.argmax(dim=-1).item())


def _run_cell(num_hazards, seed, episode_ticks):
    env = CausalGridWorldV2(seed=SEED_BASE + seed, **_env_kwargs(num_hazards))
    cfg = REEConfig.from_dims(**_cfg_kwargs(env))
    agent = REEAgent(cfg)
    agent.reset()
    _, obs = env.reset()
    vals = {sig: [] for sig in SIGNALS}
    n_valid = 0
    for step in range(episode_ticks):
        latent = _sense(agent, obs)
        zha = None
        if latent.z_harm_a is not None:
            zha = float(latent.z_harm_a.detach().norm().item())
        sp = obs.get("safety_proximity_harm")
        prox = float(sp.item()) if sp is not None else None
        valid = (
            zha is not None and np.isfinite(zha) and zha > MIN_APPARATUS_VALUE
            and prox is not None and np.isfinite(prox)
        )
        if valid:
            n_valid += 1
        if zha is not None and np.isfinite(zha):
            vals[CONTROL_SIGNAL].append(zha)
        if prox is not None and np.isfinite(prox):
            vals[DEDICATED_SIGNAL].append(prox)
        _ = _act(agent, latent)
        _flat, _harm, done, _info, obs = env.step(random.randint(0, 4))
        if done:
            _, obs = env.reset()
        if (step + 1) % 50 == 0:
            print(f"  [train] arm=h{num_hazards} seed={seed} ep {step + 1}/{episode_ticks}",
                  flush=True)
    _ZG.observe(agent)   # AFTER stepping -- no goal use case here, genuine goal-OFF measured-zero
    row = {
        "num_hazards": num_hazards,
        "seed": seed,
        "n_ticks": episode_ticks,
        "n_valid_apparatus": n_valid,
    }
    for sig in SIGNALS:
        s = vals[sig]
        row[f"mean_{sig}"] = float(np.mean(s)) if s else 0.0
        row[f"std_{sig}"] = float(np.std(s)) if s else 0.0
        row[f"max_{sig}"] = float(np.max(s)) if s else 0.0
        row[f"min_{sig}"] = float(np.min(s)) if s else 0.0
        row[f"fire_rate_by_threshold_{sig}"] = {
            f"{tau:g}": (float(np.mean([1.0 if v < tau else 0.0 for v in s])) if s else 0.0)
            for tau in SWEEP_THRESHOLDS
        }
    return row


def _run_arm(num_hazards, seeds, episode_ticks):
    rows = []
    for seed in seeds:
        print(f"Seed {seed} Condition h{num_hazards}", flush=True)
        with arm_cell(
            seed,
            config_slice={
                "num_hazards": num_hazards,
                "env_kwargs": _env_kwargs(num_hazards),
                "substrate_common": _SUBSTRATE_COMMON,
                "harm_obs_a_dim": HARM_OBS_A_DIM,
                "episode_ticks": episode_ticks,
            },
            script_path=Path(__file__),
            config_slice_declared=True,
        ) as cell:
            row = _run_cell(num_hazards, seed, episode_ticks)
            cell.stamp(row)
        print(f"  [h{num_hazards} seed={seed}] mean_dedicated={row['mean_' + DEDICATED_SIGNAL]:.4f} "
              f"mean_zharma={row['mean_' + CONTROL_SIGNAL]:.4f} "
              f"n_valid={row['n_valid_apparatus']}/{row['n_ticks']}", flush=True)
        print("verdict: PASS", flush=True)   # per-cell run completion marker
        rows.append(row)
    return rows


def _auc(pos_scores, neg_scores):
    """Rank-biserial AUC (Mann-Whitney U / (n_pos*n_neg)) -- P(pos_score > neg_score), ties
    0.5. Higher score = more 'SAFE'-predicting (the gate fires more often)."""
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


def _signal_analysis(rows_by_hazard, sig):
    """Per-threshold reachability + AUC (SAFE_GROUP vs UNSAFE_GROUP) for one signal, plus the
    recommended tau (max AUC among qualifying)."""
    fkey = f"fire_rate_by_threshold_{sig}"
    per_threshold = []
    best = None
    for tau in SWEEP_THRESHOLDS:
        key = f"{tau:g}"
        safe_scores = [r[fkey][key] for h in SAFE_GROUP for r in rows_by_hazard[h]]
        unsafe_scores = [r[fkey][key] for h in UNSAFE_GROUP for r in rows_by_hazard[h]]
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
    density_means = {h: float(np.mean([r[f"mean_{sig}"] for r in rows_by_hazard[h]])) for h in DENSITY_LEVELS}
    best_auc_any = max((e["auc_safe_vs_unsafe"] for e in per_threshold), default=0.5)
    return {
        "per_threshold": per_threshold,
        "recommended": best,
        "best_auc_any_threshold": best_auc_any,
        "density_condition_means": density_means,
        "middle_condition_mean": float(np.mean([r[f"mean_{sig}"] for h in MIDDLE_GROUP for r in rows_by_hazard[h]])),
        "density_spread_std": float(np.std(list(density_means.values()))),
    }


def run_experiment(seeds, episode_ticks):
    per_cell = []
    rows_by_hazard = {}
    for num_hazards in DENSITY_LEVELS:
        rows = _run_arm(num_hazards, seeds, episode_ticks)
        rows_by_hazard[num_hazards] = rows
        per_cell.extend(rows)

    analysis = {sig: _signal_analysis(rows_by_hazard, sig) for sig in SIGNALS}

    # Readiness: both signals produce a valid reading essentially every tick.
    total_ticks = sum(r["n_ticks"] for r in per_cell)
    total_valid = sum(r["n_valid_apparatus"] for r in per_cell)
    apparatus_valid_frac = float(total_valid) / float(total_ticks) if total_ticks else 0.0
    # The LOAD-BEARING readiness statistic: the dedicated signal varies with hazard density
    # (same range/spread statistic the AUC criterion routes on -- NOT a magnitude proxy).
    dedicated_spread = analysis[DEDICATED_SIGNAL]["density_spread_std"]
    spread_met = bool(dedicated_spread >= DENSITY_SPREAD_FLOOR)
    apparatus_met = bool(apparatus_valid_frac >= APPARATUS_VALID_FRAC_FLOOR)
    readiness_met = bool(apparatus_met and spread_met)

    c1 = analysis[DEDICATED_SIGNAL]["recommended"] is not None      # LOAD-BEARING
    c2 = analysis[CONTROL_SIGNAL]["recommended"] is not None        # reported control
    dedicated_best_auc = analysis[DEDICATED_SIGNAL]["best_auc_any_threshold"]
    meets_acceptance_target = bool(dedicated_best_auc >= ACCEPTANCE_AUC_TARGET)

    # Non-degeneracy: the AUC computation actually varied for the dedicated signal.
    dedicated_auc_nondegenerate = any(
        e["safe_scores_nondegenerate"] or e["unsafe_scores_nondegenerate"]
        for e in analysis[DEDICATED_SIGNAL]["per_threshold"]
    )
    criteria_non_degenerate = {
        "dedicated_density_battery_nondegenerate": bool(dedicated_spread > 1e-6),
        "dedicated_auc_computation_nondegenerate": bool(dedicated_auc_nondegenerate),
    }

    if not readiness_met:
        outcome = "FAIL"
        evidence_direction = "non_contributory"
        label = "substrate_not_ready_requeue"
        overall_pass = False
    else:
        overall_pass = bool(c1)
        outcome = "PASS" if overall_pass else "FAIL"
        if c1 and not c2:
            label = "mech303_dedicated_signal_discriminates_zharma_does_not"
        elif c1 and c2:
            label = "mech303_dedicated_signal_discriminates_zharma_also_discriminates"
        else:
            label = "mech303_dedicated_signal_fails_to_discriminate"
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
            "substrate_not_ready" if not readiness_met else "zero_spread_or_auc"),
        "timestamp_utc": datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
        "arm_results": per_cell,
        "analysis_by_signal": analysis,
        "dedicated_signal_best_auc": dedicated_best_auc,
        "acceptance_auc_target": ACCEPTANCE_AUC_TARGET,
        "meets_acceptance_target": meets_acceptance_target,
        "readiness": {
            "apparatus_valid_frac": apparatus_valid_frac,
            "apparatus_valid_frac_floor": APPARATUS_VALID_FRAC_FLOOR,
            "apparatus_met": apparatus_met,
            "dedicated_density_spread_std": dedicated_spread,
            "density_spread_floor": DENSITY_SPREAD_FLOOR,
            "spread_met": spread_met,
            "readiness_met": readiness_met,
        },
        "interpretation": {
            "label": label,
            "preconditions": [
                {"name": "both_signals_apparatus_reachable",
                 "description": "fraction of battery ticks where BOTH z_harm_a.norm() and safety_proximity_harm return a finite value -- the gate signals' own reachability, not their discrimination",
                 "measured": apparatus_valid_frac, "threshold": APPARATUS_VALID_FRAC_FLOOR,
                 "direction": "lower",
                 "control": "every arm in the battery (all 5 density levels)",
                 "met": apparatus_met},
                {"name": "dedicated_signal_varies_with_density",
                 "description": "std across the 5 per-density means of the DEDICATED signal -- the SAME cross-density range statistic the load-bearing AUC criterion routes on (not a magnitude proxy); a flat signal means the dedicated channel is inert/misconfigured",
                 "measured": dedicated_spread, "threshold": DENSITY_SPREAD_FLOOR,
                 "direction": "lower",
                 "control": "positive control = hazard density genuinely varies 0->8 across the battery",
                 "met": spread_met},
            ],
            "criteria_non_degenerate": criteria_non_degenerate,
        },
        "criteria": [
            {"name": "C1_dedicated_proximity_signal_discriminates (reachable+AUC>=%.2f)" % AUC_BAR,
             "load_bearing": True, "passed": bool(c1)},
            {"name": "C2_damage_sourced_zharma_discriminates (control -- expected FAIL, reproduces 917)",
             "load_bearing": False, "passed": bool(c2)},
        ],
        "combination_rule": (
            "overall PASS iff C1 (the DEDICATED proximity signal finds a reachable+discriminating "
            "threshold under limb_damage_enabled=True). C2 (does the damage-sourced z_harm_a "
            "discriminate under the same scenario) is a REPORTED control, NOT load-bearing: it is "
            "expected to FAIL (reproducing V3-EXQ-917's chance-level C1) and its failure confirms "
            "the dedicated signal was necessary; a surprising C2 PASS is a context note, not a "
            "run failure."
        ),
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
        "sleep_driver_pattern": "n/a",
        "notes": (
            "Validation of SD-MECH303-THRESHOLD-SOURCING (landed 2026-08-14, ree-v3 b257e7ad14). "
            "Measures BOTH candidate MECH-303 gate signals per tick from the SAME live path under "
            "the production scenario (limb_damage_enabled=True) that V3-EXQ-917 showed defeats "
            "z_harm_a: the DEDICATED proximity signal obs_dict['safety_proximity_harm'] (C1, "
            "load-bearing) vs the damage-sourced z_harm_a.norm() (C2, control). Acceptance: the "
            "dedicated signal clears a reachable+discriminating threshold (AUC>=%.2f pre-registered "
            "bar; ~%.2f acceptance target from the failure record) that the damage-sourced signal "
            "cannot -- validating that the decoupling gives MECH-303's gate a working signal even "
            "when SD-022 body-damage sourcing is active. GOV-REUSE-1: the dedicated signal did not "
            "exist before b257e7ad14, so no prior manifest on any substrate_hash carries this "
            "readout -- not recoverable, ran fresh. C1/C2 combination is NOT a plain AND (see "
            "combination_rule): C1 alone gates the PASS."
        ) % (AUC_BAR, ACCEPTANCE_AUC_TARGET),
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
        "density_levels": DENSITY_LEVELS,
        "safe_group": SAFE_GROUP,
        "unsafe_group": UNSAFE_GROUP,
        "middle_group": MIDDLE_GROUP,
        "sweep_thresholds": SWEEP_THRESHOLDS,
        "reach_floor": REACH_FLOOR,
        "auc_bar": AUC_BAR,
        "acceptance_auc_target": ACCEPTANCE_AUC_TARGET,
        "density_spread_floor": DENSITY_SPREAD_FLOOR,
        "grid_size": GRID_SIZE,
        "num_resources": NUM_RESOURCES,
        "substrate_common": _SUBSTRATE_COMMON,
        "harm_obs_a_dim": HARM_OBS_A_DIM,
        "env_scenario": "limb_damage_enabled=True + safety_proximity_signal_enabled=True",
        "episode_ticks": episode_ticks,
    }

    out_path = write_flat_manifest(
        manifest, None, dry_run=args.dry_run,
        config=full_config, seeds=seeds, script_path=Path(__file__), started_at=t0,
        z_goal_stream_stats=_ZG.stats(),
    )
    print(f"outcome={manifest['outcome']} evidence_direction={manifest['evidence_direction']} "
          f"label={manifest['interpretation']['label']} "
          f"dedicated_best_auc={manifest['dedicated_signal_best_auc']:.4f}", flush=True)
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
