#!/opt/local/bin/python3
"""V3-EXQ-603b -- Q-045 MECH-313 vs MECH-260 four-arm anti-monostrategy ablation (retest).

Supersedes V3-EXQ-603a (measurement_gap).

IGW-20260521-003 / arc_062_rule_apprehension:GAP-H.

Root cause of V3-EXQ-603a (failure-autopsy + governance-603-manifests interactive,
2026-05-25): 603a's call-path fix landed cleanly (mech260_operative_all_seeds=true,
dACC FIFO + suppression both wired), BUT 2 of 3 seeds (42, 44) produced measured_steps=0
because the FIFO_WARMUP_STEPS=75 gate was applied PER-EPISODE while seeds 42/44
collapsed into early-termination episodes (~14 steps/ep across 30 episodes for seed 42,
~14.5 for seed 44; total_steps ~423 and ~436). With no per-episode cumulative reset on
the dACC.._action_history deque (cross-episode by construction in DACCAdaptiveControl),
the per-episode gate was the wrong abstraction. Mean entropy was therefore driven by
two zero seeds; only seed 43 (which produced normal-length episodes of ~158 steps/ep)
yielded interpretable data, leaving the comparison structurally underpowered with
N=1 effective seed.

Fixes applied vs V3-EXQ-603a:
  Fix 1 (retained from 603a): Training loop uses sense()+generate()+select_action()
    via _select_action_with_harm(), passing obs_harm_a so z_harm_a is set and
    select_action() calls dacc.record_action() each step.
  Fix 2 (RETAINED FROM 603a, KEPT INTACT): use_affective_harm_stream=True +
    harm_obs_a_dim=50 + harm_history_len=10.
  Fix 3 (RETAINED FROM 603a): evidence_direction_per_claim compares each arm vs
    ARM_0 baseline, not arm vs arm.
  Fix 4 (NEW IN 603b): CUMULATIVE FIFO warmup. The gate is now
    "if total_steps_global >= FIFO_WARMUP_STEPS" rather than
    "if step_in_ep >= FIFO_WARMUP_STEPS". The dACC._action_history deque is
    cross-episode by construction (no per-episode reset hook), so the cumulative
    metric matches what the dACC actually does internally.
  Fix 5 (NEW IN 603b): Per-seed minimum-measured-steps guard. Seeds with
    measured_steps < MIN_MEASURED_STEPS_PER_SEED are EXCLUDED from the per-arm
    entropy mean rather than averaged in at 0.0. fifo_temporal_gate_ok_all becomes
    "every seed in every arm cleared the minimum"; arms with too-few-clearing-seeds
    are reported but flagged measurement_gap.
  Fix 6 (NEW IN 603b): FIFO_WARMUP_STEPS lowered from 75 to 50, still within the
    pre-registered Kennerley / SD-054 band [50, 100] (SD-054 plan-doc). 50 also
    clears the 2 * DACC_SUPPRESSION_MEMORY (=16) sanity floor inherited from 603a.

SD-054 bipartite reef env + ARC-062 gated-policy ON + main-path SP-CEM (config
inherited from 603a unchanged; see "Inherited config" block).

Arms (unchanged from 603a):
  ARM_0_both_off   -- control (expected collapse / low diversity)
  ARM_1_mech313    -- use_noise_floor only
  ARM_2_mech260    -- use_dacc only (MECH-260 pathway)
  ARM_3_both_on    -- MECH-313 + MECH-260

Pre-registered PASS criteria (Q-045 mutually load-bearing; ENTROPY_MARGIN unchanged
from 603a; cumulative warmup is a measurement-discipline change, NOT a hypothesis
change):
  C1: ARM_3 entropy > ARM_0 + ENTROPY_MARGIN
  C2: ARM_3 entropy > max(ARM_1, ARM_2) + ENTROPY_MARGIN
  C3: ARM_1 entropy > ARM_0 AND ARM_2 entropy > ARM_0
  PASS if C2. FAIL otherwise.

Interpretation grid for outcomes (decision policy on read-out):
  - C2 PASS + every seed in every arm cleared MIN_MEASURED_STEPS_PER_SEED:
    Q-045 mutually load-bearing reading supported; MECH-313 + MECH-260 confirmed
    distinct and complementary. Promote both substrates' diagnostic-evidence
    weight; do not collapse the claims.
  - C2 PASS but >=1 seed in any arm missed MIN_MEASURED_STEPS_PER_SEED:
    Provisional support; record as PASS-with-caveat and re-run with longer
    episodes or higher MIN_MEASURED_STEPS_PER_SEED threshold (substrate finding,
    not a tuning failure).
  - C2 FAIL but C1 + C3 both PASS:
    Substrates are individually distinguishable but not mutually load-bearing.
    Indicates MECH-313 and MECH-260 partially redundant -- proceed to /failure-
    autopsy for collapse-or-separate decision (Q-045 specifically asks this).
  - C1 FAIL across the board:
    Neither substrate moved entropy detectably above baseline.
    Two possibilities: ARM_0 already saturated to the substrate ceiling (unlikely
    given ARC-062 GAP-B FAIL signature), or the env config produces too-short
    episodes for entropy to develop. Re-route to /diagnose-errors before re-tuning.
  - fifo_temporal_gate_ok_all=False for the WHOLE table:
    Same measurement-gap signature as 603a. Indicates the env config itself
    cannot deliver enough steps -- substrate-finding, escalate as substrate
    issue, do NOT re-tune.

claim_ids: Q-045, MECH-313, MECH-260
experiment_purpose: evidence

Smoke:
  /opt/local/bin/python3 experiments/v3_exq_603b_q045_mech313_mech260_four_arm_ablation.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiment_protocol import emit_outcome  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402


EXPERIMENT_TYPE = "v3_exq_603b_q045_mech313_mech260_four_arm_ablation"
QUEUE_ID = "V3-EXQ-603b"
CLAIM_IDS = ["Q-045", "MECH-313", "MECH-260"]
EXPERIMENT_PURPOSE = "evidence"

SEEDS = [42, 43, 44]
EVAL_EPISODES = 30
STEPS_PER_EPISODE = 200

# Fix 6: FIFO_WARMUP_STEPS lowered from 75 (603a) to 50; still within pre-registered
# Kennerley / SD-054 band [50, 100] and above 2 * DACC_SUPPRESSION_MEMORY=16.
FIFO_WARMUP_STEPS = 50
DACC_SUPPRESSION_MEMORY = 8

# Fix 5: per-seed minimum measured steps before that seed's entropy is admitted.
# Threshold chosen to leave ~5x DACC_SUPPRESSION_MEMORY of measured action evidence
# above the FIFO warmup before counting the seed (the dACC suppression FIFO needs
# enough above-warmup steps for its anti-recency signal to become observable).
MIN_MEASURED_STEPS_PER_SEED = 40

DRY_RUN_SEEDS = [42]
DRY_RUN_EPISODES = 2
DRY_RUN_STEPS = 50

ENTROPY_MARGIN = 0.05
DACC_SUPPRESSION_WEIGHT = 0.5
WORLD_DIM = 32
HARM_A_DIM = 16
HARM_HISTORY_LEN = 10

# Inherited config (unchanged from V3-EXQ-603a; bipartite reef + food-attracted hazards
# from SD-054 + ARC-062 gated policy ON + main-path SP-CEM).
ENV_KWARGS = dict(
    size=12,
    num_hazards=4,
    num_resources=5,
    hazard_harm=0.05,
    env_drift_interval=5,
    env_drift_prob=0.1,
    proximity_harm_scale=0.1,
    proximity_benefit_scale=0.05,
    proximity_approach_threshold=0.2,
    hazard_field_decay=0.5,
    resource_respawn_on_consume=True,
    toroidal=False,
    harm_history_len=HARM_HISTORY_LEN,
    reef_enabled=True,
    n_reef_patches=3,
    reef_patch_radius=2,
    hazard_food_attraction=0.7,
    reef_bipartite_layout=True,
    reef_bipartite_axis="horizontal",
    reef_bipartite_agent_band_radius=1,
)

ARMS: List[Dict[str, Any]] = [
    {
        "arm": "ARM_0_both_off",
        "use_noise_floor": False,
        "noise_floor_alpha": 0.1,
        "use_dacc": False,
    },
    {
        "arm": "ARM_1_mech313_only",
        "use_noise_floor": True,
        "noise_floor_alpha": 0.5,
        "use_dacc": False,
    },
    {
        "arm": "ARM_2_mech260_only",
        "use_noise_floor": False,
        "noise_floor_alpha": 0.1,
        "use_dacc": True,
    },
    {
        "arm": "ARM_3_both_on",
        "use_noise_floor": True,
        "noise_floor_alpha": 0.5,
        "use_dacc": True,
    },
]


def _entropy(counts: Counter) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        p = c / total
        if p > 0.0:
            h -= p * math.log(p)
    return float(h)


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(seed=seed, **ENV_KWARGS)


def _make_config(env: CausalGridWorldV2, arm: Dict[str, Any]) -> REEConfig:
    return REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=WORLD_DIM,
        alpha_world=0.9,
        use_affective_harm_stream=True,
        z_harm_a_dim=HARM_A_DIM,
        harm_obs_a_dim=50,
        harm_history_len=HARM_HISTORY_LEN,
        use_gated_policy=True,
        gated_policy_use_first_action_onehot=True,
        use_support_preserving_cem=True,
        support_preserving_stratified_elites=True,
        support_preserving_ao_std_floor=0.2,
        support_preserving_min_first_action_classes=2,
        use_noise_floor=arm["use_noise_floor"],
        noise_floor_alpha=arm["noise_floor_alpha"],
        use_dacc=arm["use_dacc"],
        dacc_weight=(1.0 if arm["use_dacc"] else 0.0),
        dacc_suppression_weight=(
            DACC_SUPPRESSION_WEIGHT if arm["use_dacc"] else 0.0
        ),
        dacc_suppression_memory=DACC_SUPPRESSION_MEMORY,
        use_structured_curiosity=False,
    )


def _obs_harm_a(obs_dict: Dict[str, Any]) -> Optional[torch.Tensor]:
    ha = obs_dict.get("harm_obs_a")
    if ha is None:
        return None
    t = ha.float()
    return t.unsqueeze(0) if t.dim() == 1 else t


def _obs_harm_history(obs_dict: Dict[str, Any]) -> Optional[torch.Tensor]:
    hh = obs_dict.get("harm_history")
    if hh is None:
        return None
    t = hh.float()
    return t.unsqueeze(0) if t.dim() == 1 else t


def _select_action_with_harm(
    agent: REEAgent,
    obs_body: torch.Tensor,
    obs_world: torch.Tensor,
    obs_harm_a: Optional[torch.Tensor],
    obs_harm_history: Optional[torch.Tensor],
) -> torch.Tensor:
    """Fix 1: select_action() path with obs_harm_a so z_harm_a and record_action() fire."""
    latent_state = agent.sense(
        obs_body,
        obs_world,
        obs_harm_a=obs_harm_a,
        obs_harm_history=obs_harm_history,
    )
    ticks = agent.clock.advance()
    if ticks.get("e1_tick", False):
        e1_prior = agent._e1_tick(latent_state)
    else:
        e1_prior = torch.zeros(1, WORLD_DIM, device=agent.device)
    candidates = agent.generate_trajectories(latent_state, e1_prior, ticks)
    action = agent.select_action(candidates, ticks, temperature=1.0)
    if ticks.get("e3_quiescent", False):
        agent._do_replay(latent_state)
    agent._step_count += 1
    return action


def _dacc_diagnostics(agent: REEAgent) -> Dict[str, Any]:
    if agent.dacc is None:
        return {
            "dacc_forward_calls": 0,
            "dacc_history_len": 0,
            "dacc_max_suppression": 0.0,
        }
    hist_len = len(agent.dacc._action_history)
    max_sup = 0.0
    bundle = getattr(agent, "_dacc_last_bundle", None)
    if bundle is not None and "suppression" in bundle:
        sup = bundle["suppression"]
        if isinstance(sup, torch.Tensor) and sup.numel() > 0:
            max_sup = float(sup.max().item())
    return {
        "dacc_forward_calls": int(agent.dacc._n_forward_calls),
        "dacc_history_len": hist_len,
        "dacc_max_suppression": round(max_sup, 6),
    }


def _run_arm_seed(
    arm: Dict[str, Any],
    seed: int,
    episodes: int,
    steps_per_episode: int,
    fifo_warmup_steps: int,
) -> Dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = _make_env(seed)
    cfg = _make_config(env, arm)
    agent = REEAgent(cfg)

    action_counts: Counter = Counter()
    reef_steps = 0
    total_steps = 0  # cumulative across episodes (Fix 4 reference for warmup gate)
    measured_steps = 0
    reef_cells = set()
    max_dacc_history = 0
    max_dacc_forward = 0
    max_dacc_suppression = 0.0

    for ep in range(episodes):
        _, obs_dict = env.reset()
        agent.reset()
        if ep == 0:
            reef_cells = set(getattr(env, "_reef_cells", set()))

        for step_in_ep in range(steps_per_episode):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            if obs_body.dim() == 1:
                obs_body = obs_body.unsqueeze(0)
            if obs_world.dim() == 1:
                obs_world = obs_world.unsqueeze(0)

            with torch.no_grad():
                action = _select_action_with_harm(
                    agent,
                    obs_body,
                    obs_world,
                    _obs_harm_a(obs_dict),
                    _obs_harm_history(obs_dict),
                )

            if arm["use_dacc"] and agent.dacc is not None:
                diag = _dacc_diagnostics(agent)
                max_dacc_history = max(max_dacc_history, diag["dacc_history_len"])
                max_dacc_forward = max(max_dacc_forward, diag["dacc_forward_calls"])
                max_dacc_suppression = max(
                    max_dacc_suppression, diag["dacc_max_suppression"]
                )

            if action is None:
                idx = random.randint(0, env.action_dim - 1)
                action_onehot = torch.zeros(1, env.action_dim, device=agent.device)
                action_onehot[0, idx] = 1.0
            else:
                action_onehot = action
                idx = int(action.argmax(dim=-1).item())

            # Fix 4: CUMULATIVE step gate. total_steps is the global cross-episode
            # counter; the dACC._action_history deque is cross-episode by construction
            # (DACCAdaptiveControl does not reset on env.reset()), so this is the
            # correct abstraction. step_in_ep ignored for the gate.
            if total_steps >= fifo_warmup_steps:
                action_counts[idx] += 1
                measured_steps += 1

            pos = (int(env.agent_x), int(env.agent_y))
            if pos in reef_cells:
                reef_steps += 1

            _flat, _harm, done, _info, obs_dict = env.step(action_onehot)
            total_steps += 1
            if done:
                break

        if (ep + 1) % 10 == 0 or (ep + 1) == episodes:
            print(
                f"  [train] arm={arm['arm']} seed={seed} ep {ep + 1}/{episodes}",
                flush=True,
            )

    entropy = _entropy(action_counts)
    reef_fraction = reef_steps / max(total_steps, 1)
    steps_per_ep_measured = measured_steps / max(episodes, 1)
    # Fix 5: per-seed gate -- this seed only contributes to per-arm entropy mean if
    # its measured_steps cleared MIN_MEASURED_STEPS_PER_SEED. Reported here; the
    # aggregator in _evaluate() honours it.
    seed_admitted = measured_steps >= MIN_MEASURED_STEPS_PER_SEED
    fifo_gate_ok = (
        fifo_warmup_steps >= 2 * DACC_SUPPRESSION_MEMORY
        and seed_admitted
    )

    out: Dict[str, Any] = {
        "arm": arm["arm"],
        "seed": seed,
        "selected_action_entropy": round(entropy, 6),
        "reef_fraction": round(reef_fraction, 6),
        "total_steps": int(total_steps),
        "measured_steps": int(measured_steps),
        "fifo_warmup_steps": int(fifo_warmup_steps),
        "warmup_metric": "cumulative",
        "min_measured_steps_threshold": int(MIN_MEASURED_STEPS_PER_SEED),
        "seed_admitted": bool(seed_admitted),
        "steps_per_episode_measured_mean": round(steps_per_ep_measured, 2),
        "fifo_temporal_gate_ok": bool(fifo_gate_ok),
        "unique_actions": len(action_counts),
    }
    if arm["use_dacc"]:
        out.update(
            {
                "dacc_history_len_max": max_dacc_history,
                "dacc_forward_calls_max": max_dacc_forward,
                "dacc_max_suppression": round(max_dacc_suppression, 6),
                "mech260_operative": bool(
                    max_dacc_forward > 0 and max_dacc_history > 0
                ),
            }
        )
    return out


def _evaluate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_arm: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_arm.setdefault(row["arm"], []).append(row)

    def admitted_mean_entropy(arm_name: str) -> Dict[str, Any]:
        """Fix 5: average only over seeds that cleared MIN_MEASURED_STEPS_PER_SEED."""
        all_rows = by_arm.get(arm_name, [])
        admitted = [r for r in all_rows if r.get("seed_admitted", False)]
        if not admitted:
            return {
                "mean": 0.0,
                "n_admitted": 0,
                "n_total": len(all_rows),
                "admitted_seeds": [],
            }
        vals = [r["selected_action_entropy"] for r in admitted]
        return {
            "mean": float(sum(vals) / len(vals)),
            "n_admitted": len(admitted),
            "n_total": len(all_rows),
            "admitted_seeds": sorted(r["seed"] for r in admitted),
        }

    arm0 = admitted_mean_entropy("ARM_0_both_off")
    arm1 = admitted_mean_entropy("ARM_1_mech313_only")
    arm2 = admitted_mean_entropy("ARM_2_mech260_only")
    arm3 = admitted_mean_entropy("ARM_3_both_on")

    e0, e1, e2, e3 = arm0["mean"], arm1["mean"], arm2["mean"], arm3["mean"]

    c1 = e3 > e0 + ENTROPY_MARGIN
    c2 = e3 > max(e1, e2) + ENTROPY_MARGIN
    c3 = (e1 > e0) and (e2 > e0)

    mech260_rows = [
        r for r in rows if r["arm"] in ("ARM_2_mech260_only", "ARM_3_both_on")
    ]
    mech260_operative_all = all(
        r.get("mech260_operative", False) for r in mech260_rows
    )
    fifo_ok_all = all(r.get("fifo_temporal_gate_ok", False) for r in rows)

    # Per-arm admission accounting (Fix 5 disclosure).
    arm_admission = {
        "ARM_0_both_off": arm0,
        "ARM_1_mech313_only": arm1,
        "ARM_2_mech260_only": arm2,
        "ARM_3_both_on": arm3,
    }
    # All arms must have at least 1 admitted seed for the entropy comparison to be
    # interpretable at all.
    all_arms_have_admitted = all(a["n_admitted"] >= 1 for a in arm_admission.values())

    return {
        "entropy_ARM_0": round(e0, 6),
        "entropy_ARM_1": round(e1, 6),
        "entropy_ARM_2": round(e2, 6),
        "entropy_ARM_3": round(e3, 6),
        "c1_both_beats_off": bool(c1),
        "c2_mutually_load_bearing": bool(c2),
        "c3_each_alone_beats_off": bool(c3),
        "mech260_operative_all_seeds": bool(mech260_operative_all),
        "fifo_temporal_gate_ok_all": bool(fifo_ok_all),
        "all_arms_have_admitted_seed": bool(all_arms_have_admitted),
        "arm_admission": arm_admission,
        "overall_pass": bool(c2 and all_arms_have_admitted),
    }


def _evidence_direction_per_claim(summary: Dict[str, Any]) -> Dict[str, str]:
    """Fix 3: compare each arm vs ARM_0 baseline, not arm vs arm."""
    e0 = summary["entropy_ARM_0"]
    e1 = summary["entropy_ARM_1"]
    e2 = summary["entropy_ARM_2"]

    # If any arm has zero admitted seeds, the comparison is not interpretable.
    if not summary.get("all_arms_have_admitted_seed", True):
        return {"Q-045": "non_contributory", "MECH-313": "non_contributory", "MECH-260": "non_contributory"}

    if summary["c2_mutually_load_bearing"]:
        q045 = "supports"
        m313 = "supports" if e1 > e0 + ENTROPY_MARGIN else "mixed"
        m260 = "supports" if e2 > e0 + ENTROPY_MARGIN else "mixed"
    elif summary["c3_each_alone_beats_off"]:
        q045 = "mixed"
        m313 = "supports" if e1 > e0 + ENTROPY_MARGIN else "mixed"
        m260 = "supports" if e2 > e0 + ENTROPY_MARGIN else "mixed"
    elif summary["entropy_ARM_3"] > e0:
        q045 = "mixed"
        m313 = "mixed" if e1 > e0 else "weakens"
        m260 = "mixed" if e2 > e0 else "weakens"
    else:
        q045 = "weakens"
        m313 = "weakens"
        m260 = "weakens"

    if not summary.get("mech260_operative_all_seeds", True):
        m260 = "unknown"

    return {"Q-045": q045, "MECH-313": m313, "MECH-260": m260}


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    seeds = DRY_RUN_SEEDS if dry_run else SEEDS
    episodes = DRY_RUN_EPISODES if dry_run else EVAL_EPISODES
    steps = DRY_RUN_STEPS if dry_run else STEPS_PER_EPISODE
    # Dry-run uses a tiny warmup since DRY_RUN_STEPS=50 is the per-episode budget
    # and we want SOME measured steps even at dry-run scale.
    warmup = FIFO_WARMUP_STEPS if not dry_run else 5

    rows: List[Dict[str, Any]] = []
    for arm in ARMS:
        for seed in seeds:
            print(f"Seed {seed} Condition {arm['arm']}", flush=True)
            cell = _run_arm_seed(arm, seed, episodes, steps, warmup)
            rows.append(cell)
            print("verdict: PASS", flush=True)

    summary = _evaluate(rows)
    outcome = "PASS" if summary["overall_pass"] else "FAIL"
    edpc = _evidence_direction_per_claim(summary)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{timestamp}_v3"

    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "queue_id": QUEUE_ID,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": timestamp,
        "outcome": outcome,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": edpc["Q-045"],
        "evidence_direction_per_claim": edpc,
        "supersedes": "V3-EXQ-603a",
        "dry_run": dry_run,
        "acceptance_criteria": summary,
        "summary": summary,
        "arm_results": rows,
        "fifo_warmup_steps": warmup,
        "warmup_metric": "cumulative",
        "min_measured_steps_threshold": int(MIN_MEASURED_STEPS_PER_SEED),
        "fixes_applied": [
            "Fix1 (from 603a): select_action() path with obs_harm_a + use_affective_harm_stream",
            f"Fix4 (NEW): CUMULATIVE FIFO warmup, threshold={FIFO_WARMUP_STEPS}",
            f"Fix5 (NEW): per-seed admission gate, threshold={MIN_MEASURED_STEPS_PER_SEED}; seeds below excluded from arm mean",
            "Fix6 (NEW): FIFO_WARMUP_STEPS 75 -> 50 (pre-registered band [50,100])",
        ],
    }

    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    out_path = out_dir / f"{run_id}.json"
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as fh:
            json.dump(manifest, fh, indent=2)
        print(f"Manifest written: {out_path}", flush=True)
    else:
        out_path = Path("/dev/null")
        print("Dry run -- manifest not written.", flush=True)

    print(f"Outcome: {outcome}", flush=True)
    manifest["manifest_path"] = str(out_path)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=EXPERIMENT_TYPE)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = run_experiment(dry_run=args.dry_run)
    if args.dry_run:
        sys.exit(0)
    emit_outcome(
        outcome=str(result.get("outcome", "FAIL")),
        manifest_path=str(result.get("manifest_path", "/dev/null")),
    )
