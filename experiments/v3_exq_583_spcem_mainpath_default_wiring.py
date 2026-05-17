"""V3-EXQ-583 -- SP-CEM main-path default-wiring validation (DIAGNOSTIC).

On 2026-05-17 the support-preserving + stratified CEM ("SP-CEM") substrate was
landed as the REE-v3 MAIN-PATH DEFAULT: the HippocampalConfig dataclass and the
REEConfig.from_dims() signature defaults were flipped to the V3-EXQ-567
ARM_1-validated combination
  use_support_preserving_cem        False -> True
  support_preserving_stratified_elites False -> True
  support_preserving_ao_std_floor   0.0   -> 0.2
(support_preserving_min_first_action_classes stays 2).

This experiment validates the DEFAULT WIRING, not the substrate. The substrate
itself is already validated by V3-EXQ-567 PASS (2026-05-15:
selected_action_class_entropy 0.0124 -> 0.4965; candidate support 1.007 ->
2.810). The open question this run answers is narrower: does an agent built via
a BARE REEConfig.from_dims() -- passing NO support_preserving_* kwargs -- now
behave identically to one that sets the V3-EXQ-567 ARM_1 flags explicitly, and
materially differently from one that explicitly opts back into the legacy
collapsing CEM?

Arms (env + harness + metrics mirror V3-EXQ-567 exactly for comparability):
  ARM_default       -- from_dims(...) with NO support_preserving_* kwargs
                       (relies entirely on the new defaults)
  ARM_explicit_on   -- from_dims(..., use_support_preserving_cem=True,
                       support_preserving_stratified_elites=True,
                       support_preserving_ao_std_floor=0.2,
                       support_preserving_min_first_action_classes=2)
  ARM_explicit_off  -- from_dims(..., use_support_preserving_cem=False,
                       support_preserving_stratified_elites=False,
                       support_preserving_ao_std_floor=0.0,
                       support_preserving_min_first_action_classes=2)
                       (the bit-identical legacy opt-out)

Pre-registered acceptance criteria:
  P1 WIRING EQUIVALENCE -- for every matched seed, ARM_default equals
     ARM_explicit_on within EQUIV_TOL on every float metric AND exactly on the
     action-count / candidate-count dicts. (Same seed + identical effective
     config => deterministic => bit-identical. A non-zero gap means a config
     layer is leaking between from_dims and the gate.)
  P2 EFFECT REPRODUCED WITH ZERO FLAGS -- mean ARM_default
     selected_action_class_entropy > mean ARM_explicit_off + P2_ENTROPY_MARGIN
     AND mean ARM_default candidate_unique_first_action_classes_mean >=
     P2_MIN_UNIQUE. (Reproduces the V3-EXQ-567 ARM_1-vs-ARM_0 separation with
     NO flags passed in ARM_default.)
  overall PASS = P1 and P2.

Diagnostic interpretation grid (one row per outcome -> next action):
  P1 PASS + P2 PASS -> default wiring confirmed; SP-CEM is genuinely the
     main-path default; close the implement-substrate validation.
  P1 FAIL (ARM_default != ARM_explicit_on) -> a config layer overrides
     from_dims pass-through; re-open the config audit
     (config.py:3122-3146 unconditional assignment block).
  P1 PASS + P2 FAIL (ARM_default == ARM_explicit_off) -> the flipped defaults
     are not reaching the gate; re-audit HippocampalConfig dataclass defaults
     and the from_dims signature defaults.

Purpose: diagnostic. claim_ids: [] -- this is a substrate-wiring check, not
governance evidence for ARC-065 (that is V3-EXQ-569's matched-entropy job).
Excluded from governance confidence/conflict scoring.

Dry run:
  /opt/local/bin/python3 experiments/v3_exq_583_spcem_mainpath_default_wiring.py --dry-run
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
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiment_protocol import emit_outcome  # noqa: E402
from experiments._harness import StepHarness, StepHooks  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorld  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402


EXPERIMENT_TYPE = "v3_exq_583_spcem_mainpath_default_wiring"
QUEUE_ID = "V3-EXQ-583"
SUPERSEDES_QUEUE_ID = None
CLAIM_IDS: List[str] = []
EXPERIMENT_PURPOSE = "diagnostic"

SEEDS = [42, 43, 44]
EVAL_EPISODES = 40
STEPS_PER_EPISODE = 200

DRY_RUN_SEEDS = [42]
DRY_RUN_EPISODES = 2
DRY_RUN_STEPS = 50

# Acceptance thresholds (pre-registered)
EQUIV_TOL = 1e-9            # P1: max abs float gap ARM_default vs ARM_explicit_on
P2_ENTROPY_MARGIN = 0.05    # P2: ARM_default entropy > ARM_explicit_off + this
P2_MIN_UNIQUE = 2.0         # P2: ARM_default candidate unique-class mean >= this

STD_FLOOR = 0.2             # the landed support_preserving_ao_std_floor default

# Float metrics compared for P1 wiring equivalence.
EQUIV_FLOAT_KEYS = [
    "selected_action_class_entropy",
    "candidate_unique_first_action_classes_mean",
    "candidate_first_action_entropy_mean",
    "action_0_fraction",
]
# Exact-match (dict / int) metrics for P1 wiring equivalence.
EQUIV_EXACT_KEYS = [
    "total_steps",
    "unique_actions_taken",
    "action_counts",
    "candidate_first_action_counts",
    "support_preserving_active_steps",
]

ARMS: List[Dict[str, Any]] = [
    {"arm": "ARM_default", "pass_flags": False},
    {
        "arm": "ARM_explicit_on",
        "pass_flags": True,
        "use_support_preserving_cem": True,
        "support_preserving_stratified_elites": True,
        "support_preserving_ao_std_floor": STD_FLOOR,
    },
    {
        "arm": "ARM_explicit_off",
        "pass_flags": True,
        "use_support_preserving_cem": False,
        "support_preserving_stratified_elites": False,
        "support_preserving_ao_std_floor": 0.0,
    },
]


def _make_env(seed: int) -> CausalGridWorld:
    return CausalGridWorld(
        size=8,
        num_hazards=2,
        num_resources=10,
        hazard_harm=0.01,
        resource_benefit=0.25,
        energy_decay=0.015,
        use_proxy_fields=True,
        proximity_benefit_scale=0.18,
        proximity_harm_scale=0.01,
        resource_respawn_on_consume=True,
        seed=seed,
    )


def _make_config(env: CausalGridWorld, arm: Dict[str, Any]) -> REEConfig:
    # ARM_default: pass NO support_preserving_* kwargs -- exercise the landed
    # from_dims defaults exactly as an ordinary experiment that never opts in.
    if not arm["pass_flags"]:
        return REEConfig.from_dims(
            body_obs_dim=env.body_obs_dim,
            world_obs_dim=env.world_obs_dim,
            action_dim=env.action_dim,
            alpha_world=0.9,
            world_dim=32,
        )
    # ARM_explicit_on / ARM_explicit_off: set the three flags explicitly.
    return REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        alpha_world=0.9,
        world_dim=32,
        use_support_preserving_cem=bool(arm["use_support_preserving_cem"]),
        support_preserving_stratified_elites=bool(
            arm["support_preserving_stratified_elites"]
        ),
        support_preserving_ao_std_floor=float(
            arm["support_preserving_ao_std_floor"]
        ),
        support_preserving_min_first_action_classes=2,
    )


def _entropy(counts: Counter) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    value = 0.0
    for count in counts.values():
        p = count / total
        if p > 0.0:
            value -= p * math.log(p)
    return float(value)


def _mean(
    values: Iterable[float], default: Optional[float] = 0.0
) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return default
    return sum(vals) / len(vals)


def _round6(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), 6)


def _counter_from_dict(data: Dict[Any, Any]) -> Counter:
    counter: Counter = Counter()
    for key, value in data.items():
        counter[int(key)] += int(value)
    return counter


def _run_arm_seed(
    arm: Dict[str, Any],
    seed: int,
    episodes: int,
    steps_per_episode: int,
) -> Dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = _make_env(seed)
    cfg = _make_config(env, arm)
    agent = REEAgent(cfg)

    action_counts: Counter = Counter()
    candidate_first_action_counts: Counter = Counter()
    unique_candidate_classes: List[float] = []
    candidate_entropies: List[float] = []
    support_preserving_active_steps = 0
    total_steps = 0

    def on_action(*, agent, action, step, **_kw):  # type: ignore[no-untyped-def]
        nonlocal support_preserving_active_steps

        idx = int(action.argmax(dim=-1).item())
        action_counts[idx] += 1

        hdiag = agent.hippocampal.get_last_propose_diagnostics()
        if hdiag:
            candidate_first_action_counts.update(
                _counter_from_dict(hdiag.get("candidate_first_action_counts", {}))
            )
            unique_candidate_classes.append(
                float(hdiag.get("candidate_unique_first_action_classes", 0))
            )
            candidate_entropies.append(
                float(hdiag.get("candidate_first_action_entropy", 0.0))
            )
            if bool(hdiag.get("support_preserving_active", False)):
                support_preserving_active_steps += 1

    hooks = StepHooks(on_action=on_action)
    harness = StepHarness(agent, env, train_mode=False, hooks=hooks, seed=seed)

    for ep in range(episodes):
        _, obs_dict = env.reset()
        agent.reset()
        harness.reset()
        for _ in range(steps_per_episode):
            result = harness.step(obs_dict)
            total_steps += 1
            obs_dict = result.next_obs_dict
            if result.done:
                break
        if (ep + 1) % 10 == 0 or (ep + 1) == episodes:
            print(
                f"  [train] arm={arm['arm']} seed={seed} ep {ep + 1}/{episodes}",
                flush=True,
            )

    action_total = sum(action_counts.values())
    selected_entropy = _entropy(action_counts)
    candidate_unique_mean = _round6(_mean(unique_candidate_classes, None))
    candidate_entropy_mean = _round6(_mean(candidate_entropies, None))

    return {
        "arm": arm["arm"],
        "seed": seed,
        "total_steps": int(total_steps),
        "selected_action_class_entropy": round(selected_entropy, 6),
        "action_0_fraction": round(
            action_counts.get(0, 0) / action_total if action_total else 0.0,
            6,
        ),
        "unique_actions_taken": int(len(action_counts)),
        "action_counts": dict(sorted(action_counts.items())),
        "candidate_unique_first_action_classes_mean": candidate_unique_mean,
        "candidate_first_action_entropy_mean": candidate_entropy_mean,
        "candidate_first_action_counts": dict(
            sorted(candidate_first_action_counts.items())
        ),
        "support_preserving_active_steps": int(support_preserving_active_steps),
    }


def _arm_rows(rows: List[Dict[str, Any]], arm_name: str) -> List[Dict[str, Any]]:
    return [row for row in rows if row.get("arm") == arm_name]


def _mean_key(rows: List[Dict[str, Any]], key: str, default: float = 0.0) -> float:
    values = [row.get(key) for row in rows if row.get(key) is not None]
    return float(_mean(values, default) or default)


def _evaluate(arm_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    default_rows = _arm_rows(arm_results, "ARM_default")
    on_rows = _arm_rows(arm_results, "ARM_explicit_on")
    off_rows = _arm_rows(arm_results, "ARM_explicit_off")

    # --- P1: per-seed wiring equivalence ARM_default vs ARM_explicit_on ---
    default_by_seed = {row["seed"]: row for row in default_rows}
    on_by_seed = {row["seed"]: row for row in on_rows}
    seeds_matched = sorted(set(default_by_seed) & set(on_by_seed))

    max_float_gap = 0.0
    worst_float_key = None
    exact_mismatches: List[str] = []
    per_seed_gap: Dict[int, float] = {}

    for s in seeds_matched:
        d = default_by_seed[s]
        o = on_by_seed[s]
        seed_gap = 0.0
        for k in EQUIV_FLOAT_KEYS:
            dv = d.get(k)
            ov = o.get(k)
            if dv is None and ov is None:
                continue
            if dv is None or ov is None:
                exact_mismatches.append(f"seed{s}:{k}:None-mismatch")
                continue
            gap = abs(float(dv) - float(ov))
            if gap > seed_gap:
                seed_gap = gap
            if gap > max_float_gap:
                max_float_gap = gap
                worst_float_key = k
        for k in EQUIV_EXACT_KEYS:
            if d.get(k) != o.get(k):
                exact_mismatches.append(f"seed{s}:{k}")
        per_seed_gap[s] = round(seed_gap, 12)

    p1 = bool(
        len(seeds_matched) > 0
        and max_float_gap <= EQUIV_TOL
        and len(exact_mismatches) == 0
    )

    # --- P2: ARM_default reproduces the effect with zero flags passed ---
    default_entropy = _mean_key(default_rows, "selected_action_class_entropy")
    off_entropy = _mean_key(off_rows, "selected_action_class_entropy")
    default_support = _mean_key(
        default_rows, "candidate_unique_first_action_classes_mean"
    )
    off_support = _mean_key(
        off_rows, "candidate_unique_first_action_classes_mean"
    )

    p2_entropy = bool(default_entropy > off_entropy + P2_ENTROPY_MARGIN)
    p2_support = bool(default_support >= P2_MIN_UNIQUE)
    p2 = bool(p2_entropy and p2_support)

    return {
        "seeds_matched": seeds_matched,
        "p1_max_float_gap": round(max_float_gap, 12),
        "p1_worst_float_key": worst_float_key,
        "p1_exact_mismatches": exact_mismatches,
        "p1_per_seed_gap": per_seed_gap,
        "p1_wiring_equivalence": p1,
        "default_selected_entropy_mean": round(default_entropy, 6),
        "explicit_off_selected_entropy_mean": round(off_entropy, 6),
        "entropy_delta_default_minus_off": round(
            default_entropy - off_entropy, 6
        ),
        "default_candidate_support_mean": round(default_support, 6),
        "explicit_off_candidate_support_mean": round(off_support, 6),
        "p2_entropy_reproduced": p2_entropy,
        "p2_support_reproduced": p2_support,
        "p2_effect_reproduced": p2,
        "overall_pass": bool(p1 and p2),
    }


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    seeds = DRY_RUN_SEEDS if dry_run else SEEDS
    episodes = DRY_RUN_EPISODES if dry_run else EVAL_EPISODES
    steps = DRY_RUN_STEPS if dry_run else STEPS_PER_EPISODE

    arm_results: List[Dict[str, Any]] = []
    for arm in ARMS:
        for seed in seeds:
            print(f"Seed {seed} Condition {arm['arm']}", flush=True)
            cell = _run_arm_seed(arm, seed, episodes, steps)
            arm_results.append(cell)
            passed = cell["selected_action_class_entropy"] >= 0.0
            print(f"verdict: {'PASS' if passed else 'FAIL'}", flush=True)

    summary = _evaluate(arm_results)
    outcome = "PASS" if summary["overall_pass"] else "FAIL"

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
        "evidence_direction": "non_contributory",
        "evidence_direction_note": (
            "DIAGNOSTIC substrate-wiring check (experiment_purpose=diagnostic, "
            "claim_ids=[]). Validates that the 2026-05-17 SP-CEM main-path "
            "default flip wires through from_dims identically to the explicit "
            "V3-EXQ-567 ARM_1 flags (P1) and reproduces the diversity effect "
            "with zero flags passed (P2). Not governance evidence for ARC-065 "
            "(that is V3-EXQ-569's matched-entropy job). Pre-registered "
            "non_contributory -- excluded from confidence/conflict scoring; "
            "NOT a force-mapped FAIL."
        ),
        "supersedes_queue_id": SUPERSEDES_QUEUE_ID,
        "dry_run": dry_run,
        "config": {
            "seeds": seeds,
            "eval_episodes": episodes,
            "steps_per_episode": steps,
            "equiv_tol": EQUIV_TOL,
            "p2_entropy_margin": P2_ENTROPY_MARGIN,
            "p2_min_unique": P2_MIN_UNIQUE,
            "std_floor": STD_FLOOR,
            "arms": [arm["arm"] for arm in ARMS],
        },
        "acceptance_criteria": {
            "P1_wiring_equivalence": summary["p1_wiring_equivalence"],
            "P2_effect_reproduced": summary["p2_effect_reproduced"],
        },
        "summary": summary,
        "arm_results": arm_results,
    }

    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.json"

    if not dry_run:
        with open(out_path, "w") as fh:
            json.dump(manifest, fh, indent=2)
        print(f"Manifest written: {out_path}", flush=True)
    else:
        out_path = Path("/dev/null")
        print("Dry run -- manifest not written.", flush=True)

    print(f"Outcome: {outcome}", flush=True)
    for key, value in manifest["acceptance_criteria"].items():
        print(f"  {key}: {value}", flush=True)

    manifest["manifest_path"] = str(out_path)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="V3-EXQ-583 SP-CEM main-path default-wiring validation"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Short smoke run; no manifest written.",
    )
    args = parser.parse_args()

    result = run_experiment(dry_run=args.dry_run)

    if args.dry_run:
        sys.exit(0)

    _outcome_raw = str(result.get("outcome", "FAIL")).upper()
    _outcome = _outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL"
    emit_outcome(
        outcome=_outcome,
        manifest_path=str(result.get("manifest_path", Path("/dev/null"))),
    )
    sys.exit(0)
