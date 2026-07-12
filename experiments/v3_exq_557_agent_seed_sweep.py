#!/opt/local/bin/python3
"""
V3-EXQ-557 -- Agent-seed sweep diagnostic.

Claims: [] (monostrategy-investigation diagnostic; no substrate claim under test)

Purpose (evidence_direction_note: diagnostic)
---------------------------------------------
V3-EXQ-555 (completed 2026-05-12T05:21Z) showed that under fixed env_seed=42,
agent_seed=7 sustains action_class_entropy ~ 0.68 while agent_seed=42 collapses
to entropy=0. Two data points are not enough to know the size of the
monostrategy basin in agent-init space. This experiment runs 30 different
agent_seeds at env_seed=42 fixed and produces a histogram of P1 entropy
values.

This bounds basin size and tells us whether seed-7-style escapes are
common-but-fragile or rare-and-special.

Design
------
* 30 cells, one per agent_seed.
* env_seed = 42 FIXED across all cells (matches V3-EXQ-555 C3 / C2 baseline).
* Single arm (ARM_NORMAL); no warmup override.
* Per-cell: P0 = 40 ep training + P1 = 60 ep eval, 200 steps/episode --
  mirrors V3-EXQ-555 ARM_NORMAL verbatim. Same ENV_KWARGS as V3-EXQ-555.
* Per cell logs P1 action_class_entropy, P1 action count, action-class
  distribution (counts dict).

Seed selection
--------------
30 values: replicates of V3-EXQ-552 seeds (7, 42, 17), the V3-EXQ-555 cells
(7, 42), low magnitudes 0..9, mid magnitudes 11..47 (primes / nearby), and
large magnitudes (100, 101, 200, 300, 500, 1000, 12345). Spread of
magnitudes guards against systematic seed-range bias.

Seed factorization
------------------
Reuses V3-EXQ-555's _make_agent_and_env helper unchanged via direct import:
sets torch.manual_seed / random.seed / np.random.seed to agent_seed BEFORE
REEAgent construction (controls weight init), passes env_seed=42
independently to CausalGridWorldV2, and re-seeds globals from agent_seed
after env construction. action-fallback RNG is seeded with env_seed=42 so
env-side stochasticity is constant across cells.

Pre-registered interpretation grid (4 rows; embedded in
evidence_direction_note)
--------------------------------------------------------------------------
  R1 deep_collapse_basin: >= 80% of seeds (>= 24/30) have entropy < 0.10
    -> monostrategy basin dominates agent-init space; seed-7-style
       escapes are rare. Reinforces the need for a substrate-side init
       fix rather than relying on lucky seeds.
  R2 bimodal_split: clear bimodal histogram (cluster near 0, cluster
       >= 0.5), with the collapsed cluster comprising 50%-80% of seeds
    -> basin is large but escapable; routes to characterising what
       distinguishes the two clusters (V3-EXQ-556 module-init swap on a
       member of each cluster).
  R3 mostly_diverse: >= 50% of seeds (>= 15/30) have entropy >= 0.30
    -> monostrategy is the minority outcome; V3-EXQ-555's seed=42 was
       unlucky, not seed=7 lucky.
  R4 continuous_spread: no bimodality, smooth distribution across
       [0, 0.7]
    -> not a basin at all; some seeds are partially diverse. Routes to
       a different framing entirely.

PASS = all 30 cells produce finite metrics, no ERROR. Interpretation
drives the next experiment, not the verdict.

experiment_purpose=diagnostic (decomposes a known anomaly; no falsifiable
substrate claim under test).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.optim as optim

from experiment_protocol import emit_outcome
from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2

# Reuse the V3-EXQ-555 factored-seed helper verbatim. This is the load-bearing
# import: _make_agent_and_env contains the agent_seed-vs-env_seed split logic.
# We also reuse ENV_KWARGS and the latent dims so the smoke-replication test
# (seeds 7/42 reproducing the V3-EXQ-555 entropies) is meaningful.
from experiments.v3_exq_555_seed7_env_factorization import (
    ENV_KWARGS,
    LR_E1,
    LR_E2_WF,
    LR_E3_HARM,
    LR_ENC_AUX,
    P0_TRAIN_EPISODES,
    P1_EVAL_EPISODES,
    STEPS_PER_EPISODE,
    _make_agent_and_env,
    _run_one_phase,
    _shannon_entropy,
)
from experiments.pack_writer import write_flat_manifest  # noqa: E402


EXPERIMENT_TYPE = "v3_exq_557_agent_seed_sweep"
QUEUE_ID = "V3-EXQ-557"
CLAIM_IDS: List[str] = []  # diagnostic; no substrate claim under test
EXPERIMENT_PURPOSE = "diagnostic"

# Fixed env seed across all 30 cells (matches V3-EXQ-555 C3 / C2).
ENV_SEED_FIXED = 42

# 30 agent seeds. Include 7 / 42 / 17 for V3-EXQ-552 / V3-EXQ-555
# replication anchors plus spread of magnitudes.
AGENT_SEEDS: List[int] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
    11, 13, 17, 19, 23, 29, 31, 37, 41, 42,
    43, 47, 53, 100, 101, 200, 300, 500, 1000, 12345,
]


# ---------------------------------------------------------------------------
# Per-cell run (matches V3-EXQ-555 _run_cell but parameterised on agent_seed
# only -- env_seed is held at ENV_SEED_FIXED).
# ---------------------------------------------------------------------------

def _run_cell(agent_seed: int) -> Dict:
    """Run one (env_seed=ENV_SEED_FIXED, agent_seed) cell.

    Returns dict with P1 action_class_counts + action_class_entropy.
    """
    agent, env = _make_agent_and_env(
        env_seed=ENV_SEED_FIXED, agent_seed=agent_seed,
    )

    e1_optimizer = optim.Adam(agent.e1.parameters(), lr=LR_E1)
    e2_wf_optimizer = optim.Adam(
        list(agent.e2.world_transition.parameters())
        + list(agent.e2.world_action_encoder.parameters()),
        lr=LR_E2_WF,
    )
    harm_eval_optimizer = optim.Adam(
        agent.e3.harm_eval_head.parameters(), lr=LR_E3_HARM,
    )
    aux_params = list(agent.latent_stack.parameters())
    aux_optimizer = optim.Adam(aux_params, lr=LR_ENC_AUX)

    wf_buf: List = []
    harm_eval_buf: List = []

    optimizers_and_params = {
        "e1_optimizer": e1_optimizer,
        "e2_wf_optimizer": e2_wf_optimizer,
        "harm_eval_optimizer": harm_eval_optimizer,
        "aux_optimizer": aux_optimizer,
        "aux_params": aux_params,
        "wf_buf": wf_buf,
        "harm_eval_buf": harm_eval_buf,
    }

    # Action-fallback RNG seeded from ENV_SEED_FIXED so env-side
    # stochasticity is constant across cells. Variation is attributable
    # to agent_seed alone.
    rng_module = random.Random(ENV_SEED_FIXED)

    p0_diag = _run_one_phase(
        agent=agent,
        env=env,
        phase_label="P0",
        num_episodes=P0_TRAIN_EPISODES,
        steps_per_episode=STEPS_PER_EPISODE,
        train=True,
        optimizers_and_params=optimizers_and_params,
        rng_module=rng_module,
        action_count_window=None,
    )
    print(
        f"  [train] agent_seed={agent_seed} P0 "
        f"{P0_TRAIN_EPISODES}/{P0_TRAIN_EPISODES} "
        f"n_total={p0_diag['n_total_actions']}",
        flush=True,
    )

    action_count_window: Dict[int, int] = {}
    p1_diag = _run_one_phase(
        agent=agent,
        env=env,
        phase_label="P1",
        num_episodes=P1_EVAL_EPISODES,
        steps_per_episode=STEPS_PER_EPISODE,
        train=False,
        optimizers_and_params=None,
        rng_module=rng_module,
        action_count_window=action_count_window,
    )
    p1_entropy = _shannon_entropy(action_count_window)
    p1_n_actions = sum(action_count_window.values())
    print(
        f"  [eval]  agent_seed={agent_seed} P1 "
        f"{P1_EVAL_EPISODES}/{P1_EVAL_EPISODES} "
        f"entropy={p1_entropy:.4f} n_actions={p1_n_actions}",
        flush=True,
    )

    return {
        "agent_seed": agent_seed,
        "env_seed": ENV_SEED_FIXED,
        "p1_action_class_counts": action_count_window,
        "p1_action_class_entropy": p1_entropy,
        "p1_n_actions": p1_n_actions,
        "p0_n_total_actions": p0_diag["n_total_actions"],
        "p1_n_total_actions": p1_diag["n_total_actions"],
    }


# ---------------------------------------------------------------------------
# Interpretation grid (4 rows)
# ---------------------------------------------------------------------------

def _classify_interpretation(entropies: List[float]) -> Tuple[str, str]:
    """Map the 30-cell entropy distribution to one of 4 interpretation rows.

    Thresholds:
      collapse: entropy < 0.10
      diverse:  entropy >= 0.30
      bimodal_high: entropy >= 0.50
    """
    n = len(entropies)
    if n == 0:
        return (
            "R_no_data",
            "Zero cells produced metrics. Routes to /diagnose-errors.",
        )

    n_collapse = sum(1 for e in entropies if e < 0.10)
    n_diverse = sum(1 for e in entropies if e >= 0.30)
    n_bimodal_high = sum(1 for e in entropies if e >= 0.50)
    n_mid = sum(1 for e in entropies if 0.10 <= e < 0.30)

    frac_collapse = n_collapse / n
    frac_diverse = n_diverse / n
    frac_mid = n_mid / n

    # R1 deep_collapse_basin: >= 80% at entropy < 0.10.
    if frac_collapse >= 0.80:
        return (
            "R1_deep_collapse_basin",
            (
                f"{n_collapse}/{n} ({frac_collapse:.0%}) of agent_seeds at "
                f"entropy < 0.10. Monostrategy basin dominates agent-init "
                f"space; seed-7-style escapes are rare. Reinforces the "
                f"need for a substrate-side init fix rather than relying "
                f"on lucky seeds."
            ),
        )

    # R3 mostly_diverse: >= 50% at entropy >= 0.30.
    if frac_diverse >= 0.50:
        return (
            "R3_mostly_diverse",
            (
                f"{n_diverse}/{n} ({frac_diverse:.0%}) of agent_seeds at "
                f"entropy >= 0.30. Monostrategy is the minority outcome; "
                f"V3-EXQ-555's seed=42 was unlucky, not seed=7 lucky."
            ),
        )

    # R2 bimodal_split: clear bimodal histogram (collapsed cluster
    # 50%-80%, with a second cluster >= 0.50 making up most of the rest
    # and the transition band 0.10-0.30 sparse).
    if (
        0.50 <= frac_collapse < 0.80
        and n_bimodal_high >= max(1, int(0.15 * n))
        and frac_mid < 0.30
    ):
        return (
            "R2_bimodal_split",
            (
                f"{n_collapse}/{n} ({frac_collapse:.0%}) collapsed cluster "
                f"(entropy < 0.10) and {n_bimodal_high}/{n} "
                f"({n_bimodal_high / n:.0%}) high-entropy cluster "
                f"(entropy >= 0.50) with sparse transition band "
                f"({n_mid}/{n} in [0.10, 0.30)). Basin is large but "
                f"escapable; routes to characterising what distinguishes "
                f"the two clusters (run V3-EXQ-556-style module-init swap "
                f"on a member of each)."
            ),
        )

    # R4 continuous_spread: no bimodality; smooth distribution.
    return (
        "R4_continuous_spread",
        (
            f"No clear bimodality: collapse {n_collapse}/{n} "
            f"({frac_collapse:.0%}), transition band {n_mid}/{n} "
            f"({frac_mid:.0%}), diverse {n_diverse}/{n} "
            f"({frac_diverse:.0%}). Some agent_seeds are partially "
            f"diverse rather than cleanly collapsed or cleanly escaping; "
            f"the metaphor of a discrete basin is wrong. Routes to a "
            f"different framing entirely."
        ),
    )


# ---------------------------------------------------------------------------
# Plan / smoke / main
# ---------------------------------------------------------------------------

def _print_plan() -> None:
    print(
        f"{EXPERIMENT_TYPE} -- 30-seed agent-init sweep at env_seed={ENV_SEED_FIXED}",
        flush=True,
    )
    print(f"Cells: {len(AGENT_SEEDS)} agent_seeds = {AGENT_SEEDS}", flush=True)
    print(
        f"P0 train: {P0_TRAIN_EPISODES} ep x {STEPS_PER_EPISODE} steps",
        flush=True,
    )
    print(
        f"P1 eval:  {P1_EVAL_EPISODES} ep x {STEPS_PER_EPISODE} steps",
        flush=True,
    )
    print(
        "Metric: P1 action_class_entropy per agent_seed (single cell per "
        "seed; the seed IS the variable).",
        flush=True,
    )
    print(
        "4-row interpretation grid: R1 deep_collapse_basin / R2 bimodal_split "
        "/ R3 mostly_diverse / R4 continuous_spread.",
        flush=True,
    )
    print(f"experiment_purpose={EXPERIMENT_PURPOSE}", flush=True)


def _run_smoke() -> None:
    """3-seed smoke (7, 42, 100) x 1 ep x 20 steps per phase; no manifest write.

    Seeds 7 and 42 are the V3-EXQ-555 anchors -- they should approximately
    replicate the V3-EXQ-555 entropies (~0.68 and 0.0 respectively) at FULL
    depth. At the shortened smoke depth (1 ep x 20 steps each) the
    monostrategy collapse has not had time to lock in, so smoke output is
    primarily a wiring sanity check rather than a quantitative replication.
    Seed 100 is a fresh test value.
    """
    print(
        "SMOKE MODE: agent_seeds [7, 42, 100] x 1 ep x 20 steps per phase; "
        "no manifest write. Seeds 7/42 are V3-EXQ-555 anchors -- at this "
        "depth, smoke output is a wiring check, not a quantitative "
        "replication of EXQ-555's 0.68 / 0.0 entropies.",
        flush=True,
    )

    smoke_seeds = [7, 42, 100]
    smoke_results: List[Tuple[int, float, int]] = []
    for seed in smoke_seeds:
        agent, env = _make_agent_and_env(
            env_seed=ENV_SEED_FIXED, agent_seed=seed,
        )

        e1_optimizer = optim.Adam(agent.e1.parameters(), lr=LR_E1)
        e2_wf_optimizer = optim.Adam(
            list(agent.e2.world_transition.parameters())
            + list(agent.e2.world_action_encoder.parameters()),
            lr=LR_E2_WF,
        )
        harm_eval_optimizer = optim.Adam(
            agent.e3.harm_eval_head.parameters(), lr=LR_E3_HARM,
        )
        aux_params = list(agent.latent_stack.parameters())
        aux_optimizer = optim.Adam(aux_params, lr=LR_ENC_AUX)
        wf_buf: List = []
        harm_eval_buf: List = []
        optimizers_and_params = {
            "e1_optimizer": e1_optimizer,
            "e2_wf_optimizer": e2_wf_optimizer,
            "harm_eval_optimizer": harm_eval_optimizer,
            "aux_optimizer": aux_optimizer,
            "aux_params": aux_params,
            "wf_buf": wf_buf,
            "harm_eval_buf": harm_eval_buf,
        }
        rng_module = random.Random(ENV_SEED_FIXED)
        _run_one_phase(
            agent=agent, env=env, phase_label="P0",
            num_episodes=1, steps_per_episode=20,
            train=True, optimizers_and_params=optimizers_and_params,
            rng_module=rng_module, action_count_window=None,
        )
        action_count_window: Dict[int, int] = {}
        _run_one_phase(
            agent=agent, env=env, phase_label="P1",
            num_episodes=1, steps_per_episode=20,
            train=False, optimizers_and_params=None,
            rng_module=rng_module, action_count_window=action_count_window,
        )
        ent = _shannon_entropy(action_count_window)
        n_act = sum(action_count_window.values())
        print(
            f"  smoke agent_seed={seed} P1 action_counts="
            f"{dict(action_count_window)} entropy={ent:.4f} "
            f"n_actions={n_act}",
            flush=True,
        )
        smoke_results.append((seed, ent, n_act))

    print(
        "Note: smoke depth (1 ep x 20 steps) is too short for the "
        "V3-EXQ-555 monostrategy lock-in pattern to manifest; smoke checks "
        "wiring and non-crashing execution end-to-end, not quantitative "
        "replication.",
        flush=True,
    )
    print("verdict: PASS", flush=True)
    print("SMOKE OK", flush=True)


def main() -> Tuple[Optional[str], Optional[str]]:
    parser = argparse.ArgumentParser(
        description=f"{EXPERIMENT_TYPE} 30-seed agent-init sweep",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print plan and exit; do not execute.",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="3-seed (7, 42, 100) x 1 ep x 20 step wiring smoke.",
    )
    args = parser.parse_args()
    _run_started = datetime.now(timezone.utc)

    if args.dry_run:
        _print_plan()
        print("DRY RUN OK", flush=True)
        return (None, None)

    if args.smoke:
        _run_smoke()
        return (None, None)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"
    out_dir = (
        Path(__file__).resolve().parents[2]
        / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    per_cell_results: List[Dict] = []
    for agent_seed in AGENT_SEEDS:
        print(
            f"Cell agent_seed={agent_seed} (env_seed={ENV_SEED_FIXED})",
            flush=True,
        )
        r = _run_cell(agent_seed)
        print("verdict: PASS", flush=True)
        per_cell_results.append(r)

    entropies = [r["p1_action_class_entropy"] for r in per_cell_results]
    n_cells = len(entropies)
    n_collapse = sum(1 for e in entropies if e < 0.10)
    n_mid = sum(1 for e in entropies if 0.10 <= e < 0.30)
    n_diverse = sum(1 for e in entropies if e >= 0.30)
    n_bimodal_high = sum(1 for e in entropies if e >= 0.50)

    mean_entropy = float(np.mean(entropies)) if entropies else 0.0
    std_entropy = float(np.std(entropies)) if entropies else 0.0
    min_entropy = float(np.min(entropies)) if entropies else 0.0
    max_entropy = float(np.max(entropies)) if entropies else 0.0

    # Anchor lookup: where do seeds 7, 42, 17 sit in the distribution?
    by_seed = {r["agent_seed"]: r for r in per_cell_results}
    anchor_summary = {
        "seed_7_entropy": by_seed.get(7, {}).get("p1_action_class_entropy"),
        "seed_17_entropy": by_seed.get(17, {}).get("p1_action_class_entropy"),
        "seed_42_entropy": by_seed.get(42, {}).get("p1_action_class_entropy"),
    }

    row_label, row_description = _classify_interpretation(entropies)

    summary = {
        "n_cells": n_cells,
        "n_collapse_lt_0p10": n_collapse,
        "n_mid_0p10_to_0p30": n_mid,
        "n_diverse_ge_0p30": n_diverse,
        "n_bimodal_high_ge_0p50": n_bimodal_high,
        "mean_entropy": mean_entropy,
        "std_entropy": std_entropy,
        "min_entropy": min_entropy,
        "max_entropy": max_entropy,
        "interpretation_row": row_label,
        "interpretation_description": row_description,
        "anchor_entropies": anchor_summary,
    }

    # Outcome convention: all 30 cells produced metrics = PASS; otherwise
    # FAIL (the runner converts an exception to ERROR before reaching here).
    if n_cells == len(AGENT_SEEDS):
        outcome = "PASS"
        evidence_direction = "non_contributory"
    else:
        outcome = "FAIL"
        evidence_direction = "inconclusive"

    print(f"\nOutcome: {outcome}", flush=True)
    print(f"Interpretation row: {row_label}", flush=True)
    print(
        f"  n_cells={n_cells}  mean={mean_entropy:.4f}  "
        f"std={std_entropy:.4f}  min={min_entropy:.4f}  "
        f"max={max_entropy:.4f}",
        flush=True,
    )
    print(
        f"  n_collapse(<0.10)={n_collapse}  n_mid([0.10, 0.30))={n_mid}  "
        f"n_diverse(>=0.30)={n_diverse}  n_high(>=0.50)={n_bimodal_high}",
        flush=True,
    )
    print(
        f"  anchors: seed_7={anchor_summary['seed_7_entropy']}  "
        f"seed_17={anchor_summary['seed_17_entropy']}  "
        f"seed_42={anchor_summary['seed_42_entropy']}",
        flush=True,
    )
    print(f"  {row_description}", flush=True)

    output = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": ts,
        "outcome": outcome,
        "evidence_direction": evidence_direction,
        "evidence_direction_note": (
            "Monostrategy-investigation diagnostic. Samples 30 agent_seeds "
            "at fixed env_seed=42 to characterise the monostrategy basin "
            "in agent-init space. Replicates V3-EXQ-555 ARM_NORMAL config "
            "verbatim (P0=40, P1=60, 200 steps/ep, same ENV_KWARGS) and "
            "reuses the V3-EXQ-555 _make_agent_and_env factored-seed "
            "helper unchanged. "
            "Pre-registered 4-row interpretation grid: "
            "(R1) deep_collapse_basin -- >= 80% of seeds at entropy < "
            "0.10; monostrategy basin dominates agent-init space; "
            "substrate-side init fix needed rather than lucky-seed "
            "reliance. "
            "(R2) bimodal_split -- collapsed cluster 50%-80% with a "
            "second cluster >= 0.50 and sparse transition band; basin "
            "is large but escapable; routes to V3-EXQ-556-style "
            "module-init swap on a member of each cluster. "
            "(R3) mostly_diverse -- >= 50% of seeds at entropy >= 0.30; "
            "monostrategy is the minority outcome; V3-EXQ-555 seed=42 "
            "was unlucky, not seed=7 lucky. "
            "(R4) continuous_spread -- no bimodality, smooth "
            "distribution; basin metaphor wrong; routes to different "
            "framing. "
            "experiment_purpose=diagnostic; evidence_direction set to "
            "non_contributory on PASS so governance does not weight any "
            "claim from this run."
        ),
        "pass_criteria_summary": summary,
        "per_cell_results": per_cell_results,
        "config": {
            "env_seed_fixed": ENV_SEED_FIXED,
            "agent_seeds": AGENT_SEEDS,
            "p0_train_episodes": P0_TRAIN_EPISODES,
            "p1_eval_episodes": P1_EVAL_EPISODES,
            "steps_per_episode": STEPS_PER_EPISODE,
            "env_kwargs": ENV_KWARGS,
            "ran_at_full_555_length": True,
            "factored_seed_helper_source": (
                "experiments.v3_exq_555_seed7_env_factorization."
                "_make_agent_and_env"
            ),
        },
    }

    out_file = write_flat_manifest(
        output,
        out_dir,
        dry_run=False,
        config=output.get("config"),
        seeds=None,
        script_path=Path(__file__),
        elapsed_seconds=(datetime.now(timezone.utc) - _run_started).total_seconds(),
    )
    print(f"Result written to: {out_file}", flush=True)

    return (outcome, str(out_file))


if __name__ == "__main__":
    _outcome, _manifest_path = main()
    if _outcome is not None and _manifest_path is not None:
        emit_outcome(outcome=_outcome, manifest_path=_manifest_path)
    sys.exit(0)
