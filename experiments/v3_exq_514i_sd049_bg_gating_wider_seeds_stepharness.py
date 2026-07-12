#!/opt/local/bin/python3
"""V3-EXQ-514i: SD-049 BG gating wider-seed sweep under StepHarness.

Next-letter re-run after V3-EXQ-514h completed PASS at 2026-05-10T17:32:00Z
on ree-cloud-1 but the manifest was never written before SIGTERM landed at
17:32:00.5 (auto-scaler shut the box down mid-completion under the buggy
"very stale heartbeat = safe to shutdown" rule, since fixed). The PASS is
recorded in runner_status with empty output_file, but no metrics survive.
514i re-runs identical logic to recover the metrics. Script logic and design
are unchanged from 514h; only the identifier and SUPERSEDES change.

Tier-1 retest: re-run the EXQ-514d BG gating diagnostic with rv-live substrate
(StepHarness) and a wider seed sweep. Supersedes EXQ-514d's narrow [42, 43, 44]
sweep with a 6-seed sweep and routes the per-tick loop through StepHarness.

Motivation
----------
EXQ-514d (3 seeds [42, 43, 44]) showed ever_committed=True only on seed 44; the
other two never crossed the commit threshold. The seed-44 signal suggests the
BG gate WOULD fire under SD-049 multi-resource if rv were free to fall below
the commit_threshold (0.40), but the inline loop never called update_residue()
so rv was pinned at precision_init=0.5 -- the running variance never updated,
and the BG gate could only fire by chance through other code paths.

The Q-042 regression contract (tests/contracts/test_running_variance_contract.py)
locks the failure mode: rv only moves when update_residue is called once per
env step. StepHarness encodes this invariant.

Hypothesis: with rv free to evolve under StepHarness, more seeds will produce
ever_committed=True, AND mean_committed_frac_last_third should be > 0.05 in
the majority of seeds. If so, the BG gate IS functional under SD-049
multi-resource and the previous near-zero commit rates were a substrate-level
plumbing artifact, not an SD-049 architectural problem.

Design
------
- StepHarness drives the per-tick loop (sense, _e1_tick, generate, update_z_goal,
  select_action, env.step, update_residue). Joint training same as EXQ-514d
  inline (compute_prediction_loss + classifier_loss_weight * id_loss after
  harness.step()).
- 6 seeds: [42, 43, 44, 45, 46, 47]. Doubles the EXQ-514d sweep so the
  monomodality of "seed 44 fires, others don't" is clearly distinguished from
  "all/most seeds fire" or "few/none fire" patterns.
- 90 episodes x 300 steps, same as EXQ-514d.
- Same env: SD-049 multi_resource_heterogeneity, 3 resource types, 8x8 grid,
  0 hazards, 15 resources. Holds the SD-049 substrate fixed so the only
  changes from EXQ-514d are (a) rv now live via StepHarness, (b) wider seeds.

experiment_purpose = "diagnostic" -- tracks BG state under SD-049, does not
adjudicate ARC-016 / SD-049 evidence directions.

Verdict mapping
---------------
- ever_committed_count >= 4/6 AND mean_committed_frac_last_third >= 0.05:
  PASS. Confirms BG gate IS functional under SD-049 once rv is live; previous
  near-zero commit rates in 514d/e were substrate-plumbing artifact.
  Next step: re-evaluate non_contributory tags on the 514f / 530b cohort with
  a substrate-aware lens.
- ever_committed_count <= 2/6: FAIL (gate-still-dormant verdict). SD-049
  multi-resource genuinely produces an environment where BG cannot commit
  even with rv free; route to substrate-conditional review or commit_threshold
  retuning.
- Mixed (3/6, or ever_committed high but committed_frac_last_third < 0.05):
  PASS but flagged for follow-up. Gate fires but does not stabilise -- design
  a bistable-on retest.

claim_ids: []
experiment_purpose: diagnostic
supersedes: V3-EXQ-514h
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.optim as optim

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from experiment_protocol import emit_outcome  # noqa: E402
from experiments._harness import StepHarness  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402

QUEUE_ID = "V3-EXQ-514i"
EXPERIMENT_TYPE = "v3_exq_514i_sd049_bg_gating_wider_seeds_stepharness"
CLAIM_IDS: List[str] = []
EXPERIMENT_PURPOSE = "diagnostic"
SUPERSEDES = "V3-EXQ-514h"

# Wider-seed sweep: 6 seeds doubles 514d's [42, 43, 44] window.
SEEDS = [42, 43, 44, 45, 46, 47]
N_EPISODES = 90
STEPS_PER_EPISODE = 300

GRID_SIZE = 8
N_HAZARDS = 0
N_RESOURCES = 15

COMMIT_THRESHOLD = 0.40
PRECISION_INIT_BASELINE = 0.5
RV_DIFF_FLOOR = 1e-6
LR = 1e-3
CLASSIFIER_LOSS_WEIGHT = 0.1

# Verdict thresholds (pre-registered).
PASS_EVER_COMMITTED_FRAC = 4.0 / 6.0   # >= 4 of 6 seeds.
PASS_COMMIT_FRAC_FLOOR = 0.05

DRY_RUN_EPISODES = 3


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        size=GRID_SIZE,
        num_hazards=N_HAZARDS,
        num_resources=N_RESOURCES,
        seed=seed,
        multi_resource_heterogeneity_enabled=True,
        n_resource_types=3,
        per_axis_drive_enabled=True,
    )


def _make_config(env: CausalGridWorldV2) -> REEConfig:
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        alpha_world=0.9,
        world_dim=32,
        drive_weight=2.0,
        z_goal_enabled=True,
    )
    cfg.latent.use_resource_encoder = True
    cfg.latent.z_resource_dim = 32
    cfg.latent.use_identity_classifier = True
    cfg.latent.identity_classifier_n_types = 3
    cfg.goal.goal_dim = cfg.latent.world_dim
    return cfg


def _entropy(counts: Dict[int, int]) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    ent = 0.0
    for c in counts.values():
        if c > 0:
            p = c / total
            ent -= p * math.log(p + 1e-12)
    return ent


def run_diagnostic_for_seed(seed: int, n_episodes: int, device: torch.device) -> Dict:
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = _make_env(seed)
    cfg = _make_config(env)
    agent = REEAgent(cfg)
    agent.train()
    opt = optim.Adam(agent.parameters(), lr=LR)

    harness = StepHarness(agent, env, train_mode=True, seed=seed)

    ep_committed_frac: List[float] = []
    ep_rv_mean: List[float] = []
    ep_rv_final: List[float] = []
    ep_action_entropy: List[float] = []
    ep_resource_contacts: List[int] = []

    # Boundary line for runner progress display.
    print(f"Seed {seed} Condition single", flush=True)

    for ep in range(n_episodes):
        _, obs_dict = env.reset()
        agent.reset()
        harness.reset()

        step_committed = 0
        step_rv_sum = 0.0
        step_rv_n = 0
        action_counts: Dict[int, int] = {}
        resource_contacts = 0

        for _step in range(STEPS_PER_EPISODE):
            # Capture pre-step BG / rv readings (matches EXQ-514d ordering:
            # BG state read AFTER select_action, BEFORE env.step). With
            # StepHarness, sense->select_action all happen inside step(),
            # and update_residue runs at the end. Reading after step() gives
            # rv after this tick's update; we add a pre-step rv read above
            # for parity with 514d.
            rv_pre = float(agent.e3._running_variance)

            result = harness.step(obs_dict)

            is_committed = bool(agent.beta_gate.is_elevated)
            if is_committed:
                step_committed += 1
            step_rv_sum += rv_pre
            step_rv_n += 1

            action_idx = int(result.action.argmax(dim=-1).item())
            action_counts[action_idx] = action_counts.get(action_idx, 0) + 1

            ttype = result.info.get("transition_type", "none")
            if ttype == "resource":
                resource_contacts += 1

            # Joint training step (same losses as EXQ-514d inline).
            opt.zero_grad()
            loss = agent.compute_prediction_loss()
            latent = result.latent
            if (
                agent.latent_stack.resource_encoder is not None
                and getattr(latent, "resource_prox_pred_r", None) is not None
            ):
                prox_target_val = float(result.info.get("resource_field_at_agent", 0.0))
                prox_target = torch.tensor(
                    [[prox_target_val]], dtype=torch.float32, device=device
                )
                res_loss = agent.compute_resource_encoder_loss(prox_target, latent)
                loss = loss + res_loss
            if getattr(agent.config.latent, "use_identity_classifier", False):
                target_type = int(result.info.get("sd049_consumed_type_tag_this_tick", 0))
                id_loss = agent.compute_resource_identity_loss(target_type, latent)
                loss = loss + CLASSIFIER_LOSS_WEIGHT * id_loss

            if loss.requires_grad:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                opt.step()

            obs_dict = result.next_obs_dict
            if result.done:
                break

        ep_committed_frac.append(step_committed / STEPS_PER_EPISODE)
        ep_rv_mean.append(step_rv_sum / max(1, step_rv_n))
        ep_rv_final.append(float(agent.e3._running_variance))
        ep_action_entropy.append(_entropy(action_counts))
        ep_resource_contacts.append(resource_contacts)

        if (ep + 1) % 10 == 0 or ep + 1 == n_episodes:
            print(
                f"  [train] seed={seed} ep {ep + 1}/{n_episodes} "
                f"committed_frac={ep_committed_frac[-1]:.3f} "
                f"rv_final={ep_rv_final[-1]:.6f}",
                flush=True,
            )

    def _mean(lst: List[float]) -> float:
        return sum(lst) / max(1, len(lst))

    n = len(ep_committed_frac)
    third = max(1, n // 3)
    first_frac = ep_committed_frac[:third]
    last_frac = ep_committed_frac[n - third:]
    first_rv = ep_rv_mean[:third]
    last_rv = ep_rv_mean[n - third:]

    ever_committed = any(f > 0 for f in ep_committed_frac)
    first_commit_ep = next((i for i, f in enumerate(ep_committed_frac) if f > 0), -1)

    return {
        "seed": seed,
        "commit_threshold": COMMIT_THRESHOLD,
        "ever_committed": ever_committed,
        "first_commit_episode": first_commit_ep,
        "committed_frac_first_third": _mean(first_frac),
        "committed_frac_last_third": _mean(last_frac),
        "rv_mean_first_third": _mean(first_rv),
        "rv_mean_last_third": _mean(last_rv),
        "rv_final": ep_rv_final[-1] if ep_rv_final else -1.0,
        "rv_diff_from_init": (
            abs(ep_rv_final[-1] - PRECISION_INIT_BASELINE) if ep_rv_final else 0.0
        ),
        "action_entropy_mean": _mean(ep_action_entropy),
        "action_entropy_final": ep_action_entropy[-1] if ep_action_entropy else -1.0,
        "resource_contacts_mean": _mean([float(c) for c in ep_resource_contacts]),
        "resource_contacts_total": sum(ep_resource_contacts),
        "ep_committed_frac": ep_committed_frac,
        "ep_rv_final": ep_rv_final,
        "ep_resource_contacts": ep_resource_contacts,
    }


def main_run(dry_run: bool = False) -> Dict:
    device = torch.device("cpu")
    n_eps = DRY_RUN_EPISODES if dry_run else N_EPISODES
    seeds = SEEDS[:1] if dry_run else SEEDS

    per_seed_results: List[Dict] = []
    t0_total = time.time()
    for seed in seeds:
        print(f"[514i] seed={seed} episodes={n_eps}", flush=True)
        t0 = time.time()
        res = run_diagnostic_for_seed(seed, n_eps, device)
        elapsed = time.time() - t0
        per_seed_results.append(res)
        # Per-seed verdict line for runner progress display.
        seed_pass = res["ever_committed"] and res["rv_diff_from_init"] > RV_DIFF_FLOOR
        print(
            f"  ever_committed={res['ever_committed']}  "
            f"first_commit_ep={res['first_commit_episode']}  "
            f"rv_final={res['rv_final']:.6f}  "
            f"rv_diff_from_init={res['rv_diff_from_init']:.6f}  "
            f"committed_frac_last_third={res['committed_frac_last_third']:.3f}  "
            f"contacts={res['resource_contacts_total']}  "
            f"elapsed={elapsed:.1f}s",
            flush=True,
        )
        print(f"verdict: {'PASS' if seed_pass else 'FAIL'}", flush=True)

    elapsed_total = time.time() - t0_total

    n_ever_committed = sum(1 for r in per_seed_results if r["ever_committed"])
    n_rv_moved = sum(1 for r in per_seed_results if r["rv_diff_from_init"] > RV_DIFF_FLOOR)
    mean_rv_final = (
        sum(r["rv_final"] for r in per_seed_results) / len(per_seed_results)
    ) if per_seed_results else 0.0
    mean_committed_frac_last = (
        sum(r["committed_frac_last_third"] for r in per_seed_results)
        / len(per_seed_results)
    ) if per_seed_results else 0.0

    n_seeds_total = len(per_seed_results)
    ever_committed_frac = (n_ever_committed / n_seeds_total) if n_seeds_total else 0.0

    # Substrate sanity: rv must be live in ALL seeds for verdicts to mean anything.
    substrate_live = (n_rv_moved == n_seeds_total) if n_seeds_total else False

    bg_gate_recovered = (
        substrate_live
        and ever_committed_frac >= PASS_EVER_COMMITTED_FRAC
        and mean_committed_frac_last >= PASS_COMMIT_FRAC_FLOOR
    )

    outcome = "PASS" if bg_gate_recovered else "FAIL"

    return {
        "outcome": outcome,
        "elapsed_seconds": elapsed_total,
        "n_seeds": n_seeds_total,
        "n_ever_committed": n_ever_committed,
        "ever_committed_frac": ever_committed_frac,
        "n_rv_moved": n_rv_moved,
        "substrate_live": substrate_live,
        "mean_rv_final": mean_rv_final,
        "mean_committed_frac_last_third": mean_committed_frac_last,
        "bg_gate_recovered_verdict": bg_gate_recovered,
        "per_seed_results": per_seed_results,
        "config": {
            "seeds": seeds,
            "n_episodes": n_eps,
            "steps_per_episode": STEPS_PER_EPISODE,
            "grid_size": GRID_SIZE,
            "n_hazards": N_HAZARDS,
            "n_resources": N_RESOURCES,
            "commit_threshold": COMMIT_THRESHOLD,
            "lr": LR,
            "classifier_loss_weight": CLASSIFIER_LOSS_WEIGHT,
            "pass_ever_committed_frac": PASS_EVER_COMMITTED_FRAC,
            "pass_commit_frac_floor": PASS_COMMIT_FRAC_FLOOR,
        },
    }


def write_manifest(result: Dict, run_id: str) -> str:
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_dir = (
        Path(__file__).resolve().parent.parent.parent
        / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.json"

    manifest = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "supersedes": SUPERSEDES,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": ts,
        "outcome": result["outcome"],
        "evidence_direction": "non_contributory",
        "n_seeds": result["n_seeds"],
        "n_seeds_ever_committed": result["n_ever_committed"],
        "ever_committed_frac": result["ever_committed_frac"],
        "n_rv_moved": result["n_rv_moved"],
        "substrate_live": result["substrate_live"],
        "mean_rv_final": result["mean_rv_final"],
        "mean_committed_frac_last_third": result["mean_committed_frac_last_third"],
        "bg_gate_recovered_verdict": result["bg_gate_recovered_verdict"],
        "elapsed_seconds": result["elapsed_seconds"],
        "config": result["config"],
        "per_seed_results": [
            {k: v for k, v in r.items()
             if k not in ("ep_committed_frac", "ep_rv_final", "ep_resource_contacts")}
            for r in result["per_seed_results"]
        ],
        "per_seed_episode_traces": [
            {
                "seed": r["seed"],
                "ep_committed_frac": r["ep_committed_frac"],
                "ep_rv_final": r["ep_rv_final"],
                "ep_resource_contacts": r["ep_resource_contacts"],
            }
            for r in result["per_seed_results"]
        ],
        "supersedes_note": (
            "V3-EXQ-514d (3 seeds [42, 43, 44]) showed ever_committed=True only "
            "on seed 44; the inline loop never called update_residue() so rv was "
            "pinned at precision_init=0.5 and BG gate had no live commit-threshold "
            "signal to read. This retest routes through StepHarness (Q-042 "
            "regression contract guarantees rv moves under update_residue) and "
            "doubles the seed window to [42-47] to clearly distinguish "
            "monomodal-seed-44 from gate-functional-across-seeds."
        ),
    }

    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=False,
        config=manifest.get("config"),
        seeds=SEEDS,
        script_path=Path(__file__),
    )
    print(f"[514i] written -> {out_path}", flush=True)
    return str(out_path)


def main():
    parser = argparse.ArgumentParser(description=EXPERIMENT_TYPE)
    parser.add_argument("--dry-run", action="store_true",
                        help="Quick smoke test (1 seed x 3 episodes)")
    args = parser.parse_args()

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"

    result = main_run(dry_run=args.dry_run)
    out_path = None
    if not args.dry_run:
        out_path = write_manifest(result, run_id)
    else:
        print("DRY_RUN_COMPLETE", flush=True)
        print(
            f"[dry-run] outcome={result['outcome']}  "
            f"bg_gate_recovered={result['bg_gate_recovered_verdict']}  "
            f"mean_rv_final={result['mean_rv_final']:.6f}  "
            f"substrate_live={result['substrate_live']}",
            flush=True,
        )
    return result, run_id, out_path


if __name__ == "__main__":
    _result, _run_id, _out_path = main()
    _outcome_emit = _result["outcome"] if _result["outcome"] in ("PASS", "FAIL") else "FAIL"
    emit_outcome(
        outcome=_outcome_emit,
        manifest_path=_out_path,
        run_id=_run_id,
        exit_reason="ok" if _outcome_emit == "PASS" else "fail",
    )
