"""
V3-EXQ-590: EXQ-ISEF-004 Novelty Bonus Goldilocks Calibration.

infant_substrate:GAP-13 closure experiment.

Scientific question: At what novelty_bonus_weight does the MECH-111 E3
curiosity signal provide maximum exploratory benefit without stochastic
attractor capture in the infant environment
(interoceptive_noise_enabled=True)?

GAP-4 binding constraint (passed downstream): SD-048 interoceptive noise
is the primary irreducible stochastic attractor. MECH-314a (z_world RBF
novelty) is structurally safe. This experiment tests MECH-111 (broadcast
EMA of E1 prediction error used in score_trajectory); MECH-314
StructuredCuriosity (per-candidate score_bias channel) is a separate
mechanism requiring a separate calibration experiment.

Design: grid search novelty_bonus_weight in [0.1, 0.3, 0.5, 0.7, 1.0].
Environment: interoceptive_noise_enabled=True (stochastic attractor present),
             microhabitat_enabled=True (structured environment),
             harm_gradient_enabled=True, harm_gradient_scale=0.30
             (infant Phase 2 configuration).
Agent: alpha_world=0.9; novelty_bonus_weight=<arm_weight> in E3Config.
Novelty EMA: updated per step from E1 prediction error via
             agent.compute_prediction_loss() with torch.no_grad()
             (no backprop required).
N: 3 seeds x 1000 episodes x 200 steps per arm (5 arms total).

Primary metrics (last 100 episodes per seed, eval window):
  H_pos: info["pos_entropy"] at episode end (Shannon entropy, nats).
         Max theoretical: ln(100) ~ 4.6 for size=10 grid.
  residue_coverage_pct: ResidueField.get_coverage_telemetry()
  novelty_ema_at_ep_end: agent.e3._novelty_ema

Acceptance criteria (pre-registered):
  C1 (signal active): any arm with mean_novelty_ema > 0.001 in >=2/3 seeds
  C2 (exploration): >=3/5 arms with mean_coverage > 0.05 in >=2/3 seeds
  PASS = C1 AND C2

Goldilocks report: arm with highest joint_metric = (norm_coverage +
norm_H_pos) / 2 (always reported as calibration output).

Interpretation grid:
  Outcome                                        | Diagnosis / next action
  -----------------------------------------------|-------------------------------
  C1 + C2 PASS, nonmonotone joint metric         | Goldilocks point identified.
    (inverted-U: peak arm > both neighbors)      |   Adopt optimal weight for
                                                 |   infant curriculum Phase 1.
  C1 + C2 PASS, monotone joint metric            | MECH-111 broadcast is weak
    (no inverted-U at these weights)             |   but active; report best
                                                 |   weight; recommend
                                                 |   MECH-314a per-candidate
                                                 |   novelty as upgrade path.
  C1 FAIL: novelty_ema flat <= 0.001             | E1 prediction error not
    across all arms                              |   accumulating; escalate
                                                 |   to /diagnose-errors.
  C2 FAIL: <3/5 arms with coverage in 2/3 seeds  | Exploration null; harm
                                                 |   gradient or residue path
                                                 |   needs tuning; queue
                                                 |   /diagnose-errors.

Expected runtime: ~500 min (3,000,000 total steps + per-step E1 novelty
forward pass: 5 arms x 3 seeds x 1000 ep x 200 steps/ep).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from experiment_protocol import emit_outcome  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_590_isef004_novelty_bonus_goldilocks"
QUEUE_ID = os.environ.get("REE_QUEUE_ID", "V3-EXQ-590")
CLAIM_IDS: List[str] = ["MECH-314"]
EXPERIMENT_PURPOSE = "evidence"

# Arm sweep: novelty_bonus_weight values to compare
NOVELTY_WEIGHTS: List[float] = [0.1, 0.3, 0.5, 0.7, 1.0]
SEEDS: List[int] = [42, 43, 44]
N_EPISODES: int = 1000
STEPS_PER_EPISODE: int = 200
EVAL_WINDOW: int = 100  # last N episodes for metric aggregation

# Standard V3 obs dims (default env size=10, no landmarks)
BODY_OBS_DIM: int = 12
WORLD_OBS_DIM: int = 250
ACTION_DIM: int = 4

# Acceptance thresholds (pre-registered)
C1_NOVELTY_EMA_MIN: float = 0.001   # any arm mean_novelty_ema must exceed
C1_MIN_SEEDS_PASSING: int = 2       # in at least this many seeds (out of 3)
C2_COVERAGE_MIN: float = 0.05       # mean_coverage threshold per arm
C2_MIN_ARMS_PASSING: int = 3        # at least this many arms must pass C2
C2_MIN_SEEDS_PER_ARM: int = 2       # per arm: must pass in this many seeds


def _arm_name(weight: float) -> str:
    """ARM_nbw01 ... ARM_nbw10 (tenths)."""
    label = f"{int(round(weight * 10)):02d}"
    return f"ARM_nbw{label}"


def _build_agent(*, novelty_bonus_weight: float) -> REEAgent:
    cfg = REEConfig.from_dims(
        body_obs_dim=BODY_OBS_DIM,
        world_obs_dim=WORLD_OBS_DIM,
        action_dim=ACTION_DIM,
        novelty_bonus_weight=novelty_bonus_weight,
    )
    cfg.latent.alpha_world = 0.9
    return REEAgent(cfg)


def _build_env(*, seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed,
        interoceptive_noise_enabled=True,
        microhabitat_enabled=True,
        harm_gradient_enabled=True,
        harm_gradient_scale=0.30,
        resource_respawn_on_consume=True,
    )


def _default_checkpoint_path() -> Path:
    return (
        REPO_ROOT.parent
        / "REE_assembly"
        / "evidence"
        / "experiments"
        / "_partial"
        / EXPERIMENT_TYPE
        / f"{QUEUE_ID}.json"
    )


def _load_checkpoint(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
    except Exception as exc:
        print(f"[checkpoint] ignoring unreadable checkpoint {path}: {exc}", flush=True)
        return {}
    if payload.get("queue_id") != QUEUE_ID:
        print(
            f"[checkpoint] ignoring checkpoint for queue_id={payload.get('queue_id')!r}",
            flush=True,
        )
        return {}
    arm_results = payload.get("arm_results")
    if not isinstance(arm_results, dict):
        return {}
    return {
        str(arm): [r for r in results if isinstance(r, dict)]
        for arm, results in arm_results.items()
        if isinstance(results, list)
    }


def _write_checkpoint(
    path: Path,
    arm_results: Dict[str, List[Dict[str, Any]]],
    *,
    dry_run: bool,
    complete: bool = False,
) -> None:
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "queue_id": QUEUE_ID,
        "experiment_type": EXPERIMENT_TYPE,
        "updated_at_utc": datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
        "complete": complete,
        "arm_results": arm_results,
    }
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2) + "\n")
    tmp_path.replace(path)


def _extract_obs(obs_dict: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
    obs_body = obs_dict["body_state"].float()
    if obs_body.shape[0] < BODY_OBS_DIM:
        obs_body = torch.cat([obs_body, torch.zeros(BODY_OBS_DIM - obs_body.shape[0])])
    elif obs_body.shape[0] > BODY_OBS_DIM:
        obs_body = obs_body[:BODY_OBS_DIM]
    obs_world = obs_dict["world_state"].float()
    if obs_world.shape[0] < WORLD_OBS_DIM:
        obs_world = torch.cat([obs_world, torch.zeros(WORLD_OBS_DIM - obs_world.shape[0])])
    elif obs_world.shape[0] > WORLD_OBS_DIM:
        obs_world = obs_world[:WORLD_OBS_DIM]
    return obs_body, obs_world


def _run_arm_seed(
    *,
    seed: int,
    novelty_bonus_weight: float,
    arm_name: str,
    dry_run: bool,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    agent = _build_agent(novelty_bonus_weight=novelty_bonus_weight)
    env = _build_env(seed=seed)

    n_episodes = 2 if dry_run else N_EPISODES
    steps_per_ep = 10 if dry_run else STEPS_PER_EPISODE

    ep_h_pos: List[float] = []
    ep_coverage: List[float] = []
    ep_novelty_ema: List[float] = []

    for ep in range(n_episodes):
        _flat, obs_dict = env.reset()
        obs_body, obs_world = _extract_obs(obs_dict)

        last_info: Dict = {}
        for _step in range(steps_per_ep):
            with torch.no_grad():
                action = agent.act_with_split_obs(obs_body=obs_body, obs_world=obs_world)

            action_idx = int(action.argmax().item()) % env.action_dim
            _flat_obs, harm_signal, done, info, obs_dict = env.step(action_idx)
            last_info = info
            agent.update_residue(float(harm_signal))

            # Update MECH-111 novelty EMA from E1 prediction error (no backprop).
            # compute_prediction_loss() returns zero when buffer < 2 (safe).
            if len(agent._world_experience_buffer) >= 2:
                with torch.no_grad():
                    novelty_err = float(agent.compute_prediction_loss().item())
                agent.e3.update_novelty_ema(novelty_err)

            if done:
                _flat, obs_dict = env.reset()
            obs_body, obs_world = _extract_obs(obs_dict)

        # Episode telemetry
        h_pos = float(last_info.get("pos_entropy", -1.0))
        telemetry = agent.residue_field.get_coverage_telemetry()
        coverage = float(telemetry["residue_coverage_pct"])
        novelty_ema = float(agent.e3._novelty_ema)

        ep_h_pos.append(h_pos)
        ep_coverage.append(coverage)
        ep_novelty_ema.append(novelty_ema)

        want_print = (ep + 1) % 100 == 0 or ep == n_episodes - 1
        if want_print or dry_run:
            print(
                f"  [train] {arm_name} seed={seed} ep {ep + 1}/{n_episodes}"
                f" H_pos={h_pos:.3f} coverage={coverage:.3f}"
                f" novelty_ema={novelty_ema:.5f}",
                flush=True,
            )

    # Aggregate over eval window (last EVAL_WINDOW episodes)
    eval_win = min(EVAL_WINDOW, n_episodes)
    mean_h_pos = sum(ep_h_pos[-eval_win:]) / eval_win
    mean_coverage = sum(ep_coverage[-eval_win:]) / eval_win
    mean_novelty_ema = sum(ep_novelty_ema[-eval_win:]) / eval_win

    c1_seed_pass = mean_novelty_ema > C1_NOVELTY_EMA_MIN
    c2_seed_pass = mean_coverage > C2_COVERAGE_MIN
    verdict = "PASS" if (c1_seed_pass and c2_seed_pass) else "FAIL"
    print(f"verdict: {verdict} (seed={seed} arm={arm_name})", flush=True)

    return {
        "arm": arm_name,
        "seed": seed,
        "novelty_bonus_weight": novelty_bonus_weight,
        "mean_h_pos_eval": mean_h_pos,
        "mean_coverage_eval": mean_coverage,
        "mean_novelty_ema_eval": mean_novelty_ema,
        "c1_seed_pass": c1_seed_pass,
        "c2_seed_pass": c2_seed_pass,
        "ep_h_pos_last5": ep_h_pos[-5:] if not dry_run else ep_h_pos,
        "ep_coverage_last5": ep_coverage[-5:] if not dry_run else ep_coverage,
        "ep_novelty_ema_last5": ep_novelty_ema[-5:] if not dry_run else ep_novelty_ema,
    }


def run_experiment(
    *,
    dry_run: bool = False,
    checkpoint_path: Optional[Path] = None,
    resume: bool = True,
) -> Dict[str, Any]:
    seeds = [SEEDS[0]] if dry_run else SEEDS
    checkpoint_path = checkpoint_path or _default_checkpoint_path()

    print(
        f"{QUEUE_ID}: EXQ-ISEF-004 novelty bonus Goldilocks calibration",
        flush=True,
    )
    print(
        f"  dry_run={dry_run} seeds={seeds}"
        f" n_episodes={2 if dry_run else N_EPISODES}"
        f" steps={10 if dry_run else STEPS_PER_EPISODE}"
        f" novelty_weights={NOVELTY_WEIGHTS}",
        flush=True,
    )
    if not dry_run:
        print(
            f"  checkpoint={checkpoint_path} resume={resume}",
            flush=True,
        )

    arm_results: Dict[str, List[Dict[str, Any]]] = (
        _load_checkpoint(checkpoint_path) if resume and not dry_run else {}
    )

    for w in NOVELTY_WEIGHTS:
        arm_name = _arm_name(w)
        arm_results.setdefault(arm_name, [])
        for seed in seeds:
            if any(r.get("seed") == seed for r in arm_results[arm_name]):
                print(f"[checkpoint] skip completed {arm_name} seed={seed}", flush=True)
                continue
            print(f"Seed {seed} Condition {arm_name}", flush=True)
            result = _run_arm_seed(
                seed=seed,
                novelty_bonus_weight=w,
                arm_name=arm_name,
                dry_run=dry_run,
            )
            arm_results[arm_name].append(result)
            _write_checkpoint(checkpoint_path, arm_results, dry_run=dry_run)

    _write_checkpoint(checkpoint_path, arm_results, dry_run=dry_run, complete=True)

    # Aggregate per arm across seeds
    arm_summary: Dict[str, Dict[str, Any]] = {}
    for arm_name, seed_results in arm_results.items():
        mean_h = sum(r["mean_h_pos_eval"] for r in seed_results) / len(seed_results)
        mean_cov = sum(r["mean_coverage_eval"] for r in seed_results) / len(seed_results)
        mean_nema = sum(r["mean_novelty_ema_eval"] for r in seed_results) / len(seed_results)
        c1_seeds = sum(1 for r in seed_results if r["c1_seed_pass"])
        c2_seeds = sum(1 for r in seed_results if r["c2_seed_pass"])
        arm_summary[arm_name] = {
            "novelty_bonus_weight": seed_results[0]["novelty_bonus_weight"],
            "mean_h_pos_across_seeds": mean_h,
            "mean_coverage_across_seeds": mean_cov,
            "mean_novelty_ema_across_seeds": mean_nema,
            "c1_seeds_passing": c1_seeds,
            "c2_seeds_passing": c2_seeds,
            "c1_arm_pass": c1_seeds >= C1_MIN_SEEDS_PASSING,
            "c2_arm_pass": c2_seeds >= C2_MIN_SEEDS_PER_ARM,
        }

    # C1: any arm has novelty signal active in >=2/3 seeds
    c1_pass = any(s["c1_arm_pass"] for s in arm_summary.values())

    # C2: >=3/5 arms with coverage in >=2/3 seeds
    c2_arms_passing = sum(1 for s in arm_summary.values() if s["c2_arm_pass"])
    c2_pass = c2_arms_passing >= C2_MIN_ARMS_PASSING

    # Goldilocks: arm with highest joint_metric = (norm_coverage + norm_H_pos) / 2
    coverage_vals = [s["mean_coverage_across_seeds"] for s in arm_summary.values()]
    h_pos_vals = [s["mean_h_pos_across_seeds"] for s in arm_summary.values()]
    max_coverage = max(coverage_vals) if max(coverage_vals) > 0.0 else 1.0
    max_h_pos = max(h_pos_vals) if max(h_pos_vals) > 0.0 else 1.0

    best_arm: Optional[str] = None
    best_joint: float = -1.0
    for arm_name, s in arm_summary.items():
        norm_cov = s["mean_coverage_across_seeds"] / max_coverage
        norm_h = max(0.0, s["mean_h_pos_across_seeds"]) / max_h_pos
        joint = (norm_cov + norm_h) / 2.0
        s["joint_metric"] = joint
        if joint > best_joint:
            best_joint = joint
            best_arm = arm_name

    # Detect nonmonotone (inverted-U): peak is in the interior
    arm_names_ordered = [_arm_name(w) for w in NOVELTY_WEIGHTS]
    joint_vals = [arm_summary[a]["joint_metric"] for a in arm_names_ordered]
    best_idx = joint_vals.index(max(joint_vals))
    nonmonotone = (best_idx > 0) and (best_idx < len(joint_vals) - 1)

    if dry_run:
        c1_pass = True
        c2_pass = True

    outcome = "PASS" if (c1_pass and c2_pass) else "FAIL"

    # Summary output
    print("", flush=True)
    print("--- Per-arm summary ---", flush=True)
    for arm_name in arm_names_ordered:
        s = arm_summary[arm_name]
        print(
            f"  {arm_name} (nbw={s['novelty_bonus_weight']:.1f}):"
            f" coverage={s['mean_coverage_across_seeds']:.3f}"
            f" H_pos={s['mean_h_pos_across_seeds']:.3f}"
            f" novelty_ema={s['mean_novelty_ema_across_seeds']:.5f}"
            f" joint={s.get('joint_metric', 0.0):.3f}"
            f" C1={s['c1_seeds_passing']}/3 C2={s['c2_seeds_passing']}/3",
            flush=True,
        )
    print(
        f"C1 (signal active, any arm in >=2/3 seeds): {'PASS' if c1_pass else 'FAIL'}",
        flush=True,
    )
    print(
        f"C2 ({c2_arms_passing}/5 arms with coverage in >=2/3 seeds):"
        f" {'PASS' if c2_pass else 'FAIL'}",
        flush=True,
    )
    goldilocks_weight: Optional[float] = (
        arm_summary[best_arm]["novelty_bonus_weight"] if best_arm else None
    )
    print(
        f"Goldilocks arm: {best_arm}"
        f" (nbw={goldilocks_weight})"
        f" nonmonotone={nonmonotone}",
        flush=True,
    )
    print(f"Overall outcome: {outcome}", flush=True)

    return {
        "outcome": outcome,
        "c1_pass": c1_pass,
        "c2_pass": c2_pass,
        "c2_arms_passing": c2_arms_passing,
        "goldilocks_arm": best_arm,
        "goldilocks_weight": goldilocks_weight,
        "nonmonotone_detected": nonmonotone,
        "best_joint_metric": best_joint,
        "arm_summary": arm_summary,
        "arm_results_per_seed": arm_results,
        "checkpoint_path": str(checkpoint_path),
    }


def main(
    *,
    dry_run: bool = False,
    checkpoint_path: Optional[Path] = None,
    resume: bool = True,
) -> Tuple[str, Path]:
    result = run_experiment(
        dry_run=dry_run,
        checkpoint_path=checkpoint_path,
        resume=resume,
    )
    outcome = result["outcome"]

    run_id = (
        f"{EXPERIMENT_TYPE}_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_v3"
    )
    out_dir = (
        REPO_ROOT.parent
        / "REE_assembly"
        / "evidence"
        / "experiments"
        / EXPERIMENT_TYPE
    )
    out_path = out_dir / f"{run_id}.json"

    evidence_dir = "supports" if result["c1_pass"] else "does_not_support"
    mech314_dir = "non_contributory" if result["c1_pass"] else "does_not_support"

    manifest = {
        "run_id": run_id,
        "queue_id": QUEUE_ID,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "outcome": outcome,
        "evidence_direction": evidence_dir,
        "evidence_direction_per_claim": {
            "MECH-314": mech314_dir,
        },
        "evidence_direction_note": (
            "GAP-13 calibrates MECH-111 (E3 broadcast novelty EMA from E1 "
            "prediction error in score_trajectory), NOT MECH-314 "
            "(StructuredCuriosity per-candidate score_bias). "
            "non_contributory to MECH-314 because the per-candidate "
            "curiosity pathway is not under test here. "
            "PASS = signal active (C1) + exploration confirmed (C2); "
            "Goldilocks weight adopted for infant curriculum Phase 1. "
            "If nonmonotone_detected=True: stochastic attractor boundary "
            "visible at high novelty_bonus_weight. "
            "MECH-314a (per-candidate RBF novelty) is the upgrade path "
            "when MECH-111 broadcast shows weak directional effect."
        ),
        "timestamp_utc": datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
        "dry_run": dry_run,
        "config": {
            "seeds": list(SEEDS if not dry_run else [SEEDS[0]]),
            "n_episodes": N_EPISODES if not dry_run else 2,
            "steps_per_episode": STEPS_PER_EPISODE if not dry_run else 10,
            "eval_window_episodes": EVAL_WINDOW,
            "novelty_weights_swept": NOVELTY_WEIGHTS,
            "env_interoceptive_noise_enabled": True,
            "env_microhabitat_enabled": True,
            "env_harm_gradient_enabled": True,
            "env_harm_gradient_scale": 0.30,
            "env_resource_respawn_on_consume": True,
            "agent_alpha_world": 0.9,
            "novelty_ema_update": (
                "per_step_via_compute_prediction_loss_no_grad"
            ),
        },
        "acceptance_criteria": {
            "C1_signal_active": (
                f"any arm mean_novelty_ema > {C1_NOVELTY_EMA_MIN}"
                f" in >={C1_MIN_SEEDS_PASSING}/3 seeds"
            ),
            "C2_exploration": (
                f">={C2_MIN_ARMS_PASSING}/5 arms with mean_coverage"
                f" > {C2_COVERAGE_MIN} in >={C2_MIN_SEEDS_PER_ARM}/3 seeds"
            ),
            "Goldilocks_calibration": (
                "arm with highest joint_metric = (norm_coverage + norm_H_pos)"
                " / 2; always reported regardless of C1/C2 outcome"
            ),
        },
        "criteria_results": {
            "C1_pass": result["c1_pass"],
            "C2_pass": result["c2_pass"],
            "c2_arms_passing": result["c2_arms_passing"],
        },
        "metrics": {
            "goldilocks_arm": result["goldilocks_arm"],
            "goldilocks_weight": result["goldilocks_weight"],
            "nonmonotone_detected": result["nonmonotone_detected"],
            "best_joint_metric": result["best_joint_metric"],
            "arm_summary": result["arm_summary"],
        },
        "per_seed_results": result["arm_results_per_seed"],
        "checkpoint_path": result["checkpoint_path"],
        "notes": (
            "infant_substrate:GAP-13 closure. EXQ-ISEF-004 novelty bonus "
            "Goldilocks calibration. Sweeps E3 novelty_bonus_weight (MECH-111 "
            "broadcast mechanism) across [0.1, 0.3, 0.5, 0.7, 1.0] in the "
            "infant Phase 2 environment (interoceptive noise ON = stochastic "
            "attractor risk active, microhabitat ON, harm gradient 0.30). "
            "Novelty EMA updated per step from E1 prediction error "
            "(compute_prediction_loss, no backprop; returns 0.0 when buffer < "
            "2). Primary outputs: (1) C1 signal-active confirmation, (2) C2 "
            "exploration quality confirmation, (3) Goldilocks weight for infant "
            "curriculum Phase 1. nonmonotone_detected=True signals that "
            "stochastic attractor boundary is visible (peak arm outperforms "
            "higher weights). MECH-314a per-candidate RBF novelty is the "
            "recommended upgrade path when MECH-111 broadcast shows weak "
            "directional exploration effect."
        ),
    }

    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = write_flat_manifest(
            manifest,
            out_dir,
            dry_run=False,
            config=manifest.get("config"),
            seeds=SEEDS,
            script_path=Path(__file__),
        )
        print(f"Result written to: {out_path}", flush=True)
    else:
        print("[dry-run] manifest not written", flush=True)
        print(json.dumps(manifest, indent=2), flush=True)

    return outcome, out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    _outcome, _out_path = main(
        dry_run=args.dry_run,
        checkpoint_path=args.checkpoint_path,
        resume=not args.no_resume,
    )

    emit_outcome(
        outcome=_outcome if _outcome in ("PASS", "FAIL") else "FAIL",
        manifest_path=_out_path,
    )
    sys.exit(0)
