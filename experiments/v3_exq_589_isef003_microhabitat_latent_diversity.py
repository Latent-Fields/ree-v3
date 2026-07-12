"""
V3-EXQ-589: EXQ-ISEF-003 Microhabitat Zones vs Homogeneous Geography -- Latent Diversity.

infant_substrate:GAP-12 closure experiment.

Scientific question: does 3-zone microhabitat structure (microhabitat_enabled=True,
n_microhabitats=3) produce greater z_world latent space diversity -- measured by PCA
absolute variance across the top 3 components -- than homogeneous geography at
episode 1000?

Design:
  ARM_0_control: microhabitat_enabled=False (homogeneous geography, standard infant
        params with resource_respawn_on_consume=True)
  ARM_1_treatment: microhabitat_enabled=True, n_microhabitats=3

Both arms: same agent config (alpha_world=0.9). Fresh agent per arm x seed.
No backprop -- agent acts under torch.no_grad(). z_world diversity reflects
environmental structure propagated through the (randomly-initialised) encoder.
Agent persists across all 1000 episodes within a seed x arm run so that the
_traj_store accumulates pairwise diversity across episodes.

Primary metric: PCA absolute variance (sum of top-3 singular_values^2 / (n-1))
from z_world vectors collected at every step of the snapshot episode (ep 999 =
episode 1000).

C1 (gate): ARM_1 top3_abs_var_sum > 1.2 * ARM_0 top3_abs_var_sum at snapshot
ep999, in >= 4/5 seeds.

C2 (advisory): ARM_1 mean traj_pairwise_cosine_mean > ARM_0 at ep999 snapshot.
Interpretation: more diverse trajectories in structured environment.

C3 (advisory): ARM_1 zone_coverage has >= 2 zones each with > 0.1 fraction in
>= 3/5 seeds. Checks that the agent visits multiple microhabitat zones, not
just one.

Interpretation grid:
  Outcome                                      | Diagnosis / next action
  ---------------------------------------------|--------------------------------------------
  C1 PASS (>= 4/5 seeds, ratio >= 1.2x)        | Microhabitat zones create richer z_world
                                               |   structure; substrate-readiness for
                                               |   ARC-065 diversity experiments confirmed;
                                               |   GAP-12 CLOSED; DEV-NEED-001/007 unblocked
  C1 FAIL (ARM_1 ~ ARM_0 in z_world variance) | Zone map not propagating into obs_world;
                                               |   check zone-specific resource/hazard
                                               |   density in obs channels; may need
                                               |   zone_id included in obs or larger
                                               |   zone factor contrasts
  C1 FAIL + C3 FAIL (zone_coverage < 2 zones  | Agent anchored to one zone; check
    > 0.1 in treatment)                       |   zone_C_ambient_bonus and hazard/resource
                                               |   contrast across zones; zone map present
                                               |   but agent not exploring it
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from experiment_protocol import emit_outcome  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_589_isef003_microhabitat_latent_diversity"
QUEUE_ID = "V3-EXQ-589"
CLAIM_IDS: List[str] = ["ARC-065"]
EXPERIMENT_PURPOSE = "diagnostic"

SEEDS = [42, 43, 44, 45, 46]
BODY_OBS_DIM = 12
WORLD_OBS_DIM = 250
ACTION_DIM = 4
N_EPISODES = 1000
STEPS_PER_EPISODE = 200

# Snapshot episodes (0-indexed): ep 499 = episode 500, ep 999 = episode 1000.
SNAPSHOT_EPS = {499, 999}

# Pre-registered acceptance thresholds.
# C1: ARM_1 top-3 PCA abs variance sum > this ratio x ARM_0 in >= min seeds.
C1_VAR_RATIO_MIN = 1.2
C1_MIN_SEEDS_PASSING = 4

# C3: treatment arm must have >= this many zones each with > this coverage fraction.
C3_MIN_ZONES_VISITED = 2
C3_ZONE_COVERAGE_THRESHOLD = 0.1


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


def _build_agent() -> REEAgent:
    cfg = REEConfig.from_dims(
        body_obs_dim=BODY_OBS_DIM,
        world_obs_dim=WORLD_OBS_DIM,
        action_dim=ACTION_DIM,
    )
    cfg.latent.alpha_world = 0.9
    return REEAgent(cfg)


def _build_env(*, microhabitat_enabled: bool, seed: int) -> CausalGridWorldV2:
    if microhabitat_enabled:
        return CausalGridWorldV2(
            seed=seed,
            resource_respawn_on_consume=True,
            microhabitat_enabled=True,
            n_microhabitats=3,
        )
    else:
        return CausalGridWorldV2(
            seed=seed,
            resource_respawn_on_consume=True,
            microhabitat_enabled=False,
        )


def _compute_pca_top3_var(z_world_matrix: np.ndarray) -> Tuple[float, List[float]]:
    """Compute PCA absolute variance (singular_values^2 / (n-1)) for top-3 components.

    Returns (top3_sum, [var_pc1, var_pc2, var_pc3]).
    Handles degenerate case (n < 4 rows) by returning zeros.
    """
    n, d = z_world_matrix.shape
    if n < 4:
        return 0.0, [0.0, 0.0, 0.0]
    centered = z_world_matrix - z_world_matrix.mean(axis=0)
    # Full SVD on centered matrix; singular values S satisfy S^2 = eigenvalues * (n-1).
    try:
        _, s, _ = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return 0.0, [0.0, 0.0, 0.0]
    # Absolute variance for each component.
    n_eff = max(1, n - 1)
    abs_vars = (s ** 2) / float(n_eff)
    # Pad to at least 3 components.
    while len(abs_vars) < 3:
        abs_vars = np.append(abs_vars, 0.0)
    var_top3 = [float(abs_vars[0]), float(abs_vars[1]), float(abs_vars[2])]
    top3_sum = sum(var_top3)
    return top3_sum, var_top3


def _run_arm(
    *,
    seed: int,
    arm_name: str,
    microhabitat_enabled: bool,
    dry_run: bool,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    agent = _build_agent()
    env = _build_env(microhabitat_enabled=microhabitat_enabled, seed=seed)

    n_episodes = 2 if dry_run else N_EPISODES
    # In dry_run, snapshot at the final episode; in real run, at eps 500 and 1000.
    active_snapshot_eps = {n_episodes - 1} if dry_run else SNAPSHOT_EPS

    # Snapshot storage: ep_idx -> {"z_world_vecs": list, "traj_cosine": float,
    #                               "zone_coverage": dict}
    snapshots: Dict[int, Dict[str, Any]] = {}

    for ep in range(n_episodes):
        _flat, obs_dict = env.reset()
        obs_body, obs_world = _extract_obs(obs_dict)

        collect_z_world = ep in active_snapshot_eps
        z_world_buf: List[np.ndarray] = []
        last_info: Optional[Dict[str, Any]] = None

        for _step in range(STEPS_PER_EPISODE):
            with torch.no_grad():
                action = agent.act_with_split_obs(
                    obs_body=obs_body, obs_world=obs_world
                )

            action_idx = int(action.argmax().item()) % env.action_dim
            _flat_obs, harm_signal, done, info, obs_dict = env.step(action_idx)
            agent.update_residue(float(harm_signal))

            last_info = info

            # Collect z_world at snapshot episodes.
            if collect_z_world and agent._current_latent is not None:
                zw = agent._current_latent.z_world.detach().squeeze(0).cpu().numpy()
                z_world_buf.append(zw)

            if done:
                _flat, obs_dict = env.reset()

            obs_body, obs_world = _extract_obs(obs_dict)

        # At snapshot boundary: compute PCA and record telemetry.
        if ep in active_snapshot_eps:
            if z_world_buf:
                z_mat = np.stack(z_world_buf, axis=0)
                top3_sum, var_top3 = _compute_pca_top3_var(z_mat)
            else:
                top3_sum, var_top3 = 0.0, [0.0, 0.0, 0.0]

            traj_cosine = float(last_info.get("traj_pairwise_cosine_mean", -1.0)) if last_info else -1.0
            zone_cov = last_info.get("zone_coverage", {}) if last_info else {}

            snapshots[ep] = {
                "n_z_world_vecs": len(z_world_buf),
                "top3_abs_var_sum": top3_sum,
                "var_pc1": var_top3[0],
                "var_pc2": var_top3[1],
                "var_pc3": var_top3[2],
                "traj_pairwise_cosine_mean": traj_cosine,
                "zone_coverage": {str(k): v for k, v in zone_cov.items()},
            }

        if (ep + 1) % 100 == 0 or (ep + 1) == n_episodes:
            snap_str = ""
            if ep in snapshots:
                sn = snapshots[ep]
                snap_str = (
                    f" top3var={sn['top3_abs_var_sum']:.4f}"
                    f" traj_cosine={sn['traj_pairwise_cosine_mean']:.3f}"
                )
            print(
                f"  [train] {arm_name} seed={seed} ep {ep + 1}/{n_episodes}{snap_str}",
                flush=True,
            )

    # Extract final snapshot (ep999 in full run; last episode in dry_run).
    final_ep = n_episodes - 1
    snap_final = snapshots.get(final_ep, {
        "top3_abs_var_sum": 0.0,
        "var_pc1": 0.0,
        "var_pc2": 0.0,
        "var_pc3": 0.0,
        "traj_pairwise_cosine_mean": -1.0,
        "zone_coverage": {},
    })

    # C3: count zones in treatment arm with > threshold coverage.
    zone_cov_dict = snap_final.get("zone_coverage", {})
    n_zones_visited = sum(
        1 for v in zone_cov_dict.values() if float(v) > C3_ZONE_COVERAGE_THRESHOLD
    )

    passed_arm = snap_final["top3_abs_var_sum"] > 0.0
    print(f"verdict: {'PASS' if passed_arm else 'FAIL'}", flush=True)

    return {
        "arm": arm_name,
        "top3_abs_var_sum_ep1000": snap_final["top3_abs_var_sum"],
        "var_pc1_ep1000": snap_final["var_pc1"],
        "var_pc2_ep1000": snap_final["var_pc2"],
        "var_pc3_ep1000": snap_final["var_pc3"],
        "traj_pairwise_cosine_mean_ep1000": snap_final["traj_pairwise_cosine_mean"],
        "n_zones_visited_ep1000": n_zones_visited,
        "zone_coverage_ep1000": zone_cov_dict,
        "snapshots": {str(k): v for k, v in snapshots.items()},
    }


def _run_seed(*, seed: int, dry_run: bool) -> Dict[str, Any]:
    arm_results: Dict[str, Dict[str, Any]] = {}
    for arm_name, microhabitat_enabled in [
        ("ARM_0_control", False),
        ("ARM_1_treatment", True),
    ]:
        print(f"Seed {seed} Condition {arm_name}", flush=True)
        result = _run_arm(
            seed=seed,
            arm_name=arm_name,
            microhabitat_enabled=microhabitat_enabled,
            dry_run=dry_run,
        )
        arm_results[arm_name] = result
    return {"seed": seed, "arm_results": arm_results}


def run_experiment(*, dry_run: bool = False) -> Dict[str, Any]:
    seeds = [SEEDS[0]] if dry_run else SEEDS
    print(
        f"V3-EXQ-589: EXQ-ISEF-003 microhabitat latent diversity",
        flush=True,
    )
    print(
        f"  dry_run={dry_run} seeds={seeds} "
        f"n_episodes={2 if dry_run else N_EPISODES} "
        f"steps={STEPS_PER_EPISODE}",
        flush=True,
    )
    print(
        f"  ARM_0: microhabitat_enabled=False (homogeneous geography)",
        flush=True,
    )
    print(
        f"  ARM_1: microhabitat_enabled=True n_microhabitats=3",
        flush=True,
    )
    print(
        f"  C1 gate: ARM_1 top3_abs_var_sum > {C1_VAR_RATIO_MIN}x ARM_0 "
        f"in >= {C1_MIN_SEEDS_PASSING}/5 seeds at ep999",
        flush=True,
    )

    all_seed_results: List[Dict[str, Any]] = []
    for seed in seeds:
        result = _run_seed(seed=seed, dry_run=dry_run)
        all_seed_results.append(result)

    # C1: per-seed top3_abs_var_sum ratio at ep999.
    seed_c1_results: List[Dict[str, Any]] = []
    for r in all_seed_results:
        arm0_var = r["arm_results"]["ARM_0_control"]["top3_abs_var_sum_ep1000"]
        arm1_var = r["arm_results"]["ARM_1_treatment"]["top3_abs_var_sum_ep1000"]
        ratio = arm1_var / arm0_var if arm0_var > 1e-9 else 999.0
        seed_pass = ratio >= C1_VAR_RATIO_MIN
        seed_c1_results.append({
            "seed": r["seed"],
            "arm0_top3_var_sum": arm0_var,
            "arm1_top3_var_sum": arm1_var,
            "ratio": ratio,
            "c1_pass": seed_pass,
        })

    seeds_passing_c1 = sum(1 for s in seed_c1_results if s["c1_pass"])
    c1_pass = seeds_passing_c1 >= C1_MIN_SEEDS_PASSING or dry_run

    def _mean_f(xs: List[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    # C2: advisory -- mean traj_pairwise_cosine_mean at ep999.
    arm0_traj = [r["arm_results"]["ARM_0_control"]["traj_pairwise_cosine_mean_ep1000"]
                 for r in all_seed_results]
    arm1_traj = [r["arm_results"]["ARM_1_treatment"]["traj_pairwise_cosine_mean_ep1000"]
                 for r in all_seed_results]
    arm0_traj_mean = _mean_f([v for v in arm0_traj if v >= 0])
    arm1_traj_mean = _mean_f([v for v in arm1_traj if v >= 0])
    c2_pass = arm1_traj_mean > arm0_traj_mean

    # C3: advisory -- treatment arm has >= 2 zones with > 0.1 coverage in >= 3/5 seeds.
    arm1_zones = [r["arm_results"]["ARM_1_treatment"]["n_zones_visited_ep1000"]
                  for r in all_seed_results]
    c3_seeds_passing = sum(1 for n in arm1_zones if n >= C3_MIN_ZONES_VISITED)
    c3_pass = c3_seeds_passing >= 3 or dry_run

    outcome = "PASS" if c1_pass else "FAIL"

    print("", flush=True)
    print(f"=== V3-EXQ-589 Results ===", flush=True)
    print(
        f"C1 gate: ARM_1 top3_abs_var_sum > {C1_VAR_RATIO_MIN}x ARM_0 "
        f"in >= {C1_MIN_SEEDS_PASSING}/5 seeds",
        flush=True,
    )
    print(f"  Seeds passing C1: {seeds_passing_c1}/{len(seeds)}", flush=True)
    for sc in seed_c1_results:
        print(
            f"  seed={sc['seed']} arm0_var={sc['arm0_top3_var_sum']:.4f} "
            f"arm1_var={sc['arm1_top3_var_sum']:.4f} ratio={sc['ratio']:.3f} "
            f"c1={'PASS' if sc['c1_pass'] else 'FAIL'}",
            flush=True,
        )
    print(f"C1 overall: {'PASS' if c1_pass else 'FAIL'}", flush=True)
    print(
        f"C2 [advisory] traj_pairwise_cosine_mean: "
        f"arm0={arm0_traj_mean:.4f} arm1={arm1_traj_mean:.4f} "
        f"arm1_exceeds={'YES' if c2_pass else 'NO'}",
        flush=True,
    )
    print(
        f"C3 [advisory] zones > {C3_ZONE_COVERAGE_THRESHOLD} in treatment: "
        f"{c3_seeds_passing}/5 seeds have >= {C3_MIN_ZONES_VISITED} zones visited -> "
        f"{'PASS' if c3_pass else 'FAIL'}",
        flush=True,
    )
    print(f"Overall outcome: {outcome}", flush=True)

    return {
        "outcome": outcome,
        "c1_pass": c1_pass,
        "c2_pass": c2_pass,
        "c3_pass": c3_pass,
        "seeds_passing_c1": seeds_passing_c1,
        "c3_seeds_passing": c3_seeds_passing,
        "seed_c1_results": seed_c1_results,
        "arm0_traj_cosine_mean": arm0_traj_mean,
        "arm1_traj_cosine_mean": arm1_traj_mean,
        "all_seed_results": all_seed_results,
    }


def main(*, dry_run: bool = False) -> Tuple[str, Path]:
    result = run_experiment(dry_run=dry_run)
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

    c1_dir = "supports" if result["c1_pass"] else "does_not_support"

    manifest = {
        "run_id": run_id,
        "queue_id": QUEUE_ID,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "outcome": outcome,
        "evidence_direction": c1_dir,
        "evidence_direction_per_claim": {
            "ARC-065": c1_dir,
        },
        "timestamp_utc": datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
        "dry_run": dry_run,
        "config": {
            "seeds": list(SEEDS if not dry_run else [SEEDS[0]]),
            "n_episodes": N_EPISODES if not dry_run else 2,
            "steps_per_episode": STEPS_PER_EPISODE,
            "snapshot_episodes": sorted(SNAPSHOT_EPS),
            "arm0_microhabitat_enabled": False,
            "arm1_microhabitat_enabled": True,
            "arm1_n_microhabitats": 3,
            "alpha_world": 0.9,
            "c1_var_ratio_min": C1_VAR_RATIO_MIN,
            "c1_min_seeds_passing": C1_MIN_SEEDS_PASSING,
            "c3_min_zones_visited": C3_MIN_ZONES_VISITED,
            "c3_zone_coverage_threshold": C3_ZONE_COVERAGE_THRESHOLD,
        },
        "acceptance_criteria": {
            "C1_gate": (
                f"ARM_1 top3_abs_var_sum (PCA sum of top-3 absolute variances of z_world "
                f"at ep999) > {C1_VAR_RATIO_MIN}x ARM_0 in >= {C1_MIN_SEEDS_PASSING}/5 seeds"
            ),
            "C2_advisory": "ARM_1 mean traj_pairwise_cosine_mean > ARM_0 at ep999",
            "C3_advisory": (
                f"ARM_1 treatment zone_coverage has >= {C3_MIN_ZONES_VISITED} zones "
                f"each > {C3_ZONE_COVERAGE_THRESHOLD} in >= 3/5 seeds"
            ),
        },
        "criteria_results": {
            "C1_pass": result["c1_pass"],
            "seeds_passing_c1": result["seeds_passing_c1"],
            "C2_traj_cosine_advisory": result["c2_pass"],
            "C3_zone_coverage_advisory": result["c3_pass"],
            "c3_seeds_passing": result["c3_seeds_passing"],
        },
        "metrics": {
            "arm0_traj_pairwise_cosine_mean": result["arm0_traj_cosine_mean"],
            "arm1_traj_pairwise_cosine_mean": result["arm1_traj_cosine_mean"],
        },
        "per_seed_c1_results": result["seed_c1_results"],
        "per_seed_results": result["all_seed_results"],
        "notes": (
            "EXQ-ISEF-003: infant_substrate:GAP-12 closure. "
            "Microhabitat zones vs homogeneous geography latent diversity comparison. "
            "Primary criterion: z_world top-3 PCA absolute variance sum (sum of "
            "singular_values[:3]^2 / (n-1)) at ep999 (episode 1000) is >= 1.2x "
            "control in >= 4/5 seeds. Variance collected from all steps of the "
            "snapshot episode (200 vectors x 32-dim z_world). C2: trajectory "
            "pairwise cosine diversity (advisory; reads from env info dict at last "
            "step of snapshot episode). C3: zone visitation breadth in treatment "
            "(advisory; >= 2 zones with > 0.1 coverage fraction in >= 3/5 seeds). "
            "PASS closes GAP-12 and confirms ARC-065 substrate-readiness; unblocks "
            "DEV-NEED-001 / DEV-NEED-007. FAIL indicates zone map not propagating "
            "into z_world encoder -- check obs_world channel content per zone."
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
        summary = {k: v for k, v in manifest.items() if k != "per_seed_results"}
        print(json.dumps(summary, indent=2), flush=True)

    return outcome, out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    _outcome, _out_path = main(dry_run=args.dry_run)

    emit_outcome(
        outcome=_outcome if _outcome in ("PASS", "FAIL") else "FAIL",
        manifest_path=_out_path,
    )
    sys.exit(0)
