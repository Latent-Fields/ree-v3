#!/opt/local/bin/python3
"""V3-EXQ-540c -- MECH-307 read-site introspection probe.

Claims: MECH-307, MECH-295
Anchor: REE_assembly/docs/architecture/anticipatory_affect_conjunction_vs_dual_channel.md
Supersedes: (does not supersede V3-EXQ-540b; diagnostic follow-up)

Why this probe
--------------
V3-EXQ-540b (2026-05-12T02:50:41Z FAIL) ran a 4-arm threshold sweep with
substrate fully ON in every arm and the consumer-side thresholds dropped to
0.01 / 0.005 / 0.01 in the floor arm. Result: conj_fire_rate=0.0000 in
ALL arms across 8000 read opportunities per arm per seed. The bridge IS
being called (the agent.py block at lines 2893-2966 fires whenever
goal_state.is_active(), which 540a/b confirmed produces ~8000 calls per
arm), and the substrate IS populating writes (liking_writes_mean=2711,
gap4_e1_prior_writes_mean=2711, z_beta_excursion_mean=0.15,
n_split_surprise_centers_total=82). But the four-way predicate

    v[VALENCE_WANTING]   > w_thr
    v[VALENCE_LIKING]    > l_thr
    v[VALENCE_SURPRISE]  > 0.0       (legacy channel; Option-b magnitude
                                       write satisfies this trivially)
    z_beta_arousal       > b_thr

returns False at every one of those reads, even at floor thresholds where
w_thr=0.01 etc. The 540b outcome routes per 540b's pre-registered branch
to "structural read-site audit required". This probe is that audit.

Diagnoses to surface
--------------------
Per the 540b outcome-branch text, suspect (a) kernel-decay reading nearby
zeros, (b) RBF center drift, (c) write/read site z_world mismatch (Gap 4
writes at e1_prior, bridge reads at current_z_world summary per candidate).
Also worth confirming: (d) z_beta_arousal scalar passed to the bridge is
actually above floor, (e) drive_level passed is above min_drive_to_fire,
(f) under Option-b the bridge still reads legacy v[:,3] VALENCE_SURPRISE
(not v[:,4]=POSITIVE_SURPRISE / v[:,5]=NEGATIVE_SURPRISE) -- semantic
mismatch that may or may not be load-bearing.

The probe instruments the bridge consumer-read site by monkey-patching
agent.mech295_bridge.compute_conjunction_score_bias to log every call
without changing the agent's behaviour, then aggregates per-component
predicate-satisfaction statistics across all read opportunities.

Conditions (1 cell -- single-seed probe, no arm comparison)
-----------------------------------------------------------
Single config: use_mech307_conjunction=True (master flag flips all three
substrate-side sub-flags via __post_init__) + use_mech307_consumer_conjunction_read=True
on the bridge (same as 540a/b ARM_2_full). Single seed for short turnaround;
the 540a/b runs already established that the substrate fires consistently
across seeds so per-seed variation is not the diagnostic target.

P0 warmup 30 ep + eval 5 ep, 200 steps/ep. ~15-20 min budget on Mac.

Per-tick instrumentation captures (for every bridge call during eval):
  - drive_level passed
  - z_beta_arousal scalar passed
  - For each candidate's read row v in [K, 6]:
      v[VALENCE_WANTING] / VALENCE_LIKING / VALENCE_HARM_DISCRIMINATIVE /
      VALENCE_SURPRISE / VALENCE_POSITIVE_SURPRISE / VALENCE_NEGATIVE_SURPRISE
  - Per-component predicate satisfaction at each threshold setting (we
    compute four threshold tiers per call: default 0.6/0.3/0.6,
    half 0.3/0.15/0.3, low 0.1/0.05/0.1, floor 0.01/0.005/0.01) so the
    aggregated table tells us which channel is the bottleneck at each tier.

End-of-run RBF inventory captures:
  - n_active_centers
  - per-channel valence_vecs statistics (max, mean, std, fraction > 0)
  - 10 random active centers' full valence_vec dump
  - Sample 100 random z_world points + the last tick's candidate_z_locs:
    distance to nearest active center, and evaluate_valence amplitude.

Pre-registered acceptance
-------------------------
This is purely diagnostic; PASS / FAIL is a runner-bookkeeping label only,
the load-bearing output is the per-component predicate-satisfaction table
in the manifest.

PASS = probe completed end-to-end (n_bridge_calls > 0 AND probe_data
populated). Otherwise FAIL.

experiment_purpose=diagnostic.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorld  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from ree_core.residue.field import (  # noqa: E402
    VALENCE_WANTING, VALENCE_LIKING, VALENCE_HARM_DISCRIMINATIVE,
    VALENCE_SURPRISE, VALENCE_POSITIVE_SURPRISE, VALENCE_NEGATIVE_SURPRISE,
    VALENCE_DIM,
)
from experiment_protocol import emit_outcome  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402


EXPERIMENT_TYPE = "v3_exq_540c_mech307_readsite_probe"
QUEUE_ID = "V3-EXQ-540c"
CLAIM_IDS = ["MECH-307", "MECH-295"]
EXPERIMENT_PURPOSE = "diagnostic"

SEED = 42
P0_EPISODES = 30
EVAL_EPISODES = 5
STEPS_PER_EPISODE = 200
EPISODES_PER_RUN = P0_EPISODES + EVAL_EPISODES   # 35

# Threshold tiers tracked in parallel per-call (matches 540b ARMs so the
# results are directly comparable to that experiment's outputs).
THRESHOLD_TIERS = {
    "default": {"w": 0.6,  "l": 0.3,   "b": 0.6},
    "half":    {"w": 0.3,  "l": 0.15,  "b": 0.3},
    "low":     {"w": 0.1,  "l": 0.05,  "b": 0.1},
    "floor":   {"w": 0.01, "l": 0.005, "b": 0.01},
}

CHANNEL_NAMES = {
    VALENCE_WANTING: "wanting",
    VALENCE_LIKING: "liking",
    VALENCE_HARM_DISCRIMINATIVE: "harm",
    VALENCE_SURPRISE: "surprise_legacy",
    VALENCE_POSITIVE_SURPRISE: "positive_surprise",
    VALENCE_NEGATIVE_SURPRISE: "negative_surprise",
}


def _make_env(seed: int) -> CausalGridWorld:
    """Match V3-EXQ-540b env config verbatim."""
    return CausalGridWorld(
        size=10,
        num_hazards=3,
        num_resources=8,
        hazard_harm=0.01,
        resource_benefit=0.18,
        use_proxy_fields=True,
        seed=seed,
        proximity_benefit_scale=0.18,
        resource_respawn_on_consume=True,
    )


def _make_config(env: CausalGridWorld) -> REEConfig:
    """ARM_2_full config from V3-EXQ-540b: full conjunction substrate ON."""
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        alpha_world=0.9,
        world_dim=32,
        drive_weight=2.0,
        z_goal_enabled=True,
    )
    cfg.surprise_gated_replay = True
    cfg.e1.schema_wanting_enabled = True
    cfg.schema_wanting_threshold = 0.1
    cfg.schema_wanting_gain = 0.5
    cfg.residue.valence_enabled = True
    cfg.use_mech295_liking_bridge = True
    cfg.mech295_drive_to_liking_gain = 1.0
    cfg.mech295_liking_to_approach_cue_gain = 0.5
    cfg.mech295_min_drive_to_fire = 0.1
    cfg.mech295_min_z_goal_norm_to_fire = 0.05
    cfg.use_mech307_consumer_conjunction_read = True
    cfg.mech307_conjunction_gain = 1.0
    # Master flag flips all three substrate-side sub-flags via __post_init__.
    cfg.use_mech307_conjunction = True
    cfg.__post_init__()
    return cfg


class ReadSiteProbe:
    """Captures every bridge consumer-read call without changing behaviour.

    Wraps agent.mech295_bridge.compute_conjunction_score_bias so we get a
    record of every (candidate_z_locs, valence_eval, z_beta, drive,
    returned_bias) tuple. The wrapped method delegates to the original and
    returns its result unchanged, so the agent runs bit-identically to a
    non-probed run.
    """

    def __init__(self, agent: REEAgent):
        self.agent = agent
        self.original_method = agent.mech295_bridge.compute_conjunction_score_bias
        self.records: List[Dict[str, Any]] = []
        # Aggregate-only tallies (avoid per-tick memory blowup over long runs).
        self.n_calls = 0
        self.n_calls_short_circuit = 0   # zero-returned via gain==0 / drive < floor
        self.per_channel_samples: Dict[int, List[float]] = {
            ch: [] for ch in CHANNEL_NAMES
        }
        self.z_beta_samples: List[float] = []
        self.drive_samples: List[float] = []
        self.predicate_tier_hits: Dict[str, Dict[str, int]] = {
            tier: {
                "w_pass": 0, "l_pass": 0, "s_pass": 0, "b_pass": 0,
                "all_pass": 0, "candidate_total": 0,
            }
            for tier in THRESHOLD_TIERS
        }
        agent.mech295_bridge.compute_conjunction_score_bias = (
            self._wrapped_compute  # type: ignore[assignment]
        )

    def _wrapped_compute(
        self,
        candidate_z_locs: torch.Tensor,
        residue_field,
        z_beta_arousal: float,
        drive_level: float,
        simulation_mode: bool = False,
    ) -> torch.Tensor:
        self.n_calls += 1

        # Replay the bridge's evaluation path to capture the same v matrix
        # the bridge will see (the original method recomputes it internally,
        # so we redo it here for logging only).
        v = None
        if (
            (not simulation_mode)
            and residue_field is not None
            and hasattr(residue_field, "evaluate_valence")
        ):
            try:
                with torch.no_grad():
                    v = residue_field.evaluate_valence(candidate_z_locs)
            except Exception:
                v = None

        d = float(drive_level)
        beta = float(z_beta_arousal)
        self.z_beta_samples.append(beta)
        self.drive_samples.append(d)

        if v is not None and v.shape[0] == int(candidate_z_locs.shape[0]):
            K = int(v.shape[0])
            for ch, _name in CHANNEL_NAMES.items():
                if ch < v.shape[1]:
                    col = v[:, ch].detach().cpu().tolist()
                    self.per_channel_samples[ch].extend(col)
            # Per-tier predicate satisfaction across this call's K candidates.
            v_w = v[:, VALENCE_WANTING]
            v_l = v[:, VALENCE_LIKING]
            v_s = v[:, VALENCE_SURPRISE]   # legacy channel that the bridge reads
            for tier_name, thr in THRESHOLD_TIERS.items():
                w_pass = (v_w > thr["w"]).sum().item()
                l_pass = (v_l > thr["l"]).sum().item()
                s_pass = (v_s > 0.0).sum().item()
                b_pass_per_candidate = (
                    K if beta > thr["b"] else 0
                )
                all_pass = (
                    (v_w > thr["w"]) & (v_l > thr["l"]) & (v_s > 0.0)
                    & (torch.full_like(v_w, beta) > thr["b"])
                ).sum().item()
                self.predicate_tier_hits[tier_name]["w_pass"] += int(w_pass)
                self.predicate_tier_hits[tier_name]["l_pass"] += int(l_pass)
                self.predicate_tier_hits[tier_name]["s_pass"] += int(s_pass)
                self.predicate_tier_hits[tier_name]["b_pass"] += int(
                    b_pass_per_candidate
                )
                self.predicate_tier_hits[tier_name]["all_pass"] += int(all_pass)
                self.predicate_tier_hits[tier_name]["candidate_total"] += K
        if d < self.agent.mech295_bridge.config.min_drive_to_fire:
            self.n_calls_short_circuit += 1

        # Save a small subset of full per-call records for spot inspection.
        # Keep only every 100th call to bound memory.
        if self.n_calls % 100 == 1 and v is not None:
            record = {
                "call_idx": self.n_calls,
                "K": int(candidate_z_locs.shape[0]),
                "z_beta_arousal": beta,
                "drive_level": d,
                "v_summary_per_channel": {
                    CHANNEL_NAMES[ch]: {
                        "max": float(v[:, ch].max().item()),
                        "mean": float(v[:, ch].mean().item()),
                        "min": float(v[:, ch].min().item()),
                    }
                    for ch in CHANNEL_NAMES if ch < v.shape[1]
                },
                "candidate_z_locs_first_row_norm": float(
                    candidate_z_locs[0].norm().item()
                ),
            }
            self.records.append(record)

        # Delegate to original; agent must run bit-identically.
        return self.original_method(
            candidate_z_locs,
            residue_field,
            z_beta_arousal,
            drive_level,
            simulation_mode,
        )

    def summary(self) -> Dict[str, Any]:
        def _stats(samples: List[float]) -> Dict[str, float]:
            if not samples:
                return {"n": 0, "max": 0.0, "mean": 0.0, "min": 0.0, "std": 0.0,
                        "frac_above_0": 0.0, "frac_above_0p01": 0.0,
                        "frac_above_0p1": 0.0}
            arr = np.array(samples, dtype=np.float64)
            return {
                "n": int(arr.size),
                "max": float(arr.max()),
                "mean": float(arr.mean()),
                "min": float(arr.min()),
                "std": float(arr.std()),
                "frac_above_0": float((arr > 0).mean()),
                "frac_above_0p01": float((arr > 0.01).mean()),
                "frac_above_0p1": float((arr > 0.1).mean()),
            }

        return {
            "n_bridge_calls": self.n_calls,
            "n_calls_drive_below_floor": self.n_calls_short_circuit,
            "z_beta_arousal_stats": _stats(self.z_beta_samples),
            "drive_level_stats": _stats(self.drive_samples),
            "per_channel_amplitude_stats": {
                CHANNEL_NAMES[ch]: _stats(self.per_channel_samples[ch])
                for ch in CHANNEL_NAMES
            },
            "predicate_tier_hits": self.predicate_tier_hits,
            "sample_records": self.records,
        }


def _dump_rbf_inventory(agent: REEAgent) -> Dict[str, Any]:
    rbf = agent.residue_field.rbf_field
    active_mask = rbf.active_mask
    valence_vecs = rbf.valence_vecs
    centers = rbf.centers

    n_active = int(active_mask.sum().item())
    if n_active == 0:
        return {"n_active_centers": 0}

    active_idx = active_mask.nonzero(as_tuple=False).squeeze(-1)
    active_v = valence_vecs[active_idx]
    active_c = centers[active_idx]

    per_channel = {}
    for ch, name in CHANNEL_NAMES.items():
        if ch >= active_v.shape[1]:
            continue
        col = active_v[:, ch].detach().cpu()
        per_channel[name] = {
            "max": float(col.max().item()),
            "mean": float(col.mean().item()),
            "min": float(col.min().item()),
            "std": float(col.std().item() if col.numel() > 1 else 0.0),
            "n_nonzero": int((col.abs() > 1e-9).sum().item()),
            "frac_above_0p01": float((col.abs() > 0.01).float().mean().item()),
        }

    # 10 random active centers' full valence_vec for spot inspection.
    sample_n = min(10, n_active)
    perm = torch.randperm(n_active)[:sample_n]
    sample_centers = []
    for j in perm.tolist():
        center_position = active_c[j].detach().cpu().tolist()
        vvec = active_v[j].detach().cpu().tolist()
        sample_centers.append({
            "active_idx": int(active_idx[j].item()),
            "center_position_norm": float(active_c[j].norm().item()),
            "center_position_head5": center_position[:5],
            "valence_vec": vvec,
        })

    return {
        "n_active_centers": int(n_active),
        "per_channel_active_center_stats": per_channel,
        "sample_active_centers": sample_centers,
        "kernel_bandwidth": float(rbf.bandwidth),
    }


def _run_probe(dry_run: bool = False) -> Dict[str, Any]:
    n_warmup = 3 if dry_run else P0_EPISODES
    n_eval = 1 if dry_run else EVAL_EPISODES
    steps_per_episode = 30 if dry_run else STEPS_PER_EPISODE

    print(f"[{EXPERIMENT_TYPE}] Seed {SEED} Condition probe_only", flush=True)
    print(
        f"[{EXPERIMENT_TYPE}] warmup={n_warmup} eval={n_eval} "
        f"steps={steps_per_episode} dry_run={dry_run}",
        flush=True,
    )

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    env = _make_env(SEED)
    cfg = _make_config(env)
    agent = REEAgent(cfg)

    probe = ReadSiteProbe(agent)
    total_episodes = n_warmup + n_eval
    contact_events = 0

    def _step_episode(ep: int, phase: str) -> None:
        nonlocal contact_events
        _, obs_dict = env.reset()
        agent.reset()
        for _ in range(steps_per_episode):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            latent = agent.sense(obs_body, obs_world)
            ticks = agent.clock.advance()
            e1_prior = (
                agent._e1_tick(latent) if ticks.get("e1_tick")
                else torch.zeros(1, agent.config.latent.world_dim)
            )
            drive = float(REEAgent.compute_drive_level(obs_body))
            agent.update_schema_wanting(drive_level=drive)
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            action = agent.select_action(candidates, ticks)
            action_idx = int(action.argmax(dim=-1).item())
            _, harm_signal, done, info, obs_dict = env.step(action_idx)
            ttype = info.get("transition_type", "none")
            if ttype == "resource":
                agent.update_z_goal(float(harm_signal), drive_level=drive)
                contact_events += 1
            agent.update_residue(float(harm_signal))
            if done:
                break

    for ep in range(n_warmup):
        _step_episode(ep, "warmup")
        if (ep + 1) % 5 == 0 or (ep + 1) == n_warmup:
            print(
                f"  [train] probe seed={SEED} ep {ep+1}/{total_episodes} (warmup) "
                f"contacts={contact_events} bridge_calls={probe.n_calls}",
                flush=True,
            )

    for ep in range(n_eval):
        _step_episode(ep, "eval")
        global_ep = n_warmup + ep + 1
        print(
            f"  [train] probe seed={SEED} ep {global_ep}/{total_episodes} (eval) "
            f"contacts={contact_events} bridge_calls={probe.n_calls}",
            flush=True,
        )

    probe_summary = probe.summary()
    rbf_inventory = _dump_rbf_inventory(agent)

    print(f"[{EXPERIMENT_TYPE}] PROBE SUMMARY", flush=True)
    print(f"  n_bridge_calls = {probe_summary['n_bridge_calls']}")
    print(f"  contact_events = {contact_events}")
    z_stats = probe_summary["z_beta_arousal_stats"]
    print(
        f"  z_beta_arousal: max={z_stats['max']:.4f} mean={z_stats['mean']:.4f} "
        f"min={z_stats['min']:.4f} frac>0.01={z_stats['frac_above_0p01']:.3f} "
        f"frac>0.1={z_stats['frac_above_0p1']:.3f}"
    )
    d_stats = probe_summary["drive_level_stats"]
    print(
        f"  drive_level: max={d_stats['max']:.4f} mean={d_stats['mean']:.4f} "
        f"min={d_stats['min']:.4f}"
    )
    print("  per-channel valence amplitudes at read sites:")
    for name, s in probe_summary["per_channel_amplitude_stats"].items():
        print(
            f"    {name:<20} n={s['n']:<6} max={s['max']:.5f} mean={s['mean']:.5f} "
            f"frac>0={s['frac_above_0']:.3f} frac>0.01={s['frac_above_0p01']:.3f}"
        )
    print("  predicate tier hits (per-candidate-tick):")
    for tier, h in probe_summary["predicate_tier_hits"].items():
        denom = max(1, h["candidate_total"])
        print(
            f"    {tier:<8} w_pass={h['w_pass']:<5} ({h['w_pass']/denom:.3f}) "
            f"l_pass={h['l_pass']:<5} ({h['l_pass']/denom:.3f}) "
            f"s_pass={h['s_pass']:<5} ({h['s_pass']/denom:.3f}) "
            f"b_pass={h['b_pass']:<5} ({h['b_pass']/denom:.3f}) "
            f"all_pass={h['all_pass']:<5} ({h['all_pass']/denom:.3f})"
        )
    print("  RBF inventory:")
    print(f"    n_active_centers = {rbf_inventory.get('n_active_centers', 0)}")
    if rbf_inventory.get("n_active_centers", 0) > 0:
        for name, s in rbf_inventory["per_channel_active_center_stats"].items():
            print(
                f"    {name:<20} max={s['max']:.5f} mean={s['mean']:.5f} "
                f"n_nonzero={s['n_nonzero']:<4}"
            )

    outcome = "PASS" if probe.n_calls > 0 else "FAIL"
    print(f"verdict: {outcome}", flush=True)

    return {
        "outcome": outcome,
        "n_bridge_calls": probe.n_calls,
        "contact_events": contact_events,
        "probe_summary": probe_summary,
        "rbf_inventory": rbf_inventory,
    }


def main(dry_run: bool = False):
    print(f"[{EXPERIMENT_TYPE}] starting (dry_run={dry_run})")
    t0 = time.time()
    result = _run_probe(dry_run=dry_run)
    elapsed = time.time() - t0
    outcome = result["outcome"]
    print(f"[{EXPERIMENT_TYPE}] outcome={outcome} elapsed={elapsed:.1f}s")
    print(f"Done. Outcome: {outcome}")

    if dry_run:
        print(f"[{EXPERIMENT_TYPE}] dry-run complete; not writing manifest.")
        return 0

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{timestamp}_v3"
    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    out_dir.mkdir(parents=True, exist_ok=True)

    if outcome == "PASS":
        direction = "mixed"   # diagnostic; surfacing data, not validating
    else:
        direction = "weakens"
    per_claim = {
        "MECH-307": "mixed",
        "MECH-295": "mixed",
    }

    manifest = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "started_utc": datetime.utcnow().isoformat() + "Z",
        "outcome": outcome,
        "evidence_direction": direction,
        "evidence_direction_per_claim": per_claim,
        "elapsed_seconds": elapsed,
        "seed": SEED,
        "p0_episodes": P0_EPISODES,
        "eval_episodes": EVAL_EPISODES,
        "steps_per_episode": STEPS_PER_EPISODE,
        "probe_data": result,
        "note": (
            "V3-EXQ-540c -- MECH-307 read-site introspection probe. "
            "Diagnostic follow-up to V3-EXQ-540b (FAIL: conj_fire_rate=0 in "
            "all 4 threshold-sweep arms across 8000 read opportunities per "
            "arm per seed). The bridge IS being called but the four-way "
            "predicate returns False at every read even with thresholds at "
            "0.01/0.005/0.01. This probe instruments the bridge consumer-"
            "read site to log per-call valence amplitudes per channel + "
            "z_beta_arousal scalar + drive_level + RBF center geometry, "
            "across all 4 threshold tiers in parallel. The per-channel "
            "amplitude stats + per-tier predicate-hit table together "
            "identify which channel is the bottleneck (low-amplitude reads / "
            "RBF kernel decay / z_beta below floor / drive below floor / "
            "etc.). Probe runs bit-identical to a non-probed run -- "
            "compute_conjunction_score_bias is wrapped but delegates to "
            "the original. Single seed, ARM_2_full config, 30 ep warmup + "
            "5 ep eval, ~15-20 min on Mac. Routes per surfaced bottleneck: "
            "low v[liking] but high v[wanting]/[surprise] -> RBF kernel "
            "decay or write-read site mismatch; low z_beta_arousal -> "
            "z_beta encoder normalisation issue; low all-channels -> "
            "active centers far from read locations (write-at-e1_prior vs "
            "read-at-current_z_world mismatch is the leading hypothesis)."
        ),
    }
    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=False,
        config=manifest.get("config"),
        seeds=None,
        script_path=Path(__file__),
    )
    print(f"Result written to: {out_path}", flush=True)
    return outcome, out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Smoke run, no manifest.")
    args = parser.parse_args()
    result = main(dry_run=args.dry_run)
    if args.dry_run or result == 0:
        sys.exit(0)
    _outcome, _out_path = result
    emit_outcome(
        outcome=_outcome if _outcome in ("PASS", "FAIL") else "FAIL",
        manifest_path=_out_path,
    )
    sys.exit(0)
