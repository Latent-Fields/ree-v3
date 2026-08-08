"""V3-EXQ-904 -- ARC-070 decomposition trigger-selectivity + zoom-in confirming DV.

WHAT THIS IS (GOV-CONFIRM-1). ARC-070 (policy_decomposition_on_prediction_failure)
has a BUILT substrate (ree_core/policy/policy_decomposition.py, MECH-321, 2026-07-24)
and lit_conf 0.855 but genuine_exp_count = 0 -- zero claim-weighting experimental
evidence. This is its first. It is a WALL-INDEPENDENT representation/functional-signature
confirming DV: it confirms the mechanism produces ARC-070's predicted internal signature
(trigger-conditional re-segmentation into finer primitives), and is deliberately NOT a
behavioural / task-outcome test.

WHY WALL-INDEPENDENT (the design that unblocks a line 816/820/830 could not). The whole
816 -> 830 campaign self-routed substrate_not_ready_requeue for one reason: in a trained
encoder the region-V_s proxy SATURATES near 1.0 and never drops below threshold in the
measure window (failure_autopsy_816-820-policy-decomposition-cluster_2026-07-26), so the
R1 V_s trigger could never be exercised via emergent env dynamics. ARC-070's own
falsifiable prediction (claims.yaml functional_restatement) anticipates this and calls for
the trigger to be "artificially induced". This driver does exactly that: PolicyDecomposition
.evaluate() takes region_vs as a CALLER-SUPPLIED float (module.py:830-831 -- it never reads
MECH-269 substrate directly), so we override HippocampalModule._region_vs() to inject
region_vs as a CONTROLLED IV. The boundary path (real MECH-288 event_segmenter) stays
fully live over genuine rollout latents; only the V_s scalar is controlled. This sidesteps
the env-saturation wall entirely -- it does not depend on the env producing low-V_s states,
and it does not depend on the agent breaking any task-performance wall.

WHAT IT CONFIRMS vs WHAT STAYS OPEN (honest scoping). This confirms the OPERATION half of
ARC-070: given a trigger, the integrated proposer re-segments a chunk into finer primitives,
selectively (both R1-OR trigger sources drive it; the confident-V_s case does not spuriously
mass-decompose; an at-cap chunk is marked-unreliable not decomposed). It does NOT and cannot
confirm the TRIGGER-FIRES-UNDER-NATURAL-PREDICTION-FAILURE half -- that is env-blocked by the
V_s saturation above and is a separate (behavioural) question tracked on the MECH-321
harm-outcome line (V3-EXQ-844/867/867a/867b). So a PASS is bears_on-strong evidence for
ARC-070's mechanism signature, not a claim that decomposition currently fires in the live env.

ARMS (4) x SEEDS (4), matched shared seeds. region_vs is the controlled IV:
  ARM_OFF        -- use_policy_decomposition=False. Negative control (structural zero,
                    bit-identical-when-off is a landed contract). No region_vs override.
  ARM_ON_LOWVS   -- ON, region_vs forced to 0.1 (< threshold 0.5 -> V_s trigger). Confirms
                    the V_s trigger drives real integrated decomposition.
  ARM_ON_HIGHVS  -- ON, region_vs forced to 0.9 (> threshold -> V_s trigger arithmetically
                    OFF). HEADLINE: the real MECH-288 boundary detector, over genuine rollout
                    latents, independently drives decomposition (R1 OR + R2 shared substrate),
                    and the mechanism does NOT spuriously decompose when V_s is confident.
  ARM_ON_DEPTHCAP-- ON, region_vs forced to 0.1 but decomposition_depth_cap=1. Confirms R3:
                    an at-cap triggering chunk is MARKED UNRELIABLE, not decomposed.

MANIPULATION CHECK vs EVIDENCE -- the DV-symmetry discipline. decomp_n_vs_trigger is
`float(region_vs) < threshold`; with an injected region_vs it is ARITHMETICALLY FIXED
(1.0 of evaluations in LOWVS, 0 in HIGHVS). That is a MANIPULATION-VALIDITY check, NOT
claim weight -- reporting it as a measured effect would be the DV-symmetry trap
(failure_autopsy_V3-EXQ-604c_2026-07-20). Claim weight rests only on quantities the
injection does NOT determine:
  - ARM_ON_HIGHVS DV = decomp_n_boundary_fires / decomp_n_decomposed_precommit. Symmetry:
    the boundary detector's firing is a function of the real z_world rollout trajectory and
    is ORTHOGONAL to the region_vs scalar we inject -- the manipulation is not invariant
    under (indeed does not touch) this DV. A broken/dead detector reads 0 here.
  - ARM_ON_LOWVS DV = decomp_n_decomposed_precommit + grain_tiles_from_seeded_chunk. The
    injection sets whether the trigger fires; the DV measures whether the OPERATION
    (decompose_sequence tiling + proposer injection) actually completes and produces finer
    grain -- a broken tiler reads decomposed=0 or grain<=1. Not invariant under the DV.
  - ARM_ON_DEPTHCAP DV = decomp_n_marked_unreliable / decomp_n_decomposed_precommit. The
    cap manipulation is not invariant: a broken cap would decompose an at-cap chunk.
  - ARM_OFF: no measured delta -- a structural gating check, not an effect.

NON-DEGENERACY. The one genuine, non-forced load-bearing measurement is the HIGHVS boundary
path. If it never engages (total boundary_fires == 0), the load-bearing criterion is STARVED,
not falsified -- non_degenerate=False, direction unknown, self-route
boundary_path_measurement_starved (a longer rollout, not a substrate defect). Probe on this
config: 16 boundary fires in 8 episodes, so 20 episodes is comfortably above the floor.

SLEEP DRIVER: none (no sleep flags set).
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ree_core.utils.config import REEConfig  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.policy import ChunkedPrimitive, ChunkState  # noqa: E402
from experiment_protocol import emit_outcome  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_904_arc070_decomposition_trigger_selectivity"
EXPERIMENT_PURPOSE = "evidence"
CLAIM_IDS = ["ARC-070"]

SEEDS = [11, 23, 47, 71]
ARMS = ["ARM_OFF", "ARM_ON_LOWVS", "ARM_ON_HIGHVS", "ARM_ON_DEPTHCAP"]
N_EPISODES = 20
STEPS_PER_EPISODE = 24

VS_THRESHOLD = 0.5
DEPTH_CAP_DEFAULT = 3
DEPTH_CAP_DEGENERATE = 1  # ARM_ON_DEPTHCAP: forces at-cap -> mark-unreliable path
REGION_VS_LOW = 0.1       # < VS_THRESHOLD -> V_s trigger fires
REGION_VS_HIGH = 0.9      # > VS_THRESHOLD -> V_s trigger silent (boundary path only)
SEEDED_CHUNK_SEQUENCE = (0, 1, 2)

# Pre-registered thresholds (constants, not derived from run statistics).
SEED_PASS_FRACTION = 0.75          # 3 of 4 seeds
GRAIN_TILES_MIN = 2                # zoom-in: a decomposed chunk yields >= 2 finer tiles
MIN_DECOMPOSED = 1                 # per seed, LOWVS
MIN_BOUNDARY_FIRES = 1            # per seed, HIGHVS (the genuine, non-forced measurement)
MIN_MARKED_UNRELIABLE = 1         # per seed, DEPTHCAP

# Captured for z_goal-stream + enabled-default-off-flags recording (last ON cell).
_LAST_ON_AGENT: Any = None


def _arm_spec(arm: str):
    """(use_policy_decomposition, depth_cap, region_vs_override)."""
    if arm == "ARM_OFF":
        return False, DEPTH_CAP_DEFAULT, None
    if arm == "ARM_ON_LOWVS":
        return True, DEPTH_CAP_DEFAULT, REGION_VS_LOW
    if arm == "ARM_ON_HIGHVS":
        return True, DEPTH_CAP_DEFAULT, REGION_VS_HIGH
    if arm == "ARM_ON_DEPTHCAP":
        return True, DEPTH_CAP_DEGENERATE, REGION_VS_LOW
    raise ValueError(f"unknown arm {arm}")


def _build_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(size=8, num_hazards=0, num_resources=6,
                             use_proxy_fields=True, seed=seed)


def _arm_flags(arm: str) -> Dict[str, Any]:
    on, cap, _ = _arm_spec(arm)
    return {
        # Constant substrate stack in every arm (see 815): without these the
        # CEM pool has no chunk for MECH-321 to evaluate.
        "use_event_segmenter": True,
        "use_policy_chunking": True,
        "use_chunk_proposal_injection": True,
        "use_per_stream_vs": True,
        # The manipulation(s).
        "use_policy_decomposition": on,
        "decomposition_vs_threshold": VS_THRESHOLD,
        "decomposition_depth_cap": cap,
    }


def _build_agent(env: CausalGridWorldV2, arm: str) -> REEAgent:
    agent = REEAgent(REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=4,
        alpha_world=0.9,  # SD-008 z_world fidelity
        **_arm_flags(arm),
    ))
    chunk = ChunkedPrimitive(
        sequence=SEEDED_CHUNK_SEQUENCE, depth=1,
        state=ChunkState.CRYSTALLISED, selection_weight=1.0,
    )
    agent.policy_chunking.library.register(chunk)
    return agent


def _config_slice(arm: str) -> Dict[str, Any]:
    on, cap, rv = _arm_spec(arm)
    slice_ = {
        "env": {"size": 8, "num_hazards": 0, "num_resources": 6,
                "use_proxy_fields": True},
        "schedule": {"n_episodes": N_EPISODES, "steps": STEPS_PER_EPISODE},
        "alpha_world": 0.9,
        "seeded_chunk_sequence": list(SEEDED_CHUNK_SEQUENCE),
        # The driver-applied manipulation, declared so the fingerprint reflects it.
        "region_vs_override": rv,
    }
    slice_.update(_arm_flags(arm))
    return slice_


def _grain_tiles(agent: REEAgent) -> int:
    """Directly measure the zoom-in: how many finer tiles decompose_sequence
    produces from the seeded chunk at depth 1. Deterministic mechanism check on
    the same substrate the integrated path uses (evaluate() calls this)."""
    pd = agent.policy_decomposition
    if pd is None:
        return 0
    try:
        subs = pd.decompose_sequence(
            tuple(int(a) for a in SEEDED_CHUNK_SEQUENCE), 1,
            agent.policy_chunking.library)
        return int(len(subs) if subs else 0)
    except Exception:
        return -1  # measurement failed -> C_GRAIN cannot pass


def _run_cell(arm: str, seed: int) -> Dict[str, Any]:
    on, cap, region_vs_override = _arm_spec(arm)
    print(f"Seed {seed} Condition {arm}", flush=True)
    with arm_cell(seed, config_slice=_config_slice(arm),
                  script_path=Path(__file__)) as cell:
        env = _build_env(seed)
        agent = _build_agent(env, arm)
        # Inject the controlled trigger IV. _region_vs() is the SOLE region_vs
        # source for the decomposition path (module.py:809); overriding it leaves
        # the boundary path (real MECH-288 over genuine latents) untouched.
        if on and region_vs_override is not None:
            agent.hippocampal._region_vs = (lambda v=region_vs_override: float(v))
        wd = agent.config.latent.world_dim

        for ep in range(N_EPISODES):
            _, obs = env.reset()
            agent.reset()
            for _ in range(STEPS_PER_EPISODE):
                latent = agent.sense(obs["body_state"], obs["world_state"])
                ticks = agent.clock.advance()
                e1 = (agent._e1_tick(latent) if ticks.get("e1_tick")
                      else torch.zeros(1, wd, device=agent.device))
                cands = agent.generate_trajectories(latent, e1, ticks)
                action = agent.select_action(cands, ticks)
                _flat, _harm, done, _info, obs = env.step(
                    int(action.argmax(dim=-1).item()))
                if done:
                    break
            if (ep + 1) % 5 == 0:
                st = agent.get_policy_decomposition_state()
                print(f"  [train] decomp seed={seed} arm={arm} ep {ep+1}/{N_EPISODES} "
                      f"vs_trig={st.get('decomp_n_vs_trigger', 0)} "
                      f"boundary={st.get('decomp_n_boundary_fires', 0)} "
                      f"decomposed={st.get('decomp_n_decomposed_precommit', 0)}",
                      flush=True)

        st = agent.get_policy_decomposition_state()
        row = {
            "arm": arm,
            "seed": seed,
            "region_vs_override": region_vs_override,
            "depth_cap": cap,
            "decomposition_instantiated": bool(agent.policy_decomposition is not None),
            "decomp_n_evaluated_precommit": int(st.get("decomp_n_evaluated_precommit", 0)),
            "decomp_n_decomposed_precommit": int(st.get("decomp_n_decomposed_precommit", 0)),
            "decomp_n_marked_unreliable": int(st.get("decomp_n_marked_unreliable", 0)),
            "decomp_n_vs_trigger": int(st.get("decomp_n_vs_trigger", 0)),
            "decomp_n_boundary_fires": int(st.get("decomp_n_boundary_fires", 0)),
            "grain_tiles_from_seeded_chunk": _grain_tiles(agent),
        }
        cell.stamp(row)
        if on:
            global _LAST_ON_AGENT
            _LAST_ON_AGENT = agent  # for z_goal-stream recording at manifest write

    # One verdict line per (seed x arm) cell (runner progress contract).
    on2, _, _ = _arm_spec(arm)
    if arm == "ARM_OFF":
        cell_ok = (not row["decomposition_instantiated"]
                   and row["decomp_n_evaluated_precommit"] == 0)
    elif arm == "ARM_ON_LOWVS":
        cell_ok = (row["decomp_n_decomposed_precommit"] >= MIN_DECOMPOSED
                   and row["grain_tiles_from_seeded_chunk"] >= GRAIN_TILES_MIN)
    elif arm == "ARM_ON_HIGHVS":
        cell_ok = (row["decomp_n_vs_trigger"] == 0
                   and row["decomp_n_boundary_fires"] >= MIN_BOUNDARY_FIRES
                   and row["decomp_n_decomposed_precommit"] >= MIN_DECOMPOSED)
    else:  # ARM_ON_DEPTHCAP
        cell_ok = (row["decomp_n_marked_unreliable"] >= MIN_MARKED_UNRELIABLE
                   and row["decomp_n_decomposed_precommit"] == 0)
    print(f"verdict: {'PASS' if cell_ok else 'FAIL'}", flush=True)
    return row


def run_experiment() -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for arm in ARMS:
        for seed in SEEDS:
            rows.append(_run_cell(arm, seed))

    by_arm = {a: [r for r in rows if r["arm"] == a] for a in ARMS}

    def frac(arm: str, pred) -> float:
        cells = by_arm[arm]
        return (sum(1 for r in cells if pred(r)) / len(cells)) if cells else 0.0

    # --- Manipulation-validity checks (arithmetic; NOT claim weight) ---
    mc_vs_low = all(r["decomp_n_vs_trigger"] >= 1 for r in by_arm["ARM_ON_LOWVS"])
    mc_vs_high = all(r["decomp_n_vs_trigger"] == 0 for r in by_arm["ARM_ON_HIGHVS"])

    # --- Load-bearing (genuine, non-forced): HIGHVS boundary path ---
    c_boundary_live = frac(
        "ARM_ON_HIGHVS",
        lambda r: (r["decomp_n_vs_trigger"] == 0
                   and r["decomp_n_boundary_fires"] >= MIN_BOUNDARY_FIRES
                   and r["decomp_n_decomposed_precommit"] >= MIN_DECOMPOSED),
    ) >= SEED_PASS_FRACTION

    # --- Supporting confirming criteria ---
    c_decomp_fires = frac(
        "ARM_ON_LOWVS",
        lambda r: (r["decomp_n_vs_trigger"] >= 1
                   and r["decomp_n_decomposed_precommit"] >= MIN_DECOMPOSED),
    ) >= SEED_PASS_FRACTION

    c_grain = frac(
        "ARM_ON_LOWVS",
        lambda r: r["grain_tiles_from_seeded_chunk"] >= GRAIN_TILES_MIN,
    ) >= SEED_PASS_FRACTION

    c_depthcap = frac(
        "ARM_ON_DEPTHCAP",
        lambda r: (r["decomp_n_marked_unreliable"] >= MIN_MARKED_UNRELIABLE
                   and r["decomp_n_decomposed_precommit"] == 0),
    ) >= SEED_PASS_FRACTION

    c_off = all(
        (not r["decomposition_instantiated"])
        and r["decomp_n_evaluated_precommit"] == 0
        and r["decomp_n_decomposed_precommit"] == 0
        and r["decomp_n_vs_trigger"] == 0
        and r["decomp_n_boundary_fires"] == 0
        for r in by_arm["ARM_OFF"]
    )

    # --- Non-degeneracy: the genuine boundary measurement must actually engage.
    highvs_boundary_total = sum(r["decomp_n_boundary_fires"] for r in by_arm["ARM_ON_HIGHVS"])
    non_degenerate = highvs_boundary_total >= 1

    overall_pass = (non_degenerate and c_boundary_live and c_decomp_fires
                    and c_grain and c_depthcap and c_off)

    if not non_degenerate:
        label = "boundary_path_measurement_starved"
        direction = "unknown"
    elif overall_pass:
        label = "arc070_decomposition_operation_confirmed"
        direction = "supports"
    elif not c_boundary_live:
        label = "boundary_trigger_not_applied"
        direction = "weakens"
    else:
        label = "decomposition_operation_partial"
        direction = "mixed"

    metrics = {
        "c_boundary_live": c_boundary_live,
        "c_decomp_fires": c_decomp_fires,
        "c_grain": c_grain,
        "c_depthcap": c_depthcap,
        "c_off": c_off,
        "mc_vs_low_ok": mc_vs_low,
        "mc_vs_high_ok": mc_vs_high,
        "highvs_boundary_fires_total": int(highvs_boundary_total),
        "highvs_decomposed_total": sum(
            r["decomp_n_decomposed_precommit"] for r in by_arm["ARM_ON_HIGHVS"]),
        "lowvs_decomposed_total": sum(
            r["decomp_n_decomposed_precommit"] for r in by_arm["ARM_ON_LOWVS"]),
        "lowvs_grain_tiles_mean": statistics.fmean(
            [r["grain_tiles_from_seeded_chunk"] for r in by_arm["ARM_ON_LOWVS"]]),
        "depthcap_marked_unreliable_total": sum(
            r["decomp_n_marked_unreliable"] for r in by_arm["ARM_ON_DEPTHCAP"]),
        "depthcap_decomposed_total": sum(
            r["decomp_n_decomposed_precommit"] for r in by_arm["ARM_ON_DEPTHCAP"]),
        "off_evaluated_total": sum(
            r["decomp_n_evaluated_precommit"] for r in by_arm["ARM_OFF"]),
    }

    interpretation = {
        "label": label,
        "manipulation_checks": {
            # Arithmetically forced by the injected region_vs -- validity, not evidence.
            "vs_trigger_fires_when_region_vs_below_threshold": mc_vs_low,
            "vs_trigger_silent_when_region_vs_above_threshold": mc_vs_high,
            "note": ("decomp_n_vs_trigger == (region_vs < threshold) is a tautology under "
                     "an injected region_vs; recorded as a manipulation check, carries no "
                     "claim weight (DV-symmetry discipline, failure_autopsy_V3-EXQ-604c)."),
        },
        "preconditions": [
            {
                "name": "highvs_boundary_path_engaged",
                "description": ("The one non-forced load-bearing measurement: the real "
                                "MECH-288 detector must fire on genuine rollout latents in "
                                "ARM_ON_HIGHVS. Below floor means the measurement is "
                                "STARVED (rollout too short), not that ARC-070 is falsified."),
                "measured": float(highvs_boundary_total),
                "threshold": 1.0,
                "direction": "lower",
                "control": ("total decomp_n_boundary_fires across ARM_ON_HIGHVS cells; "
                            "probe on this config gave 16 fires in 8 episodes"),
                "met": non_degenerate,
            },
        ],
        "criteria": [
            {"name": "C_BOUNDARY_LIVE_real_mech288_drives_decomp", "load_bearing": True,
             "passed": c_boundary_live},
            {"name": "C_DECOMP_FIRES_vs_trigger_drives_decomp", "load_bearing": False,
             "passed": c_decomp_fires},
            {"name": "C_GRAIN_decompose_yields_finer_tiles", "load_bearing": False,
             "passed": c_grain},
            {"name": "C_DEPTHCAP_at_cap_marked_unreliable", "load_bearing": False,
             "passed": c_depthcap},
            {"name": "C_OFF_structural_zero", "load_bearing": False, "passed": c_off},
        ],
        "criteria_non_degenerate": {
            "C_BOUNDARY_LIVE": non_degenerate,
            "C_DECOMP_FIRES": non_degenerate,
            "C_GRAIN": non_degenerate,
            "C_DEPTHCAP": non_degenerate,
            # Structural inertness assertion -- meaningful regardless of firing.
            "C_OFF": True,
        },
        "scope_note": ("Confirms ARC-070's OPERATION half (given a trigger, the proposer "
                       "re-segments into finer primitives, selectively). Does NOT confirm "
                       "that the trigger fires under NATURAL prediction failure in the live "
                       "env -- that is env-blocked by V_s saturation (816/830) and is the "
                       "separate behavioural MECH-321 line."),
    }

    return {
        "outcome": "PASS" if overall_pass else "FAIL",
        "evidence_direction": direction,
        "metrics": metrics,
        "per_seed_rows": rows,
        "arm_results": rows,
        "interpretation": interpretation,
        "non_degenerate": non_degenerate,
        "degeneracy_reason": (
            None if non_degenerate
            else ("ARM_ON_HIGHVS produced zero MECH-288 boundary fires across all seeds, so "
                  "the sole non-forced load-bearing measurement was starved; requeue with a "
                  "longer rollout rather than reading this as a falsification")
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    t0 = time.perf_counter()

    global SEEDS, N_EPISODES
    if args.dry_run:
        SEEDS = [11]
        N_EPISODES = 6

    result = run_experiment()
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    full_config = {
        "seeds": SEEDS, "arms": ARMS, "n_episodes": N_EPISODES,
        "steps_per_episode": STEPS_PER_EPISODE,
        "decomposition_vs_threshold": VS_THRESHOLD,
        "depth_cap_default": DEPTH_CAP_DEFAULT,
        "depth_cap_degenerate": DEPTH_CAP_DEGENERATE,
        "region_vs_low": REGION_VS_LOW,
        "region_vs_high": REGION_VS_HIGH,
        "seeded_chunk_sequence": list(SEEDED_CHUNK_SEQUENCE),
        "seed_pass_fraction": SEED_PASS_FRACTION,
        "grain_tiles_min": GRAIN_TILES_MIN,
        "arm_config_slices": {a: _config_slice(a) for a in ARMS},
    }
    manifest = {
        "run_id": f"{EXPERIMENT_TYPE}_{ts}_v3",
        "experiment_type": EXPERIMENT_TYPE,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "evidence_direction": result["evidence_direction"],
        "outcome": result["outcome"],
        "timestamp_utc": ts,
        "metrics": result["metrics"],
        "per_seed_rows": result["per_seed_rows"],
        "arm_results": result["arm_results"],
        "interpretation": result["interpretation"],
        "non_degenerate": result["non_degenerate"],
        "degeneracy_reason": result["degeneracy_reason"],
    }
    out_path = write_flat_manifest(
        manifest,
        Path(__file__).resolve().parents[2] / "REE_assembly" / "evidence" / "experiments",
        dry_run=args.dry_run,
        config=full_config,
        seeds=SEEDS,
        script_path=Path(__file__),
        started_at=t0,
        agent=_LAST_ON_AGENT,  # records z_goal-stream stats + enabled default-off flags
    )
    m = result["metrics"]
    print(f"outcome: {result['outcome']}", flush=True)
    print(f"direction: {result['evidence_direction']}", flush=True)
    print(f"label: {result['interpretation']['label']}", flush=True)
    print(f"C_BOUNDARY_LIVE={m['c_boundary_live']} C_DECOMP_FIRES={m['c_decomp_fires']} "
          f"C_GRAIN={m['c_grain']} C_DEPTHCAP={m['c_depthcap']} C_OFF={m['c_off']}", flush=True)
    print(f"HIGHVS boundary_fires={m['highvs_boundary_fires_total']} "
          f"decomposed={m['highvs_decomposed_total']} | LOWVS decomposed={m['lowvs_decomposed_total']} "
          f"grain_mean={m['lowvs_grain_tiles_mean']:.2f} | DEPTHCAP marked={m['depthcap_marked_unreliable_total']}",
          flush=True)
    print(f"manip_check vs_low={m['mc_vs_low_ok']} vs_high={m['mc_vs_high_ok']}", flush=True)
    print(f"wrote: {out_path}", flush=True)
    return result, out_path, args.dry_run


if __name__ == "__main__":
    _result, _out_path, _dry_run = main()
    _outcome_raw = str(_result["outcome"]).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=str(_out_path),
        dry_run=_dry_run,
    )
