#!/opt/local/bin/python3
"""
V3-EXQ-450 -- SD-032a Salience Coordinator Behavioral Confirmation.

Claims: SD-032a, MECH-259, MECH-261
Supersedes: V3-EXQ-446 (wiring-only diagnostic; untrained agent, zero action entropy)

V3-EXQ-446 PASS confirmed wiring: coordinator instantiates, operating_mode has non-zero
entropy under synthetic injection, write_gate values are in [0, 1]. But trigger_count_natural
was 0 (only synthetic pe=10 injection fired). The 6-episode untrained run gave action
entropy=0 in BOTH arms -- coordinator had no real signal to respond to.

This experiment adds phased training (P0 EMA warmup + P1 E2_harm_a) so the dACC bundle
produces calibrated PE signals. With a partially trained E2_harm_a, harm-approach events
produce real prediction errors that the coordinator can respond to.

Conditions (3 seeds x 2 conditions):
  COORD_OFF: use_dacc=True, use_e2_harm_a=True, use_salience_coordinator=False
  COORD_ON:  use_dacc=True, use_e2_harm_a=True, use_salience_coordinator=True

Phased training (per condition, per seed):
  P0 (50 eps): EMA warmup -- forward passes only, no optimizer. Calibrates
               z_harm_a EMA so dACC PE has a running baseline.
  P1 (100 eps): E2_harm_a training on frozen z_harm_a targets. After this,
                dACC PE reflects genuine prediction errors, not noise.
  P2 (50 eps): Evaluation. Collect coordinator metrics per step.

Metrics collected during P2 (COORD_ON):
  operating_mode_entropy_per_step  -- H(operating_mode) > 0 means mode mixing
  trigger_count_natural            -- times mode_switch_trigger fired without injection
  write_gate_e3_policy_per_step    -- gate value trajectory over P2
  mean_score_bias_abs              -- dACC bias magnitude (same as 445 diagnostic)
  action_counts                    -- action class distribution (entropy)

PASS criteria:
  C1 (MECH-259 trigger): trigger_count_natural > 0 in >=2/3 seeds for COORD_ON
                          (mode_switch_trigger fires on real dACC PE, not injection)
  C2 (SD-032a mode distribution): mean_operating_mode_entropy > 0.3 nats across
                          P2 steps in >=2/3 seeds for COORD_ON
                          (mode vector has genuine mixed distribution)
  C3 (MECH-261 gate modulation): write_gate("e3_policy") std_dev > 0.01 across
                          P2 steps in >=2/3 seeds for COORD_ON
                          (gate is modulated per step, not constant)
  C4 (compat check): COORD_OFF write_gate std_dev < 0.001 in all seeds
                          (no coordinator -> no gate modulation)

PASS = C1 AND C2 AND C3.

Note: C1 requires dACC PE to exceed coordinator salience_aggregate threshold (default 1.0).
With partially trained E2_harm_a, approach events may drive z_harm_a spikes above baseline.
If C1 FAILs despite C2/C3 PASSing, the diagnosis is: PE stays below threshold even with
training (likely environment doesn't generate enough harm variation). Recommend 450a with
stronger harm signal (higher hazard_harm=0.1) or lowered threshold (switch_threshold=0.5).

claim_ids: ["SD-032a", "MECH-259", "MECH-261"]
experiment_purpose: "evidence"
  Rationale: Directly tests whether coordinator responds to naturalistic dACC PE.
  PASS = coordinator is usable in trained-agent conditions (evidence for SD-032a,
  MECH-259, MECH-261). FAIL = PE insufficient under default environment (diagnostic gap).
"""

import sys
import json
import argparse
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_455_sd032a_salience_behavioral"
CLAIM_IDS = ["SD-032a", "MECH-259", "MECH-261"]
EXPERIMENT_PURPOSE = "evidence"

SEEDS = [42, 7, 13]
STEPS_PER_EP = 120
P0_EPS = 50
P1_EPS = 100
P2_EPS = 50

CONDITIONS = ["COORD_OFF", "COORD_ON"]


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed,
        size=10,
        num_hazards=3,
        num_resources=4,
        hazard_harm=0.06,
        proximity_harm_scale=0.15,
        proximity_benefit_scale=0.18,
        proximity_approach_threshold=0.15,
        hazard_field_decay=0.5,
        energy_decay=0.005,
        use_proxy_fields=True,
        resource_respawn_on_consume=True,
        harm_history_len=10,
    )


def _make_agent(env: CausalGridWorldV2, condition: str) -> REEAgent:
    use_coord = condition == "COORD_ON"
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=32,
        alpha_world=0.9,
        alpha_self=0.3,
        use_harm_stream=True,
        harm_obs_dim=51,
        use_affective_harm_stream=True,
        harm_obs_a_dim=50,
        harm_history_len=10,
        z_harm_a_dim=16,
        use_e2_harm_a=True,
        use_shared_harm_trunk=False,
        e2_harm_a_lr=5e-4,
        use_dacc=True,
        dacc_weight=0.5,
        dacc_interaction_weight=0.3,
        dacc_foraging_weight=0.2,
        dacc_suppression_weight=0.3,
        dacc_suppression_memory=8,
        dacc_precision_scale=5000.0,
        dacc_effort_cost=0.1,
        dacc_drive_coupling=0.0,
        use_salience_coordinator=use_coord,
        salience_switch_threshold=1.0,
        salience_stability_scaling=1.0,
        salience_softmax_temperature=1.0,
        salience_external_task_bias=1.0,
        salience_dacc_pe_weight=1.0,
        salience_dacc_foraging_weight=0.5,
        salience_apply_to_dacc_bias=False,
    )
    return REEAgent(cfg)


def _entropy(probs: Dict[str, float]) -> float:
    ent = 0.0
    for v in probs.values():
        if v > 1e-12:
            ent -= v * math.log(v)
    return ent


def _action_entropy(counts: Dict[int, int]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    ent = 0.0
    for c in counts.values():
        if c <= 0:
            continue
        p = c / total
        ent -= p * math.log(p)
    return ent


def _obs_tensors(obs_dict):
    body = obs_dict["body_state"].float().unsqueeze(0)
    world = obs_dict["world_state"].float().unsqueeze(0)
    harm = obs_dict["harm_obs"].float().unsqueeze(0) if "harm_obs" in obs_dict else None
    harm_a = obs_dict["harm_obs_a"].float().unsqueeze(0) if "harm_obs_a" in obs_dict else None
    harm_hist = obs_dict["harm_history"].float().unsqueeze(0) if "harm_history" in obs_dict else None
    return body, world, harm, harm_a, harm_hist


def _run_condition(seed: int, condition: str, verbose: bool = True) -> Dict:
    torch.manual_seed(seed)

    env = _make_env(seed)
    agent = _make_agent(env, condition)

    optim_e2_a = torch.optim.Adam(agent.e2_harm_a.parameters(), lr=5e-4)

    total_eps = P0_EPS + P1_EPS + P2_EPS
    phase_p1_start = P0_EPS
    phase_p2_start = P0_EPS + P1_EPS

    # P2 collection
    mode_entropy_per_step: List[float] = []
    gate_e3_per_step: List[float] = []
    trigger_count_natural: int = 0
    action_counts: Dict[int, int] = {}
    score_bias_abs_sum = 0.0
    score_bias_count = 0

    prev_z_harm_a: Optional[torch.Tensor] = None

    for ep_idx in range(total_eps):
        agent.reset()
        _obs, obs_dict = env.reset()

        phase_is_p1 = phase_p1_start <= ep_idx < phase_p2_start
        phase_is_p2 = ep_idx >= phase_p2_start

        for step in range(STEPS_PER_EP):
            body, world, harm, harm_a, harm_hist = _obs_tensors(obs_dict)
            latent = agent.sense(
                obs_body=body,
                obs_world=world,
                obs_harm=harm,
                obs_harm_a=harm_a,
                obs_harm_history=harm_hist,
            )
            ticks = agent.clock.advance()
            wdim = latent.z_world.shape[-1]
            e1_prior = (
                agent._e1_tick(latent) if ticks.get("e1_tick", False)
                else torch.zeros(1, wdim, device=agent.device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            action = agent.select_action(candidates, ticks)
            a_idx = int(action[0].argmax().item())

            # P1: train E2_harm_a
            if phase_is_p1 and prev_z_harm_a is not None and latent.z_harm_a is not None:
                z_prev_det = prev_z_harm_a.detach()
                z_next_det = latent.z_harm_a.detach()
                a_det = action.detach()
                z_pred = agent.e2_harm_a(z_prev_det, a_det)
                loss = agent.e2_harm_a.compute_loss(z_pred, z_next_det)
                optim_e2_a.zero_grad()
                loss.backward()
                optim_e2_a.step()

            # P2: collect coordinator metrics
            if phase_is_p2:
                action_counts[a_idx] = action_counts.get(a_idx, 0) + 1
                if agent._dacc_last_bias is not None:
                    score_bias_abs_sum += float(agent._dacc_last_bias.abs().mean().item())
                    score_bias_count += 1

                if agent.salience is not None:
                    tick_out = agent._salience_last_tick
                    if tick_out is not None:
                        mode_entropy = _entropy(tick_out["operating_mode"])
                        mode_entropy_per_step.append(mode_entropy)
                        if tick_out["mode_switch_trigger"]:
                            trigger_count_natural += 1
                    gate_val = agent.salience.write_gate("e3_policy")
                    gate_e3_per_step.append(gate_val)

            _obs, harm_signal, done, _info, obs_dict = env.step(action)
            prev_z_harm_a = (
                latent.z_harm_a.detach().clone() if latent.z_harm_a is not None else None
            )
            if done:
                break

    mean_mode_entropy = (
        sum(mode_entropy_per_step) / len(mode_entropy_per_step)
        if mode_entropy_per_step else 0.0
    )
    gate_std = 0.0
    if len(gate_e3_per_step) >= 2:
        mean_g = sum(gate_e3_per_step) / len(gate_e3_per_step)
        gate_std = math.sqrt(
            sum((g - mean_g) ** 2 for g in gate_e3_per_step) / len(gate_e3_per_step)
        )
    mean_score_bias_abs = (
        score_bias_abs_sum / score_bias_count if score_bias_count > 0 else 0.0
    )
    act_entropy = _action_entropy(action_counts)

    result = {
        "seed": seed,
        "condition": condition,
        "mean_operating_mode_entropy": float(mean_mode_entropy),
        "trigger_count_natural": int(trigger_count_natural),
        "write_gate_e3_policy_std": float(gate_std),
        "write_gate_e3_policy_mean": float(
            sum(gate_e3_per_step) / len(gate_e3_per_step) if gate_e3_per_step else 0.0
        ),
        "action_class_entropy": float(act_entropy),
        "mean_score_bias_abs": float(mean_score_bias_abs),
        "n_p2_steps": len(mode_entropy_per_step) if agent.salience is not None else len(action_counts),
        "action_counts": {str(k): v for k, v in action_counts.items()},
    }

    if verbose:
        print(
            f"  [seed={seed} {condition}] "
            f"mode_entropy={mean_mode_entropy:.3f} "
            f"trigger_natural={trigger_count_natural} "
            f"gate_e3_std={gate_std:.4f} "
            f"act_entropy={act_entropy:.3f} "
            f"bias_abs={mean_score_bias_abs:.2f}",
            flush=True,
        )

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    if args.dry_run:
        print("Smoke: seed=42, tiny P0=2/P1=3/P2=3 steps=20")
        global P0_EPS, P1_EPS, P2_EPS, STEPS_PER_EP
        P0_EPS, P1_EPS, P2_EPS, STEPS_PER_EP = 2, 3, 3, 20
        for cond in CONDITIONS:
            r = _run_condition(seed=42, condition=cond, verbose=True)
            assert r["mean_score_bias_abs"] >= 0.0
        print("Smoke test PASSED")
        return

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        script_dir = Path(__file__).resolve().parents[1]
        out_dir = (
            script_dir.parent / "REE_assembly" / "evidence"
            / "experiments" / EXPERIMENT_TYPE
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for seed in SEEDS:
        print(f"\nSeed {seed}", flush=True)
        for cond in CONDITIONS:
            print(f"  Running {cond}...", flush=True)
            r = _run_condition(seed=seed, condition=cond)
            all_results.append(r)

    def by_cond(c):
        return [r for r in all_results if r["condition"] == c]

    off_res = by_cond("COORD_OFF")
    on_res = by_cond("COORD_ON")

    # C1: trigger_count_natural > 0 in >=2/3 seeds for COORD_ON
    c1_wins = sum(1 for r in on_res if r["trigger_count_natural"] > 0)
    c1 = c1_wins >= 2

    # C2: mean_operating_mode_entropy > 0.3 nats in >=2/3 seeds for COORD_ON
    c2_wins = sum(1 for r in on_res if r["mean_operating_mode_entropy"] > 0.3)
    c2 = c2_wins >= 2

    # C3: write_gate_e3_policy_std > 0.01 in >=2/3 seeds for COORD_ON
    c3_wins = sum(1 for r in on_res if r["write_gate_e3_policy_std"] > 0.01)
    c3 = c3_wins >= 2

    # C4 compat: COORD_OFF gate_std < 0.001 (no coordinator -> no modulation)
    c4_wins = sum(1 for r in off_res if r["write_gate_e3_policy_std"] < 0.001)
    c4 = c4_wins == len(off_res)

    outcome = "PASS" if (c1 and c2 and c3) else "FAIL"

    summary = {
        "c1_mech259_trigger_natural": {
            "on_wins": c1_wins,
            "threshold": 2,
            "pass": c1,
            "desc": "trigger_count_natural > 0 in >=2/3 seeds for COORD_ON",
        },
        "c2_sd032a_mode_entropy": {
            "on_wins": c2_wins,
            "threshold": 2,
            "pass": c2,
            "desc": "mean_operating_mode_entropy > 0.3 nats in >=2/3 seeds for COORD_ON",
        },
        "c3_mech261_gate_modulation": {
            "on_wins": c3_wins,
            "threshold": 2,
            "pass": c3,
            "desc": "write_gate_e3_policy std > 0.01 in >=2/3 seeds for COORD_ON",
        },
        "c4_backward_compat": {
            "off_no_modulation_count": c4_wins,
            "pass": c4,
            "desc": "COORD_OFF gate std < 0.001 (no coordinator = no gate modulation)",
        },
    }

    per_claim = {
        "SD-032a": "supports" if (c2 and c3) else "weakens",
        "MECH-259": "supports" if c1 else "weakens",
        "MECH-261": "supports" if c3 else "weakens",
    }

    print(f"\nOutcome: {outcome}", flush=True)
    for k, v in summary.items():
        print(f"  {k}: {v}", flush=True)

    output = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": ts,
        "outcome": outcome,
        "evidence_direction": "supports" if outcome == "PASS" else "weakens",
        "evidence_direction_per_claim": per_claim,
        "supersedes": "v3_exq_446_sd032a_salience_coordinator",
        "pass_criteria_summary": summary,
        "per_seed_results": all_results,
        "config": {
            "conditions": CONDITIONS,
            "seeds": SEEDS,
            "p0_eps": P0_EPS,
            "p1_eps": P1_EPS,
            "p2_eps": P2_EPS,
            "steps_per_ep": STEPS_PER_EP,
        },
    }

    out_file = out_dir / f"{run_id}.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nOutput written to: {out_file}", flush=True)


if __name__ == "__main__":
    main()
