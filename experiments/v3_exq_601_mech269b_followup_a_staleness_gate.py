#!/opt/local/bin/python3
"""
V3-EXQ-601 -- MECH-269b-followup-A staleness-corrected VsRolloutGate diagnostic.

MECH-269b-followup-A (substrate_queue): wire MECH-284 region staleness into
VsRolloutGate threshold comparison (effective_vs = raw_vs - staleness[s]),
mirroring AnchorSet.tick_hysteresis when use_mech284_hysteresis is on.

Substrate landed 2026-04-29 (use_vs_gate_staleness_lookup). This experiment
validates that wiring at realistic gate thresholds (0.4 / 0.5 defaults),
without the smoke overrides used by V3-EXQ-490b (0.85 / 0.95).

Arms (2 x 3 seeds):
  LOOKUP_OFF -- use_vs_rollout_gating=True, use_vs_gate_staleness_lookup=False
  LOOKUP_ON  -- use_vs_rollout_gating=True, use_vs_gate_staleness_lookup=True

Both arms: full V_s invalidation circuit ON (per_stream_vs, anchor_sets,
staleness_accumulator, mech284_hysteresis, event_segmenter, invalidation).

Pre-registered acceptance (substrate readiness):
  C1: LOOKUP_ON arm registers vs_gate_staleness_lookup_calls > 0 in >= 2/3 seeds.
  C2: LOOKUP_ON (vs_gate_total_held_e1 + vs_gate_total_held_e2) strictly exceeds
      LOOKUP_OFF totals in >= 2/3 seeds (staleness severance).

claim_ids: [MECH-269b]
experiment_purpose: diagnostic
supersedes: (none; complements V3-EXQ-490b smoke-threshold probe)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig
from experiment_protocol import emit_outcome
from experiments.pack_writer import write_flat_manifest  # noqa: E402


EXPERIMENT_TYPE = "v3_exq_601_mech269b_followup_a_staleness_gate"
QUEUE_ID = "V3-EXQ-601"
CLAIM_IDS = ["MECH-269b"]
EXPERIMENT_PURPOSE = "diagnostic"

SEEDS = [42, 7, 19]
ARMS = [
    {"id": "LOOKUP_OFF", "use_vs_gate_staleness_lookup": False},
    {"id": "LOOKUP_ON", "use_vs_gate_staleness_lookup": True},
]

ENV_KWARGS = dict(
    size=10,
    num_hazards=3,
    num_resources=5,
    hazard_harm=0.05,
    env_drift_interval=4,
    env_drift_prob=0.15,
    proximity_harm_scale=0.1,
    proximity_benefit_scale=0.05,
    proximity_approach_threshold=0.2,
    hazard_field_decay=0.5,
    resource_respawn_on_consume=True,
    use_proxy_fields=True,
    toroidal=False,
    harm_history_len=10,
    limb_damage_enabled=True,
    damage_increment=0.15,
    failure_prob_scale=0.3,
    heal_rate=0.002,
    n_landmarks_b=2,
)

WARMUP_EPISODES = 40
EVAL_EPISODES = 3
STEPS_PER_EPISODE = 150

WORLD_DIM = 32
SELF_DIM = 32
HARM_DIM = 32
HARM_A_DIM = 16
HARM_HISTORY_LEN = 10

WF_BUF_MAX = 1500
HARM_EVAL_BUF_MAX = 1500
BATCH_SIZE = 32
LR_E1 = 1e-4
LR_E2_WF = 3e-4
LR_E3_HARM = 1e-3
def _action_to_onehot(idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, idx] = 1.0
    return v


def _make_agent_and_env(seed: int, arm: Dict) -> Tuple[REEAgent, CausalGridWorldV2]:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    env = CausalGridWorldV2(seed=seed, **ENV_KWARGS)
    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
        harm_dim=HARM_DIM,
        alpha_world=0.9,
        alpha_self=0.3,
        reafference_action_dim=env.action_dim,
        use_harm_stream=True,
        z_harm_dim=HARM_DIM,
        use_affective_harm_stream=True,
        z_harm_a_dim=HARM_A_DIM,
        harm_history_len=HARM_HISTORY_LEN,
        use_resource_proximity_head=True,
        resource_proximity_weight=0.5,
        benefit_eval_enabled=True,
        benefit_weight=1.0,
        z_goal_enabled=True,
        goal_weight=0.5,
        drive_weight=2.0,
        e1_goal_conditioned=True,
        limb_damage_enabled=True,
        damage_increment=0.15,
        failure_prob_scale=0.3,
        heal_rate=0.002,
        use_gabaergic_decay=True,
        use_pag_freeze_gate=True,
        use_broadcast_override=True,
        use_dacc=True,
        use_e2_harm_a=True,
        use_per_stream_vs=True,
        use_event_segmenter=True,
        use_invalidation_trigger=True,
        use_anchor_sets=True,
        use_per_region_vs=True,
        use_staleness_accumulator=True,
        use_mech284_hysteresis=True,
        use_vs_commit_release=True,
        use_vs_rollout_gating=True,
        use_vs_gate_staleness_lookup=arm["use_vs_gate_staleness_lookup"],
    )
    config.e3.commitment_threshold = 0.5
    config.heartbeat.beta_gate_bistable = True
    return REEAgent(config), env


def _warmup_train(agent: REEAgent, env: CausalGridWorldV2, num_episodes: int, steps: int) -> None:
    device = agent.device
    action_dim = env.action_dim
    e1_optimizer = optim.Adam(agent.e1.parameters(), lr=LR_E1)
    e2_wf_optimizer = optim.Adam(
        list(agent.e2.world_transition.parameters())
        + list(agent.e2.world_action_encoder.parameters()),
        lr=LR_E2_WF,
    )
    harm_eval_optimizer = optim.Adam(agent.e3.harm_eval_head.parameters(), lr=LR_E3_HARM)
    wf_buf: List = []
    harm_eval_buf: List = []
    agent.train()

    for _ in range(num_episodes):
        _, obs_dict = env.reset()
        agent.reset()
        z_world_prev = None
        action_prev = None
        for _ in range(steps):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            latent = agent.sense(
                obs_body,
                obs_world,
                obs_harm=obs_dict.get("harm_obs"),
                obs_harm_a=obs_dict.get("harm_obs_a"),
                obs_harm_history=obs_dict.get("harm_history"),
            )
            z_world_curr = latent.z_world.detach()
            if z_world_prev is not None and action_prev is not None:
                agent.record_transition(z_world_prev, action_prev, latent.z_self.detach())
            ticks = agent.clock.advance()
            if ticks.get("e1_tick", False):
                agent._e1_tick(latent)
            candidates = agent.generate_trajectories(
                latent,
                torch.zeros(1, WORLD_DIM, device=device),
                ticks,
            )
            agent.update_z_goal(
                benefit_exposure=max(0.0, float(obs_dict.get("benefit_exposure", 0.0))),
                drive_level=REEAgent.compute_drive_level(obs_body),
            )
            action = agent.select_action(candidates, ticks, temperature=1.0)
            if action is None:
                action = _action_to_onehot(random.randint(0, action_dim - 1), action_dim, device)
            agent._last_action = action
            _, harm_signal, done, _, obs_dict = env.step(action)
            if z_world_prev is not None and action_prev is not None:
                wf_buf.append((z_world_prev.cpu(), action_prev.cpu(), z_world_curr.cpu()))
                if len(wf_buf) > WF_BUF_MAX:
                    wf_buf = wf_buf[-WF_BUF_MAX:]
            harm_target = abs(float(harm_signal)) if float(harm_signal) < 0 else 0.0
            harm_eval_buf.append((z_world_curr.cpu(), torch.tensor([harm_target])))
            if len(harm_eval_buf) > HARM_EVAL_BUF_MAX:
                harm_eval_buf = harm_eval_buf[-HARM_EVAL_BUF_MAX:]
            if len(wf_buf) >= BATCH_SIZE:
                idxs = torch.randperm(len(wf_buf))[:BATCH_SIZE].tolist()
                zw_b = torch.cat([wf_buf[i][0] for i in idxs]).to(device)
                a_b = torch.cat([wf_buf[i][1] for i in idxs]).to(device)
                zw1_b = torch.cat([wf_buf[i][2] for i in idxs]).to(device)
                wf_pred = agent.e2.world_forward(zw_b, a_b)
                wf_loss = F.mse_loss(wf_pred, zw1_b)
                if wf_loss.requires_grad:
                    e2_wf_optimizer.zero_grad()
                    wf_loss.backward()
                    e2_wf_optimizer.step()
            if len(harm_eval_buf) >= BATCH_SIZE:
                idxs = torch.randperm(len(harm_eval_buf))[:BATCH_SIZE].tolist()
                zw_b = torch.cat([harm_eval_buf[i][0] for i in idxs]).to(device)
                ht_b = torch.cat([harm_eval_buf[i][1] for i in idxs]).to(device)
                hp = agent.e3.harm_eval(zw_b)
                he_loss = F.mse_loss(hp.squeeze(), ht_b.squeeze())
                if he_loss.requires_grad:
                    harm_eval_optimizer.zero_grad()
                    he_loss.backward()
                    harm_eval_optimizer.step()
            if len(agent._world_experience_buffer) >= 2:
                e1_loss = agent.compute_prediction_loss()
                if e1_loss.requires_grad:
                    e1_optimizer.zero_grad()
                    e1_loss.backward()
                    e1_optimizer.step()
            z_world_prev = z_world_curr
            action_prev = action.detach()
            if done:
                break


def _eval_agent(agent: REEAgent, env: CausalGridWorldV2, num_episodes: int, steps: int) -> Dict:
    device = agent.device
    action_dim = env.action_dim
    agent.eval()
    total_held_e1 = 0
    total_held_e2 = 0
    staleness_calls = 0
    max_staleness_z_world = 0.0

    for _ in range(num_episodes):
        _, obs_dict = env.reset()
        agent.reset()
        z_world_prev = None
        action_prev = None
        for _ in range(steps):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            with torch.no_grad():
                latent = agent.sense(
                    obs_body,
                    obs_world,
                    obs_harm=obs_dict.get("harm_obs"),
                    obs_harm_a=obs_dict.get("harm_obs_a"),
                    obs_harm_history=obs_dict.get("harm_history"),
                )
                if z_world_prev is not None and action_prev is not None:
                    agent.record_transition(z_world_prev, action_prev, latent.z_self.detach())
                ticks = agent.clock.advance()
                if ticks.get("e1_tick", False):
                    agent._e1_tick(latent)
                candidates = agent.generate_trajectories(
                    latent,
                    torch.zeros(1, WORLD_DIM, device=device),
                    ticks,
                )
                agent.update_z_goal(
                    benefit_exposure=max(0.0, float(obs_dict.get("benefit_exposure", 0.0))),
                    drive_level=REEAgent.compute_drive_level(obs_body),
                )
                action = agent.select_action(candidates, ticks, temperature=1.0)
                if action is None:
                    action = _action_to_onehot(random.randint(0, action_dim - 1), action_dim, device)
                agent._last_action = action
            _, _, done, _, obs_dict = env.step(action)
            z_world_prev = latent.z_world.detach()
            action_prev = action.detach()
            if done:
                break

    if agent.vs_rollout_gate is not None:
        diag = agent.vs_rollout_gate.get_diagnostics()
        total_held_e1 = int(diag.get("vs_gate_total_held_e1", 0))
        total_held_e2 = int(diag.get("vs_gate_total_held_e2", 0))
        staleness_calls = int(diag.get("vs_gate_staleness_lookup_calls", 0))
        max_staleness_z_world = float(diag.get("vs_gate_max_staleness_z_world", 0.0))

    return {
        "vs_gate_total_held_e1": total_held_e1,
        "vs_gate_total_held_e2": total_held_e2,
        "vs_gate_total_held": total_held_e1 + total_held_e2,
        "vs_gate_staleness_lookup_calls": staleness_calls,
        "vs_gate_max_staleness_z_world": max_staleness_z_world,
    }


def _evaluate(per_seed: List[Dict]) -> Dict:
    """C1/C2 across seeds."""
    c1_hits = 0
    c2_hits = 0
    for row in per_seed:
        on = row.get("LOOKUP_ON", {})
        off = row.get("LOOKUP_OFF", {})
        if on.get("vs_gate_staleness_lookup_calls", 0) > 0:
            c1_hits += 1
        if on.get("vs_gate_total_held", 0) > off.get("vs_gate_total_held", 0):
            c2_hits += 1
    n = len(per_seed)
    need = max(2, (n + 1) // 2) if n >= 2 else 1
    c1 = c1_hits >= need
    c2 = c2_hits >= need
    return {
        "c1_staleness_lookup_calls_on": c1,
        "c2_on_held_gt_off": c2,
        "c1_hits": c1_hits,
        "c2_hits": c2_hits,
        "n_seeds": n,
        "pass": c1 and c2,
    }


def main(dry_run: bool = False) -> int:
    seeds = [SEEDS[0]] if dry_run else SEEDS
    warmup = 6 if dry_run else WARMUP_EPISODES
    eval_eps = 1 if dry_run else EVAL_EPISODES
    steps = 30 if dry_run else STEPS_PER_EPISODE

    print(f"[{EXPERIMENT_TYPE}] dry_run={dry_run} seeds={seeds}", flush=True)
    per_seed_rows: List[Dict] = []

    for seed in seeds:
        arm_metrics: Dict[str, Dict] = {}
        for arm in ARMS:
            agent, env = _make_agent_and_env(seed, arm)
            _warmup_train(agent, env, warmup, steps)
            arm_metrics[arm["id"]] = _eval_agent(agent, env, eval_eps, steps)
            print(
                f"  seed={seed} arm={arm['id']} "
                f"held={arm_metrics[arm['id']]['vs_gate_total_held']} "
                f"staleness_calls={arm_metrics[arm['id']]['vs_gate_staleness_lookup_calls']}",
                flush=True,
            )
        per_seed_rows.append(arm_metrics)

    acceptance = _evaluate(per_seed_rows)
    outcome = "PASS" if acceptance["pass"] else "FAIL"
    print(f"[{EXPERIMENT_TYPE}] acceptance={acceptance} outcome={outcome}", flush=True)

    if dry_run:
        print("DRY RUN OK", flush=True)
        return 0

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"
    out_dir = (
        Path(__file__).resolve().parents[2]
        / "REE_assembly"
        / "evidence"
        / "experiments"
        / EXPERIMENT_TYPE
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.json"
    manifest = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": ts,
        "outcome": outcome,
        "evidence_direction": "supports" if outcome == "PASS" else "weakens",
        "evidence_direction_per_claim": {
            "MECH-269b": "supports" if outcome == "PASS" else "weakens",
        },
        "acceptance": acceptance,
        "per_seed": per_seed_rows,
        "arms": [a["id"] for a in ARMS],
        "note": (
            "MECH-269b-followup-A substrate validation at default gate thresholds "
            "(0.4 hold / 0.5 refresh). Does not test Q-040b behavioural sufficiency."
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
    print(f"Result written to: {out_path}", flush=True)
    emit_outcome(outcome=outcome, manifest_path=out_path)
    return 0 if outcome == "PASS" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    sys.exit(main(dry_run=args.dry_run))
