"""
V3-EXQ-004 — ARC-021 Three-Loop Incommensurability

Claim: ARC-021 / MECH-069 — E1 (sensory prediction), E2 (motor-sensory), and E3
(harm/goal) error signals are incommensurable. Collapsing them into a single loss
misattributes credit and degrades E3 harm evaluation.

Experimental logic:
  Two training conditions, identical warmup (300 episodes, RANDOM policy, 8 hazards):

  SEPARATE: Three separate optimizers:
    - opt_e12: Adam(E1+E2 params, lr=1e-3)  — sensory prediction + motor-sensory
    - opt_e3:  Adam(E3 harm_eval params, lr=1e-4) — harm/goal evaluation
    Gradients from E3 harm loss cannot contaminate E1/E2, and vice versa.

  COLLAPSED: Single optimizer, combined loss:
    - opt_all: Adam(all params, lr=1e-3)
    - loss = E1_loss + E2_loss + lambda_e3 * E3_harm_loss  (lambda_e3 = 0.1)
    E3 harm gradients flow through all parameters simultaneously. Credit from
    harm events may be assigned to E1/E2 weights rather than E3.harm_eval.

After warmup: probe-based calibration eval (10 grid resets).
  near_hazard probes: agent 1 step from hazard, toward-hazard action
  safe probes: agent > 3 Manhattan dist from all hazards

calibration_gap = mean(causal_sig | near_hazard) - mean(causal_sig | safe)

ARC-021 prediction: SEPARATE >> COLLAPSED in calibration_gap.
Collapsing forces E3 to compete with E1/E2 for gradient bandwidth at each step.
With imbalanced harm events (~12% rate), the collapsed optimizer spends most steps
adjusting E3 weights on E1/E2-only gradients, corrupting harm_eval.

PASS criteria (ALL must hold):
  C1: SEPARATE calibration_gap > 0.05
  C2: SEPARATE calibration_gap > COLLAPSED calibration_gap (separation helps)
  C3: COLLAPSED calibration_gap < SEPARATE * 0.7 OR COLLAPSED harm_eval degenerate
  C4: warmup harm events > 100 for SEPARATE condition
  C5: fatal_error_count == 0
  C6: n_probes >= 10 each for both conditions
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from typing import Dict, List, Tuple
import random

import torch
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorld
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_004_arc021_incommensurability"
CLAIM_IDS = ["ARC-021", "MECH-069"]

CONDITION_SEPARATE  = "SEPARATE"
CONDITION_COLLAPSED = "COLLAPSED"

LAMBDA_E3 = 0.1  # scale factor for E3 loss in collapsed condition


def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


def _random_cf_action(actual_idx: int, num_actions: int) -> int:
    choices = [a for a in range(num_actions) if a != actual_idx]
    return random.choice(choices) if choices else 0


# ---------------------------------------------------------------------------
# SEPARATE condition training
# ---------------------------------------------------------------------------

def _train_separate(
    agent: REEAgent,
    env: CausalGridWorld,
    opt_e12: optim.Optimizer,
    opt_e3: optim.Optimizer,
    num_episodes: int,
    steps_per_episode: int,
) -> Dict[str, float]:
    """Separate E1+E2 and E3 optimizers — reference condition (as in EXQ-002r3)."""
    agent.train()

    world_buf: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    harm_buf:    List[torch.Tensor] = []
    no_harm_buf: List[torch.Tensor] = []
    total_harm = 0

    for ep in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        z_world_prev = None
        action_prev  = None

        for step in range(steps_per_episode):
            latent = agent.sense(obs_dict["body_state"], obs_dict["world_state"])
            agent.clock.advance()

            if z_world_prev is not None and action_prev is not None:
                world_buf.append((z_world_prev.detach(), action_prev.detach(), latent.z_world.detach()))
                if len(world_buf) > 500:
                    world_buf = world_buf[-500:]

            action_idx = random.randint(0, env.action_dim - 1)
            action = _action_to_onehot(action_idx, env.action_dim, agent.device)
            agent._last_action = action
            z_world_prev = latent.z_world.detach()
            action_prev  = action.detach()

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)

            if harm_signal < 0:
                total_harm += 1
                harm_buf.append(latent.z_world.detach())
                if len(harm_buf) > 500: harm_buf = harm_buf[-500:]
            else:
                if step % 3 == 0:
                    no_harm_buf.append(latent.z_world.detach())
                    if len(no_harm_buf) > 500: no_harm_buf = no_harm_buf[-500:]

            # E1 + E2 backward (separate)
            e1_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()
            e2_world_loss = e1_loss.new_zeros(())
            if len(world_buf) >= 4:
                n = min(16, len(world_buf))
                idxs = torch.randperm(len(world_buf))[:n].tolist()
                zw_t, acts, zw_t1 = zip(*[world_buf[i] for i in idxs])
                e2_world_loss = F.mse_loss(
                    agent.e2.world_forward(torch.cat(zw_t), torch.cat(acts)),
                    torch.cat(zw_t1)
                )
            e12_total = e1_loss + e2_loss + e2_world_loss
            if e12_total.requires_grad:
                opt_e12.zero_grad()
                e12_total.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                opt_e12.step()

            # E3 backward (separate, lower lr)
            n_h, n_nh = len(harm_buf), len(no_harm_buf)
            if n_h >= 4 and n_nh >= 4:
                k = min(16, n_h, n_nh)
                zw_h  = torch.cat([harm_buf[i]    for i in torch.randperm(n_h)[:k].tolist()])
                zw_nh = torch.cat([no_harm_buf[i] for i in torch.randperm(n_nh)[:k].tolist()])
                labels = torch.cat([torch.ones(k, 1, device=agent.device),
                                    torch.zeros(k, 1, device=agent.device)])
                pred = agent.e3.harm_eval(torch.cat([zw_h, zw_nh]))
                if not torch.isnan(pred).any():
                    e3_loss = F.binary_cross_entropy(pred.clamp(1e-6, 1-1e-6), labels)
                    opt_e3.zero_grad()
                    e3_loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.e3.harm_eval_head.parameters(), 0.5)
                    opt_e3.step()

            if done:
                break

        if (ep + 1) % 50 == 0 or ep == num_episodes - 1:
            print(f"  [SEPARATE ep {ep+1}/{num_episodes}] "
                  f"harm={total_harm} harm_buf={len(harm_buf)}", flush=True)

    return {"total_harm_events": total_harm}


# ---------------------------------------------------------------------------
# COLLAPSED condition training
# ---------------------------------------------------------------------------

def _train_collapsed(
    agent: REEAgent,
    env: CausalGridWorld,
    opt_all: optim.Optimizer,
    num_episodes: int,
    steps_per_episode: int,
    lambda_e3: float = LAMBDA_E3,
) -> Dict[str, float]:
    """Single optimizer combining E1 + E2 + E3 harm losses — collapsed condition."""
    agent.train()

    world_buf: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    harm_buf:    List[torch.Tensor] = []
    no_harm_buf: List[torch.Tensor] = []
    total_harm = 0

    for ep in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        z_world_prev = None
        action_prev  = None

        for step in range(steps_per_episode):
            latent = agent.sense(obs_dict["body_state"], obs_dict["world_state"])
            agent.clock.advance()

            if z_world_prev is not None and action_prev is not None:
                world_buf.append((z_world_prev.detach(), action_prev.detach(), latent.z_world.detach()))
                if len(world_buf) > 500: world_buf = world_buf[-500:]

            action_idx = random.randint(0, env.action_dim - 1)
            action = _action_to_onehot(action_idx, env.action_dim, agent.device)
            agent._last_action = action
            z_world_prev = latent.z_world.detach()
            action_prev  = action.detach()

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)

            if harm_signal < 0:
                total_harm += 1
                harm_buf.append(latent.z_world.detach())
                if len(harm_buf) > 500: harm_buf = harm_buf[-500:]
            else:
                if step % 3 == 0:
                    no_harm_buf.append(latent.z_world.detach())
                    if len(no_harm_buf) > 500: no_harm_buf = no_harm_buf[-500:]

            # COLLAPSED: all losses in one backward pass
            e1_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()
            e2_world_loss = e1_loss.new_zeros(())
            if len(world_buf) >= 4:
                n = min(16, len(world_buf))
                idxs = torch.randperm(len(world_buf))[:n].tolist()
                zw_t, acts, zw_t1 = zip(*[world_buf[i] for i in idxs])
                e2_world_loss = F.mse_loss(
                    agent.e2.world_forward(torch.cat(zw_t), torch.cat(acts)),
                    torch.cat(zw_t1)
                )

            total_loss = e1_loss + e2_loss + e2_world_loss

            # Add E3 harm loss to the same graph
            n_h, n_nh = len(harm_buf), len(no_harm_buf)
            if n_h >= 4 and n_nh >= 4:
                k = min(16, n_h, n_nh)
                zw_h  = torch.cat([harm_buf[i]    for i in torch.randperm(n_h)[:k].tolist()])
                zw_nh = torch.cat([no_harm_buf[i] for i in torch.randperm(n_nh)[:k].tolist()])
                labels = torch.cat([torch.ones(k, 1, device=agent.device),
                                    torch.zeros(k, 1, device=agent.device)])
                pred = agent.e3.harm_eval(torch.cat([zw_h, zw_nh]))
                if not torch.isnan(pred).any():
                    e3_loss = F.binary_cross_entropy(pred.clamp(1e-6, 1-1e-6), labels)
                    total_loss = total_loss + lambda_e3 * e3_loss

            if total_loss.requires_grad:
                opt_all.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                opt_all.step()

            if done:
                break

        if (ep + 1) % 50 == 0 or ep == num_episodes - 1:
            print(f"  [COLLAPSED ep {ep+1}/{num_episodes}] "
                  f"harm={total_harm} harm_buf={len(harm_buf)}", flush=True)

    return {"total_harm_events": total_harm}


# ---------------------------------------------------------------------------
# Probe-based evaluation (shared)
# ---------------------------------------------------------------------------

def _eval_probes(
    agent: REEAgent,
    env: CausalGridWorld,
    num_resets: int,
    condition_label: str,
) -> Dict:
    agent.eval()
    near_sigs: List[float] = []
    safe_sigs:  List[float] = []
    fatal_errors = 0

    wall_type   = env.ENTITY_TYPES["wall"]
    hazard_type = env.ENTITY_TYPES["hazard"]

    def _run_probe(ax: int, ay: int, actual_idx: int) -> float:
        env.agent_x = ax
        env.agent_y = ay
        obs_dict = env._get_observation_dict()
        with torch.no_grad():
            latent  = agent.sense(obs_dict["body_state"], obs_dict["world_state"])
            z_world = latent.z_world
            cf_idx  = _random_cf_action(actual_idx, env.action_dim)
            a_act = _action_to_onehot(actual_idx, env.action_dim, agent.device)
            a_cf  = _action_to_onehot(cf_idx,     env.action_dim, agent.device)
            h_act = torch.nan_to_num(agent.e3.harm_eval(agent.e2.world_forward(z_world, a_act)), nan=0.5)
            h_cf  = torch.nan_to_num(agent.e3.harm_eval(agent.e2.world_forward(z_world, a_cf)),  nan=0.5)
            return float((h_act - h_cf).item())

    try:
        for _ in range(num_resets):
            env.reset()
            for hx, hy in env.hazards:
                for action_idx, (dx, dy) in env.ACTIONS.items():
                    if action_idx == 4:
                        continue
                    ax, ay = hx - dx, hy - dy
                    if 0 <= ax < env.size and 0 <= ay < env.size:
                        if int(env.grid[ax, ay]) not in (wall_type, hazard_type):
                            near_sigs.append(_run_probe(ax, ay, action_idx))
            for px in range(env.size):
                for py in range(env.size):
                    if int(env.grid[px, py]) in (wall_type, hazard_type):
                        continue
                    if min(abs(px-hx)+abs(py-hy) for hx, hy in env.hazards) > 3:
                        safe_sigs.append(_run_probe(px, py, random.randint(0, 3)))
    except Exception:
        import traceback
        fatal_errors += 1
        print(f"  FATAL [{condition_label}]: {traceback.format_exc()}", flush=True)

    mean_near = float(sum(near_sigs) / max(1, len(near_sigs)))
    mean_safe = float(sum(safe_sigs)  / max(1, len(safe_sigs)))
    gap = mean_near - mean_safe
    degenerate = abs(mean_near) < 1e-6 and abs(mean_safe) < 1e-6

    print(f"  [{condition_label}] near={len(near_sigs)} safe={len(safe_sigs)}  "
          f"gap={gap:.4f}  {'[DEGENERATE]' if degenerate else ''}", flush=True)

    return {
        "condition": condition_label,
        "calibration_gap": gap,
        "mean_near": mean_near,
        "mean_safe": mean_safe,
        "n_near": len(near_sigs),
        "n_safe": len(safe_sigs),
        "harm_eval_degenerate": degenerate,
        "fatal_errors": fatal_errors,
    }


# ---------------------------------------------------------------------------
# Main run()
# ---------------------------------------------------------------------------

def run(
    seed: int = 0,
    warmup_episodes: int = 300,
    eval_probe_resets: int = 10,
    steps_per_episode: int = 200,
    self_dim: int = 32,
    world_dim: int = 32,
    lr: float = 1e-3,
    **kwargs,
) -> dict:
    torch.manual_seed(seed)
    random.seed(seed)

    env = CausalGridWorld(seed=seed, num_hazards=8)
    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=self_dim,
        world_dim=world_dim,
    )

    fatal_errors = 0
    results = {}
    warmup_harm_separate = 0

    # ------------------------------------------------------------------ #
    # CONDITION: SEPARATE                                                  #
    # ------------------------------------------------------------------ #
    print(f"\n[V3-EXQ-004] Seed {seed} Condition SEPARATE", flush=True)
    agent_sep = REEAgent(config)
    # Exclude harm_eval_head from E12 optimizer — two Adam optimisers on the
    # same weights corrupt each other's momentum/variance estimates.
    e12_params = [p for n, p in agent_sep.named_parameters() if "harm_eval" not in n]
    opt_e12 = optim.Adam(e12_params, lr=lr)
    opt_e3  = optim.Adam(agent_sep.e3.harm_eval_head.parameters(), lr=1e-4)
    print(f"  Warmup: {warmup_episodes} episodes (RANDOM policy, SEPARATE optimizers) ...", flush=True)
    m = _train_separate(agent_sep, env, opt_e12, opt_e3, warmup_episodes, steps_per_episode)
    warmup_harm_separate = m["total_harm_events"]
    print(f"  Probe eval ...", flush=True)
    r_sep = _eval_probes(agent_sep, env, eval_probe_resets, CONDITION_SEPARATE)
    results[CONDITION_SEPARATE] = r_sep
    fatal_errors += r_sep["fatal_errors"]

    # ------------------------------------------------------------------ #
    # CONDITION: COLLAPSED                                                 #
    # ------------------------------------------------------------------ #
    print(f"\n[V3-EXQ-004] Seed {seed} Condition COLLAPSED", flush=True)
    torch.manual_seed(seed + 5000)
    random.seed(seed + 5000)
    agent_col = REEAgent(config)
    opt_all = optim.Adam(agent_col.parameters(), lr=lr)
    print(f"  Warmup: {warmup_episodes} episodes (RANDOM policy, COLLAPSED optimizer, lambda_e3={LAMBDA_E3}) ...", flush=True)
    m2 = _train_collapsed(agent_col, env, opt_all, warmup_episodes, steps_per_episode)
    warmup_harm_collapsed = m2["total_harm_events"]
    print(f"  Probe eval ...", flush=True)
    r_col = _eval_probes(agent_col, env, eval_probe_resets, CONDITION_COLLAPSED)
    results[CONDITION_COLLAPSED] = r_col
    fatal_errors += r_col["fatal_errors"]

    # ------------------------------------------------------------------ #
    # PASS / FAIL                                                          #
    # ------------------------------------------------------------------ #
    gap_sep = results[CONDITION_SEPARATE]["calibration_gap"]
    gap_col = results[CONDITION_COLLAPSED]["calibration_gap"]
    n_near  = results[CONDITION_SEPARATE]["n_near"]
    n_safe  = results[CONDITION_SEPARATE]["n_safe"]
    col_deg = results[CONDITION_COLLAPSED]["harm_eval_degenerate"]

    crit1_pass = gap_sep > 0.05
    crit2_pass = gap_sep > gap_col
    crit3_pass = (gap_col < gap_sep * 0.7) or col_deg
    crit4_pass = warmup_harm_separate > 100
    crit5_pass = fatal_errors == 0
    crit6_pass = n_near >= 10 and n_safe >= 10

    all_pass = all([crit1_pass, crit2_pass, crit3_pass, crit4_pass, crit5_pass, crit6_pass])
    status = "PASS" if all_pass else "FAIL"
    criteria_met = sum([crit1_pass, crit2_pass, crit3_pass, crit4_pass, crit5_pass, crit6_pass])

    failure_notes = []
    if not crit1_pass: failure_notes.append(f"C1 FAIL: SEPARATE gap {gap_sep:.4f} <= 0.05")
    if not crit2_pass: failure_notes.append(f"C2 FAIL: SEPARATE gap {gap_sep:.4f} <= COLLAPSED gap {gap_col:.4f}")
    if not crit3_pass: failure_notes.append(f"C3 FAIL: COLLAPSED gap {gap_col:.4f} not < {gap_sep*0.7:.4f} and not degenerate")
    if not crit4_pass: failure_notes.append(f"C4 FAIL: warmup harm events {warmup_harm_separate} <= 100")
    if not crit5_pass: failure_notes.append(f"C5 FAIL: fatal_errors={fatal_errors}")
    if not crit6_pass: failure_notes.append(f"C6 FAIL: insufficient probes (near={n_near} safe={n_safe})")

    print(f"\nARC-021 / V3-EXQ-004 verdict: {status}  ({criteria_met}/6)", flush=True)
    for note in failure_notes:
        print(f"  {note}", flush=True)

    s = results[CONDITION_SEPARATE]
    c = results[CONDITION_COLLAPSED]

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(f"- {n}" for n in failure_notes)

    summary_markdown = f"""# V3-EXQ-004 -- ARC-021 Three-Loop Incommensurability

**Status:** {status}
**Warmup:** {warmup_episodes} episodes, RANDOM policy
**Probe eval:** {eval_probe_resets} grid resets x (near-hazard + safe positions)
**Seed:** {seed}

## ARC-021 Prediction

Collapsing E1/E2/E3 gradients into a single optimizer forces E3.harm_eval to
compete with much larger E1/E2 prediction gradients. On steps with no harm events
(~88% of steps), E3 parameters receive only E1/E2 gradient signal, corrupting
the harm_eval function. Separate optimizers isolate the E3 learning channel.

## PASS Criteria

| Criterion | Result | Value |
|---|---|---|
| C1: SEPARATE calibration_gap > 0.05 | {"PASS" if crit1_pass else "FAIL"} | {gap_sep:.4f} |
| C2: SEPARATE gap > COLLAPSED gap | {"PASS" if crit2_pass else "FAIL"} | {gap_sep:.4f} vs {gap_col:.4f} |
| C3: COLLAPSED gap < SEPARATE * 0.7 or degenerate | {"PASS" if crit3_pass else "FAIL"} | {gap_col:.4f} {'[DEGENERATE]' if col_deg else ''} |
| C4: Warmup harm events > 100 | {"PASS" if crit4_pass else "FAIL"} | {warmup_harm_separate} |
| C5: No fatal errors | {"PASS" if crit5_pass else "FAIL"} | {fatal_errors} |
| C6: Probe coverage >= 10 each | {"PASS" if crit6_pass else "FAIL"} | near={n_near} safe={n_safe} |

## Calibration Results

| Condition | Optimizer | mean_near | mean_safe | calibration_gap |
|---|---|---|---|---|
| SEPARATE  | E12 (lr=1e-3) + E3 (lr=1e-4) | {s["mean_near"]:.4f} | {s["mean_safe"]:.4f} | {gap_sep:.4f} |
| COLLAPSED | All params (lr=1e-3, lambda_e3={LAMBDA_E3}) | {c["mean_near"]:.4f} | {c["mean_safe"]:.4f} | {gap_col:.4f} |

Criteria met: {criteria_met}/6 -> **{status}**
{failure_section}
"""

    metrics = {
        "fatal_error_count": float(fatal_errors),
        "warmup_harm_separate": float(warmup_harm_separate),
        "warmup_harm_collapsed": float(warmup_harm_collapsed),
        "separate_calibration_gap": float(gap_sep),
        "collapsed_calibration_gap": float(gap_col),
        "separate_mean_near": float(s["mean_near"]),
        "separate_mean_safe": float(s["mean_safe"]),
        "collapsed_mean_near": float(c["mean_near"]),
        "collapsed_mean_safe": float(c["mean_safe"]),
        "n_near_hazard_probes": float(n_near),
        "n_safe_probes": float(n_safe),
        "collapsed_harm_eval_degenerate": 1.0 if col_deg else 0.0,
        "crit1_pass": 1.0 if crit1_pass else 0.0,
        "crit2_pass": 1.0 if crit2_pass else 0.0,
        "crit3_pass": 1.0 if crit3_pass else 0.0,
        "crit4_pass": 1.0 if crit4_pass else 0.0,
        "crit5_pass": 1.0 if crit5_pass else 0.0,
        "crit6_pass": 1.0 if crit6_pass else 0.0,
        "criteria_met": float(criteria_met),
    }

    return {
        "status": status,
        "metrics": metrics,
        "summary_markdown": summary_markdown,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": "supports" if all_pass else ("mixed" if criteria_met >= 3 else "weakens"),
        "experiment_type": EXPERIMENT_TYPE,
        "fatal_error_count": fatal_errors,
    }


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",         type=int, default=0)
    parser.add_argument("--warmup",       type=int, default=300)
    parser.add_argument("--probe-resets", type=int, default=10)
    parser.add_argument("--steps",        type=int, default=200)
    args = parser.parse_args()

    result = run(
        seed=args.seed,
        warmup_episodes=args.warmup,
        eval_probe_resets=args.probe_resets,
        steps_per_episode=args.steps,
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"] = ts
    result["claim"] = CLAIM_IDS[0]
    result["verdict"] = result["status"]

    out_dir = Path(__file__).resolve().parents[1] / "evidence" / "experiments" / EXPERIMENT_TYPE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{EXPERIMENT_TYPE}_{ts}.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"\nResult written to: {out_path}", flush=True)
    print(f"Status: {result['status']}", flush=True)
    for k, v in result["metrics"].items():
        print(f"  {k}: {v}", flush=True)
