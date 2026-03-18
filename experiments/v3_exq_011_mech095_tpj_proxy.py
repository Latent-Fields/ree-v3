"""
V3-EXQ-011 — MECH-095 TPJ Agency-Detection Proxy

Claim: MECH-095 — At the z_self/z_world interface, the efference-copy-predicted
z_self change must be compared against the observed z_self change. Match → self-caused
(no residue). Mismatch → world-contributed cause (residue candidate).

Experimental logic:
  This experiment tests two necessary conditions for the TPJ comparator to function:

  CONDITION A (mismatch detects unexpected events):
    The E2 motor-sensory prediction error ||E2.predict_next_self(z_self_t, a_t) -
    z_self_observed_{t+1}|| is significantly ELEVATED at harm events compared to
    safe-movement steps. This confirms the comparator would fire at the right time.

  CONDITION B (z_self does not discriminate harm type):
    The mismatch is NOT significantly different between agent_caused_hazard and
    env_caused_hazard steps. The z_self channel is blind to the causal origin of
    harm — it only knows something unexpected happened to the body state. The
    discrimination between agent-caused and env-caused must come from z_world.
    This confirms that z_world routing (the full MECH-095 mechanism) is necessary.

    If this condition FAILS (z_self mismatch DOES discriminate), it would mean the
    SD-005 z_self/z_world split has leaked contamination signal into z_self — the
    MECH-096 dual-stream encoder is not maintaining architectural separation.

Design:
  Phase 1 — Training (RANDOM policy):
    Train E2 motor-sensory model (predict_next_self) over WARMUP_EPISODES.
    At each step record (z_self_t, action, z_self_observed_{t+1}) for mismatch
    computation, grouped by transition_type.

  Phase 2 — Eval (RANDOM policy, no training):
    Collect mismatch measurements across EVAL_EPISODES. Record mismatch per step
    with transition_type label from environment ground truth.

Metrics:
  mismatch_safe:         mean ||predicted - observed|| on safe-movement steps
  mismatch_harm:         mean ||predicted - observed|| on all harm steps
  mismatch_agent_caused: mean on agent_caused_hazard steps only
  mismatch_env_caused:   mean on env_caused_hazard steps only
  harm_safe_gap:         mismatch_harm - mismatch_safe  (should be > 0)
  agent_env_gap:         |mismatch_agent_caused - mismatch_env_caused| (should be ≈ 0)

PASS criteria (ALL must hold):
  C1: harm_safe_gap > 0.005     (TPJ fires at harm events: unexpected body state change)
  C2: agent_env_gap < 0.05      (z_self doesn't discriminate cause type; z_world needed)
  C3: n_harm >= 30              (enough harm events for reliable estimate)
  C4: n_safe >= 200             (enough safe steps for baseline)
  C5: fatal_error_count == 0
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


EXPERIMENT_TYPE = "v3_exq_011_mech095_tpj_proxy"
CLAIM_IDS = ["MECH-095", "SD-005"]


def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


def _collect_mismatch(
    agent: REEAgent,
    env: CausalGridWorld,
    num_episodes: int,
    steps_per_episode: int,
    train: bool,
    optimizer=None,
) -> Tuple[Dict[str, List[float]], int]:
    """
    Run episodes and collect E2 z_self prediction mismatches grouped by transition_type.

    Returns:
        mismatch_by_type: dict mapping transition_type -> list of mismatch scalars
        fatal_errors: count of fatal exceptions
    """
    agent.train(train)
    mismatch_by_type: Dict[str, List[float]] = {
        "safe": [],
        "agent_caused_hazard": [],
        "env_caused_hazard": [],
        "resource": [],
        "other": [],
    }
    fatal_errors = 0

    harm_buffer: List[torch.Tensor] = []
    no_harm_buffer: List[torch.Tensor] = []

    try:
        for ep in range(num_episodes):
            flat_obs, obs_dict = env.reset()
            agent.reset()

            for step in range(steps_per_episode):
                obs_body  = obs_dict["body_state"]
                obs_world = obs_dict["world_state"]

                latent_t = agent.sense(obs_body, obs_world)
                agent.clock.advance()

                z_self_t = latent_t.z_self.detach()

                # RANDOM policy
                action_idx = random.randint(0, env.action_dim - 1)
                action = _action_to_onehot(action_idx, env.action_dim, agent.device)
                agent._last_action = action

                # Efference copy prediction BEFORE env step
                with torch.no_grad():
                    z_self_predicted = agent.e2.predict_next_self(z_self_t, action)

                flat_obs, harm_signal, done, info, obs_dict = env.step(action)

                # Observed z_self_{t+1} AFTER env step
                latent_t1 = agent.sense(obs_dict["body_state"], obs_dict["world_state"])
                z_self_observed = latent_t1.z_self.detach()

                # Efference-copy mismatch: ||predicted - observed||
                mismatch = torch.norm(
                    z_self_predicted - z_self_observed, p=2
                ).item()

                # Group by transition_type
                tt = info.get("transition_type", "other") if info else "other"
                if harm_signal < 0:
                    group = tt if tt in ("agent_caused_hazard", "env_caused_hazard") else "other"
                    mismatch_by_type[group].append(mismatch)
                    mismatch_by_type["other"]  # keep "other" exists
                    harm_buffer.append(latent_t1.z_world.detach())
                    if len(harm_buffer) > 500:
                        harm_buffer = harm_buffer[-500:]
                else:
                    mismatch_by_type["safe"].append(mismatch)
                    no_harm_buffer.append(latent_t1.z_world.detach())
                    if len(no_harm_buffer) > 500:
                        no_harm_buffer = no_harm_buffer[-500:]

                if train and optimizer is not None:
                    # Train E2 motor-sensory: z_self_t + action → z_self_{t+1}
                    agent.record_transition(
                        z_self_t,
                        action.detach(),
                        z_self_observed,
                    )
                    e1_loss = agent.compute_prediction_loss()
                    e2_loss = agent.compute_e2_loss()

                    # E3 harm eval training (balanced)
                    e3_loss = agent.e1.parameters().__next__().new_zeros(())
                    n_h, n_nh = len(harm_buffer), len(no_harm_buffer)
                    if n_h >= 4 and n_nh >= 4:
                        k = min(16, n_h, n_nh)
                        zw_h  = torch.cat([harm_buffer[i]    for i in torch.randperm(n_h)[:k].tolist()], 0)
                        zw_nh = torch.cat([no_harm_buffer[i] for i in torch.randperm(n_nh)[:k].tolist()], 0)
                        zw_b  = torch.cat([zw_h, zw_nh], 0)
                        lbls  = torch.cat([
                            torch.ones(k,  1, device=agent.device),
                            torch.zeros(k, 1, device=agent.device),
                        ], 0)
                        hp = agent.e3.harm_eval(zw_b)
                        if not torch.isnan(hp).any():
                            e3_loss = F.binary_cross_entropy(
                                hp.clamp(1e-6, 1 - 1e-6), lbls
                            )

                    total_loss = e1_loss + e2_loss + e3_loss
                    if total_loss.requires_grad:
                        optimizer.zero_grad()
                        total_loss.backward()
                        torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                        optimizer.step()

                if done:
                    break

    except Exception:
        import traceback
        fatal_errors += 1
        print(f"  FATAL: {traceback.format_exc()}", flush=True)

    return mismatch_by_type, fatal_errors


def run(
    seed: int = 0,
    warmup_episodes: int = 1000,
    eval_episodes: int = 200,
    steps_per_episode: int = 200,
    self_dim: int = 32,
    world_dim: int = 32,
    lr: float = 1e-3,
    harm_scale: float = 0.02,
    alpha_world: float = 0.9,
    **kwargs,
) -> dict:
    torch.manual_seed(seed)
    random.seed(seed)

    env = CausalGridWorld(seed=seed, num_hazards=8,
                          hazard_harm=harm_scale, contaminated_harm=harm_scale)
    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=self_dim,
        world_dim=world_dim,
        alpha_world=alpha_world,
    )
    agent = REEAgent(config)
    optimizer = optim.Adam(agent.parameters(), lr=lr)

    # ── Phase 1: Training ────────────────────────────────────────────────
    print(f"\n[V3-EXQ-011] Seed {seed} — Phase 1: Training ({warmup_episodes} eps)",
          flush=True)
    _, train_fatal = _collect_mismatch(
        agent, env, warmup_episodes, steps_per_episode,
        train=True, optimizer=optimizer,
    )

    # ── Phase 2: Eval (collect mismatch data) ───────────────────────────
    print(f"[V3-EXQ-011] Seed {seed} — Phase 2: Eval ({eval_episodes} eps)", flush=True)
    mismatch_by_type, eval_fatal = _collect_mismatch(
        agent, env, eval_episodes, steps_per_episode,
        train=False, optimizer=None,
    )
    fatal_errors = train_fatal + eval_fatal

    # ── Compute metrics ──────────────────────────────────────────────────
    def mean_or_nan(lst):
        return float(sum(lst) / len(lst)) if lst else float("nan")

    m_safe   = mean_or_nan(mismatch_by_type["safe"])
    m_agent  = mean_or_nan(mismatch_by_type["agent_caused_hazard"])
    m_env    = mean_or_nan(mismatch_by_type["env_caused_hazard"])
    m_harm   = mean_or_nan(
        mismatch_by_type["agent_caused_hazard"] + mismatch_by_type["env_caused_hazard"]
    )

    n_safe   = len(mismatch_by_type["safe"])
    n_agent  = len(mismatch_by_type["agent_caused_hazard"])
    n_env    = len(mismatch_by_type["env_caused_hazard"])
    n_harm   = n_agent + n_env

    harm_safe_gap  = m_harm - m_safe   if (m_harm == m_harm and m_safe == m_safe) else float("nan")
    agent_env_gap  = abs(m_agent - m_env) if (m_agent == m_agent and m_env == m_env) else float("nan")

    print(f"  mismatch_safe={m_safe:.5f}  n={n_safe}", flush=True)
    print(f"  mismatch_harm={m_harm:.5f}  n={n_harm}  (agent={m_agent:.5f} n={n_agent}  env={m_env:.5f} n={n_env})", flush=True)
    print(f"  harm_safe_gap={harm_safe_gap:.5f}  agent_env_gap={agent_env_gap:.5f}", flush=True)

    # ── PASS / FAIL ──────────────────────────────────────────────────────
    c1 = harm_safe_gap > 0.005 if harm_safe_gap == harm_safe_gap else False
    c2 = agent_env_gap < 0.05  if agent_env_gap == agent_env_gap else False
    c3 = n_harm >= 30
    c4 = n_safe >= 200
    c5 = fatal_errors == 0

    all_pass = all([c1, c2, c3, c4, c5])
    status = "PASS" if all_pass else "FAIL"
    criteria_met = sum([c1, c2, c3, c4, c5])

    failure_notes = []
    if not c1: failure_notes.append(f"C1 FAIL: harm_safe_gap {harm_safe_gap:.5f} <= 0.005 (TPJ does not fire at harm events)")
    if not c2: failure_notes.append(f"C2 FAIL: agent_env_gap {agent_env_gap:.5f} >= 0.05 (z_self discriminates cause type — possible encoder leak)")
    if not c3: failure_notes.append(f"C3 FAIL: n_harm={n_harm} < 30")
    if not c4: failure_notes.append(f"C4 FAIL: n_safe={n_safe} < 200")
    if not c5: failure_notes.append(f"C5 FAIL: fatal_errors={fatal_errors}")

    print(f"\n[V3-EXQ-011] {status}  ({criteria_met}/5)", flush=True)
    for note in failure_notes:
        print(f"  {note}", flush=True)

    metrics = {
        "mismatch_safe":         float(m_safe),
        "mismatch_harm":         float(m_harm),
        "mismatch_agent_caused": float(m_agent),
        "mismatch_env_caused":   float(m_env),
        "harm_safe_gap":         float(harm_safe_gap),
        "agent_env_gap":         float(agent_env_gap),
        "n_safe":                float(n_safe),
        "n_harm":                float(n_harm),
        "n_agent_caused":        float(n_agent),
        "n_env_caused":          float(n_env),
        "fatal_error_count":     float(fatal_errors),
        "crit1_pass":            1.0 if c1 else 0.0,
        "crit2_pass":            1.0 if c2 else 0.0,
        "crit3_pass":            1.0 if c3 else 0.0,
        "crit4_pass":            1.0 if c4 else 0.0,
        "crit5_pass":            1.0 if c5 else 0.0,
        "criteria_met":          float(criteria_met),
    }

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(f"- {n}" for n in failure_notes)

    summary_markdown = f"""# V3-EXQ-011 — MECH-095 TPJ Agency-Detection Proxy

**Status:** {status}
**Seed:** {seed}
**Training:** {warmup_episodes} episodes (RANDOM policy)
**Eval:** {eval_episodes} episodes (RANDOM policy)

## Claim Under Test

MECH-095: TPJ comparator detects world-contributed z_self changes via efference-copy
mismatch (||E2.predict_next_self(z_self_t, a_t) - z_self_observed_t+1||).

## Two-Condition Test

**C1 — Mismatch detects unexpected events (harm_safe_gap > 0.005):**
E2 motor-sensory mismatch should be elevated at harm steps vs safe steps.
The body experienced something the motor model could not predict from the action alone.
This is what the TPJ comparator would detect and flag as `residue_flag=True`.

**C2 — z_self blind to cause type (agent_env_gap < 0.05):**
The mismatch should NOT discriminate agent_caused from env_caused harm.
Both types cause unexpected body-state changes. Only z_world carries the causal
origin information. If C2 fails, SD-005 z_self/z_world separation has leaked.

## Results

| Metric | Value |
|---|---|
| mismatch_safe | {m_safe:.5f}  (n={n_safe}) |
| mismatch_harm | {m_harm:.5f}  (n={n_harm}) |
| mismatch_agent_caused | {m_agent:.5f}  (n={n_agent}) |
| mismatch_env_caused | {m_env:.5f}  (n={n_env}) |
| harm_safe_gap | {harm_safe_gap:.5f}  [threshold > 0.005] |
| agent_env_gap | {agent_env_gap:.5f}  [threshold < 0.05] |

## PASS Criteria

| Criterion | Result | Value |
|---|---|---|
| C1: harm_safe_gap > 0.005 | {"PASS" if c1 else "FAIL"} | {harm_safe_gap:.5f} |
| C2: agent_env_gap < 0.05 | {"PASS" if c2 else "FAIL"} | {agent_env_gap:.5f} |
| C3: n_harm >= 30 | {"PASS" if c3 else "FAIL"} | {n_harm} |
| C4: n_safe >= 200 | {"PASS" if c4 else "FAIL"} | {n_safe} |
| C5: no fatal errors | {"PASS" if c5 else "FAIL"} | {fatal_errors} |

Criteria met: {criteria_met}/5 → **{status}**
{failure_section}
"""

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
    parser.add_argument("--seed",           type=int,   default=0)
    parser.add_argument("--warmup",         type=int,   default=1000)
    parser.add_argument("--eval",           type=int,   default=200)
    parser.add_argument("--steps",          type=int,   default=200)
    parser.add_argument("--harm-scale",     type=float, default=0.02)
    parser.add_argument("--alpha-world",    type=float, default=0.9)
    args = parser.parse_args()

    result = run(
        seed=args.seed,
        warmup_episodes=args.warmup,
        eval_episodes=args.eval,
        steps_per_episode=args.steps,
        harm_scale=args.harm_scale,
        alpha_world=args.alpha_world,
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
