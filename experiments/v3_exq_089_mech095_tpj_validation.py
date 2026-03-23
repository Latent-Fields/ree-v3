"""
V3-EXQ-089 -- MECH-095: TPJ Agency Comparator Validation

Claims: MECH-095

Tests the live TPJ comparator (ree_core/comparator/tpj_comparator.py).

Mechanism under test:
    z_self_pred = E2.predict_next_self(z_self_t, action_taken)
    z_self_obs  = encode(new_obs).z_self
    agency_signal, is_self_caused = tpj.compare(z_self_pred, z_self_obs)

Hypothesis:
  A) Agency mismatch is HIGHER after world-caused events (env_caused_hazard) than
     after self-caused locomotion events (none) -- TPJ fires for unexpected changes.
  B) Agency signal correctly classifies self-caused vs world-caused transitions
     above chance (accuracy > 0.70, using optimal threshold search).

Schizophrenia failure mode reference (Frith et al. 2000):
  The comparator should assign LOW agency signal to world-caused events.
  If agency_signal remains high for world-caused events, the boundary has failed:
  the agent cannot distinguish its own effects from the environment's.

Protocol:
  Phase 1 -- E2 training (300 eps, random policy): train self_transition + world_forward.
  Phase 2 -- TPJ evaluation (200 eps): freeze, record agency_signal per transition type.
    NO task-specific training for TPJ -- it is a purely unsupervised comparator.
    The comparator uses E2's existing predict_next_self and the encoder's z_self.

PASS criteria (ALL):
  C1: mean_mismatch_world_caused > mean_mismatch_self_caused + 0.02
      (mismatch elevates after world-caused events)
  C2: classification accuracy > 0.70 (optimal threshold on agency_signal)
  C3: false_attribution_rate < 0.15 (world-caused events labelled as self-caused)
  C4: n_world_caused_eval >= 30 (sufficient world events)
  C5: agency_signal_none < agency_signal_none_std * 3 + 0.5
      (agency_signal distributional coherence -- not saturated at 1.0)
"""

import sys
import random
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.comparator.tpj_comparator import TPJComparator
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_089_mech095_tpj_validation"
CLAIM_IDS = ["MECH-095"]


def _action_to_onehot(idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, idx] = 1.0
    return v


def _is_world_caused(ttype: str) -> bool:
    """World-caused: env-initiated harm events (not agent stepping into hazard)."""
    return ttype == "env_caused_hazard"


def run(
    seed: int = 0,
    phase1_episodes: int = 300,
    eval_episodes: int = 200,
    steps_per_episode: int = 200,
    world_dim: int = 32,
    self_dim: int = 32,
    lr: float = 1e-3,
    alpha_world: float = 0.9,
    **kwargs,
) -> dict:
    torch.manual_seed(seed)
    random.seed(seed)

    env = CausalGridWorldV2(
        seed=seed, size=12, num_hazards=5, num_resources=3,
        hazard_harm=0.05,
        env_drift_interval=3, env_drift_prob=0.3,  # frequent env-caused events
        proximity_harm_scale=0.05,
        proximity_approach_threshold=0.15,
        hazard_field_decay=0.5,
        use_proxy_fields=True,
    )

    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=self_dim,
        world_dim=world_dim,
        alpha_world=alpha_world,
        alpha_self=0.3,
        reafference_action_dim=env.action_dim,
    )
    agent = REEAgent(config)

    tpj = TPJComparator(self_dim=self_dim, agency_threshold=0.5)

    num_actions = env.action_dim

    print(
        f"[V3-EXQ-089] MECH-095 TPJ Agency Comparator Validation\n"
        f"  agency_signal = 1 / (1 + ||z_self_pred - z_self_obs||)\n"
        f"  Phases: E2 training ({phase1_episodes} eps) -> TPJ eval ({eval_episodes} eps)",
        flush=True,
    )

    # -- Phase 1: E2 self_transition training --------------------------------
    std_params = [p for n, p in agent.named_parameters()
                  if "world_transition" not in n and "world_action_encoder" not in n]
    opt_std = optim.Adam(std_params, lr=lr)

    agent.train()

    for ep in range(phase1_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()

        for _ in range(steps_per_episode):
            latent = agent.sense(obs_dict["body_state"], obs_dict["world_state"])
            agent.clock.advance()

            action_idx = random.randint(0, num_actions - 1)
            action = _action_to_onehot(action_idx, num_actions, agent.device)
            agent._last_action = action

            flat_obs, _, done, _, obs_dict = env.step(action)

            total = agent.compute_prediction_loss() + agent.compute_e2_loss()
            if total.requires_grad:
                opt_std.zero_grad(); total.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                opt_std.step()

            if done: break

        if (ep + 1) % 100 == 0 or ep == phase1_episodes - 1:
            print(f"  [P1] ep {ep+1}/{phase1_episodes}", flush=True)

    # -- Phase 2: TPJ evaluation (no training) --------------------------------
    print(f"\n[P2] TPJ evaluation ({eval_episodes} eps)...", flush=True)
    agent.eval()

    mismatch_by_type: Dict[str, List[float]] = {}
    agency_by_type:   Dict[str, List[float]] = {}

    for ep in range(eval_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        z_self_prev = None
        action_prev = None

        for _ in range(steps_per_episode):
            with torch.no_grad():
                latent = agent.sense(obs_dict["body_state"], obs_dict["world_state"])
                agent.clock.advance()
                z_self_obs = latent.z_self.detach()

                if z_self_prev is not None and action_prev is not None:
                    # Efference copy: what did E2 predict from the LAST step?
                    z_self_pred = agent.e2.predict_next_self(z_self_prev, action_prev)
                    agency_signal, is_self = tpj.compare(z_self_pred, z_self_obs)
                    mismatch = (z_self_pred - z_self_obs).norm(dim=-1).mean().item()
                    agency_val = agency_signal.mean().item()

                    mismatch_by_type.setdefault(ttype_prev, []).append(mismatch)
                    agency_by_type.setdefault(ttype_prev, []).append(agency_val)

            action_idx = random.randint(0, num_actions - 1)
            action = _action_to_onehot(action_idx, num_actions, agent.device)
            agent._last_action = action
            flat_obs, _, done, info, obs_dict = env.step(action)
            ttype_prev = info.get("transition_type", "none")
            z_self_prev = z_self_obs
            action_prev = action.detach()

            if done: break

        if (ep + 1) % 50 == 0 or ep == eval_episodes - 1:
            n_world = len(mismatch_by_type.get("env_caused_hazard", []))
            n_self  = len(mismatch_by_type.get("none", []))
            print(f"  [P2] ep {ep+1}/{eval_episodes}  n_world={n_world}  n_self={n_self}", flush=True)

    # -- Metrics --------------------------------------------------------------
    def _mean(lst): return float(np.mean(lst)) if lst else 0.0
    def _std(lst):  return float(np.std(lst))  if lst else 0.0

    none_mismatch    = mismatch_by_type.get("none", [])
    world_mismatch   = mismatch_by_type.get("env_caused_hazard", [])
    agent_mismatch   = mismatch_by_type.get("agent_caused_hazard", [])
    none_agency      = agency_by_type.get("none", [])
    world_agency     = agency_by_type.get("env_caused_hazard", [])

    mean_mismatch_self  = _mean(none_mismatch)
    mean_mismatch_world = _mean(world_mismatch)
    mean_agency_none    = _mean(none_agency)
    mean_agency_world   = _mean(world_agency)
    std_agency_none     = _std(none_agency)
    n_world             = len(world_mismatch)

    # C2/C3: find optimal threshold for classification
    all_labels  = [0.0] * len(none_agency) + [1.0] * len(world_agency)
    all_agency  = list(none_agency) + list(world_agency)  # high agency = self (0), low = world (1)

    best_acc    = 0.0
    best_thresh = 0.5
    false_attr  = 1.0

    if len(all_labels) >= 10:
        for thresh in np.linspace(0.1, 0.9, 17):
            # world-caused = low agency (signal < thresh -> predict world-caused)
            preds = [1.0 if a < thresh else 0.0 for a in all_agency]
            acc = sum(1 for p, l in zip(preds, all_labels) if abs(p - l) < 0.5) / len(all_labels)
            if acc > best_acc:
                best_acc = acc
                best_thresh = float(thresh)

        # False attribution: world-caused events labelled as self-caused
        world_preds = [1.0 if a < best_thresh else 0.0 for a in world_agency]
        false_attr = (world_preds.count(0.0) / max(len(world_preds), 1))

    print(f"\n  --- EXQ-089 results ---", flush=True)
    print(f"  mean_mismatch_self:   {mean_mismatch_self:.6f}", flush=True)
    print(f"  mean_mismatch_world:  {mean_mismatch_world:.6f}", flush=True)
    print(f"  mismatch_gap:         {mean_mismatch_world - mean_mismatch_self:.6f}", flush=True)
    print(f"  mean_agency_none:     {mean_agency_none:.4f}  std={std_agency_none:.4f}", flush=True)
    print(f"  mean_agency_world:    {mean_agency_world:.4f}", flush=True)
    print(f"  classification_acc:   {best_acc:.4f}  @thresh={best_thresh:.2f}", flush=True)
    print(f"  false_attribution:    {false_attr:.4f}", flush=True)
    print(f"  n_world_caused:       {n_world}", flush=True)

    c1 = mean_mismatch_world > mean_mismatch_self + 0.02
    c2 = best_acc            > 0.70
    c3 = false_attr          < 0.15
    c4 = n_world             >= 30
    c5 = not (mean_agency_none > 0.95 and std_agency_none < 0.01)  # not saturated at 1.0

    all_pass = c1 and c2 and c3 and c4 and c5
    status   = "PASS" if all_pass else "FAIL"
    n_met    = sum([c1, c2, c3, c4, c5])

    failure_notes = []
    if not c1: failure_notes.append(
        f"C1 FAIL: mismatch gap={mean_mismatch_world - mean_mismatch_self:.6f} <= 0.02. "
        f"E2 predict_next_self does not differentiate self vs world-caused changes.")
    if not c2: failure_notes.append(
        f"C2 FAIL: classification_acc={best_acc:.4f} <= 0.70.")
    if not c3: failure_notes.append(
        f"C3 FAIL: false_attribution={false_attr:.4f} >= 0.15. "
        f"World-caused events misattributed as self-caused (schizophrenic failure mode).")
    if not c4: failure_notes.append(
        f"C4 FAIL: n_world_caused={n_world} < 30. Too few world-caused events.")
    if not c5: failure_notes.append(
        f"C5 FAIL: agency_signal saturated at ~1.0. E2 predictions too accurate "
        f"or encoder outputs too smooth.")

    print(f"\nV3-EXQ-089 verdict: {status}  ({n_met}/5)", flush=True)
    for note in failure_notes:
        print(f"  {note}", flush=True)

    metrics = {
        "mean_mismatch_self":   float(mean_mismatch_self),
        "mean_mismatch_world":  float(mean_mismatch_world),
        "mismatch_gap":         float(mean_mismatch_world - mean_mismatch_self),
        "mean_agency_none":     float(mean_agency_none),
        "mean_agency_world":    float(mean_agency_world),
        "classification_acc":   float(best_acc),
        "optimal_threshold":    float(best_thresh),
        "false_attribution":    float(false_attr),
        "n_world_caused":       float(n_world),
        "crit1_pass":           1.0 if c1 else 0.0,
        "crit2_pass":           1.0 if c2 else 0.0,
        "crit3_pass":           1.0 if c3 else 0.0,
        "crit4_pass":           1.0 if c4 else 0.0,
        "crit5_pass":           1.0 if c5 else 0.0,
        "criteria_met":         float(n_met),
        "fatal_error_count":    0.0,
    }

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(f"- {n}" for n in failure_notes)

    summary_markdown = f"""# V3-EXQ-089 -- MECH-095: TPJ Agency Comparator Validation

**Status:** {status}
**Claims:** MECH-095
**World:** CausalGridWorldV2 (5 hazards, frequent env_drift for world-caused events)
**Protocol:** E2 training ({phase1_episodes} eps) -> TPJ eval ({eval_episodes} eps, no TPJ training)

## TPJ Mechanism

agency_signal = 1 / (1 + ||E2.predict_next_self(z_self_t, a_t) - z_self_{{t+1}}||)

High agency_signal: action produced predicted z_self change -> self-caused.
Low agency_signal: z_self changed unexpectedly -> world-caused or action error.

Schizophrenia failure mode (Frith et al. 2000): if the comparator is absent/noisy,
self-generated actions appear world-caused (passivity experiences).

## Results

| Metric | Value |
|--------|-------|
| mean mismatch (self-caused) | {mean_mismatch_self:.6f} |
| mean mismatch (world-caused) | {mean_mismatch_world:.6f} |
| mismatch gap | {mean_mismatch_world - mean_mismatch_self:.6f} |
| classification accuracy | {best_acc:.4f} |
| false attribution rate | {false_attr:.4f} |
| n_world_caused_eval | {n_world} |

## PASS Criteria

| Criterion | Result | Value |
|-----------|--------|-------|
| C1: mismatch_world > mismatch_self + 0.02 | {"PASS" if c1 else "FAIL"} | gap={mean_mismatch_world - mean_mismatch_self:.6f} |
| C2: classification_acc > 0.70 | {"PASS" if c2 else "FAIL"} | {best_acc:.4f} |
| C3: false_attribution < 0.15 | {"PASS" if c3 else "FAIL"} | {false_attr:.4f} |
| C4: n_world_caused >= 30 | {"PASS" if c4 else "FAIL"} | {n_world} |
| C5: agency_signal not saturated | {"PASS" if c5 else "FAIL"} | mean={mean_agency_none:.3f} std={std_agency_none:.3f} |

Criteria met: {n_met}/5 -> **{status}**
{failure_section}
"""

    return {
        "status":             status,
        "metrics":            metrics,
        "summary_markdown":   summary_markdown,
        "claim_ids":          CLAIM_IDS,
        "evidence_direction": "supports" if all_pass else ("mixed" if n_met >= 3 else "weakens"),
        "experiment_type":    EXPERIMENT_TYPE,
        "fatal_error_count":  0,
    }


if __name__ == "__main__":
    import argparse, json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",   type=int, default=0)
    parser.add_argument("--phase1", type=int, default=300)
    parser.add_argument("--eval",   type=int, default=200)
    parser.add_argument("--steps",  type=int, default=200)
    args = parser.parse_args()

    result = run(
        seed=args.seed,
        phase1_episodes=args.phase1,
        eval_episodes=args.eval,
        steps_per_episode=args.steps,
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"]      = ts
    result["claim"]              = CLAIM_IDS[0]
    result["verdict"]            = result["status"]
    result["run_id"]             = f"{EXPERIMENT_TYPE}_{ts}_v3"
    result["architecture_epoch"] = "ree_hybrid_guardrails_v1"

    out_dir = (
        Path(__file__).resolve().parents[2]
        / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{EXPERIMENT_TYPE}_{ts}.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"\nResult written to: {out_path}", flush=True)
    print(f"Status: {result['status']}", flush=True)
    for k, v in result["metrics"].items():
        print(f"  {k}: {v}", flush=True)
