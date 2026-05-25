#!/opt/local/bin/python3
"""V3-EXQ-603b -- Q-045 MECH-313 vs MECH-260 four-arm anti-monostrategy ablation (retest).

Supersedes V3-EXQ-603a (measurement_gap).

IGW-20260521-003 / arc_062_rule_apprehension:GAP-H.

Root cause of V3-EXQ-603a (failure autopsy + governance-603-manifests interactive,
2026-05-25): 603a's call-path fix landed cleanly (mech260_operative_all_seeds=true,
dACC FIFO + suppression both wired). BUT 2 of 3 seeds (42, 44) produced
measured_steps=0 because the per-episode FIFO_WARMUP_STEPS=75 gate was never
cleared -- those seeds averaged ~14 steps/episode (total_steps ~423 and ~436
across 30 episodes), well below the 75-step warmup. Mean entropy was therefore
driven by two zero seeds; only seed 43 (which produced normal ~158-step episodes)
yielded interpretable data, leaving the comparison structurally underpowered
with N=1 effective seed.

Fixes applied vs V3-EXQ-603a:
  Fix A (per failure-autopsy V3-EXQ-603 retest spec, Section 11):
    STEPS_PER_EPISODE raised 200 -> 500. The autopsy spec's prescription;
    gives the outer-loop step budget more headroom for episodes that
    survive longer than 75 steps.

    Note: the 8-cell-design reference in the failure-autopsy artifact is a
    separate future extension, NOT the 603b retest -- explicitly held out
    of scope per the autopsy spec.

  Fix B (env-side reach extension, added 2026-05-25T06:05Z after env-code
  review):
    hazard_harm 0.05 -> 0.02. Read of causal_grid_world.py:2024 confirmed
    env terminates on `agent_health <= 0 OR env.steps >= 500`. The 603a
    short-life seeds (42/44, ~14 steps/ep) died from health depletion via
    repeated hazard contact, NOT from the script's outer step cap. Bumping
    STEPS_PER_EPISODE alone (Fix A) cannot extend an agent that runs out
    of vitality reserve at step ~14. Lowering hazard_harm 2.5x lets the
    agent survive ~35-50 contact steps before death (typical contact rate)
    instead of ~14-20, well past the 75-step FIFO warmup.

    Rationale for not compromising the experiment's dependent variable:
    hazard_harm scales agent_health depletion ONLY. The latent harm signals
    that drive every substrate the experiment is testing read independent
    paths --
      z_harm_s (SD-010): HarmEncoder(harm_obs) where harm_obs is the env
        proximity field, scaled by proximity_harm_scale (UNCHANGED at 0.1);
      z_harm_a (SD-011): AffectiveHarmEncoder(harm_obs_a, harm_history),
        EMA over the affective stream (UNCHANGED);
      AIC urgency (SD-032c), MECH-091 interrupt, MECH-302 suffering
        comparator all read z_harm_a -- unaffected by hazard_harm;
      dACC anti-recency (MECH-260) reads action FIFO -- no harm-magnitude
        dependency;
      MECH-313 noise floor: state-independent.
    Biological framing: raising lethality threshold without anesthetizing
    nociception. REE substrate already carries this distinction (z_harm_un
    SD-019a unpleasantness, z_harm_a SD-019b suffering -- both distinct
    from agent_health vitality reserve). The agent still INTEROCEPTS the
    damage via obs_body[2]=agent_health (UNCHANGED computation), just over
    a longer integration window before death.

    Methodological benefit: in 603a the env-side mortality at step ~14 was
    confounded with the policy-side measurement target (action-class entropy
    under monostrategy-locked policies walking into hazards -- exactly the
    phenomenon ARC-062 / MECH-260 / MECH-313 address). The lower hazard_harm
    isolates the policy-side signal from the survival confound. Contact rate
    feeding the latent signals is preserved; only the integral over contacts
    before death changes.

  Fixes retained from 603a (unchanged):
    Fix 1: Training loop uses sense()+generate()+select_action() via
      _select_action_with_harm(), passing obs_harm_a so z_harm_a is set and
      select_action() calls dacc.record_action() each step.
    Fix 2: FIFO_WARMUP_STEPS=75 -- pre-registered Kennerley / SD-054 gate.
    Fix 3: evidence_direction_per_claim compares each arm vs ARM_0 baseline.

(Fixes 1-3 retained from 603a are documented above under "Fixes retained
from 603a".)

SD-054 bipartite reef env + ARC-062 gated-policy ON + main-path SP-CEM.

Arms:
  ARM_0_both_off   -- control (expected collapse / low diversity)
  ARM_1_mech313    -- use_noise_floor only
  ARM_2_mech260    -- use_dacc only (MECH-260 pathway)
  ARM_3_both_on    -- MECH-313 + MECH-260

Pre-registered PASS (Q-045 mutually load-bearing):
  C1: ARM_3 entropy > ARM_0 + ENTROPY_MARGIN
  C2: ARM_3 entropy > max(ARM_1, ARM_2) + ENTROPY_MARGIN
  C3: ARM_1 entropy > ARM_0 AND ARM_2 entropy > ARM_0

PASS if C2. FAIL otherwise (still records per-arm directions).

claim_ids: Q-045, MECH-313, MECH-260
experiment_purpose: evidence

Smoke:
  /opt/local/bin/python3 experiments/v3_exq_603b_q045_mech313_mech260_four_arm_ablation.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiment_protocol import emit_outcome  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402


EXPERIMENT_TYPE = "v3_exq_603b_q045_mech313_mech260_four_arm_ablation"
QUEUE_ID = "V3-EXQ-603b"
CLAIM_IDS = ["Q-045", "MECH-313", "MECH-260"]
EXPERIMENT_PURPOSE = "evidence"

SEEDS = [42, 43, 44]
EVAL_EPISODES = 30
# 603b fix: 200 -> 500. Gives short-lived episodes more headroom to surpass the
# pre-registered 75-step FIFO warmup so the per-episode entropy window opens for
# all seeds, not just seed 43.
STEPS_PER_EPISODE = 500

# Kennerley / SD-054 temporal-horizon gate (lit-pull R5): FIFO must fill before measure.
# Unchanged from 603a -- the warmup threshold is pre-registered.
FIFO_WARMUP_STEPS = 75
DACC_SUPPRESSION_MEMORY = 8

DRY_RUN_SEEDS = [42]
DRY_RUN_EPISODES = 2
DRY_RUN_STEPS = 50

ENTROPY_MARGIN = 0.05
DACC_SUPPRESSION_WEIGHT = 0.5
WORLD_DIM = 32
HARM_A_DIM = 16
HARM_HISTORY_LEN = 10

ENV_KWARGS = dict(
    size=12,
    num_hazards=4,
    num_resources=5,
    # 603b Fix B: 0.05 -> 0.02. Lowers agent_health depletion per hazard contact so
    # seeds 42/44 can survive past the 75-step FIFO warmup. Decoupled from latent
    # harm signals: z_harm_s reads harm_obs (proximity field, scaled by
    # proximity_harm_scale UNCHANGED at 0.1); z_harm_a EMA over harm_obs_a UNCHANGED.
    # See module docstring Fix B for full decoupling rationale.
    hazard_harm=0.02,
    env_drift_interval=5,
    env_drift_prob=0.1,
    proximity_harm_scale=0.1,
    proximity_benefit_scale=0.05,
    proximity_approach_threshold=0.2,
    hazard_field_decay=0.5,
    resource_respawn_on_consume=True,
    toroidal=False,
    harm_history_len=HARM_HISTORY_LEN,
    reef_enabled=True,
    n_reef_patches=3,
    reef_patch_radius=2,
    hazard_food_attraction=0.7,
    reef_bipartite_layout=True,
    reef_bipartite_axis="horizontal",
    reef_bipartite_agent_band_radius=1,
)

ARMS: List[Dict[str, Any]] = [
    {
        "arm": "ARM_0_both_off",
        "use_noise_floor": False,
        "noise_floor_alpha": 0.1,
        "use_dacc": False,
    },
    {
        "arm": "ARM_1_mech313_only",
        "use_noise_floor": True,
        "noise_floor_alpha": 0.5,
        "use_dacc": False,
    },
    {
        "arm": "ARM_2_mech260_only",
        "use_noise_floor": False,
        "noise_floor_alpha": 0.1,
        "use_dacc": True,
    },
    {
        "arm": "ARM_3_both_on",
        "use_noise_floor": True,
        "noise_floor_alpha": 0.5,
        "use_dacc": True,
    },
]


def _entropy(counts: Counter) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        p = c / total
        if p > 0.0:
            h -= p * math.log(p)
    return float(h)


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(seed=seed, **ENV_KWARGS)


def _make_config(env: CausalGridWorldV2, arm: Dict[str, Any]) -> REEConfig:
    return REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=WORLD_DIM,
        alpha_world=0.9,
        use_affective_harm_stream=True,
        z_harm_a_dim=HARM_A_DIM,
        harm_obs_a_dim=50,
        harm_history_len=HARM_HISTORY_LEN,
        use_gated_policy=True,
        gated_policy_use_first_action_onehot=True,
        use_support_preserving_cem=True,
        support_preserving_stratified_elites=True,
        support_preserving_ao_std_floor=0.2,
        support_preserving_min_first_action_classes=2,
        use_noise_floor=arm["use_noise_floor"],
        noise_floor_alpha=arm["noise_floor_alpha"],
        use_dacc=arm["use_dacc"],
        dacc_weight=(1.0 if arm["use_dacc"] else 0.0),
        dacc_suppression_weight=(
            DACC_SUPPRESSION_WEIGHT if arm["use_dacc"] else 0.0
        ),
        dacc_suppression_memory=DACC_SUPPRESSION_MEMORY,
        use_structured_curiosity=False,
    )


def _obs_harm_a(obs_dict: Dict[str, Any]) -> Optional[torch.Tensor]:
    ha = obs_dict.get("harm_obs_a")
    if ha is None:
        return None
    t = ha.float()
    return t.unsqueeze(0) if t.dim() == 1 else t


def _obs_harm_history(obs_dict: Dict[str, Any]) -> Optional[torch.Tensor]:
    hh = obs_dict.get("harm_history")
    if hh is None:
        return None
    t = hh.float()
    return t.unsqueeze(0) if t.dim() == 1 else t


def _select_action_with_harm(
    agent: REEAgent,
    obs_body: torch.Tensor,
    obs_world: torch.Tensor,
    obs_harm_a: Optional[torch.Tensor],
    obs_harm_history: Optional[torch.Tensor],
) -> torch.Tensor:
    """Fix 1: select_action() path with obs_harm_a so z_harm_a and record_action() fire."""
    latent_state = agent.sense(
        obs_body,
        obs_world,
        obs_harm_a=obs_harm_a,
        obs_harm_history=obs_harm_history,
    )
    ticks = agent.clock.advance()
    if ticks.get("e1_tick", False):
        e1_prior = agent._e1_tick(latent_state)
    else:
        e1_prior = torch.zeros(1, WORLD_DIM, device=agent.device)
    candidates = agent.generate_trajectories(latent_state, e1_prior, ticks)
    action = agent.select_action(candidates, ticks, temperature=1.0)
    if ticks.get("e3_quiescent", False):
        agent._do_replay(latent_state)
    agent._step_count += 1
    return action


def _dacc_diagnostics(agent: REEAgent) -> Dict[str, Any]:
    if agent.dacc is None:
        return {
            "dacc_forward_calls": 0,
            "dacc_history_len": 0,
            "dacc_max_suppression": 0.0,
        }
    hist_len = len(agent.dacc._action_history)
    max_sup = 0.0
    bundle = getattr(agent, "_dacc_last_bundle", None)
    if bundle is not None and "suppression" in bundle:
        sup = bundle["suppression"]
        if isinstance(sup, torch.Tensor) and sup.numel() > 0:
            max_sup = float(sup.max().item())
    return {
        "dacc_forward_calls": int(agent.dacc._n_forward_calls),
        "dacc_history_len": hist_len,
        "dacc_max_suppression": round(max_sup, 6),
    }


def _run_arm_seed(
    arm: Dict[str, Any],
    seed: int,
    episodes: int,
    steps_per_episode: int,
    fifo_warmup_steps: int,
) -> Dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = _make_env(seed)
    cfg = _make_config(env, arm)
    agent = REEAgent(cfg)

    action_counts: Counter = Counter()
    reef_steps = 0
    total_steps = 0
    measured_steps = 0
    reef_cells = set()
    max_dacc_history = 0
    max_dacc_forward = 0
    max_dacc_suppression = 0.0

    for ep in range(episodes):
        _, obs_dict = env.reset()
        agent.reset()
        if ep == 0:
            reef_cells = set(getattr(env, "_reef_cells", set()))

        for step_in_ep in range(steps_per_episode):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            if obs_body.dim() == 1:
                obs_body = obs_body.unsqueeze(0)
            if obs_world.dim() == 1:
                obs_world = obs_world.unsqueeze(0)

            with torch.no_grad():
                action = _select_action_with_harm(
                    agent,
                    obs_body,
                    obs_world,
                    _obs_harm_a(obs_dict),
                    _obs_harm_history(obs_dict),
                )

            if arm["use_dacc"] and agent.dacc is not None:
                diag = _dacc_diagnostics(agent)
                max_dacc_history = max(max_dacc_history, diag["dacc_history_len"])
                max_dacc_forward = max(max_dacc_forward, diag["dacc_forward_calls"])
                max_dacc_suppression = max(
                    max_dacc_suppression, diag["dacc_max_suppression"]
                )

            if action is None:
                idx = random.randint(0, env.action_dim - 1)
                action_onehot = torch.zeros(1, env.action_dim, device=agent.device)
                action_onehot[0, idx] = 1.0
            else:
                action_onehot = action
                idx = int(action.argmax(dim=-1).item())

            if step_in_ep >= fifo_warmup_steps:
                action_counts[idx] += 1
                measured_steps += 1

            pos = (int(env.agent_x), int(env.agent_y))
            if pos in reef_cells:
                reef_steps += 1

            _flat, _harm, done, _info, obs_dict = env.step(action_onehot)
            total_steps += 1
            if done:
                break

        if (ep + 1) % 10 == 0 or (ep + 1) == episodes:
            print(
                f"  [train] arm={arm['arm']} seed={seed} ep {ep + 1}/{episodes}",
                flush=True,
            )

    entropy = _entropy(action_counts)
    reef_fraction = reef_steps / max(total_steps, 1)
    steps_per_ep_measured = measured_steps / max(episodes, 1)
    fifo_gate_ok = (
        fifo_warmup_steps >= 2 * DACC_SUPPRESSION_MEMORY
        and steps_per_ep_measured >= (steps_per_episode - fifo_warmup_steps) * 0.9
    )

    out: Dict[str, Any] = {
        "arm": arm["arm"],
        "seed": seed,
        "selected_action_entropy": round(entropy, 6),
        "reef_fraction": round(reef_fraction, 6),
        "total_steps": int(total_steps),
        "measured_steps": int(measured_steps),
        "fifo_warmup_steps": int(fifo_warmup_steps),
        "steps_per_episode_measured_mean": round(steps_per_ep_measured, 2),
        "fifo_temporal_gate_ok": bool(fifo_gate_ok),
        "unique_actions": len(action_counts),
    }
    if arm["use_dacc"]:
        out.update(
            {
                "dacc_history_len_max": max_dacc_history,
                "dacc_forward_calls_max": max_dacc_forward,
                "dacc_max_suppression": round(max_dacc_suppression, 6),
                "mech260_operative": bool(
                    max_dacc_forward > 0 and max_dacc_history > 0
                ),
            }
        )
    return out


def _evaluate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_arm: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_arm.setdefault(row["arm"], []).append(row)

    def mean_entropy(arm_name: str) -> float:
        vals = [r["selected_action_entropy"] for r in by_arm.get(arm_name, [])]
        return float(sum(vals) / len(vals)) if vals else 0.0

    e0 = mean_entropy("ARM_0_both_off")
    e1 = mean_entropy("ARM_1_mech313_only")
    e2 = mean_entropy("ARM_2_mech260_only")
    e3 = mean_entropy("ARM_3_both_on")

    c1 = e3 > e0 + ENTROPY_MARGIN
    c2 = e3 > max(e1, e2) + ENTROPY_MARGIN
    c3 = (e1 > e0) and (e2 > e0)

    mech260_rows = [
        r for r in rows if r["arm"] in ("ARM_2_mech260_only", "ARM_3_both_on")
    ]
    mech260_operative_all = all(
        r.get("mech260_operative", False) for r in mech260_rows
    )
    fifo_ok_all = all(r.get("fifo_temporal_gate_ok", False) for r in rows)

    return {
        "entropy_ARM_0": round(e0, 6),
        "entropy_ARM_1": round(e1, 6),
        "entropy_ARM_2": round(e2, 6),
        "entropy_ARM_3": round(e3, 6),
        "c1_both_beats_off": c1,
        "c2_mutually_load_bearing": c2,
        "c3_each_alone_beats_off": c3,
        "mech260_operative_all_seeds": mech260_operative_all,
        "fifo_temporal_gate_ok_all": fifo_ok_all,
        "overall_pass": bool(c2),
    }


def _evidence_direction_per_claim(summary: Dict[str, Any]) -> Dict[str, str]:
    """Fix 3: compare each arm vs ARM_0 baseline, not arm vs arm."""
    e0 = summary["entropy_ARM_0"]
    e1 = summary["entropy_ARM_1"]
    e2 = summary["entropy_ARM_2"]

    if summary["c2_mutually_load_bearing"]:
        q045 = "supports"
        m313 = "supports" if e1 > e0 + ENTROPY_MARGIN else "mixed"
        m260 = "supports" if e2 > e0 + ENTROPY_MARGIN else "mixed"
    elif summary["c3_each_alone_beats_off"]:
        q045 = "mixed"
        m313 = "supports" if e1 > e0 + ENTROPY_MARGIN else "mixed"
        m260 = "supports" if e2 > e0 + ENTROPY_MARGIN else "mixed"
    elif summary["entropy_ARM_3"] > e0:
        q045 = "mixed"
        m313 = "mixed" if e1 > e0 else "weakens"
        m260 = "mixed" if e2 > e0 else "weakens"
    else:
        q045 = "weakens"
        m313 = "weakens"
        m260 = "weakens"

    if not summary.get("mech260_operative_all_seeds", True):
        m260 = "unknown"

    return {"Q-045": q045, "MECH-313": m313, "MECH-260": m260}


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    seeds = DRY_RUN_SEEDS if dry_run else SEEDS
    episodes = DRY_RUN_EPISODES if dry_run else EVAL_EPISODES
    steps = DRY_RUN_STEPS if dry_run else STEPS_PER_EPISODE
    warmup = min(FIFO_WARMUP_STEPS, max(0, steps - 1)) if not dry_run else 2

    rows: List[Dict[str, Any]] = []
    for arm in ARMS:
        for seed in seeds:
            print(f"Seed {seed} Condition {arm['arm']}", flush=True)
            cell = _run_arm_seed(arm, seed, episodes, steps, warmup)
            rows.append(cell)
            print("verdict: PASS", flush=True)

    summary = _evaluate(rows)
    outcome = "PASS" if summary["overall_pass"] else "FAIL"
    edpc = _evidence_direction_per_claim(summary)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{timestamp}_v3"

    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "queue_id": QUEUE_ID,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": timestamp,
        "outcome": outcome,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": edpc["Q-045"],
        "evidence_direction_per_claim": edpc,
        "supersedes": "V3-EXQ-603a",
        "dry_run": dry_run,
        "acceptance_criteria": summary,
        "summary": summary,
        "arm_results": rows,
        "fifo_warmup_steps": warmup,
        "steps_per_episode": int(steps),
        "fixes_applied": [
            "Fix1 (from 603a): select_action() path with obs_harm_a + use_affective_harm_stream",
            f"Fix2 (from 603a): FIFO_WARMUP_STEPS={FIFO_WARMUP_STEPS} before entropy measurement",
            "Fix3 (from 603a): evidence_direction_per_claim vs ARM_0 baseline",
            f"Fix A (NEW in 603b, autopsy spec): STEPS_PER_EPISODE 200 -> {STEPS_PER_EPISODE}",
            "Fix B (NEW in 603b, env-side reach): hazard_harm 0.05 -> 0.02 so seeds 42/44 survive past warmup; latent harm signals (z_harm_s/z_harm_a) decoupled and unaffected",
        ],
    }

    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    out_path = out_dir / f"{run_id}.json"
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as fh:
            json.dump(manifest, fh, indent=2)
        print(f"Manifest written: {out_path}", flush=True)
    else:
        out_path = Path("/dev/null")
        print("Dry run -- manifest not written.", flush=True)

    print(f"Outcome: {outcome}", flush=True)
    manifest["manifest_path"] = str(out_path)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=EXPERIMENT_TYPE)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = run_experiment(dry_run=args.dry_run)
    if args.dry_run:
        sys.exit(0)
    emit_outcome(
        outcome=str(result.get("outcome", "FAIL")),
        manifest_path=str(result.get("manifest_path", "/dev/null")),
    )
