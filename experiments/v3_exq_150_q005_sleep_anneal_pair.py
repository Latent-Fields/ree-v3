#!/opt/local/bin/python3
"""
V3-EXQ-150 -- Q-005: Sleep Annealing / Reset of R(x,t)

Claim:    Q-005
Proposal: EXP-0094 (EVB-0074)

Q-005 asks:
  "Can offline integration be modeled as a resetting or annealing of R(x,t)?"

  R(x,t) is the astrocytic regulatory field over L-space positions (MECH-001).
  In biology, sleep is associated with synaptic homeostasis: the brain downscales
  synaptic weights accumulated during waking (Tononi & Cirelli 2014), and hippocampal
  SWR replay transfers information to cortex while simultaneously down-weighting
  transient episodic traces.

  For the REE residue field, two distinct offline operations are possible:
    (A) ANNEAL: multiply all RBF weights by a decay factor < 1 (partial forgetting).
        Preserves the structure of the residue field but reduces magnitude. Analogous
        to synaptic homeostasis: the spatial pattern of harm knowledge is retained but
        scaled down, preventing saturation across episodes.
    (B) RESET:  zero all RBF weights (full erasure). Tabula rasa between waking phases.
        Represents the view that episodic harm memory should not carry over -- the agent
        re-learns harm from scratch each waking period.

  The null (NO_ANNEAL) condition: residue field persists unchanged between episodes,
  continuing to accumulate without any offline correction. This is the current default
  in V3.

  Scientific question: do ANNEAL or RESET produce better (lower) harm across an extended
  multi-session protocol? If ANNEAL wins: offline partial decay is beneficial (residue
  homeostasis is a useful mechanism). If NO_ANNEAL wins: accumulated residue is load-bearing
  across episodes (wiping it is harmful). If RESET wins: episodic residue should not
  persist (harm knowledge is session-local).

  Experimental operationalisation:
    - "waking phase": 100 training episodes (agent moves, accumulates residue, learns).
    - "sleep phase": apply offline operation to residue field (anneal, reset, or none).
    - Protocol: 3 waking phases separated by 2 sleep phases = 300 training episodes total.
    - After each sleep phase, measure harm in the NEXT waking phase.
    - Primary metric: mean_harm across phases 2 and 3 (post-sleep evaluation).
    - Secondary metric: residue_contrast (spatial differentiation of harm loci).

  This tests whether offline R(x,t) modification changes how quickly the agent re-learns
  harm avoidance in subsequent waking phases.

Pre-registered thresholds
--------------------------
C1: ANNEAL reduces harm vs NO_ANNEAL: mean_harm_anneal < mean_harm_no_anneal * THRESH_ANNEAL_WIN
    across waking phases 2+3 (both seeds). THRESH_ANNEAL_WIN = 0.92 (anneal must be >=8%
    lower harm than no-anneal on average).

C2: RESET does not catastrophically worsen harm: mean_harm_reset < mean_harm_no_anneal
    * THRESH_NO_REGRESSION (both seeds). THRESH_NO_REGRESSION = 1.15 (reset must not be
    >15% worse than no-anneal -- confirms reset is not catastrophically destructive).

C3: ANNEAL residue_contrast > THRESH_CONTRAST_ANNEAL after all sleep phases (both seeds).
    The annealed field must retain some spatial structure (not collapse to zero).
    THRESH_CONTRAST_ANNEAL = 0.03.

C4: NO_ANNEAL residue_contrast > THRESH_CONTRAST_NO_ANNEAL (both seeds, waking phase 3).
    The persistent field should show accumulated differentiation.
    THRESH_CONTRAST_NO_ANNEAL = 0.02.

C5: Seed consistency: C1 direction (or its absence) agrees across both seeds.

PASS: C1 + C3 + C5  => ANNEAL is beneficial; offline partial R(x,t) decay is load-bearing.
PARTIAL (NO_ANNEAL wins): C1 fails + C4 + C5 => persistent residue is better; no offline op.
PARTIAL (RESET acceptable): C2 + C5 but not C1 => neither ANNEAL nor NO_ANNEAL clearly best.
FAIL: C3 fails (anneal collapsed to zero) OR C4 fails (no_anneal never builds contrast).

PASS     => Q-005 guidance: ANNEAL model for offline integration (synaptic homeostasis analog).
PARTIAL (NO_ANNEAL) => Q-005 guidance: no offline operation; persistent residue is better.
PARTIAL (RESET)     => Q-005 guidance: partial evidence; offline erasure acceptable but not best.
FAIL     => implementation problem; residue field does not build or retain usable structure.

Conditions
----------
NO_ANNEAL:
  Residue field persists unchanged between waking phases.
  This is the current default V3 behaviour.

ANNEAL:
  After each waking phase, multiply all RBF weights by ANNEAL_FACTOR = 0.3.
  Preserves the relative spatial pattern of harm knowledge, but reduces magnitude.
  Prevents multi-episode saturation. Analogous to synaptic homeostasis.

RESET:
  After each waking phase, zero all RBF weights (full erasure).
  The agent starts each waking phase with a fresh residue field.

Seeds: [42, 123] (matched -- same env seed per condition)
Env:   CausalGridWorldV2 size=10, 2 hazards, 3 resources, hazard_harm=0.05
Protocol: 3 waking phases x 100 episodes x 200 steps, separated by 2 sleep phases.
Eval: measured across waking phases 2 and 3 (post-sleep).
Estimated runtime: ~45 min any machine (3 conditions x 2 seeds x ~7.5 min each)
"""

import sys
import random
import json
import uuid
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig, ResidueConfig


EXPERIMENT_TYPE = "v3_exq_150_q005_sleep_anneal_pair"
CLAIM_IDS = ["Q-005"]

# ---------------------------------------------------------------------------
# Pre-registered thresholds
# ---------------------------------------------------------------------------
# C1: ANNEAL must reduce mean harm vs NO_ANNEAL by at least this fraction
THRESH_ANNEAL_WIN = 0.92

# C2: RESET must not exceed NO_ANNEAL harm by more than this fraction
THRESH_NO_REGRESSION = 1.15

# C3: ANNEAL condition residue_contrast must exceed this after sleep phases
THRESH_CONTRAST_ANNEAL = 0.03

# C4: NO_ANNEAL residue_contrast must exceed this in waking phase 3
THRESH_CONTRAST_NO_ANNEAL = 0.02

# Anneal factor: fraction of RBF weight retained after each sleep phase
ANNEAL_FACTOR = 0.3

# ---------------------------------------------------------------------------
# Protocol constants
# ---------------------------------------------------------------------------
WAKING_PHASES = 3           # total waking phases
EPISODES_PER_PHASE = 100    # training episodes per waking phase
STEPS_PER_EPISODE = 200     # steps per episode
LR = 3e-4

SEEDS = [42, 123]

# Condition names
CONDITIONS = ["NO_ANNEAL", "ANNEAL", "RESET"]

# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------
BODY_OBS_DIM = 12
WORLD_OBS_DIM = 250   # CausalGridWorldV2 size=10
ACTION_DIM = 5
WORLD_DIM = 32


# ---------------------------------------------------------------------------
# Environment and config factories
# ---------------------------------------------------------------------------

def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed,
        size=10,
        num_hazards=2,
        num_resources=3,
        hazard_harm=0.05,
        env_drift_interval=15,
        env_drift_prob=0.2,
    )


def _make_config() -> REEConfig:
    cfg = REEConfig()
    cfg.latent.world_dim = WORLD_DIM
    cfg.latent.self_dim = 16
    cfg.latent.body_obs_dim = BODY_OBS_DIM
    cfg.latent.world_obs_dim = WORLD_OBS_DIM
    cfg.latent.observation_dim = BODY_OBS_DIM + WORLD_OBS_DIM
    cfg.latent.alpha_world = 0.95   # SD-008
    cfg.latent.alpha_self = 0.3
    cfg.residue.world_dim = WORLD_DIM
    cfg.residue.accumulation_rate = 0.1
    cfg.residue.kernel_bandwidth = 1.0
    cfg.residue.num_basis_functions = 32
    cfg.residue.decay_rate = 0.0    # invariant: no waking-phase decay; offline only
    cfg.residue.benefit_terrain_enabled = False
    return cfg


# ---------------------------------------------------------------------------
# Sleep operation: apply offline R(x,t) modification
# ---------------------------------------------------------------------------

def _apply_sleep_op(agent: REEAgent, condition: str) -> None:
    """
    Apply the offline (sleep) operation to the agent's residue field.

    NO_ANNEAL: no-op (residue persists unchanged).
    ANNEAL:    multiply all RBF weights by ANNEAL_FACTOR.
    RESET:     zero all RBF weights.
    """
    if condition == "NO_ANNEAL":
        return  # no operation

    with torch.no_grad():
        if condition == "ANNEAL":
            agent.residue_field.rbf_field.weights.data *= ANNEAL_FACTOR
        elif condition == "RESET":
            agent.residue_field.rbf_field.weights.data.zero_()
        else:
            raise ValueError(f"Unknown condition: {condition}")


# ---------------------------------------------------------------------------
# Run one waking phase
# ---------------------------------------------------------------------------

def _run_waking_phase(
    agent: REEAgent,
    env: CausalGridWorldV2,
    e1_opt: optim.Optimizer,
    e3_opt: optim.Optimizer,
    num_episodes: int,
    steps_per_episode: int,
    phase_idx: int,
    condition: str,
    dry_run: bool,
) -> Dict:
    """
    Run one waking phase. Returns metrics for this phase.
    Phase_idx is 0-indexed (0 = first waking phase).
    """
    agent.train()

    phase_harms: List[float] = []
    phase_residue_at_harm: List[float] = []
    phase_residue_elsewhere: List[float] = []

    for ep in range(num_episodes):
        _, obs_dict = env.reset()
        agent.reset()
        ep_harm = 0.0

        for _step in range(steps_per_episode):
            obs_body  = torch.tensor(obs_dict["body_state"],  dtype=torch.float32)
            obs_world = torch.tensor(obs_dict["world_state"], dtype=torch.float32)

            agent.sense(obs_body, obs_world)
            agent.clock.advance()

            e1_loss = agent.compute_prediction_loss()
            if e1_loss is not None and e1_loss.requires_grad:
                e1_opt.zero_grad()
                e1_loss.backward()
                e1_opt.step()

            if agent._current_latent is not None:
                z_world_cur = agent._current_latent.z_world.detach()
                harm_pred = agent.e3.harm_eval(z_world_cur)

                action_idx = random.randint(0, ACTION_DIM - 1)
                action = torch.zeros(1, ACTION_DIM)
                action[0, action_idx] = 1.0
                _, reward, done, _, obs_dict = env.step(action)

                harm_signal = float(min(0.0, reward))
                ep_harm += abs(harm_signal)

                if harm_signal < 0.0:
                    agent.residue_field.accumulate(
                        z_world_cur,
                        harm_magnitude=abs(harm_signal),
                        hypothesis_tag=False,
                    )
                    phase_residue_at_harm.append(
                        float(agent.residue_field.evaluate(z_world_cur).item())
                    )
                else:
                    if agent._current_latent is not None:
                        phase_residue_elsewhere.append(
                            float(agent.residue_field.evaluate(z_world_cur).item())
                        )

                # E3 harm eval training
                harm_target = torch.tensor([abs(harm_signal)], dtype=torch.float32)
                e3_loss = nn.functional.mse_loss(harm_pred.view(-1), harm_target)
                if e3_loss.requires_grad:
                    e3_opt.zero_grad()
                    e3_loss.backward()
                    e3_opt.step()

                if done:
                    _, obs_dict = env.reset()
                    agent.reset()
            else:
                action_idx = random.randint(0, ACTION_DIM - 1)
                action = torch.zeros(1, ACTION_DIM)
                action[0, action_idx] = 1.0
                _, _, done, _, obs_dict = env.step(action)
                if done:
                    _, obs_dict = env.reset()
                    agent.reset()

        phase_harms.append(ep_harm)

    mean_harm = float(sum(phase_harms) / max(len(phase_harms), 1))
    mean_at_harm = float(sum(phase_residue_at_harm) / max(len(phase_residue_at_harm), 1))
    mean_elsewhere = float(sum(phase_residue_elsewhere) / max(len(phase_residue_elsewhere), 1))
    residue_contrast = max(0.0, mean_at_harm - mean_elsewhere)

    residue_stats = agent.residue_field.get_statistics()
    total_residue = float(residue_stats["total_residue"].item())

    print(
        f"  [{condition}] phase={phase_idx+1} mean_harm={mean_harm:.4f}"
        f" residue_contrast={residue_contrast:.4f}"
        f" total_residue={total_residue:.3f}",
        flush=True,
    )

    return {
        "phase_idx": phase_idx,
        "mean_harm": mean_harm,
        "residue_contrast": residue_contrast,
        "mean_residue_at_harm": mean_at_harm,
        "mean_residue_elsewhere": mean_elsewhere,
        "total_residue": total_residue,
        "n_episodes": num_episodes,
    }


# ---------------------------------------------------------------------------
# Run one full condition x seed (3 waking phases + 2 sleep phases)
# ---------------------------------------------------------------------------

def _run_condition(
    seed: int,
    condition: str,
    episodes_per_phase: int,
    steps_per_episode: int,
    lr: float,
    dry_run: bool,
) -> Dict:
    """
    Run the full multi-phase protocol for one condition + seed.
    Returns per-phase metrics and summary.
    """
    if dry_run:
        episodes_per_phase = 2
        steps_per_episode = 10

    torch.manual_seed(seed)
    random.seed(seed)

    env = _make_env(seed)
    cfg = _make_config()
    agent = REEAgent(cfg)

    e1_opt = optim.Adam(agent.e1.parameters(), lr=lr)
    e3_opt = optim.Adam(agent.e3.parameters(), lr=lr)

    print(
        f"\n--- [{condition}] seed={seed}"
        f" phases={WAKING_PHASES} eps_per_phase={episodes_per_phase}"
        f" steps={steps_per_episode} ---",
        flush=True,
    )

    phase_results: List[Dict] = []

    for phase_idx in range(WAKING_PHASES):
        result = _run_waking_phase(
            agent=agent,
            env=env,
            e1_opt=e1_opt,
            e3_opt=e3_opt,
            num_episodes=episodes_per_phase,
            steps_per_episode=steps_per_episode,
            phase_idx=phase_idx,
            condition=condition,
            dry_run=dry_run,
        )
        phase_results.append(result)

        # Apply sleep op between waking phases (not after the last phase)
        if phase_idx < WAKING_PHASES - 1:
            _apply_sleep_op(agent, condition)
            print(f"  [{condition}] seed={seed} sleep op applied ({condition})", flush=True)

    # Summary: mean harm across post-sleep phases (phases 2 and 3 = indices 1 and 2)
    post_sleep_phases = phase_results[1:]  # phases after first sleep
    mean_harm_post_sleep = float(
        sum(p["mean_harm"] for p in post_sleep_phases)
        / max(len(post_sleep_phases), 1)
    )

    # Residue contrast in final waking phase (phase 3 = index 2)
    final_contrast = phase_results[-1]["residue_contrast"]

    # Residue contrast immediately after last sleep (= residue_contrast at end of phase 3)
    # For ANNEAL: check that contrast > THRESH_CONTRAST_ANNEAL
    anneal_contrast_after_sleep = final_contrast

    return {
        "condition": condition,
        "seed": seed,
        "mean_harm_post_sleep": mean_harm_post_sleep,
        "final_contrast": final_contrast,
        "anneal_contrast_after_sleep": anneal_contrast_after_sleep,
        "phase_results": phase_results,
    }


# ---------------------------------------------------------------------------
# Criterion evaluation
# ---------------------------------------------------------------------------

def _evaluate_criteria(results_by_condition: Dict[str, List[Dict]]) -> Dict[str, bool]:
    """Evaluate pre-registered criteria across all conditions and seeds."""

    no_anneal = results_by_condition["NO_ANNEAL"]
    anneal    = results_by_condition["ANNEAL"]
    reset_res = results_by_condition["RESET"]

    # C1: ANNEAL reduces mean_harm_post_sleep vs NO_ANNEAL (both seeds)
    c1_seeds = []
    for i in range(len(anneal)):
        c1_seeds.append(
            anneal[i]["mean_harm_post_sleep"]
            < no_anneal[i]["mean_harm_post_sleep"] * THRESH_ANNEAL_WIN
        )
    c1 = all(c1_seeds)

    # C2: RESET does not catastrophically worsen harm vs NO_ANNEAL (both seeds)
    c2_seeds = []
    for i in range(len(reset_res)):
        c2_seeds.append(
            reset_res[i]["mean_harm_post_sleep"]
            < no_anneal[i]["mean_harm_post_sleep"] * THRESH_NO_REGRESSION
        )
    c2 = all(c2_seeds)

    # C3: ANNEAL residue_contrast after sleep phases > THRESH_CONTRAST_ANNEAL (both seeds)
    c3_seeds = [r["anneal_contrast_after_sleep"] > THRESH_CONTRAST_ANNEAL for r in anneal]
    c3 = all(c3_seeds)

    # C4: NO_ANNEAL final_contrast > THRESH_CONTRAST_NO_ANNEAL (both seeds)
    c4_seeds = [r["final_contrast"] > THRESH_CONTRAST_NO_ANNEAL for r in no_anneal]
    c4 = all(c4_seeds)

    # C5: C1 direction agrees across both seeds (or absence of improvement is consistent)
    if c1:
        # ANNEAL wins both seeds -- check directionality consistent
        c5_seeds = [
            anneal[i]["mean_harm_post_sleep"] < no_anneal[i]["mean_harm_post_sleep"]
            for i in range(len(anneal))
        ]
        c5 = all(c5_seeds)
    else:
        # No ANNEAL win -- check that the direction is consistent (either NO_ANNEAL or RESET
        # consistently dominates, not mixed-seed flip)
        c5_seeds = []
        for i in range(len(anneal)):
            # Direction: is NO_ANNEAL <= ANNEAL for this seed?
            dir_i = (no_anneal[i]["mean_harm_post_sleep"] <= anneal[i]["mean_harm_post_sleep"])
            # Compare to seed 0 direction
            dir_0 = (no_anneal[0]["mean_harm_post_sleep"] <= anneal[0]["mean_harm_post_sleep"])
            c5_seeds.append(dir_i == dir_0)
        c5 = all(c5_seeds)

    return {
        "C1_anneal_wins": c1,
        "C2_reset_no_regression": c2,
        "C3_anneal_contrast": c3,
        "C4_no_anneal_contrast": c4,
        "C5_seed_consistent": c5,
    }


def _determine_outcome(criteria: Dict[str, bool]) -> str:
    c1 = criteria["C1_anneal_wins"]
    c2 = criteria["C2_reset_no_regression"]
    c3 = criteria["C3_anneal_contrast"]
    c4 = criteria["C4_no_anneal_contrast"]
    c5 = criteria["C5_seed_consistent"]

    if not c3 or not c4:
        return "FAIL"   # residue field does not build/retain usable structure
    if c1 and c5:
        return "PASS"   # ANNEAL beneficial; offline partial decay is useful
    # Partial outcomes
    return "PARTIAL"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_experiment(dry_run: bool = False) -> Dict:
    """Run all conditions and compile the result pack."""
    print("=== V3-EXQ-150: Q-005 Sleep Annealing of R(x,t) Discriminative Pair ===", flush=True)
    print(f"Conditions: {CONDITIONS}  Seeds: {SEEDS}", flush=True)
    print("Pre-registered thresholds:", flush=True)
    print(f"  C1 THRESH_ANNEAL_WIN       = {THRESH_ANNEAL_WIN}", flush=True)
    print(f"  C2 THRESH_NO_REGRESSION    = {THRESH_NO_REGRESSION}", flush=True)
    print(f"  C3 THRESH_CONTRAST_ANNEAL  = {THRESH_CONTRAST_ANNEAL}", flush=True)
    print(f"  C4 THRESH_CONTRAST_NO_ANNEAL = {THRESH_CONTRAST_NO_ANNEAL}", flush=True)
    print(f"  ANNEAL_FACTOR              = {ANNEAL_FACTOR}", flush=True)

    results_by_condition: Dict[str, List[Dict]] = {c: [] for c in CONDITIONS}

    for condition in CONDITIONS:
        print(f"\n=== Condition: {condition} ===", flush=True)
        for seed in SEEDS:
            result = _run_condition(
                seed=seed,
                condition=condition,
                episodes_per_phase=EPISODES_PER_PHASE,
                steps_per_episode=STEPS_PER_EPISODE,
                lr=LR,
                dry_run=dry_run,
            )
            results_by_condition[condition].append(result)

    print("\n=== Evaluating criteria ===", flush=True)
    criteria = _evaluate_criteria(results_by_condition)
    outcome = _determine_outcome(criteria)

    for k, v in criteria.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}", flush=True)
    print(f"Overall outcome: {outcome}", flush=True)

    # Summary metrics (mean over seeds, per condition)
    def _mean_over_seeds(cond: str, key: str) -> float:
        vals = [r[key] for r in results_by_condition[cond]]
        return float(sum(vals) / max(len(vals), 1))

    summary_metrics: Dict = {}
    for cond in CONDITIONS:
        prefix = cond.lower()
        summary_metrics[f"{prefix}_mean_harm_post_sleep"] = _mean_over_seeds(
            cond, "mean_harm_post_sleep"
        )
        summary_metrics[f"{prefix}_final_contrast"] = _mean_over_seeds(cond, "final_contrast")

    # Guidance and evidence direction
    if outcome == "PASS":
        guidance = "anneal_r_field_offline"
        evidence_direction = "supports"   # Q-005: offline annealing of R(x,t) is beneficial
    elif outcome == "PARTIAL":
        # Determine which partial
        no_a_harm = _mean_over_seeds("NO_ANNEAL", "mean_harm_post_sleep")
        ann_harm  = _mean_over_seeds("ANNEAL", "mean_harm_post_sleep")
        if no_a_harm <= ann_harm:
            guidance = "persistent_residue_preferred"
        else:
            guidance = "anneal_or_no_clear_winner"
        evidence_direction = "mixed"
    else:
        guidance = "residue_field_contrast_failed"
        evidence_direction = "mixed"

    run_id = (
        "v3_exq_150_q005_sleep_anneal_"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "_v3"
    )

    pack = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "claim_ids": CLAIM_IDS,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "outcome": outcome,
        "evidence_direction": evidence_direction,
        "evidence_class": "discriminative_pair",
        "guidance": guidance,
        "criteria": criteria,
        "pre_registered_thresholds": {
            "THRESH_ANNEAL_WIN": THRESH_ANNEAL_WIN,
            "THRESH_NO_REGRESSION": THRESH_NO_REGRESSION,
            "THRESH_CONTRAST_ANNEAL": THRESH_CONTRAST_ANNEAL,
            "THRESH_CONTRAST_NO_ANNEAL": THRESH_CONTRAST_NO_ANNEAL,
            "ANNEAL_FACTOR": ANNEAL_FACTOR,
        },
        "summary_metrics": summary_metrics,
        "protocol": {
            "waking_phases": WAKING_PHASES,
            "episodes_per_phase": EPISODES_PER_PHASE,
            "steps_per_episode": STEPS_PER_EPISODE,
            "sleep_phases": WAKING_PHASES - 1,
        },
        "seeds": SEEDS,
        "scenario": (
            "Three-condition sleep annealing test: NO_ANNEAL (residue persists unchanged),"
            " ANNEAL (RBF weights *= 0.3 after each waking phase, synaptic homeostasis analog),"
            " RESET (RBF weights zeroed after each waking phase, tabula rasa)."
            " Protocol: 3 waking phases x 100 episodes x 200 steps separated by 2 sleep phases."
            " Primary metric: mean_harm_post_sleep (mean over waking phases 2+3)."
            " 2 seeds x 3 conditions = 6 cells. CausalGridWorldV2 size=10 2 hazards 3 resources."
        ),
        "interpretation": (
            f"PASS => Q-005 guidance: ANNEAL (factor={ANNEAL_FACTOR}) is beneficial;"
            f" offline partial R(x,t) decay models sleep homeostasis correctly."
            f" PARTIAL (NO_ANNEAL) => Q-005 guidance: persistent residue is better;"
            f" no offline operation needed."
            f" PARTIAL (other) => Q-005 inconclusive; neither condition clearly better."
            f" FAIL => residue field does not build usable spatial contrast (C3/C4 failed)."
        ),
        "per_seed_results": {cond: results_by_condition[cond] for cond in CONDITIONS},
        "dry_run": dry_run,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    if not dry_run:
        out_dir = Path(__file__).resolve().parents[2] / "REE_assembly" / "evidence" / "experiments"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{run_id}.json"
        with open(out_path, "w") as f:
            json.dump(pack, f, indent=2)
        print(f"\nResult pack written to: {out_path}", flush=True)
    else:
        print("\n[dry_run] Result pack NOT written.", flush=True)
        print(json.dumps(pack, indent=2), flush=True)

    return pack


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    result = run_experiment(dry_run=dry_run)
    print(f"Done. Outcome: {result['outcome']}", flush=True)
