#!/opt/local/bin/python3
"""
V3-EXQ-149 -- Q-004: tau_R Calibration Relative to E1/E2 Timescales

Claim:    Q-004
Proposal: EXP-0092 (EVB-0073)

Q-004 asks:
  "How should tau_R be calibrated relative to E1/E2 timescales?"

  tau_R is the effective integration timescale of the residue field R(x,t).
  In the current V3 implementation, tau_R is not a single explicit parameter --
  it emerges from the interaction of two ResidueConfig fields:
    - accumulation_rate:  how strongly each harm event writes into the RBF field
    - kernel_bandwidth:   spatial spread of each RBF basis function

  Together these determine how quickly the residue field becomes saturated after
  harm events (fast tau) vs. how slowly it integrates many weak events over time
  (slow tau). The open question is whether a FAST or SLOW tau is better calibrated
  to the E1/E2 timescales:
    - E2 forward model: rollout_horizon=30 steps (fast; predicts immediate consequences)
    - E1 LSTM predictor: prediction_horizon=20, integrates over many steps (slow context)

  If tau_R is too fast: the residue field saturates immediately after a single harm
  event and becomes uninformative (every hazard cell looks equally bad regardless
  of how recently the harm occurred or how frequently the agent visits it).

  If tau_R is too slow: harm events produce negligible residue signals -- the field
  never builds enough contrast to guide E3 trajectory evaluation or shape hippocampal
  navigation.

  The discriminative question: does FAST_TAU or SLOW_TAU produce lower cumulative
  harm? The winning calibration gives E3 more useful gradient signal for harm avoidance.
  A PARTIAL result (neither significantly better) is also meaningful: it would suggest
  that tau_R is not a critical free parameter within the range tested.

Experiment design:
  Two conditions, 2 seeds each (matched):

  FAST_TAU (matched to E2 forward-model timescale ~30 steps):
    accumulation_rate = 0.5   -- large accumulation per harm event
    kernel_bandwidth  = 0.5   -- narrow spatial influence (localised residue)
    Interpretation: residue field responds quickly and precisely to individual
    harm events. After ~2-3 hits, RBF centers at a hazard location reach high
    weight and clearly signal danger. E3 can detect danger with few exposures.

  SLOW_TAU (matched to E1 LSTM integration timescale ~200+ steps):
    accumulation_rate = 0.02  -- small accumulation per harm event
    kernel_bandwidth  = 2.0   -- wide spatial influence (diffuse residue)
    Interpretation: residue builds slowly over many harm events and spreads
    broadly across z_world space. Requires many exposures before the field
    builds enough contrast. E3 sees a slowly-increasing background signal.

  Control condition (DEFAULT_TAU):
    accumulation_rate = 0.1   -- current default
    kernel_bandwidth  = 1.0   -- current default
    Provides a baseline to detect monotone trends (e.g. if faster is always
    better, FAST > DEFAULT > SLOW would be expected).

  Three conditions, 2 seeds each = 6 cells total.

Pre-registered thresholds
--------------------------
C1: FAST_TAU reduces harm vs SLOW_TAU: mean_harm_fast < mean_harm_slow * THRESH_FAST_WIN
    (both seeds). Fast tau produces at least FAST_WIN_MIN fraction less harm than slow.
    Pre-registered: THRESH_FAST_WIN = 0.90 (fast must be >= 10% lower harm than slow).

C2: SLOW_TAU does not catastrophically fail: mean_harm_slow < mean_harm_fast * THRESH_SLOW_WIN
    (both seeds). Slow tau produces at least SLOW_WIN_MIN fraction less harm than fast.
    Pre-registered: THRESH_SLOW_WIN = 0.90 (slow must be >= 10% lower harm than fast).
    (C1 and C2 are mutually exclusive by definition if thresholds are symmetric.)

C3: DEFAULT_TAU ranks between FAST and SLOW: the ordering fast < default < slow OR
    slow < default < fast holds (both seeds). This checks that the calibration axis
    is monotone -- the default is not an outlier.

C4: Residue field contrast non-trivial: mean_residue_contrast > THRESH_CONTRAST in the
    FAST_TAU condition (both seeds). This confirms the field actually builds usable
    signal during training (mean RBF weight at harm locations / mean elsewhere > threshold).
    Pre-registered: THRESH_CONTRAST = 0.05.

C5: Seed consistency: C1 (or C2 if slow wins) agrees across both seeds (not one PASS /
    one FAIL).

PASS (FAST wins):    C1 + C4 + C5  => tau_R should track E2 timescale (fast, ~30 steps)
PASS (SLOW wins):    C2 + C4 + C5  => tau_R should track E1 timescale (slow, ~200 steps)
PARTIAL:             C4 only (no clear winner) => tau_R is not a critical parameter in this range
FAIL:                C4 fails (neither condition builds usable residue contrast)

PASS (fast) => Q-004 guidance: calibrate tau_R to E2 forward-model horizon (~30 steps).
PASS (slow) => Q-004 guidance: calibrate tau_R to E1 LSTM integration horizon (~200 steps).
PARTIAL     => Q-004 guidance: tau_R is robust in this range; default is acceptable.
FAIL        => implementation problem; accumulation_rate or bandwidth too extreme.

Conditions
----------
FAST_TAU:
  accumulation_rate = 0.5
  kernel_bandwidth  = 0.5
  Captures rapid per-event harm signaling.

SLOW_TAU:
  accumulation_rate = 0.02
  kernel_bandwidth  = 2.0
  Captures slow integrative harm context.

DEFAULT_TAU:
  accumulation_rate = 0.1
  kernel_bandwidth  = 1.0
  Current default -- calibration baseline.

Seeds: [42, 123] (matched -- same env seed per condition)
Env:   CausalGridWorldV2 size=10, 2 hazards, 3 resources, hazard_harm=0.05
Train: 300 episodes x 200 steps
Eval:  100 episodes x 200 steps
Estimated runtime: ~75 min any machine (3 conditions x 2 seeds x ~12 min each)
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


EXPERIMENT_TYPE = "v3_exq_149_q004_tau_r_calibration_pair"
CLAIM_IDS = ["Q-004"]

# ---------------------------------------------------------------------------
# Pre-registered thresholds
# ---------------------------------------------------------------------------
# C1: fast tau must produce this fraction LESS harm than slow tau to win
THRESH_FAST_WIN = 0.90

# C2: slow tau must produce this fraction LESS harm than fast tau to win
THRESH_SLOW_WIN = 0.90

# C4: residue contrast in FAST condition must exceed this
THRESH_CONTRAST = 0.05

# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------
BODY_OBS_DIM = 12
WORLD_OBS_DIM = 250   # CausalGridWorldV2 size=10
ACTION_DIM = 5
WORLD_DIM = 32

SEEDS = [42, 123]
WARMUP_EPISODES = 300
EVAL_EPISODES = 100
STEPS_PER_EPISODE = 200
LR = 3e-4

# tau_R condition parameters
CONDITIONS = {
    "FAST_TAU": {
        "accumulation_rate": 0.5,
        "kernel_bandwidth": 0.5,
    },
    "DEFAULT_TAU": {
        "accumulation_rate": 0.1,
        "kernel_bandwidth": 1.0,
    },
    "SLOW_TAU": {
        "accumulation_rate": 0.02,
        "kernel_bandwidth": 2.0,
    },
}


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed,
        size=10,
        num_hazards=2,
        num_resources=3,
        hazard_harm=0.05,   # slightly higher than default to give residue a clear signal
        env_drift_interval=15,
        env_drift_prob=0.2,
    )


# ---------------------------------------------------------------------------
# Config factory
# ---------------------------------------------------------------------------

def _make_config(accumulation_rate: float, kernel_bandwidth: float) -> REEConfig:
    cfg = REEConfig()
    cfg.latent.world_dim = WORLD_DIM
    cfg.latent.self_dim = 16
    cfg.latent.body_obs_dim = BODY_OBS_DIM
    cfg.latent.world_obs_dim = WORLD_OBS_DIM
    cfg.latent.observation_dim = BODY_OBS_DIM + WORLD_OBS_DIM
    cfg.latent.alpha_world = 0.95   # SD-008
    cfg.latent.alpha_self = 0.3
    cfg.residue.world_dim = WORLD_DIM
    cfg.residue.accumulation_rate = accumulation_rate
    cfg.residue.kernel_bandwidth = kernel_bandwidth
    cfg.residue.num_basis_functions = 32
    cfg.residue.decay_rate = 0.0    # invariant: residue cannot be erased
    cfg.residue.benefit_terrain_enabled = False  # not needed for this test
    return cfg


# ---------------------------------------------------------------------------
# Run one condition x seed
# ---------------------------------------------------------------------------

def _run_condition(
    seed: int,
    condition: str,
    warmup_episodes: int,
    eval_episodes: int,
    steps_per_episode: int,
    lr: float,
    dry_run: bool,
) -> Dict:
    """Run one condition for one seed. Returns per-seed result dict."""
    if dry_run:
        warmup_episodes = 2
        eval_episodes = 2
        steps_per_episode = 10

    params = CONDITIONS[condition]
    torch.manual_seed(seed)
    random.seed(seed)

    env = _make_env(seed)
    cfg = _make_config(
        accumulation_rate=params["accumulation_rate"],
        kernel_bandwidth=params["kernel_bandwidth"],
    )
    agent = REEAgent(cfg)
    agent.train()

    e1_opt = optim.Adam(agent.e1.parameters(), lr=lr)
    e3_opt = optim.Adam(agent.e3.parameters(), lr=lr)

    print(
        f"  [{condition}] seed={seed}"
        f" warmup={warmup_episodes} eval={eval_episodes}"
        f" steps={steps_per_episode}"
        f" accum_rate={params['accumulation_rate']}"
        f" bandwidth={params['kernel_bandwidth']}",
        flush=True,
    )

    # -----------------------------------------------------------------------
    # Phase 1: Warmup training
    # -----------------------------------------------------------------------
    for ep in range(warmup_episodes):
        _, obs_dict = env.reset()
        agent.reset()

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
                if harm_signal < 0.0:
                    z_world_np = agent._current_latent.z_world.detach()
                    agent.residue_field.accumulate(
                        z_world_np,
                        harm_magnitude=abs(harm_signal),
                        hypothesis_tag=False,
                    )

                # E3 harm eval training: supervised by harm signal
                harm_target = torch.tensor([abs(harm_signal)], dtype=torch.float32)
                # harm_pred shape: [1] or [1, 1] -- flatten to 1-D for loss
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

        if (ep + 1) % 50 == 0:
            print(f"    warmup ep {ep+1}/{warmup_episodes}", flush=True)

    # -----------------------------------------------------------------------
    # Phase 2: Evaluation (no weight updates)
    # -----------------------------------------------------------------------
    agent.eval()

    eval_harms: List[float] = []
    eval_residue_at_harm: List[float] = []
    eval_residue_elsewhere: List[float] = []

    for ep in range(eval_episodes):
        _, obs_dict = env.reset()
        agent.reset()
        ep_harm = 0.0

        for _step in range(steps_per_episode):
            obs_body  = torch.tensor(obs_dict["body_state"],  dtype=torch.float32)
            obs_world = torch.tensor(obs_dict["world_state"], dtype=torch.float32)

            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world)
                agent.clock.advance()

                action_idx = random.randint(0, ACTION_DIM - 1)
                action = torch.zeros(1, ACTION_DIM)
                action[0, action_idx] = 1.0
                _, reward, done, _, obs_dict = env.step(action)

                harm_signal = float(min(0.0, reward))
                ep_harm += abs(harm_signal)

                if agent._current_latent is not None:
                    z_world_cur = agent._current_latent.z_world.detach()
                    residue_val = float(agent.residue_field.evaluate(z_world_cur).item())
                    if harm_signal < 0.0:
                        eval_residue_at_harm.append(residue_val)
                    else:
                        eval_residue_elsewhere.append(residue_val)

                if done:
                    _, obs_dict = env.reset()
                    agent.reset()

        eval_harms.append(ep_harm)

    mean_harm = float(sum(eval_harms) / max(len(eval_harms), 1))

    # Residue contrast: how much higher is residue at harm locations vs elsewhere?
    mean_at_harm = float(sum(eval_residue_at_harm) / max(len(eval_residue_at_harm), 1))
    mean_elsewhere = float(sum(eval_residue_elsewhere) / max(len(eval_residue_elsewhere), 1))
    residue_contrast = max(0.0, mean_at_harm - mean_elsewhere)

    residue_stats = agent.residue_field.get_statistics()
    total_residue = float(residue_stats["total_residue"].item())
    num_harm_events = int(residue_stats["num_harm_events"].item())
    active_centers = int(residue_stats["active_centers"].item())

    print(
        f"  [{condition}] seed={seed} mean_harm={mean_harm:.4f}"
        f" residue_contrast={residue_contrast:.4f}"
        f" total_residue={total_residue:.3f}"
        f" harm_events={num_harm_events}",
        flush=True,
    )

    return {
        "condition": condition,
        "seed": seed,
        "mean_harm": mean_harm,
        "residue_contrast": residue_contrast,
        "mean_residue_at_harm": mean_at_harm,
        "mean_residue_elsewhere": mean_elsewhere,
        "total_residue": total_residue,
        "num_harm_events": num_harm_events,
        "active_centers": active_centers,
        "n_eval_episodes": len(eval_harms),
        "accumulation_rate": params["accumulation_rate"],
        "kernel_bandwidth": params["kernel_bandwidth"],
    }


# ---------------------------------------------------------------------------
# Criterion evaluation
# ---------------------------------------------------------------------------

def _evaluate_criteria(
    results_by_condition: Dict[str, List[Dict]]
) -> Dict[str, bool]:
    """
    Evaluate pre-registered criteria across all conditions and seeds.

    Returns dict of criterion_id -> bool.
    """
    fast_results  = results_by_condition["FAST_TAU"]
    slow_results  = results_by_condition["SLOW_TAU"]
    def_results   = results_by_condition["DEFAULT_TAU"]

    # C1: FAST_TAU wins over SLOW_TAU (both seeds)
    c1_seeds = []
    for i in range(len(fast_results)):
        c1_seeds.append(fast_results[i]["mean_harm"] < slow_results[i]["mean_harm"] * THRESH_FAST_WIN)
    c1 = all(c1_seeds)

    # C2: SLOW_TAU wins over FAST_TAU (both seeds)
    c2_seeds = []
    for i in range(len(slow_results)):
        c2_seeds.append(slow_results[i]["mean_harm"] < fast_results[i]["mean_harm"] * THRESH_SLOW_WIN)
    c2 = all(c2_seeds)

    # C3: DEFAULT_TAU ranks between FAST and SLOW on mean_harm (both seeds)
    c3_seeds = []
    for i in range(len(def_results)):
        fast_h = fast_results[i]["mean_harm"]
        slow_h = slow_results[i]["mean_harm"]
        def_h  = def_results[i]["mean_harm"]
        # monotone: fast < default < slow OR slow < default < fast
        monotone = (fast_h <= def_h <= slow_h) or (slow_h <= def_h <= fast_h)
        c3_seeds.append(monotone)
    c3 = all(c3_seeds)

    # C4: FAST_TAU residue contrast > THRESH_CONTRAST (both seeds)
    c4_seeds = [r["residue_contrast"] > THRESH_CONTRAST for r in fast_results]
    c4 = all(c4_seeds)

    # C5: C1 (or C2) agrees across both seeds (no mixed seed result)
    # C5 checks that whichever direction wins in C1/C2, it does so for both seeds
    if c1:
        # C1 passed -- check seed consistency for fast winning
        c5_seeds = [fast_results[i]["mean_harm"] < slow_results[i]["mean_harm"] for i in range(len(fast_results))]
        c5 = all(c5_seeds)
    elif c2:
        # C2 passed -- check seed consistency for slow winning
        c5_seeds = [slow_results[i]["mean_harm"] < fast_results[i]["mean_harm"] for i in range(len(slow_results))]
        c5 = all(c5_seeds)
    else:
        # No clear directional winner -- C5 checks that neither condition dominates either seed
        # (partial result: consistent across seeds means no seed-level flip)
        c5_seeds = []
        for i in range(len(fast_results)):
            # "consistent" means direction is same across seeds (even if threshold not met)
            c5_seeds.append(
                (fast_results[i]["mean_harm"] < slow_results[i]["mean_harm"]) ==
                (fast_results[0]["mean_harm"] < slow_results[0]["mean_harm"])
            )
        c5 = all(c5_seeds)

    return {
        "C1_fast_wins": c1,
        "C2_slow_wins": c2,
        "C3_default_ranked": c3,
        "C4_fast_contrast": c4,
        "C5_seed_consistent": c5,
    }


def _determine_outcome(criteria: Dict[str, bool]) -> str:
    c1 = criteria["C1_fast_wins"]
    c2 = criteria["C2_slow_wins"]
    c4 = criteria["C4_fast_contrast"]
    c5 = criteria["C5_seed_consistent"]

    if not c4:
        return "FAIL"   # fast tau doesn't build usable residue contrast
    if c1 and c5:
        return "PASS"   # fast tau wins -- calibrate to E2 timescale
    if c2 and c5:
        return "PASS"   # slow tau wins -- calibrate to E1 timescale
    return "PARTIAL"    # no clear winner; tau_R robust in this range


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_experiment(dry_run: bool = False) -> Dict:
    """Run all conditions and compile the result pack."""
    print("=== V3-EXQ-149: Q-004 tau_R Calibration Discriminative Pair ===", flush=True)
    print(f"Conditions: {list(CONDITIONS.keys())}  Seeds: {SEEDS}", flush=True)
    print(f"Pre-registered thresholds:", flush=True)
    print(f"  C1 THRESH_FAST_WIN    = {THRESH_FAST_WIN}", flush=True)
    print(f"  C2 THRESH_SLOW_WIN    = {THRESH_SLOW_WIN}", flush=True)
    print(f"  C4 THRESH_CONTRAST    = {THRESH_CONTRAST}", flush=True)

    results_by_condition: Dict[str, List[Dict]] = {k: [] for k in CONDITIONS}

    for condition in CONDITIONS:
        print(f"\n--- Condition: {condition} ---", flush=True)
        for seed in SEEDS:
            result = _run_condition(
                seed=seed,
                condition=condition,
                warmup_episodes=WARMUP_EPISODES,
                eval_episodes=EVAL_EPISODES,
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

    summary_metrics = {}
    for cond in CONDITIONS:
        prefix = cond.lower()
        summary_metrics[f"{prefix}_mean_harm"] = _mean_over_seeds(cond, "mean_harm")
        summary_metrics[f"{prefix}_residue_contrast"] = _mean_over_seeds(cond, "residue_contrast")
        summary_metrics[f"{prefix}_total_residue"] = _mean_over_seeds(cond, "total_residue")
        summary_metrics[f"{prefix}_num_harm_events"] = _mean_over_seeds(cond, "num_harm_events")

    # Direction determination (for governance)
    if outcome == "PASS":
        if criteria["C1_fast_wins"]:
            guidance = "fast_tau_preferred"
            evidence_direction = "supports"   # Q-004: calibrate to E2 timescale
        else:
            guidance = "slow_tau_preferred"
            evidence_direction = "supports"   # Q-004: calibrate to E1 timescale
    elif outcome == "PARTIAL":
        guidance = "tau_R_robust_in_range"
        evidence_direction = "mixed"
    else:
        guidance = "accumulation_pathological"
        evidence_direction = "mixed"

    run_id = f"v3_exq_149_q004_tau_r_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_v3"

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
            "THRESH_FAST_WIN": THRESH_FAST_WIN,
            "THRESH_SLOW_WIN": THRESH_SLOW_WIN,
            "THRESH_CONTRAST": THRESH_CONTRAST,
        },
        "summary_metrics": summary_metrics,
        "condition_params": CONDITIONS,
        "seeds": SEEDS,
        "scenario": (
            "Three-condition tau_R calibration test: FAST_TAU (accum=0.5, bw=0.5),"
            " DEFAULT_TAU (accum=0.1, bw=1.0), SLOW_TAU (accum=0.02, bw=2.0)."
            " 2 seeds x 3 conditions = 6 cells. 300 warmup + 100 eval eps x 200 steps."
            " CausalGridWorldV2 size=10 2 hazards 3 resources."
        ),
        "interpretation": (
            f"PASS (fast) => calibrate tau_R to E2 forward-model timescale (~30 steps)."
            f" PASS (slow) => calibrate tau_R to E1 LSTM integration timescale (~200 steps)."
            f" PARTIAL => tau_R is not a critical free parameter in [0.02, 0.5] range."
            f" FAIL => pathological accumulation (contrast < {THRESH_CONTRAST})."
        ),
        "per_seed_results": {
            cond: results_by_condition[cond]
            for cond in CONDITIONS
        },
        "dry_run": dry_run,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    if not dry_run:
        # Write result pack to REE_assembly evidence directory
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
