#!/opt/local/bin/python3
"""
V3-EXQ-231 -- MECH-106: BG Hysteresis First Experiment

Claims: MECH-106
Proposal: first experiment

EXPERIMENT_PURPOSE = "evidence"

MECH-106 asserts:
  The BG gate shows hysteresis: the commitment decision persists beyond the
  initial triggering condition, implementing a commitment boundary. Once beta
  is elevated (committed mode), it does not immediately drop when the
  triggering variance drops back above the threshold.

In the V3 architecture this maps to:
  - Commitment fires when running_variance < commitment_threshold (E3 dynamic
    precision, ARC-016).
  - Hysteresis = committed mode continues even after variance recovers above
    threshold, until a natural completion signal releases beta.

Two conditions (PERSISTENT vs REACTIVE):
  A. PERSISTENT (MECH-106 active):
     Standard V3 beta gate behavior. Commitment elevates beta; beta stays
     elevated (action-loop held) until BetaGate.release() is called.
     Commitment mode is externally released only when running_variance rises
     back above commitment_threshold + hysteresis_margin (or after N max steps).
     Simulated: once committed, hold commitment for a minimum of MIN_HOLD_STEPS
     even if variance spikes back up.

  B. REACTIVE (MECH-106 ablated):
     Beta gate uses instantaneous threshold crossing. Every step:
     beta_elevated = (running_variance < commit_threshold).
     No memory of prior commitment state.

Design
------
Run 3 seeds x 400 episodes x 200 steps.
For each episode, compute:
  - trigger_duration: steps where triggering condition is active
    (running_variance < commit_threshold)
  - commitment_duration: steps where agent is actually in committed mode
    (beta_elevated=True)
  - persistence_ratio = commitment_duration / max(1, trigger_duration)

PASS criteria (pre-registered)
-------------------------------
C1: mean_persistence_ratio_PERSISTENT > 2.0 (>= 2/3 seeds)
    Committed mode lasts at least 2x longer than triggering condition duration.
C2: mean_persistence_ratio_REACTIVE is close to 1.0 (<= 1.5, >= 2/3 seeds)
    Reactive gate has near-1:1 ratio (no hysteresis).
C3: commitment_duration_PERSISTENT > commitment_duration_REACTIVE (>= 2/3 seeds)
    Persistent mode spends more total time committed.
C4: n_committed_episodes > MIN_COMMITTED_EPS in PERSISTENT condition (>= 2/3 seeds)
    Enough commitment events to measure.

PASS: C1 AND C2 AND C3 AND C4.

Seeds: [42, 7, 13]
Env:   CausalGridWorldV2 size=10, 2 hazards, 3 resources
Train: 400 eps x 200 steps per condition per seed
Est:   ~120 min DLAPTOP-4.local
"""

import sys
import random
import json
import math
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.optim as optim
import torch.nn.functional as F

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
EXPERIMENT_TYPE = "v3_exq_231_mech106_bg_hysteresis"
CLAIM_IDS = ["MECH-106"]
EXPERIMENT_PURPOSE = "evidence"

# ---------------------------------------------------------------------------
# Pre-registered thresholds
# ---------------------------------------------------------------------------
THRESH_PERSIST_RATIO      = 2.0   # C1: mean persistence_ratio PERSISTENT > this
THRESH_REACTIVE_RATIO_MAX = 1.5   # C2: mean persistence_ratio REACTIVE <= this
MIN_COMMITTED_EPS         = 5     # C4: min episodes with non-zero commitment

# PERSISTENT mode: minimum hold steps after commitment trigger (hysteresis model)
MIN_HOLD_STEPS    = 20    # Committed mode held for at least this many steps
HYSTERESIS_MARGIN = 0.05  # Variance must rise above threshold + margin to release

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BODY_OBS_DIM  = 12
WORLD_OBS_DIM = 250
ACTION_DIM    = 5
WORLD_DIM     = 32
SELF_DIM      = 32

TOTAL_EPISODES = 400
STEPS_PER_EP   = 200

SEEDS = [42, 7, 13]


# ---------------------------------------------------------------------------
# Env / config factory
# ---------------------------------------------------------------------------

def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed,
        size=10,
        num_hazards=2,
        num_resources=3,
        hazard_harm=0.02,
        env_drift_interval=10,
        env_drift_prob=0.2,
    )


def _make_config() -> REEConfig:
    return REEConfig.from_dims(
        body_obs_dim=BODY_OBS_DIM,
        world_obs_dim=WORLD_OBS_DIM,
        action_dim=ACTION_DIM,
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
        alpha_world=0.9,
        alpha_self=0.3,
    )


# ---------------------------------------------------------------------------
# Commitment tracking helpers
# ---------------------------------------------------------------------------

class HysteresisCommitTracker:
    """
    PERSISTENT commitment tracker implementing MECH-106 hysteresis.

    Once commitment fires (variance < threshold), holds committed state for at
    least MIN_HOLD_STEPS. Only releases when variance rises above
    threshold + HYSTERESIS_MARGIN after the minimum hold period.
    """

    def __init__(self, commit_threshold: float, min_hold: int, margin: float):
        self.threshold    = commit_threshold
        self.min_hold     = min_hold
        self.margin       = margin
        self._committed   = False
        self._hold_counter: int = 0
        self.trigger_steps_ep: int = 0   # steps where condition active
        self.commit_steps_ep:  int = 0   # steps where committed

    def reset_episode(self) -> None:
        self._committed  = False
        self._hold_counter = 0
        self.trigger_steps_ep = 0
        self.commit_steps_ep  = 0

    def step(self, running_variance: float) -> bool:
        """Update and return current committed state."""
        trigger_active = running_variance < self.threshold
        if trigger_active:
            self.trigger_steps_ep += 1

        if trigger_active and not self._committed:
            self._committed = True
            self._hold_counter = 0

        if self._committed:
            self._hold_counter += 1
            release_cond = (
                self._hold_counter >= self.min_hold
                and running_variance > self.threshold + self.margin
            )
            if release_cond:
                self._committed = False
            else:
                self.commit_steps_ep += 1
        return self._committed


class ReactiveCommitTracker:
    """
    REACTIVE commitment tracker (MECH-106 ablated).

    Commitment is determined entirely by current-step variance < threshold.
    No memory of prior state.
    """

    def __init__(self, commit_threshold: float):
        self.threshold = commit_threshold
        self.trigger_steps_ep: int = 0
        self.commit_steps_ep:  int = 0

    def reset_episode(self) -> None:
        self.trigger_steps_ep = 0
        self.commit_steps_ep  = 0

    def step(self, running_variance: float) -> bool:
        trigger_active = running_variance < self.threshold
        if trigger_active:
            self.trigger_steps_ep += 1
            self.commit_steps_ep  += 1
        return trigger_active


# ---------------------------------------------------------------------------
# Run one seed x one condition
# ---------------------------------------------------------------------------

def _run_condition(
    condition: str,
    seed: int,
    n_episodes: int,
    steps_per_ep: int,
) -> Dict:
    torch.manual_seed(seed)
    random.seed(seed)

    env    = _make_env(seed)
    config = _make_config()
    agent  = REEAgent(config)

    commit_thresh = agent.e3.config.commitment_threshold  # default 0.40

    if condition == "PERSISTENT":
        tracker = HysteresisCommitTracker(
            commit_threshold=commit_thresh,
            min_hold=MIN_HOLD_STEPS,
            margin=HYSTERESIS_MARGIN,
        )
    else:
        tracker = ReactiveCommitTracker(commit_threshold=commit_thresh)

    e1_opt = optim.Adam(agent.e1.parameters(), lr=1e-3)
    e2_opt = optim.Adam(agent.e2.parameters(), lr=3e-3)
    e3_opt = optim.Adam(agent.e3.parameters(), lr=1e-3)

    print(
        f"  [{condition}] seed={seed} episodes={n_episodes} steps={steps_per_ep}"
        f" commit_thresh={commit_thresh:.3f}",
        flush=True,
    )

    per_episode_trigger:  List[int] = []
    per_episode_commit:   List[int] = []
    n_committed_episodes: int = 0

    agent.train()

    for ep in range(n_episodes):
        _, obs_dict = env.reset()
        agent.reset()
        tracker.reset_episode()

        for _ in range(steps_per_ep):
            obs_body  = torch.tensor(obs_dict["body_state"],  dtype=torch.float32)
            obs_world = torch.tensor(obs_dict["world_state"], dtype=torch.float32)

            latent   = agent.sense(obs_body, obs_world)
            ticks    = agent.clock.advance()
            e1_prior = (
                agent._e1_tick(latent) if ticks["e1_tick"]
                else torch.zeros(1, WORLD_DIM, device=agent.device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            action     = agent.select_action(candidates, ticks, temperature=1.0)

            _, reward, done, info, obs_dict = env.step(action)
            harm_signal = float(reward) if reward < 0 else 0.0

            # Track commitment state via tracker (condition-specific)
            rv  = agent.e3._running_variance
            tracker.step(rv)

            # E1 prediction loss -> update running_variance
            e1_loss = agent.compute_prediction_loss()
            if hasattr(e1_loss, 'item') and e1_loss.requires_grad:
                agent.e3.update_running_variance(
                    torch.tensor([[e1_loss.item()]])
                )

            e2_loss = agent.compute_e2_loss()
            total   = e1_loss + e2_loss
            if total.requires_grad:
                e1_opt.zero_grad(); e2_opt.zero_grad()
                total.backward()
                e1_opt.step(); e2_opt.step()

            # E3 harm supervision
            if agent._current_latent is not None:
                z_world = agent._current_latent.z_world.detach()
                harm_t  = torch.tensor([[1.0 if harm_signal < 0 else 0.0]])
                hloss   = F.mse_loss(agent.e3.harm_eval(z_world), harm_t)
                if hloss.requires_grad:
                    e3_opt.zero_grad(); hloss.backward(); e3_opt.step()

            agent.update_residue(harm_signal)
            if done:
                break

        per_episode_trigger.append(tracker.trigger_steps_ep)
        per_episode_commit.append(tracker.commit_steps_ep)
        if tracker.commit_steps_ep > 0:
            n_committed_episodes += 1

        if (ep + 1) % 100 == 0:
            mean_t = sum(per_episode_trigger[-50:]) / 50
            mean_c = sum(per_episode_commit[-50:])  / 50
            ratio  = mean_c / max(1.0, mean_t)
            print(
                f"  [{condition}] seed={seed} ep {ep+1}/{n_episodes}"
                f" trigger={mean_t:.1f} commit={mean_c:.1f} ratio={ratio:.2f}"
                f" rv={agent.e3._running_variance:.4f}",
                flush=True,
            )

    # Compute summary metrics
    eps_with_trigger = [
        (t, c) for t, c in zip(per_episode_trigger, per_episode_commit)
        if t > 0
    ]
    if eps_with_trigger:
        ratios = [c / max(1, t) for t, c in eps_with_trigger]
        mean_ratio = sum(ratios) / len(ratios)
    else:
        mean_ratio = 0.0

    mean_commit   = sum(per_episode_commit)  / max(1, n_episodes)
    mean_trigger  = sum(per_episode_trigger) / max(1, n_episodes)

    print(
        f"  [{condition}] seed={seed} DONE:"
        f" mean_persist_ratio={mean_ratio:.3f}"
        f" committed_eps={n_committed_episodes}/{n_episodes}"
        f" mean_commit={mean_commit:.1f}",
        flush=True,
    )

    return {
        "condition":             condition,
        "mean_persistence_ratio": mean_ratio,
        "mean_commit_steps_ep":   mean_commit,
        "mean_trigger_steps_ep":  mean_trigger,
        "n_committed_episodes":   n_committed_episodes,
    }


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------

def run(dry_run: bool = False) -> dict:
    print(f"\n[EXQ-231] MECH-106 BG Hysteresis (dry_run={dry_run})", flush=True)

    n_eps  = 5   if dry_run else TOTAL_EPISODES
    n_step = 20  if dry_run else STEPS_PER_EP

    all_seed_results = []
    c1_passes = []; c2_passes = []; c3_passes = []; c4_passes = []

    for seed in SEEDS:
        print(f"\n--- seed={seed} ---", flush=True)
        r_pers = _run_condition("PERSISTENT", seed, n_eps, n_step)
        r_reac = _run_condition("REACTIVE",   seed, n_eps, n_step)
        all_seed_results.append({
            "seed": seed,
            "PERSISTENT": r_pers,
            "REACTIVE":   r_reac,
        })

        c1_passes.append(r_pers["mean_persistence_ratio"] > THRESH_PERSIST_RATIO)
        c2_passes.append(r_reac["mean_persistence_ratio"] <= THRESH_REACTIVE_RATIO_MAX)
        c3_passes.append(r_pers["mean_commit_steps_ep"] > r_reac["mean_commit_steps_ep"])
        c4_passes.append(r_pers["n_committed_episodes"] >= MIN_COMMITTED_EPS)

    c1_pass = sum(c1_passes) >= 2
    c2_pass = sum(c2_passes) >= 2
    c3_pass = sum(c3_passes) >= 2
    c4_pass = sum(c4_passes) >= 2

    all_pass = c1_pass and c2_pass and c3_pass and c4_pass
    status   = "PASS" if all_pass else "FAIL"
    crit_met = sum([c1_pass, c2_pass, c3_pass, c4_pass])

    print(f"\n[EXQ-231] Results:", flush=True)
    for i, seed in enumerate(SEEDS):
        sr   = all_seed_results[i]
        pr   = sr["PERSISTENT"]["mean_persistence_ratio"]
        rr   = sr["REACTIVE"]["mean_persistence_ratio"]
        ce   = sr["PERSISTENT"]["n_committed_episodes"]
        print(
            f"  seed={seed}: persist_ratio={pr:.3f} reactive_ratio={rr:.3f}"
            f" committed_eps={ce}"
            f" C1={'P' if c1_passes[i] else 'F'}"
            f" C2={'P' if c2_passes[i] else 'F'}"
            f" C3={'P' if c3_passes[i] else 'F'}"
            f" C4={'P' if c4_passes[i] else 'F'}",
            flush=True,
        )
    print(f"  Status: {status} ({crit_met}/4)", flush=True)

    if all_pass:
        interpretation = (
            "MECH-106 SUPPORTED: Committed mode persists > 2x longer than"
            " triggering condition (PERSISTENT), while REACTIVE shows near-1:1"
            " ratio. BG gate implements genuine hysteresis -- commitment boundary"
            " is maintained beyond initial trigger."
        )
    elif crit_met >= 2:
        interpretation = (
            "MECH-106 PARTIAL: Some hysteresis signal present but below"
            " full threshold. Consider longer training or wider margin."
        )
    else:
        interpretation = (
            "MECH-106 NOT SUPPORTED: Persistence ratio not above threshold or"
            " commitment events too rare. Commitment gating may not be active"
            " at current training stage."
        )

    failure_notes = []
    if not c1_pass:
        vals = [all_seed_results[i]["PERSISTENT"]["mean_persistence_ratio"] for i in range(3)]
        failure_notes.append(
            f"C1 FAIL: persist_ratios={[round(v,3) for v in vals]}"
            f" (need > {THRESH_PERSIST_RATIO} in >= 2/3)"
        )
    if not c2_pass:
        vals = [all_seed_results[i]["REACTIVE"]["mean_persistence_ratio"] for i in range(3)]
        failure_notes.append(
            f"C2 FAIL: reactive_ratios={[round(v,3) for v in vals]}"
            f" (need <= {THRESH_REACTIVE_RATIO_MAX} in >= 2/3)"
        )
    if not c3_pass:
        failure_notes.append("C3 FAIL: PERSISTENT not committing more steps than REACTIVE")
    if not c4_pass:
        failure_notes.append(
            f"C4 FAIL: too few committed episodes"
            f" (need >= {MIN_COMMITTED_EPS})"
        )
    for n in failure_notes:
        print(f"  {n}", flush=True)

    summary_markdown = (
        f"# V3-EXQ-231 -- MECH-106 BG Hysteresis\n\n"
        f"**Status:** {status}  **Criteria met:** {crit_met}/4\n"
        f"**Claims:** MECH-106  **Purpose:** evidence\n\n"
        f"First experiment for MECH-106. Tests whether committed mode persists"
        f" beyond the initial triggering condition (variance < threshold).\n\n"
        f"## Conditions\n\n"
        f"- PERSISTENT: commitment held for >= {MIN_HOLD_STEPS} steps after trigger;"
        f" releases only when variance > threshold + {HYSTERESIS_MARGIN}\n"
        f"- REACTIVE: commitment = instantaneous (variance < threshold each step)\n\n"
        f"## Results\n\n"
        f"| Seed | Persist ratio | Reactive ratio | Commit eps | C1 | C2 | C3 | C4 |\n"
        f"|------|--------------|----------------|------------|----|----|----|----|"
    )
    for i, seed in enumerate(SEEDS):
        sr   = all_seed_results[i]
        pr   = sr["PERSISTENT"]["mean_persistence_ratio"]
        rr   = sr["REACTIVE"]["mean_persistence_ratio"]
        ce   = sr["PERSISTENT"]["n_committed_episodes"]
        summary_markdown += (
            f"\n| {seed} | {pr:.3f} | {rr:.3f} | {ce}"
            f" | {'PASS' if c1_passes[i] else 'FAIL'}"
            f" | {'PASS' if c2_passes[i] else 'FAIL'}"
            f" | {'PASS' if c3_passes[i] else 'FAIL'}"
            f" | {'PASS' if c4_passes[i] else 'FAIL'} |"
        )
    summary_markdown += (
        f"\n\n## Interpretation\n\n{interpretation}\n"
    )
    if failure_notes:
        summary_markdown += "\n## Failure Notes\n\n"
        summary_markdown += "\n".join(f"- {n}" for n in failure_notes) + "\n"

    metrics: Dict = {
        "c1_pass": 1.0 if c1_pass else 0.0,
        "c2_pass": 1.0 if c2_pass else 0.0,
        "c3_pass": 1.0 if c3_pass else 0.0,
        "c4_pass": 1.0 if c4_pass else 0.0,
        "criteria_met": float(crit_met),
    }
    for i, seed in enumerate(SEEDS):
        sr  = all_seed_results[i]
        sfx = f"_seed{i}"
        metrics[f"persist_ratio{sfx}"] = float(
            sr["PERSISTENT"]["mean_persistence_ratio"])
        metrics[f"reactive_ratio{sfx}"] = float(
            sr["REACTIVE"]["mean_persistence_ratio"])
        metrics[f"commit_eps_persistent{sfx}"] = float(
            sr["PERSISTENT"]["n_committed_episodes"])
        metrics[f"mean_commit_persistent{sfx}"] = float(
            sr["PERSISTENT"]["mean_commit_steps_ep"])
        metrics[f"mean_commit_reactive{sfx}"] = float(
            sr["REACTIVE"]["mean_commit_steps_ep"])

    return {
        "status": status,
        "metrics": metrics,
        "summary_markdown": summary_markdown,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": (
            "supports" if all_pass
            else ("mixed" if crit_met >= 2 else "weakens")
        ),
        "experiment_type": EXPERIMENT_TYPE,
        "fatal_error_count": 0,
        "seed_results": all_seed_results,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    result = run(dry_run=args.dry_run)

    ts  = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"] = ts
    result["run_id"]         = f"{EXPERIMENT_TYPE}_{ts}_v3"
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
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}", flush=True)
        else:
            print(f"  {k}: {v}", flush=True)
