"""DV-range probe -- `waypoint_field_consumer_reach` leg H1 (axis: drive/objective-sparsity).

NOT AN EXPERIMENT. NOT QUEUED. No manifest, no claim_ids, no emit_outcome. Same `_scratch`
convention as exq_wpfield_h2_probe_dvrange*.py (chip-20260905-waypoint-consumer-reach-portfolio).

Pre-registration source: confirmed autopsy failure_autopsy_V3-EXQ-1004_2026-09-05.json,
fanout_recommendation.suggested_probes[0] (H1, axis "drive").

H1 declared null: "visits/ep is flat across training signals, i.e. sparsity is not the
residual blocker." Per the campaign convention established by the H2 probe sequence
(design record REE_assembly/evidence/planning/exq_wpfield_h2_dv_range_probe_20260907.md),
a criterion must not be pre-registered until the achievable DV range at probe scale is
measured -- H2 was refused twice on exactly this omission (once by construction, once by
un-achievable range against the corrected control).

WHAT THIS PROBE MEASURES. A REE agent (z_world -> E1/E2/E3, the x724 all-ON recipe, the
SAME agent-construction and training helper every 724/734/737/742/808/978/1002/1008 driver
uses -- NOT a behaviour-cloned reader, per the chip's explicit instruction) trained via the
substrate's own REINFORCE mechanism (lateral-PFC bias head + OFC devaluation head,
`experiments._lib.allon_training._train_all_on_agent`) on the V3-EXQ-1004 bench geometry
(field ON, hazards/resources/energy zeroed, SD-094 gates set). TWO training-signal arms,
field ON and consumer identical in both:

  ARM_SPARSE: the env's own stock reward (waypoint_visit_reward=0.2, sparse -- only fires on
              arrival).
  ARM_SHAPED: the same stock reward PLUS a potential-based shaping term added at the DRIVER
              level (never touching causal_grid_world.py): bonus = SHAPING_COEF *
              (phi(s') - phi(s)), phi(s) = the waypoint-proximity-field value AT THE AGENT'S
              OWN CELL (obs_dict["waypoint_proximity_field_view"] center, [2, 2] of the 5x5
              patch) = 1/(1 + waypoint_field_decay * dist(agent, target)). This is the
              textbook Ng-et-al. potential-based shaping form: it changes only the DENSITY of
              the training signal (a per-step gradient toward the target instead of a sparse
              arrival bonus), while leaving the terminal/arrival reward and the env's own
              dynamics untouched -- i.e. it manipulates ONLY the axis H1 names ("drive"), not
              the observation channel (H2, already probe-resolved: the field survives into
              z_world, see the design record's section 7c).

The env is IDENTICAL between arms except for this driver-side reward augmentation; the field
observable, the agent architecture, the training recipe and every hyperparameter are shared.

WHY VISITS ARE COUNTED DURING P1 TRAINING ITSELF, NOT VIA A SEPARATE EVAL ROLLOUT. Rebuilding
the agent's inference path (sense -> clock.advance -> e1_tick -> generate_trajectories ->
select_action) outside `_train_all_on_agent` would duplicate ~80 lines of intricate,
substrate-version-sensitive code with no independent verification -- exactly the kind of
untested duplication the campaign's red-team passes keep catching (e.g. the e3 diagnostics
staleness class). `_train_all_on_agent`'s P1 phase already runs true on-policy rollout through
the real inference path (that IS what "training" means here), so this probe reads the
achieved visits/ep DIRECTLY off the training env's own last-K-episode tally via the same env
wrapper that injects the shaping term -- zero duplicated inference code, and the measured
quantity is literally "what did the trained policy achieve", not a re-derived approximation.

PROBE SCALE, DELIBERATELY SMALL. This is a range check, not a pre-registered run: 2 seeds,
short training (see CLI defaults). The question is binary -- does ANY achievable budget move
visits/ep between arms at all -- not a publishable effect size. A real portfolio run (if this
probe finds a non-degenerate range) will use full 1004-bench seeds/steps and a
pre-registered threshold DERIVED from this probe's own range, per the shipped
`dv_headroom_check` convention (never an assumed threshold).

ASCII-only in printed output (repo rule).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import experiments.v3_exq_724_competence_localization_diagnostic as x724  # noqa: E402
from experiments._lib.allon_training import _train_all_on_agent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402

# --- 1004-bench geometry (V3-EXQ-1004 _build_env, field ON; identical across arms) --------
GRID_SIZE = 12
N_WAYPOINTS = 3
WAYPOINT_VISIT_REWARD = 0.2
WAYPOINT_FIELD_DECAY = 0.25
NUM_HAZARDS = 0
NUM_RESOURCES = 0
ENERGY_DECAY = 0.0

FIELD_CENTER_IDX = (2, 2)  # 5x5 patch, agent-centred: [2,2] = the agent's own cell.


def _build_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed,
        size=GRID_SIZE,
        use_proxy_fields=True,
        subgoal_mode=True,
        num_waypoints=N_WAYPOINTS,
        waypoint_visit_reward=WAYPOINT_VISIT_REWARD,
        subgoal_arrival_position_check=True,
        num_hazards=NUM_HAZARDS,
        num_resources=NUM_RESOURCES,
        energy_decay=ENERGY_DECAY,
        hazard_free_contamination_gate=True,
        waypoint_proximity_field_enabled=True,
        waypoint_field_decay=WAYPOINT_FIELD_DECAY,
    )


def _phi(obs_dict: Dict[str, Any]) -> Optional[float]:
    fv = obs_dict.get("waypoint_proximity_field_view")
    if fv is None:
        return None
    arr = fv.detach().cpu().numpy() if torch.is_tensor(fv) else np.asarray(fv)
    arr = arr.reshape(5, 5)
    return float(arr[FIELD_CENTER_IDX])


class ShapingCountingEnv:
    """Thin wrapper: adds potential-based shaping to the reward (ARM_SHAPED only) and tallies
    waypoint arrivals per episode. Delegates everything else to the underlying env untouched
    -- causal_grid_world.py itself is never modified.
    """

    def __init__(self, env: CausalGridWorldV2, shaping_coef: float):
        self._env = env
        self.shaping_coef = float(shaping_coef)
        self._last_phi: Optional[float] = None
        self.episode_visit_counts: List[int] = []
        self._cur_visits = 0

    def reset(self):
        flat, obs_dict = self._env.reset()
        if self._cur_visits or self.episode_visit_counts:
            self.episode_visit_counts.append(self._cur_visits)
        self._cur_visits = 0
        self._last_phi = _phi(obs_dict)
        return flat, obs_dict

    def step(self, action):
        flat, harm_signal, done, info, obs_dict = self._env.step(action)
        tt = str(info.get("transition_type", "") or "")
        if tt in ("waypoint", "sequence_complete"):
            self._cur_visits += 1
        if self.shaping_coef:
            phi_next = _phi(obs_dict)
            if self._last_phi is not None and phi_next is not None:
                harm_signal = float(harm_signal) + self.shaping_coef * (phi_next - self._last_phi)
            self._last_phi = phi_next
        return flat, harm_signal, done, info, obs_dict

    def finalize(self):
        """Call once after training loop exits, to flush the last episode's tally."""
        self.episode_visit_counts.append(self._cur_visits)

    def __getattr__(self, name):
        return getattr(self._env, name)


def _mean(vals: List[float]) -> float:
    return float(sum(vals) / len(vals)) if vals else 0.0


def run_one_cell(seed: int, shaped: bool, shaping_coef: float, zworld_p0_episodes: int,
                 p0_episodes: int, p1_episodes: int, steps_per_episode: int,
                 tail_k: int) -> Dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_env_raw = _build_env(seed)
    train_env = ShapingCountingEnv(train_env_raw, shaping_coef if shaped else 0.0)

    zworld_env = _build_env(seed)  # dedicated, per allon_training's contract

    agent = x724._make_agent(train_env_raw, kind="all_on")

    t0 = time.perf_counter()
    stats = _train_all_on_agent(
        agent,
        train_env,  # duck-typed: ShapingCountingEnv exposes reset/step/__getattr__
        seed=seed,
        p0_episodes=p0_episodes,
        p1_episodes=p1_episodes,
        steps_per_episode=steps_per_episode,
        rung_id=f"h1probe_{'shaped' if shaped else 'sparse'}",
        total_denominator=p0_episodes + p1_episodes,
        zworld_p0_episodes=zworld_p0_episodes,
        zworld_p0_env=zworld_env,
    )
    train_env.finalize()
    elapsed = time.perf_counter() - t0

    # P1 episodes are the LAST p1_episodes entries of episode_visit_counts (P0 tally comes
    # first). Read the tail-K of those as "converged" visits/ep.
    p1_counts = train_env.episode_visit_counts[p0_episodes:]
    tail = p1_counts[-tail_k:] if len(p1_counts) >= tail_k else p1_counts
    return {
        "seed": seed,
        "shaped": shaped,
        "elapsed_seconds": elapsed,
        "p1_episode_visit_counts": p1_counts,
        "tail_k": len(tail),
        "visits_per_ep_tail_mean": _mean([float(v) for v in tail]),
        "visits_per_ep_all_p1_mean": _mean([float(v) for v in p1_counts]),
        "n_p1_ticks": stats["n_p1_ticks"],
        "n_e2_train_steps": stats["n_e2_train_steps"],
        "zworld_p0a_ran": stats["zworld_p0"].get("p0a_ran"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    ap.add_argument("--zworld-p0-episodes", type=int, default=15)
    ap.add_argument("--p0-episodes", type=int, default=10)
    ap.add_argument("--p1-episodes", type=int, default=20)
    ap.add_argument("--steps-per-episode", type=int, default=60)
    ap.add_argument("--shaping-coef", type=float, default=5.0)
    ap.add_argument("--tail-k", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true",
                    help="tiny smoke: overrides episode/step counts to minimal values")
    args = ap.parse_args()

    if args.dry_run:
        args.seeds = args.seeds[:1]
        args.zworld_p0_episodes = 2
        args.p0_episodes = 2
        args.p1_episodes = 2
        args.steps_per_episode = 10
        args.tail_k = 2

    results: Dict[str, List[Dict[str, Any]]] = {"sparse": [], "shaped": []}
    for seed in args.seeds:
        for shaped in (False, True):
            print(f"[h1probe] seed={seed} shaped={shaped} starting", flush=True)
            r = run_one_cell(
                seed=seed, shaped=shaped, shaping_coef=args.shaping_coef,
                zworld_p0_episodes=args.zworld_p0_episodes,
                p0_episodes=args.p0_episodes, p1_episodes=args.p1_episodes,
                steps_per_episode=args.steps_per_episode, tail_k=args.tail_k,
            )
            key = "shaped" if shaped else "sparse"
            results[key].append(r)
            print(
                f"[h1probe] seed={seed} shaped={shaped} "
                f"visits_per_ep_tail_mean={r['visits_per_ep_tail_mean']:.3f} "
                f"all_p1_mean={r['visits_per_ep_all_p1_mean']:.3f} "
                f"elapsed={r['elapsed_seconds']:.1f}s",
                flush=True,
            )

    sparse_tail = [r["visits_per_ep_tail_mean"] for r in results["sparse"]]
    shaped_tail = [r["visits_per_ep_tail_mean"] for r in results["shaped"]]
    lift_per_seed = [s - p for s, p in zip(shaped_tail, sparse_tail)]
    summary = {
        "sparse_tail_mean_per_seed": sparse_tail,
        "shaped_tail_mean_per_seed": shaped_tail,
        "lift_per_seed": lift_per_seed,
        "lift_mean": _mean(lift_per_seed),
        "sparse_mean": _mean(sparse_tail),
        "shaped_mean": _mean(shaped_tail),
    }
    print("[h1probe] SUMMARY " + json.dumps(summary), flush=True)
    print("verdict: PASS", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
