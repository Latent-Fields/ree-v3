"""Canonical OFF/baseline arm + shared cell runner for the ARC-071 chunking lineage.

Arm-reuse Phase 0 (instrument-only). Design plan:
REE_assembly/evidence/planning/arm_reuse_fingerprint_plan.md (sections 2, 7b, 9.7).

WHAT THIS MODULE IS
-------------------
The single source of truth for the OFF / baseline arm of the ARC-071
policy-chunking substrate-readiness lineage (V3-EXQ-810a -> 810b -> ...), and
for the CELL LOOP every arm of that lineage runs. Both live here so that (a) a
later sibling constructs its OFF arm from the same bytes -- which is what makes
the arm fingerprint match by construction rather than by luck -- and (b) the
treatment arms cannot silently drift away from the OFF arm's loop, because
there is only one loop.

The module is matched by the arm-fingerprint substrate glob
`experiments/_lib/**/*.py`, so any edit here correctly flips the substrate hash
and REFUSES a stale reuse.

WHY THE LOOP LIVES HERE AND NOT IN THE DRIVER (the 810 lesson)
--------------------------------------------------------------
V3-EXQ-810 hand-rolled a REDUCED agent loop in its driver:
  sense -> clock.advance -> _e1_tick -> generate_trajectories -> select_action
  -> env.step -> note_chunk_outcome
It never called `update_residue`, and MECH-091's salient-event
`clock.phase_reset()` lives INSIDE `REEAgent.update_residue`. So no salient
event could ever shorten the E3 interval, the E3 tick was PERFECTLY periodic,
and (with `e3_steps_per_tick = 10`) a 24-step episode yielded exactly THREE
recorded symbols -- measured buf = 3.00 on every seed. `note_outcome` breaks
out at `len(actions) < size`, so chunk sizes 4 and 5 were STRUCTURALLY
UNREACHABLE and the whole size/depth budget was inert. This module therefore
drives the CANONICAL loop via `experiments/_harness.StepHarness`, whose
invariant 3 pins exactly one `update_residue` per env step.

Note the second half of that fix is the ENV, not the loop: `phase_reset()` is
reached only on `harm_signal < 0`, so `num_hazards = 0` (810's setting) would
have kept the tick periodic even under the full loop. `ENV_KWARGS` here carries
hazards for that reason, and it is a load-bearing part of the OFF slice.

THE CONTRACT
------------
A future 810b/810c that wants to REUSE a baseline minted from this module MUST
build its OFF arm by importing `off_arm_flags` / `run_cell` from here, not by
re-deriving the OFF path inline. The OFF cell's computation is the conjunction
of:
  * ENV_KWARGS               -- 8x8 proxy-field grid WITH hazards
  * SCHEDULE                 -- N_EPISODES x STEPS_PER_EPISODE
  * CHUNK_PARAMS             -- the probe-scaled R_min / window / C_min, which
                                are passed to REEConfig in EVERY arm (they are
                                inert when chunking is off, but they are part of
                                the config the cell is built from)
  * ALPHA_WORLD              -- SD-008 z_world fidelity, all arms
  * off_arm_flags()          -- the OFF arm's own switches (chunking off)
It does NOT depend on the ON arms' switches, the acceptance thresholds, or the
arm labels, so none of those appear in `off_path_config_slice()`.

ASCII-only output (repo rule).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import sys

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ree_core.utils.config import REEConfig  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from experiments._harness import StepHarness  # noqa: E402

# --- Canonical OFF-path constants -------------------------------------------

# num_hazards > 0 is LOAD-BEARING, not a taste call: harm_signal < 0 is the sole
# trigger for MECH-091's clock.phase_reset() (ree_core/agent.py, inside
# update_residue), so a hazard-free grid leaves the E3 tick perfectly periodic
# however the loop is driven. 810 ran num_hazards=0 and measured exactly 3.00
# symbols per trial on every seed as a result.
ENV_KWARGS: Dict[str, Any] = {
    "size": 8,
    "num_hazards": 2,
    "num_resources": 6,
    "use_proxy_fields": True,
}

# 72 steps, not 810's 24. E3 ticks per episode set the accumulator buffer, and
# the buffer must reach chunk_max_size for the size range to be reachable at
# all. At e3_steps_per_tick = 10 a 72-step episode gives ~8 periodic symbols
# (step 1 plus steps 10..70) before any phase-reset extras.
N_EPISODES = 120
STEPS_PER_EPISODE = 72
SCHEDULE: Dict[str, int] = {
    "n_episodes": N_EPISODES,
    "steps_per_episode": STEPS_PER_EPISODE,
}

# Chunking parameters, scaled DOWN from the registered defaults (R_min 20 /
# W 100 / C_min 5) to fit a 120-episode probe, exactly as V3-EXQ-810 did. The
# RATIOS and every structural property (hysteresis gap, evaluative gate, depth
# cap, size budget) are preserved; only the absolute counts are reduced.
#
# Do NOT lower CHUNK_MIN_REPETITIONS further. V3-EXQ-810's autopsy measured
# repetition reaching 37 against a bar of 5, with the variance gate passing
# too: repetition was never the binding constraint, so lowering it cannot help
# and would only manufacture the conjunction the gate exists to test.
CHUNK_MIN_REPETITIONS = 5
CHUNK_WINDOW_TRIALS = 60
CHUNK_CRYSTALLISATION_MIN = 2

# Substrate defaults, restated here because the readiness preconditions are
# expressed against them (ree_core/utils/config.py: chunk_min_size / chunk_max_size).
CHUNK_MIN_SIZE = 2
CHUNK_MAX_SIZE = 5

ALPHA_WORLD = 0.9  # SD-008: z_world fidelity


def off_arm_flags() -> Dict[str, Any]:
    """The OFF/baseline arm's own switches. Chunking never instantiated."""
    return {
        "use_policy_chunking": False,
        "use_chunk_maintenance": False,
        "use_chunk_all_position_credit": False,
        # Injection OFF in every arm of this lineage: E3 never sees a chunk, so
        # the action stream is untouched by chunk formation and a readiness
        # probe cannot perturb the behaviour a later run would measure.
        "use_chunk_proposal_injection": False,
        # MECH-094 strict gate: the sleep-replay carve-out stays OFF. Its
        # inertness is an audited property of this lineage, not an assumption.
        "use_chunk_replay_origin_path": False,
    }


def shared_config_kwargs() -> Dict[str, Any]:
    """Config every arm is built with, independent of the chunking switches."""
    return {
        "alpha_world": ALPHA_WORLD,
        "chunk_min_repetitions": CHUNK_MIN_REPETITIONS,
        "chunk_window_trials": CHUNK_WINDOW_TRIALS,
        "chunk_crystallisation_min": CHUNK_CRYSTALLISATION_MIN,
    }


def off_path_config_slice() -> Dict[str, Any]:
    """The fingerprint slice for the OFF cell.

    Declares ONLY what the OFF computation reads: env, schedule, the
    substrate-operating config every arm runs, and the OFF arm's own flags.
    Never the ON-arm switches, the acceptance thresholds, or the arm labels.
    """
    slice_: Dict[str, Any] = {
        "env": dict(ENV_KWARGS),
        "schedule": dict(SCHEDULE),
    }
    slice_.update(shared_config_kwargs())
    slice_.update(off_arm_flags())
    return slice_


def arm_config_slice(arm_flags: Dict[str, Any]) -> Dict[str, Any]:
    """The fingerprint slice for any cell of this lineage.

    Identical in shape to `off_path_config_slice()`; for the OFF arm's flags it
    returns exactly that slice, which is what makes a later sibling's OFF cell
    hash-identical to this lineage's mint.
    """
    slice_: Dict[str, Any] = {
        "env": dict(ENV_KWARGS),
        "schedule": dict(SCHEDULE),
    }
    slice_.update(shared_config_kwargs())
    slice_.update(dict(arm_flags))
    return slice_


def build_env(seed: int) -> Any:
    return CausalGridWorldV2(seed=seed, **ENV_KWARGS)


def build_agent(env: Any, arm_flags: Dict[str, Any]) -> REEAgent:
    return REEAgent(REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=4,
        **shared_config_kwargs(),
        **arm_flags,
    ))


def formed_chunk_lengths(agent: REEAgent) -> Dict[str, int]:
    """Histogram of formed-chunk sequence lengths, keyed by length as a string.

    Empty when chunking is off. This is the readout that makes the 810
    structural-inertness finding directly checkable: at buf = 3.00 nothing above
    length 3 could ever appear, whatever the size budget said.
    """
    pc = getattr(agent, "policy_chunking", None)
    if pc is None:
        return {}
    hist: Dict[str, int] = {}
    for chunk in pc.library.all_chunks():
        key = str(len(chunk.sequence))
        hist[key] = hist.get(key, 0) + 1
    return hist


def run_cell(arm_flags: Dict[str, Any], seed: int, *,
             n_episodes: int, steps_per_episode: int,
             progress_every: int = 30,
             progress_label: str = "") -> Dict[str, Any]:
    """Run one (arm, seed) cell on the canonical full agent loop.

    Returns the raw per-cell readout dict. The caller owns the arm_cell()
    context (RNG reset + fingerprint stamping) and the verdict line.

    THE LOOP. `StepHarness.step` is the canonical sequence -- one `sense()`, one
    `update_z_goal(...)`, one `select_action(...)`, one `update_residue(...)` per
    env step. `update_residue` is the piece 810 omitted, and it is what lets
    MECH-091 fire `clock.phase_reset()` on a harm event so the E3 tick stops
    being perfectly periodic. `agent.select_action` is the sole `record_step`
    call site (ree_core/agent.py), so the accumulator is fed only by genuinely
    deliberated, executed actions -- hypothesis_tag is False by construction
    there (MECH-094).
    """
    env = build_env(seed)
    agent = build_agent(env, arm_flags)
    # train_mode=False (torch.no_grad) is CORRECT here, not merely faster. This
    # lineage is a readiness probe: it constructs no optimiser and calls no
    # backward pass, and neither does the agent (`ree_core/agent.py` contains no
    # `.backward()` at all -- there is no internal learning step to preserve
    # gradients for). Under train_mode=True the autograd graph would accumulate
    # across every one of the 72 steps in an episode and then be discarded,
    # which is pure cost and a memory hazard. Measured on the hub, the
    # gradient-tracking version ran at roughly 2 s/step, which put the full
    # 4-arm x 8-seed grid out of reach entirely.
    harness = StepHarness(agent, env, train_mode=False, seed=seed)

    episode_outcomes: List[float] = []
    per_episode_symbols: List[int] = []
    n_harm_steps = 0
    n_steps_total = 0
    prev_symbols = 0

    for ep in range(n_episodes):
        results = harness.run_episode(steps_per_episode)
        ep_reward = 0.0
        for r in results:
            ep_reward += r.harm_signal
            if r.harm_signal < 0:
                n_harm_steps += 1
        n_steps_total += len(results)

        # The trial-boundary outcome report. No-op when chunking is off.
        agent.note_chunk_outcome(ep_reward)
        episode_outcomes.append(ep_reward)

        st_now = agent.get_chunking_state()
        symbols_now = int(st_now.get("chunk_acc_n_steps", 0))
        per_episode_symbols.append(symbols_now - prev_symbols)
        prev_symbols = symbols_now

        if (ep + 1) % progress_every == 0:
            print(f"  [train] chunk {progress_label} ep {ep+1}/{n_episodes} "
                  f"formed={st_now.get('chunk_acc_n_formed', 0)} "
                  f"cryst={st_now.get('chunk_lib_n_crystallised', 0)} "
                  f"symbols/ep={per_episode_symbols[-1]}", flush=True)

    st = agent.get_chunking_state()
    return {
        "chunking_state": st,
        "episode_outcomes": episode_outcomes,
        "per_episode_symbols": per_episode_symbols,
        "n_harm_steps": n_harm_steps,
        "n_steps_total": n_steps_total,
        "formed_chunk_lengths": formed_chunk_lengths(agent),
        "chunking_instantiated": bool(agent.policy_chunking is not None),
        "z_goal_stream_stats": harness.z_goal_stream_stats(),
    }


def run_off_cell(seed: int, **kwargs) -> Dict[str, Any]:
    """The canonical OFF cell. Kept as a named entry point for reusers."""
    return run_cell(off_arm_flags(), seed,
                    n_episodes=kwargs.pop("n_episodes", N_EPISODES),
                    steps_per_episode=kwargs.pop("steps_per_episode",
                                                 STEPS_PER_EPISODE),
                    **kwargs)
