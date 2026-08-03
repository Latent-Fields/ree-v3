"""Canonical OFF-path baseline for the MECH-426 progress-velocity lineage
(V3-EXQ-885 and any successor 885a / 885b / ...).

WHY THIS MODULE EXISTS
----------------------
Two reasons, both structural rather than stylistic:

(1) ARM REUSE (the "mint as you go" default). V3-EXQ-885 is the FIRST
    experiment of this lineage, so per the `/queue-experiment` skill's
    producer-side rule it mints its own OFF/baseline arms reuse-ELIGIBLE
    in-line, with `include_driver_script_in_hash=False`, rather than
    spawning a redundant baseline-only job. A later, DIFFERENT driver can
    only cache-HIT that mint if it constructs its OFF arm from exactly this
    module -- re-deriving the OFF path inline would produce a different
    `config_slice` and refuse. Hence `off_path_config_slice()` is the
    single declaration of what the OFF computation reads.

(2) SUBSTRATE-HASH INCLUSION. `experiments/_lib/arm_fingerprint.py` hashes a
    cell's computation from `ree_core/**` + `experiments/_lib/**` (+ the
    driver script, unless excluded). Cell-executed code that lives in a
    DRIVER file and is then excluded by `include_driver_script_in_hash=False`
    is invisible to the hash -- the under-inclusion defect documented in
    `experiments/_lib/allon_training.py`. So the code a cell EXECUTES lives
    here; the arm table, the acceptance gates and the manifest shape stay in
    the driver.

THE CONTRACT (the part that must be exactly right)
--------------------------------------------------
The OFF arm's computation is the conjunction of:
  * ENV_REGIMES[regime]              -- the confirmation-density regime
                                        (num_resources; everything else held)
  * the schedule                     -- P0A_EPISODES/P0A_STEPS_PER_EP,
                                        N_EPISODES_P2, STEPS_PER_EP
  * the substrate-operating config   -- WORLD_DIM, z_goal_enabled=True, the
                                        GoalConfig defaults every arm runs on
  * the SD-093 knobs                 -- use_progress_velocity_effort_modulation
                                        (the ablation) plus gain / window / cap,
                                        declared on EVERY arm (see
                                        `arm_config_slice` for why the tempting
                                        ON-arms-only scoping is a false-HIT bug)
It does NOT depend on the acceptance thresholds or on the arm label strings --
neither can change a cell's computation, so folding them in would only cause
false misses.

WHY THE OFF ARM IS A TRUE NO-OP CONTROL
---------------------------------------
`use_progress_velocity_effort_modulation=False` is the SD-093 master switch
and is bit-identical OFF by construction, verified by contract
(`tests/contracts/test_mech426_progress_velocity.py` C2/C9):
`GoalState.record_progress()` returns 0.0 without touching the history deque,
`progress_velocity_effort_modulation` returns exactly 0.0, and
`E3TrajectorySelector.select()`'s modulation branch is never entered. So the
OFF arm is the stock substrate, not a differently-parameterised one.

MECH-094: not applicable. Everything here runs on live waking observations;
nothing is written to any memory store in a non-waking state.

SLEEP DRIVER: N/A -- no sleep loop is enabled anywhere in this module.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorld
from ree_core.utils.config import REEConfig

from experiments._lib.capability_eval import RandomPolicy
from experiments._lib.zworld_p0_warmup import run_zworld_p0

# --------------------------------------------------------------------- #
# Schedule + substrate-operating constants (part of the OFF-path slice)
# --------------------------------------------------------------------- #
GRID_SIZE = 10
NUM_HAZARDS = 0
# CONTAMINATED_HARM = 0.0 is load-bearing, and num_hazards=0 alone does NOT
# achieve what it looks like it achieves. CausalGridWorld marks the cell the
# agent VACATES as "contaminated" (causal_grid_world.py:2210), and re-entering
# one costs `contaminated_harm` health as an "agent_caused_hazard". With the
# 0.4 default, three re-visits of its own footprint kill the agent: measured
# under a random walk with num_hazards=0, episodes terminated after ~29 steps
# (health <= 0), not the 500-step cap. That silently converts a long-horizon
# maintenance experiment into a short-episode one -- exactly the regime in
# which there is no long gap between confirmations to maintain anything across.
# Zeroing it makes `done` be `steps >= 500` alone.
# It also removes a SECOND, uncontrolled modulator of the very quantity under
# test: SD-011 urgency scales the SAME `effective_threshold` this experiment
# manipulates (e3_selector.py, immediately above the SD-093 block). That path
# is already inert here because sense() is called without harm args so z_harm_a
# is None -- zeroing the harm makes it doubly so, by construction rather than
# by call-site convention.
CONTAMINATED_HARM = 0.0
WORLD_DIM = 32

STEPS_PER_EP = 500       # CausalGridWorld.step()'s own episode cap
N_EPISODES_P2 = 6
P0A_EPISODES = 20
P0A_STEPS_PER_EP = 60

# Dry-run (smoke) budgets -- exercise every code path at a fraction of the cost.
STEPS_PER_EP_DRY = 60
N_EPISODES_P2_DRY = 1
P0A_EPISODES_DRY = 2
P0A_STEPS_PER_EP_DRY = 20

# --------------------------------------------------------------------- #
# Factor 2: confirmation density.
#
# A z_goal REFRESH ("terminal confirmation") is a resource CONSUMPTION event:
# GoalState.update() pulls z_goal toward z_world only when
# `benefit_exposure * ... > benefit_threshold` (0.1), and the env's
# `resource_benefit` (0.3) clears that while the continuous
# `proximity_benefit_scale` (0.03) does not. So confirmation rate is set by
# how often the agent actually consumes a resource, i.e. by num_resources.
# `resource_respawn_on_consume=True` in BOTH regimes so neither runs out --
# the regimes differ in RATE, not in eventual availability.
# --------------------------------------------------------------------- #
ENV_REGIMES: Dict[str, Dict[str, Any]] = {
    "SPARSE": {"num_resources": 2},
    "DENSE": {"num_resources": 8},
}
REGIMES: Tuple[str, ...] = ("SPARSE", "DENSE")


# --------------------------------------------------------------------- #
# Fingerprint slice
# --------------------------------------------------------------------- #

def _schedule_slice(dry_run: bool) -> Dict[str, Any]:
    return {
        "steps_per_ep": STEPS_PER_EP_DRY if dry_run else STEPS_PER_EP,
        "n_episodes_p2": N_EPISODES_P2_DRY if dry_run else N_EPISODES_P2,
        "p0a_episodes": P0A_EPISODES_DRY if dry_run else P0A_EPISODES,
        "p0a_steps_per_ep": P0A_STEPS_PER_EP_DRY if dry_run else P0A_STEPS_PER_EP,
    }


def off_path_config_slice(regime: str, *, dry_run: bool = False,
                          velocity_gain: float = 10.0,
                          velocity_window: int = 5,
                          velocity_max: float = 0.3) -> Dict[str, Any]:
    """The canonical declared slice for this lineage's OFF (velocity-ablated)
    arm in `regime`. A consumer MUST call this -- not re-derive it -- for its
    `try_reuse_cell(config_slice=...)` to match the mint.

    The three SD-093 knob defaults here are V3-EXQ-885's minting values; a
    consumer that ran its ON arms at a different gain must pass that gain to
    hit these cells (see `arm_config_slice` for why they are declared on the
    OFF arm at all)."""
    return arm_config_slice(
        velocity_on=False, regime=regime, dry_run=dry_run,
        velocity_gain=velocity_gain, velocity_window=velocity_window,
        velocity_max=velocity_max,
    )


def arm_config_slice(*, velocity_on: bool, regime: str,
                     dry_run: bool = False,
                     velocity_gain: Optional[float] = None,
                     velocity_window: Optional[int] = None,
                     velocity_max: Optional[float] = None) -> Dict[str, Any]:
    """Declared slice for any arm of this lineage.

    ALL THREE SD-093 knobs are declared on EVERY arm, including the OFF arms.
    An earlier draft scoped gain/window/cap to the ON arms on the reasoning that
    the master switch short-circuits ahead of them, so the OFF slice would stay
    invariant to an ON-only re-tune and reuse would hit more often. That is the
    wrong trade and `validate_experiments.py --checks config_slice_declaration`
    flags it: under-approximating the slice is a false-cache-HIT bug, and a
    false HIT corrupts a conclusion while a false MISS only wastes compute
    (arm_reuse_fingerprint_plan.md 7b). It is also not even true of
    `progress_velocity_window`, which sizes the GoalState history deque in
    __init__ REGARDLESS of the master switch. So the safe direction is taken:
    a successor that re-tunes the gain will re-run the OFF cells rather than
    risk reading them under a different scheme."""
    slice_: Dict[str, Any] = {
        "lineage": "exq885_mech426_velocity",
        "regime": regime,
        "grid_size": GRID_SIZE,
        "num_hazards": NUM_HAZARDS,
        "contaminated_harm": CONTAMINATED_HARM,
        "num_resources": ENV_REGIMES[regime]["num_resources"],
        "resource_respawn_on_consume": True,
        "world_dim": WORLD_DIM,
        "z_goal_enabled": True,
        "use_progress_velocity_effort_modulation": bool(velocity_on),
        "progress_velocity_effort_gain": float(velocity_gain),
        "progress_velocity_window": int(velocity_window),
        "progress_velocity_effort_max": float(velocity_max),
        "dry_run": bool(dry_run),
    }
    slice_.update(_schedule_slice(dry_run))
    return slice_


# --------------------------------------------------------------------- #
# Env + agent builders
# --------------------------------------------------------------------- #

def build_env(regime: str, seed: int) -> CausalGridWorld:
    return CausalGridWorld(
        size=GRID_SIZE,
        num_hazards=NUM_HAZARDS,
        contaminated_harm=CONTAMINATED_HARM,
        num_resources=ENV_REGIMES[regime]["num_resources"],
        resource_respawn_on_consume=True,
        seed=seed,
    )


def build_warmup_env(regime: str, seed: int) -> CausalGridWorld:
    """A DEDICATED env for the SD-070 P0a warmup rollout.

    run_zworld_p0's own docstring requires this: the rollout consumes env RNG,
    so reusing the measurement env would shift the layout sequence P2 then
    sees. Built with the same regime and seed so the warmup sees the matched
    state distribution."""
    return build_env(regime, seed)


def build_agent(env: CausalGridWorld, *, velocity_on: bool,
                velocity_gain: float, velocity_window: int,
                velocity_max: float) -> REEAgent:
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        world_dim=WORLD_DIM,
        z_goal_enabled=True,
    )
    # SD-093 / MECH-426 -- THE ABLATION. False is the substrate default and is
    # bit-identical OFF (contract C2/C9), so the OFF arm is the stock substrate.
    cfg.goal.use_progress_velocity_effort_modulation = bool(velocity_on)
    cfg.goal.progress_velocity_window = int(velocity_window)
    cfg.goal.progress_velocity_effort_gain = float(velocity_gain)
    cfg.goal.progress_velocity_effort_max = float(velocity_max)
    agent = REEAgent(cfg)
    # V3-EXQ-571 per-component decomposition. Diagnostics-only and explicitly
    # bit-identical when off; enabled IDENTICALLY in every arm, so it cannot
    # confound the contrast. It is the only route by which `effective_threshold`
    # / `commit_variance` / `committed` escape select() -- they are otherwise
    # pure locals, and they are exactly what the readiness gates measure.
    agent.e3.e3_score_decomp_enabled = True
    return agent


def _benefit_and_drive(obs_body: torch.Tensor) -> Tuple[float, float]:
    """body_state[11] = benefit_exposure, body_state[3] = energy (the same
    indices `experiments/goal_stream_stages_sd054.py` reads)."""
    flat = obs_body.reshape(-1)
    benefit = float(flat[11].item()) if flat.shape[0] > 11 else 0.0
    energy = float(flat[3].item()) if flat.shape[0] > 3 else 0.5
    drive = max(0.0, min(1.0, 1.0 - energy))
    return benefit, drive


def _mean_run_length(flags: List[bool]) -> float:
    """Mean length of maximal consecutive True runs. 0.0 if never True."""
    runs: List[int] = []
    cur = 0
    for f in flags:
        if f:
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    return sum(runs) / len(runs) if runs else 0.0


# --------------------------------------------------------------------- #
# The executed cell
# --------------------------------------------------------------------- #

def run_cell(*, velocity_on: bool, regime: str, seed: int,
             velocity_gain: float, velocity_window: int, velocity_max: float,
             zg: Any, dry_run: bool, arm_label: str,
             episode_counter_base: int, episodes_per_run: int) -> Dict[str, Any]:
    """One (arm x seed) cell: P0a encoder warmup -> P2 long-horizon measurement.

    NO POLICY LEARNING. The agent's action selection is its own stock E3 path
    throughout; no optimizer is built and no REINFORCE/actor-critic step is
    taken. This is deliberate and is what makes the contrast clean: MECH-426's
    mechanism is not learned (SD-093 is rolling-window arithmetic plus a clip),
    so introducing a policy-learning phase would let an ON-vs-OFF difference
    ride on divergent LEARNING rather than on the maintenance dynamics the
    claim is about. What P0a DOES train is the z_world encoder -- necessary,
    because `goal_proximity()` is a distance in z_world space and an untrained
    encoder makes it a random projection (the `zworld_p0_warmup` defect), which
    would leave "progress estimate" an overclaim.

    MEASUREMENT WINDOW: every readout below is accumulated ONLY after the
    episode's FIRST confirmation. Before it, z_goal is zero-init and inactive,
    so there is by construction nothing to maintain and no modulation to apply
    -- including those ticks would dilute every arm's DV by an amount set by
    how long it took to stumble onto a resource, which is a regime property,
    not a treatment effect.
    """
    steps_per_ep = STEPS_PER_EP_DRY if dry_run else STEPS_PER_EP
    n_episodes = N_EPISODES_P2_DRY if dry_run else N_EPISODES_P2
    p0a_eps = P0A_EPISODES_DRY if dry_run else P0A_EPISODES
    p0a_steps = P0A_STEPS_PER_EP_DRY if dry_run else P0A_STEPS_PER_EP

    print(f"Seed {seed} Condition {arm_label}", flush=True)

    device = torch.device("cpu")
    env = build_env(regime, seed)
    agent = build_agent(
        env, velocity_on=velocity_on, velocity_gain=velocity_gain,
        velocity_window=velocity_window, velocity_max=velocity_max,
    )

    # ---- P0a: SD-070 z_world encoder warmup (RNG-neutral; dedicated env) ----
    warmup_env = build_warmup_env(regime, seed)
    p0a = run_zworld_p0(
        agent, warmup_env, seed=seed, episodes=p0a_eps,
        steps_per_episode=p0a_steps, policy=RandomPolicy(seed),
        label=arm_label, dry_run=dry_run,
    )
    ep_done = episode_counter_base
    for _i in range(p0a_eps):
        ep_done += 1
        if ep_done % 5 == 0 or ep_done == episodes_per_run:
            print(f"  [train] {arm_label} seed={seed} ep {ep_done}/{episodes_per_run} "
                  f"phase=P0a_zworld_warmup", flush=True)

    # ---- P2: long-horizon measurement ----
    agent.eval()

    veff_vals: List[float] = []          # per-tick velocity effort modulation
    velocity_vals: List[float] = []      # per-tick raw progress velocity
    committed_flags: List[bool] = []     # per genuine E3 selection
    flip_flags: List[bool] = []          # would the commit decision differ unmodulated?
    goal_active_flags: List[bool] = []
    goal_norms: List[float] = []         # per env step, post-first-confirmation
    n_latched_ticks = 0                  # E3 cadence held -> no fresh selection
    n_ticks_measured = 0
    n_steps_measured = 0
    n_confirmations = 0
    n_episodes_with_confirmation = 0
    per_episode: List[Dict[str, Any]] = []

    for ep in range(n_episodes):
        _flat, obs_dict = env.reset()
        agent.reset()
        seen_confirmation = False
        ep_confirmations = 0
        ep_steps_measured = 0

        for _step in range(steps_per_ep):
            ob = obs_dict["body_state"].to(device)
            ow = obs_dict["world_state"].to(device)
            with torch.no_grad():
                latent = agent.sense(ob, ow)
                ticks = agent.clock.advance()
                e1_prior = (
                    agent._e1_tick(latent)
                    if ticks.get("e1_tick", True)
                    else torch.zeros(1, WORLD_DIM, device=device)
                )
                candidates = agent.generate_trajectories(latent, e1_prior, ticks)
                # E3 DIAGNOSTICS LATCH (queue-experiment "Sample-size integrity"):
                # last_score_diagnostics is populated ONLY inside select() and
                # LATCHES -- on a tick where the E3 cadence held the previous
                # action it still carries the PREVIOUS selection's values. The
                # cadence is heartbeat.e3_steps_per_tick (default 10), NOT the
                # beta gate, so a per-env-step read without this clear would be
                # ~10x pseudo-replicated. Cleared to {} rather than None because
                # select() does item-assignment into it.
                agent.e3.last_score_diagnostics = {}
                action = agent.select_action(candidates, ticks)
            action_idx = int(action.argmax(dim=-1).item())

            diag = agent.e3.last_score_diagnostics
            fresh = "committed" in diag
            # Scoped to the measurement window, so that (n_ticks_measured,
            # n_latched_ticks) is a coherent numerator/denominator pair: the
            # true yield of genuine E3 selections over the ticks this cell
            # actually scored.
            if seen_confirmation and not fresh:
                n_latched_ticks += 1

            if seen_confirmation and fresh:
                eff_thr = float(diag["effective_threshold"])
                cvar = float(diag["commit_variance"])
                committed = bool(diag["committed"])
                gs = agent.goal_state
                active = bool(gs is not None and gs.is_active())
                veff = float(gs.progress_velocity_effort_modulation) if active else 0.0
                vel = float(gs.progress_velocity) if gs is not None else 0.0
                # The counterfactual UNMODULATED threshold. select() applies the
                # modulation as `eff_thr *= (1 + veff)`, so dividing recovers the
                # threshold this same tick would have used with the flag off --
                # every other term (sweep, SD-011 urgency) is common to both.
                thr_unmod = eff_thr / (1.0 + veff) if veff != 0.0 else eff_thr
                flip = (cvar < eff_thr) != (cvar < thr_unmod)

                veff_vals.append(veff)
                velocity_vals.append(vel)
                committed_flags.append(committed)
                flip_flags.append(flip)
                goal_active_flags.append(active)
                n_ticks_measured += 1

            _flat, _harm, done, info, obs_dict = env.step(action_idx)

            benefit, drive = _benefit_and_drive(obs_dict["body_state"].to(device))
            agent.update_z_goal(benefit_exposure=benefit, drive_level=drive)

            if info.get("transition_type", "none") == "resource":
                if not seen_confirmation:
                    seen_confirmation = True
                    n_episodes_with_confirmation += 1
                else:
                    # Only confirmations INSIDE the measurement window count
                    # toward the completion-rate DV, so the window-opening
                    # event itself is not scored as an outcome of maintenance.
                    ep_confirmations += 1
                    n_confirmations += 1

            if seen_confirmation:
                gs = agent.goal_state
                if gs is not None:
                    goal_norms.append(float(gs.goal_norm()))
                n_steps_measured += 1
                ep_steps_measured += 1

            if done:
                break

        per_episode.append({
            "episode": ep,
            "confirmed": seen_confirmation,
            "confirmations_in_window": ep_confirmations,
            "steps_measured": ep_steps_measured,
        })
        ep_done += 1
        if ep_done % 5 == 0 or ep_done == episodes_per_run:
            print(f"  [train] {arm_label} seed={seed} ep {ep_done}/{episodes_per_run} "
                  f"phase=P2_measurement", flush=True)

    if zg is not None:
        zg.observe(agent)

    # ---- DVs ----
    peak = max(goal_norms) if goal_norms else 0.0
    goal_retention = (
        (sum(goal_norms) / len(goal_norms)) / peak if goal_norms and peak > 1e-12 else 0.0
    )
    commit_run_len_mean = _mean_run_length(committed_flags)
    confirmations_per_1k = (
        1000.0 * n_confirmations / n_steps_measured if n_steps_measured else 0.0
    )
    velocity_effort_range = (max(veff_vals) - min(veff_vals)) if veff_vals else 0.0
    velocity_range = (max(velocity_vals) - min(velocity_vals)) if velocity_vals else 0.0
    commit_flip_frac = (
        sum(1 for f in flip_flags if f) / len(flip_flags) if flip_flags else 0.0
    )
    goal_active_frac = (
        sum(1 for f in goal_active_flags if f) / len(goal_active_flags)
        if goal_active_flags else 0.0
    )
    committed_frac = (
        sum(1 for f in committed_flags if f) / len(committed_flags)
        if committed_flags else 0.0
    )
    frac_eps_confirmed = n_episodes_with_confirmation / max(1, n_episodes)

    print(
        f"  [eval] {arm_label} seed={seed} retention={goal_retention:.4f} "
        f"commit_run={commit_run_len_mean:.2f} conf/1k={confirmations_per_1k:.2f} "
        f"veff_range={velocity_effort_range:.5f} flip_frac={commit_flip_frac:.4f} "
        f"ticks={n_ticks_measured} latched={n_latched_ticks}",
        flush=True,
    )

    return {
        "arm": arm_label,
        "regime": regime,
        "velocity_on": bool(velocity_on),
        "seed": seed,
        # --- DVs ---
        "goal_retention": goal_retention,
        "commit_run_len_mean": commit_run_len_mean,
        "confirmations_per_1k_steps": confirmations_per_1k,
        "committed_frac": committed_frac,
        # --- readiness statistics ---
        "velocity_effort_range": velocity_effort_range,
        "velocity_effort_mean_abs": (
            sum(abs(v) for v in veff_vals) / len(veff_vals) if veff_vals else 0.0
        ),
        "velocity_range": velocity_range,
        "commit_flip_frac": commit_flip_frac,
        "goal_active_frac": goal_active_frac,
        "frac_episodes_with_confirmation": frac_eps_confirmed,
        # --- denominators + provenance (generous recording) ---
        "n_ticks_measured": n_ticks_measured,
        "n_latched_ticks": n_latched_ticks,
        "n_steps_measured": n_steps_measured,
        "n_confirmations_in_window": n_confirmations,
        "n_episodes": n_episodes,
        "n_episodes_with_confirmation": n_episodes_with_confirmation,
        "steps_per_ep": steps_per_ep,
        "goal_norm_peak": peak,
        "goal_norm_mean": (sum(goal_norms) / len(goal_norms)) if goal_norms else 0.0,
        "per_episode": per_episode,
        "p0a_zworld_warmup": p0a,
    }
