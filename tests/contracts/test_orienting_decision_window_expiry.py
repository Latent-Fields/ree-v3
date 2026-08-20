"""SD-ORIENTING-DECISION-SCALE amend -- post-override decision-window expiry.

WHAT THIS PINS. `substrate_queue.json`'s SD-ORIENTING-DECISION-SCALE entry
carries a SECOND failure_record (severity=degrading) beside the already-fixed
norm-vs-benefit scale mismatch: `decision_counts` in the 910 lineage is
decoupled from the intended N-tick persistence window (V3-EXQ-910 sum 206 vs a
theoretical max of 21 x 5 = 105, and byte-identical to 910a's total despite
half the override count). `failure_autopsy_V3-EXQ-910a_2026-08-11` could not
pin the exact mode; `evidence/planning/
mech489_decision_counts_defect_staged_20260817.md` section 2(d) pinned it
statically against ree-v3 HEAD f5837e9:

    Outside reset(), the ONLY clear of `_orienting_decision` sat INSIDE the
    score-bias application block, gated on five conditions including
    `_orienting_decision_ticks_remaining > 0` and `_orienting_decision in
    ("approach", "withdraw")`. A "resume" decision sets ticks_remaining = 0 and
    is excluded by BOTH, so it could never reach the clear and persisted for
    the rest of the episode. Any tick on which `candidates` was empty or lacked
    world_states likewise froze the countdown without clearing, making
    `orienting_post_override_bias_ticks` a LOWER BOUND on persistence, not a
    bound.

The amend moves the countdown into its own unconditional step, gated only on
`_orienting_decision is not None`, leaving the score-bias APPLICATION
conditional exactly as it was.

WHY THIS IS FORWARD-CRITICAL AND NOT COSMETIC. Both runs to date recorded
resume = 0, so the leak has never yet fired. It becomes active precisely WHEN
the Component 4/5 scale fix starts working -- corrupting the owed retest's
counter only in the branch that indicates success, biasing it toward the null
in a way no reader could detect from the manifest.

THE NEGATIVE CONTROLS ARE THE OTHER HALF, and they are what stop a later
session "simplifying" the expiry back inside the bias block or widening it into
a behaviour change:

  * `test_window_length_is_exactly_configured_ticks` -- the window is EXACTLY
    `orienting_post_override_bias_ticks` E3 ticks, not 1 and not unbounded. A
    fix that cleared eagerly (e.g. placed AFTER the bias block, where a resume
    would be cleared on the tick it is set) would shorten it and destroy the
    mechanism; this pins the count in both directions.
  * `test_gate_off_leaves_decision_state_untouched` -- with
    `use_defensive_orienting=False` the whole SD-099 guard is skipped, so the
    new clear must be inert and the flag bit-identical OFF.
  * `test_reset_still_clears` -- reset() remains the episode-boundary clear.

DENOMINATION. The countdown is in E3 TICKS: this block is downstream of
`select_action()`'s non-E3-tick early return, so at the default
`heartbeat.e3_steps_per_tick = 10` a nominal 5-tick window spans ~50 env
steps. `_e3_ticks_elapsed` counts `DefensiveOrientingGate.tick()` calls, which
is exactly one per executed E3 tick, so these tests are written in the same
units the substrate is.

WHAT IS DELIBERATELY SCRIPTED, AND WHY THAT IS NOT VACUOUS. `defensive_
orienting.tick()` is stubbed to a quiescent output (no trigger, no override) so
no spontaneous override can re-arm the window mid-measurement. Everything under
test -- the expiry step, its placement inside the SD-099 guard, and the
score-bias block's own gate -- is the REAL `agent.select_action()` code path,
driven by a REAL `REEAgent` over a REAL `CausalGridWorldV2` rollout. Only the
gate's verdict is scripted, not the behaviour being pinned.

VERIFIED DIFFERENTIALLY against the pre-amend `agent.py` (2026-08-20): 3 of
these 5 FAIL there and all 5 pass after. The pre-amend traces are the first
LIVE measurement of both leaks -- `failure_autopsy_V3-EXQ-910a_2026-08-11`
could only infer them from aggregates, and the staged doc pinned them
statically:

    resume:   [(2, 'resume', 0), (3, 'resume', 0), (4, 'resume', 0)]
              -- never cleared, exactly as predicted.
    withdraw: [(2, 'withdraw', 5), ... (8, 'withdraw', 5)]
              -- countdown pinned at 5 for 7 straight block executions with the
                 bias gate closed, i.e. frozen, not merely slow.

The 2 that pass in BOTH versions are the invariance controls
(`test_gate_off_leaves_decision_state_untouched`, `test_reset_still_clears`):
they must be insensitive to this amend, and a change that flipped either of
them would be changing something this amend has no business touching.
"""

import torch

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.pag.defensive_orienting import DefensiveOrientingOutput
from ree_core.utils.config import REEConfig

SEED = 489
STEPS = 220  # ~22 E3 ticks at the default e3_steps_per_tick = 10
BIAS_TICKS = 5


def _build(orienting_on):
    torch.manual_seed(SEED)
    env = CausalGridWorldV2()
    _, obs = env.reset()
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=32,
        reafference_action_dim=env.action_dim,
        use_defensive_orienting=orienting_on,
        orienting_post_override_bias_ticks=BIAS_TICKS,
    )
    agent = REEAgent(cfg)
    agent.reset()
    return env, obs, agent


def _quiesce(agent):
    """Stub the gate to a no-trigger, no-override verdict and count its calls.

    One call == one executed E3 tick that reached the SD-099 block, which is
    the unit the persistence window is denominated in.
    """
    calls = {"n": 0}

    def _tick(residue_surprise_now, z_harm_s_norm_now, simulation_mode=False):
        calls["n"] += 1
        return DefensiveOrientingOutput()

    agent.defensive_orienting.tick = _tick
    return calls


def _drive(env, obs, agent, calls, max_e3_ticks, max_steps=STEPS):
    """Drive the canonical loop, sampling the decision state after every
    select_action() on an E3 tick. Returns [(block_executions, decision, ticks)].

    `block_executions` is `calls["n"]` -- the number of times the SD-099 block
    has run, which is the unit the countdown is denominated in. NOTE it is NOT
    equal to the number of E3 ticks: the FIRST env step of an episode has
    `_last_action is None`, so select_action()'s non-E3-tick early return does
    not fire and the block runs on that step too, whether or not it is an E3
    tick. That is pre-existing SD-099 behaviour, identical before and after this
    amend, and it is exactly why the assertions below key off `calls["n"]`
    rather than off a positional index into the trace.
    """
    world_dim = agent.config.latent.world_dim
    trace = []
    for _ in range(max_steps):
        if calls is not None and calls["n"] >= max_e3_ticks:
            break
        latent = agent.sense(obs["body_state"], obs["world_state"])
        ticks = agent.clock.advance()
        e1_prior = (
            agent._e1_tick(latent)
            if ticks.get("e1_tick")
            else torch.zeros(1, world_dim, device=agent.device)
        )
        candidates = agent.generate_trajectories(latent, e1_prior, ticks)
        agent.update_z_goal(
            benefit_exposure=0.0,
            drive_level=REEAgent.compute_drive_level(obs["body_state"]),
        )
        action = agent.select_action(candidates, ticks)
        if ticks.get("e3_tick"):
            trace.append(
                (
                    None if calls is None else calls["n"],
                    agent._orienting_decision,
                    agent._orienting_decision_ticks_remaining,
                )
            )
        _flat, harm, _done, _info, obs = env.step(int(action.argmax(dim=-1).item()))
        agent.update_residue(harm)
    return trace


def test_resume_decision_does_not_leak_past_one_e3_tick():
    """THE PRIMARY LEAK. A "resume" decision carries ticks_remaining = 0 and is
    excluded by the bias block's `in ("approach", "withdraw")` condition, so
    pre-amend it could never reach any clear. It must now expire on the very
    next E3 tick that runs the block.
    """
    env, obs, agent = _build(orienting_on=True)
    calls = _quiesce(agent)
    agent._orienting_decision = "resume"
    agent._orienting_decision_ticks_remaining = 0

    trace = _drive(env, obs, agent, calls, max_e3_ticks=4)

    assert trace, "no E3 tick executed -- test drove nothing"
    assert trace[0][1] is None, (
        "resume decision survived its first E3 tick: "
        f"{trace[0]!r} (full trace {trace!r})"
    )
    assert all(d is None for _n, d, _t in trace), f"resume re-appeared: {trace!r}"


def test_countdown_advances_when_the_score_bias_gate_never_opens():
    """THE SECOND LEAK. With `_orienting_trigger_z_world` None the bias block's
    third condition never holds, so pre-amend the countdown froze and an
    approach/withdraw decision persisted indefinitely. The window must expire
    on schedule regardless of whether any bias was applied.
    """
    env, obs, agent = _build(orienting_on=True)
    calls = _quiesce(agent)
    agent._orienting_decision = "withdraw"
    agent._orienting_decision_ticks_remaining = BIAS_TICKS
    agent._orienting_trigger_z_world = None  # bias gate can never open

    trace = _drive(env, obs, agent, calls, max_e3_ticks=BIAS_TICKS + 3)

    assert len(trace) >= BIAS_TICKS + 1, f"too few E3 ticks driven: {trace!r}"
    assert trace[-1][1] is None, f"withdraw decision never expired: {trace!r}"
    assert agent._orienting_decision_ticks_remaining == 0, (
        "countdown left in a non-zero state after expiry: "
        f"{agent._orienting_decision_ticks_remaining}"
    )


def test_window_length_is_exactly_configured_ticks():
    """NEGATIVE CONTROL (both directions). Pins the EXACT countdown mapping: on
    the sample where the SD-099 block has run `n` times since the decision was
    primed with `ticks_remaining = N`, `ticks_remaining` must read `max(N-n, 0)`
    and the decision must be live iff `n < N`.

    This pins against an over-eager clear just as hard as against the unbounded
    persistence being fixed. An eager clear -- e.g. moving the expiry to AFTER
    the score-bias block, where a "resume" (ticks_remaining = 0) would be
    cleared on the very tick it is set -- would shorten the window, make the
    bias mechanism inert, and make "resume" invisible to any per-step readout:
    the same degenerate `resume = 0` reading the retest exists to escape.

    It also pins that the happy-path window is UNCHANGED by this amend: N block
    executions of bias, exactly as the old decrement-after-applying placement
    gave when the bias gate was open every tick.
    """
    env, obs, agent = _build(orienting_on=True)
    calls = _quiesce(agent)
    agent._orienting_decision = "withdraw"
    agent._orienting_decision_ticks_remaining = BIAS_TICKS
    agent._orienting_trigger_z_world = None

    trace = _drive(env, obs, agent, calls, max_e3_ticks=BIAS_TICKS + 2)

    assert trace, "no sample taken -- test drove nothing"
    assert any(n < BIAS_TICKS for n, _d, _t in trace), (
        "every sample was already past the window -- the trace cannot "
        f"distinguish an early clear from a correct one: {trace!r}"
    )
    assert any(n >= BIAS_TICKS for n, _d, _t in trace), (
        f"trace never reached the expiry point: {trace!r}"
    )
    for n, decision, remaining in trace:
        assert remaining == max(BIAS_TICKS - n, 0), (
            f"countdown off-schedule at block execution {n}: remaining="
            f"{remaining}, expected {max(BIAS_TICKS - n, 0)} (trace {trace!r})"
        )
        if n < BIAS_TICKS:
            assert decision == "withdraw", (
                f"decision expired EARLY at block execution {n} of "
                f"{BIAS_TICKS}: {trace!r}"
            )
        else:
            assert decision is None, (
                f"decision outlived its {BIAS_TICKS}-execution window at "
                f"block execution {n}: {trace!r}"
            )


def test_gate_off_leaves_decision_state_untouched():
    """NEGATIVE CONTROL. The expiry lives inside the `if self.defensive_orienting
    is not None:` guard, so with the SD-099 master switch OFF it must never run
    -- the flag stays bit-identical inert (tests/test_flag_inertness.py's
    `use_defensive_orienting` entry is the fleet-wide statement of this; here it
    is pinned for the new code specifically).
    """
    env, obs, agent = _build(orienting_on=False)
    assert agent.defensive_orienting is None, "master switch OFF but gate built"
    agent._orienting_decision = "resume"
    agent._orienting_decision_ticks_remaining = 0

    _drive(env, obs, agent, None, max_e3_ticks=0, max_steps=30)

    assert agent._orienting_decision == "resume", (
        "the new expiry ran with the SD-099 master switch OFF -- the flag is no "
        "longer inert"
    )


def test_reset_still_clears():
    """NEGATIVE CONTROL. reset() remains the episode-boundary clear for BOTH
    fields; the amend adds a within-episode expiry, it does not replace this.
    """
    _env, _obs, agent = _build(orienting_on=True)
    agent._orienting_decision = "approach"
    agent._orienting_decision_ticks_remaining = BIAS_TICKS
    agent.reset()
    assert agent._orienting_decision is None
    assert agent._orienting_decision_ticks_remaining == 0
