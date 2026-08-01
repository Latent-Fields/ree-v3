"""ARC-071/MECH-090 E3-tick reselection short-circuit.

WHAT THIS PINS. `diagnostic_arc071_e3_reselection_probe_2026-08-01.md`
confirmed, via a real 53,063-step hazard-exposed rollout, that
`select_action`'s E3-tick branch unconditionally re-deliberates and installs
a brand-new trajectory object on EVERY E3 tick, even when already committed
to an unexpired `arc071_chunk` program -- `premature_reselection` rates of
36.8% / 41.4% / 55.1% at `chunk_max_size` 5 / 8 / 15 (0% at 2 / 3, the two
sizes below the ~8-10-step steady-state E3 cadence observed in that probe).
`substrate_queue.json` entry `arc071_e3_reselection_on_committed_program`'s
`diagnosis.recommended_next_step` named the fix: an "already committed to an
unexpired arc071_chunk program, skip re-selection" short-circuit in
`select_action`'s E3-tick branch, gated on `use_e3_reselection_shortcircuit`
(default False -> bit-identical; requires `use_persistent_committed_program_
handle=True` to have any candidate trajectory to check against).

METHOD. Rather than reproduce the probe's expensive 120-episode real-hazard
formation schedule, this test drives the exact mechanism the diagnosis found
dominant: MECH-091's `clock.phase_reset()` forces an E3 tick on the very next
`advance()` regardless of the periodic counter. Forcing `ticks["e3_tick"] =
True` on every subsequent step after a chunk commits reproduces the diagnosed
worst-case cadence (gap=1) deterministically, well below any `chunk_max_size`
tested here, without depending on hazard signals, torch RNG, or chunk-
formation timing. A chunk is registered directly in the policy_chunking
library (same idiom as `test_arc071_commit_latch_persistence.py`) rather than
grown through curriculum, since the mechanism under test is E3's per-tick
re-selection decision, not chunk formation itself.

CLASSIFICATION mirrors the probe's own priority order: a re-commit only
counts as `premature_reselection` when the PRIOR persistent trajectory was
itself `arc071_chunk`-sourced and had not yet reached `chunk_length - 1` --
matching the probe's own `non_chunk_reselect` / `premature_reselection`
distinction (a re-commit away from an *ordinary* CEM candidate, or after the
chunk has already run to its own horizon, is not a defect and must not be
counted as one).

NEGATIVE CONTROL: with the flag OFF, the identical forced-tick schedule must
reproduce the diagnosed defect exactly -- a distinct trajectory object
installed on effectively every forced E3 tick, `_committed_step_idx` never
advancing past its immediate post-commit value.

SAFETY REGRESSION: a separate test confirms the short-circuit does not
swallow the MECH-091 urgency-interrupt release -- forcing `z_harm_a` norm
above `urgency_interrupt_threshold` while short-circuit conditions otherwise
hold must still release `beta_gate`, exactly as with the flag off.
"""

import torch

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.policy import ChunkedPrimitive, ChunkState
from ree_core.utils.config import REEConfig

SEED = 71
SEQ5 = (0, 1, 2, 1, 0)
SEQ15 = tuple([0, 1, 2, 1][i % 4] for i in range(15))
MAX_COMMIT_STEPS = 60
N_FORCED_TICKS = 25


def _make_agent(seq, shortcircuit_on, seed=SEED):
    torch.manual_seed(seed)
    env = CausalGridWorldV2()
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=32,
        reafference_action_dim=env.action_dim,
        use_event_segmenter=True,
        use_policy_chunking=True,
        use_chunk_proposal_injection=True,
        use_persistent_committed_program_handle=True,
        use_e3_reselection_shortcircuit=shortcircuit_on,
    )
    agent = REEAgent(cfg)
    agent.policy_chunking.library.register(
        ChunkedPrimitive(
            sequence=seq,
            depth=1,
            state=ChunkState.CRYSTALLISED,
            selection_weight=1.0,
        )
    )
    return env, agent


def _step_once(env, agent, obs, force_e3_tick=False):
    world_dim = agent.config.latent.world_dim
    latent = agent.sense(obs["body_state"], obs["world_state"])
    ticks = agent.clock.advance()
    if force_e3_tick:
        ticks["e3_tick"] = True
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
    pre_idx = agent._committed_step_idx
    pre_traj = agent.e3._persistent_committed_trajectory
    pre_meta = pre_traj.metadata if pre_traj is not None else None
    pre_was_chunk = bool(pre_meta) and pre_meta.get("source") == "arc071_chunk"
    action = agent.select_action(candidates, ticks)
    _flat, harm, _done, _info, obs = env.step(int(action.argmax(dim=-1).item()))
    agent.update_residue(harm)
    return obs, pre_idx, pre_was_chunk, id(pre_traj) if pre_traj is not None else None


def _drive_until_chunk_committed(env, agent, obs, seq_len):
    """Step the canonical (unforced) loop until the persistent handle holds an
    arc071_chunk commitment of the expected length. Returns obs to resume from.
    """
    for _ in range(MAX_COMMIT_STEPS):
        obs, _pre_idx, _pre_was_chunk, _pre_id = _step_once(env, agent, obs)
        traj = agent.e3._persistent_committed_trajectory
        meta = traj.metadata if traj is not None else None
        if (
            meta is not None
            and meta.get("source") == "arc071_chunk"
            and len(meta.get("chunk_sequence", ())) == seq_len
        ):
            return obs
    raise AssertionError(
        "fixture broken: an arc071_chunk commitment of the registered length "
        "never formed within MAX_COMMIT_STEPS"
    )


def _force_ticks_and_classify(env, agent, obs, seq_len, n_ticks):
    """Force an E3 tick on every subsequent step (gap=1, the diagnosed
    worst-case cadence) and classify each resulting re-commit exactly as the
    probe did: premature only if the PRIOR trajectory was arc071_chunk-sourced
    and had not yet reached seq_len - 1.

    Returns (max_committed_step_idx, any_premature_reselection: bool).
    """
    max_idx = agent._committed_step_idx
    any_premature = False
    for _ in range(n_ticks):
        id_before = (
            id(agent.e3._persistent_committed_trajectory)
            if agent.e3._persistent_committed_trajectory is not None
            else None
        )
        obs, pre_idx, pre_was_chunk, _pre_id = _step_once(
            env, agent, obs, force_e3_tick=True
        )
        traj_after = agent.e3._persistent_committed_trajectory
        id_after = id(traj_after) if traj_after is not None else None
        reselected = id_after != id_before
        if reselected and pre_was_chunk and pre_idx < seq_len - 1:
            any_premature = True
        max_idx = max(max_idx, agent._committed_step_idx)
    return max_idx, any_premature


def _run_cell(seq, shortcircuit_on):
    env, agent = _make_agent(seq, shortcircuit_on)
    _, obs = env.reset()
    obs = _drive_until_chunk_committed(env, agent, obs, len(seq))
    return _force_ticks_and_classify(env, agent, obs, len(seq), N_FORCED_TICKS)


def test_shortcircuit_on_reaches_horizon_chunk5():
    max_idx, premature = _run_cell(SEQ5, shortcircuit_on=True)
    assert not premature, (
        "ARC-071 E3-tick reselection short-circuit failed to hold at "
        "chunk_max_size=5: a prior unexpired arc071_chunk commitment was cut "
        "short by a fresh reselection before reaching its own horizon"
    )
    assert max_idx >= len(SEQ5) - 1, (
        f"_committed_step_idx never reached chunk_length-1={len(SEQ5) - 1} "
        f"(max observed: {max_idx})"
    )


def test_shortcircuit_off_reproduces_premature_reselection_chunk5():
    max_idx, premature = _run_cell(SEQ5, shortcircuit_on=False)
    assert premature, (
        "negative control failed to reproduce the diagnosed defect at "
        "chunk_max_size=5 -- if the flag is truly a no-op OFF, forcing an E3 "
        "tick on every step must still cut the chunk short exactly as the "
        "confirmed pre-fix behaviour did"
    )
    assert max_idx < len(SEQ5) - 1, (
        f"OFF path is no longer bit-identical: _committed_step_idx reached "
        f"{max_idx} (>= chunk_length-1={len(SEQ5) - 1}) without the "
        "short-circuit engaging, which should be structurally impossible"
    )


def test_shortcircuit_on_reaches_horizon_chunk15():
    max_idx, premature = _run_cell(SEQ15, shortcircuit_on=True)
    assert not premature, (
        "ARC-071 E3-tick reselection short-circuit failed to hold at "
        "chunk_max_size=15 (the diagnosis's own worst-case size, 55.1% "
        "premature-reselection pre-fix)"
    )
    assert max_idx >= len(SEQ15) - 1, (
        f"_committed_step_idx never reached chunk_length-1={len(SEQ15) - 1} "
        f"(max observed: {max_idx})"
    )


def test_shortcircuit_off_reproduces_premature_reselection_chunk15():
    max_idx, premature = _run_cell(SEQ15, shortcircuit_on=False)
    assert premature, (
        "negative control failed to reproduce the diagnosed defect at "
        "chunk_max_size=15"
    )
    assert max_idx < len(SEQ15) - 1, (
        f"OFF path is no longer bit-identical: _committed_step_idx reached "
        f"{max_idx} without the short-circuit engaging"
    )


def test_shortcircuit_does_not_block_mech091_urgency_release():
    """MECH-091 must NEVER be swallowed by the short-circuit. Commit a chunk,
    force an E3 tick with an artificially high z_harm_a (norm far above
    urgency_interrupt_threshold), and confirm the urgency-interrupt release
    fires within that same select_action call.

    NOTE on the assertion target: `beta_gate.is_elevated` is NOT the right
    signal here. MECH-091 is "abort the committed program and fall through to
    FRESH E3 selection" (per its own docstring), not "stay uncommitted" -- a
    legitimate fresh commitment (even a re-commit to the same chunk) can form
    later in the SAME select_action call, re-elevating beta_gate before the
    call returns. That is correct behaviour, not a swallowed release.
    `self._ncl_mech091_fired` is the flag the MECH-091 block itself sets
    (reset to False at the top of every select_action call, read by the
    natural-commit latch-hold re-assertion a few dozen lines later in the
    SAME call, per its own "the hold must NEVER override MECH-091" comment)
    -- it is the authoritative record of whether MECH-091 fired during this
    specific call, independent of what a later fresh deliberation does.
    """
    env, agent = _make_agent(SEQ15, shortcircuit_on=True)
    _, obs = env.reset()
    obs = _drive_until_chunk_committed(env, agent, obs, len(SEQ15))
    assert agent.beta_gate.is_elevated, (
        "fixture broken: expected an active commitment (beta_gate elevated) "
        "before testing the MECH-091 interrupt against it"
    )

    world_dim = agent.config.latent.world_dim
    latent = agent.sense(obs["body_state"], obs["world_state"])
    # z_harm_a is only populated when the SD-011 dual-nociceptive-stream
    # config is on; this fixture doesn't enable it, so inject it directly at
    # the dimension the config declares (config.latent.z_harm_a_dim) rather
    # than depending on latent.z_harm_a already being non-None.
    z_dim = int(agent.config.latent.z_harm_a_dim)
    # Norm = 5 * sqrt(z_dim), far above the default urgency_interrupt_threshold
    # of 0.8 for any realistic z_harm_a_dim.
    agent._current_latent.z_harm_a = torch.full((1, z_dim), 5.0)

    ticks = agent.clock.advance()
    ticks["e3_tick"] = True
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
    pre_idx = agent._committed_step_idx
    assert pre_idx < len(SEQ15) - 1, (
        "fixture broken: the short-circuit's own precondition "
        "(_committed_step_idx < chunk_length - 1) must hold going into this "
        "tick, otherwise a fallthrough-to-full-deliberation here would be "
        "natural chunk completion, not evidence the interrupt overrode "
        "anything"
    )
    agent.select_action(candidates, ticks)

    assert agent._ncl_mech091_fired, (
        "MECH-091 urgency-interrupt release did not fire under forced high "
        "z_harm_a with use_e3_reselection_shortcircuit=True, even though the "
        "short-circuit's own precondition (unexpired arc071_chunk "
        "commitment) held going into the tick -- the short-circuit is "
        "swallowing the safety-critical release"
    )
