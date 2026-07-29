"""SD-084 -- MECH-321 R4 mid-execution hook: NATURAL reachability.

WHAT THIS PINS, AND WHY IT IS A DIFFERENT ASSERTION FROM THE SHIPPED CONTRACT.

`tests/contracts/test_mech321_scale_resolved_boundary.py` already exercises the
mid-execution hook -- but it reaches it by hand-injecting every precondition the
gate tests:

    fake_traj.metadata = {"source": "arc071_chunk", ...}
    agent.e3._committed_trajectory = fake_traj
    agent._committed_step_idx = 1
    agent.beta_gate.elevate()

That is a correct contract, and it carries a correct anti-vacuity guard. It
proves reachability IN PRINCIPLE: given the gate state, the hook runs. What it
structurally CANNOT catch is whether that gate state is ever ATTAINABLE under a
real driver loop -- and it was not. `E3Selector.post_action_update`'s last
statement unconditionally destroys `_committed_trajectory`, and every driver
calls `update_residue` each step, so the hook has never executed in any
experiment: V3-EXQ-830 measured `decomp_n_evaluated_midexec = 0` in all 10 cells
(5 seeds x 2 arms) against `decomp_n_evaluated_precommit` of 1862-2618.
(Diagnosis: `failure_autopsy_V3-EXQ-830_2026-07-29.md` section 3.)

So this file asserts reachability IN PRACTICE. It drives the canonical
`sense -> generate_trajectories -> update_z_goal -> select_action -> env.step ->
update_residue` loop -- the same idiom as the V3-EXQ-830 driver -- and injects
NOTHING. No metadata is written, no `_committed_step_idx` is set, no
`beta_gate.elevate()` is called. Every gate must be satisfied by the substrate's
own dynamics or the test fails.

THE NEGATIVE CONTROL IS THE OTHER HALF, and it is what makes the positive
assertion mean anything. With the flag OFF the same loop must yield exactly
zero mid-execution evaluations WHILE STILL PRODUCING multi-action commits --
proving the zero is the structural teardown and not merely an absence of
committed programs to re-evaluate. That distinction is precisely what a
hand-injected contract cannot make.

Seed 71 is pinned because the natural path is seed-dependent: E3 commits a
multi-action ARC-071 chunk only when the chunk beats the CEM-optimised
candidates on score, which happens on some seeds and not others (measured
2026-07-29: seeds 47 and 71 yield multi-action commits at this config; 11, 23
and 97 yield none). The anti-vacuity assertions below fail loudly rather than
passing silently if a substrate change ever makes that stop happening.
"""

import warnings

import torch

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.policy import ChunkedPrimitive, ChunkState
from ree_core.utils.config import REEConfig

# A crystallised 3-action chunk. Length 3 matters: the hook's gate (6) requires
# len(remaining) > 1, and _committed_step_idx is 1 on the tick after a commit,
# so a 3-action program leaves exactly 2 unexecuted actions.
SEQ = (0, 1, 2)
SEED = 71
EPISODES = 2
STEPS = 40
# decomposition_depth_cap=3 is the config DEFAULT and is non-degenerate. Do not
# lower it to 1 to make chunks survive decomposition: the substrate itself warns
# that depth_cap=1 is degenerate (every ARC-071 chunk has depth >= 1, so every
# triggering chunk is marked unreliable rather than re-tiled and decomposition
# never fires at all), which would make this contract pass for the wrong reason.
DEPTH_CAP = 3
# 0.99 == near-certain V_s trigger, the same value the shipped MECH-321
# contracts use (vs_trigger is `region_vs < threshold`).
VS_THRESHOLD = 0.99


def _drive_real_rollout(handle_on):
    """Drive the canonical loop. Returns (midexec_evals, multi_action_commits).

    NOTHING about the mid-execution gate is injected. The only setup is
    registering a chunk in the library -- which is ordinary ARC-071 usage, not a
    precondition of the hook.
    """
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
        use_event_segmenter=True,
        use_policy_chunking=True,
        use_chunk_proposal_injection=True,
        use_policy_decomposition=True,
        decomposition_vs_threshold=VS_THRESHOLD,
        decomposition_depth_cap=DEPTH_CAP,
        use_persistent_committed_program_handle=handle_on,
    )
    agent = REEAgent(cfg)

    def _register():
        agent.policy_chunking.library.register(
            ChunkedPrimitive(
                sequence=SEQ,
                depth=1,
                state=ChunkState.CRYSTALLISED,
                selection_weight=1.0,
            )
        )

    _register()
    world_dim = cfg.latent.world_dim
    multi_action_commits = 0

    for _ in range(EPISODES):
        _, obs = env.reset()
        agent.reset()
        if not agent.policy_chunking.library.all_chunks():
            _register()
        for _ in range(STEPS):
            latent = agent.sense(obs["body_state"], obs["world_state"])
            ticks = agent.clock.advance()
            e1_prior = (
                agent._e1_tick(latent)
                if ticks.get("e1_tick")
                else torch.zeros(1, world_dim, device=agent.device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            body = obs["body_state"]
            agent.update_z_goal(
                benefit_exposure=0.0,
                drive_level=REEAgent.compute_drive_level(body),
            )
            action = agent.select_action(candidates, ticks)
            # Read the FRESH per-tick commit, before update_residue tears it
            # down. Read-only -- this observes the loop, it does not steer it.
            committed = agent.e3._committed_trajectory
            if committed is not None:
                meta = committed.metadata or {}
                if len(meta.get("chunk_sequence", ())) > 1:
                    multi_action_commits += 1
            _flat, harm, _done, _info, obs = env.step(
                int(action.argmax(dim=-1).item())
            )
            agent.update_residue(harm)

    state = agent.get_policy_decomposition_state()
    return int(state["decomp_n_evaluated_midexec"]), multi_action_commits


def test_sd084_midexec_hook_fires_in_a_real_rollout_with_nothing_injected():
    """ON: the hook executes under the standard driver loop.

    This is the assertion that did not exist before SD-084, and the one whose
    absence let a structurally-unreachable hook ship correct, flagged and
    contract-covered while never once running.
    """
    midexec, multi_commits = _drive_real_rollout(handle_on=True)
    assert multi_commits > 0, (
        "anti-vacuity: no multi-action program was committed, so the hook had "
        "nothing to re-evaluate and this test would pass for the wrong reason"
    )
    assert midexec > 0, (
        "the MECH-321 R4 mid-execution hook did not fire in a real rollout "
        "even with use_persistent_committed_program_handle=True -- SD-084 has "
        "regressed and the R4 half is unmeasurable again"
    )


def test_sd084_off_path_is_structurally_zero_despite_multi_action_commits():
    """OFF: the negative control, and the reason the ON assertion has content.

    Multi-action programs ARE committed on this seed, so the zero below is not
    'nothing to evaluate'. It is post_action_update's unconditional teardown
    destroying the committed trajectory before the next tick's gate (4) can see
    it -- exactly the V3-EXQ-830 structural zero, reproduced here as a unit.
    """
    midexec, multi_commits = _drive_real_rollout(handle_on=False)
    assert multi_commits > 0, (
        "anti-vacuity: the OFF arm must still COMMIT multi-action programs, "
        "otherwise midexec == 0 is trivially true and controls nothing"
    )
    assert midexec == 0, (
        "OFF path is no longer bit-identical: the mid-execution hook fired "
        f"{midexec} times without the SD-084 handle, which contradicts the "
        "unconditional _committed_trajectory teardown in post_action_update"
    )


def test_sd084_flag_defaults_off_and_is_wired_through_from_dims():
    """The three-site from_dims wiring pin.

    A REEConfig knob needs the field, the from_dims signature AND the from_dims
    assignment; wiring two of the three leaves it silently unreachable, which
    reads out as a clean null rather than as a bug (memory
    reference-reeconfig-from-dims-silent-kwargs). Asserted on from_dims rather
    than on REEConfig() so a missing signature/assignment cannot pass.
    """
    dims = dict(
        body_obs_dim=4,
        world_obs_dim=8,
        action_dim=5,
        self_dim=32,
        world_dim=32,
        reafference_action_dim=5,
    )
    assert REEConfig.from_dims(**dims).use_persistent_committed_program_handle is False
    assert (
        REEConfig.from_dims(
            **dims, use_persistent_committed_program_handle=True
        ).use_persistent_committed_program_handle
        is True
    )


def test_sd084_handle_survives_post_action_update_but_dies_on_beta_release():
    """The handle's two defining properties, asserted directly.

    (a) It is NOT torn down by post_action_update -- that is the whole point,
        and the property _committed_trajectory lacks.
    (b) It IS reaped once beta is no longer elevated, so the hook can never fire
        against a program that has already been released. (b) is an INVARIANT
        rather than a list of clear sites on purpose: agent.py has ten
        beta_gate.release() sites and only five clear _committed_trajectory,
        because post_action_update backstops the rest -- a backstop this handle
        removes.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        env = CausalGridWorldV2()
        _, obs = env.reset()
        cfg = REEConfig.from_dims(
            body_obs_dim=env.body_obs_dim,
            world_obs_dim=env.world_obs_dim,
            action_dim=env.action_dim,
            self_dim=32,
            world_dim=32,
            reafference_action_dim=env.action_dim,
            use_persistent_committed_program_handle=True,
        )
        agent = REEAgent(cfg)

    class _StubTrajectory:
        """Minimal stand-in: post_action_update reads only .world_states on the
        reference trajectory, and skips the residue branch when harm is False."""

        world_states = None
        metadata = None

    traj = _StubTrajectory()
    agent.e3._committed_trajectory = traj
    agent.e3._persistent_committed_trajectory = traj

    # (a) post_action_update destroys the F-driven handle, not the persistent one.
    agent.e3.post_action_update(
        torch.zeros(1, cfg.latent.world_dim), harm_occurred=False
    )
    assert agent.e3._committed_trajectory is None
    assert agent.e3._persistent_committed_trajectory is traj, (
        "the persistent handle must survive post_action_update -- if it does "
        "not, the mid-execution hook is unreachable again"
    )

    # (b) episode reset clears it: no program carries across an episode boundary.
    agent.reset()
    assert agent.e3._persistent_committed_trajectory is None
