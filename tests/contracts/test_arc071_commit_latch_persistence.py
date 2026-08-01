"""ARC-071 commit-latch persistence -- between-E3-tick stepping branch.

WHAT THIS PINS. `failure_autopsy_V3-EXQ-841_2026-07-31` Finding B measured
`realised_mean_committed_length` pinned at exactly 1 step in 100% of 32
chunking cells with any commitment, independent of configured
`chunk_max_size` in {2, 3, 5}. `diagnostic_arc071_commit_latch_h1h2_probe_
2026-07-31.md` confirmed H1 (real ree_core wiring defect) and refuted H2
(experiment readout artifact) by direct instrumentation: `agent.py`'s
between-E3-tick "step through committed trajectory" branch (`select_action`,
the block starting `if not ticks["e3_tick"] and self._last_action is not
None:`) read only `self.e3._committed_trajectory` (torn down unconditionally
every tick by `E3Selector.post_action_update`, `e3_selector.py:3910`, by
SD-084 design) or `self.e3._closure_committed_trajectory` (off by default) --
never `self.e3._persistent_committed_trajectory`, the SD-084 handle
specifically built to survive that teardown. The MECH-321 hook a few dozen
lines away (`agent.py:5584-5588`,
`tests/contracts/test_mech321_midexec_natural_reachability.py`) already did
this fallback correctly; the fix mirrors that exact pattern into the stepping
branch.

THE NEGATIVE CONTROL IS THE OTHER HALF, same idiom as the MECH-321 contract
above it: with the flag OFF, the same real rollout must reproduce the
pre-fix pinned-at-1 signature EXACTLY (never observing
`_committed_step_idx > 1` post-tick) -- proving the ON-path advance is the
fallback actually engaging, not an unrelated change to the stepping branch's
arithmetic.

Seed 71 / chunk (0, 1, 2) is reused verbatim from
`test_mech321_midexec_natural_reachability.py`, which measured (2026-07-29)
that this seed naturally yields multi-action ARC-071 commits under the
canonical loop with nothing hand-injected -- i.e. the between-tick stepping
branch is naturally reached, not forced.
"""

import torch

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.policy import ChunkedPrimitive, ChunkState
from ree_core.utils.config import REEConfig

SEQ = (0, 1, 2)
SEED = 71
EPISODES = 2
STEPS = 40


def _drive_real_rollout(handle_on):
    """Drive the canonical loop, tracking `_committed_step_idx` post-tick on
    every non-E3-tick step. Nothing about the stepping branch is injected --
    the only setup is registering a chunk in the library, ordinary ARC-071
    usage, not a precondition of the branch under test.

    Returns (max_post_tick_step_idx_on_non_e3_ticks, multi_action_commits).
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
    max_step_idx_non_e3_tick = 0

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
            committed = agent.e3._committed_trajectory
            if committed is not None:
                meta = committed.metadata or {}
                if len(meta.get("chunk_sequence", ())) > 1:
                    multi_action_commits += 1
            if not ticks.get("e3_tick"):
                max_step_idx_non_e3_tick = max(
                    max_step_idx_non_e3_tick, agent._committed_step_idx
                )
            _flat, harm, _done, _info, obs = env.step(
                int(action.argmax(dim=-1).item())
            )
            agent.update_residue(harm)

    return max_step_idx_non_e3_tick, multi_action_commits


def test_arc071_step_idx_advances_past_one_with_persistent_handle_on():
    """ON: the between-tick stepping branch now steps through the SD-084
    persistent handle, so the committed step index advances past its first
    post-commit value across a chunk's horizon.
    """
    max_idx, multi_commits = _drive_real_rollout(handle_on=True)
    assert multi_commits > 0, (
        "anti-vacuity: no multi-action program was committed, so the "
        "stepping branch had nothing to step through and this test would "
        "pass for the wrong reason"
    )
    assert max_idx > 1, (
        "ARC-071 commit-latch persistence fix has regressed: "
        "_committed_step_idx never advanced past 1 on a non-E3-tick step "
        "even with use_persistent_committed_program_handle=True"
    )


def test_arc071_step_idx_pinned_at_one_with_persistent_handle_off():
    """OFF: the negative control, and the reason the ON assertion has
    content. Multi-action programs ARE committed on this seed, so the
    pinned-at-1 ceiling below is not 'nothing to step through' -- it is
    post_action_update's unconditional teardown of _committed_trajectory
    leaving the stepping branch's `_step_traj` None on every non-E3-tick
    step, exactly the confirmed failure_autopsy_V3-EXQ-841_2026-07-31
    signature, reproduced here as a unit.
    """
    max_idx, multi_commits = _drive_real_rollout(handle_on=False)
    assert multi_commits > 0, (
        "anti-vacuity: the OFF arm must still COMMIT multi-action programs, "
        "otherwise max_idx<=1 is trivially true and controls nothing"
    )
    assert max_idx <= 1, (
        "OFF path is no longer bit-identical: _committed_step_idx advanced "
        f"to {max_idx} on a non-E3-tick step without the SD-084 handle, "
        "which contradicts the unconditional _committed_trajectory teardown "
        "in post_action_update -- the fallback may have engaged even with "
        "the flag off"
    )
