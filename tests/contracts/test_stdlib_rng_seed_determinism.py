"""Contract tests for REEConfig.stdlib_rng_seed (2026-07-28).

Three ree_core call sites draw from the stdlib `random` module, which auto-seeds
from OS entropy at import while experiment drivers seed only torch and numpy:

  S1  HippocampalModule.diverse_replay(mode="auto")     -- per-step mode roll
  S2  HippocampalModule._sample_exploration_trajectory  -- zero-weight fallback
  S3  SelfModelAggregator.offline_gradient_pass         -- waking-pair sample

`stdlib_rng_seed` makes those reproducible ON DEMAND. It is deliberately opt-in:
seeding CHANGES RESULTS for any run that reaches a site, so the default must stay
bit-identical to every landed run.

Both directions are pinned, per the scaffold_env_seed precedent
(tests/contracts/test_scaffold_env_seed_determinism.py):

  DEFAULT IS A NO-OP -- with the knob unset, each site still resolves through the
  process-global `random` instance (identity-checked, not merely
  behaviour-checked), and its draws still track random.seed().

  SET IS DETERMINISTIC -- with the knob set, repeated construction reproduces the
  same draw sequence, the sequence is INDEPENDENT of the global RNG state, and
  the global RNG is never perturbed by the seeding.
"""

import random

import pytest
import torch

from ree_core.agent import REEAgent
from ree_core.predictors.e2_fast import Trajectory
from ree_core.sleep.self_model_aggregator import (
    SelfModelAggregator,
    SelfModelAggregatorConfig,
)
from ree_core.utils.config import (
    STDLIB_RNG_STREAM_HIPPOCAMPAL_REPLAY,
    STDLIB_RNG_STREAM_SELF_MODEL_WRITEBACK,
    derive_stdlib_rng_seed,
)

from tests.fixtures.seed_utils import set_all_seeds
from tests.fixtures.tiny_configs import make_tiny_config
from tests.fixtures.tiny_env import make_tiny_env


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #
def _build_agent(*, stdlib_rng_seed=None):
    """Tiny agent with the MECH-165 diverse-replay path enabled."""
    set_all_seeds(0)
    env = make_tiny_env(seed=0)
    cfg = make_tiny_config(
        env,
        replay_diversity_enabled=True,
        stdlib_rng_seed=stdlib_rng_seed,
    )
    agent = REEAgent(cfg)
    agent.reset()
    return cfg, agent


def _fill_exploration_buffer(cfg, agent, n=3):
    z_self = torch.zeros(1, cfg.latent.self_dim)
    z_world = torch.zeros(1, cfg.latent.world_dim)
    actions = torch.zeros(1, 1, cfg.e2.action_dim)
    actions[0, 0, 0] = 1.0
    agent.hippocampal._exploration_buffer.clear()
    for i in range(n):
        agent.hippocampal.record_exploration_trajectory(
            Trajectory(
                states=[z_self.clone(), z_self.clone()],
                actions=actions.clone(),
                world_states=[z_world.clone(), z_world.clone()],
                memory_strength=float(i + 1),
            )
        )


def _replay_rolls(agent, n=24):
    """The raw S1 mode-roll sequence, read straight off the module's RNG source.

    Reading the rolls directly (rather than inferring them from returned
    trajectories) keeps the contract on the RNG plumbing under test and off the
    downstream replay machinery, which has its own torch-seeded stochasticity.
    """
    return [agent.hippocampal._rng.random() for _ in range(n)]


def _aggregator(rng_seed=None):
    return SelfModelAggregator(
        SelfModelAggregatorConfig(offline_n_steps=2, rng_seed=rng_seed)
    )


# --------------------------------------------------------------------------- #
# D1-D3: DEFAULT IS A NO-OP                                                    #
# --------------------------------------------------------------------------- #
def test_d1_default_leaves_hippocampal_on_the_global_random_instance():
    """Unset knob -> the module's RNG source IS the `random` module.

    Identity, not equivalence: `self._rng.random` must be the very same bound
    method the pre-knob `random.random()` call resolved to, which is what makes
    the default path bit-identical rather than merely statistically alike.
    """
    _cfg, agent = _build_agent()
    assert agent.hippocampal._rng is random
    assert agent.hippocampal._rng.random is random.random
    assert agent.hippocampal._rng.choice is random.choice


def test_d2_default_leaves_aggregator_on_the_global_random_instance():
    agg = _aggregator()
    assert agg._rng is random
    assert agg._rng.choices is random.choices


def test_d3_default_draws_still_track_the_global_seed():
    """Negative control for D1/D2: unseeded, the sites really do consume the
    global RNG -- so a global random.seed() reproduces them. This is what the
    seeded case below must BREAK away from; without it, D4/D5 could pass on a
    site that never touched stdlib random at all."""
    _cfg, agent = _build_agent()
    _fill_exploration_buffer(_cfg, agent)

    random.seed(1234)
    first = _replay_rolls(agent)
    random.seed(1234)
    second = _replay_rolls(agent)
    assert first == second

    random.seed(4321)
    third = _replay_rolls(agent)
    assert first != third, "unseeded module did not consume the global RNG"


def test_d3b_derive_returns_none_for_an_unset_base():
    """The derivation must pass None straight through, so every consumer's
    'seed is None -> keep the global instance' branch is the one taken."""
    assert derive_stdlib_rng_seed(None, STDLIB_RNG_STREAM_HIPPOCAMPAL_REPLAY) is None
    assert derive_stdlib_rng_seed(None, STDLIB_RNG_STREAM_SELF_MODEL_WRITEBACK) is None


# --------------------------------------------------------------------------- #
# D4-D7: SET IS DETERMINISTIC                                                  #
# --------------------------------------------------------------------------- #
def test_d4_seeded_hippocampal_reproduces_across_constructions():
    _cfg_a, agent_a = _build_agent(stdlib_rng_seed=7)
    _cfg_b, agent_b = _build_agent(stdlib_rng_seed=7)

    assert agent_a.hippocampal._rng is not random
    assert isinstance(agent_a.hippocampal._rng, random.Random)
    assert _replay_rolls(agent_a) == _replay_rolls(agent_b)


def test_d5_seeded_hippocampal_is_independent_of_the_global_rng():
    """The whole point: two PROCESSES differ only in their OS-entropy global
    seed, which the global random.seed() calls here stand in for."""
    _cfg_a, agent_a = _build_agent(stdlib_rng_seed=7)
    random.seed(999)
    rolls_a = _replay_rolls(agent_a)

    _cfg_b, agent_b = _build_agent(stdlib_rng_seed=7)
    random.seed(111)
    rolls_b = _replay_rolls(agent_b)

    assert rolls_a == rolls_b


def test_d6_distinct_seeds_give_distinct_streams():
    _cfg_a, agent_a = _build_agent(stdlib_rng_seed=7)
    _cfg_b, agent_b = _build_agent(stdlib_rng_seed=8)
    assert _replay_rolls(agent_a) != _replay_rolls(agent_b)


def test_d7_seeding_never_perturbs_the_global_rng():
    """A module-local random.Random, never random.seed(). If this regressed to
    a global reseed, building an agent would silently move every other stdlib
    random consumer in the host process."""
    # Build FIRST, then seed. _build_agent calls set_all_seeds(), which reseeds
    # the global itself -- seeding before construction would measure the test
    # fixture rather than the substrate.
    _cfg, agent = _build_agent(stdlib_rng_seed=7)
    _fill_exploration_buffer(_cfg, agent)

    # Independent reference stream, so capturing the baseline does not itself
    # consume the global draws under test.
    ref = random.Random(2026)
    expected = [ref.random() for _ in range(5)]

    random.seed(2026)
    agent.hippocampal.diverse_replay(
        torch.zeros(3, 1, _cfg.latent.world_dim), num_replay_steps=6, mode="auto"
    )
    # Zero global draws consumed and no global reseed: the next 5 global draws
    # are still the first 5 of the seed-2026 stream.
    assert [random.random() for _ in range(5)] == expected


# --------------------------------------------------------------------------- #
# D8-D9: S2 fallback + S3 writeback carry the same guarantees                  #
# --------------------------------------------------------------------------- #
def test_d8_seeded_zero_weight_fallback_is_reproducible():
    """S2 fires only when the BLA retrieval_bias zeroes every weight."""
    zero_bias = torch.zeros(3)

    def picks(seed, global_seed):
        cfg, agent = _build_agent(stdlib_rng_seed=seed)
        _fill_exploration_buffer(cfg, agent)
        buf = agent.hippocampal._exploration_buffer
        random.seed(global_seed)
        out = []
        for _ in range(12):
            traj = agent.hippocampal._sample_exploration_trajectory(
                retrieval_bias=zero_bias
            )
            # Identity lookup: which buffer slot came back.
            out.append(next(i for i, t in enumerate(buf) if t is traj))
        return out

    # Seeded: identical across differing global RNG states.
    assert picks(7, 111) == picks(7, 999)
    # Unseeded (default): follows the global RNG -- the negative control.
    assert picks(None, 111) != picks(None, 999)


def test_d9_seeded_aggregator_writeback_sample_is_reproducible():
    from ree_core.predictors.e2_harm_s import E2HarmSConfig, E2HarmSForward
    from ree_core.sleep.phase_manager import SleepPhase
    from ree_core.sleep.routing_gate import RoutedEvent

    z_dim, a_dim = 8, 4
    regions = [("fast", "0.1"), ("fast", "0.2")]
    buffer = [
        (torch.full((1, z_dim), float(i)), torch.full((1, a_dim), float(i)))
        for i in range(8)
    ]

    def sampled_rows(rng_seed, global_seed):
        e2 = E2HarmSForward(
            E2HarmSConfig(
                use_e2_harm_s_forward=True,
                z_harm_dim=z_dim,
                action_dim=a_dim,
                hidden_dim=16,
            )
        )
        agg = _aggregator(rng_seed=rng_seed)
        for r in regions:
            agg.update(
                RoutedEvent(
                    event=r,
                    anchor_channel=0.6,
                    probe_channel=0.4,
                    phase=SleepPhase.SWS_ANALOG,
                ),
                evidence=2.0,
                domain="self",
            )
        random.seed(global_seed)
        seen = []
        for _ in range(10):
            seen.append(tuple(float(p[0][0, 0]) for p in agg._rng.choices(buffer, k=2)))
        return seen

    # Seeded: identical across differing global RNG states.
    assert sampled_rows(5, 111) == sampled_rows(5, 999)
    # Unseeded (default): follows the global RNG -- the negative control.
    assert sampled_rows(None, 111) != sampled_rows(None, 999)


# --------------------------------------------------------------------------- #
# D10: stream namespacing                                                      #
# --------------------------------------------------------------------------- #
def test_d10_consumers_get_distinct_streams_from_one_base():
    """One base seed must not hand two consumers the same sequence."""
    base = 7
    hip = derive_stdlib_rng_seed(base, STDLIB_RNG_STREAM_HIPPOCAMPAL_REPLAY)
    sma = derive_stdlib_rng_seed(base, STDLIB_RNG_STREAM_SELF_MODEL_WRITEBACK)
    assert hip != sma
    a = [random.Random(hip).random() for _ in range(3)]
    b = [random.Random(sma).random() for _ in range(3)]
    assert a != b


def test_d11_from_dims_mirrors_the_knob_onto_the_hippocampal_subconfig():
    """The from_dims THREE-SITE requirement: field, signature, assignment. A
    knob wired at only two of them is silently unreachable."""
    env = make_tiny_env(seed=0)
    cfg_off = make_tiny_config(env)
    assert cfg_off.stdlib_rng_seed is None
    assert cfg_off.hippocampal.replay_rng_seed is None

    cfg_on = make_tiny_config(env, stdlib_rng_seed=7)
    assert cfg_on.stdlib_rng_seed == 7
    assert cfg_on.hippocampal.replay_rng_seed == derive_stdlib_rng_seed(
        7, STDLIB_RNG_STREAM_HIPPOCAMPAL_REPLAY
    )


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
