"""C5: imagined/acted isolation (MECH-094 gate semantics).

The contract: when update_residue(..., hypothesis_tag=True), ResidueField
state must not change. When hypothesis_tag=False, state changes (given a
non-trivial harm signal at a non-trivial z_world location).

We deliberately test the GATE, not the tag mechanism. Any future refactor
that keeps this behavioural isolation (e.g. a mode flag on the agent rather
than a kwarg) should still make this test pass.
"""

import torch

from ree_core.agent import REEAgent

from tests.fixtures.seed_utils import set_all_seeds
from tests.fixtures.tiny_env import make_tiny_env
from tests.fixtures.tiny_configs import make_tiny_config
from tests.fixtures.tiny_loop import _obs_tensors


def _residue_snapshot(agent: REEAgent) -> tuple:
    """Flat snapshot of residue-carrying state (center weights + total)."""
    rbf = agent.residue_field.rbf_field
    weights = rbf.weights.detach().clone()
    centers = rbf.centers.detach().clone()
    active = rbf.active_mask.detach().clone()
    total = agent.residue_field.total_residue.detach().clone()
    return weights, centers, active, total


def _snapshots_equal(a, b) -> bool:
    return all(torch.equal(x, y) for x, y in zip(a, b))


def _prime_agent(seed: int = 0):
    set_all_seeds(seed)
    env = make_tiny_env(seed=seed)
    cfg = make_tiny_config(env)
    agent = REEAgent(cfg)
    agent.reset()
    _flat, obs_dict = env.reset()
    body, world = _obs_tensors(obs_dict)
    with torch.no_grad():
        agent.sense(obs_body=body, obs_world=world)
    return agent


HARM = -0.5  # harm_signal must be negative to trigger residue accumulation


def test_hypothesis_tag_true_blocks_residue_write():
    agent = _prime_agent()
    before = _residue_snapshot(agent)
    agent.update_residue(harm_signal=HARM, hypothesis_tag=True)
    after = _residue_snapshot(agent)
    assert _snapshots_equal(before, after), \
        "MECH-094: update_residue(hypothesis_tag=True) mutated ResidueField state"


def test_hypothesis_tag_false_permits_residue_write():
    agent = _prime_agent()
    before = _residue_snapshot(agent)
    agent.update_residue(harm_signal=HARM, hypothesis_tag=False)
    after = _residue_snapshot(agent)
    assert not _snapshots_equal(before, after), \
        "Waking update_residue(hypothesis_tag=False, harm_signal=-0.5) produced " \
        "no state change -- the contract test cannot distinguish tagged/untagged behaviour"
