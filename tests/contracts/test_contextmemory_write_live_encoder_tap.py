"""SD-CM-LIVETAP contracts: the ContextMemory live-encoder write tap.

WHAT THIS PINS. Every write-side training signal in ContextMemory used to reach
write_addr_tagger and stop, because compute_write_addressing_loss takes an
already-detached batch and the agent-side SD-016 Part B2 hook detaches before
writing. V3-EXQ-972a measured the consequence: 0 of 49 latent_stack parameters
changed, and trained-vs-UNTRAINED_ENCODER lineages hash-identical on 8 of 8
comparisons. The tap restores a gradient route from a write-side loss back into
the encoder, without altering the write itself.

ASSERTION POLICY, inherited from the sibling write-selection contract files: no
test here claims content-discrimination is achieved. That is exactly the open
question the validation experiment must answer. These assert mechanical
correctness -- gradient reachability, the default staying bit-identical, and
loud failure where a silent zero would be a negative instrument.

Machine-portability: nothing here asserts a committed action or any quantity
downstream of torch.multinomial, so a worker-green run IS a gate for this file
(CLAUDE.md, "Running the test suite").
"""
import pytest
import torch

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.predictors.e1_deep import ContextMemory
from ree_core.utils.config import REEConfig

LATENT_DIM = 64


def _cm(tap=0, selection="gumbel_learned"):
    return ContextMemory(latent_dim=LATENT_DIM, memory_dim=32, num_slots=16,
                         write_selection=selection, live_encoder_tap=tap)


def _action_onehot(idx, n, device):
    v = torch.zeros(1, n, device=device)
    v[0, idx] = 1.0
    return v


def _agent(tap, seed=42, self_dim=32, world_dim=32):
    torch.manual_seed(seed)
    env = CausalGridWorldV2(seed=7)
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim, world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim, self_dim=self_dim, world_dim=world_dim,
        sd016_writepath_mode="sense_only",
        contextmemory_write_selection="gumbel_learned",
    )
    # NOT a from_dims kwarg yet: the E1Config field is deferred (config.py was
    # held by another session at build time) and from_dims SILENTLY SWALLOWS
    # unknown kwargs, so passing it there would run with the tap off.
    cfg.e1.contextmemory_write_live_encoder_tap = tap
    return REEAgent(cfg), env


def _rollout(agent, env, steps=16):
    _, obs = env.reset()
    agent.reset()
    agent.e1.reset_hidden_state()
    detached = []
    for _ in range(steps):
        latent = agent.sense(obs["body_state"], obs["world_state"],
                             obs_harm=obs.get("harm_obs", None))
        detached.append(
            torch.cat([latent.z_self.detach(), latent.z_world.detach()], dim=-1)
        )
        a = _action_onehot(int(torch.randint(0, env.action_dim, (1,)).item()),
                           env.action_dim, agent.device)
        agent._last_action = a
        _, _h, done, _i, obs = env.step(a)
        if done:
            _, obs = env.reset()
    return detached


# --- default stays bit-identical ---------------------------------------------

def test_default_is_off():
    cm = _cm()
    assert cm.live_encoder_tap == 0
    assert cm.live_encoder_tap_enabled is False


def test_disabled_adds_no_state_dict_key():
    """The tap must not perturb checkpoints for any existing mode."""
    base = set(_cm(tap=0).state_dict().keys())
    assert set(_cm(tap=8).state_dict().keys()) == base


def test_disabled_adds_no_parameter_or_buffer():
    off, on = _cm(tap=0), _cm(tap=8)
    assert len(list(on.parameters())) == len(list(off.parameters()))
    assert len(list(on.named_buffers())) == len(list(off.named_buffers()))


def test_record_is_a_noop_when_disabled():
    cm = _cm(tap=0)
    cm.record_live_write_state(torch.randn(1, LATENT_DIM, requires_grad=True))
    assert cm._live_write_states is None


def test_negative_tap_clamps_to_off():
    assert _cm(tap=-5).live_encoder_tap == 0


def test_tap_works_in_every_selection_mode():
    """The tap captures state; it is orthogonal to the address rule."""
    for mode in ("argmin", "refractory", "gumbel_learned"):
        cm = _cm(tap=4, selection=mode)
        assert cm.live_encoder_tap_enabled is True
        cm.record_live_write_state(torch.randn(1, LATENT_DIM))
        assert cm.take_live_write_states().shape == (1, LATENT_DIM)


# --- the ring -----------------------------------------------------------------

def test_ring_is_bounded():
    cm = _cm(tap=3)
    for _ in range(10):
        cm.record_live_write_state(torch.randn(1, LATENT_DIM))
    assert cm.take_live_write_states().shape[0] == 3


def test_ring_keeps_the_most_recent():
    cm = _cm(tap=2)
    marks = []
    for i in range(5):
        t = torch.full((1, LATENT_DIM), float(i))
        marks.append(float(i))
        cm.record_live_write_state(t)
    got = cm.take_live_write_states()[:, 0].tolist()
    assert got == marks[-2:]


def test_take_clears_the_ring():
    """A second consumption in the same window must not re-use stale graph."""
    cm = _cm(tap=4)
    cm.record_live_write_state(torch.randn(1, LATENT_DIM))
    cm.take_live_write_states()
    with pytest.raises(RuntimeError, match="empty"):
        cm.take_live_write_states()


def test_clear_drops_without_consuming():
    cm = _cm(tap=4)
    cm.record_live_write_state(torch.randn(1, LATENT_DIM))
    cm.clear_live_write_states()
    with pytest.raises(RuntimeError, match="empty"):
        cm.take_live_write_states()


# --- loud, not silent ---------------------------------------------------------

def test_take_raises_when_tap_disabled():
    """Never an empty tensor: a caller would build a loss that trains nothing."""
    with pytest.raises(RuntimeError, match="live-encoder tap"):
        _cm(tap=0).take_live_write_states()


def test_live_loss_raises_when_tap_disabled():
    with pytest.raises(RuntimeError, match="live-encoder tap"):
        _cm(tap=0).compute_write_addressing_loss_live()


def test_live_loss_raises_on_under_filled_ring():
    """The sibling returns 0.0 at n<2, which is right for a caller-chosen batch
    and wrong here -- an under-filled ring means the tap did not capture what the
    caller assumed, and a silent 0.0 would be summed into a total and train
    nothing."""
    cm = _cm(tap=4)
    cm.record_live_write_state(torch.randn(1, LATENT_DIM))
    with pytest.raises(RuntimeError, match=">= 2"):
        cm.compute_write_addressing_loss_live()


def test_sibling_still_returns_zero_at_n1():
    """The existing method's contract is unchanged -- this build adds a path,
    it does not alter one."""
    cm = _cm(tap=0)
    out = cm.compute_write_addressing_loss(torch.randn(1, LATENT_DIM))
    assert float(out.item()) == 0.0


# --- the point of the whole build --------------------------------------------

def test_gradient_reaches_latent_stack_only_with_the_tap():
    """THE defect and its fix, in one comparison.

    OFF must reproduce V3-EXQ-972a's measured reading exactly (0 latent_stack
    parameters reached). ON must exceed it. Asserted as a strict inequality plus
    an exact zero, not a threshold, because the OFF reading is a hard structural
    zero rather than a small number.
    """
    agent_off, env_off = _agent(tap=0)
    detached = _rollout(agent_off, env_off)
    agent_off.zero_grad(set_to_none=True)
    agent_off.e1.context_memory.compute_write_addressing_loss(
        torch.cat(detached[-8:], dim=0)
    ).backward()
    off = sum(1 for p in agent_off.latent_stack.parameters()
              if p.grad is not None and torch.any(p.grad != 0))

    agent_on, env_on = _agent(tap=8)
    _rollout(agent_on, env_on)
    agent_on.zero_grad(set_to_none=True)
    agent_on.e1.context_memory.compute_write_addressing_loss_live().backward()
    on = sum(1 for p in agent_on.latent_stack.parameters()
             if p.grad is not None and torch.any(p.grad != 0))

    assert off == 0, "OFF must reproduce the 972a frozen-encoder reading"
    assert on > 0, "INERT: the tap reaches no latent_stack parameter"
    assert on > off


def test_gradient_still_reaches_the_tagger():
    """The existing guarantee must not regress."""
    agent, env = _agent(tap=8)
    _rollout(agent, env)
    cm = agent.e1.context_memory
    agent.zero_grad(set_to_none=True)
    cm.compute_write_addressing_loss_live().backward()
    assert any(p.grad is not None and torch.any(p.grad != 0)
               for p in cm.write_addr_tagger.parameters())


def test_take_live_states_carries_graph():
    """The general surface -- any loss routed through it reaches the encoder,
    which is what lets the standalone auxiliary loss and the read-path task
    gradient be compared rather than chosen between."""
    agent, env = _agent(tap=8)
    _rollout(agent, env)
    states = agent.e1.context_memory.take_live_write_states()
    assert states.requires_grad, "tapped states must carry graph"
    agent.zero_grad(set_to_none=True)
    states.sum().backward()
    assert any(p.grad is not None and torch.any(p.grad != 0)
               for p in agent.latent_stack.parameters())


def test_write_itself_stays_no_grad_with_the_tap_on():
    """The tap must not give the WRITE a graph -- write() is no_grad and updates
    self.memory by raw .data writes; that is unchanged."""
    agent, env = _agent(tap=8)
    _rollout(agent, env)
    cm = agent.e1.context_memory
    assert cm.memory.grad is None or torch.all(cm.memory.grad == 0)


def test_written_content_is_unchanged_by_the_tap():
    """Same seed, tap on vs off: the memory the agent actually wrote must be
    identical. The tap observes the write path, it does not alter it."""
    a_off, e_off = _agent(tap=0, seed=123)
    _rollout(a_off, e_off)
    a_on, e_on = _agent(tap=8, seed=123)
    _rollout(a_on, e_on)
    assert torch.equal(a_off.e1.context_memory.memory.detach(),
                       a_on.e1.context_memory.memory.detach())
    assert a_off.e1.context_memory.slot_write_counts.tolist() == \
        a_on.e1.context_memory.slot_write_counts.tolist()
