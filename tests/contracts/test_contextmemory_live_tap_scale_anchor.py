"""SD-CM-LIVETAP scale-anchor contracts (options (b) + (c), 2026-09-24).

WHAT THIS PINS. With the live-encoder write tap as the encoder's only gradient
source, the write-side losses are met by INFLATING z_world rather than by
discriminating content (validation pilots, 2026-09-24: H3_LIVE z_world norm
0.37 -> ~140, DIV_LIVE 0.37 -> 6.3, DETACHED twins flat ~0.41; an SD-070 warmup
still ended 31.8x / 96.6x above DETACHED). Two default-off flags anchor it:

  cfg.e1.contextmemory_write_tagger_scale_invariant  (b) per-block unit norm
      at the write_addr_tagger input, removing the radial gradient.
  cfg.e1.contextmemory_write_live_scale_anchor_margin (c) hinge on the tapped
      states' per-block norm against the first tapped batch's norm.

(b) ALONE WAS MEASURED NOT TO HOLD in the fixture below (DIV objective, 200 tap
updates): DETACHED 0.41, unanchored LIVE 212, LIVE with (b) 2.6 end / 5.7 peak.
(b)+(c): 0.30 end / 0.59 peak. So the load-bearing test pins the PAIR.

The load-bearing test MEASURES the live z_world norm; it does not assume it.
Against the pre-anchor e1_deep.py it FAILS (verified at build time: the two
cfg attributes are never read and live_scale_anchor_loss does not exist; with
that call removed the LIVE arm is exactly the unanchored path, measured ~500x
DETACHED). The unanchored arm is kept IN the test as a canary asserting the
fixture still inflates past 2x, so a fixture change that stopped reproducing
the defect cannot turn the anchored assertion vacuously true.

Machine-portability: nothing here depends on torch.multinomial (actions come
from a seeded torch.randint generator), so a worker-green run IS a gate.
"""
import pytest
import torch
import torch.nn.functional as F

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.predictors.e1_deep import ContextMemory
from ree_core.utils.config import REEConfig

SELF_DIM = 32
WORLD_DIM = 32
LATENT_DIM = SELF_DIM + WORLD_DIM
TAP_UPDATES = 200
TICKS_PER_UPDATE = 4
MARGIN = 2.0


def _cm(**kw):
    kw.setdefault("write_selection", "gumbel_learned")
    return ContextMemory(latent_dim=LATENT_DIM, memory_dim=32, num_slots=16, **kw)


def _agent(scale_invariant, margin, seed=42):
    torch.manual_seed(seed)
    env = CausalGridWorldV2(seed=7)
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim, world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim, self_dim=SELF_DIM, world_dim=WORLD_DIM,
        sd016_writepath_mode="sense_only",
        contextmemory_write_selection="gumbel_learned",
    )
    # None of these are E1Config fields / from_dims kwargs yet (from_dims
    # silently swallows unknown kwargs) -- set on cfg.e1 directly, exactly as
    # the tap's own contract does.
    cfg.e1.contextmemory_write_live_encoder_tap = 8
    cfg.e1.contextmemory_write_tagger_scale_invariant = scale_invariant
    cfg.e1.contextmemory_write_live_scale_anchor_margin = margin
    return REEAgent(cfg), env


def _tap_phase_zworld_norms(arm, scale_invariant, margin, seed=42):
    """TAP_UPDATES optimizer steps whose ONLY loss is the DIV write-addressing
    objective over the tapped states (plus the hinge when enabled). Returns the
    mean z_world norm observed per update. DETACHED = the same states detached,
    so the encoder receives no gradient: the reference twin."""
    agent, env = _agent(scale_invariant, margin, seed)
    cm = agent.e1.context_memory
    opt = torch.optim.Adam(
        list(agent.latent_stack.parameters())
        + list(cm.write_addr_tagger.parameters()),
        lr=1e-3,
    )
    gen = torch.Generator().manual_seed(seed)
    _, obs = env.reset()
    agent.reset()
    agent.e1.reset_hidden_state()
    norms = []
    for _ in range(TAP_UPDATES):
        tick_norms = []
        for _ in range(TICKS_PER_UPDATE):
            lat = agent.sense(obs["body_state"], obs["world_state"],
                              obs_harm=obs.get("harm_obs", None))
            tick_norms.append(float(lat.z_world.detach().norm(dim=-1).mean()))
            a = torch.zeros(1, env.action_dim)
            a[0, int(torch.randint(0, env.action_dim, (1,), generator=gen))] = 1.0
            agent._last_action = a
            _, _h, done, _i, obs = env.step(a)
            if done:
                _, obs = env.reset()
        norms.append(sum(tick_norms) / len(tick_norms))
        states = cm.take_live_write_states()
        if arm == "DETACHED":
            states = states.detach()
        probs = F.softmax(-cm.write_addr_tagger(states), dim=-1)
        pn = F.normalize(probs, dim=-1)
        n = states.shape[0]
        loss = (pn @ pn.T * (1.0 - torch.eye(n))).pow(2).sum() / (n * (n - 1))
        if margin > 0.0:
            loss = loss + cm.live_scale_anchor_loss(states)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return norms


# --- the point of the whole build --------------------------------------------

def test_live_zworld_norm_stays_within_2x_of_detached_with_anchor():
    detached = _tap_phase_zworld_norms("DETACHED", True, MARGIN)
    live = _tap_phase_zworld_norms("LIVE", True, MARGIN)
    unanchored = _tap_phase_zworld_norms("LIVE", False, 0.0)

    assert len(live) >= 200
    # Canary: the fixture must still reproduce the defect, or the assertion
    # below would pass vacuously. Measured ~500x at build time.
    assert max(unanchored) > 2.0 * max(detached), (
        "fixture no longer inflates the unanchored LIVE arm -- it cannot "
        f"discriminate an anchor (unanchored max {max(unanchored):.3f}, "
        f"detached max {max(detached):.3f})")
    # Load-bearing: measured over every one of the >= 200 tap updates.
    assert max(live) <= 2.0 * max(detached), (
        f"LIVE z_world norm peaked at {max(live):.3f}, > 2x DETACHED "
        f"{max(detached):.3f}")
    assert live[-1] >= 0.5 * detached[-1], (
        f"anchor over-shrank z_world: {live[-1]:.3f} vs DETACHED {detached[-1]:.3f}")


# --- (b): scale-invariant tagger input ---------------------------------------

def test_tagger_is_scale_invariant_per_block_when_on():
    torch.manual_seed(0)
    cm = _cm(write_tagger_scale_invariant=True, write_tagger_norm_split=SELF_DIM)
    x = torch.randn(5, LATENT_DIM)
    scaled = x.clone()
    scaled[:, :SELF_DIM] *= 3.0
    scaled[:, SELF_DIM:] *= 50.0  # z_world inflated independently
    assert torch.allclose(cm.write_addr_tagger(x), cm.write_addr_tagger(scaled),
                          atol=1e-5)


def test_tagger_is_not_scale_invariant_when_off():
    """Negative control: the default tagger DOES see scale -- which is the
    inflation route the anchor closes."""
    torch.manual_seed(0)
    cm = _cm()
    x = torch.randn(5, LATENT_DIM)
    assert not torch.allclose(cm.write_addr_tagger(x),
                              cm.write_addr_tagger(x * 50.0), atol=1e-3)


def test_default_off_is_bit_identical():
    torch.manual_seed(1)
    legacy = _cm(live_encoder_tap=8)
    torch.manual_seed(1)
    explicit = _cm(live_encoder_tap=8, write_tagger_scale_invariant=False,
                   write_tagger_norm_split=SELF_DIM,
                   live_scale_anchor_margin=0.0)
    a, b = legacy.state_dict(), explicit.state_dict()
    assert list(a) == list(b)
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert not any(k.startswith("live_scale_ref") for k, _ in explicit.named_buffers())
    x = torch.randn(3, LATENT_DIM)
    assert torch.equal(legacy.write_addr_tagger(x), explicit.write_addr_tagger(x))


def test_default_agent_is_bit_identical():
    """Agent-level: the new getattr reads with the attrs absent construct the
    same E1 as before."""
    agent_a, _ = _agent(False, 0.0)
    torch.manual_seed(42)
    env = CausalGridWorldV2(seed=7)
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim, world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim, self_dim=SELF_DIM, world_dim=WORLD_DIM,
        sd016_writepath_mode="sense_only",
        contextmemory_write_selection="gumbel_learned",
    )
    cfg.e1.contextmemory_write_live_encoder_tap = 8
    agent_b = REEAgent(cfg)
    a, b = agent_a.e1.state_dict(), agent_b.e1.state_dict()
    assert list(a) == list(b)
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_scale_invariant_requires_gumbel_learned():
    with pytest.raises(ValueError, match="gumbel_learned"):
        _cm(write_selection="argmin", write_tagger_scale_invariant=True)


# --- (c): running-norm hinge -------------------------------------------------

def test_anchor_requires_the_tap():
    with pytest.raises(ValueError, match="live-encoder tap"):
        _cm(live_scale_anchor_margin=2.0)


def test_anchor_loss_raises_when_disabled():
    cm = _cm(live_encoder_tap=8)
    with pytest.raises(RuntimeError, match="margin"):
        cm.live_scale_anchor_loss(torch.randn(2, LATENT_DIM))


def test_anchor_zero_in_band_positive_outside_and_reaches_states():
    cm = _cm(live_encoder_tap=8, write_tagger_norm_split=SELF_DIM,
             live_scale_anchor_margin=2.0)
    for _ in range(4):
        cm.record_live_write_state(torch.randn(1, LATENT_DIM))
    base = cm.take_live_write_states()          # calibrates the reference
    assert float(cm.live_scale_anchor_loss(base)) == 0.0
    cm.record_live_write_state(torch.randn(1, LATENT_DIM))
    cm.record_live_write_state(torch.randn(1, LATENT_DIM))
    big = cm.take_live_write_states().clone()
    big[:, SELF_DIM:] *= 10.0
    big.requires_grad_(True)
    loss = cm.live_scale_anchor_loss(big)
    assert float(loss.detach()) > 0.0
    loss.backward()
    assert big.grad is not None and torch.any(big.grad[:, SELF_DIM:] != 0)


def test_reference_is_frozen_after_first_take():
    cm = _cm(live_encoder_tap=8, write_tagger_norm_split=SELF_DIM,
             live_scale_anchor_margin=2.0)
    cm.record_live_write_state(torch.randn(2, LATENT_DIM))
    s = cm.take_live_write_states()
    ref = cm.live_scale_ref.clone()
    cm.live_scale_anchor_loss(s)
    cm.record_live_write_state(torch.randn(2, LATENT_DIM) * 100.0)
    cm.live_scale_anchor_loss(cm.take_live_write_states())
    assert torch.equal(cm.live_scale_ref, ref)


def test_forgotten_anchor_term_raises_on_next_take():
    """A driver that takes the tap but never adds the hinge would train
    unanchored while reporting nothing -- that must fail loudly."""
    cm = _cm(live_encoder_tap=8, live_scale_anchor_margin=2.0)
    cm.record_live_write_state(torch.randn(2, LATENT_DIM))
    cm.take_live_write_states()
    cm.record_live_write_state(torch.randn(2, LATENT_DIM))
    with pytest.raises(RuntimeError, match="never computed"):
        cm.take_live_write_states()


def test_clear_discharges_the_owed_anchor():
    cm = _cm(live_encoder_tap=8, live_scale_anchor_margin=2.0)
    cm.record_live_write_state(torch.randn(2, LATENT_DIM))
    cm.take_live_write_states()
    cm.clear_live_write_states()
    cm.record_live_write_state(torch.randn(2, LATENT_DIM))
    cm.take_live_write_states()  # must not raise


def test_addressing_loss_live_includes_anchor_when_on():
    """compute_write_addressing_loss_live consumes the tap itself, so it must
    add the hinge itself -- and discharge the guard."""
    torch.manual_seed(3)
    cm = _cm(live_encoder_tap=8, write_tagger_norm_split=SELF_DIM,
             live_scale_anchor_margin=2.0)
    for _ in range(3):
        cm.record_live_write_state(torch.randn(1, LATENT_DIM))
    cm.compute_write_addressing_loss_live()
    for _ in range(3):
        cm.record_live_write_state(torch.randn(1, LATENT_DIM) * 20.0)
    with_anchor = cm.compute_write_addressing_loss_live()
    assert float(with_anchor) > 1.0  # hinge term dominates: 20x the reference
    cm.record_live_write_state(torch.randn(2, LATENT_DIM))
    cm.take_live_write_states()  # guard was discharged -> no raise here
