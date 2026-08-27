"""Contracts for ContextMemory write-address selection, THIRD mechanism:
`contextmemory_write_selection="gumbel_learned"`.

substrate_queue `contextmemory-write-path-addressing-degeneracy` (severity
`corrupting`). See test_contextmemory_write_usage_balancing.py (conscience
bias) and test_contextmemory_write_address_selection.py (refractory) for the
first two mechanisms and the shared defect background -- not duplicated here.

WHY A THIRD MECHANISM, AND WHY THIS ONE IS DIFFERENT FROM THE REJECTED DRAFT.
HUMAN DECISION 2026-08-26 (recorded in substrate_queue.json): neither the
conscience bias nor refractory closes the corrupting defect as a matter of
addressing POLICY -- both are mechanical occupancy workarounds, not a
learned/content-based write-selection mechanism. The substrate_queue entry's
own implementation_hint names the real fix: apply annealed Gumbel-softmax
(V3-EXQ-908, confirmed on the READ path) to the WRITE address.

The superseded draft already tried exactly that literally (a Gumbel-perturbed
argmax over the SAME untrained `query_proj(state) @ memory.T` score write()
already computes) and it was measured content-blind -- 2-cluster Jaccard
EXACTLY 1.000 on 5/5 seeds -- and NOT landed, because
`write() runs entirely under torch.no_grad()`: gumbel-softmax sharpens a
TRAINED distribution on the read path (terrain_loss/action-prediction losses
shape cue_slot_tagger); with nothing shaping the write-side score, the same
operator just randomises an untrained one.

THIS mode answers that specific critique with a real, verified gradient path
(compute_write_addressing_loss(), see its own docstring for the two-attempt
design history) -- but MEASURED THE SAME empirical signature under a toy
training loop (2-cluster Jaccard 1.000 on 5/5 seeds, see that method's
docstring "FIRST-ATTEMPT DESIGN" section) before the loss was redesigned to
mirror compute_diversification_loss()'s pairwise-cosine structure instead of
a batch-mean load-balancing term. THE REDESIGNED LOSS'S CONTENT-DISCRIMINATION
EFFECT IS NOT YET DEMONSTRATED EITHER -- gradient reachability is proven at
the unit level (this file), but moving that gradient enough to produce
genuine content-conditioned addressing needs real training, exactly as
compute_diversification_loss() itself needed a real experiment (V3-EXQ-907)
rather than a toy loop to show effect (EXQ-418d: "v2 read+write gradients
alone cannot break slot symmetry"). See
REE_assembly/docs/architecture/contextmemory_write_address_selection.md for
the full measurement record and status.

ASSERTION POLICY, extending the sibling file's: no assertion here claims
content-discrimination is achieved. Assertions are on (a) mechanical
correctness -- gradient reaches write_addr_tagger, RNG consumption is exactly
where documented, defaults are untouched, config validation is fail-closed --
and (b) the one thing this mode DOES decisively deliver regardless of
training: occupancy, via Gumbel noise alone, independent of what the tagger
has learned.
"""

import math

import torch
import pytest

from ree_core.predictors.e1_deep import ContextMemory

LATENT_DIM, MEMORY_DIM, NUM_SLOTS = 64, 128, 16


def _stream(seed, n, jitter=0.0078, clusters=1):
    gen = torch.Generator().manual_seed(seed)
    bases = [torch.randn(1, LATENT_DIM, generator=gen) * 0.078 for _ in range(clusters)]
    return [
        (i % clusters, bases[i % clusters] + torch.randn(1, LATENT_DIM, generator=gen) * jitter)
        for i in range(n)
    ]


def _run(seed, n=1500, clusters=1, **kwargs):
    torch.manual_seed(seed)
    cm = ContextMemory(LATENT_DIM, MEMORY_DIM, NUM_SLOTS,
                       gated_content_write=True, **kwargs)
    per_cluster = {}
    sequence = []
    for cid, state in _stream(seed, n, clusters=clusters):
        cm.write(state)
        per_cluster.setdefault(cid, set()).add(cm.last_write_index)
        sequence.append(cm.last_write_index)
    return cm, per_cluster, sequence


LOCKING_SEEDS = (0, 100)
ROTATING_SEEDS = (7, 13, 42)
ALL_SEEDS = LOCKING_SEEDS + ROTATING_SEEDS

GUMBEL_LEARNED = {"write_selection": "gumbel_learned"}


# --------------------------------------------------------------------------
# Backward compatibility. Constructing with the new kwargs present but the
# mode not selected must not change anything.
# --------------------------------------------------------------------------

def test_default_selection_is_still_argmin():
    cm = ContextMemory(LATENT_DIM, MEMORY_DIM, NUM_SLOTS)
    assert cm.write_selection == "argmin"
    assert cm.write_addr_tagger is None


def test_write_addr_tagger_constructed_only_in_gumbel_learned_mode():
    for kwargs in ({}, {"write_selection": "refractory"}):
        cm = ContextMemory(LATENT_DIM, MEMORY_DIM, NUM_SLOTS, **kwargs)
        assert cm.write_addr_tagger is None, kwargs
    cm = ContextMemory(LATENT_DIM, MEMORY_DIM, NUM_SLOTS, **GUMBEL_LEARNED)
    assert cm.write_addr_tagger is not None


def test_default_is_bit_identical_to_the_legacy_expression():
    """Same negative control as the sibling file, re-run here so a regression
    introduced by THIS mode's changes to write()/__init__ would be caught by
    either file independently."""
    for seed in ALL_SEEDS:
        torch.manual_seed(seed)
        cm = ContextMemory(LATENT_DIM, MEMORY_DIM, NUM_SLOTS, gated_content_write=True)
        torch.manual_seed(seed)
        ref = ContextMemory(LATENT_DIM, MEMORY_DIM, NUM_SLOTS, gated_content_write=True)
        for _, state in _stream(seed, 200):
            with torch.no_grad():
                legacy_idx = int(
                    torch.mm(ref.query_proj(state), ref.memory.t()).mean(0).argmin().item()
                )
                signal = ref.write_gate(state) * ref.write_content(state)
                ref.memory.data[legacy_idx] = (
                    0.9 * ref.memory.data[legacy_idx] + 0.1 * signal.mean(0)
                )
            cm.write(state)
            assert cm.last_write_index == legacy_idx
        assert torch.equal(cm.memory.data, ref.memory.data)


def test_state_dict_gains_tagger_keys_only_in_gumbel_learned_mode():
    off_keys = set(ContextMemory(LATENT_DIM, MEMORY_DIM, NUM_SLOTS).state_dict().keys())
    assert not any(k.startswith("write_addr_tagger") for k in off_keys)

    on_keys = set(
        ContextMemory(LATENT_DIM, MEMORY_DIM, NUM_SLOTS, **GUMBEL_LEARNED).state_dict().keys()
    )
    tagger_keys = {k for k in on_keys if k.startswith("write_addr_tagger")}
    assert tagger_keys == {
        "write_addr_tagger.0.weight", "write_addr_tagger.0.bias",
        "write_addr_tagger.2.weight", "write_addr_tagger.2.bias",
    }, tagger_keys


# --------------------------------------------------------------------------
# Config validation.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("bad", ["bogus", "usage", "gumbel", "Gumbel_Learned", ""])
def test_invalid_selection_mode_is_rejected(bad):
    """`gumbel` (bare) stays rejected on purpose -- it is NOT an accepted
    spelling of this mode, to avoid any reader mistaking a stray reference to
    the superseded, rejected mode for a silent resurrection under the same
    string. This is intentionally the SAME parametrize list (plus a near-miss
    case) as the sibling file's test of the same name -- if that file's list
    and this one ever diverge on the shared bad values, one of them is wrong."""
    with pytest.raises(ValueError, match="contextmemory_write_selection"):
        ContextMemory(LATENT_DIM, MEMORY_DIM, NUM_SLOTS, write_selection=bad)


def test_gumbel_learned_is_accepted():
    cm = ContextMemory(LATENT_DIM, MEMORY_DIM, NUM_SLOTS, **GUMBEL_LEARNED)
    assert cm.write_selection == "gumbel_learned"


def test_config_plumbs_through_from_dims():
    from ree_core.utils.config import REEConfig
    cfg = REEConfig.from_dims(
        body_obs_dim=8, world_obs_dim=8, action_dim=5, self_dim=32, world_dim=32,
        contextmemory_write_selection="gumbel_learned",
        contextmemory_write_gumbel_tau_init=0.8,
        contextmemory_write_gumbel_tau_min=0.05,
        contextmemory_write_gumbel_anneal_steps=500,
        contextmemory_write_gumbel_tagger_hidden=48,
        contextmemory_write_addressing_loss_weight=0.3,
    )
    assert cfg.e1.contextmemory_write_selection == "gumbel_learned"
    assert cfg.e1.contextmemory_write_gumbel_tau_init == 0.8
    assert cfg.e1.contextmemory_write_gumbel_tau_min == 0.05
    assert cfg.e1.contextmemory_write_gumbel_anneal_steps == 500
    assert cfg.e1.contextmemory_write_gumbel_tagger_hidden == 48
    # TOP-LEVEL, not e1-scoped -- see the config.py note beside
    # sd016_diversification_weight for why.
    assert cfg.contextmemory_write_addressing_loss_weight == 0.3

    default = REEConfig.from_dims(body_obs_dim=8, world_obs_dim=8, action_dim=5,
                                  self_dim=32, world_dim=32)
    assert default.e1.contextmemory_write_selection == "argmin"
    assert default.e1.contextmemory_write_gumbel_tau_init == 1.0
    assert default.e1.contextmemory_write_gumbel_tau_min == 0.1
    assert default.e1.contextmemory_write_gumbel_anneal_steps == 2000
    assert default.e1.contextmemory_write_gumbel_tagger_hidden == 32
    assert default.contextmemory_write_addressing_loss_weight == 0.0


def test_e1_predictor_wires_the_selection_through():
    from ree_core.predictors.e1_deep import E1DeepPredictor
    from ree_core.utils.config import REEConfig
    cfg = REEConfig.from_dims(
        body_obs_dim=8, world_obs_dim=8, action_dim=5, self_dim=32, world_dim=32,
        contextmemory_write_selection="gumbel_learned",
        contextmemory_write_gumbel_tagger_hidden=48,
    )
    e1 = E1DeepPredictor(cfg.e1)
    assert e1.context_memory.write_selection == "gumbel_learned"
    assert e1.context_memory.write_addr_tagger is not None
    assert e1.context_memory.write_addr_tagger[0].out_features == 48


# --------------------------------------------------------------------------
# RNG consumption. Deliberately different from argmin/refractory -- this
# mode's whole point is stochastic exploration during training.
# --------------------------------------------------------------------------

def test_eval_mode_consumes_no_rng_and_matches_plain_argmin():
    """Eval-mode gumbel_learned must be deterministic AND must equal exactly
    what `selection_scores.argmin()` would give on write_addr_tagger's own
    scores -- not merely close. A monotonic temperature rescaling cannot
    change the argmax of -selection_scores, so this is an exact equivalence,
    not an approximation."""
    states = [s for _, s in _stream(0, 30)]

    torch.manual_seed(1)
    cm = ContextMemory(LATENT_DIM, MEMORY_DIM, NUM_SLOTS, **GUMBEL_LEARNED)
    cm.eval()

    torch.manual_seed(123)
    before = torch.rand(3)
    torch.manual_seed(123)
    written = []
    for s in states:
        with torch.no_grad():
            expected = cm.write_addr_tagger(s).mean(0).argmin().item()
        cm.write(s)
        written.append((cm.last_write_index, expected))
    after = torch.rand(3)

    assert torch.equal(before, after), "eval-mode write() consumed RNG"
    for got, expected in written:
        assert got == expected, f"eval-mode selection {got} != plain argmin {expected}"


def test_train_mode_consumes_rng():
    states = [s for _, s in _stream(0, 30)]
    torch.manual_seed(1)
    cm = ContextMemory(LATENT_DIM, MEMORY_DIM, NUM_SLOTS, **GUMBEL_LEARNED)
    assert cm.training  # nn.Module default

    torch.manual_seed(123)
    before = torch.rand(3)
    torch.manual_seed(123)
    for s in states:
        cm.write(s)
    after = torch.rand(3)
    assert not torch.equal(before, after), "train-mode write() did not consume RNG"


@pytest.mark.parametrize("kwargs", [{}, {"write_selection": "refractory"}],
                         ids=["argmin", "refractory"])
def test_other_modes_still_consume_no_rng_in_train_mode(kwargs):
    """NEGATIVE CONTROL. This mode's RNG consumption must not leak into the
    other two -- they stay fully deterministic in every mode, exactly as
    pinned by the sibling file."""
    states = [s for _, s in _stream(0, 30)]
    torch.manual_seed(1)
    cm = ContextMemory(LATENT_DIM, MEMORY_DIM, NUM_SLOTS, gated_content_write=True, **kwargs)
    cm.train()
    torch.manual_seed(123)
    before = torch.rand(3)
    torch.manual_seed(123)
    for s in states:
        cm.write(s)
    after = torch.rand(3)
    assert torch.equal(before, after)


# --------------------------------------------------------------------------
# Occupancy. What this mode decisively delivers regardless of training.
# --------------------------------------------------------------------------

def test_gumbel_learned_clears_the_registered_failure_floor():
    """V3-EXQ-436f's registered target: >= 2 occupied slots on >= 3/5 seeds.
    Delivered by Gumbel noise alone during training -- true even for a freshly
    initialised, completely untrained write_addr_tagger, independent of
    whether compute_write_addressing_loss() has been optimised at all."""
    passing = sum(
        1 for seed in ALL_SEEDS
        if len(_run(seed, **GUMBEL_LEARNED)[0].occupied_slots()) >= 2
    )
    assert passing == len(ALL_SEEDS), f"only {passing}/{len(ALL_SEEDS)} seeds cleared the floor"


def test_gumbel_learned_breaks_the_locking_seeds():
    for seed in LOCKING_SEEDS:
        cm, _, _ = _run(seed, **GUMBEL_LEARNED)
        assert len(cm.occupied_slots()) >= 2, f"seed {seed} still locked"


def test_write_counts_track_occupancy():
    cm, _, _ = _run(0, n=300, **GUMBEL_LEARNED)
    assert float(cm.slot_write_counts.sum()) == 300.0
    assert cm.occupied_slots() == [
        i for i in range(NUM_SLOTS) if float(cm.slot_write_counts[i]) > 0
    ]


def test_write_records_the_slot_it_actually_wrote():
    """Same instrumentation invariant the sibling file pins for refractory --
    re-checked here since this mode's selection expression is entirely new."""
    torch.manual_seed(0)
    cm = ContextMemory(LATENT_DIM, MEMORY_DIM, NUM_SLOTS, gated_content_write=True,
                       **GUMBEL_LEARNED)
    for _, state in _stream(0, 50):
        before = cm.memory.data.clone()
        cm.write(state)
        changed = [i for i in range(NUM_SLOTS)
                   if not torch.equal(before[i], cm.memory.data[i])]
        assert changed == [cm.last_write_index]


# --------------------------------------------------------------------------
# Composition with the landed conscience bias. No claim of usefulness, only
# that it does not error and both knobs remain independently readable.
# --------------------------------------------------------------------------

def test_composes_with_usage_balancing_without_error():
    cm, _, _ = _run(0, n=200, write_usage_balancing=True, **GUMBEL_LEARNED)
    assert float(cm.slot_write_counts.sum()) == 200.0
    assert len(cm.occupied_slots()) >= 1


# --------------------------------------------------------------------------
# compute_write_addressing_loss(): the gradient path that answers the
# superseded draft's "no gradient at all" critique.
# --------------------------------------------------------------------------

def test_raises_outside_gumbel_learned_mode():
    for kwargs in ({}, {"write_selection": "refractory"}):
        cm = ContextMemory(LATENT_DIM, MEMORY_DIM, NUM_SLOTS, **kwargs)
        with pytest.raises(RuntimeError, match="gumbel_learned"):
            cm.compute_write_addressing_loss(torch.randn(4, LATENT_DIM))


def test_single_state_batch_is_a_vacuous_zero_not_a_crash():
    cm = ContextMemory(LATENT_DIM, MEMORY_DIM, NUM_SLOTS, **GUMBEL_LEARNED)
    loss = cm.compute_write_addressing_loss(torch.randn(1, LATENT_DIM))
    assert float(loss) == 0.0


def test_loss_is_well_formed():
    """Bounded, finite, and matches the pairwise-mean-squared-cosine-similarity
    definition compute_diversification_loss() uses for memory rows, applied
    here to per-example selection distributions instead."""
    torch.manual_seed(0)
    cm = ContextMemory(LATENT_DIM, MEMORY_DIM, NUM_SLOTS, **GUMBEL_LEARNED)
    states = torch.randn(12, LATENT_DIM) * 0.078
    loss = cm.compute_write_addressing_loss(states)
    assert torch.isfinite(loss)
    assert 0.0 <= float(loss.detach()) <= 1.0 + 1e-4

    with torch.no_grad():
        import torch.nn.functional as F
        scores = cm.write_addr_tagger(states)
        probs = F.softmax(-scores, dim=-1)
        probs_norm = F.normalize(probs, dim=-1)
        sim = probs_norm @ probs_norm.T
        n = states.shape[0]
        mask = 1.0 - torch.eye(n)
        expected = (sim * mask).pow(2).sum() / (n * (n - 1))
    assert torch.allclose(loss, expected, atol=1e-6)


def test_gradient_reaches_write_addr_tagger():
    """THE load-bearing contract for this whole mode: proof that, unlike the
    superseded draft's rejected "gumbel" (which perturbed a permanently
    untrained score), this mechanism has a real, checkable gradient path into
    the parameters that determine write-address selection."""
    torch.manual_seed(0)
    cm = ContextMemory(LATENT_DIM, MEMORY_DIM, NUM_SLOTS, **GUMBEL_LEARNED)
    states = torch.randn(16, LATENT_DIM) * 0.078
    loss = cm.compute_write_addressing_loss(states)
    loss.backward()
    for name, p in cm.write_addr_tagger.named_parameters():
        assert p.grad is not None, f"{name} received no gradient"
        assert float(p.grad.abs().sum()) > 0.0, f"{name} gradient is all-zero"


def test_write_itself_stays_fully_no_grad():
    """NEGATIVE CONTROL, the other half of the design's safety argument:
    write() must NOT retain any graph through write_addr_tagger, even in
    train mode -- compute_write_addressing_loss() is the ONLY gradient path,
    called explicitly and separately by the training loop. If this ever
    fails, write() has started building a per-call graph, which is exactly
    the retained-stale-graph hazard compute_write_addressing_loss()'s
    docstring documents avoiding."""
    torch.manual_seed(0)
    cm = ContextMemory(LATENT_DIM, MEMORY_DIM, NUM_SLOTS, **GUMBEL_LEARNED)
    cm.train()
    for _, state in _stream(0, 10):
        cm.write(state)
    for p in cm.write_addr_tagger.parameters():
        assert p.grad is None, "write() left a gradient on write_addr_tagger"
    assert cm.memory.grad is None
