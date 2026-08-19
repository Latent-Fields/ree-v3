"""Contracts for ContextMemory write-address selection (the `refractory` mode).

substrate_queue `contextmemory-write-path-addressing-degeneracy` (severity
`corrupting`). The legacy write address is a hard `scores.mean(0).argmin()`,
which under a near-constant query stream is a deterministic single-slot fixed
point: V3-EXQ-436e and V3-EXQ-436f both recorded n_occupied_slots = 1 of 16 in
BOTH arms on 3/5 seeds despite thousands of write() calls per arm, and the
resulting 1-slot bank yields a well-formed null that reads as a genuine
"sleep has no effect" finding.

TWO mechanisms now address it, and this file covers the SECOND one:

  * `contextmemory_write_usage_balancing` (ree-v3 76cbf844) -- a conscience-bias
    SCORE adjustment. Its own contracts are in
    test_contextmemory_write_usage_balancing.py and are NOT duplicated here.
  * `contextmemory_write_selection="refractory"` (this file) -- an ELIGIBILITY
    rule: the k most-recently-written slots are ineligible.

They are orthogonal and compose; all four combinations are legal.

These tests are time-independent and construct ContextMemory directly, so they
do not depend on the environment, the runner, or any queue state.

DESIGN NOTE FOR ANYONE RELAXING THESE. Roughly half the assertions are NEGATIVE
CONTROLS: that the default is bit-identical, that the state_dict is unchanged,
that bookkeeping does not perturb the legacy path, that the landed conscience
bias is not altered by this mode's arrival, and above all
test_landed_usage_balancing_is_a_fixed_cycle_and_refractory_is_not.

That last one is the load-bearing one, and it replaces two tests from this
file's pre-landing draft that pinned `gumbel`/`usage` modes which were measured
content-blind and deliberately NOT landed (see the reconciliation table in
REE_assembly/evidence/planning/contextmemory_refractory_mode_dataflow_plan_20260819.md).
Its job is the same: stop a later session promoting a mode on its occupancy
number alone. Every mechanism here clears the registered n_occupied floor, so
occupancy cannot discriminate them.

ASSERTION POLICY -- READ BEFORE ADDING A TEST. Every numeric assertion below is
on a DETERMINISTIC quantity (occupancy count, self-repeat, round-robin
agreement, slot identity). There is deliberately NO assertion on the
occupied-slot cosine similarity. The independently pre-registered probe
(REE_assembly/evidence/planning/contextmemory_write_selection_comparison_20260819.md;
pre-registration REE_assembly fcfb311e4b, results b7e072ddf0) established that
that column cannot discriminate these arms at 5 seeds -- every contrast
|dz| <= 0.47, |t(4)| <= 1.04, sign-inconsistent across seeds, INCLUDING the
+0.6060 -> +0.5919 refractory-over-legacy gap (dz = -0.06, t(4) = -0.13) that
the pre-landing draft of this file cited as the recommendation. A test pinning
a quantity with that spread at n=5 is a flake generator wearing a contract's
clothes. Required n at 80% power is 38-2485 depending on the contrast.
"""

import math

import torch
import pytest

from ree_core.predictors.e1_deep import ContextMemory

LATENT_DIM, MEMORY_DIM, NUM_SLOTS = 64, 128, 16


def _stream(seed, n, jitter=0.0078, clusters=1):
    """Query stream at the measured operating point (state rms ~0.078).

    clusters=1 reproduces the near-constant stream that triggers the lock;
    clusters=2 gives a varied stream on which context-conditioning is testable.
    """
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


# Seeds 0 and 100 lock under the legacy rule at this operating point; 7/13/42
# rotate. Keeping both groups is what makes the fix tests meaningful rather
# than vacuous -- a mode that only ever rotates would pass on 7/13/42 alone.
LOCKING_SEEDS = (0, 100)
ROTATING_SEEDS = (7, 13, 42)
ALL_SEEDS = LOCKING_SEEDS + ROTATING_SEEDS

REFRACTORY = {"write_selection": "refractory", "write_refractory_k": 2}
USAGE_BALANCING = {"write_usage_balancing": True}


# --------------------------------------------------------------------------
# The defect itself. If this stops failing, the operating point has drifted and
# every other test here is measuring nothing.
# --------------------------------------------------------------------------

def test_legacy_argmin_locks_to_a_single_slot():
    """NEGATIVE CONTROL / defect pin: the legacy rule really does lock."""
    for seed in LOCKING_SEEDS:
        cm, _, _ = _run(seed)
        assert len(cm.occupied_slots()) == 1, (
            f"seed {seed}: expected the documented single-slot fixed point, "
            f"got {len(cm.occupied_slots())} occupied slots. If this changed, the "
            "probe operating point has drifted -- re-derive before trusting the "
            "fix tests below."
        )


def test_sign_discriminator_predicts_lock_versus_rotate():
    """V3-EXQ-436e's closed-form discriminator q . (write_signal - memory[argmin]):
    negative predicts LOCK, positive predicts ROTATE. This is the mechanism, and
    it is what says the defect is addressing-space misalignment rather than bad
    luck in the initialisation."""
    for seed in ALL_SEEDS:
        torch.manual_seed(seed)
        cm = ContextMemory(LATENT_DIM, MEMORY_DIM, NUM_SLOTS, gated_content_write=True)
        _, state = _stream(seed, 1)[0]
        with torch.no_grad():
            query = cm.query_proj(state)
            idx = int(torch.mm(query, cm.memory.t()).mean(0).argmin().item())
            signal = cm.write_gate(state) * cm.write_content(state)
            disc = float((query.mean(0) * (signal.mean(0) - cm.memory.data[idx])).sum())
        observed_lock = len(_run(seed)[0].occupied_slots()) == 1
        assert (disc < 0) == observed_lock, (
            f"seed {seed}: discriminator {disc:+.5f} predicts "
            f"{'LOCK' if disc < 0 else 'ROTATE'}, observed lock={observed_lock}"
        )


# --------------------------------------------------------------------------
# Backward compatibility. The default must keep the bug, bit-identically.
# --------------------------------------------------------------------------

def test_default_selection_is_argmin():
    cm = ContextMemory(LATENT_DIM, MEMORY_DIM, NUM_SLOTS)
    assert cm.write_selection == "argmin"
    assert cm.write_refractory_k == 2  # inert unless the mode is "refractory"


def test_default_is_bit_identical_to_the_legacy_expression():
    """The default path must reproduce the pre-change code exactly -- same slot
    sequence AND same final memory tensor. Bookkeeping must not perturb it."""
    for seed in ALL_SEEDS:
        torch.manual_seed(seed)
        cm = ContextMemory(LATENT_DIM, MEMORY_DIM, NUM_SLOTS, gated_content_write=True)
        torch.manual_seed(seed)
        ref = ContextMemory(LATENT_DIM, MEMORY_DIM, NUM_SLOTS, gated_content_write=True)

        for _, state in _stream(seed, 200):
            with torch.no_grad():  # legacy expression, verbatim
                legacy_idx = int(
                    torch.mm(ref.query_proj(state), ref.memory.t()).mean(0).argmin().item()
                )
                signal = ref.write_gate(state) * ref.write_content(state)
                ref.memory.data[legacy_idx] = (
                    0.9 * ref.memory.data[legacy_idx] + 0.1 * signal.mean(0)
                )
            cm.write(state)
            assert cm.last_write_index == legacy_idx, f"seed {seed}: slot diverged"

        assert torch.equal(cm.memory.data, ref.memory.data), (
            f"seed {seed}: final memory differs from the legacy path"
        )


def test_refractory_consumes_no_rng():
    """NEGATIVE CONTROL. `refractory` is fully deterministic -- no sampling. So
    enabling it must not advance the global RNG stream, or every downstream
    seeded trajectory in an experiment would shift for reasons unrelated to the
    mode. (This is a substantive difference from the `gumbel` mode that was
    measured and deliberately not landed.)"""
    for kwargs in ({}, REFRACTORY):
        torch.manual_seed(0)
        cm = ContextMemory(LATENT_DIM, MEMORY_DIM, NUM_SLOTS,
                           gated_content_write=True, **kwargs)
        for _, state in _stream(0, 50):
            cm.write(state)
        after = torch.rand(3)
        torch.manual_seed(0)
        ContextMemory(LATENT_DIM, MEMORY_DIM, NUM_SLOTS,
                      gated_content_write=True, **kwargs)
        expected = torch.rand(3)
        assert torch.equal(after, expected), (
            f"write() consumed RNG with kwargs={kwargs}"
        )


def test_state_dict_is_unchanged_by_the_new_buffer():
    """slot_write_counts is persistent=False, so existing checkpoints load
    untouched."""
    keys = set(ContextMemory(LATENT_DIM, MEMORY_DIM, NUM_SLOTS).state_dict().keys())
    assert "slot_write_counts" not in keys


def test_new_buffer_does_not_disturb_the_landed_usage_ema_control():
    """NEGATIVE CONTROL for the LANDED contract. slot_write_counts IS always
    constructed, so named_buffers() gains a name. The landed file's
    test_off_constructs_no_extra_buffer is scoped to names starting
    'write_usage_ema'; this pins that the scoping still holds, so the two
    contracts cannot silently start contradicting each other."""
    cm = ContextMemory(LATENT_DIM, MEMORY_DIM, NUM_SLOTS)
    names = {n for n, _ in cm.named_buffers()}
    assert not any(n.startswith("write_usage_ema") for n in names), names
    assert "slot_write_counts" in names


# --------------------------------------------------------------------------
# The fix.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("k", [1, 2, 4])
def test_refractory_structurally_guarantees_occupancy(k):
    """The k most-recently-written slots are ineligible, so occupancy is
    >= k+1 by construction -- not by luck, and not dependent on the stream.
    This structural guarantee is the whole case for the mode; the conscience
    bias reaches full occupancy empirically, on the streams measured."""
    for seed in ALL_SEEDS:
        cm, _, _ = _run(seed, write_selection="refractory", write_refractory_k=k)
        assert len(cm.occupied_slots()) >= k + 1, (
            f"seed {seed}, k={k}: occupancy {len(cm.occupied_slots())} < k+1"
        )


def test_refractory_clears_the_registered_failure_floor():
    """V3-EXQ-436f's registered target: >= 2 occupied slots on >= 3/5 seeds.
    Legacy manages 3/5 only because 3 seeds happen not to lock."""
    passing = sum(
        1 for seed in ALL_SEEDS
        if len(_run(seed, **REFRACTORY)[0].occupied_slots()) >= 2
    )
    assert passing == len(ALL_SEEDS), f"only {passing}/{len(ALL_SEEDS)} seeds cleared the floor"


@pytest.mark.parametrize("kwargs", [REFRACTORY, USAGE_BALANCING, {**REFRACTORY, **USAGE_BALANCING}],
                         ids=["refractory", "usage_balancing", "both"])
def test_every_available_mechanism_breaks_the_lock(kwargs):
    """Both mechanisms, and their composition, clear the lock. This is exactly
    why occupancy CANNOT be used to choose between them -- see
    test_landed_usage_balancing_is_a_fixed_cycle_and_refractory_is_not."""
    for seed in LOCKING_SEEDS:
        cm, _, _ = _run(seed, **kwargs)
        assert len(cm.occupied_slots()) >= 2, f"{kwargs} still locked on seed {seed}"


def test_refractory_never_writes_into_the_refractory_window():
    """The invariant the structural guarantee rests on."""
    k = 3
    torch.manual_seed(0)
    cm = ContextMemory(LATENT_DIM, MEMORY_DIM, NUM_SLOTS, gated_content_write=True,
                       write_selection="refractory", write_refractory_k=k)
    history = []
    for _, state in _stream(0, 300):
        cm.write(state)
        assert cm.last_write_index not in history[-k:], (
            f"slot {cm.last_write_index} rewritten inside the k={k} window {history[-k:]}"
        )
        history.append(cm.last_write_index)


def test_refractory_k_cannot_mask_every_slot():
    """k >= num_slots must still leave a writable slot rather than deadlock."""
    cm, _, _ = _run(0, n=200, write_selection="refractory",
                    write_refractory_k=NUM_SLOTS + 5)
    assert len(cm.occupied_slots()) >= 2


def test_refractory_k_zero_is_the_legacy_path():
    """k=0 disables the mask, so the mode degrades to argmin rather than
    silently doing something else."""
    for seed in LOCKING_SEEDS:
        a, _, seq_a = _run(seed, n=200, write_selection="refractory", write_refractory_k=0)
        b, _, seq_b = _run(seed, n=200)
        assert seq_a == seq_b
        assert torch.equal(a.memory.data, b.memory.data)


# --------------------------------------------------------------------------
# Content-conditioning. The load-bearing negative controls.
# --------------------------------------------------------------------------

def _jaccard(per_cluster):
    a, b = per_cluster.get(0, set()), per_cluster.get(1, set())
    return len(a & b) / max(len(a | b), 1)


def _round_robin_agreement(sequence):
    """Fraction of writes whose slot is the strict least-recently-used choice.

    This is the metric the probe's section 6 says should have been
    pre-registered instead of Jaccard: a period-16 LRU cycle aliases against a
    2-cluster alternation, giving Jaccard exactly 0.0 on 3/5 seeds and exactly
    1.0 on 2/5 -- a bimodal artifact whose mean looks moderate. The round-robin
    index has no such failure mode and is exactly reproducible.
    """
    last_seen, hits, total = {}, 0, 0
    for t, idx in enumerate(sequence):
        if t > 0:
            lru = min(range(NUM_SLOTS), key=lambda i: last_seen.get(i, -1))
            total += 1
            hits += (idx == lru)
        last_seen[idx] = t
    return hits / max(total, 1)


def test_landed_usage_balancing_is_a_fixed_cycle_and_refractory_is_not():
    """LOAD-BEARING NEGATIVE CONTROL.

    Both mechanisms clear the occupancy floor on every seed. They are NOT
    equivalent, and the distinction is not visible in the occupancy number.

    With the conscience bias on, the sqrt(memory_dim) = 11.31 scaling puts the
    usage term 2-3 orders of magnitude above the ~0.026 across-slot spread of
    mean_scores, so after the first pass the write address is a function of the
    write COUNTER rather than of the query: a strict LRU cycle on ~99% of
    writes. That is real occupancy and a real improvement on the defect -- it is
    NOT globally content-blind, since the cycle's ORDER is content-determined
    once -- but it is occupancy without addressing.

    `refractory` masks only the k most recent slots and leaves selection among
    the rest as the unmodified content argmin, so it scores 0.000 on the same
    index.

    This test exists so that a later session cannot promote either mechanism as
    "the" fix on the strength of its occupancy number without first contradicting
    a red test. If it starts failing, the selector changed: re-measure before
    changing any recommendation.
    """
    legacy = [_round_robin_agreement(_run(s)[2]) for s in ALL_SEEDS]
    landed = [_round_robin_agreement(_run(s, **USAGE_BALANCING)[2]) for s in ALL_SEEDS]
    refr = [_round_robin_agreement(_run(s, **REFRACTORY)[2]) for s in ALL_SEEDS]

    assert min(landed) > 0.90, (
        f"conscience bias round-robin agreement {landed} -- expected ~0.99 on every "
        "seed. Re-measure the bias scaling before recommending this mode."
    )
    assert max(refr) < 0.10, f"refractory round-robin agreement {refr} -- expected ~0.0"
    assert max(legacy) < 0.10, f"legacy round-robin agreement {legacy} -- expected ~0.0"


def test_refractory_occupancy_is_content_determined_and_the_bias_is_not():
    """The same distinction from the other side, and the sharper form of it.

    The conscience bias visits ALL 16 slots on EVERY seed -- the count carries
    no information about the content stream. `refractory`'s slot count varies
    with the stream and tracks the legacy path's exactly on the seeds where
    legacy does not lock, because on those seeds the mask almost never binds.
    """
    legacy = [len(_run(s)[0].occupied_slots()) for s in ALL_SEEDS]
    landed = [len(_run(s, **USAGE_BALANCING)[0].occupied_slots()) for s in ALL_SEEDS]
    refr = [len(_run(s, **REFRACTORY)[0].occupied_slots()) for s in ALL_SEEDS]

    assert landed == [NUM_SLOTS] * len(ALL_SEEDS), (
        f"conscience bias occupancy {landed} -- expected all {NUM_SLOTS}"
    )
    assert len(set(refr)) > 1, (
        f"refractory occupancy {refr} is constant across seeds -- it has stopped "
        "tracking content and has become a counter-driven cycle"
    )
    # On the rotating seeds the mask barely binds, so refractory stays close to
    # legacy rather than saturating.
    for seed, leg, ref in zip(ALL_SEEDS, legacy, refr):
        if seed in ROTATING_SEEDS:
            assert abs(ref - leg) <= 2, (
                f"seed {seed}: refractory occupancy {ref} vs legacy {leg} -- the "
                "mask should barely bind on a non-locking seed"
            )


def test_refractory_preserves_content_conditioning():
    """Two distinct contexts must still map to distinguishable slot sets.
    Refractory only removes the k most recent slots from an otherwise unmodified
    content argmin, so conditioning stays at the legacy level.

    TWO NOTES ON THE INSTRUMENT, both learned the hard way.

    (1) Jaccard is used here ONLY for the refractory-vs-legacy contrast. It must
    NOT be used to assess the conscience-bias arm -- a period-16 cycle aliases
    against a 2-cluster alternation and Jaccard reports a meaningless bimodal
    0.0/1.0 whose mean looks moderate. Use _round_robin_agreement for that.

    (2) The margin assertion is on the MEAN across seeds, not per seed, and this
    is deliberate. The pre-landing draft of this file asserted the 0.25 margin
    per seed; measured, the per-seed delta is [0.000, 0.000, -0.137, +0.235,
    +0.077] at n=3000 and [0.000, 0.000, -0.077, +0.255, +0.077] at n=1500 --
    sign-inconsistent, and seed 13 STRADDLES the 0.25 margin depending only on
    the write count, so that assertion was never stable at this helper's own
    default n. The mean, by contrast, reproduces the probe to three decimals
    (legacy 0.329, refractory 0.364 at n=3000). This is the probe's own section-6
    finding showing up inside the contract: these per-seed differentiation
    numbers do not support per-seed thresholds. Pinning the stable statistic is
    a correction, not a relaxation -- the strong per-seed claim (not content-
    blind) is retained below and passes with a wide margin on every seed.
    """
    legacy_j, refr_j = [], []
    for seed in ALL_SEEDS:
        _, legacy, _ = _run(seed, clusters=2)
        _, refr, _ = _run(seed, clusters=2, **REFRACTORY)
        legacy_j.append(_jaccard(legacy))
        refr_j.append(_jaccard(refr))
        assert refr_j[-1] < 0.95, (
            f"seed {seed}: refractory Jaccard {refr_j[-1]:.3f} -- contexts no "
            "longer separable, addressing has gone content-blind"
        )
    mean_legacy = sum(legacy_j) / len(legacy_j)
    mean_refr = sum(refr_j) / len(refr_j)
    assert mean_refr <= mean_legacy + 0.25, (
        f"refractory mean Jaccard {mean_refr:.3f} materially worse than legacy "
        f"{mean_legacy:.3f} (per-seed refractory {[round(x, 3) for x in refr_j]}, "
        f"legacy {[round(x, 3) for x in legacy_j]})"
    )


# --------------------------------------------------------------------------
# Composition with the landed conscience bias.
# --------------------------------------------------------------------------

def test_the_two_mechanisms_compose_without_error():
    """All four combinations are legal and produce a well-formed bank. The
    composition is deliberate rather than a missing mutual exclusion: the bias
    decides how good each slot looks, the mask decides which slots may be
    looked at."""
    for ub in (False, True):
        for sel in ("argmin", "refractory"):
            cm, _, _ = _run(0, n=200, write_usage_balancing=ub, write_selection=sel)
            assert float(cm.slot_write_counts.sum()) == 200.0
            assert len(cm.occupied_slots()) >= 1


def test_the_conscience_bias_subsumes_the_refractory_mask_at_default_weight():
    """MEASURED, and recorded so nobody proposes the combination as a third arm
    without knowing this. At the default bias weight the usage term dominates
    content by 2-3 orders of magnitude, so the composed arm is IDENTICAL to the
    bias alone -- the mask never binds, because the LRU choice is already
    outside the last-k window. Enabling both is legal but buys nothing.

    If this ever fails, the relative scaling of the two terms has changed and
    the composed arm has become a genuinely distinct mechanism -- which would
    need its own measurement before use.
    """
    for seed in ALL_SEEDS:
        _, _, bias_only = _run(seed, n=300, **USAGE_BALANCING)
        _, _, composed = _run(seed, n=300, **{**REFRACTORY, **USAGE_BALANCING})
        assert bias_only == composed, (
            f"seed {seed}: the refractory mask now changes the conscience-bias "
            "sequence. Re-measure before treating the combination as an arm."
        )


# --------------------------------------------------------------------------
# Instrumentation. The reason V3-EXQ-436f's tracker was fragile.
# --------------------------------------------------------------------------

def test_write_records_the_slot_it_actually_wrote():
    """V3-EXQ-436f's occupancy tracker re-derived the index by duplicating
    write()'s own argmin expression, which silently reports the WRONG slot once
    the selection rule changes. Instrumentation must read these instead."""
    torch.manual_seed(0)
    cm = ContextMemory(LATENT_DIM, MEMORY_DIM, NUM_SLOTS, gated_content_write=True,
                       **REFRACTORY)
    for _, state in _stream(0, 50):
        before = cm.memory.data.clone()
        cm.write(state)
        changed = [i for i in range(NUM_SLOTS)
                   if not torch.equal(before[i], cm.memory.data[i])]
        assert changed == [cm.last_write_index], (
            f"last_write_index={cm.last_write_index} but slots {changed} changed"
        )


def test_last_write_index_is_none_before_any_write():
    assert ContextMemory(LATENT_DIM, MEMORY_DIM, NUM_SLOTS).last_write_index is None


@pytest.mark.parametrize("kwargs", [{}, REFRACTORY, USAGE_BALANCING],
                         ids=["argmin", "refractory", "usage_balancing"])
def test_write_counts_track_occupancy_in_every_mode(kwargs):
    cm, _, _ = _run(0, n=300, **kwargs)
    assert float(cm.slot_write_counts.sum()) == 300.0, "lost writes"
    assert cm.occupied_slots() == [
        i for i in range(NUM_SLOTS) if float(cm.slot_write_counts[i]) > 0
    ]


def test_stale_reimplementation_of_the_old_rule_disagrees():
    """Pins the desync hazard explicitly: re-deriving the legacy argmin no
    longer identifies the written slot once selection changes."""
    torch.manual_seed(0)
    cm = ContextMemory(LATENT_DIM, MEMORY_DIM, NUM_SLOTS, gated_content_write=True,
                       **REFRACTORY)
    disagreements = 0
    for _, state in _stream(0, 200):
        with torch.no_grad():
            stale = int(torch.mm(cm.query_proj(state), cm.memory.t()).mean(0).argmin().item())
        cm.write(state)
        disagreements += (stale != cm.last_write_index)
    assert disagreements > 0, (
        "the stale re-derivation agreed on every write -- this test is vacuous "
        "and the desync hazard it documents is unpinned"
    )


# --------------------------------------------------------------------------
# Config validation.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("bad", ["bogus", "usage", "gumbel", "Argmin", ""])
def test_invalid_selection_mode_is_rejected(bad):
    """Fail closed. Note `usage` and `gumbel` are rejected BY NAME: they were
    implemented on the salvaged branch, measured content-blind, and deliberately
    not landed. A config carrying them over must error, not silently fall back
    to the defect."""
    with pytest.raises(ValueError, match="contextmemory_write_selection"):
        ContextMemory(LATENT_DIM, MEMORY_DIM, NUM_SLOTS, write_selection=bad)


def test_config_plumbs_through_from_dims():
    """The documented 3-site hazard: from_dims silently swallows unknown kwargs,
    so a knob wired at only 2 of the 3 sites fails open and silently."""
    from ree_core.utils.config import REEConfig
    cfg = REEConfig.from_dims(
        body_obs_dim=8, world_obs_dim=8, action_dim=5, self_dim=32, world_dim=32,
        contextmemory_write_selection="refractory", contextmemory_write_refractory_k=3,
    )
    assert cfg.e1.contextmemory_write_selection == "refractory"
    assert cfg.e1.contextmemory_write_refractory_k == 3

    default = REEConfig.from_dims(body_obs_dim=8, world_obs_dim=8, action_dim=5,
                                  self_dim=32, world_dim=32)
    assert default.e1.contextmemory_write_selection == "argmin"
    assert default.e1.contextmemory_write_refractory_k == 2


def test_e1_predictor_wires_the_selection_through():
    from ree_core.predictors.e1_deep import E1DeepPredictor
    from ree_core.utils.config import REEConfig
    cfg = REEConfig.from_dims(
        body_obs_dim=8, world_obs_dim=8, action_dim=5, self_dim=32, world_dim=32,
        contextmemory_write_selection="refractory", contextmemory_write_refractory_k=3,
    )
    e1 = E1DeepPredictor(cfg.e1)
    assert e1.context_memory.write_selection == "refractory"
    assert e1.context_memory.write_refractory_k == 3


def test_the_landed_usage_balancing_knobs_are_untouched():
    """NEGATIVE CONTROL. This landing must not have altered the conscience-bias
    surface it sits alongside -- the user's disposition was to KEEP it."""
    from ree_core.utils.config import REEConfig
    d = REEConfig.from_dims(body_obs_dim=8, world_obs_dim=8, action_dim=5,
                            self_dim=32, world_dim=32).e1
    assert d.contextmemory_write_usage_balancing is False
    assert d.contextmemory_write_usage_bias_weight == 1.0
    assert d.contextmemory_write_usage_decay == 0.99
