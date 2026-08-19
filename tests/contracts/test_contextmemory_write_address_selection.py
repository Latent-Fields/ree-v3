"""Contracts for ContextMemory write-address selection.

substrate_queue `contextmemory-write-path-addressing-degeneracy` (severity
`corrupting`). The legacy write address is a hard `scores.mean(0).argmin()`,
which under a near-constant query stream is a deterministic single-slot fixed
point: V3-EXQ-436e and V3-EXQ-436f both recorded n_occupied_slots = 1 of 16 in
BOTH arms on 3/5 seeds despite thousands of write() calls per arm, and the
resulting 1-slot bank yields a well-formed null that reads as a genuine
"sleep has no effect" finding.

These tests are time-independent and construct ContextMemory directly, so they
do not depend on the environment, the runner, or any queue state.

DESIGN NOTE FOR ANYONE RELAXING THESE: roughly half of the assertions below are
NEGATIVE CONTROLS -- that the default is bit-identical, that the state_dict is
unchanged, that bookkeeping does not perturb the legacy path, and above all that
`gumbel`/`usage` are NOT content-conditioned. That last group is the load-bearing
one. Every non-legacy mode satisfies the registered n_occupied floor, so a future
session reading occupancy alone would reasonably conclude any of them is a fix.
The measurement says otherwise: `gumbel`/`usage` reach occupancy by making
selection content-blind, which makes the differentiation DV worse than the
defect they replace. test_gumbel_is_not_content_conditioned pins that so the
recommendation cannot quietly invert.
"""

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
    for cid, state in _stream(seed, n, clusters=clusters):
        cm.write(state)
        per_cluster.setdefault(cid, set()).add(cm.last_write_index)
    return cm, per_cluster


# Seeds 0 and 100 lock under the legacy rule at this operating point; 7/13/42
# rotate. Keeping both groups is what makes the fix tests meaningful rather
# than vacuous -- a mode that only ever rotates would pass on 7/13/42 alone.
LOCKING_SEEDS = (0, 100)
ROTATING_SEEDS = (7, 13, 42)
ALL_SEEDS = LOCKING_SEEDS + ROTATING_SEEDS


# --------------------------------------------------------------------------
# The defect itself. If this stops failing, the operating point has drifted and
# every other test here is measuring nothing.
# --------------------------------------------------------------------------

def test_legacy_argmin_locks_to_a_single_slot():
    """NEGATIVE CONTROL / defect pin: the legacy rule really does lock."""
    for seed in LOCKING_SEEDS:
        cm, _ = _run(seed)
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


def test_state_dict_is_unchanged_by_the_new_buffers():
    """slot_usage / slot_write_counts are persistent=False, so existing
    checkpoints load untouched."""
    keys = set(ContextMemory(LATENT_DIM, MEMORY_DIM, NUM_SLOTS).state_dict().keys())
    assert "slot_usage" not in keys
    assert "slot_write_counts" not in keys


# --------------------------------------------------------------------------
# The fix.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("k", [1, 2, 4])
def test_refractory_structurally_guarantees_occupancy(k):
    """The k most-recently-written slots are ineligible, so occupancy is
    >= k+1 by construction -- not by luck, and not dependent on the stream."""
    for seed in ALL_SEEDS:
        cm, _ = _run(seed, write_selection="refractory", write_refractory_k=k)
        assert len(cm.occupied_slots()) >= k + 1, (
            f"seed {seed}, k={k}: occupancy {len(cm.occupied_slots())} < k+1"
        )


def test_refractory_clears_the_registered_failure_floor():
    """V3-EXQ-436f's registered target: >= 2 occupied slots on >= 3/5 seeds.
    Legacy manages 3/5 only because 3 seeds happen not to lock."""
    passing = sum(
        1 for seed in ALL_SEEDS
        if len(_run(seed, write_selection="refractory", write_refractory_k=2)[0].occupied_slots()) >= 2
    )
    assert passing == len(ALL_SEEDS), f"only {passing}/{len(ALL_SEEDS)} seeds cleared the floor"


@pytest.mark.parametrize("mode", ["refractory", "usage", "gumbel"])
def test_every_non_legacy_mode_breaks_the_lock(mode):
    for seed in LOCKING_SEEDS:
        cm, _ = _run(seed, write_selection=mode)
        assert len(cm.occupied_slots()) >= 2, f"{mode} still locked on seed {seed}"


def test_refractory_never_writes_into_the_refractory_window():
    """The invariant the guarantee rests on."""
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
    cm, _ = _run(0, n=200, write_selection="refractory", write_refractory_k=NUM_SLOTS + 5)
    assert len(cm.occupied_slots()) >= 2


# --------------------------------------------------------------------------
# Content-conditioning. The load-bearing negative controls.
# --------------------------------------------------------------------------

def _jaccard(per_cluster):
    a, b = per_cluster.get(0, set()), per_cluster.get(1, set())
    return len(a & b) / max(len(a | b), 1)


def test_refractory_preserves_content_conditioning():
    """Two distinct contexts must still map to distinguishable slot sets.
    Refractory only removes the k most recent slots from an otherwise unmodified
    content argmin, so conditioning stays at the legacy level."""
    for seed in ALL_SEEDS:
        _, legacy = _run(seed, clusters=2)
        _, refr = _run(seed, clusters=2, write_selection="refractory", write_refractory_k=2)
        assert _jaccard(refr) < 0.95, (
            f"seed {seed}: refractory Jaccard {_jaccard(refr):.3f} -- contexts no "
            "longer separable, addressing has gone content-blind"
        )
        assert _jaccard(refr) <= _jaccard(legacy) + 0.25, (
            f"seed {seed}: refractory Jaccard {_jaccard(refr):.3f} materially worse "
            f"than legacy {_jaccard(legacy):.3f}"
        )


def test_gumbel_is_not_content_conditioned():
    """LOAD-BEARING NEGATIVE CONTROL. `gumbel` clears the occupancy floor on
    every seed while writing both contexts to the SAME slots -- occupancy bought
    by noise, not by context. This test exists so that a later session cannot
    promote `gumbel` to the recommended mode on the strength of its occupancy
    number without first contradicting a red test. If this ever starts failing,
    the selector changed: re-measure the differentiation DV before changing any
    recommendation."""
    overlaps = [_jaccard(_run(seed, clusters=2, write_selection="gumbel")[1])
                for seed in ALL_SEEDS]
    mean_overlap = sum(overlaps) / len(overlaps)
    assert mean_overlap > 0.8, (
        f"gumbel mean Jaccard {mean_overlap:.3f} -- expected near-total overlap "
        "(content-blind). Re-measure before recommending this mode."
    )


def test_recommended_mode_beats_noise_modes_on_context_conditioning():
    """The comparison that should govern mode choice, pinned as a test."""
    refr = [_jaccard(_run(s, clusters=2, write_selection="refractory",
                          write_refractory_k=2)[1]) for s in ALL_SEEDS]
    gumb = [_jaccard(_run(s, clusters=2, write_selection="gumbel")[1]) for s in ALL_SEEDS]
    assert sum(refr) / len(refr) < sum(gumb) / len(gumb), (
        "refractory must retain more context-conditioning than gumbel"
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
                       write_selection="refractory", write_refractory_k=2)
    for _, state in _stream(0, 50):
        before = cm.memory.data.clone()
        cm.write(state)
        changed = [i for i in range(NUM_SLOTS)
                   if not torch.equal(before[i], cm.memory.data[i])]
        assert changed == [cm.last_write_index], (
            f"last_write_index={cm.last_write_index} but slots {changed} changed"
        )


def test_write_counts_track_occupancy_in_every_mode():
    for mode in ("argmin", "refractory", "usage", "gumbel"):
        cm, _ = _run(0, n=300, write_selection=mode)
        assert float(cm.slot_write_counts.sum()) == 300.0, f"{mode}: lost writes"
        assert cm.occupied_slots() == [
            i for i in range(NUM_SLOTS) if float(cm.slot_write_counts[i]) > 0
        ]


def test_stale_reimplementation_of_the_old_rule_disagrees():
    """Pins the desync hazard explicitly: re-deriving the legacy argmin no
    longer identifies the written slot once selection changes."""
    torch.manual_seed(0)
    cm = ContextMemory(LATENT_DIM, MEMORY_DIM, NUM_SLOTS, gated_content_write=True,
                       write_selection="refractory", write_refractory_k=2)
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

def test_invalid_selection_mode_is_rejected():
    with pytest.raises(ValueError, match="contextmemory_write_selection"):
        ContextMemory(LATENT_DIM, MEMORY_DIM, NUM_SLOTS, write_selection="bogus")


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
