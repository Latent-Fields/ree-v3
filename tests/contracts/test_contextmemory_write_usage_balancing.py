"""Contract: ContextMemory write-address usage balancing
(substrate_queue contextmemory-write-path-addressing-degeneracy,
V3-EXQ-436e/436f follow-up).

ContextMemory.write() (ree_core/predictors/e1_deep.py) addresses by a hard
`scores.mean(0).argmin()`, which under a near-constant query stream is a
deterministic single-slot fixed point -- established by
failure_autopsy_V3-EXQ-436e_2026-08-13.md via a closed-form sign
discriminator `q . (write_signal - memory[argmin])` that predicted
lock-vs-rotate 5/5, and confirmed live at V3-EXQ-436f (n_occupied_slots = 1
of 16 in BOTH arms on 3/5 seeds despite 2,837-4,903 write() calls per arm,
with the full SD-016 production combination armed and engaged). The
READ-path fix (SD-016 cue_slot_tagger + gumbel selection) does not touch
this: write() runs entirely under torch.no_grad() and
compute_diversification_loss() only ever trains self.memory, never the
write-address selection itself.

Pins the two properties this repair establishes:

  1. OFF-inert -- write_usage_balancing=False is bit-identical to the legacy
     argmin path and constructs no extra buffer, so no in-flight experiment
     changes semantics.
  2. ON-diversifying -- write_usage_balancing=True breaks the deterministic
     lock: replaying a perfectly constant query stream that locks the
     legacy path to a single slot yields >= 2 occupied slots under the fix,
     on every seed observed to lock under the legacy path -- matching the
     substrate_queue entry's own registered acceptance target (">= 2
     occupied slots in both arms on >= 3/5 seeds").

Time-independent; CPU-only; fixed seeds.
"""
import pytest
import torch

from ree_core.predictors.e1_deep import ContextMemory

LATENT, MEM, SLOTS = 64, 128, 16
N_WRITES = 3000  # matches the failure_autopsy's own degenerate-query probe

# Seeds confirmed (by direct replay against this module's own RNG stream) to
# lock the LEGACY path to a single slot under a perfectly constant query
# stream (jitter=0.0) -- the worst case in the failure_autopsy's jitter
# sweep. Seed choice is incidental (per the autopsy: "the particular seeds
# that lock differ ... the regime, not the seed list, is what transfers")
# -- these are simply seeds observed to reproduce the LOCK regime, so the
# fix is tested against a genuine instance of the defect rather than an
# arbitrary seed that happens never to trigger it.
LOCKING_SEEDS = [1, 5, 12]


def _make(seed: int, usage_balancing: bool, weight: float = 1.0,
          decay: float = 0.99) -> ContextMemory:
    torch.manual_seed(seed)
    return ContextMemory(latent_dim=LATENT, memory_dim=MEM, num_slots=SLOTS,
                          write_usage_balancing=usage_balancing,
                          write_usage_bias_weight=weight, write_usage_decay=decay)


def _selection_index(cm: ContextMemory, state: torch.Tensor) -> int:
    """Recompute write()'s own selection index, read-only -- mirrors the
    instance-level wrapper V3-EXQ-436e's harness uses around the same bound
    write() call, so occupancy tracking cannot drift from what write()
    itself actually selects."""
    with torch.no_grad():
        query = cm.query_proj(state)
        mean_scores = torch.mm(query, cm.memory.t()).mean(0)
        if cm.write_usage_balancing:
            bias = cm.write_usage_bias_weight * cm.write_usage_ema * (cm.memory_dim ** 0.5)
            selection_scores = mean_scores + bias
        else:
            selection_scores = mean_scores
        return int(selection_scores.argmin())


def _replay_occupancy(cm: ContextMemory, seed: int, n_writes: int = N_WRITES) -> int:
    """Drive a perfectly constant query stream (the failure_autopsy's
    H-degenerate-query regime, jitter=0.0) and return the count of distinct
    slots write() ever selected."""
    gen = torch.Generator().manual_seed(seed + 9000)
    state = torch.randn(1, LATENT, generator=gen)
    occupied = set()
    for _ in range(n_writes):
        occupied.add(_selection_index(cm, state))
        cm.write(state)
    return len(occupied)


def test_off_constructs_no_extra_buffer():
    cm = _make(0, usage_balancing=False)
    assert cm.write_usage_balancing is False
    assert cm.write_usage_ema is None
    names = {n for n, _ in cm.named_buffers()}
    assert not any(n.startswith("write_usage_ema") for n in names), names


def test_on_constructs_the_usage_buffer():
    cm = _make(0, usage_balancing=True)
    assert cm.write_usage_balancing is True
    assert isinstance(cm.write_usage_ema, torch.Tensor)
    assert cm.write_usage_ema.shape == (SLOTS,)
    assert torch.equal(cm.write_usage_ema, torch.zeros(SLOTS))


def test_off_is_bit_identical_to_the_legacy_write():
    """The legacy path is `min_idx = scores.mean(0).argmin()`."""
    cm = _make(1, usage_balancing=False)
    state = torch.randn(1, LATENT, generator=torch.Generator().manual_seed(3))

    with torch.no_grad():
        expected_signal = cm.write_gate(state).mean(0)
        query = cm.query_proj(state)
        idx = int((query @ cm.memory.t()).mean(0).argmin())
        expected_slot = 0.9 * cm.memory.data[idx] + 0.1 * expected_signal

    cm.write(state)
    assert torch.equal(cm.memory.data[idx], expected_slot)


def test_off_default_matches_explicit_off():
    """Omitting the kwarg must give the legacy path, not the new one."""
    default = _make(4, usage_balancing=False)
    torch.manual_seed(4)
    implicit = ContextMemory(latent_dim=LATENT, memory_dim=MEM, num_slots=SLOTS)
    assert implicit.write_usage_balancing is False
    assert implicit.write_usage_ema is None
    assert torch.equal(implicit.memory.data, default.memory.data)


@pytest.mark.parametrize("seed", LOCKING_SEEDS)
def test_off_reproduces_the_deterministic_lock(seed):
    """Regression pin on the DEFECT itself: the legacy argmin path locks
    onto a single slot under a perfectly constant query stream -- what
    V3-EXQ-436f measured (n_occupied_slots = 1 of 16 despite thousands of
    write() calls) and the failure_autopsy's H-degenerate-query probe
    reproduced synthetically. If this ever fails because the lock stops
    reproducing, the ON tests below are no longer testing a real repair."""
    cm = _make(seed, usage_balancing=False)
    n_occupied = _replay_occupancy(cm, seed)
    assert n_occupied == 1, n_occupied


@pytest.mark.parametrize("seed", LOCKING_SEEDS)
def test_on_breaks_the_lock(seed):
    """The repair: the same degenerate query stream that locks the legacy
    path must NOT lock the usage-balanced path."""
    cm = _make(seed, usage_balancing=True)
    n_occupied = _replay_occupancy(cm, seed)
    assert n_occupied >= 2, n_occupied


@pytest.mark.parametrize("seed", LOCKING_SEEDS)
def test_on_strictly_improves_on_off_under_the_same_load(seed):
    """Paired comparison -- same seed, same query stream, flag the only change."""
    off = _replay_occupancy(_make(seed, usage_balancing=False), seed)
    on = _replay_occupancy(_make(seed, usage_balancing=True), seed)
    assert on > off, (on, off)


def test_usage_ema_tracks_the_winning_slot():
    """Direct unit check of the EMA bookkeeping, isolated from the argmin
    dynamics above via bias_weight=0.0 (the bias never feeds back into
    selection, so the same slot wins every write by the pure legacy
    dynamics): a slot that wins every write converges its own usage_ema
    toward 1.0, and every other slot's toward 0.0."""
    seed = LOCKING_SEEDS[0]
    cm = _make(seed, usage_balancing=True, weight=0.0)
    state = torch.randn(1, LATENT, generator=torch.Generator().manual_seed(seed + 9000))
    with torch.no_grad():
        forced_idx = int((cm.query_proj(state) @ cm.memory.t()).mean(0).argmin())
    for _ in range(2000):
        cm.write(state)
    assert cm.write_usage_ema[forced_idx].item() == pytest.approx(1.0, abs=1e-3)
    others = torch.cat([cm.write_usage_ema[:forced_idx], cm.write_usage_ema[forced_idx + 1:]])
    assert others.abs().max().item() < 1e-3


def test_config_flag_reaches_the_module():
    from ree_core.utils.config import REEConfig
    from ree_core.predictors.e1_deep import E1DeepPredictor

    for flag in (False, True):
        cfg = REEConfig.from_dims(
            body_obs_dim=8, world_obs_dim=8, action_dim=4,
            self_dim=32, world_dim=32,
            contextmemory_write_usage_balancing=flag,
        )
        assert cfg.e1.contextmemory_write_usage_balancing is flag
        e1 = E1DeepPredictor(cfg.e1)
        assert e1.context_memory.write_usage_balancing is flag
        assert (e1.context_memory.write_usage_ema is not None) is flag
