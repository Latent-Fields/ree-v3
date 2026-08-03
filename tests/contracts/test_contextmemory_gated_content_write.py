"""Contract: ContextMemory gated content write (V3-EXQ-436c / V3-EXQ-861a follow-up).

Pins the two properties the 2026-08-03 write-path audit established:

  1. OFF-inert -- gated_content_write=False is bit-identical to the legacy path
     and constructs no extra parameter, so no in-flight experiment changes
     semantics.
  2. ON-differentiating -- gated_content_write=True removes the sigmoid-midpoint
     constant that homogenizes the slot bank, and replaying 436c's SWS write load
     then DECREASES whole-bank slot cosine similarity instead of driving it to 1.

It also pins the negative result that is the whole point of the audit: removing
write_gate's BIAS (the fix the failure_autopsy proposed, by analogy to key_proj /
SD-016 Part A) does NOT repair the homogenization. Without that assertion a later
session re-deriving the autopsy's hypothesis would apply the bias fix, observe
nothing, and re-open the same investigation.

Time-independent; CPU-only; fixed seeds.
"""
import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from ree_core.predictors.e1_deep import ContextMemory

LATENT, MEM, SLOTS = 64, 128, 16
# Operating point measured live on a 436c-configured agent during run_sws_schema_pass():
# state ||x|| mean 0.6266, state rms mean 0.0783.
STATE_RMS = 0.078
N_WRITES = 800  # 436c's pooled sws_n_writes


def _pair_sim(mat: torch.Tensor) -> float:
    """Mean off-diagonal cosine similarity of the rows of mat."""
    n = mat.shape[0]
    normed = F.normalize(mat, dim=-1)
    sim = normed @ normed.t()
    return float(sim[~torch.eye(n, dtype=torch.bool, device=mat.device)].mean())


def _make(seed: int, gated: bool) -> ContextMemory:
    torch.manual_seed(seed)
    return ContextMemory(latent_dim=LATENT, memory_dim=MEM, num_slots=SLOTS,
                         gated_content_write=gated)


def _replay(cm: ContextMemory, seed: int, n_writes: int = N_WRITES) -> float:
    """Drive 436c's SWS write load (batch=1 per write) and return final slot cos sim."""
    gen = torch.Generator().manual_seed(seed + 9000)
    for _ in range(n_writes):
        state = torch.randn(1, LATENT, generator=gen) * STATE_RMS
        cm.write(state)
    return _pair_sim(cm.memory.data)


def test_off_constructs_no_extra_parameter():
    cm = _make(0, gated=False)
    assert cm.gated_content_write is False
    assert cm.write_content is None
    names = {n for n, _ in cm.named_parameters()}
    assert not any(n.startswith("write_content") for n in names), names


def test_on_constructs_the_content_projection():
    cm = _make(0, gated=True)
    assert cm.gated_content_write is True
    assert isinstance(cm.write_content, nn.Linear)
    assert cm.write_content.in_features == LATENT
    assert cm.write_content.out_features == MEM


def test_off_is_bit_identical_to_the_legacy_write():
    """The legacy path is `memory[i] = 0.9*memory[i] + 0.1*sigmoid(W x + b)`."""
    cm = _make(1, gated=False)
    state = torch.randn(1, LATENT, generator=torch.Generator().manual_seed(3)) * STATE_RMS

    with torch.no_grad():
        expected_signal = cm.write_gate(state).mean(0)
        query = cm.query_proj(state)
        idx = int((query @ cm.memory.t()).mean(0).argmin())
        expected_slot = 0.9 * cm.memory.data[idx] + 0.1 * expected_signal

    cm.write(state)
    assert torch.equal(cm.memory.data[idx], expected_slot)


def test_off_default_matches_explicit_off():
    """Omitting the kwarg must give the legacy path, not the new one."""
    default = _make(4, gated=False)
    torch.manual_seed(4)
    implicit = ContextMemory(latent_dim=LATENT, memory_dim=MEM, num_slots=SLOTS)
    assert implicit.gated_content_write is False
    assert torch.equal(implicit.memory.data, default.memory.data)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_off_homogenizes_the_bank_reproducing_436c(seed):
    """Regression pin on the DEFECT itself: the legacy path drives slot cosine
    similarity to ~1.0, which is what V3-EXQ-436c measured (~0.9999-1.0, 4/5
    seeds) and the opposite of the SD-017/ARC-045/MECH-166 prediction."""
    cm = _make(seed, gated=False)
    before = _pair_sim(cm.memory.data)
    after = _replay(cm, seed)
    assert before < 0.10, before
    assert after > 0.99, after


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_on_does_not_homogenize_the_bank(seed):
    """The repair: the same write load must NOT collapse the bank.

    Measured 0.0062 / 0.0491 / 0.0434 for these seeds (0.0154-0.0491 over seeds
    0-5). The 0.30 bound is deliberately loose relative to that -- it exists to
    separate repaired (~0.03) from defective (~0.9999), not to pin a value.
    """
    cm = _make(seed, gated=True)
    after = _replay(cm, seed)
    assert after < 0.30, after


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_on_strictly_improves_on_off_under_the_same_load(seed):
    """Paired comparison -- same seed, same write sequence, flag the only change."""
    off = _replay(_make(seed, gated=False), seed)
    on = _replay(_make(seed, gated=True), seed)
    assert on < off, (on, off)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_on_written_vectors_are_not_near_collinear(seed):
    """Root cause, measured directly: with the legacy path the vectors actually
    written are mutually cosine-similar at ~0.9995+ (they are all ~0.5*ones);
    gating restores per-input variation."""
    gen = torch.Generator().manual_seed(seed + 77)
    states = torch.randn(256, LATENT, generator=gen) * STATE_RMS

    legacy = _make(seed, gated=False)
    gated = _make(seed, gated=True)
    with torch.no_grad():
        legacy_signal = legacy.write_gate(states)
        gated_signal = gated.write_gate(states) * gated.write_content(states)

    assert _pair_sim(legacy_signal) > 0.99, _pair_sim(legacy_signal)
    assert _pair_sim(gated_signal) < 0.90, _pair_sim(gated_signal)


def test_legacy_constant_is_the_sigmoid_midpoint_not_the_bias():
    """The audit's decisive negative result.

    The failure_autopsy hypothesised that write_gate carries key_proj's
    bias-over-content collapse (SD-016 Part A / EXQ-477). The bias ratio IS
    elevated -- but the constant that homogenizes the bank is sigmoid's own
    midpoint, ||0.5*ones(MEM)||, which zeroing the bias cannot move. This test
    fails if anyone "fixes" this by removing the bias.
    """
    cm = _make(5, gated=False)
    gen = torch.Generator().manual_seed(5150)
    states = torch.randn(256, LATENT, generator=gen) * STATE_RMS

    def const_over_varying(signal):
        mean_vec = signal.mean(0)
        dev = (signal - mean_vec).norm(dim=-1).mean()
        return float(mean_vec.norm() / dev.clamp(min=1e-12)), float(mean_vec.norm())

    with torch.no_grad():
        ratio_with_bias, const_with_bias = const_over_varying(cm.write_gate(states))
        # The bias IS larger than the content term -- the autopsy's premise holds.
        lin = cm.write_gate[0]
        content_norm = (states @ lin.weight.t()).norm(dim=-1).mean()
        assert float(lin.bias.norm() / content_norm) > 1.0

        # ... and removing it changes essentially nothing.
        nn.init.zeros_(lin.bias)
        ratio_no_bias, const_no_bias = const_over_varying(cm.write_gate(states))

    sigmoid_midpoint = 0.5 * math.sqrt(MEM)
    assert const_with_bias == pytest.approx(sigmoid_midpoint, rel=0.01)
    assert const_no_bias == pytest.approx(sigmoid_midpoint, rel=0.01)
    assert ratio_with_bias > 20.0, ratio_with_bias
    assert ratio_no_bias > 20.0, ratio_no_bias
    assert ratio_no_bias == pytest.approx(ratio_with_bias, rel=0.25)


def test_config_flag_reaches_the_module():
    from ree_core.utils.config import REEConfig
    from ree_core.predictors.e1_deep import E1DeepPredictor

    for flag in (False, True):
        cfg = REEConfig.from_dims(
            body_obs_dim=8, world_obs_dim=8, action_dim=4,
            self_dim=32, world_dim=32,
            contextmemory_gated_content_write=flag,
        )
        assert cfg.e1.contextmemory_gated_content_write is flag
        e1 = E1DeepPredictor(cfg.e1)
        assert e1.context_memory.gated_content_write is flag
        assert (e1.context_memory.write_content is not None) is flag
