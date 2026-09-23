"""
Contract tests for the dedicated HARM-residue RBF bandwidth
(SD residue-field-kernel-resolution; GFLAG-0337; MECH-023 / INV-023 / MECH-018).

THE DEFECT. ResidueField's harm RBF (self.rbf_field) inherits the shared
ResidueConfig.kernel_bandwidth = 1.0, while the reachable z_world cloud spans
~0.125 (max pairwise; median 0.065, igw-242 2026-09-17). The two most distant
points the agent can reach therefore read exp(-0.125^2 / 2) = 0.9922 -- 0.8%
apart. The harm field is a near-broadcast constant across everywhere the agent
can go, and a ratio-of-means readout over it SATURATES. Measured 2026-09-23
(pilot, session inv023-dvrestructure-20260923): harm/safe ratio at bandwidth 1.0
was 1.006675 (seed 0) and 1.000454 (seed 1) -- seed-INDEPENDENT saturation.

SD-067 fixed exactly this for the MECH-303 SAFETY terrain
(ResidueConfig.safety_terrain_bandwidth). The harm-geometry case was unowned;
this is it. The two knobs are independent and these tests pin that.

SCOPE OF THE CLAIM, stated because it is the caveat most easily lost between a
build and the experiment that consumes it: this knob fixes the INSTRUMENT. A
bandwidth ~8x the reachable manifold cannot resolve separation WHEN separation
exists. It is NOT a claim that the separation is reliably there -- it was
measured adequate on 1 of 2 seeds (seed 1's harm/safe centroid separation,
0.080564, is BELOW the within-cluster spread of both classes, 0.120749 /
0.085289; its rank AUC is flat at 0.5732-0.5940 across a 15x bandwidth range).

Contracts:
  C1  OFF by default, and the default path is BIT-IDENTICAL to the pre-knob code
      -- including against a config object that does not carry the attribute at
      all (which is what the pre-knob ResidueConfig was).
  C2  When set, the value REACHES the field: both the harm RBF read AND
      integrate()'s distillation sampling scale. Construction succeeding is not
      the assertion -- the realised bandwidth at each consumer is.
  C3  Discrimination: at the shared (wide) bandwidth the harm/safe read is
      saturated; at the recommended tighter bandwidth it resolves into an
      absolute gap. Plus the mechanism behind the documented FLOOR.
  C4  from_dims wiring end-to-end, and the negative-instrument guard: from_dims
      SILENTLY SWALLOWS unknown kwargs via **kwargs, so a knob wired at two of
      its three sites is accepted and ignored. C4 pins the positive path AND
      demonstrates the swallow, so the positive test is known to discriminate.
"""

import copy

import pytest
import torch

from ree_core.utils.config import ResidueConfig, REEConfig
from ree_core.residue.field import ResidueField


WORLD_DIM = 8


def _cfg(harm_bw=None, kernel_bw=1.0, world_dim=WORLD_DIM):
    cfg = ResidueConfig()
    cfg.world_dim = world_dim
    cfg.num_basis_functions = 32
    cfg.kernel_bandwidth = kernel_bw
    cfg.harm_field_bandwidth = harm_bw
    return cfg


# ---------------------------------------------------------------------------
# C1  OFF by default; default path bit-identical to the pre-knob code
# ---------------------------------------------------------------------------

def test_c1_config_default_is_none():
    # Read from the module under test -- not restated as a literal elsewhere.
    assert ResidueConfig().harm_field_bandwidth is None


@pytest.mark.parametrize("kernel_bw", [0.25, 1.0, 2.5])
def test_c1_none_falls_back_to_kernel_bandwidth(kernel_bw):
    """Fallback is asserted against the config's OWN kernel_bandwidth.

    Parametrised so a hardcoded 1.0 cannot pass vacuously: if the fallback were
    dropped and the RBF took some fixed default, two of the three cells fail.
    """
    cfg = _cfg(harm_bw=None, kernel_bw=kernel_bw)
    rf = ResidueField(cfg)
    assert rf.rbf_field.bandwidth == cfg.kernel_bandwidth
    assert rf.effective_harm_bandwidth == cfg.kernel_bandwidth


class _PreKnobConfigView:
    """A ResidueConfig as it existed BEFORE this knob: no such attribute at all.

    This is the honest control for the bit-identity proof. It is also the real
    deployed case -- an old pickled/checkpointed config reaching the new code --
    and it is what exercises the getattr(config, "harm_field_bandwidth", None)
    fallback rather than the None-valued field.
    """

    _HIDDEN = ("harm_field_bandwidth",)

    def __init__(self, cfg):
        object.__setattr__(self, "_cfg", cfg)

    def __getattr__(self, name):
        if name in _PreKnobConfigView._HIDDEN:
            raise AttributeError(name)
        return getattr(object.__getattribute__(self, "_cfg"), name)

    def __setattr__(self, name, value):
        setattr(object.__getattribute__(self, "_cfg"), name, value)


def _workload(rf, seed):
    """Deterministic exercise of every harm-bandwidth consumer; returns tensors."""
    torch.manual_seed(seed)
    pts = [torch.randn(WORLD_DIM) * 0.05 for _ in range(12)]
    for p in pts:
        rf.accumulate(p, harm_magnitude=0.3)
    torch.manual_seed(seed + 1000)
    queries = torch.randn(6, WORLD_DIM) * 0.05
    out = {
        "evaluate": rf.evaluate(queries).detach().clone(),
        "weights": rf.rbf_field.weights.detach().clone(),
        "centers": rf.rbf_field.centers.detach().clone(),
        "total": rf.total_residue.detach().clone(),
    }
    torch.manual_seed(seed + 2000)
    out["integrate"] = rf.integrate(num_steps=5)
    return out


def test_c1_default_path_bitidentical_to_config_without_the_attribute():
    """The strong bit-identity proof.

    The pre-knob ResidueConfig had NO harm_field_bandwidth attribute at all, so
    the honest control is a config object with the attribute DELETED -- which
    exercises the getattr(..., None) fallback exactly as old configs (and old
    pickled/checkpointed configs) do. Every tensor and every integrate() metric
    must match bitwise.
    """
    legacy = _PreKnobConfigView(_cfg(harm_bw=None))
    # The control is only a control if the attribute is genuinely ABSENT. A
    # dataclass field with a default lives on the CLASS, so delattr() on the
    # instance leaves hasattr() True and the control silently degrades into a
    # copy of the test case -- assert the absence rather than assuming it.
    assert not hasattr(legacy, "harm_field_bandwidth")

    modern = _cfg(harm_bw=None)
    assert modern.harm_field_bandwidth is None

    torch.manual_seed(7)
    rf_legacy = ResidueField(legacy)
    torch.manual_seed(7)
    rf_modern = ResidueField(modern)

    a = _workload(rf_legacy, seed=11)
    b = _workload(rf_modern, seed=11)

    for key in ("evaluate", "weights", "centers", "total"):
        assert torch.equal(a[key], b[key]), f"{key} not bit-identical"
    assert a["integrate"].keys() == b["integrate"].keys()
    for k in a["integrate"]:
        assert a["integrate"][k] == b["integrate"][k], f"integrate[{k}] differs"


def test_c1_off_does_not_disturb_the_sd067_safety_knob():
    """The two bandwidths are independent knobs, in both directions."""
    cfg = _cfg(harm_bw=0.15)
    cfg.safety_terrain_enabled = True
    cfg.safety_terrain_bandwidth = None
    rf = ResidueField(cfg)
    assert rf.rbf_field.bandwidth == 0.15
    assert rf.safety_terrain_rbf_field.bandwidth == cfg.kernel_bandwidth

    cfg2 = _cfg(harm_bw=None)
    cfg2.safety_terrain_enabled = True
    cfg2.safety_terrain_bandwidth = 0.03
    rf2 = ResidueField(cfg2)
    assert rf2.rbf_field.bandwidth == cfg2.kernel_bandwidth
    assert rf2.safety_terrain_rbf_field.bandwidth == 0.03


# ---------------------------------------------------------------------------
# C2  The value REACHES each consumer (not merely: construction succeeded)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("harm_bw", [0.15, 0.2])
def test_c2_knob_reaches_the_harm_rbf_read(harm_bw):
    cfg = _cfg(harm_bw=harm_bw, kernel_bw=1.0)
    rf = ResidueField(cfg)
    # Distinct from the fallback, so "it fell back" cannot satisfy this.
    assert rf.rbf_field.bandwidth != cfg.kernel_bandwidth
    assert rf.rbf_field.bandwidth == pytest.approx(harm_bw)


def test_c2_knob_reaches_integrate_sampling_noise():
    """Measure the REALISED sampling scale inside integrate().

    integrate() draws `randn_like(harm_locations) * <bandwidth>` and distils the
    RBF onto neural_field at those points. A knob wired into the RBF read but
    NOT here would sample at the shared 1.0 -- far outside a narrowed field's
    support -- making targets ~0 everywhere and the loop vacuously green. So the
    scale is measured, not assumed: randn_like is pinned to ones, and the RBF's
    own forward input is captured, giving the offset directly.
    """
    import ree_core.residue.field as field_mod

    harm_bw = 0.15
    cfg = _cfg(harm_bw=harm_bw, kernel_bw=1.0)
    rf = ResidueField(cfg)

    torch.manual_seed(3)
    locations = [torch.randn(WORLD_DIM) * 0.05 for _ in range(6)]
    for p in locations:
        rf.accumulate(p, harm_magnitude=0.3)

    captured = []

    def _hook(_module, inputs):
        captured.append(inputs[0].detach().clone())

    handle = rf.rbf_field.register_forward_pre_hook(_hook)
    real_randn_like = field_mod.torch.randn_like
    field_mod.torch.randn_like = lambda t, *a, **k: torch.ones_like(t)
    try:
        rf.integrate(num_steps=1)
    finally:
        field_mod.torch.randn_like = real_randn_like
        handle.remove()

    assert captured, "integrate() never called the harm RBF"
    sample_points = captured[0]
    base = torch.stack(locations)
    offsets = (sample_points - base).flatten()
    realised = float(offsets.mean())

    # Read the expectation from the module under test, and require it to be
    # distinguishable from the fallback the defect would produce.
    assert realised == pytest.approx(rf.effective_harm_bandwidth, abs=1e-6)
    assert realised == pytest.approx(harm_bw, abs=1e-6)
    assert abs(realised - cfg.kernel_bandwidth) > 0.5


# ---------------------------------------------------------------------------
# C3  Saturation at the shared bandwidth; resolution at the tighter one
# ---------------------------------------------------------------------------

def _harm_safe_read(harm_bw):
    """Harm cluster vs a safe cluster offset at the MEASURED z_world scale.

    Geometry mirrors the 2026-09-23 pilot's seed 0 (the run where separation was
    adequate): centroid_dist ~0.188, within-cluster spread ~0.13.
    """
    cfg = _cfg(harm_bw=harm_bw, kernel_bw=1.0)
    rf = ResidueField(cfg)
    torch.manual_seed(5)
    base = torch.zeros(WORLD_DIM)
    harm_pts = [base + 0.045 * torch.randn(WORLD_DIM) for _ in range(20)]
    for p in harm_pts:
        rf.accumulate(p, harm_magnitude=0.5)

    safe_center = base.clone()
    safe_center[0] += 0.188                      # measured centroid_dist
    torch.manual_seed(6)
    harm_q = torch.stack([base + 0.045 * torch.randn(WORLD_DIM) for _ in range(16)])
    safe_q = torch.stack([safe_center + 0.045 * torch.randn(WORLD_DIM) for _ in range(16)])
    h = float(rf.evaluate(harm_q).mean().detach())
    s = float(rf.evaluate(safe_q).mean().detach())
    return h, s


def test_c3_shared_bandwidth_saturates_and_tighter_resolves():
    wide_h, wide_s = _harm_safe_read(None)      # falls back to kernel_bandwidth 1.0
    tight_h, tight_s = _harm_safe_read(0.15)    # recommended value

    wide_ratio = wide_h / (wide_s + 1e-12)
    tight_ratio = tight_h / (tight_s + 1e-12)

    # Direction must be right in both cases: harm reads higher than safe.
    assert wide_ratio > 1.0 and tight_ratio > 1.0
    # The defect: at the shared bandwidth the gap is a rounding error.
    assert wide_ratio < 1.05, f"expected saturation at the shared bandwidth, got {wide_ratio}"
    # The fix: the tighter bandwidth turns it into an absolute gap.
    assert tight_ratio > 1.5, f"tighter bandwidth failed to resolve, got {tight_ratio}"
    assert tight_ratio > wide_ratio * 1.4


def test_c3_below_the_within_cluster_spread_the_read_collapses():
    """The mechanism behind the documented 0.15 FLOOR.

    The recommended value is the smallest ABOVE every measured within-cluster
    spread (max 0.131291, seed 0 harm). Below that floor the kernel is narrower
    than the class itself, so it stops generalising within a class and the read
    collapses toward zero even AT the harm cluster -- which is how seed 0
    INVERTED at bw 0.065 (ratio 0.748631, AUC 0.3587). Pinned as a magnitude
    collapse, which is deterministic, rather than by reproducing the inversion,
    which was seed-dependent.
    """
    at_floor_h, _ = _harm_safe_read(0.15)
    below_floor_h, _ = _harm_safe_read(0.02)
    assert below_floor_h < at_floor_h * 0.5, (
        "a kernel far below the within-cluster spread should collapse the "
        f"in-class read; got {below_floor_h} vs {at_floor_h}"
    )


# ---------------------------------------------------------------------------
# C4  from_dims wiring, and the swallowed-kwarg negative-instrument guard
# ---------------------------------------------------------------------------

_DIMS = dict(body_obs_dim=12, world_obs_dim=250, action_dim=5)


def test_c4_from_dims_default_is_none():
    cfg = REEConfig.from_dims(**_DIMS)
    assert cfg.residue.harm_field_bandwidth is None


def test_c4_from_dims_threads_value_all_the_way_to_the_field():
    """End-to-end: the kwarg must reach a live ResidueField, not just the config.

    The item's own warning is that from_dims accepts and ignores. Asserting the
    config attribute alone would still pass if a later assignment clobbered it,
    so the built field is what is checked.
    """
    cfg = REEConfig.from_dims(harm_field_bandwidth=0.15, **_DIMS)
    assert cfg.residue.harm_field_bandwidth == pytest.approx(0.15)
    rf = ResidueField(cfg.residue)
    assert rf.rbf_field.bandwidth == pytest.approx(0.15)
    assert rf.rbf_field.bandwidth != cfg.residue.kernel_bandwidth


def test_c4_unknown_kwarg_is_silently_swallowed_the_hazard_this_guards():
    """NEGATIVE-INSTRUMENT GUARD -- documents the failure the C4 tests detect.

    from_dims ends in **kwargs and drops what it does not name, with no error.
    This test pins that a MISSPELLED knob is accepted and ignored: that is
    precisely the state the build would be in if the signature entry (one of the
    three wiring sites) were missing. Its passing is what makes
    test_c4_from_dims_threads_value_all_the_way_to_the_field a discriminating
    test rather than a tautology.
    """
    cfg = REEConfig.from_dims(harm_field_bandwidth_TYPO=0.15, **_DIMS)
    assert cfg.residue.harm_field_bandwidth is None
    rf = ResidueField(cfg.residue)
    assert rf.rbf_field.bandwidth == cfg.residue.kernel_bandwidth


def test_c4_from_dims_does_not_disturb_the_safety_knob():
    cfg = REEConfig.from_dims(harm_field_bandwidth=0.15, **_DIMS)
    assert cfg.residue.safety_terrain_bandwidth is None
