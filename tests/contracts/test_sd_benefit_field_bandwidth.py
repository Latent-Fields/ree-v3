"""
Contract tests for the dedicated BENEFIT-terrain RBF bandwidth
(SD benefit-field-kernel-resolution; SD-024 / MECH-232, SD-025 / ARC-057).

THE DEFECT. ResidueField's benefit terrain (self.benefit_rbf_field) inherited the
shared ResidueConfig.kernel_bandwidth = 1.0, while the reachable z_world cloud on
the live untrained-encoder manifold spans 0.07 (MAX pairwise over 180 visited
states across 7 grid cells, ||z|| ~0.33; measured 2026-09-24, session
orch0924-sd024, red-team BLOCKING F-1 on EXP-1391). The two most distant points
the agent can reach therefore read exp(-0.07^2 / 2) = 0.99755 apart -- a
near-broadcast constant. Measured consequence: compute_benefit_density returned
~11.5 at EVERY visited state with 23 active centers. The benefit map had no
spatial structure, so SD-024's density falsifier and the SD-025 curiosity drive
(which follows that density) both read a constant.

This is the THIRD instance of the SD-067 saturation class, after the MECH-303
safety terrain (safety_terrain_bandwidth) and the harm field
(harm_field_bandwidth). The three knobs are independent and C1 pins that.

WHAT THIS BUILD DELIBERATELY DOES NOT DO -- read before copying a number out of
here. It does not choose the scientific operating value of the bandwidth. The
user's 2026-09-24 decision (option B, rec-20260924-3a88e1e9) was to build the
knob and NOT to calibrate it; the operating value is pre-registered by the
re-queued EXP-1391 from its own P0 measurement of the live manifold.
_RESOLVABILITY_DEMO_BW below exists ONLY to demonstrate that the knob can
resolve the measured geometry at all, and is NOT a recommendation. In particular
do not reach for harm_field_bandwidth's 0.15: that was tuned against a different
(0.125) manifold and a different readout, and it exceeds this whole 0.07 manifold.

Contracts:
  C1  OFF by default, and the default path is BIT-IDENTICAL to the pre-knob code
      -- including against a config object that does not carry the attribute at
      all (which is what the pre-knob ResidueConfig was). Plus independence from
      the harm and safety bandwidths in both directions.
  C2  When set, the value REACHES every benefit consumer: the field read
      (evaluate_benefit), the weight-independent density read
      (compute_benefit_density), and the SD-024 DA cluster allocator's
      per-center narrowing base. Construction succeeding is not the assertion --
      the realised bandwidth at each consumer is.
  C3  RESOLVABILITY, the chip's validation criterion: on the measured manifold
      geometry, benefit density at a held-out state in a DIFFERENT grid cell is
      < 0.5x the density at the contact state when the knob is armed, and
      indistinguishable from it (the defect) when it is not.
  C4  from_dims wiring end-to-end, and the negative-instrument guard: from_dims
      SILENTLY SWALLOWS unknown kwargs via **kwargs, so a knob wired at two of
      its three sites is accepted and ignored. C4 pins the positive path AND
      demonstrates the swallow, so the positive test is known to discriminate.
  C5  The HARM field is UNCHANGED by this knob -- including integrate(), whose
      distillation sampling scale is the one place the harm knob needed a second
      wiring site. The benefit field has no such site (integrate() is harm-only)
      and C5 is what keeps that true.
"""

import pytest
import torch

from ree_core.utils.config import ResidueConfig, REEConfig
from ree_core.residue.field import ResidueField


WORLD_DIM = 8

# MEASURED live-manifold geometry (orch0924-sd024, 2026-09-24, quoted via the chip
# prompt -- the primary artifacts are uncommitted and machine-local to DLAPTOP, so
# EXP-1391's P0 must RE-MEASURE rather than cite these).
MANIFOLD_MAX_PAIRWISE = 0.07    # max pairwise distance over 180 visited states
MANIFOLD_RADIUS = 0.33          # ||z_world||
N_BENEFIT_CENTERS = 23          # centers active in the measured run

# ASSUMED FOR THIS FIXTURE -- NOT MEASURED. The source measurement gives 7 grid cells
# over a 0.07 manifold but no per-cell separation and no within-cell spread, and the
# registry entry's own ready_blocked_by makes the within-cell spread something
# EXP-1391's P0 still OWES. They are set here only to build a geometry the criterion
# can be exercised against.
#
# READ THIS BEFORE RE-DERIVING A BANDWIDTH FROM THEM. Together they bracket one:
# WITHIN_CELL_SPREAD < bw < GRID_CELL_SEP gives 0.004 < bw < 0.05, i.e. ~0.02 -- which
# is exactly the fixture value below. That bracket is an artifact of these two ASSUMED
# numbers, not a result. The floor half in particular is the one the harm knob measured
# an INVERSION below (see constraint (b) in the ResidueConfig comment), so a real value
# needs a real spread measurement.
GRID_CELL_SEP = 0.05            # two states in different grid cells, inside 0.07
WITHIN_CELL_SPREAD = 0.004

# A bandwidth that RESOLVES the geometry above -- a test fixture, NOT the
# operating value (see the module docstring). Arithmetic: at bw 0.02 a
# GRID_CELL_SEP-distant point reads exp(-0.05^2 / (2 * 0.02^2)) = 0.044 of a
# co-located one, so the criterion has real margin; at the shared 1.0 it reads
# exp(-0.05^2 / 2) = 0.99875, which is the defect.
_RESOLVABILITY_DEMO_BW = 0.02


def _cfg(benefit_bw=None, kernel_bw=1.0, benefit_on=True, da=False):
    cfg = ResidueConfig()
    cfg.world_dim = WORLD_DIM
    cfg.num_basis_functions = 64
    cfg.kernel_bandwidth = kernel_bw
    cfg.benefit_field_bandwidth = benefit_bw
    cfg.benefit_terrain_enabled = benefit_on
    if da:
        cfg.use_da_modulated_rbf_density = True
        cfg.da_allocation_scale = 4.0
        # 0.01, NOT the config default 0.1 -- and that is a deliberate coupling
        # choice, not an arbitrary constant. The default (and the 0.3 the live SD-024
        # run used) scatters DA-allocated centers outside any manifold-resolving
        # kernel, which zeroes the density at the reward site. See
        # test_f1_da_cluster_dv_inverts_at_the_default_jitter_radius, which is the
        # cell that pins that failure rather than hiding behind this override.
        cfg.da_jitter_radius = 0.01
        cfg.da_bandwidth_narrowing = 0.5
    return cfg


# ---------------------------------------------------------------------------
# C1  OFF by default; default path bit-identical to the pre-knob code
# ---------------------------------------------------------------------------

def test_c1_config_default_is_none():
    # Read from the module under test -- not restated as a literal elsewhere.
    assert ResidueConfig().benefit_field_bandwidth is None


@pytest.mark.parametrize("kernel_bw", [0.25, 1.0, 2.5])
def test_c1_none_falls_back_to_kernel_bandwidth(kernel_bw):
    """Fallback is asserted against the config's OWN kernel_bandwidth.

    Parametrised so a hardcoded 1.0 cannot pass vacuously: if the fallback were
    dropped and the RBF took some fixed default, two of the three cells fail.
    """
    cfg = _cfg(benefit_bw=None, kernel_bw=kernel_bw)
    rf = ResidueField(cfg)
    assert rf.benefit_rbf_field.bandwidth == cfg.kernel_bandwidth
    assert rf.effective_benefit_bandwidth == cfg.kernel_bandwidth


def test_c1_off_branch_does_not_coerce_the_type():
    """Bit-identity by IDENTITY, not by a numerical argument.

    The pre-knob code passed config.kernel_bandwidth straight through, so a
    config carrying an INT kernel_bandwidth handed RBFLayer an int. A float()
    around the OFF branch would silently change that, which is exactly the kind
    of "numerically equal, not identical" drift the harm knob's comment calls out.
    """
    cfg = _cfg(benefit_bw=None, kernel_bw=1)      # int on purpose
    rf = ResidueField(cfg)
    assert isinstance(rf.effective_benefit_bandwidth, int)
    assert isinstance(rf.benefit_rbf_field.bandwidth, int)


class _PreKnobConfigView:
    """A ResidueConfig as it existed BEFORE this knob: no such attribute at all.

    This is the honest control for the bit-identity proof. It is also the real
    deployed case -- an old pickled/checkpointed config reaching the new code --
    and it is what exercises the getattr(config, "benefit_field_bandwidth", None)
    fallback rather than the None-valued field.
    """

    _HIDDEN = ("benefit_field_bandwidth",)

    def __init__(self, cfg):
        object.__setattr__(self, "_cfg", cfg)

    def __getattr__(self, name):
        if name in _PreKnobConfigView._HIDDEN:
            raise AttributeError(name)
        return getattr(object.__getattribute__(self, "_cfg"), name)

    def __setattr__(self, name, value):
        setattr(object.__getattribute__(self, "_cfg"), name, value)


def _workload(rf, seed):
    """Deterministic exercise of every benefit-bandwidth consumer + the harm path."""
    torch.manual_seed(seed)
    pts = [torch.randn(WORLD_DIM) * 0.02 for _ in range(10)]
    for i, p in enumerate(pts):
        # Alternate the DA cluster path and the single-center path so both
        # allocators are covered when the DA master switch is on.
        rf.accumulate_benefit(p, benefit_magnitude=0.7, dopamine_signal=(0.6 if i % 2 else 0.0))
        rf.accumulate(p * 1.5, harm_magnitude=0.3)
    torch.manual_seed(seed + 1000)
    q = torch.randn(6, WORLD_DIM) * 0.02
    out = {
        "benefit": rf.evaluate_benefit(q).detach().clone(),
        "density": rf.compute_benefit_density(q).detach().clone(),
        "b_weights": rf.benefit_rbf_field.weights.detach().clone(),
        "b_centers": rf.benefit_rbf_field.centers.detach().clone(),
        "total_benefit": rf.total_benefit.detach().clone(),
        "harm": rf.evaluate(q).detach().clone(),
        "h_weights": rf.rbf_field.weights.detach().clone(),
    }
    torch.manual_seed(seed + 2000)
    out["integrate"] = rf.integrate(num_steps=5)
    return out


@pytest.mark.parametrize("da", [False, True])
def test_c1_default_path_bitidentical_to_config_without_the_attribute(da):
    """The strong bit-identity proof.

    The pre-knob ResidueConfig had NO benefit_field_bandwidth attribute at all,
    so the honest control is a config object with the attribute ABSENT -- which
    exercises the getattr(..., None) fallback exactly as old configs (and old
    pickled/checkpointed configs) do. Every tensor and every integrate() metric
    must match bitwise, on BOTH the single-center and the SD-024 DA cluster
    allocation path.
    """
    legacy = _PreKnobConfigView(_cfg(benefit_bw=None, da=da))
    # The control is only a control if the attribute is genuinely ABSENT. A
    # dataclass field with a default lives on the CLASS, so delattr() on the
    # instance leaves hasattr() True and the control silently degrades into a
    # copy of the test case -- assert the absence rather than assuming it.
    assert not hasattr(legacy, "benefit_field_bandwidth")

    modern = _cfg(benefit_bw=None, da=da)
    assert modern.benefit_field_bandwidth is None

    torch.manual_seed(7)
    rf_legacy = ResidueField(legacy)
    torch.manual_seed(7)
    rf_modern = ResidueField(modern)

    a = _workload(rf_legacy, seed=11)
    b = _workload(rf_modern, seed=11)

    for key in ("benefit", "density", "b_weights", "b_centers", "total_benefit",
                "harm", "h_weights"):
        assert torch.equal(a[key], b[key]), f"{key} not bit-identical"
    assert a["integrate"].keys() == b["integrate"].keys()
    for k in a["integrate"]:
        assert a["integrate"][k] == b["integrate"][k], f"integrate[{k}] differs"


def test_c1_the_three_bandwidth_knobs_are_independent():
    """Benefit / harm / safety are three separate levers, in every direction."""
    cfg = _cfg(benefit_bw=_RESOLVABILITY_DEMO_BW)
    cfg.safety_terrain_enabled = True
    cfg.safety_terrain_bandwidth = None
    cfg.harm_field_bandwidth = None
    rf = ResidueField(cfg)
    assert rf.benefit_rbf_field.bandwidth == _RESOLVABILITY_DEMO_BW
    assert rf.rbf_field.bandwidth == cfg.kernel_bandwidth
    assert rf.safety_terrain_rbf_field.bandwidth == cfg.kernel_bandwidth

    cfg2 = _cfg(benefit_bw=None)
    cfg2.safety_terrain_enabled = True
    cfg2.safety_terrain_bandwidth = 0.03
    cfg2.harm_field_bandwidth = 0.15
    rf2 = ResidueField(cfg2)
    assert rf2.benefit_rbf_field.bandwidth == cfg2.kernel_bandwidth
    assert rf2.rbf_field.bandwidth == 0.15
    assert rf2.safety_terrain_rbf_field.bandwidth == 0.03


def test_c1_resolution_happens_even_with_the_terrain_disabled():
    """The effective scale resolves unconditionally.

    Resolving inside the benefit_terrain_enabled block would leave the attribute
    missing for any caller that enables the terrain afterwards, and would make
    the OFF-terrain case unassertable. Pinned so a later "tidy-up" that moves the
    resolution into the gate fails here.
    """
    rf = ResidueField(_cfg(benefit_bw=_RESOLVABILITY_DEMO_BW, benefit_on=False))
    assert rf.effective_benefit_bandwidth == _RESOLVABILITY_DEMO_BW
    assert not hasattr(rf, "benefit_rbf_field")


# ---------------------------------------------------------------------------
# C2  The value REACHES each consumer (not merely: construction succeeded)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("benefit_bw", [0.005, 0.02])
def test_c2_knob_reaches_the_benefit_field_read(benefit_bw):
    cfg = _cfg(benefit_bw=benefit_bw, kernel_bw=1.0)
    rf = ResidueField(cfg)
    # Distinct from the fallback, so "it fell back" cannot satisfy this.
    assert rf.benefit_rbf_field.bandwidth != cfg.kernel_bandwidth
    assert rf.benefit_rbf_field.bandwidth == pytest.approx(benefit_bw)

    # And the realised READ moves with it: a point one grid cell away from the
    # only center must be attenuated, which it is not at the shared bandwidth.
    torch.manual_seed(2)
    centre = torch.zeros(WORLD_DIM)
    rf.accumulate_benefit(centre, benefit_magnitude=1.0)
    away = centre.clone()
    away[0] += GRID_CELL_SEP
    near = float(rf.evaluate_benefit(centre.unsqueeze(0))[0].detach())
    far = float(rf.evaluate_benefit(away.unsqueeze(0))[0].detach())
    assert far < 0.5 * near, f"read not attenuated at bw={benefit_bw}: {far} vs {near}"


def test_c2_knob_reaches_the_density_read():
    """compute_benefit_density passes bandwidth=None -> the layer's own scale.

    The density read is the SD-024 falsifier's actual instrument and is a
    SEPARATE code path from forward() (compute_local_density has its own
    bandwidth override argument), so it is measured rather than assumed.
    """
    cfg = _cfg(benefit_bw=_RESOLVABILITY_DEMO_BW)
    rf = ResidueField(cfg)
    centre = torch.zeros(WORLD_DIM)
    rf.accumulate_benefit(centre, benefit_magnitude=1.0)
    away = centre.clone()
    away[0] += GRID_CELL_SEP

    d_near = float(rf.compute_benefit_density(centre.unsqueeze(0))[0].detach())
    d_far = float(rf.compute_benefit_density(away.unsqueeze(0))[0].detach())
    assert d_far < 0.5 * d_near

    # The explicit override still wins -- pinned so wiring the constructor did
    # not accidentally hard-code the layer scale over the argument.
    d_over = float(rf.compute_benefit_density(away.unsqueeze(0), bandwidth=1.0)[0].detach())
    assert d_over > 0.99


def test_c2_knob_reaches_the_da_cluster_allocator_base():
    """SD-024 per-center narrowing must be based on the BENEFIT scale.

    add_residue_cluster computes per_bw from float(self.bandwidth) and floors it
    at 0.5 * self.bandwidth. If the layer had been built at the shared
    kernel_bandwidth, every DA-allocated center would carry a bandwidth ~50x the
    whole manifold and the narrowing would be meaningless.
    """
    cfg = _cfg(benefit_bw=_RESOLVABILITY_DEMO_BW, da=True)
    rf = ResidueField(cfg)
    assert rf.benefit_rbf_field.per_center_bandwidth is True
    # Initialised from the constructor argument.
    init = rf.benefit_rbf_field.center_bandwidths.clone()
    assert torch.allclose(init, torch.full_like(init, _RESOLVABILITY_DEMO_BW))

    torch.manual_seed(4)
    rf.accumulate_benefit(torch.zeros(WORLD_DIM), benefit_magnitude=1.0, dopamine_signal=0.6)
    touched = rf.benefit_rbf_field.center_bandwidths[rf.benefit_rbf_field.active_mask]
    assert touched.numel() > 1, "DA cluster path did not allocate multiple centers"
    # Narrowed relative to the benefit scale, and floored at half of it -- never
    # anywhere near the shared kernel_bandwidth.
    assert float(touched.max()) <= _RESOLVABILITY_DEMO_BW + 1e-9
    assert float(touched.min()) >= 0.5 * _RESOLVABILITY_DEMO_BW - 1e-9
    assert float(touched.max()) < 0.1 * cfg.kernel_bandwidth


# ---------------------------------------------------------------------------
# C3  The chip's validation criterion: resolvability on the measured manifold
# ---------------------------------------------------------------------------

def _measured_manifold_density(benefit_bw):
    """Reproduce the measured live geometry and read density at two grid cells.

    23 benefit centers clustered at a contact state on a ||z|| ~0.33 shell, and a
    held-out real state GRID_CELL_SEP away -- i.e. a DIFFERENT grid cell, still
    well inside the 0.07 manifold. Returns (contact_density, heldout_density).
    """
    rf = ResidueField(_cfg(benefit_bw=benefit_bw))
    torch.manual_seed(5)
    contact = torch.zeros(WORLD_DIM)
    contact[0] = MANIFOLD_RADIUS / (WORLD_DIM ** 0.5)
    for _ in range(N_BENEFIT_CENTERS):
        rf.accumulate_benefit(
            contact + WITHIN_CELL_SPREAD * torch.randn(WORLD_DIM), benefit_magnitude=1.0
        )
    held_out = contact.clone()
    held_out[1] += GRID_CELL_SEP
    d_contact = float(rf.compute_benefit_density(contact.unsqueeze(0))[0].detach())
    d_held = float(rf.compute_benefit_density(held_out.unsqueeze(0))[0].detach())
    return d_contact, d_held


def test_c3_shared_bandwidth_is_spatially_constant_the_measured_defect():
    """The defect, reproduced: density is the same everywhere on the manifold.

    This is the F-1 finding -- ~11.5 at EVERY visited state with 23 centers. It
    is asserted as a near-EQUALITY rather than an absolute value, because the
    absolute number depends on center count and the claim is about structure.
    """
    d_contact, d_held = _measured_manifold_density(None)   # -> kernel_bandwidth 1.0
    assert d_contact > 1.0, "no centers active -- the probe itself is broken"
    assert d_held / d_contact > 0.99, (
        "expected a spatially constant benefit map at the shared bandwidth; got "
        f"contact={d_contact} heldout={d_held}"
    )


def test_c3_armed_knob_meets_the_resolvability_criterion():
    """THE VALIDATION CRITERION registered with this substrate item.

    'benefit density at a held-out real state in a different grid cell < 0.5x the
    density at the contact state, on the live manifold at the chosen bandwidth.'

    _RESOLVABILITY_DEMO_BW is a fixture demonstrating the knob CAN satisfy this,
    not the operating value -- EXP-1391 pre-registers that from its own P0.
    """
    d_contact, d_held = _measured_manifold_density(_RESOLVABILITY_DEMO_BW)
    assert d_contact > 1.0, "no centers active -- the probe itself is broken"
    assert d_held < 0.5 * d_contact, (
        f"resolvability criterion failed: contact={d_contact} heldout={d_held}"
    )


def test_c3_the_two_cases_differ_so_the_criterion_discriminates():
    """Guard against both cells passing for a reason unrelated to the knob."""
    off_c, off_h = _measured_manifold_density(None)
    on_c, on_h = _measured_manifold_density(_RESOLVABILITY_DEMO_BW)
    assert (off_h / off_c) > 10.0 * (on_h / on_c)


# ---------------------------------------------------------------------------
# C4  from_dims wiring, and the swallowed-kwarg negative-instrument guard
# ---------------------------------------------------------------------------

_DIMS = dict(body_obs_dim=12, world_obs_dim=250, action_dim=5)


def test_c4_from_dims_default_is_none():
    cfg = REEConfig.from_dims(**_DIMS)
    assert cfg.residue.benefit_field_bandwidth is None


def test_c4_from_dims_threads_value_all_the_way_to_the_field():
    """End-to-end: the kwarg must reach a live ResidueField, not just the config.

    from_dims accepts and ignores unknown kwargs (see the guard below), so
    asserting the config attribute alone would still pass if a later assignment
    clobbered it. The built field is what is checked.
    """
    cfg = REEConfig.from_dims(benefit_field_bandwidth=_RESOLVABILITY_DEMO_BW, **_DIMS)
    assert cfg.residue.benefit_field_bandwidth == pytest.approx(_RESOLVABILITY_DEMO_BW)
    cfg.residue.benefit_terrain_enabled = True
    rf = ResidueField(cfg.residue)
    assert rf.benefit_rbf_field.bandwidth == pytest.approx(_RESOLVABILITY_DEMO_BW)
    assert rf.benefit_rbf_field.bandwidth != cfg.residue.kernel_bandwidth


def test_c4_unknown_kwarg_is_silently_swallowed_the_hazard_this_guards():
    """NEGATIVE-INSTRUMENT GUARD -- documents the failure the C4 tests detect.

    from_dims ends in **kwargs and drops what it does not name, with no error.
    This pins that a MISSPELLED knob is accepted and ignored: precisely the state
    the build would be in if the signature entry (one of the three wiring sites)
    were missing. Its passing is what makes
    test_c4_from_dims_threads_value_all_the_way_to_the_field a discriminating
    test rather than a tautology.
    """
    cfg = REEConfig.from_dims(benefit_field_bandwidth_TYPO=0.02, **_DIMS)
    assert cfg.residue.benefit_field_bandwidth is None
    cfg.residue.benefit_terrain_enabled = True
    rf = ResidueField(cfg.residue)
    assert rf.benefit_rbf_field.bandwidth == cfg.residue.kernel_bandwidth


def test_c4_from_dims_does_not_disturb_the_sibling_knobs():
    cfg = REEConfig.from_dims(benefit_field_bandwidth=_RESOLVABILITY_DEMO_BW, **_DIMS)
    assert cfg.residue.harm_field_bandwidth is None
    assert cfg.residue.safety_terrain_bandwidth is None


# ---------------------------------------------------------------------------
# C5  The harm field is UNCHANGED (the chip's third validation criterion)
# ---------------------------------------------------------------------------

def _harm_only_workload(cfg, seed=13):
    torch.manual_seed(7)
    rf = ResidueField(cfg)
    torch.manual_seed(seed)
    pts = [torch.randn(WORLD_DIM) * 0.05 for _ in range(12)]
    for p in pts:
        rf.accumulate(p, harm_magnitude=0.3)
    torch.manual_seed(seed + 1000)
    q = torch.randn(6, WORLD_DIM) * 0.05
    out = {
        "harm": rf.evaluate(q).detach().clone(),
        "weights": rf.rbf_field.weights.detach().clone(),
        "centers": rf.rbf_field.centers.detach().clone(),
        "bandwidth": rf.rbf_field.bandwidth,
        "effective_harm": rf.effective_harm_bandwidth,
    }
    torch.manual_seed(seed + 2000)
    out["integrate"] = rf.integrate(num_steps=5)
    return out


def test_c5_arming_the_benefit_knob_leaves_the_harm_field_bit_identical():
    """Including integrate().

    The harm knob needed TWO wiring sites because integrate() distils the harm
    RBF onto neural_field at randn * bandwidth offsets; a benefit knob that
    leaked into that scale would silently corrupt the harm distillation. The
    benefit field has no distillation counterpart, and this is what pins it.
    """
    off = _harm_only_workload(_cfg(benefit_bw=None))
    on = _harm_only_workload(_cfg(benefit_bw=_RESOLVABILITY_DEMO_BW))

    for key in ("harm", "weights", "centers"):
        assert torch.equal(off[key], on[key]), f"harm {key} changed"
    assert off["bandwidth"] == on["bandwidth"]
    assert off["effective_harm"] == on["effective_harm"]
    for k in off["integrate"]:
        assert off["integrate"][k] == on["integrate"][k], f"integrate[{k}] changed"


def test_c5_harm_knob_still_wins_on_the_harm_field_when_both_are_armed():
    """Both armed at DIFFERENT values: each field must take its own."""
    cfg = _cfg(benefit_bw=_RESOLVABILITY_DEMO_BW)
    cfg.harm_field_bandwidth = 0.15
    rf = ResidueField(cfg)
    assert rf.rbf_field.bandwidth == pytest.approx(0.15)
    assert rf.benefit_rbf_field.bandwidth == pytest.approx(_RESOLVABILITY_DEMO_BW)
    assert rf.effective_harm_bandwidth != rf.effective_benefit_bandwidth


# ---------------------------------------------------------------------------
# Red-team findings, 2026-09-25 (this build's own adversarial review).
# Each of these FAILED to be covered by C1-C5 and is pinned here so the gap
# cannot silently reopen. F-1 is the one that changes a SCIENTIFIC reading.
# ---------------------------------------------------------------------------

def _da_cfg(benefit_bw, jitter):
    cfg = _cfg(benefit_bw=benefit_bw, da=True)
    cfg.da_jitter_radius = jitter
    cfg.da_bandwidth_narrowing = 0.0      # isolate the JITTER term
    return cfg


def _single_vs_cluster_density(benefit_bw, jitter):
    """SD-024's DV: does a DA-allocated CLUSTER read denser than a single center?"""
    z = torch.zeros(WORLD_DIM)
    rf1 = ResidueField(_da_cfg(benefit_bw, jitter))
    rf1.accumulate_benefit(z, benefit_magnitude=1.0, dopamine_signal=0.0)
    d_single = float(rf1.compute_benefit_density(z.unsqueeze(0))[0].detach())

    torch.manual_seed(1)
    rf2 = ResidueField(_da_cfg(benefit_bw, jitter))
    rf2.accumulate_benefit(z, benefit_magnitude=1.0, dopamine_signal=0.6)
    d_cluster = float(rf2.compute_benefit_density(z.unsqueeze(0))[0].detach())
    assert int(rf2.benefit_rbf_field.active_mask.sum()) > 1, "DA cluster did not allocate"
    return d_single, d_cluster


def test_f1_da_cluster_dv_inverts_at_the_default_jitter_radius():
    """F-1: narrowing the bandwidth fixes the SPATIAL defect and INVERTS SD-024's DV.

    add_residue_cluster jitters each allocated center by randn * da_jitter_radius.
    At the CONFIG DEFAULT (0.1) -- and at the 0.3 the live SD-024 run used -- the
    cluster lands outside a manifold-resolving kernel, so density at the reward site
    goes to ZERO, the opposite of what SD-024 predicts. The bandwidth and the jitter
    radius must be pre-registered as a PAIR.

    Asserted as a DIRECTION change, not a magnitude, so it is robust to the fixture
    geometry. The default is read from ResidueConfig rather than restated, so a change
    to that default cannot leave this test asserting a stale number.
    """
    default_jitter = ResidueConfig().da_jitter_radius
    assert default_jitter > _RESOLVABILITY_DEMO_BW, (
        "this contract assumes the default jitter is WIDER than a resolving bandwidth; "
        f"default={default_jitter} demo_bw={_RESOLVABILITY_DEMO_BW}"
    )

    # Knob OFF: the DV points the right way at the default jitter.
    off_single, off_cluster = _single_vs_cluster_density(None, default_jitter)
    assert off_cluster > off_single, (
        f"baseline DV already broken: single={off_single} cluster={off_cluster}"
    )

    # Knob ARMED at the same jitter: the DV INVERTS. This is the finding.
    on_single, on_cluster = _single_vs_cluster_density(
        _RESOLVABILITY_DEMO_BW, default_jitter
    )
    assert on_cluster < on_single, (
        "expected the DA-cluster density to COLLAPSE below the single-center density "
        f"at jitter={default_jitter} with the knob armed; got single={on_single} "
        f"cluster={on_cluster}. If this now passes the other way the coupling was "
        "fixed -- update constraint (a) in the ResidueConfig comment."
    )


def test_f1_co_scaling_the_jitter_restores_the_dv():
    """The other half: the inversion is CURABLE, so F-1 is a coupling not a dead end.

    Without this cell the test above could be satisfied by a knob that breaks the DV
    unconditionally, which would be a reason not to ship rather than a constraint.
    """
    co_scaled = _RESOLVABILITY_DEMO_BW / 2.0
    single, cluster = _single_vs_cluster_density(_RESOLVABILITY_DEMO_BW, co_scaled)
    assert cluster > single, (
        f"co-scaled jitter {co_scaled} should restore the DV; "
        f"got single={single} cluster={cluster}"
    )


def test_f4_state_dict_restore_can_silently_defeat_the_knob():
    """F-4: on the SD-024 per-center path the constructor only SEEDS the scale.

    center_bandwidths is a registered BUFFER and _two_bw_sq PREFERS it over
    self.bandwidth, so a checkpoint written before this knob restores the OLD scale
    while effective_benefit_bandwidth still reports the armed one -- structurally
    present, functionally inert. No live call site restores the benefit RBF's
    state_dict today, so this pins a LATENT hazard; if a future change adds one, this
    cell is what says the knob needs re-asserting after the load.
    """
    z = torch.zeros(WORLD_DIM)
    stale = ResidueField(_da_cfg(None, 0.004))        # pre-knob: inherits 1.0
    torch.manual_seed(3)
    for _ in range(8):
        stale.accumulate_benefit(z + 0.004 * torch.randn(WORLD_DIM), 1.0, dopamine_signal=0.6)
    ckpt = {k: v.clone() for k, v in stale.state_dict().items()}
    assert set(ckpt["benefit_rbf_field.center_bandwidths"].tolist()) == {1.0}

    armed = ResidueField(_da_cfg(_RESOLVABILITY_DEMO_BW, 0.004))
    assert armed.benefit_rbf_field.center_bandwidths.max() == pytest.approx(
        _RESOLVABILITY_DEMO_BW
    )
    armed.load_state_dict(ckpt)

    # The scalar and the reported effective value BOTH still say "armed" ...
    assert armed.benefit_rbf_field.bandwidth == pytest.approx(_RESOLVABILITY_DEMO_BW)
    assert armed.effective_benefit_bandwidth == pytest.approx(_RESOLVABILITY_DEMO_BW)
    # ... while the buffer that actually governs the read has reverted.
    assert set(armed.benefit_rbf_field.center_bandwidths.tolist()) == {1.0}

    held = z.clone()
    held[1] += GRID_CELL_SEP
    d_c = float(armed.compute_benefit_density(z.unsqueeze(0))[0].detach())
    d_h = float(armed.compute_benefit_density(held.unsqueeze(0))[0].detach())
    assert d_h > 0.9 * d_c, (
        "the restored 1.0 should put the read back into saturation; "
        f"contact={d_c} heldout={d_h}"
    )


def test_f4_the_scalar_path_has_no_such_buffer():
    """Bounds F-4 to the DA path -- without this the hazard reads broader than it is."""
    rf = ResidueField(_cfg(benefit_bw=_RESOLVABILITY_DEMO_BW, da=False))
    assert rf.benefit_rbf_field.per_center_bandwidth is False
    assert not any("center_bandwidths" in k for k in rf.state_dict())


@pytest.mark.parametrize("spread", [0.004, 0.02])
def test_f3_below_the_within_cell_spread_the_read_collapses(spread):
    """F-3: the FLOOR hazard, the mechanism harm_field_bandwidth documents at 0.15.

    A kernel narrower than the within-cell spread stops generalising within the cell,
    so the read collapses toward zero even AT the contact state -- which is how the
    harm field INVERTED at bw 0.065. Pinned as a magnitude collapse (deterministic)
    rather than by reproducing an inversion (seed-dependent). Parametrised over two
    spreads so a single lucky geometry cannot carry it.
    """
    def contact_density(bw):
        rf = ResidueField(_cfg(benefit_bw=bw))
        torch.manual_seed(5)
        contact = torch.zeros(WORLD_DIM)
        contact[0] = MANIFOLD_RADIUS / (WORLD_DIM ** 0.5)
        for _ in range(N_BENEFIT_CENTERS):
            rf.accumulate_benefit(contact + spread * torch.randn(WORLD_DIM), 1.0)
        return float(rf.compute_benefit_density(contact.unsqueeze(0))[0].detach())

    above = contact_density(spread * 5.0)
    below = contact_density(spread / 10.0)
    assert below < 0.5 * above, (
        f"at spread={spread}: a kernel far below the within-cell spread should collapse "
        f"the in-class read; got below={below} above={above}"
    )


def test_f2_the_ratio_criterion_alone_admits_a_degenerate_pass():
    """F-2: why the contract's `d_contact > 1.0` floor is load-bearing, not decoration.

    The registered criterion is a pure RATIO, and a ratio is satisfied by a kernel so
    narrow that everything reads ~0 including the contact state. This cell pins that
    the degenerate region exists and that the floor is what excludes it -- so nobody
    later "simplifies" the floor away as redundant with the ratio.
    """
    contact_d, held_d = _measured_manifold_density(1e-3)
    # The ratio criterion is satisfied ...
    assert held_d < 0.5 * contact_d
    # ... while the field is dead, which the floor catches and the ratio cannot.
    assert contact_d < 1.0, f"expected a collapsed contact read, got {contact_d}"

