"""Contract tests for infant_substrate:GAP-6 residue coverage telemetry.

ResidueField.get_coverage_telemetry() is a read-only, additive summary over
the persistent harm-residue RBF field. It must NOT change get_statistics()
or mutate any field state (EXQ-575 + every existing run depend on this).

  C1  Surface contract -- method exists, returns the 7 required keys, all
      native python int/float (no torch tensors); get_statistics() is
      unchanged (same 4 keys, still tensor-valued).
  C2  residue_coverage_pct correctness -- fresh field is 0.0; K centers
      above threshold give K/num_centers; sub-threshold active centers are
      excluded; the threshold scales with residue_scale_factor; a 0.0
      scale factor is clamped (no div-by-zero, no threshold collapse).
  C3  harm_benefit_ratio -- benefit terrain OFF yields the -1.0 sentinel
      (benefit_total -1.0); ON with harm+benefit yields harm/benefit; ON
      with zero benefit yields the -1.0 sentinel (benefit_total 0.0).
  C4  Non-invasive / bit-identical -- the call mutates no weights /
      active_mask / buffers and is idempotent across repeated calls.
  C5  Edge cases -- coverage stays in [0, 1]; active_centers <= n_centers;
      tiny num_centers still works.
"""

import torch

from ree_core.residue.field import ResidueField
from ree_core.utils.config import ResidueConfig

REQUIRED_KEYS = {
    "residue_coverage_pct",
    "residue_coverage_threshold",
    "residue_active_centers",
    "residue_n_centers",
    "harm_benefit_ratio",
    "harm_total",
    "benefit_total",
}


def _field(num_centers=10, world_dim=4, benefit=False):
    cfg = ResidueConfig(
        world_dim=world_dim,
        num_basis_functions=num_centers,
        benefit_terrain_enabled=benefit,
    )
    return ResidueField(cfg)


def _acc(field, harm_magnitude, world_dim=4):
    field.accumulate(torch.randn(1, world_dim), harm_magnitude=harm_magnitude)


# ---------------------------------------------------------------------------
# C1 -- surface contract + get_statistics() left intact
# ---------------------------------------------------------------------------

def test_c1_keys_and_native_types():
    f = _field()
    tel = f.get_coverage_telemetry()
    assert set(tel.keys()) == REQUIRED_KEYS
    for k, v in tel.items():
        assert isinstance(v, (int, float)), f"{k} is {type(v)}"
        assert not isinstance(v, torch.Tensor), f"{k} leaked a tensor"


def test_c1_get_statistics_unchanged():
    f = _field()
    _acc(f, 1.0)
    stats = f.get_statistics()
    assert set(stats.keys()) == {
        "total_residue",
        "num_harm_events",
        "active_centers",
        "mean_weight",
    }
    for v in stats.values():
        assert isinstance(v, torch.Tensor)


# ---------------------------------------------------------------------------
# C2 -- residue_coverage_pct correctness
# ---------------------------------------------------------------------------

def test_c2_fresh_field_zero_coverage():
    f = _field(num_centers=10)
    tel = f.get_coverage_telemetry()
    assert tel["residue_coverage_pct"] == 0.0
    assert tel["residue_active_centers"] == 0
    assert tel["residue_n_centers"] == 10


def test_c2_k_centers_above_threshold():
    # accumulation_rate=0.1; harm=1.0 -> weight 0.1 > thr(0.02) at rsf=1.0.
    f = _field(num_centers=10)
    for _ in range(4):
        _acc(f, 1.0)
    tel = f.get_coverage_telemetry()
    assert tel["residue_active_centers"] == 4
    assert tel["residue_coverage_pct"] == 4 / 10


def test_c2_subthreshold_active_centers_excluded():
    # harm=0.1 -> weight 0.01 < thr(0.02): active but NOT covered.
    f = _field(num_centers=10)
    for _ in range(3):
        _acc(f, 1.0)   # 0.1 each -> covered
    for _ in range(2):
        _acc(f, 0.1)   # 0.01 each -> active, sub-threshold
    tel = f.get_coverage_telemetry()
    assert tel["residue_active_centers"] == 5
    assert tel["residue_coverage_pct"] == 3 / 10


def test_c2_threshold_scales_with_residue_scale_factor():
    f = _field(num_centers=10)
    for _ in range(5):
        _acc(f, 1.0)   # weight 0.1 each
    base = f.get_coverage_telemetry(residue_scale_factor=1.0)
    assert base["residue_coverage_pct"] == 5 / 10
    assert base["residue_coverage_threshold"] == 0.02
    # rsf=10 -> thr 0.2 > 0.1 weights -> nothing covered.
    high = f.get_coverage_telemetry(residue_scale_factor=10.0)
    assert high["residue_coverage_threshold"] == 0.2
    assert high["residue_coverage_pct"] == 0.0


def test_c2_zero_scale_factor_clamped():
    f = _field(num_centers=8)
    for _ in range(3):
        _acc(f, 1.0)
    tel = f.get_coverage_telemetry(residue_scale_factor=0.0)
    assert tel["residue_coverage_threshold"] > 0.0
    # tiny clamped threshold -> every active center counts.
    assert tel["residue_coverage_pct"] == tel["residue_active_centers"] / 8


# ---------------------------------------------------------------------------
# C3 -- harm_benefit_ratio
# ---------------------------------------------------------------------------

def test_c3_benefit_terrain_off_sentinel():
    f = _field(benefit=False)
    for _ in range(3):
        _acc(f, 1.0)
    tel = f.get_coverage_telemetry()
    assert tel["harm_benefit_ratio"] == -1.0
    assert tel["benefit_total"] == -1.0
    assert tel["harm_total"] > 0.0


def test_c3_benefit_terrain_on_ratio():
    f = _field(benefit=True)
    _acc(f, 1.0)
    _acc(f, 1.0)                       # harm_total = 0.2
    f.accumulate_benefit(torch.randn(1, 4), benefit_magnitude=1.0)  # benefit 1.0
    tel = f.get_coverage_telemetry()
    assert abs(tel["harm_total"] - 0.2) < 1e-6
    assert abs(tel["benefit_total"] - 1.0) < 1e-6
    assert abs(tel["harm_benefit_ratio"] - 0.2) < 1e-6


def test_c3_benefit_terrain_on_zero_benefit_sentinel():
    f = _field(benefit=True)
    _acc(f, 1.0)
    tel = f.get_coverage_telemetry()
    assert tel["harm_benefit_ratio"] == -1.0
    assert tel["benefit_total"] == 0.0   # terrain on, no benefit yet (not -1.0)


# ---------------------------------------------------------------------------
# C4 -- non-invasive / bit-identical
# ---------------------------------------------------------------------------

def test_c4_call_does_not_mutate_state():
    f = _field(num_centers=10, benefit=True)
    for _ in range(4):
        _acc(f, 1.0)
    f.accumulate_benefit(torch.randn(1, 4), benefit_magnitude=2.0)

    w0 = f.rbf_field.weights.detach().clone()
    m0 = f.rbf_field.active_mask.clone()
    tr0 = f.total_residue.item()
    tb0 = f.total_benefit.item()

    t1 = f.get_coverage_telemetry()
    t2 = f.get_coverage_telemetry()

    assert torch.equal(f.rbf_field.weights.detach(), w0)
    assert torch.equal(f.rbf_field.active_mask, m0)
    assert f.total_residue.item() == tr0
    assert f.total_benefit.item() == tb0
    assert t1 == t2


# ---------------------------------------------------------------------------
# C5 -- edge cases
# ---------------------------------------------------------------------------

def test_c5_coverage_bounds_and_small_field():
    f = _field(num_centers=2)
    for _ in range(6):          # more accumulations than centers (recycling)
        _acc(f, 1.0)
    tel = f.get_coverage_telemetry()
    assert 0.0 <= tel["residue_coverage_pct"] <= 1.0
    assert tel["residue_active_centers"] <= tel["residue_n_centers"] == 2
