"""Contracts for SD-PP-B5: the action-sensitivity readiness gate + world margin loss.

THE TEST HALF (CLAUDE.md "The test half: a guard that supplies the thing it
asserts is not a guard"). The canary tests below read their expected values from
the MODULE's own CANARY_V3_EXQ_1073 rather than restating them as literals, so a
drift in the pinned values is visible here. `test_blind_spot_*` are the
measurement of what these contracts would actually catch: they construct the
defect and assert the guard FAILS on it.

Assertions are upstream of any discrete quantizer (CLAUDE.md "Running the test
suite"): no torch.multinomial is involved, so these are portable across machine
classes.
"""
import math
import pytest
import torch
import torch.nn as nn

from experiments._lib.action_sensitivity_gate import (
    CANARY_V3_EXQ_1073, RATIO_FLOOR, SKILL_FLOOR, ActionSensitivityVerdict,
    action_shuffle_ratio, check_canary, format_verdict, identity_predictor_mse,
    readiness_verdict, skill_vs_identity)
from ree_core.predictors.e2_fast import E2FastPredictor
from ree_core.utils.config import E2Config, REEConfig


# --------------------------------------------------------------------------- #
# Fixtures: heads with KNOWN action-sensitivity, built without training.        #
# --------------------------------------------------------------------------- #
class _IdentityHead(nn.Module):
    """Copy-the-input. Reads no action at all -- the V3-EXQ-1073 phenotype."""
    def forward(self, z, a):
        return z


class _PerfectHead(nn.Module):
    """Reproduces the battery target exactly -- maximally action-sensitive."""
    def forward(self, z, a):
        return z + a[..., 0:1] * 1.0


class _ActionSensitiveHead(nn.Module):
    """Reads the action but imperfectly -- the realistic 'ready' case."""
    def forward(self, z, a):
        return z + a[..., 0:1] * 0.75


def _battery(n=64, dim=8, adim=4, seed=0, distinct=True):
    g = torch.Generator().manual_seed(seed)
    z0 = torch.randn(n, dim, generator=g)
    idx = (torch.arange(n) % adim) if distinct else torch.zeros(n, dtype=torch.long)
    acts = torch.nn.functional.one_hot(idx, adim).float()
    z1 = z0 + acts[..., 0:1] * 1.0          # target genuinely depends on action 0
    return z0, acts, z1


# --------------------------------------------------------------------------- #
# The three-valued category is STRUCTURAL, not a bool.                          #
# --------------------------------------------------------------------------- #
def test_verdict_status_is_three_valued_and_in_to_dict():
    z0, a, z1 = _battery()
    v = readiness_verdict(_IdentityHead(), z0, a, z1)
    assert isinstance(v, ActionSensitivityVerdict)
    assert v.status in ("ready", "action_blind", "cannot_determine")
    assert "status" in v.to_dict(), "cannot_determine must survive into --json"
    assert v.to_dict()["status"] == v.status


def test_monostrategy_battery_is_cannot_determine_not_action_blind():
    """THE trap. One distinct action makes an action-map change a no-op; the ratio
    comes back exactly 1.0, which a bool would read as a negative result."""
    z0, a, z1 = _battery(distinct=False)
    v = readiness_verdict(_IdentityHead(), z0, a, z1)
    assert v.status == "cannot_determine", v.reason
    assert v.n_distinct_actions == 1
    assert "monostrategy" in v.reason


def test_thin_battery_is_cannot_determine():
    z0, a, z1 = _battery(n=4)
    v = readiness_verdict(_IdentityHead(), z0, a, z1, min_rows=16)
    assert v.status == "cannot_determine"
    assert v.n_rows == 4


def test_zero_identity_mse_is_cannot_determine_not_skill_zero():
    """A battery with no movement to predict: skill is undefined, not zero."""
    z0, a, _ = _battery()
    v = readiness_verdict(_IdentityHead(), z0, a, z0.clone())
    assert v.status == "cannot_determine"
    assert skill_vs_identity(0.0, 0.0) is None


def test_denominators_always_present_and_printed():
    z0, a, z1 = _battery()
    v = readiness_verdict(_IdentityHead(), z0, a, z1)
    assert v.n_rows == 64 and v.n_distinct_actions == 4
    out = format_verdict(v, "t")
    assert "DENOM" in out and "n_rows=64" in out and "n_distinct_actions=4" in out
    assert out.isascii(), "format_verdict must be ASCII-only"


# --------------------------------------------------------------------------- #
# Direction: the gate must separate an action-blind head from a sensitive one.   #
# --------------------------------------------------------------------------- #
def test_identity_head_is_action_blind():
    z0, a, z1 = _battery()
    v = readiness_verdict(_IdentityHead(), z0, a, z1)
    assert v.status == "action_blind", v.reason


def test_action_sensitive_head_is_ready():
    z0, a, z1 = _battery()
    v = readiness_verdict(_ActionSensitiveHead(), z0, a, z1)
    assert v.status == "ready", v.reason
    assert v.ratio > RATIO_FLOOR and v.skill > SKILL_FLOOR


def test_perfect_head_is_ready_not_cannot_determine():
    """mse_true == 0 makes the RATIO undefined by division, but the head is
    plainly action-sensitive: permuting the action makes it wrong. Returning
    cannot_determine here would be a false negative on the clearest case."""
    z0, a, z1 = _battery()
    v = readiness_verdict(_PerfectHead(), z0, a, z1)
    assert v.status == "ready", v.reason
    assert v.ratio == math.inf
    assert "inf (perfect head)" in format_verdict(v, "perfect")


def test_perfect_head_on_an_unseparating_battery_is_cannot_determine():
    """Perfect AND unaffected by the action: the battery cannot separate them."""
    z0, a, _ = _battery()
    v = readiness_verdict(_IdentityHead(), z0, a, z0.clone())
    assert v.status == "cannot_determine"


def test_identity_predictor_mse_matches_definition():
    z0, _, z1 = _battery()
    assert identity_predictor_mse(z0, z1) == pytest.approx(
        float(((z1 - z0) ** 2).mean()), rel=1e-12)


def test_action_shuffle_refuses_single_class_rather_than_returning_one():
    z0, a, z1 = _battery(distinct=False)
    ratio, mse_true, mse_shuf, n_distinct = action_shuffle_ratio(
        _IdentityHead(), z0, a, z1)
    assert ratio is None, "a no-op permutation must not yield a ratio of 1.0"
    assert n_distinct == 1 and mse_shuf is None and math.isfinite(mse_true)


# --------------------------------------------------------------------------- #
# Canary -- read from the MODULE, never restated as a literal here.             #
# --------------------------------------------------------------------------- #
def test_canary_reproduces():
    res = check_canary()
    assert res["ok"], res
    assert res["n_seeds"] == len(CANARY_V3_EXQ_1073["seeds"]) == 3


def test_canary_corpus_is_non_vacuous():
    """Guard the derivation: an empty canary makes every assertion vacuously true."""
    assert len(CANARY_V3_EXQ_1073["battery_ratio"]) == 3
    assert len(CANARY_V3_EXQ_1073["skill_vs_identity"]) == 3
    assert all(isinstance(x, float) for x in CANARY_V3_EXQ_1073["battery_ratio"])
    assert CANARY_V3_EXQ_1073["source"].endswith("failure_autopsy_V3-EXQ-1073_2026-09-22.md")


def test_canary_values_match_the_landed_autopsy():
    assert CANARY_V3_EXQ_1073["battery_ratio"] == [0.760, 0.901, 0.881]
    assert CANARY_V3_EXQ_1073["skill_vs_identity"] == [-0.071, 0.227, -0.007]


# --------------------------------------------------------------------------- #
# BLIND-SPOT MEASUREMENT: build the defect, assert the guard FAILS on it.        #
# --------------------------------------------------------------------------- #
def test_blind_spot_reciprocal_ratio_would_be_caught():
    """If the ratio were computed upside-down, 1073's seeds would read 'ready'."""
    flipped = [1.0 / r for r in CANARY_V3_EXQ_1073["battery_ratio"]]
    got = ["action_blind" if (not (r > RATIO_FLOOR)) else "ready" for r in flipped]
    assert got != CANARY_V3_EXQ_1073["expected_status"], (
        "a reciprocal-ratio defect must change the canary's classification, "
        "otherwise the canary cannot catch it")
    assert got[0] == "ready"


def test_blind_spot_skill_only_gate_would_miss_seed_123():
    """Seed 123 has skill +0.227 (> 0) but ratio 0.901 (< 1). A gate that checked
    ONLY skill would call it ready. Both conjuncts are load-bearing."""
    skill_only = ["ready" if s > SKILL_FLOOR else "action_blind"
                  for s in CANARY_V3_EXQ_1073["skill_vs_identity"]]
    assert skill_only[1] == "ready"
    assert check_canary()["seeds"][1]["got"] == "action_blind"


def test_blind_spot_bool_collapse_would_lose_cannot_determine():
    """is_ready collapses three values into two: both action_blind and
    cannot_determine read False. The contract is that callers branch on status."""
    z0, a, z1 = _battery(distinct=False)
    v_cd = readiness_verdict(_IdentityHead(), z0, a, z1)
    z0b, ab, z1b = _battery()
    v_ab = readiness_verdict(_IdentityHead(), z0b, ab, z1b)
    assert v_cd.is_ready is False and v_ab.is_ready is False
    assert v_cd.status != v_ab.status, "the bool loses a distinction the status keeps"


# --------------------------------------------------------------------------- #
# SD-PP-B5 margin loss on E2FastPredictor.world_forward.                         #
# --------------------------------------------------------------------------- #
def _e2(**kw):
    return E2FastPredictor(E2Config(self_dim=8, world_dim=8, action_dim=4, **kw))


def test_margin_loss_is_zero_when_already_separated():
    e2 = _e2(world_interventional_margin=0.0)
    z = torch.randn(16, 8)
    a0 = torch.nn.functional.one_hot(torch.zeros(16, dtype=torch.long), 4).float()
    a1 = torch.nn.functional.one_hot(torch.ones(16, dtype=torch.long), 4).float()
    assert float(e2.compute_world_interventional_loss(z, a0, a1)) == pytest.approx(0.0)


def test_margin_loss_is_positive_for_an_action_invariant_head():
    """Zero the action path: predictions collapse together, loss -> margin."""
    e2 = _e2(world_interventional_margin=0.5)
    with torch.no_grad():
        e2.world_action_encoder.weight.zero_()
        e2.world_action_encoder.bias.zero_()
    z = torch.randn(16, 8)
    a0 = torch.nn.functional.one_hot(torch.zeros(16, dtype=torch.long), 4).float()
    a1 = torch.nn.functional.one_hot(torch.ones(16, dtype=torch.long), 4).float()
    assert float(e2.compute_world_interventional_loss(z, a0, a1)) == pytest.approx(
        0.5, abs=1e-4), "offset by sqrt(eps); see the epsilon note in e2_fast.py"


def test_margin_loss_gradient_vanishes_only_at_exact_collapse():
    """The honest contract. Zero distance is a MINIMUM of ||d||, so every smooth
    function of ||d|| is stationary there -- no epsilon or squared-margin variant
    escapes it. Assert BOTH halves: flat at exact collapse, live just off it."""
    def _probe(scale):
        e2 = _e2(world_interventional_margin=0.5)
        with torch.no_grad():
            e2.world_action_encoder.weight.mul_(scale)
            e2.world_action_encoder.bias.mul_(scale)
        z = torch.randn(16, 8)
        a0 = torch.nn.functional.one_hot(torch.zeros(16, dtype=torch.long), 4).float()
        a1 = torch.nn.functional.one_hot(torch.ones(16, dtype=torch.long), 4).float()
        e2.zero_grad(set_to_none=True)
        loss = e2.compute_world_interventional_loss(z, a0, a1)
        loss.backward()
        g = e2.world_action_encoder.weight.grad
        return float(loss), (0.0 if g is None else float(g.norm()))

    loss_zero, grad_zero = _probe(0.0)
    assert grad_zero == 0.0, (
        "documented property: an EXACTLY action-invariant head is a stationary "
        "point of this loss family")
    assert loss_zero == pytest.approx(0.5, abs=1e-4), (
        "the loss VALUE still reports full violation at collapse, so the "
        "condition remains observable even where the gradient is flat")

    loss_near, grad_near = _probe(1e-3)
    assert grad_near > 0.0, (
        "just off exact collapse the term must deliver gradient to the action "
        "encoder; if this is zero the loss is inert, not merely stationary")
    assert math.isfinite(grad_near), "epsilon must bound the d/||d|| blow-up"


def test_margin_loss_delivers_gradient_on_a_randomly_initialised_head():
    """The case that actually occurs: random init, untrained head."""
    torch.manual_seed(3)
    e2 = _e2(world_interventional_margin=0.5)
    z = torch.randn(16, 8)
    a0 = torch.nn.functional.one_hot(torch.zeros(16, dtype=torch.long), 4).float()
    a1 = torch.nn.functional.one_hot(torch.ones(16, dtype=torch.long), 4).float()
    e2.zero_grad(set_to_none=True)
    e2.compute_world_interventional_loss(z, a0, a1).backward()
    g = e2.world_action_encoder.weight.grad
    assert g is not None and float(g.norm()) > 0.0


def test_margin_uses_config_value_not_a_literal():
    z = torch.randn(16, 8)
    a0 = torch.nn.functional.one_hot(torch.zeros(16, dtype=torch.long), 4).float()
    a1 = torch.nn.functional.one_hot(torch.ones(16, dtype=torch.long), 4).float()
    for m in (0.25, 1.5):
        e2 = _e2(world_interventional_margin=m)
        with torch.no_grad():
            e2.world_action_encoder.weight.zero_()
            e2.world_action_encoder.bias.zero_()
        assert float(e2.compute_world_interventional_loss(z, a0, a1)) == pytest.approx(
            m, abs=1e-4)


# --------------------------------------------------------------------------- #
# Backward compatibility: every new default is no-op.                           #
# --------------------------------------------------------------------------- #
def test_new_config_defaults_are_no_op():
    c = E2Config()
    assert c.use_world_interventional is False
    assert c.world_interventional_fraction == 0.3
    assert c.world_interventional_margin == 0.1


def test_from_dims_threads_all_three_knobs():
    """MECH-307 shape: from_dims silently swallows unknown kwargs, so a knob with
    only a dataclass field runs OFF while the driver believes it ON."""
    on = REEConfig.from_dims(
        body_obs_dim=8, world_obs_dim=8, action_dim=4, self_dim=8, world_dim=8,
        use_world_interventional=True, world_interventional_fraction=0.75,
        world_interventional_margin=0.33)
    assert on.e2.use_world_interventional is True
    assert on.e2.world_interventional_fraction == 0.75
    assert on.e2.world_interventional_margin == 0.33
    off = REEConfig.from_dims(body_obs_dim=8, world_obs_dim=8, action_dim=4,
                              self_dim=8, world_dim=8)
    assert off.e2.use_world_interventional is False
    assert off.e2.world_interventional_margin == 0.1


def test_world_forward_unchanged_when_feature_off():
    """The head's forward pass must be bit-identical with the flag at default."""
    torch.manual_seed(7)
    a_cfg = E2Config(self_dim=8, world_dim=8, action_dim=4)
    torch.manual_seed(7)
    b_cfg = E2Config(self_dim=8, world_dim=8, action_dim=4,
                     use_world_interventional=False)
    torch.manual_seed(11); e2a = E2FastPredictor(a_cfg)
    torch.manual_seed(11); e2b = E2FastPredictor(b_cfg)
    z = torch.randn(8, 8)
    a = torch.nn.functional.one_hot(torch.arange(8) % 4, 4).float()
    assert torch.equal(e2a.world_forward(z, a), e2b.world_forward(z, a))


# --------------------------------------------------------------------------- #
# The canary must DRIVE the shipped path, not re-check constants against        #
# constants. Added 2026-09-22 after a red-team pass found the original          #
# check_canary re-implemented the comparison and therefore could not fail.      #
# --------------------------------------------------------------------------- #
def test_canary_drives_the_shipped_verdict_path():
    res = check_canary()
    assert res["ok"], res
    assert "readiness_verdict" in res["drives"]
    for s in res["seeds"]:
        assert s["faithful"], (
            "the synthetic battery must really carry the pinned numbers, or the "
            "classification is about something else: %r" % s)
        assert s["measured_ratio"] == pytest.approx(s["ratio"], abs=1e-3)
        assert s["measured_skill"] == pytest.approx(s["skill"], abs=1e-3)


def test_blind_spot_canary_catches_a_reciprocal_ratio():
    """Measure the blind spot: construct the defect, assert the guard FAILS."""
    import experiments._lib.action_sensitivity_gate as g
    original = g.battery_pair_ratio
    try:
        def flipped(head, orig, cf):
            r, a, b = original(head, orig, cf)
            return ((1.0 / r) if r not in (None, 0) else r), a, b
        g.battery_pair_ratio = flipped
        assert g.check_canary()["ok"] is False, (
            "a reciprocal-ratio defect must fail the canary; if it does not, the "
            "canary is re-checking constants rather than driving the real path")
    finally:
        g.battery_pair_ratio = original
    assert g.check_canary()["ok"] is True, "canary must recover after the patch"


def test_blind_spot_canary_catches_an_inverted_skill_formula():
    import experiments._lib.action_sensitivity_gate as g
    original = g.skill_vs_identity
    try:
        g.skill_vs_identity = lambda m, i: None if not i else (m / i) - 1.0
        assert g.check_canary()["ok"] is False
    finally:
        g.skill_vs_identity = original
    assert g.check_canary()["ok"] is True


def test_both_conjuncts_are_live_on_the_pinned_set():
    """Seed 123 is the ONE pinned seed where ratio and skill disagree (skill
    +0.227 > 0 but ratio 0.901 < 1), so it is what proves the conjunction is
    load-bearing rather than one bar carrying both."""
    from experiments._lib.action_sensitivity_gate import _synthetic_battery
    head, orig, cf = _synthetic_battery(1.705e-05, 0.227, 0.901)
    assert readiness_verdict(head, *orig, counterfactual_battery=cf
                             ).status == "action_blind"
    assert readiness_verdict(head, *orig, counterfactual_battery=cf,
                             ratio_floor=-1e9).status == "ready", (
        "removing the ratio bar must flip seed 123; if it does not, the ratio "
        "conjunct is inert on the pinned set and the canary cannot guard it")


def test_canary_identity_mse_corpus_is_non_vacuous():
    assert len(CANARY_V3_EXQ_1073["identity_mse"]) == 3
    assert all(0.0 < v < 1e-3 for v in CANARY_V3_EXQ_1073["identity_mse"])
