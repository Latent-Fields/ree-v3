"""Contract tests for the MECH-048 mu/kappa stability overlays on the mode prior.

MECH-048 asserts that opponent stability overlays modulate BOTH mode-prior
sharpness (entropy) AND switching pressure. On the SalienceCoordinator mode
plane only the second half was built: pcc_stability (the mu-analogue) entered
the effective_threshold multiplier and nothing else, and
`affinity_weights["pcc_stability"]` was empty. Consequence, measured by live
execution on 2026-07-21 (session `practical-volhard-2c1876`):

    H(operating_mode) was EXACTLY invariant under pcc_stability -- 0.167605 at
    pcc_stability 0.0, 1.0 AND 3.0, identical to 6 decimal places.

So any experiment measuring mode entropy against mu would have reported an
arithmetic zero rather than a measurement. That is the DV-symmetry vacuity
class documented in failure_autopsy_V3-EXQ-604c_2026-07-20.md section 3: a
manipulation invariant under the DV's own symmetry, tested against that DV.
MECH-048's entropy half was untestable by construction, on every substrate and
every seed.

The build (2026-07-21) gives mu genuine authority over the mode PRIOR via the
softmax temperature, in the literal form specified by
docs/thoughts/2026-02-11_some_control_plane_maths_hypotheses.md:63:

    tau = softmax_temperature * exp(alpha_kappa * kappa - alpha_mu * mu)

with mu = pcc_stability (SD-032d) and kappa = aic_salience (SD-032c). It is
deliberately NOT an affinity_weights entry: that would make mu a per-mode logit
bias, i.e. a mode PREFERENCE, and control_plane.md#mech-048 states the overlays
"are not scalar reward signals; they act as stability and entropy modulators".

Contracts:
  C1. OFF (default) -- H(operating_mode) is invariant under pcc_stability, and
      effective_temperature is softmax_temperature exactly. Pins the legacy
      behaviour so the flag stays a true no-op.
  C2. OFF (default) -- the threshold half still responds to mu (that half was
      never broken and must not regress).
  C3. ON -- H(operating_mode) is STRICTLY DECREASING in pcc_stability. This is
      the contract the build exists to establish: the entropy DV is no longer
      an arithmetic zero.
  C4. ON -- effective_temperature matches the closed-form spec expression.
  C5. ON -- the kappa leg flattens (raises entropy), opposing mu, and is off by
      default so that the master switch isolates the mu leg.
  C6. ON -- the exponent clip holds at an unbounded kappa (aic_salience has no
      upper bound), so exp() cannot overflow or produce a degenerate prior.
  C7. Agent-level wiring -- REEConfig -> SalienceCoordinatorConfig passthrough,
      including that the default REEConfig leaves the feature off.

NOTE ON WHAT THESE TESTS ASSERT. Every assertion here is on a continuous
readout -- entropy, temperature, threshold. None touches a sampled discrete
mode. torch.multinomial returns a different category on linux-x86_64/torch
2.11 than on darwin-arm64/torch 2.10 from a bit-identical probability tensor at
the same seed (see the "Running the test suite" section of REE_Working/CLAUDE.md
and the memory reference-cross-machine-class-contract-divergence), so an
assertion on a discrete mode sequence would be flaky across machine classes.

NOTE ON SCIENTIFIC SCOPE. C3 is a SUBSTRATE-WIRING contract, not evidence for
MECH-048. With mu injected directly, "entropy falls as mu rises" is an
arithmetic identity of the temperature coupling -- exactly the vacuity that
sank the V3-EXQ-683 design. A validation experiment must drive mu from upstream
environmental state (safety / coherence / task success, via PCCAnalog) and read
a downstream behavioural DV. See the queue entry for the validation run.
"""

from math import exp, isclose

import pytest

from ree_core.cingulate.salience_coordinator import (
    SalienceCoordinator,
    SalienceCoordinatorConfig,
)


MU_LADDER = (0.0, 0.25, 0.5, 0.75, 1.0)


def _tick(config, mu, kappa=1.0):
    """One tick on a fresh coordinator at a given mu / kappa."""
    coordinator = SalienceCoordinator(config)
    return coordinator.tick(
        dacc_bundle={"pe": 0.5},
        drive_level=0.3,
        extra_signals={"pcc_stability": mu, "aic_salience": kappa},
    )


# -- C1 / C2: OFF path is unchanged --------------------------------------


def test_c1_off_mode_entropy_invariant_under_mu():
    """Default config: entropy does not move with mu, temperature is the base.

    This is the DEFECT being pinned, not a desirable property -- it is asserted
    so that the new flag is provably a no-op when off.
    """
    config = SalienceCoordinatorConfig()
    assert config.use_stability_temperature is False, "feature must default off"

    entropies = [_tick(config, mu)["mode_entropy"] for mu in MU_LADDER]
    temperatures = [_tick(config, mu)["effective_temperature"] for mu in MU_LADDER]

    for value in entropies[1:]:
        assert isclose(value, entropies[0], rel_tol=0.0, abs_tol=1e-12), (
            "OFF path must leave H(operating_mode) exactly invariant under mu; "
            "got %r" % (entropies,)
        )
    for value in temperatures:
        assert value == pytest.approx(config.softmax_temperature, abs=1e-12)


def test_c2_off_switch_threshold_still_scales_with_mu():
    """The switching-pressure half was never broken and must not regress."""
    config = SalienceCoordinatorConfig()
    thresholds = [_tick(config, mu)["effective_threshold"] for mu in MU_LADDER]

    for lower, higher in zip(thresholds, thresholds[1:]):
        assert higher > lower, (
            "effective_threshold must increase with mu (harder to switch when "
            "the regime is stable); got %r" % (thresholds,)
        )


# -- C3 / C4: ON path gives mu authority over the prior -------------------


def test_c3_on_mode_entropy_strictly_decreases_with_mu():
    """THE contract this build exists to establish.

    With the overlay on, H(operating_mode) must actually move with mu -- and
    move DOWN, since MECH-048's mu-analogue "increases commitment stability"
    and the source spec reads mu as a sharpening term.
    """
    config = SalienceCoordinatorConfig(use_stability_temperature=True)
    entropies = [_tick(config, mu)["mode_entropy"] for mu in MU_LADDER]

    for lower_mu_h, higher_mu_h in zip(entropies, entropies[1:]):
        assert higher_mu_h < lower_mu_h, (
            "H(operating_mode) must strictly DECREASE as mu rises; got %r"
            % (entropies,)
        )

    # And the effect must be substantive, not a rounding artefact: the whole
    # point is that the DV is no longer an arithmetic zero.
    assert entropies[0] - entropies[-1] > 0.05, (
        "mu's authority over the mode prior must be measurable, not marginal; "
        "span was %r" % (entropies[0] - entropies[-1],)
    )


def test_c4_on_effective_temperature_matches_spec_form():
    """tau = tau_0 * exp(alpha_kappa * kappa - alpha_mu * mu)."""
    config = SalienceCoordinatorConfig(
        use_stability_temperature=True,
        temperature_mu_alpha=0.8,
        temperature_kappa_alpha=0.3,
        softmax_temperature=1.5,
    )
    mu, kappa = 0.7, 2.0
    expected = 1.5 * exp(0.3 * kappa - 0.8 * mu)

    observed = _tick(config, mu, kappa=kappa)["effective_temperature"]
    assert observed == pytest.approx(expected, rel=1e-12)


# -- C5: the kappa leg is the opponent, and is opt-in ---------------------


def test_c5_kappa_leg_opposes_mu_and_is_off_by_default():
    config_default = SalienceCoordinatorConfig(use_stability_temperature=True)
    assert config_default.temperature_kappa_alpha == 0.0, (
        "the master switch must isolate the mu leg -- kappa already reaches "
        "the mode logits via affinity_weights, so enabling both at once "
        "confounds two paths"
    )

    config_both = SalienceCoordinatorConfig(
        use_stability_temperature=True, temperature_kappa_alpha=1.0
    )
    # At fixed mu, raising kappa must FLATTEN the prior (re-evaluation
    # pressure), the opposite of mu's sharpening.
    low_kappa = _tick(config_both, 0.5, kappa=0.0)
    high_kappa = _tick(config_both, 0.5, kappa=2.0)

    assert high_kappa["effective_temperature"] > low_kappa["effective_temperature"]
    assert high_kappa["mode_entropy"] > low_kappa["mode_entropy"], (
        "kappa must raise mode entropy at fixed mu (destabilise / re-evaluate)"
    )


# -- C6: numerical guard --------------------------------------------------


def test_c6_exponent_clip_bounds_temperature_at_unbounded_kappa():
    """aic_salience is unbounded above (urgency_scaled + extra_sum)."""
    config = SalienceCoordinatorConfig(
        use_stability_temperature=True,
        temperature_kappa_alpha=1.0,
        temperature_exponent_clip=4.0,
    )
    out = _tick(config, 0.0, kappa=1e6)
    assert out["effective_temperature"] == pytest.approx(exp(4.0), rel=1e-9)

    # Symmetric on the sharpening side.
    config_sharp = SalienceCoordinatorConfig(
        use_stability_temperature=True,
        temperature_mu_alpha=1e6,
        temperature_exponent_clip=4.0,
    )
    out_sharp = _tick(config_sharp, 1.0, kappa=0.0)
    assert out_sharp["effective_temperature"] == pytest.approx(exp(-4.0), rel=1e-9)
    # A clipped temperature is still a valid distribution.
    assert out_sharp["mode_entropy"] >= 0.0


# -- C7: config passthrough ----------------------------------------------


def test_c7_reeconfig_defaults_leave_the_feature_off():
    from ree_core.utils.config import REEConfig

    for field_name, expected in (
        ("salience_use_stability_temperature", False),
        ("salience_temperature_mu_alpha", 1.0),
        ("salience_temperature_kappa_alpha", 0.0),
        ("salience_temperature_exponent_clip", 4.0),
    ):
        assert hasattr(REEConfig, field_name), "missing REEConfig.%s" % field_name
        assert getattr(REEConfig, field_name) == expected, (
            "REEConfig.%s default changed -- backward compatibility of every "
            "existing experiment depends on it" % field_name
        )
