"""MECH-205 pe_surprise_threshold default is not vacuous.

GFLAG-0075 (2026-09-01): the shipped default of 0.001 was ~53x too high
against an observed mean surprise of 1.86e-5, gating surprise_gated_replay
closed on every step. V3-EXQ-258a FAILed vacuously as a direct result
(root-caused on MECH-205's evidence_quality_note); V3-EXQ-258b PASSED using
1e-5, but that value was never promoted to the default. This pins the
promoted default so it cannot silently regress back to 0.001.
"""

from ree_core.utils.config import REEConfig

_OBSERVED_MEAN_SURPRISE = 1.86e-5


def test_pe_surprise_threshold_default_is_promoted_value():
    cfg = REEConfig()

    assert cfg.pe_surprise_threshold == 1e-5


def test_pe_surprise_threshold_default_not_vacuous_against_observed_surprise():
    cfg = REEConfig()

    assert cfg.pe_surprise_threshold <= _OBSERVED_MEAN_SURPRISE


def test_from_dims_pe_surprise_threshold_default_matches_field_default():
    cfg = REEConfig.from_dims(
        body_obs_dim=12,
        world_obs_dim=54,
        action_dim=4,
        self_dim=16,
        world_dim=16,
    )

    assert cfg.pe_surprise_threshold == REEConfig().pe_surprise_threshold
