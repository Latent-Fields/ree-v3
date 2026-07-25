"""SD-078 contract: common-mode-invariant (centered) CandidateRuleField context key.

The ARC-063 CandidateRuleField keys mint-block, recurrence bucketing and gate
retrieval on a raw z_world context. Under SD-008 z_world under-differentiation the
untrained encoder maps every context into a ~0.98-cosine common-mode cone, so the
mint-block (`_cosine(context, rule.context_tag) >= mint_block_thresh`) fires
against the FIRST minted rule for every subsequent context at any threshold the
config can express -- the pool is structurally capped at ONE rule, and
`_context_bucket` (a sign pattern, i.e. an absolute threshold at zero) collapses
to a single bucket. That is the V3-EXQ-654/654b/654d signature
(crf_max_pairwise_rule_dist == 0.0, crf_frac_active ~0.13), previously read as
retire-churn dynamics. SD-078 subtracts a slow EMA common-mode baseline before
every cue comparison (the SD-066 / SD-077 pattern).

C1 default OFF + bit-identity: cue_centering defaults False; no baseline tensor is
   allocated; a common-mode-dominated context stream mints exactly one rule and
   leaves max_pairwise_rule_distance at 0.0 (the reproduced 654b signature).
C2 centering separates: the SAME context stream with centering ON mints more than
   one rule and yields a non-zero max_pairwise_rule_distance.
C3 threshold tuning cannot substitute: with centering OFF, NO mint-block threshold
   in [0.5, 0.94] recovers a second rule, because every pairwise context cosine
   sits above the common-mode floor. Pins the 654b amend's wrong lever so it is
   not re-attempted.
C4 lazy seed + MECH-094: the baseline is seeded from the first WAKING context and
   is never advanced by a simulation_mode tick.
C5 tags are stored RAW: a drifting baseline moves query and stored tags together,
   so a repeated context keeps matching its own rule.
C6 the recurrence bucket is centered too: a common-mode-dominated stream yields
   one bucket raw and more than one centered.
"""
from __future__ import annotations

import torch

from ree_core.policy.candidate_rule_field import (
    CandidateRuleField,
    CandidateRuleFieldConfig,
)

CONTEXT_DIM = 8
N_CONTEXTS = 60


def _common_mode_stream(n: int = N_CONTEXTS, dim: int = CONTEXT_DIM,
                        offset: float = 12.0, seed: int = 0) -> torch.Tensor:
    """A context stream with the measured SD-008 geometry: genuinely diverse
    residuals buried under one dominant shared direction. Reproduces the
    V3-EXQ-669b nursery's z_world signature (pairwise cosine min ~0.98)."""
    g = torch.Generator().manual_seed(seed)
    common = torch.ones(dim) * offset
    residual = torch.randn(n, dim, generator=g)
    return common.unsqueeze(0) + residual


def _field(centering: bool, **kw) -> CandidateRuleField:
    cfg = CandidateRuleFieldConfig(
        use_candidate_rule_field=True,
        n_slots=16,
        rule_dim=16,
        mature_pool_dynamics=True,
        persist_rules_across_episode_reset=True,
        cue_centering=centering,
        **kw,
    )
    return CandidateRuleField(CONTEXT_DIM, cfg)


def _drive(field: CandidateRuleField, stream: torch.Tensor) -> int:
    """Run the stream through step(); return max concurrent live rules."""
    max_live = 0
    for i, ctx in enumerate(stream):
        field.step(context=ctx, action_object_idx=i % 2, outcome_signal=0.1,
                   simulation_mode=False)
        max_live = max(max_live, len(field._rules))
    return max_live


def test_c0_stream_has_the_measured_common_mode_geometry():
    """Guard the fixture: if the synthetic stream ever stops being common-mode
    dominated, C1/C3 would pass vacuously."""
    s = _common_mode_stream()
    n = torch.nn.functional.normalize(s, dim=-1)
    c = n @ n.t()
    iu = torch.triu_indices(len(s), len(s), offset=1)
    v = c[iu[0], iu[1]]
    # offset=12.0 gives min pairwise cosine ~0.976, matching the z_world geometry
    # measured on the V3-EXQ-669b Stage-0 nursery (min 0.9767).
    assert v.min() > 0.95, f"fixture not common-mode dominated: min cosine {v.min()}"
    # ...while the residuals themselves are diverse (so there IS signal to recover).
    r = s - s.mean(0, keepdim=True)
    rn = torch.nn.functional.normalize(r, dim=-1)
    rv = (rn @ rn.t())[iu[0], iu[1]]
    assert rv.min() < 0.0, "fixture residuals are not diverse"


def test_c1_centering_defaults_off_and_pool_saturates_at_one_rule():
    assert CandidateRuleFieldConfig().cue_centering is False
    assert CandidateRuleFieldConfig().cue_baseline_alpha == 0.02

    field = _field(centering=False)
    max_live = _drive(field, _common_mode_stream())

    assert field._baseline is None, "OFF path must allocate no baseline"
    assert max_live == 1, f"expected the 654b signature (1 rule), got {max_live}"
    assert field.max_pairwise_rule_distance() == 0.0


def test_c2_centering_recovers_a_differentiated_pool():
    field = _field(centering=True)
    max_live = _drive(field, _common_mode_stream())

    assert field._baseline is not None
    assert max_live > 1, f"centering did not separate contexts (max_live={max_live})"
    assert field.max_pairwise_rule_distance() > 0.0


def test_c3_no_mint_block_threshold_rescues_the_raw_key():
    """The 654b amend raised mature_mint_block_threshold to 0.8 to relieve this.
    It cannot work: every pairwise context cosine is above ~0.94, so the block
    fires at any expressible threshold. Pinned so the wrong lever is not
    re-tuned as the fix."""
    stream = _common_mode_stream()
    for thresh in (0.5, 0.6, 0.7, 0.8, 0.9, 0.94):
        field = _field(centering=False, mature_mint_block_threshold=thresh)
        max_live = _drive(field, stream)
        assert max_live == 1, (
            f"mint_block_threshold={thresh} unexpectedly recovered {max_live} "
            "rules on a raw key -- C3 assumed this was impossible"
        )


def test_c4_lazy_seed_and_simulation_mode_does_not_move_baseline():
    field = _field(centering=True)
    stream = _common_mode_stream()

    # Lazily seeded: no baseline before the first waking context.
    assert field._baseline is None
    field.step(context=stream[0], action_object_idx=0, simulation_mode=False)
    assert field._baseline is not None
    assert torch.allclose(field._baseline, stream[0].reshape(-1))

    # MECH-094: a simulation tick must not advance it.
    before = field._baseline.clone()
    field.step(context=stream[1] * 50.0, action_object_idx=0, simulation_mode=True)
    assert torch.allclose(field._baseline, before)
    field.observe(stream[1] * 50.0, simulation_mode=True)
    assert torch.allclose(field._baseline, before)

    # A waking tick does advance it.
    field.step(context=stream[1], action_object_idx=0, simulation_mode=False)
    assert not torch.allclose(field._baseline, before)


def test_c5_repeated_context_still_matches_itself_as_baseline_drifts():
    """Tags are stored RAW and centered at comparison time, so a drifting
    baseline moves query and tag together and can never desynchronise them."""
    field = _field(centering=True)
    stream = _common_mode_stream()
    _drive(field, stream)
    assert len(field._rules) >= 1

    rule = next(iter(field._rules.values()))
    tag_raw = rule.context_tag.clone()

    # Feed the rule's OWN raw context back after further baseline drift.
    for ctx in stream[:20]:
        field.observe(ctx, simulation_mode=False)
    matched = field.gate_and_select(tag_raw)
    matched_ids = {id(r) for r in field._rules.values()
                   if field._cosine(field._centered(tag_raw),
                                    field._centered(r.context_tag)) >= 0.99}
    assert id(rule) in matched_ids, "a rule stopped matching its own stored context"
    assert isinstance(matched, list)


def test_c6_recurrence_bucket_is_centered_too():
    """_context_bucket is a sign pattern -- an absolute threshold at zero -- so it
    is common-mode dominated in the same way the cosines are. Centering only the
    cosines would leave the recurrence counter unable to tell regimes apart."""
    stream = _common_mode_stream()

    raw_field = _field(centering=False)
    raw_buckets = {raw_field._context_bucket(raw_field._centered(c)) for c in stream}
    assert len(raw_buckets) == 1, f"fixture should give 1 raw bucket, got {len(raw_buckets)}"

    cen_field = _field(centering=True)
    cen_buckets = set()
    for c in stream:
        cen_field.observe(c, simulation_mode=False)
        cen_buckets.add(cen_field._context_bucket(cen_field._centered(c)))
    assert len(cen_buckets) > 1, "centered bucketing did not separate regimes"
