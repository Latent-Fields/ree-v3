"""Contract: the propose diagnostics expose an INJECTION-FREE action-object
decoder centroid alongside the headline one.

WHY THIS EXISTS (2026-09-07)
----------------------------
`action_object_decoder_raw_output_stats` is computed over the FINAL candidate
pool, which by then may contain synthetic scaffolds spliced in by
`_inject_support_preserving_candidates` (tag `support_preserving_cem_injected`)
or by the action-class scaffold (tag `action_class_scaffold`). Both are hand-built
one-hot-then-exact-zeros rows, not samples from the proposal distribution.

Each such row drags one action dimension's mean by ~1/(num_candidates * horizon)
-- an order of magnitude above the centroid DISPLACEMENTS some drivers read as
their DV. And the support-preserving injector fires CONDITIONALLY on the pool's
first-action class count, so it is not arm-independent for any design whose arms
change elite selection: it can fire asymmetrically between two arms of a matched
pair and manufacture an apparent displacement.

These contracts pin that (a) tagged synthetic rows are excluded from the
companion fields, and (b) the companion differs from the headline EXACTLY when
the corresponding injector fired.

PORTABILITY: every assertion here is on continuous statistics, metadata tags, or
counts derived from the pool the call actually returned -- never on an exact
committed action sequence. `torch.multinomial` returns a different category on
linux-x86_64 than on darwin-arm64 from a bit-identical probability tensor
(CLAUDE.md, "Running the test suite"), so a discrete-action oracle would be a
cross-machine-class flake.
"""

from __future__ import annotations

import torch


SP_INJECTED_SOURCE = "support_preserving_cem_injected"
SCAFFOLD_SOURCE = "action_class_scaffold"


def _make_module(
    use_scaffold: bool = False,
    use_support_preserving_cem: bool = False,
):
    from ree_core.hippocampal.module import HippocampalModule
    from ree_core.predictors.e2_fast import E2Config, E2FastPredictor
    from ree_core.residue.field import ResidueConfig, ResidueField
    from ree_core.utils.config import HippocampalConfig

    cfg = HippocampalConfig(
        world_dim=8,
        action_dim=4,
        action_object_dim=8,
        hidden_dim=32,
        horizon=4,
        num_candidates=16,
        num_cem_iterations=2,
        elite_fraction=0.25,
        use_action_class_scaffold_candidates=use_scaffold,
        use_support_preserving_cem=use_support_preserving_cem,
    )
    e2 = E2FastPredictor(E2Config(
        self_dim=8,
        world_dim=8,
        action_dim=4,
        action_object_dim=8,
        hidden_dim=32,
    ))
    residue = ResidueField(ResidueConfig(
        world_dim=8,
        hidden_dim=32,
        num_basis_functions=8,
    ))
    return HippocampalModule(cfg, e2=e2, residue_field=residue)


def _collapsed_decode_for(module):
    """Force the decoder onto a single class, which is what makes the
    support-preserving injector fire. Deterministic -- no sampling, so no
    cross-machine-class divergence."""

    def collapsed_decode(action_objects):
        batch, horizon, _ao_dim = action_objects.shape
        actions = torch.zeros(
            batch,
            horizon,
            module.config.action_dim,
            device=action_objects.device,
            dtype=action_objects.dtype,
        )
        actions[..., 3] = 1.0
        return actions

    return collapsed_decode


def _sources(candidates):
    return [(getattr(c, "metadata", None) or {}).get("source") for c in candidates]


def _independent_mean_std(candidates, exclude_sources):
    """Oracle computed WITHOUT the module's own summarizer.

    Deliberately re-derives the statistic from raw torch rather than calling
    `_summarize_action_tensor` / `_summarize_trajectories_excluding_sources` --
    an oracle taken from the function under test would be vacuous.
    """
    excluded = set(exclude_sources)
    kept = [
        c for c in candidates
        if ((getattr(c, "metadata", None) or {}).get("source")) not in excluded
    ]
    if not kept:
        return [], []
    stacked = torch.stack([c.actions.detach() for c in kept], dim=0)
    flat = stacked.reshape(-1, stacked.shape[-1])
    return (
        [float(v) for v in flat.mean(dim=0).tolist()],
        [float(v) for v in flat.std(dim=0, unbiased=False).tolist()],
    )


def _max_abs_delta(a, b):
    assert len(a) == len(b)
    return max((abs(x - y) for x, y in zip(a, b)), default=0.0)


# --------------------------------------------------------------------------- #
# (a) injected trajectories are excluded from the companion fields             #
# --------------------------------------------------------------------------- #

def test_injected_scaffolds_are_excluded_from_companion_centroid():
    torch.manual_seed(19)
    module = _make_module(use_support_preserving_cem=True)
    module._decode_action_objects = _collapsed_decode_for(module)  # type: ignore[method-assign]

    candidates = module.propose_trajectories(
        z_world=torch.zeros(1, 8),
        z_self=torch.zeros(1, 8),
    )
    diags = module.get_last_propose_diagnostics()

    # Precondition: the injector actually fired, otherwise this test is vacuous.
    assert diags["support_preserving_active"] is True
    n_injected = int(diags["support_preserving_injected_candidates"])
    assert n_injected >= 1
    assert _sources(candidates).count(SP_INJECTED_SOURCE) == n_injected

    ex_injected = diags["action_object_decoder_raw_output_stats_excluding_injected"]

    # The companion equals an INDEPENDENTLY recomputed centroid over the pool
    # with the injected rows dropped.
    exp_mean, exp_std = _independent_mean_std(candidates, (SP_INJECTED_SOURCE,))
    assert ex_injected["mean_by_action_dim"] == exp_mean
    assert ex_injected["std_by_action_dim"] == exp_std

    # Denominator agrees with the number of rows actually kept.
    assert diags["candidate_samples_excluding_injected"] == (
        len(candidates) - n_injected
    )
    assert diags["candidate_samples_collected"] == len(candidates)


# --------------------------------------------------------------------------- #
# (b) the two fields differ EXACTLY when injection fired                       #
# --------------------------------------------------------------------------- #

def test_companion_differs_from_headline_when_injection_fired():
    torch.manual_seed(19)
    module = _make_module(use_support_preserving_cem=True)
    module._decode_action_objects = _collapsed_decode_for(module)  # type: ignore[method-assign]

    module.propose_trajectories(z_world=torch.zeros(1, 8), z_self=torch.zeros(1, 8))
    diags = module.get_last_propose_diagnostics()

    assert int(diags["support_preserving_injected_candidates"]) >= 1

    headline = diags["action_object_decoder_raw_output_stats"]
    companion = diags["action_object_decoder_raw_output_stats_excluding_injected"]

    # Relational, not a pinned magnitude: the pool's exact membership is allowed
    # to vary, the CONTAMINATION is not.
    assert _max_abs_delta(
        headline["mean_by_action_dim"], companion["mean_by_action_dim"]
    ) > 0.0


def test_companion_equals_headline_when_injection_did_not_fire():
    torch.manual_seed(19)
    module = _make_module(use_support_preserving_cem=False)
    module._decode_action_objects = _collapsed_decode_for(module)  # type: ignore[method-assign]

    candidates = module.propose_trajectories(
        z_world=torch.zeros(1, 8),
        z_self=torch.zeros(1, 8),
    )
    diags = module.get_last_propose_diagnostics()

    assert int(diags["support_preserving_injected_candidates"]) == 0
    assert SP_INJECTED_SOURCE not in _sources(candidates)

    headline = diags["action_object_decoder_raw_output_stats"]
    companion = diags["action_object_decoder_raw_output_stats_excluding_injected"]
    assert companion == headline
    assert diags["candidate_samples_excluding_injected"] == len(candidates)


def test_biconditional_holds_across_both_configurations():
    """The 'exactly when' half stated as one assertion over both arms, so a
    future change that makes the companion diverge unconditionally fails here
    even if the two single-arm tests above were adjusted."""
    observed = []
    for use_sp in (False, True):
        torch.manual_seed(19)
        module = _make_module(use_support_preserving_cem=use_sp)
        module._decode_action_objects = _collapsed_decode_for(module)  # type: ignore[method-assign]
        module.propose_trajectories(
            z_world=torch.zeros(1, 8),
            z_self=torch.zeros(1, 8),
        )
        diags = module.get_last_propose_diagnostics()
        fired = int(diags["support_preserving_injected_candidates"]) > 0
        differs = (
            diags["action_object_decoder_raw_output_stats_excluding_injected"]
            != diags["action_object_decoder_raw_output_stats"]
        )
        observed.append((use_sp, fired, differs))

    for use_sp, fired, differs in observed:
        assert fired == differs, (
            f"use_support_preserving_cem={use_sp}: injector fired={fired} but "
            f"companion-differs-from-headline={differs}; the companion must "
            f"diverge exactly when the injector fired"
        )
    # Both arms must be exercised, or the biconditional is vacuously satisfied.
    assert {fired for _sp, fired, _d in observed} == {False, True}


# --------------------------------------------------------------------------- #
# The action-class scaffold is the SAME synthetic construct under another tag   #
# --------------------------------------------------------------------------- #

def test_excluding_injected_does_not_filter_the_action_class_scaffold():
    """Documents the trap the `_excluding_synthetic` field exists to close: with
    the scaffold on and the injector off, `_excluding_injected` is STILL
    contaminated. A driver reading it as 'the clean centroid' would be wrong."""
    torch.manual_seed(13)
    module = _make_module(use_scaffold=True, use_support_preserving_cem=False)
    candidates = module.propose_trajectories(
        z_world=torch.zeros(1, 8),
        z_self=torch.zeros(1, 8),
    )
    diags = module.get_last_propose_diagnostics()

    n_scaffold = int(diags["action_class_scaffold_candidates_added"])
    assert n_scaffold >= 1
    assert _sources(candidates).count(SCAFFOLD_SOURCE) == n_scaffold
    assert int(diags["support_preserving_injected_candidates"]) == 0

    headline = diags["action_object_decoder_raw_output_stats"]
    ex_injected = diags["action_object_decoder_raw_output_stats_excluding_injected"]
    ex_synthetic = diags["action_object_decoder_raw_output_stats_excluding_synthetic"]

    # Injector did not fire -> the injected-only companion is the headline.
    assert ex_injected == headline
    # ... and is therefore still carrying the scaffold rows.
    assert _max_abs_delta(
        ex_injected["mean_by_action_dim"], ex_synthetic["mean_by_action_dim"]
    ) > 0.0

    exp_mean, exp_std = _independent_mean_std(
        candidates, (SP_INJECTED_SOURCE, SCAFFOLD_SOURCE)
    )
    assert ex_synthetic["mean_by_action_dim"] == exp_mean
    assert ex_synthetic["std_by_action_dim"] == exp_std
    assert diags["candidate_samples_excluding_synthetic"] == (
        len(candidates) - n_scaffold
    )


def test_excluding_synthetic_drops_both_tags_together():
    torch.manual_seed(19)
    module = _make_module(use_scaffold=True, use_support_preserving_cem=True)
    module._decode_action_objects = _collapsed_decode_for(module)  # type: ignore[method-assign]

    candidates = module.propose_trajectories(
        z_world=torch.zeros(1, 8),
        z_self=torch.zeros(1, 8),
    )
    diags = module.get_last_propose_diagnostics()

    sources = _sources(candidates)
    n_synthetic = (
        sources.count(SP_INJECTED_SOURCE) + sources.count(SCAFFOLD_SOURCE)
    )
    assert diags["candidate_samples_excluding_synthetic"] == (
        len(candidates) - n_synthetic
    )

    exp_mean, exp_std = _independent_mean_std(
        candidates, (SP_INJECTED_SOURCE, SCAFFOLD_SOURCE)
    )
    ex_synthetic = diags["action_object_decoder_raw_output_stats_excluding_synthetic"]
    assert ex_synthetic["mean_by_action_dim"] == exp_mean
    assert ex_synthetic["std_by_action_dim"] == exp_std


# --------------------------------------------------------------------------- #
# The headline field's meaning must not drift -- landed manifests read it       #
# --------------------------------------------------------------------------- #

def test_headline_field_still_covers_the_whole_pool():
    torch.manual_seed(19)
    module = _make_module(use_support_preserving_cem=True)
    module._decode_action_objects = _collapsed_decode_for(module)  # type: ignore[method-assign]

    candidates = module.propose_trajectories(
        z_world=torch.zeros(1, 8),
        z_self=torch.zeros(1, 8),
    )
    diags = module.get_last_propose_diagnostics()

    exp_mean, exp_std = _independent_mean_std(candidates, ())
    headline = diags["action_object_decoder_raw_output_stats"]
    assert headline["mean_by_action_dim"] == exp_mean
    assert headline["std_by_action_dim"] == exp_std


def test_injection_firing_is_visible_at_top_level_of_the_diagnostics():
    """`support_preserving_injected_candidates` must stay reachable without
    digging into a nested dict -- it is the flag a consumer needs to know
    whether the headline centroid can be trusted at all."""
    torch.manual_seed(19)
    module = _make_module(use_support_preserving_cem=True)
    module._decode_action_objects = _collapsed_decode_for(module)  # type: ignore[method-assign]
    module.propose_trajectories(z_world=torch.zeros(1, 8), z_self=torch.zeros(1, 8))
    diags = module.get_last_propose_diagnostics()

    for key in (
        "support_preserving_active",
        "support_preserving_injected_candidates",
        "support_preserving_injected_classes",
        "support_preserving_replaced_candidates",
        "action_class_scaffold_candidates_added",
        "action_object_decoder_raw_output_stats_excluding_injected",
        "action_object_decoder_raw_output_stats_excluding_synthetic",
        "candidate_samples_excluding_injected",
        "candidate_samples_excluding_synthetic",
    ):
        assert key in diags, f"{key} must be top-level in propose diagnostics"
