"""Contracts for hippocampal candidate-support diagnostics and repair flags."""

from __future__ import annotations

import torch


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


def _first_classes(candidates):
    return [
        int(c.actions[:, 0, :].argmax(dim=-1).flatten()[0].item())
        for c in candidates
    ]


def test_config_exposes_scaffold_default_off():
    from ree_core.utils.config import HippocampalConfig, REEConfig

    cfg = HippocampalConfig()
    assert hasattr(cfg, "use_action_class_scaffold_candidates")
    assert cfg.use_action_class_scaffold_candidates is False
    assert hasattr(cfg, "use_support_preserving_cem")
    assert cfg.use_support_preserving_cem is False
    assert cfg.support_preserving_min_first_action_classes == 2

    master = REEConfig.from_dims(
        body_obs_dim=4,
        world_obs_dim=8,
        action_dim=4,
        self_dim=8,
        world_dim=8,
        use_action_class_scaffold_candidates=True,
        use_support_preserving_cem=True,
        support_preserving_min_first_action_classes=3,
    )
    assert master.hippocampal.use_action_class_scaffold_candidates is True
    assert master.hippocampal.use_support_preserving_cem is True
    assert master.hippocampal.support_preserving_min_first_action_classes == 3


def test_propose_records_candidate_support_diagnostics():
    torch.manual_seed(42)
    module = _make_module(use_scaffold=False)
    candidates = module.propose_trajectories(
        z_world=torch.zeros(1, 8),
        z_self=torch.zeros(1, 8),
    )

    diags = module.get_last_propose_diagnostics()
    assert len(candidates) == module.config.num_candidates
    assert "candidate_first_action_counts" in diags
    assert "candidate_unique_first_action_classes" in diags
    assert "candidate_first_action_entropy" in diags
    assert "candidate_samples_collected" in diags
    assert "decoded_action_argmax_histogram" in diags
    assert "action_object_decoder_raw_output_stats" in diags
    assert diags["use_orthogonal_cem_seeding"] is False
    assert diags["use_support_preserving_cem"] is False
    assert diags["use_action_class_scaffold_candidates"] is False
    assert diags["candidate_samples_collected"] == len(candidates)
    assert len(diags["cem_iteration_diagnostics"]) == module.config.num_cem_iterations

    iter0 = diags["cem_iteration_diagnostics"][0]
    assert "ao_std_mean" in iter0
    assert "ao_std_min" in iter0
    assert "ao_std_max" in iter0
    assert "pre_refit_first_action_counts" in iter0
    assert "post_elite_refit_first_action_counts" in iter0
    assert "support_preserving_elite_active" in iter0
    assert "action_object_decoder_raw_output_stats" in iter0
    raw = iter0["action_object_decoder_raw_output_stats"]
    assert len(raw["mean_by_action_dim"]) == module.config.action_dim
    assert len(raw["std_by_action_dim"]) == module.config.action_dim


def test_direct_decoder_sampling_uses_same_initial_distribution_surface():
    torch.manual_seed(7)
    module = _make_module(use_scaffold=False)
    z_world = torch.zeros(1, 8)
    ao_mean = module._get_terrain_action_object_mean(z_world)
    samples = ao_mean + torch.randn(
        128,
        module.config.horizon,
        module.config.action_object_dim,
    )

    decoded = module._decode_action_objects(samples)
    summary = module._summarize_action_tensor(decoded)

    assert sum(summary["first_action_counts"].values()) == 128
    assert len(summary["raw_output_stats"]["mean_by_action_dim"]) == 4
    assert "all_action_argmax_histogram" in summary


def test_action_class_scaffold_reaches_e3_with_all_first_classes():
    torch.manual_seed(13)
    module = _make_module(use_scaffold=True)
    candidates = module.propose_trajectories(
        z_world=torch.zeros(1, 8),
        z_self=torch.zeros(1, 8),
    )

    classes = _first_classes(candidates)
    diags = module.get_last_propose_diagnostics()

    assert len(candidates) == module.config.num_candidates
    assert set(range(module.config.action_dim)).issubset(set(classes))
    assert diags["use_action_class_scaffold_candidates"] is True
    assert diags["action_class_scaffold_candidates_added"] == module.config.action_dim
    assert diags["candidate_unique_first_action_classes"] >= module.config.action_dim


def test_support_preserving_cem_repairs_collapsed_decoder_surface():
    torch.manual_seed(19)
    module = _make_module(use_support_preserving_cem=True)

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

    module._decode_action_objects = collapsed_decode  # type: ignore[method-assign]
    candidates = module.propose_trajectories(
        z_world=torch.zeros(1, 8),
        z_self=torch.zeros(1, 8),
    )

    classes = _first_classes(candidates)
    diags = module.get_last_propose_diagnostics()

    assert len(candidates) == module.config.num_candidates
    assert len(set(classes)) >= 2
    assert 3 in set(classes)
    assert diags["use_support_preserving_cem"] is True
    assert diags["support_preserving_active"] is True
    assert diags["support_preserving_injected_candidates"] >= 1
    assert diags["candidate_unique_first_action_classes"] >= 2
    assert diags["candidate_first_action_entropy"] > 0.0


def test_support_preserving_flag_off_leaves_collapsed_surface_collapsed():
    torch.manual_seed(19)
    module = _make_module(use_support_preserving_cem=False)

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

    module._decode_action_objects = collapsed_decode  # type: ignore[method-assign]
    candidates = module.propose_trajectories(
        z_world=torch.zeros(1, 8),
        z_self=torch.zeros(1, 8),
    )

    classes = _first_classes(candidates)
    diags = module.get_last_propose_diagnostics()

    assert len(set(classes)) == 1
    assert set(classes) == {3}
    assert diags["use_support_preserving_cem"] is False
    assert diags["support_preserving_active"] is False
