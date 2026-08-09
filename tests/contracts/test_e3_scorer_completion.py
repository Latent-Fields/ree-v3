"""SD-E3-SCORER-COMPLETION: E3 score_trajectory carries no untrained-head noise.

WHAT THIS PINS. `failure_autopsy_V3-EXQ-190a_2026-08-09` confirmed, by
exhaustive grep across `ree_core/`, that two of `E3TrajectorySelector`'s
trajectory-scoring cost sub-components read UNTRAINED `nn.Sequential` heads
that are touched by no loss anywhere in the substrate:

  * `reality_scorer` -- the "viability" term in `compute_reality_cost` (F),
    present on EVERY scoring path;
  * `harm_cost_fallback_scorer` -- the subtracted term in
    `compute_harm_cost_fallback` (the default fallback M path).

Because `select() -> score_trajectory` is on the live agent action path, that
random-initialisation noise perturbed the policy itself in both conditions of
every run using the default scoring path -- contaminating MECH-022's
V3-EXQ-190a test (C3 collapsed, including a sign flip on a repeated seed). The
fix gates both heads' CONTRIBUTION out of the score by default via
`E3Config.e3_include_untrained_fallback_scorers` (default False = the fix; the
`nn.Sequential` heads stay instantiated so state_dict / checkpoint keys are
unchanged).

METHOD. The defect is a DETERMINISTIC property -- "does the score depend on
these heads' parameters?" -- so it is checked directly by re-randomising the
two heads' weights and asserting the default score is bit-identical before and
after. A negative control flips the legacy flag True and asserts the score DOES
move, so the test cannot pass vacuously (e.g. if the heads were silently
dropped from the module, or the re-randomisation were a no-op).

No torch RNG cross-machine hazard here: the assertions are exact-equality on a
score DIFFERENCE that is identically zero on the fix path, and a
strictly-positive magnitude on the legacy path -- neither depends on a
multinomial draw.
"""

import torch

from ree_core.utils.config import E3Config
from ree_core.predictors.e3_selector import E3TrajectorySelector
from ree_core.predictors.e2_fast import Trajectory

WORLD_DIM = 8
HIDDEN_DIM = 16


def _make_trajectory(seed: int = 0) -> Trajectory:
    g = torch.Generator().manual_seed(seed)
    horizon = 4
    ws = torch.randn(1, horizon + 1, WORLD_DIM, generator=g)
    ss = torch.randn(1, horizon + 1, WORLD_DIM, generator=g)
    acts = torch.randn(1, horizon, 4, generator=g)
    return Trajectory(
        states=[ss[:, i, :] for i in range(horizon + 1)],
        actions=acts,
        world_states=[ws[:, i, :] for i in range(horizon + 1)],
    )


def _rerandomise_dead_heads(sel: E3TrajectorySelector) -> None:
    """Overwrite ONLY the two untrained heads' weights with large-variance noise."""
    for head in (sel.reality_scorer, sel.harm_cost_fallback_scorer):
        for p in head.parameters():
            torch.nn.init.normal_(p, mean=0.0, std=5.0)


def _score(sel: E3TrajectorySelector, traj: Trajectory) -> float:
    # Default fallback M path (no harm_forward_model / harm_bridge) -- the path
    # the autopsy found contaminated.
    return float(sel.score_trajectory(traj).item())


def test_default_config_gates_untrained_heads_off():
    """The default MUST be the fix, not the legacy contaminated behaviour."""
    cfg = E3Config(world_dim=WORLD_DIM, hidden_dim=HIDDEN_DIM)
    assert cfg.e3_include_untrained_fallback_scorers is False


def test_untrained_heads_do_not_influence_default_score():
    """Re-randomising the two never-trained heads leaves the default score
    bit-identical -- they contribute nothing to trajectory scoring."""
    sel = E3TrajectorySelector(E3Config(world_dim=WORLD_DIM, hidden_dim=HIDDEN_DIM))
    traj = _make_trajectory()

    before = _score(sel, traj)
    _rerandomise_dead_heads(sel)
    after = _score(sel, traj)

    assert after == before, (
        f"untrained heads still influence the default score: "
        f"before={before!r} after={after!r}"
    )


def test_legacy_flag_reintroduces_head_dependence_negative_control():
    """Negative control: with the legacy flag True the score DOES depend on the
    two heads -- so the fix test above is not vacuous."""
    sel = E3TrajectorySelector(
        E3Config(
            world_dim=WORLD_DIM,
            hidden_dim=HIDDEN_DIM,
            e3_include_untrained_fallback_scorers=True,
        )
    )
    traj = _make_trajectory()

    before = _score(sel, traj)
    _rerandomise_dead_heads(sel)
    after = _score(sel, traj)

    assert abs(after - before) > 1e-6, (
        "legacy path should route the score through the untrained heads, but "
        f"re-randomising them left it unchanged: before={before!r} after={after!r}"
    )


def test_reality_cost_default_is_coherence_only():
    """compute_reality_cost default == parameter-free coherence: re-randomising
    reality_scorer must not move it; the legacy path must."""
    fix = E3TrajectorySelector(E3Config(world_dim=WORLD_DIM, hidden_dim=HIDDEN_DIM))
    legacy = E3TrajectorySelector(
        E3Config(
            world_dim=WORLD_DIM,
            hidden_dim=HIDDEN_DIM,
            e3_include_untrained_fallback_scorers=True,
        )
    )
    legacy.load_state_dict(fix.state_dict())  # identical weights; only the flag differs
    traj = _make_trajectory(seed=3)

    f_fix = float(fix.compute_reality_cost(traj).item())
    f_legacy = float(legacy.compute_reality_cost(traj).item())
    # The two differ precisely by the gated reality_scorer(final) viability term.
    assert f_fix != f_legacy


def test_harm_cost_fallback_default_is_trained_head_only():
    """compute_harm_cost_fallback default == the TRAINED harm_eval_head sum: the
    subtracted untrained fallback term is gated out by default."""
    fix = E3TrajectorySelector(E3Config(world_dim=WORLD_DIM, hidden_dim=HIDDEN_DIM))
    legacy = E3TrajectorySelector(
        E3Config(
            world_dim=WORLD_DIM,
            hidden_dim=HIDDEN_DIM,
            e3_include_untrained_fallback_scorers=True,
        )
    )
    legacy.load_state_dict(fix.state_dict())
    traj = _make_trajectory(seed=5)

    m_fix = fix.compute_harm_cost_fallback(traj)
    m_legacy = legacy.compute_harm_cost_fallback(traj)

    # Default == harm_eval_head sum alone; re-derive it independently.
    world_seq = fix._get_world_states(traj)
    batch, horizon_p1, _ = world_seq.shape
    flat = world_seq.reshape(batch * horizon_p1, -1)
    harm_only = fix.harm_eval_head(flat).reshape(batch, horizon_p1).sum(dim=-1)

    assert torch.allclose(m_fix, harm_only)
    assert not torch.allclose(m_fix, m_legacy)
