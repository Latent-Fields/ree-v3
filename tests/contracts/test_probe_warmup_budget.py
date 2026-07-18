"""Contract tests for the probe-read BUDGET DENOMINATION in experiments/_lib/probe_warmup.

Guards the same defect class as tests/contracts/test_sample_driven_rollout.py, one
layer up. `sample_driven_rollout` was hardened for it in ree-v3 6ac2a0d
(EpisodeCapWarning + episode_cap_can_bind + the rollout_* manifest fields);
`probe_warmup` was deliberately NOT changed there, and so shipped a
`WarmupRecipe.probe_max_episodes = 40` against `probe_max_env_steps = 4000`.

Why that mattered: every episode costs at least one env step, so an episode cap
BELOW the step cap means a short-episode seed cannot spend its step budget. At the
V3-EXQ-779a seed-23 shape (~7 steps/episode) the de-saturation read got ~280 env
steps against a 120-selection floor -- i.e. it could report a saturation verdict
built on a starved sample, or "collected ZERO selections", which reads as a
sampling bug rather than as the budget defect it is. Adjudicated as MECH-063
autopsy followup #3 (failure_autopsy_MECH-063-777a-779a-cluster_2026-07-18,
targets[1].learning_extracted[0]).

The load-bearing assertions here:
  * the DEFAULT recipe is step-denominated -- the episode cap cannot bind first,
    and it stays that way when probe_max_env_steps is retuned ALONE (the sentinel
    exists precisely so a duplicated literal cannot rot back into a tight cap);
  * a tight cap is still expressible, but it is DECLARED and it is VISIBLE:
    as_dict() reports probe_episode_cap_can_bind, and the resolved cap -- not the
    sentinel -- is what reaches the cache key;
  * WarmupOutcome carries the binding constraint and the realised spend through to
    the manifest, so a reader can tell a starved read from a cheap one WITHOUT
    re-deriving it (before this, only probe_stop_reason was carried);
  * saturation_summary refuses to report an informative_yield without saying
    whether the reads behind it were budget-limited.

Pure dataclass-level tests -- no substrate, no env, no rollout. ASCII-only.
Run: pytest tests/contracts/test_probe_warmup_budget.py -q
"""

from __future__ import annotations

import warnings

# conftest puts ree-v3 root on sys.path -> `experiments._lib.*` importable.
from experiments._lib.probe_warmup import (
    WarmupOutcome,
    WarmupRecipe,
    saturation_summary,
)
from experiments._lib.sample_driven_rollout import EpisodeCapWarning, RolloutBudget


# ---------------------------------------------------------------------------
# The default must be step-denominated.
# ---------------------------------------------------------------------------

def test_default_recipe_episode_cap_cannot_bind():
    """The shipped default must not let the EPISODE cap bind before the STEP cap."""
    r = WarmupRecipe(num_episodes=5)
    assert r.resolved_probe_max_episodes >= r.probe_max_env_steps
    assert r.probe_episode_cap_can_bind is False


def test_default_recipe_survives_retuning_the_step_cap_alone():
    """The sentinel, not a duplicated literal.

    This is the regression the `0 => derive` form exists to prevent: an author who
    raises probe_max_env_steps and forgets a hardcoded episode cap would silently
    re-create the defect.
    """
    r = WarmupRecipe(num_episodes=5, probe_max_env_steps=99_000)
    assert r.resolved_probe_max_episodes == 99_000
    assert r.probe_episode_cap_can_bind is False


def test_default_recipe_budget_does_not_warn_and_agrees_with_rollout_budget():
    """probe_warmup's notion of 'can bind' must be the SAME as RolloutBudget's.

    Two independent definitions of the invariant would drift; this pins them
    together, and simultaneously proves the default raises no EpisodeCapWarning.
    """
    r = WarmupRecipe(num_episodes=5)
    # Promote EpisodeCapWarning to an error: if the default ever regresses to a
    # tight cap, this raises rather than merely recording a warning.
    with warnings.catch_warnings():
        warnings.simplefilter("error", EpisodeCapWarning)
        b = RolloutBudget(
            sample_floors={"selections": r.probe_selections},
            max_env_steps=r.probe_max_env_steps,
            steps_per_episode=r.steps_per_episode,
            max_episodes=r.resolved_probe_max_episodes,
        )
    assert b.episode_cap_can_bind is r.probe_episode_cap_can_bind is False


def test_the_shipped_defect_shape_is_now_flagged():
    """The exact pre-fix numbers (40 episodes vs 4000 steps) must read as binding."""
    r = WarmupRecipe(num_episodes=5, probe_max_episodes=40, probe_max_env_steps=4000)
    assert r.resolved_probe_max_episodes == 40
    assert r.probe_episode_cap_can_bind is True
    assert r.as_dict()["probe_episode_cap_can_bind"] is True


# ---------------------------------------------------------------------------
# A tight cap stays available, but declared and visible.
# ---------------------------------------------------------------------------

def test_as_dict_reports_resolved_cap_not_sentinel():
    """The cache key must describe the budget that ACTUALLY ran."""
    d = WarmupRecipe(num_episodes=5).as_dict()
    assert d["probe_max_episodes"] == 4000
    assert d["probe_max_episodes_derived"] is True

    d_tight = WarmupRecipe(num_episodes=5, probe_max_episodes=40).as_dict()
    assert d_tight["probe_max_episodes"] == 40
    assert d_tight["probe_max_episodes_derived"] is False


def test_tight_cap_is_distinguishable_in_the_cache_key():
    """A derived and an explicitly-tight recipe must not hash to the same key.

    as_dict() feeds _warmup_key; if these collided, a warmed checkpoint produced
    under one probe budget could be served to a run declaring the other.
    """
    derived = WarmupRecipe(num_episodes=5).as_dict()
    tight = WarmupRecipe(num_episodes=5, probe_max_episodes=40).as_dict()
    assert derived != tight


def test_allow_tight_episode_cap_is_declared_in_the_recipe():
    """Opting into a tight cap must be part of the recorded recipe, not ambient."""
    r = WarmupRecipe(num_episodes=5, probe_max_episodes=40, allow_tight_episode_cap=True)
    assert r.as_dict()["probe_allow_tight_episode_cap"] is True
    # The flag silences the warning; it must NOT hide the fact.
    assert r.probe_episode_cap_can_bind is True
    assert r.as_dict()["probe_episode_cap_can_bind"] is True


# ---------------------------------------------------------------------------
# The signal must reach the manifest.
# ---------------------------------------------------------------------------

def _outcome(seed: int, **kw) -> WarmupOutcome:
    base = dict(
        seed=seed,
        d_action_mass_mean=0.5,
        d_action_mass_std=0.1,
        regime="headroom",
        saturated=False,
        n_probe_selections=120,
        probe_stop_reason="floors_met",
        warmup_episodes=5,
        cache_hit=False,
        recipe=WarmupRecipe(num_episodes=5).as_dict(),
        n_probe_env_steps=1500,
        n_probe_episodes=12,
        probe_max_env_steps=4000,
        probe_max_episodes=4000,
        probe_episode_cap_can_bind=False,
        probe_floors_met=True,
    )
    base.update(kw)
    return WarmupOutcome(**base)


def test_manifest_fields_carry_the_binding_constraint():
    """Before this fix only probe_stop_reason was carried -- which cap DID fire,
    never which cap COULD have, nor how much budget was spent."""
    f = _outcome(1).as_manifest_fields()
    for key in (
        "probe_episode_cap_can_bind",
        "probe_max_episodes",
        "probe_max_env_steps",
        "n_probe_env_steps",
        "n_probe_episodes",
        "probe_floors_met",
    ):
        assert key in f, "manifest lost the budget field %r" % key
    assert f["probe_episode_cap_can_bind"] is False
    assert f["probe_floors_met"] is True


def test_starved_read_is_visible_in_manifest_fields():
    """A starved read and a cheap one must not look alike in the manifest."""
    starved = _outcome(
        23,
        n_probe_selections=31,
        probe_stop_reason="max_episodes",
        n_probe_env_steps=280,
        n_probe_episodes=40,
        probe_max_episodes=40,
        probe_episode_cap_can_bind=True,
        probe_floors_met=False,
    ).as_manifest_fields()
    assert starved["probe_floors_met"] is False
    assert starved["probe_episode_cap_can_bind"] is True
    assert starved["n_probe_env_steps"] < starved["probe_max_env_steps"]


# ---------------------------------------------------------------------------
# The aggregate must qualify its own yield.
# ---------------------------------------------------------------------------

def test_summary_reports_clean_budget_when_all_reads_are_clean():
    s = saturation_summary([_outcome(i) for i in (1, 2, 3)])
    assert s["probe_budget_clean"] is True
    assert s["n_probe_starved"] == 0
    assert s["n_probe_episode_cap_can_bind"] == 0


def test_summary_flags_a_budget_limited_read_and_names_the_seed():
    """An informative_yield computed over starved reads is a sampling artefact.
    The summary must say so, and name which seed, without a re-run."""
    outcomes = [
        _outcome(1),
        _outcome(2),
        _outcome(23, probe_floors_met=False, probe_episode_cap_can_bind=True,
                 probe_max_episodes=40, n_probe_selections=31, n_probe_env_steps=280),
    ]
    s = saturation_summary(outcomes)
    assert s["probe_budget_clean"] is False
    assert s["probe_starved_seeds"] == [23]
    assert s["probe_episode_cap_can_bind_seeds"] == [23]
    # The yield is still reported -- qualified, not suppressed.
    assert s["n_seeds"] == 3


def test_unmeasured_seeds_do_not_count_as_budget_defects():
    """measure=False ran no read at all, so it is neither starved nor cap-bound."""
    unmeasured = _outcome(
        9,
        d_action_mass_mean=None,
        d_action_mass_std=None,
        regime="unmeasured",
        n_probe_selections=0,
        probe_stop_reason="not_measured",
        n_probe_env_steps=0,
        n_probe_episodes=0,
        probe_max_env_steps=0,
        probe_max_episodes=0,
        probe_floors_met=True,
    )
    s = saturation_summary([_outcome(1), unmeasured])
    assert s["probe_budget_clean"] is True
    assert s["n_probe_starved"] == 0
