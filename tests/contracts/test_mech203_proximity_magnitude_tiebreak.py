"""Contracts for the MECH-203 proximity-approach classification tie-break fix.

Substrate: ree_core/environment/causal_grid_world.py CausalGridWorldV2.step(), the
"none"-transition proxy-gradient branch (~line 2287). Confirmed bug (2026-08-02):
hazard_field and resource_field are each a SUM over all sources of
1/(1+dist*decay). At default multi-source configs (decay=0.5, num_hazards and
num_resources >= 3 on a 10x10 grid) that sum never drops below ~0.44 anywhere on
the grid, so BOTH fields are >= proximity_approach_threshold (0.15 default) at
effectively 100% of cells. The legacy code resolved ties with a fixed
`if hazard_active: ... elif resource_active: ...` order, so "benefit_approach"
was not merely rare but structurally UNREACHABLE whenever any hazard field was
active nearby -- which, given the saturation above, is nearly always.

Fix: a new opt-in constructor flag, proximity_approach_magnitude_tiebreak
(default False). When both fields are simultaneously active, it classifies by
whichever RAW field value is locally larger (deterministic tie -> hazard, matching
legacy priority on the boundary case), instead of a fixed priority order.

The contracts that matter here are:
  (a) OFF (default) is bit-identical to the pre-fix elif-priority code -- the
      backward-compat invariant that keeps every existing experiment script's
      env dynamics (and therefore any already-collected evidence) unaffected;
  (b) ON lets "benefit_approach" actually fire when the agent is genuinely
      closer to a resource than a hazard, even though both fields are active;
  (c) ON does not flip hazard_approach into benefit_approach when hazard is the
      locally-dominant field -- only the actual tie is broken differently;
  (d) the flag is inert (no behavior change either way) when only one field is
      active -- there is nothing to tie-break;
  (e) at a realistic corpus config (10x10, 3 hazards, 3 resources, default
      decay/threshold/scales), ON produces a non-degenerate mix of both
      transition types over a real rollout, while OFF reproduces the confirmed
      n_benefit_approach == 0 degeneracy -- this is the direct regression test
      for the reported MECH-203 blocker (V3-EXQ-843b: "n_benefit_states=0/
      n_harm_states=100%" at every epsilon tested).
"""

from __future__ import annotations

from ree_core.environment.causal_grid_world import CausalGridWorldV2 as Env


def _mk(**kw):
    base = dict(size=10, num_hazards=0, num_resources=0, seed=1, use_proxy_fields=True)
    base.update(kw)
    return Env(**base)


def _place_and_step_into(env, agent_pos, hazard_positions, resource_positions, action):
    """reset_to() at agent_pos with the given sources, then take one step."""
    env.reset_to(agent_pos=agent_pos, hazard_positions=hazard_positions,
                 resource_positions=resource_positions)
    return env.step(action)


# --- (a) OFF is bit-identical to legacy hazard-first priority ------------------

def test_off_reproduces_legacy_hazard_always_wins_on_tie():
    """A cell where the resource field is STRONGER than the hazard field, but both
    are above threshold: legacy (and default OFF) code must still report
    hazard_approach, reproducing the confirmed bug exactly. This locks in
    backward compatibility -- it is a regression lock on old behavior, not an
    endorsement of it; test (b)/(c) below cover the fix.
    """
    env = _mk(proximity_approach_magnitude_tiebreak=False)
    # Agent at (5,5). Hazard far at (1,1) (weak contribution). Resource close
    # at (5,7) (strong contribution) -- resource field should locally dominate.
    env.reset_to(agent_pos=(5, 5), hazard_positions=[(1, 1)], resource_positions=[(5, 7)])
    h_val = float(env.hazard_field[5, 5])
    r_val = float(env.resource_field[5, 5])
    assert r_val > h_val, f"test setup invalid: expected resource-dominant cell, got h={h_val} r={r_val}"
    assert h_val >= env.proximity_approach_threshold
    assert r_val >= env.proximity_approach_threshold

    # Step "stay" (action 4) so the agent doesn't move off this cell.
    _, _, _, info, _ = env.step(4)
    assert info["transition_type"] == "hazard_approach"
    assert env.total_benefit == 0.0


def test_default_flag_value_is_false():
    env = _mk()
    assert env.proximity_approach_magnitude_tiebreak is False


# --- (b) ON: benefit_approach fires when resource is locally dominant ----------

def test_on_lets_benefit_approach_fire_when_resource_dominant():
    env = _mk(proximity_approach_magnitude_tiebreak=True)
    env.reset_to(agent_pos=(5, 5), hazard_positions=[(1, 1)], resource_positions=[(5, 7)])
    h_val = float(env.hazard_field[5, 5])
    r_val = float(env.resource_field[5, 5])
    assert r_val > h_val
    assert h_val >= env.proximity_approach_threshold
    assert r_val >= env.proximity_approach_threshold

    _, _, _, info, _ = env.step(4)
    assert info["transition_type"] == "benefit_approach"
    assert env.total_benefit > 0.0
    assert env.total_harm == 0.0


# --- (c) ON: hazard still wins when it is the locally-dominant field ----------

def test_on_still_reports_hazard_approach_when_hazard_dominant():
    env = _mk(proximity_approach_magnitude_tiebreak=True)
    # Mirror of the above with hazard/resource roles swapped: hazard close,
    # resource far.
    env.reset_to(agent_pos=(5, 5), hazard_positions=[(5, 7)], resource_positions=[(1, 1)])
    h_val = float(env.hazard_field[5, 5])
    r_val = float(env.resource_field[5, 5])
    assert h_val > r_val
    assert h_val >= env.proximity_approach_threshold
    assert r_val >= env.proximity_approach_threshold

    _, _, _, info, _ = env.step(4)
    assert info["transition_type"] == "hazard_approach"
    assert env.total_harm > 0.0
    assert env.total_benefit == 0.0


# --- (d) flag is inert when only one field is active ---------------------------

def test_flag_inert_when_only_hazard_active():
    """A single, distant hazard and no resource: only the hazard field can ever
    cross threshold. ON vs OFF must agree (nothing to tie-break)."""
    results = {}
    for tiebreak in (False, True):
        env = _mk(proximity_approach_magnitude_tiebreak=tiebreak)
        env.reset_to(agent_pos=(5, 5), hazard_positions=[(5, 6)], resource_positions=[])
        assert float(env.resource_field[5, 5]) == 0.0
        _, _, _, info, _ = env.step(4)
        results[tiebreak] = (info["transition_type"], env.total_harm, env.total_benefit)
    assert results[False] == results[True]
    assert results[False][0] == "hazard_approach"


def test_flag_inert_when_only_resource_active():
    results = {}
    for tiebreak in (False, True):
        env = _mk(proximity_approach_magnitude_tiebreak=tiebreak)
        env.reset_to(agent_pos=(5, 5), hazard_positions=[], resource_positions=[(5, 6)])
        assert float(env.hazard_field[5, 5]) == 0.0
        _, _, _, info, _ = env.step(4)
        results[tiebreak] = (info["transition_type"], env.total_harm, env.total_benefit)
    assert results[False] == results[True]
    assert results[False][0] == "benefit_approach"


def test_flag_inert_when_neither_active():
    """Both sources far enough that neither field crosses threshold: transition_type
    stays 'none' regardless of the flag."""
    results = {}
    for tiebreak in (False, True):
        env = _mk(size=30, proximity_approach_magnitude_tiebreak=tiebreak)
        env.reset_to(agent_pos=(15, 15), hazard_positions=[(1, 1)], resource_positions=[(28, 28)])
        h_val = float(env.hazard_field[15, 15])
        r_val = float(env.resource_field[15, 15])
        assert h_val < env.proximity_approach_threshold
        assert r_val < env.proximity_approach_threshold
        _, _, _, info, _ = env.step(4)
        results[tiebreak] = info["transition_type"]
    assert results[False] == results[True] == "none"


# --- (e) realistic corpus config: ON is non-degenerate, OFF reproduces the bug --

def _rollout_ttype_counts(env, n_steps=600, seed=7):
    import random
    rng = random.Random(seed)
    env.reset()
    counts = {"hazard_approach": 0, "benefit_approach": 0}
    for _ in range(n_steps):
        a = rng.randrange(5)
        _, _, done, info, _ = env.step(a)
        ttype = info.get("transition_type", "none")
        if ttype in counts:
            counts[ttype] += 1
        if done:
            env.reset()
    return counts


def test_corpus_config_off_reproduces_confirmed_degeneracy():
    """Standard 10x10 / 3-hazard / 3-resource config, default decay/threshold/
    scales, flag OFF (the current default for every existing experiment):
    benefit_approach must never fire. This is the confirmed MECH-203 blocker,
    locked in as a named regression so a future change to defaults doesn't
    silently alter it without this test failing loudly.
    """
    env = _mk(num_hazards=3, num_resources=3, proximity_approach_magnitude_tiebreak=False,
              seed=42)
    counts = _rollout_ttype_counts(env, n_steps=600, seed=7)
    assert counts["benefit_approach"] == 0
    assert counts["hazard_approach"] > 0


def test_corpus_config_on_produces_nondegenerate_mix():
    """Same config, flag ON: both transition types must actually occur over a
    real rollout -- the direct fix for the MECH-203 blocker."""
    env = _mk(num_hazards=3, num_resources=3, proximity_approach_magnitude_tiebreak=True,
              seed=42)
    counts = _rollout_ttype_counts(env, n_steps=600, seed=7)
    assert counts["benefit_approach"] > 0
    assert counts["hazard_approach"] > 0
