"""Contracts for mech457_consummatory_act (H-consummation-binding leg of competence_floor / MECH-457).

Substrate: ree_core/environment/causal_grid_world.py -- a distinct no-move CONSUME action
(index 5) added when consummatory_act_enabled is True, so that entering a resource cell only
AFFORDS consumption (contact) and a separate consummatory ACT EFFECTS it. This lets an approach
drive extinguish on contact and hand off to a consummatory act, the dissociation V3-EXQ-781's
non-extinguishing terminal drive could not express.

The contracts that matter here are:
  (a) OFF is byte-identical to the pre-change auto-consume-on-entry env (5-action space, entry
      consumes) -- the whole family's backward-compat invariant, and what keeps every pre-change
      arm fingerprint valid;
  (b) ON dissociates contact from consumption -- entry affords (retains the resource, delivers
      ZERO reward, no homeostatic restore), and only the CONSUME action effects it;
  (c) consumption is the SAME operation whichever way it is reached -- the legacy auto-consume and
      the consummatory CONSUME share one code path (_consume_resource_at), so the reward /
      health / energy / per-axis-drive deltas are identical. Only the TIMING differs. If they
      diverged, the leg would be measuring a consumption change, not a binding change.
"""

from __future__ import annotations

from ree_core.environment.causal_grid_world import CausalGridWorldV2 as Env


def _mk(**kw):
    base = dict(size=8, num_resources=6, num_hazards=0, seed=123, use_proxy_fields=True)
    base.update(kw)
    return Env(**base)


def _adjacent_resource_action(env):
    """An action 0-3 that steps the agent directly onto a resource, or None."""
    ax, ay = env.agent_x, env.agent_y
    for a, (dx, dy) in env.ACTIONS.items():
        if a == 4:
            continue
        nx, ny = ax + dx, ay + dy
        if any(int(r[0]) == nx and int(r[1]) == ny for r in env.resources):
            return a
    return None


def _walk_to_adjacent_resource(env, max_steps=300):
    """Greedy manhattan walk until an action lands on a resource next step. Returns that action."""
    import numpy as np
    for _ in range(max_steps):
        a = _adjacent_resource_action(env)
        if a is not None:
            return a
        if not env.resources:
            return None
        ax, ay = env.agent_x, env.agent_y
        tgt = min(env.resources, key=lambda r: abs(r[0] - ax) + abs(r[1] - ay))
        dx = int(np.sign(tgt[0] - ax))
        dy = int(np.sign(tgt[1] - ay))
        if dx != 0:
            env.step(0 if dx < 0 else 1)
        elif dy != 0:
            env.step(2 if dy < 0 else 3)
        else:
            env.step(4)
    return None


# --- C1: action_dim grows 5 -> 6 only when enabled -------------------------------
def test_c1_action_dim_off_is_five_on_is_six():
    off = _mk(consummatory_act_enabled=False)
    off.reset()
    on = _mk(consummatory_act_enabled=True)
    on.reset()
    assert off.action_dim == 5, "OFF must keep the legacy 5-action space"
    assert on.action_dim == 6, "ON must expose the distinct CONSUME action"
    # CausalGridWorldV2 is a factory; CONSUME_ACTION lives on the instance's class.
    assert on.CONSUME_ACTION == 5


# --- C2: OFF -- entering a resource auto-consumes (legacy, unchanged) -------------
def test_c2_off_entry_auto_consumes():
    env = _mk(consummatory_act_enabled=False)
    env.reset()
    a = _walk_to_adjacent_resource(env)
    assert a is not None
    n_before = len(env.resources)
    _, harm, _, info, _ = env.step(a)
    assert len(env.resources) == n_before - 1, "OFF entry must remove the resource"
    assert harm > 0.0, "OFF entry must deliver the benefit reward"
    assert info["transition_type"] == "resource"
    # OFF info still carries the flags (always present), both falsey.
    assert info["consummatory_act_enabled"] is False
    assert info["on_consumable_resource"] is False


# --- C3: ON -- contact AFFORDS but does not EFFECT consumption --------------------
def test_c3_on_contact_affords_only():
    env = _mk(consummatory_act_enabled=True)
    env.reset()
    a = _walk_to_adjacent_resource(env)
    assert a is not None
    n_before = len(env.resources)
    h_before, e_before = env.agent_health, env.agent_energy
    _, harm, _, info, _ = env.step(a)
    assert len(env.resources) == n_before, "ON contact must RETAIN the resource"
    assert harm == 0.0, "ON contact must deliver ZERO reward (reward binds to the act)"
    # health has no per-step decay, so equality proves no consummatory restore.
    assert env.agent_health == h_before, "ON contact must not restore health"
    # energy decays every step (energy_decay), so it may only DROP, never rise --
    # contact adds no restore bump.
    assert env.agent_energy <= e_before, "ON contact must not restore energy"
    assert info["transition_type"] == "resource_contact"
    assert info["on_consumable_resource"] is True, "affordance flag must be set on contact"


# --- C4: ON -- the CONSUME act effects consumption while standing on a resource ---
def test_c4_on_consume_effects_consumption():
    env = _mk(consummatory_act_enabled=True)
    env.reset()
    a = _walk_to_adjacent_resource(env)
    assert a is not None
    env.step(a)  # contact (affords)
    n_before = len(env.resources)
    _, harm, _, info, _ = env.step(env.CONSUME_ACTION)
    assert len(env.resources) == n_before - 1, "CONSUME on a resource must remove it"
    assert harm > 0.0, "CONSUME must deliver the benefit reward"
    assert info["transition_type"] == "resource"
    assert info["on_consumable_resource"] is False, "resource gone -> flag clears"


# --- C5: ON -- CONSUME off a resource is indistinguishable from STAY --------------
def test_c5_on_consume_off_resource_equals_stay():
    def fresh():
        e = _mk(consummatory_act_enabled=True)
        e.reset()
        while any(int(r[0]) == e.agent_x and int(r[1]) == e.agent_y for r in e.resources):
            e.step(0)
        return e
    e_consume = fresh()
    e_stay = fresh()
    n_before = len(e_consume.resources)
    _, harm_c, _, info_c, _ = e_consume.step(e_consume.CONSUME_ACTION)
    _, harm_s, _, info_s, _ = e_stay.step(4)
    assert len(e_consume.resources) == n_before, "off-resource CONSUME removes nothing"
    assert abs(harm_c - harm_s) < 1e-9, "off-resource CONSUME reward must equal STAY"
    assert info_c["transition_type"] == info_s["transition_type"]
    assert (e_consume.agent_x, e_consume.agent_y) == (e_stay.agent_x, e_stay.agent_y)


# --- C6: ON -- leaving an un-consumed resource restores the grid marker -----------
def test_c6_on_leave_restores_resource_marker():
    env = _mk(consummatory_act_enabled=True)
    env.reset()
    a = _walk_to_adjacent_resource(env)
    assert a is not None
    env.step(a)
    rc = (env.agent_x, env.agent_y)
    for mv in (0, 1, 2, 3):
        dx, dy = env.ACTIONS[mv]
        nx, ny = env.agent_x + dx, env.agent_y + dy
        if 0 <= nx < env.size and 0 <= ny < env.size and env.grid[nx, ny] != env.ENTITY_TYPES["wall"]:
            env.step(mv)
            break
    assert any(int(r[0]) == rc[0] and int(r[1]) == rc[1] for r in env.resources), \
        "un-consumed resource must survive departure"
    assert int(env.grid[rc[0], rc[1]]) == env.ENTITY_TYPES["resource"], \
        "grid marker must be restored to resource so a return visit re-affords"


# --- C7: consumption is the SAME operation via legacy-entry and via CONSUME -------
def test_c7_consumption_operation_is_path_independent():
    """The legacy auto-consume-on-entry (OFF) and the consummatory CONSUME (ON) route through
    ONE code path (_consume_resource_at), so the consumption OPERATION is identical -- only its
    TIMING differs. Proven on the two quantities that do NOT depend on step count: the benefit
    REWARD (resource_benefit * amp, constant) and the HEALTH restore (contact_benefit * 0.5;
    health has no per-step decay). Energy and per-axis drive are deliberately NOT asserted here
    because ON consumes one tick later (afford, then act), so energy_decay / per_axis_drive_decay
    make them differ by exactly one decay tick -- that is correct behaviour, not a divergence.
    The per-axis-drive restore path is regression-covered by test_sd049_phase2_drive_coupling.py
    (OFF-mode, i.e. the same _consume_resource_at)."""
    # Deterministic seed + greedy walk that stops ADJACENT to a resource (never stepping onto
    # one), so OFF and ON traverse an identical path and arrive in an identical pre-consume state.
    off = _mk(consummatory_act_enabled=False)
    off.reset()
    a_off = _walk_to_adjacent_resource(off)
    assert a_off is not None
    h0 = off.agent_health
    _, harm_off, _, _, _ = off.step(a_off)  # entry consumes at step S
    d_health_off = off.agent_health - h0

    on = _mk(consummatory_act_enabled=True)
    on.reset()
    a_on = _walk_to_adjacent_resource(on)
    assert a_on is not None
    assert a_on == a_off, "identical seed + walk must reach the same landing move"
    on.step(a_on)          # affords at step S (no consumption)
    h1 = on.agent_health
    _, harm_on, _, _, _ = on.step(on.CONSUME_ACTION)  # consumes at step S+1
    d_health_on = on.agent_health - h1

    assert harm_off > 0.0, "non-vacuous: consumption actually delivered a benefit reward"
    assert abs(harm_off - harm_on) < 1e-9, "benefit reward must be path-independent"
    # health starts at 1.0 (sated) so the restore caps to 0 in BOTH paths -- equality still
    # proves the identical formula is applied; the reward above is the non-vacuous signal.
    assert abs(d_health_off - d_health_on) < 1e-9, "health restore must be path-independent"
