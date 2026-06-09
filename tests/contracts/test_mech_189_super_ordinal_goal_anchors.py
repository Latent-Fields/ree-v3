"""MECH-189 contract: super-ordinal goal-anchor ContextMemory writes substrate
(infant_substrate:GAP-11 / DEV-NEED-006 / DEV-NEED-024).

C1 default OFF: GoalConfig.use_super_ordinal_goal_anchors defaults False;
   REEAgent.super_ordinal_goal_memory is None; update_z_goal path unchanged.
C2 write conjunction: a high-salience contact in a novel (high-complexity)
   context writes an anchor; low salience OR a covered (low-complexity) context
   does not write.
C3 reinforce vs allocate: a near-duplicate context (>= merge_similarity)
   reinforces the matched anchor (strength rises, n_occupied unchanged); a far
   context allocates a fresh slot.
C4 retrieve + complexity: a matching-context query returns the stored anchor
   with a high match; contextual complexity drops once a context is covered.
C5 freeze + MECH-094: write_enabled=False and simulation_mode=True are both
   no-ops.
C6 cross-episode persistence: the store is NOT reset by per-episode
   agent.reset(); reset_super_ordinal_anchors() clears it.
C7 agent WRITE: a forced high-salience update_z_goal in a novel context forms a
   super-ordinal anchor.
C8 agent READ (adult seeding): with a frozen childhood anchor and a fresh
   (sub-floor) z_goal, update_z_goal seeds z_goal toward the anchor in a
   matching context.
"""
from __future__ import annotations

import torch

from ree_core.goal import GoalConfig, GoalState, SuperOrdinalGoalMemory


def _store(context_dim=4, **kw):
    cfg = GoalConfig(goal_dim=4, use_super_ordinal_goal_anchors=True, **kw)
    return SuperOrdinalGoalMemory(cfg, context_dim=context_dim, device=torch.device("cpu"))


def _ctx(*vals):
    return torch.tensor([list(vals)], dtype=torch.float32)


# -- C1 default OFF -----------------------------------------------------------

def test_c1_default_off():
    cfg = GoalConfig(goal_dim=4)
    assert cfg.use_super_ordinal_goal_anchors is False
    # An agent without the flag has no store.
    from ree_core.utils.config import REEConfig
    from ree_core.agent import REEAgent
    a = REEAgent(REEConfig.from_dims(body_obs_dim=17, world_obs_dim=250, action_dim=4,
                                     z_goal_enabled=True))
    assert a.super_ordinal_goal_memory is None
    # Methods are safe no-ops when disabled.
    a.set_super_ordinal_write_enabled(False)
    a.reset_super_ordinal_anchors()


# -- C2 write conjunction -----------------------------------------------------

def test_c2_write_conjunction():
    s = _store(super_ordinal_salience_threshold=0.5,
               super_ordinal_complexity_threshold=0.3)
    # High salience + novel (empty store -> complexity 1.0) -> ALLOCATE.
    assert s.write(_ctx(1, 0, 0, 0), _ctx(9, 0, 0, 0), salience=0.9) is True
    assert s.n_occupied() == 1 and s._n_allocate == 1
    # Low salience -> blocked entirely (gate (a)).
    assert s.write(_ctx(0, 1, 0, 0), _ctx(0, 9, 0, 0), salience=0.1) is False
    assert s.n_occupied() == 1
    # High salience in the SAME (covered) context -> REINFORCES the existing
    # anchor; complexity gates NEW-anchor formation only, not reinforcement.
    assert s.write(_ctx(1, 0, 0, 0), _ctx(8, 0, 0, 0), salience=0.9) is True
    assert s.n_occupied() == 1 and s._n_reinforce == 1


# -- C3 reinforce vs allocate -------------------------------------------------

def test_c3_reinforce_vs_allocate():
    s = _store(super_ordinal_salience_threshold=0.5,
               super_ordinal_complexity_threshold=0.05,
               super_ordinal_merge_similarity=0.9)
    s.write(_ctx(1, 0, 0, 0), _ctx(9, 0, 0, 0), salience=0.9)
    assert s.n_occupied() == 1 and s._n_allocate == 1
    # A nearly-identical context (cosine ~1.0 >= merge) reinforces, not allocates.
    # complexity = 1 - cos(~1.0) ~ 0 < 0.05 would block -> use external mode to
    # force the write and exercise the reinforce branch.
    s2 = _store(super_ordinal_salience_threshold=0.5,
                super_ordinal_complexity_threshold=0.0,
                super_ordinal_merge_similarity=0.9,
                super_ordinal_complexity_mode="external")
    s2.write(_ctx(1, 0, 0, 0), _ctx(9, 0, 0, 0), salience=0.9, context_complexity=1.0)
    s2.write(_ctx(1.0, 0.05, 0, 0), _ctx(8, 0, 0, 0), salience=0.9, context_complexity=1.0)
    assert s2.n_occupied() == 1 and s2._n_reinforce == 1
    assert s2._strength.max().item() == 2.0
    # A far/orthogonal context allocates a new slot.
    s2.write(_ctx(0, 0, 1, 0), _ctx(0, 0, 9, 0), salience=0.9, context_complexity=1.0)
    assert s2.n_occupied() == 2


# -- C4 retrieve + complexity -------------------------------------------------

def test_c4_retrieve_and_complexity():
    s = _store(super_ordinal_salience_threshold=0.5,
               super_ordinal_complexity_threshold=0.05)
    s.write(_ctx(1, 0, 0, 0), _ctx(5, 0, 0, 0), salience=0.9)
    s.write(_ctx(0, 0, 1, 0), _ctx(0, 0, 5, 0), salience=0.9)
    # Query close to the first context retrieves its anchor with high match.
    val, match, idx = s.retrieve(_ctx(0.9, 0.1, 0, 0))
    assert match > 0.9
    assert torch.allclose(val, _ctx(5, 0, 0, 0))
    # A covered context has low complexity; an orthogonal one is maximally novel.
    assert s.contextual_complexity(_ctx(1, 0, 0, 0)) < 0.05
    assert s.contextual_complexity(_ctx(0, 1, 0, 0)) > 0.9


# -- C5 freeze + MECH-094 -----------------------------------------------------

def test_c5_freeze_and_simulation_noop():
    s = _store(super_ordinal_salience_threshold=0.5,
               super_ordinal_complexity_threshold=0.05)
    s.write_enabled = False
    assert s.write(_ctx(1, 0, 0, 0), _ctx(9, 0, 0, 0), salience=0.9) is False
    assert s.n_occupied() == 0
    s.write_enabled = True
    assert s.write(_ctx(1, 0, 0, 0), _ctx(9, 0, 0, 0), salience=0.9,
                   simulation_mode=True) is False
    assert s.n_occupied() == 0


# -- C6 cross-episode persistence --------------------------------------------

def test_c6_persistence_and_reset_anchors():
    from ree_core.utils.config import REEConfig
    from ree_core.agent import REEAgent
    a = REEAgent(REEConfig.from_dims(body_obs_dim=17, world_obs_dim=250, action_dim=4,
                                     z_goal_enabled=True,
                                     use_super_ordinal_goal_anchors=True))
    assert a.super_ordinal_goal_memory is not None
    a.super_ordinal_goal_memory.write(
        torch.ones(1, a.config.latent.world_dim),
        torch.ones(1, a.config.goal.goal_dim),
        salience=10.0,
    )
    n = a.super_ordinal_goal_memory.n_occupied()
    assert n == 1
    a.reset()  # per-episode reset must NOT clear the super-ordinal store
    assert a.super_ordinal_goal_memory.n_occupied() == n
    a.reset_super_ordinal_anchors()  # explicit developmental-stage clear
    assert a.super_ordinal_goal_memory.n_occupied() == 0


# -- C7 agent WRITE -----------------------------------------------------------

def _agent_with_zworld(zw_value):
    from ree_core.utils.config import REEConfig
    from ree_core.agent import REEAgent
    cfg = REEConfig.from_dims(body_obs_dim=17, world_obs_dim=250, action_dim=4,
                              z_goal_enabled=True, drive_weight=2.0,
                              use_super_ordinal_goal_anchors=True,
                              super_ordinal_salience_threshold=0.5,
                              super_ordinal_complexity_threshold=0.2,
                              super_ordinal_seed_below_norm=0.4,
                              super_ordinal_seed_match_threshold=0.3,
                              super_ordinal_seed_strength=0.5)
    a = REEAgent(cfg)
    a.reset()
    wd = a.config.latent.world_dim
    a._current_latent = a.latent_stack.init_state(batch_size=1, device=a.device)
    z = torch.zeros(1, wd)
    z[0, 0] = zw_value
    a._current_latent.z_world = z
    return a


def test_c7_agent_write_forms_anchor():
    a = _agent_with_zworld(1.0)
    # High benefit + drive -> salience = 0.9 * (1 + 2*1.0) = 2.7 >> 0.5; novel ctx.
    a.update_z_goal(benefit_exposure=0.9, drive_level=1.0)
    assert a.goal_state.is_active()
    assert a.super_ordinal_goal_memory.n_occupied() >= 1


# -- C8 agent READ (adult seeding) -------------------------------------------

def test_c8_agent_read_seeds_zgoal():
    a = _agent_with_zworld(1.0)
    # Child phase: form an anchor in this context.
    a.update_z_goal(benefit_exposure=0.9, drive_level=1.0)
    assert a.super_ordinal_goal_memory.n_occupied() >= 1
    anchor = a.super_ordinal_goal_memory.retrieve(a._current_latent.z_world)[0].clone()
    # Freeze writes (adult phase) and reset z_goal to sub-floor.
    a.set_super_ordinal_write_enabled(False)
    a.goal_state.reset()
    assert a.goal_state.goal_norm() < 0.4
    seeds_before = a.super_ordinal_goal_memory._n_seeds
    # Adult tick in the SAME context, no benefit pulse -> seeded from the anchor.
    a.update_z_goal(benefit_exposure=0.0, drive_level=0.0)
    assert a.super_ordinal_goal_memory._n_seeds == seeds_before + 1
    assert a.goal_state.goal_norm() > 0.0
    # No new anchor written while frozen.
    assert a.super_ordinal_goal_memory._n_writes == 1
    # z_goal moved toward the stored anchor direction.
    cos = torch.nn.functional.cosine_similarity(
        a.goal_state.z_goal, anchor, dim=-1).item()
    assert cos > 0.5
