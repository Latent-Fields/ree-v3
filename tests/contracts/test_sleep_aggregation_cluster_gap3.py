"""Contract tests for sleep-aggregation cluster GAP-3: unified Phase A-E
master flag (use_sleep_aggregation_cluster).

See REE_assembly/evidence/planning/sleep_substrate_plan.md GAP-3.

Before GAP-3 the offline-consolidation pathway was gated by EIGHT
independent default-False flags; an experiment had to set all eight by
hand or the cluster was silent. use_sleep_aggregation_cluster forces the
eight sub-flags True from one switch, via enable_sleep_aggregation_cluster()
called from __post_init__ (direct construction) and from from_dims() (the
factory path experiments use).

Guarantees enforced:
  C1. use_sleep_aggregation_cluster defaults False in REEConfig() and in
      REEConfig.from_dims(...) (backward-compat).
  C2. Cluster flag False with no per-flag overrides -> all eight sub-flags
      stay False (bit-identical pre-GAP-3 OFF).
  C3. Direct REEConfig(use_sleep_aggregation_cluster=True) (__post_init__
      path) resolves all eight sub-flags True.
  C4. REEConfig.from_dims(..., use_sleep_aggregation_cluster=True) (factory
      path) resolves all eight sub-flags True.
  C5. OR-only semantics: master True alongside an explicit sub-flag=False
      overrides it to True (matches the use_mech307_conjunction resolver
      convention).
  C6. End-to-end REEAgent wiring: an agent built with the cluster flag plus
      the substrate prerequisites (anchor sets, staleness accumulator,
      e2_harm_s) constructs all four Phase B-E components and the
      SleepLoopManager (none are None).
  C7. enable_sleep_aggregation_cluster() returns self, is idempotent, and
      does NOT turn on MECH-204 precision recalibration (separate GAP-1
      sibling step, deliberately not bundled).
"""

from __future__ import annotations


# The eight Phase A-E sub-flags GAP-3 unifies.
_SUB_FLAGS = (
    "use_sleep_loop",
    "sws_enabled",
    "rem_enabled",
    "use_mech285_sampler",
    "use_mech272_routing",
    "use_mech272_routing_consumer",
    "use_mech275_aggregator",
    "use_mech273_self_model",
)


# ---------------------------------------------------------------------------- #
# Helpers                                                                      #
# ---------------------------------------------------------------------------- #

def _all(cfg, value):
    return {f: getattr(cfg, f) for f in _SUB_FLAGS if getattr(cfg, f) is not value}


def _build_cluster_agent():
    """Agent with the cluster flag ON plus substrate prerequisites.

    anchor sets / staleness / e2_harm_s are deliberately NOT folded into
    the cluster flag (separate MECH-269/ARC-033 substrate switches, per
    GAP-3 scope), so the end-to-end test must enable them explicitly.
    """
    from ree_core.agent import REEAgent
    from ree_core.utils.config import REEConfig

    cfg = REEConfig.from_dims(
        body_obs_dim=12,
        world_obs_dim=250,
        action_dim=4,
        use_sleep_aggregation_cluster=True,
        sleep_loop_episodes_K=1,
        mech285_draws_per_cycle=6,
        use_anchor_sets=True,
        use_staleness_accumulator=True,
        use_e2_harm_s_forward=True,
        mech273_offline_n_steps=5,
    )
    return REEAgent(cfg)


# ---------------------------------------------------------------------------- #
# Contracts                                                                    #
# ---------------------------------------------------------------------------- #

def test_c1_default_config_backward_compatible():
    """use_sleep_aggregation_cluster defaults False (dataclass + factory)."""
    from ree_core.utils.config import REEConfig

    assert REEConfig().use_sleep_aggregation_cluster is False
    cfg = REEConfig.from_dims(body_obs_dim=12, world_obs_dim=250, action_dim=4)
    assert cfg.use_sleep_aggregation_cluster is False


def test_c2_cluster_off_subflags_all_false():
    """Cluster OFF + no overrides -> eight sub-flags stay False (bit-identical)."""
    from ree_core.utils.config import REEConfig

    cfg = REEConfig()
    offenders = _all(cfg, False)
    assert not offenders, f"Sub-flags not False under default OFF: {offenders}"

    cfg2 = REEConfig.from_dims(body_obs_dim=12, world_obs_dim=250, action_dim=4)
    offenders2 = _all(cfg2, False)
    assert not offenders2, f"from_dims default OFF leaked: {offenders2}"


def test_c3_direct_construction_resolves_all_subflags():
    """REEConfig(use_sleep_aggregation_cluster=True) -> all eight True (__post_init__)."""
    from ree_core.utils.config import REEConfig

    cfg = REEConfig(use_sleep_aggregation_cluster=True)
    offenders = _all(cfg, True)
    assert not offenders, f"Sub-flags not True via __post_init__: {offenders}"


def test_c4_from_dims_resolves_all_subflags():
    """from_dims(use_sleep_aggregation_cluster=True) -> all eight True (factory)."""
    from ree_core.utils.config import REEConfig

    cfg = REEConfig.from_dims(
        body_obs_dim=12,
        world_obs_dim=250,
        action_dim=4,
        use_sleep_aggregation_cluster=True,
    )
    offenders = _all(cfg, True)
    assert not offenders, f"Sub-flags not True via from_dims: {offenders}"


def test_c5_master_overrides_explicit_false_subflag():
    """OR-only: master True + explicit sub-flag False -> sub-flag forced True."""
    from ree_core.utils.config import REEConfig

    cfg = REEConfig.from_dims(
        body_obs_dim=12,
        world_obs_dim=250,
        action_dim=4,
        use_sleep_aggregation_cluster=True,
        use_mech273_self_model=False,
        use_mech285_sampler=False,
    )
    assert cfg.use_mech273_self_model is True
    assert cfg.use_mech285_sampler is True


def test_c6_agent_end_to_end_constructs_all_phase_components():
    """Cluster flag + substrate prereqs -> all four Phase B-E components wired."""
    agent = _build_cluster_agent()
    assert agent.sleep_loop is not None, "Phase A SleepLoopManager not constructed"
    assert agent.sleep_replay_sampler is not None, "Phase B sampler missing"
    assert agent.sleep_routing_gate is not None, "Phase C routing gate missing"
    assert agent.sleep_bayesian_aggregator is not None, "Phase D aggregator missing"
    assert agent.sleep_self_model_aggregator is not None, "Phase E self-model missing"
    # GAP-8 consumer flag propagated through to the manager.
    assert agent.sleep_loop.use_mech272_routing_consumer is True


def test_c7_enable_method_returns_self_idempotent_no_mech204():
    """enable_sleep_aggregation_cluster() returns self, idempotent, no MECH-204."""
    from ree_core.utils.config import REEConfig

    cfg = REEConfig()
    ret = cfg.enable_sleep_aggregation_cluster()
    assert ret is cfg, "enable_sleep_aggregation_cluster must return self"
    # Idempotent: second call leaves the eight flags True, no error.
    cfg.enable_sleep_aggregation_cluster()
    offenders = _all(cfg, True)
    assert not offenders, f"Not idempotent: {offenders}"
    # MECH-204 precision recalibration is a separate GAP-1 sibling step and
    # must NOT be turned on by the aggregation-cluster master flag.
    assert cfg.use_rem_precision_recalibration is False
