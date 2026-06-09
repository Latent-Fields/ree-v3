"""Contract tests for ARC-006 / MECH-045 token-instance object-file buffer.

C1 default-off no-op (agent.object_file_buffer is None; bit-identical action
   stream OFF vs ON -- v1 has no consumer, so even ON the action stream is
   unchanged).
C2 token KEY (C1 of the memo): cross-motion re-identification -- a moved entity
   keeps its token id, with a same-type distractor present, so the token is
   continuity-carried, NOT nearest-cell.
C3 persistence-through-absence then eviction (C3 of the memo): a token survives
   <= persist_ttl unseen ticks, then dies.
C4 attention capacity cap (C4 of the memo): births past max_tokens evict the
   weakest (oldest / least-confident) tokens.
C5 precision weighting (C5 of the memo): a zero-precision observation makes no
   feature-EMA move; precision weighting is on by default.
C6 MECH-094: update() is a no-op under simulation_mode (no births / no writes).

Design memo: REE_assembly/docs/architecture/mech_045_object_file_buffer.md.
"""

import numpy as np
import torch

from ree_core.agent import REEAgent
from ree_core.utils.config import REEConfig
from ree_core.entities.object_file_buffer import (
    EntityObservation,
    ObjectFileBuffer,
    ObjectFileBufferConfig,
)


def _buf(**kw):
    cfg = ObjectFileBufferConfig(use_object_file_buffer=True, **kw)
    return ObjectFileBuffer(cfg)


def _unit(dim, seed):
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(dim, generator=g)
    return v / v.norm()


def test_c1_default_off_noop_and_bit_identical():
    def run(seed, **kw):
        torch.manual_seed(seed)
        np.random.seed(seed)
        cfg = REEConfig.from_dims(
            world_obs_dim=250, body_obs_dim=12, action_dim=4, **kw
        )
        ag = REEAgent(cfg)
        if not kw.get("use_object_file_buffer", False):
            assert ag.object_file_buffer is None
        else:
            assert ag.object_file_buffer is not None
            # OFF agent's method returns {} (None buffer); ON returns a real dict.
        acts = []
        for _ in range(8):
            a = ag.act_with_split_obs(torch.zeros(1, 12), torch.zeros(1, 250))
            acts.append(int(a.argmax(dim=-1).flatten()[0].item()))
        return acts

    base = run(0)
    # v1 has no action-stream consumer -> ON is bit-identical to OFF.
    assert base == run(0, use_object_file_buffer=True)
    # default-OFF agent.object_file_buffer is None and the method returns {}.
    torch.manual_seed(0)
    ag_off = REEAgent(REEConfig.from_dims(world_obs_dim=250, body_obs_dim=12, action_dim=4))
    assert ag_off.object_file_buffer is None
    assert ag_off.update_object_file_buffer([EntityObservation(torch.zeros(2), torch.zeros(8))]) == {}


def test_c2_cross_motion_reidentification_with_same_type_distractor():
    D = 128
    vT = _unit(D, 1)   # target appearance
    vD = _unit(D, 2)   # distractor appearance (same TYPE tag, different instance)
    # non-degenerate feature separation (the memo G1 guard premise)
    assert 1.0 - float(torch.dot(vT, vD)) >= 0.1
    b = _buf(continuity_radius=2.0)
    m1 = b.update([
        EntityObservation(torch.tensor([3.0, 3.0]), vT, resource_tag=1),
        EntityObservation(torch.tensor([3.0, 7.0]), vD, resource_tag=1),
    ])
    tT, tD = m1[0], m1[1]
    assert tT != tD
    # target MOVES to a new cell (within the continuity radius); distractor stays.
    m2 = b.update([
        EntityObservation(torch.tensor([4.0, 4.0]), vT.clone(), resource_tag=1),
        EntityObservation(torch.tensor([3.0, 7.0]), vD.clone(), resource_tag=1),
    ])
    assert m2[0] == tT, "moved target must keep its token (continuity re-id)"
    assert m2[1] == tD
    # type_hint wiring hook is recorded, but it is NOT the key.
    assert b.query(tT).type_hint == 1


def test_c3_persistence_through_absence_then_eviction():
    D = 16
    vT = _unit(D, 3)
    b = _buf(persist_ttl=4, continuity_radius=2.0)
    tT = b.update([EntityObservation(torch.tensor([2.0, 2.0]), vT)])[0]
    # 4 absent ticks: token persists (<= ttl).
    for _ in range(4):
        b.update([])  # nothing perceived
        assert b.query(tT) is not None
    # 5th absent tick crosses ttl -> evicted.
    b.update([])
    assert b.query(tT) is None
    assert b.get_diagnostics()["obf_n_deaths"] >= 1


def test_c4_attention_capacity_cap():
    D = 16
    b = _buf(max_tokens=3, continuity_radius=0.0)  # radius 0 -> every obs is a new token
    # birth 5 distinct entities far apart in one tick
    obs = [EntityObservation(torch.tensor([float(i * 5), 0.0]), _unit(D, 10 + i),
                             salience=1.0) for i in range(5)]
    b.update(obs)
    assert b.n_active() == 3
    assert b.get_diagnostics()["obf_n_capacity_evictions"] == 2


def test_c5_precision_weighting():
    D = 16
    vA = _unit(D, 4)
    vB = _unit(D, 5)
    b = _buf(feature_alpha=0.5, continuity_radius=5.0, use_precision_weighting=True)
    tA = b.update([EntityObservation(torch.tensor([1.0, 1.0]), vA, precision=1.0)])[0]
    before = b.query(tA).z_features.clone()
    # a zero-precision re-observation must not move the feature EMA (alpha=0).
    b.update([EntityObservation(torch.tensor([1.0, 1.0]), vB, precision=0.0)])
    after = b.query(tA).z_features
    assert torch.allclose(before, after, atol=1e-7)


def test_c6_simulation_mode_noop():
    D = 16
    b = _buf()
    out = b.update([EntityObservation(torch.tensor([2.0, 2.0]), _unit(D, 6))],
                   simulation_mode=True)
    assert out == {}
    assert b.n_active() == 0
    assert b.get_diagnostics()["obf_n_simulation_skipped"] == 1
