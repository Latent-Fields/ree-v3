"""Contract tests for the MECH-320 no-op score-margin DV hook (2026-09-24).

WHY
---
MECH-320's committed DV (action_density) sat at exactly 1.0000 in every
V3-EXQ-951/951a cell, so it had no headroom. This hook records a CONTINUOUS
readout from E3TrajectorySelector.select: how far the no-op class was from
winning selection. It is a SECONDARY DV under MECH-320's current
what_would_answer (committed action density stays load-bearing).

TAUTOLOGY GUARD. The MECH-320 bias is composed into E3 scores BEFORE
selection. In the additive form it adds +w_passive*v_t to no-op candidates and
-w_action*v_t to action candidates, so a POST-bias margin shift between arms
is the injected manipulation read back. The hook therefore records the margin
PRE-bias (raw_scores) AND POST-bias; a criterion must use a v_t-dependent
relation, never a between-arm POST offset.

Lever: REEConfig.tonic_vigor_record_noop_margin (default False). When True the
agent passes noop_class=tonic_vigor_noop_class to E3.select. Independent of
use_tonic_vigor so a vigor-OFF control arm still records it.

CONTRACTS
  N1: default False on REEConfig and from_dims; a default agent never sends
      noop_class and E3's margin fields stay None.
  N2: from_dims reaches the dataclass field (the MECH-307 swallowed-kwarg trap).
  N3: select(noop_class=None) leaves every margin field None (default path).
  N4: _record_noop_margin arithmetic on hand-set scores: signed candidate
      margin (NEGATIVE when the committed candidate is not the argmin),
      selected no-op excluded from the alternative set, class-level action
      gap, PRE and POST read from their own tensors, None when a class is
      absent.
  N5: BIT-IDENTITY -- ON vs OFF produce identical action streams, with the
      tonic-vigor bias both off and on (pure diagnostic).
  N6: LIVENESS on a real rollout -- ON populates the dict on every tick, and
      with an active additive MECH-320 bias (authority OFF, vigor the sole
      channel) the POST - PRE action-gap shift is exactly (w_a + w_p) * v_t.
  N7: with use_modulatory_selection_authority ON (the 951-lineage bed) the
      POST - PRE shift is INVARIANT to v_t magnitude -- pinned so no criterion
      is designed on the authority-OFF identity above.
  N8: select_seq increments once per recorded select (held ticks do not
      re-stamp); post_is_selection_basis is True on the default argmin path.
  N9: the REINFORCE path (act_with_log_prob) forwards noop_class too.
"""

import random

import numpy as np
import pytest
import torch

from ree_core.utils.config import REEConfig
from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.predictors.e3_selector import E3TrajectorySelector
from experiments._harness import StepHarness


def _mk_env(seed=0):
    return CausalGridWorldV2(size=8, num_hazards=2, num_resources=3, seed=seed)


def _dims(env):
    return dict(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
    )


def _run(cfg, steps=12, seed=0, capture=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    env = _mk_env(seed=seed)
    agent = REEAgent(cfg)
    ticks = []
    if capture:
        orig = agent.e3.select

        def _wrapped(*a, **kw):
            ticks.append(("noop_class" in kw, kw.get("noop_class")))
            res = orig(*a, **kw)
            ticks[-1] = ticks[-1] + (agent.e3.last_noop_margin,)
            return res

        agent.e3.select = _wrapped
    results = StepHarness(agent, env, train_mode=True, seed=seed).run_episode(
        max_steps=steps
    )
    actions = [int(r.action.argmax().item()) for r in results]
    return agent, actions, ticks


class _Cand:
    """Minimal stand-in: _record_noop_margin reads only .actions."""

    def __init__(self, cls, action_dim=5):
        a = torch.zeros(1, 2, action_dim)
        a[0, 0, cls] = 1.0
        self.actions = a


def _selector_like():
    # _record_noop_margin is pure over its arguments + writes attributes; bind
    # it to a bare object so the arithmetic is tested without constructing E3.
    obj = E3TrajectorySelector.__new__(E3TrajectorySelector)
    obj._noop_margin_seq = 0
    return obj


def test_n1_default_off():
    env = _mk_env()
    assert REEConfig().tonic_vigor_record_noop_margin is False
    cfg = REEConfig.from_dims(**_dims(env))
    assert cfg.tonic_vigor_record_noop_margin is False
    agent, _, ticks = _run(cfg, steps=4, capture=True)
    assert ticks, "no E3 selection happened"
    assert all(t[0] is False for t in ticks), "default path sent noop_class"
    assert agent.e3.last_noop_margin is None
    assert agent.e3.last_noop_candidate_margin is None
    assert agent.e3.last_noop_candidate_margin_pre is None
    assert agent.e3.last_noop_candidate_present is None


def test_n2_from_dims_reaches_field():
    env = _mk_env()
    cfg = REEConfig.from_dims(tonic_vigor_record_noop_margin=True, **_dims(env))
    assert cfg.tonic_vigor_record_noop_margin is True


def test_n3_select_without_noop_class_leaves_none():
    env = _mk_env()
    cfg = REEConfig.from_dims(**_dims(env))
    agent, _, _ = _run(cfg, steps=2)
    # Poison the fields, then run one more default-path tick: select must
    # reset them to None rather than leave a stale latch.
    agent.e3.last_noop_margin = {"stale": True}
    agent.e3.last_noop_candidate_margin = 123.0
    torch.manual_seed(1)
    StepHarness(agent, _mk_env(seed=1), train_mode=True, seed=1).run_episode(
        max_steps=1
    )
    assert agent.e3.last_noop_margin is None
    assert agent.e3.last_noop_candidate_margin is None


def test_n4_arithmetic():
    sel = _selector_like()
    # classes: 0=noop, 1..4 action. Candidates: [noop, act, noop, act]
    cands = [_Cand(0), _Cand(2), _Cand(0), _Cand(3)]
    pre = torch.tensor([1.0, 2.0, 1.5, 0.5])
    post = torch.tensor([1.3, 1.8, 1.8, 0.3])
    # Committed candidate 1 (an action, NOT the post argmin) -> negative margin.
    sel._record_noop_margin(cands, pre, post, selected_idx=1, noop_class=0)
    d = sel.last_noop_margin
    assert d["n_noop"] == 2 and d["n_candidates"] == 4
    assert d["selected_is_noop"] is False
    assert d["noop_candidate_present"] is True
    assert d["candidate_margin_pre"] == pytest.approx(1.0 - 2.0)
    assert d["candidate_margin_post"] == pytest.approx(1.3 - 1.8)
    assert d["candidate_margin_post"] < 0
    assert d["action_gap_pre"] == pytest.approx(1.0 - 0.5)
    assert d["action_gap_post"] == pytest.approx(1.3 - 0.3)
    assert sel.last_noop_candidate_margin == d["candidate_margin_post"]
    assert sel.last_noop_candidate_margin_pre == d["candidate_margin_pre"]

    # Committed candidate IS a no-op: it is excluded from the alternative set.
    sel._record_noop_margin(cands, pre, post, selected_idx=0, noop_class=0)
    d = sel.last_noop_margin
    assert d["selected_is_noop"] is True
    assert d["candidate_margin_post"] == pytest.approx(1.8 - 1.3)

    # Only one no-op, and it was committed -> no alternative no-op.
    cands2 = [_Cand(0), _Cand(1), _Cand(4)]
    s2 = torch.tensor([0.1, 0.2, 0.3])
    sel._record_noop_margin(cands2, s2, s2, selected_idx=0, noop_class=0)
    assert sel.last_noop_candidate_present is False
    assert sel.last_noop_candidate_margin is None
    assert sel.last_noop_margin["action_gap_post"] == pytest.approx(0.1 - 0.2)

    # No no-op class at all -> both DVs None, not 0.0.
    cands3 = [_Cand(1), _Cand(2)]
    sel._record_noop_margin(cands3, s2[:2], s2[:2], selected_idx=0, noop_class=0)
    assert sel.last_noop_margin["candidate_margin_post"] is None
    assert sel.last_noop_margin["action_gap_post"] is None


@pytest.mark.parametrize("tonic", [False, True])
def test_n5_bit_identical_on_off(tonic):
    env = _mk_env()
    extra = dict(use_tonic_vigor=True, tonic_vigor_v_t_floor=0.5) if tonic else {}
    # noop_class=4 (stay) so the ON recorder actually exercises both classes.
    extra["tonic_vigor_noop_class"] = 4
    cfg_off = REEConfig.from_dims(**_dims(env), **extra)
    cfg_on = REEConfig.from_dims(
        tonic_vigor_record_noop_margin=True, **_dims(env), **extra
    )
    _, a_off, _ = _run(cfg_off, steps=10, seed=3)
    _, a_on, _ = _run(cfg_on, steps=10, seed=3)
    assert a_off == a_on


def test_n6_liveness_real_rollout():
    env = _mk_env()
    cfg = REEConfig.from_dims(
        tonic_vigor_record_noop_margin=True,
        use_tonic_vigor=True,
        tonic_vigor_form="additive",
        tonic_vigor_v_t_floor=0.5,
        # CausalGridWorldV2 action 4 is "stay". The config default (0) is a
        # MOVE class here and never appears as a no-op candidate class, so the
        # DV is unreadable at the default -- measured: 0 no-op candidates per
        # tick at noop_class=0 vs ~25/32 at 4.
        tonic_vigor_noop_class=4,
        **_dims(env),
    )
    _, _, ticks = _run(cfg, steps=30, seed=0, capture=True)
    assert ticks
    assert all(t[0] is True for t in ticks), "flag ON but noop_class not sent"
    dicts = [t[2] for t in ticks]
    assert all(isinstance(d, dict) for d in dicts), "margin not recorded every tick"
    both = [d for d in dicts if d["action_gap_post"] is not None]
    assert both, "no tick had both no-op and action candidates -- DV unreadable"
    shifts = [d["action_gap_post"] - d["action_gap_pre"] for d in both]
    # The additive bias lowers action scores and raises no-op scores, so the
    # POST-bias no-op-minus-best-action gap must exceed the PRE-bias one.
    assert max(shifts) > 1e-6, shifts
    # ...and on this bed (authority OFF by default, v_t at its floor on the
    # first select, vigor the only modulatory channel) that shift is EXACTLY
    # the injected bias,
    # (w_action + w_passive) * v_t, which is why the PRE-bias read exists.
    tv = REEConfig.from_dims(**_dims(env))
    expected = (tv.tonic_vigor_w_action + tv.tonic_vigor_w_passive) * 0.5
    assert shifts[0] == pytest.approx(expected, abs=1e-4), (shifts, expected)


def _shifts(v_t_floor, authority):
    env = _mk_env()
    kw = dict(
        tonic_vigor_record_noop_margin=True,
        use_tonic_vigor=True,
        tonic_vigor_v_t_floor=v_t_floor,
        tonic_vigor_noop_class=4,
    )
    if authority:
        kw.update(use_modulatory_selection_authority=True,
                  modulatory_authority_gain=0.5)
    cfg = REEConfig.from_dims(**kw, **_dims(env))
    _, _, ticks = _run(cfg, steps=20, seed=0, capture=True)
    ds = [t[2] for t in ticks if t[2]["action_gap_post"] is not None]
    assert ds
    return [d["action_gap_post"] - d["action_gap_pre"] for d in ds]


def test_n7_authority_on_shift_is_vt_invariant():
    # Floors kept below the point where w * v_t reaches the per-candidate bias
    # cap (tonic_vigor_bias_scale 0.1): measured 0.5 -> 0.1, 2.0 -> 0.2 (capped).
    off_lo, off_hi = _shifts(0.2, False), _shifts(0.4, False)
    # Authority OFF: the shift scales with v_t (sanity -- the lever is live).
    assert off_hi[0] == pytest.approx(2.0 * off_lo[0], rel=1e-3)
    on_lo, on_hi = _shifts(0.2, True), _shifts(0.4, True)
    n = min(len(on_lo), len(on_hi))
    assert n >= 1
    # Authority ON: POST - PRE is rescaled to gain * raw_score_range and does
    # not move with v_t magnitude.
    for a, b in zip(on_lo[:n], on_hi[:n]):
        assert a == pytest.approx(b, rel=1e-4, abs=1e-6), (on_lo, on_hi)


def test_n8_select_seq_and_basis():
    env = _mk_env()
    cfg = REEConfig.from_dims(
        tonic_vigor_record_noop_margin=True, tonic_vigor_noop_class=4,
        **_dims(env),
    )
    _, _, ticks = _run(cfg, steps=20, seed=0, capture=True)
    seqs = [t[2]["select_seq"] for t in ticks]
    assert seqs == list(range(1, len(ticks) + 1)), seqs
    assert all(t[2]["post_is_selection_basis"] is True for t in ticks)


def test_n9_act_with_log_prob_forwards_noop_class():
    env = _mk_env()
    cfg = REEConfig.from_dims(
        tonic_vigor_record_noop_margin=True, tonic_vigor_noop_class=4,
        **_dims(env),
    )
    torch.manual_seed(0)
    agent = REEAgent(cfg)
    seen = []
    orig = agent.e3.select

    def _wrapped(*a, **kw):
        seen.append(kw.get("noop_class"))
        return orig(*a, **kw)

    agent.e3.select = _wrapped
    _, obs_dict = env.reset()
    agent.reset()
    flat = torch.cat([
        torch.as_tensor(obs_dict["body_state"]).float().reshape(-1),
        torch.as_tensor(obs_dict["world_state"]).float().reshape(-1),
    ])
    agent.act_with_log_prob(flat)
    assert seen == [4]
    assert isinstance(agent.e3.last_noop_margin, dict)
