"""
Regression tests for the ARC-110 loop-segregation arbitration + the MECH-451
finer-channel decomposition plumbing.

Motivated by the V3-EXQ-707 autopsy (2026-06-28): ARM_DROP_LIMBIC was BYTE-IDENTICAL
to A1_LOOPS on all 6 seeds, making the C2 "limbic loop load-bearing" criterion
untestable. Two stacked defects were found:

  Defect 1 (substrate-wide):  REEAgent.select_action built its per-head finer-channel
    dict (`_fcg_channels`) gated on the TOP-LEVEL `self.config.use_finer_channel_gating`,
    which is never set anywhere in ree_core (always False), while the consumer gate at
    the e3.select() call site reads `self.config.e3.use_finer_channel_gating`. Net:
    `_fcg_channels` was always None, so MECH-451's named per-head channels
    (ofc/dacc/lpfc/vigour/liking/gated_policy) never reached the selector -- only the
    lumped residual/mech341/route did, all mapping to the default (associative) loop.
    The limbic loop received nothing, so the DROP-LIMBIC ablation was a no-op.

  Defect 2 (experiment config):  even with Defect 1 fixed, the limbic-loop input
    modules (OFC / liking / vigour) were not enabled, so the limbic loop carried no
    live channels. That is an experiment-config concern, NOT tested here.

These tests lock:
  (A) the arbitration HONOURS the loop_segregation_channel_map: dropping the limbic
      channels into associative empties the limbic loop (and can flip the committed idx);
  (B) Defect 1 directly: when `config.e3.use_finer_channel_gating` is on, the per-head
      finer channels reach the e3 selector NON-None during a real waking select; when
      off they stay None (bit-identical legacy path).
"""

import sys
from pathlib import Path

import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ree_core.utils.config import REEConfig
from ree_core.predictors.e3_selector import (
    E3TrajectorySelector,
    _FCG_CHANNEL_INDEX,
    _LOOP_DEFAULT_CHANNEL_MAP,
)
from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2


_DROP_LIMBIC_CHANNEL_MAP = {
    "dacc": "associative",
    "lpfc": "associative",
    "ofc": "associative",
    "liking": "associative",
    "vigour": "associative",
}


def _make_selector(drop_limbic: bool) -> E3TrajectorySelector:
    cfg = REEConfig.from_dims(
        body_obs_dim=8, world_obs_dim=8, action_dim=5, self_dim=32, world_dim=32,
        use_loop_segregation=True,
        use_d1_d2_population_split=True,
        loop_segregation_channel_map=(_DROP_LIMBIC_CHANNEL_MAP if drop_limbic else {}),
    )
    return E3TrajectorySelector(cfg.e3, None)


def _named_lcg_terms(n_elig: int):
    """Synthetic lcg_terms covering the named limbic + associative channels, with
    distinct per-candidate structure so each loop carries real range."""
    torch.manual_seed(0)
    terms = []
    for nm in ("ofc", "liking", "vigour", "dacc", "lpfc"):
        terms.append((_FCG_CHANNEL_INDEX[nm], torch.randn(n_elig)))
    return terms


# ------------------------------------------------------------------ #
# (A) Arbitration honours the channel-map ablation                     #
# ------------------------------------------------------------------ #

class TestSegregatedLoopArbitrateChannelMap:
    def test_drop_map_empties_limbic_loop(self):
        """With the DROP map, the limbic loop must carry NO channels (the ablation);
        without it, the limbic loop carries its default channels (ofc/liking/vigour)."""
        n = 5
        elig = torch.arange(n)
        raw = torch.randn(n)
        terms = _named_lcg_terms(n)

        sel_off = _make_selector(drop_limbic=False)
        sel_off._segregated_loop_arbitrate(
            elig, raw, terms, True, [None] * n, True, 1.0, True
        )
        d_off = sel_off.last_score_diagnostics
        assert d_off["loop_n_limbic_channels"] == 3, "default limbic loop must hold ofc/liking/vigour"
        assert d_off["loop_limbic_pref_range"] > 0.0, "default limbic loop must carry range"

        sel_on = _make_selector(drop_limbic=True)
        sel_on._segregated_loop_arbitrate(
            elig, raw, terms, True, [None] * n, True, 1.0, True
        )
        d_on = sel_on.last_score_diagnostics
        assert d_on["loop_n_limbic_channels"] == 0, "DROP map must empty the limbic loop"
        assert d_on["loop_limbic_pref_range"] == 0.0, "emptied limbic loop must carry no range"
        # The associative loop absorbs the remapped channels (2 default + 3 limbic = 5).
        assert d_on["loop_n_assoc_channels"] == 5, "associative loop must absorb the dropped limbic channels"

    def test_drop_map_changes_committed_index(self):
        """The committed within-eligible index must actually be able to differ between
        DROP-on and DROP-off when the limbic channels carry decisive structure -- this
        is the C2 prerequisite the 707 run could not exercise."""
        n = 4
        elig = torch.arange(n)
        # Flat motor (F): no within-eligible preference, so the non-motor loops decide.
        raw = torch.zeros(n)
        # Limbic channels strongly prefer candidate 3 (lowest -> best, COST convention);
        # associative-default channels prefer candidate 0. Whether limbic is its own loop
        # (off) or folded into associative (on) changes the cross-loop argmin.
        ofc = torch.tensor([0.0, 0.5, 0.5, -3.0])
        liking = torch.tensor([0.0, 0.5, 0.5, -3.0])
        vigour = torch.tensor([0.0, 0.5, 0.5, -3.0])
        dacc = torch.tensor([-3.0, 0.5, 0.5, 0.0])
        lpfc = torch.tensor([-3.0, 0.5, 0.5, 0.0])
        terms = [
            (_FCG_CHANNEL_INDEX["ofc"], ofc),
            (_FCG_CHANNEL_INDEX["liking"], liking),
            (_FCG_CHANNEL_INDEX["vigour"], vigour),
            (_FCG_CHANNEL_INDEX["dacc"], dacc),
            (_FCG_CHANNEL_INDEX["lpfc"], lpfc),
        ]
        loc_off = _make_selector(drop_limbic=False)._segregated_loop_arbitrate(
            elig, raw, terms, True, [None] * n, True, 1.0, True
        )
        loc_on = _make_selector(drop_limbic=True)._segregated_loop_arbitrate(
            elig, raw, terms, True, [None] * n, True, 1.0, True
        )
        assert loc_off != loc_on, (
            "DROP-limbic must be able to change the committed index when the limbic "
            "channels carry decisive structure (got identical -- the 707 failure mode)"
        )


# ------------------------------------------------------------------ #
# (B) Defect 1: finer per-head channels reach the e3 selector          #
# ------------------------------------------------------------------ #

class TestFinerChannelsReachSelector:
    """Directly guards Defect 1 (the agent.py top-level vs config.e3 flag mismatch)."""

    @staticmethod
    def _make_agent(env: CausalGridWorldV2, finer_on: bool) -> REEAgent:
        cfg = REEConfig.from_dims(
            body_obs_dim=env.body_obs_dim,
            world_obs_dim=env.world_obs_dim,
            action_dim=env.action_dim,
            self_dim=32, world_dim=32,
            use_finer_channel_gating=finer_on,
            use_gated_policy=True,
            use_lateral_pfc_analog=True,
            lateral_pfc_train_rule_bias_head=True,
            use_dacc=True,
        )
        return REEAgent(cfg)

    @staticmethod
    def _drive_one_select(env: CausalGridWorldV2, agent: REEAgent):
        """Run real waking ticks and capture the score_bias_channels kwarg that the
        agent passes into E3TrajectorySelector.select."""
        captured = {"seen": False, "value": "UNSET"}
        orig = E3TrajectorySelector.select

        def _patched(self, *a, **k):
            if not captured["seen"]:
                captured["value"] = k.get("score_bias_channels", "MISSING")
                captured["seen"] = True
            return orig(self, *a, **k)

        _, obs = env.reset()
        agent.reset()
        E3TrajectorySelector.select = _patched
        try:
            for _ in range(6):
                body = obs["body_state"].float()
                world = obs["world_state"].float()
                if body.dim() == 1:
                    body = body.unsqueeze(0)
                if world.dim() == 1:
                    world = world.unsqueeze(0)
                latent = agent.sense(obs_body=body, obs_world=world)
                ticks = agent.clock.advance()
                wdim = latent.z_world.shape[-1]
                e1_prior = (
                    agent._e1_tick(latent) if ticks.get("e1_tick", False)
                    else torch.zeros(1, wdim, device=agent.device)
                )
                candidates = agent.generate_trajectories(latent, e1_prior, ticks)
                action = agent.select_action(candidates, ticks)
                if captured["seen"]:
                    break
                _, _, done, _, obs = env.step(action)
                if done:
                    _, obs = env.reset()
        finally:
            E3TrajectorySelector.select = orig
        return captured

    def test_finer_on_channels_reach_selector(self):
        """Defect-1 guard: with finer gating ON, the per-head channel dict must reach
        the selector NON-None (pre-fix it was always None)."""
        env = CausalGridWorldV2(seed=0, size=8, num_hazards=2, num_resources=3)
        agent = self._make_agent(env, finer_on=True)
        captured = self._drive_one_select(env, agent)
        assert captured["seen"], "selector.select was never called during the waking ticks"
        sbc = captured["value"]
        assert sbc is not None and sbc != "MISSING", (
            "score_bias_channels reached the selector as None/absent with finer gating ON "
            "-- Defect 1 has regressed (agent gate must read config.e3.use_finer_channel_gating)"
        )
        # At least one named cortical channel must be present (dacc/lpfc/gated_policy are
        # the channels active in this minimal config).
        assert len(sbc) >= 1, "finer per-head channel dict reached the selector empty"

    def test_finer_off_channels_none(self):
        """Backward-compat: with finer gating OFF, the channel dict must stay None
        (bit-identical legacy single-channel path)."""
        env = CausalGridWorldV2(seed=0, size=8, num_hazards=2, num_resources=3)
        agent = self._make_agent(env, finer_on=False)
        captured = self._drive_one_select(env, agent)
        assert captured["seen"], "selector.select was never called during the waking ticks"
        sbc = captured["value"]
        assert sbc is None or sbc == "MISSING", (
            "finer gating OFF must leave score_bias_channels None (got a populated dict)"
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
