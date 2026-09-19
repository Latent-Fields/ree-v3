"""Probe 7: does the SD-061 (c) axis mask actually make the gate fire in the
ecology where mean-over-present never fired?

2026-09-18 baseline (GFLAG-0352): with 2 of 4 axes present, stuck_score pinned
at exactly stuck_threshold and duty(is_stuck) was 0.000 in every arm.
Here: same loop, declaring only the axes that actually arrive.
"""
from __future__ import annotations
import sys, random
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import numpy as np, torch
from ree_core.agent import REEAgent
from ree_core.cingulate.stuck_state_detector import StuckStateAxisUnavailable
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig

ENV_KWARGS = dict(size=8, num_resources=3, num_hazards=2, use_proxy_fields=True)
T = 100


def run(declared, tag, seed=42):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    env = CausalGridWorldV2(seed=seed, scheduled_action_block_enabled=True,
                            scheduled_action_block_interval=3,
                            scheduled_action_block_prob=1.0, **ENV_KWARGS)
    cfg = REEConfig.goal_stream(
        body_obs_dim=env.body_obs_dim, world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim, alpha_world=0.9,
        use_difficulty_gated_proposal_entropy=True,
        dgpe_candidate_widen_max=8, dgpe_temperature_gain_max=1.0,
        stuck_declared_axes=declared,
        use_sleep_loop=False, sws_enabled=False, rem_enabled=False,
        use_sleep_aggregation_cluster=False)
    ag = REEAgent(cfg); ag.reset()
    _, od = env.reset()
    if ag.goal_state is not None:
        ag.goal_state._z_goal = torch.ones(1, ag.goal_state.config.goal_dim) * 0.5
    ss = []
    try:
        for _ in range(T):
            a_ = ag.act_with_split_obs(od["body_state"], od["world_state"])
            a_t = a_[0] if isinstance(a_, tuple) else a_
            _, h, d, inf, od = env.step(a_t)
            ss.append(ag._last_stuck_score)
            if d:
                _, od = env.reset()
    except StuckStateAxisUnavailable as e:
        print("[%s] REFUSED (this is the designed behaviour): %s"
              % (tag, str(e).split(". ")[0]), flush=True)
        return
    st = ag.stuck_state_detector.get_state()
    duty = sum(1 for v in ss if v >= 0.5) / len(ss)
    print("[%s] declared=%s  stuck %.4f..%.4f  duty(is_stuck)=%.3f  "
          "advanced_ticks=%d undetermined=%d"
          % (tag, st["sd061_declared_axes"], min(ss), max(ss), duty,
             st["sd061_n_ticks"], st["sd061_n_undetermined_ticks"]), flush=True)
    print("   decays after peak? peak_idx=%d final=%.4f peak=%.4f"
          % (ss.index(max(ss)), ss[-1], max(ss)), flush=True)


run(None, "LEGACY mean-over-present (the 2026-09-18 baseline)")
run(["progress"], "DECLARED progress only  (stall axis, MECH-527's trigger)")
run(["progress", "margin"], "DECLARED progress+margin (= the axes that arrive)")
run(["progress", "difficulty"], "DECLARED progress+difficulty (difficulty never arrives)")
