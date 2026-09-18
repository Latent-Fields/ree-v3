"""Probe 4: verify the (b) answer and the (a) implementation preconditions.

B1 does use_dacc=True + use_affective_harm_stream=True populate _dacc_last_bundle,
   and does choice_difficulty then reach StuckStateDetector.update?
B2 is use_dacc=True ALONE insufficient (the current state)?
A1 is HippocampalModule.config the SAME object as REEConfig.hippocampal?
A2 is use_differentiable_cem read only at runtime (so a post-construction set works)?
"""
from __future__ import annotations
import sys, random
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import numpy as np, torch
from collections import Counter
from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig

ENV_KWARGS = dict(size=8, num_resources=3, num_hazards=2, use_proxy_fields=True)


def build(seed, **extra):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    env = CausalGridWorldV2(seed=seed, scheduled_action_block_enabled=True,
                            scheduled_action_block_interval=3,
                            scheduled_action_block_prob=1.0, **ENV_KWARGS)
    cfg = REEConfig.goal_stream(
        body_obs_dim=env.body_obs_dim, world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim, alpha_world=0.9,
        use_difficulty_gated_proposal_entropy=True,
        dgpe_candidate_widen_max=8, dgpe_temperature_gain_max=1.0,
        use_sleep_loop=False, sws_enabled=False, rem_enabled=False,
        use_sleep_aggregation_cluster=False, **extra)
    a = REEAgent(cfg); a.reset()
    return env, a, cfg


def loop(env, ag, n=60):
    seen = Counter()
    ou = ag.stuck_state_detector.update

    def upd(**k):
        for key in ("goal_proximity", "score_margin", "committed_action_class",
                    "choice_difficulty", "goal_salience"):
            if k.get(key) is not None:
                seen[key] += 1
        return ou(**k)

    ag.stuck_state_detector.update = upd
    _, od = env.reset()
    if ag.goal_state is not None:
        ag.goal_state._z_goal = torch.ones(1, ag.goal_state.config.goal_dim) * 0.5
    ss = []
    for _ in range(n):
        a_ = ag.act_with_split_obs(od["body_state"], od["world_state"])
        a_t = a_[0] if isinstance(a_, tuple) else a_
        _, h, d, inf, od = env.step(a_t)
        ss.append(ag._last_stuck_score)
        if d:
            _, od = env.reset()
    return seen, ss


print("=== B2: use_dacc=True ALONE ===", flush=True)
env, ag, cfg = build(42, use_dacc=True)
print("   agent.dacc is None?", ag.dacc is None, flush=True)
print("   latent.use_affective_harm_stream =",
      getattr(cfg.latent, "use_affective_harm_stream", "MISSING"), flush=True)
seen, ss = loop(env, ag)
print("   _dacc_last_bundle set?", getattr(ag, "_dacc_last_bundle", None) is not None, flush=True)
print("   inputs:", dict(seen), flush=True)
print("   stuck max=%.4f duty=%.3f" % (max(ss), sum(1 for v in ss if v >= 0.5)/len(ss)), flush=True)

print("=== B1: use_dacc=True AND use_affective_harm_stream=True ===", flush=True)
env, ag, cfg = build(42, use_dacc=True, use_affective_harm_stream=True)
print("   agent.dacc is None?", ag.dacc is None, flush=True)
print("   affective_harm_encoder is None?",
      getattr(ag.latent_stack, "affective_harm_encoder", "NO_ATTR") is None
      if hasattr(ag, "latent_stack") else "no latent_stack attr", flush=True)
seen, ss = loop(env, ag)
print("   _dacc_last_bundle set?", getattr(ag, "_dacc_last_bundle", None) is not None, flush=True)
print("   inputs:", dict(seen), flush=True)
print("   stuck max=%.4f duty=%.3f" % (max(ss), sum(1 for v in ss if v >= 0.5)/len(ss)), flush=True)
print("   deficits:", {k: round(v, 4) for k, v in ag.stuck_state_detector.get_state().items()
                       if k.startswith("last_deficit")}, flush=True)

print("=== A1/A2: config identity + runtime-read ===", flush=True)
_, ag2, cfg2 = build(42)
print("   hippocampal.config IS cfg.hippocampal:", ag2.hippocampal.config is cfg2.hippocampal, flush=True)
print("   before:", ag2.hippocampal.config.use_differentiable_cem, flush=True)
ag2.hippocampal.config.use_differentiable_cem = True
print("   after post-construction set:", ag2.hippocampal.config.use_differentiable_cem, flush=True)
