"""Probe 3 (short, line-buffered) for V3-EXQ-1056 pre-flight.

Q1 does use_dacc=True restore the choice_difficulty axis?
Q2 does any commitment (beta rising edge) occur in an ecological loop?
Q3 duty cycle of stuck_score (G9 pole B)
Q4 ARM-1 parity: flag=True/widen=0/gain=0 vs flag=False
"""
from __future__ import annotations
import sys, random, math
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import numpy as np, torch
from collections import Counter
from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig

ENV_KWARGS = dict(size=8, num_resources=3, num_hazards=2, use_proxy_fields=True)
T = 100


def ent(c):
    cl = []
    for x in c:
        try:
            cl.append(int(torch.argmax(x.actions[0, 0, :]).item()))
        except Exception:
            pass
    if not cl:
        return 0.0
    n = len(cl)
    e = 0.0
    for k in Counter(cl).values():
        p = k / n
        e -= p * math.log(p)
    return e


def run(seed, dgpe, widen, gain, dacc, tag):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    env = CausalGridWorldV2(seed=seed, scheduled_action_block_enabled=True,
                            scheduled_action_block_interval=3,
                            scheduled_action_block_prob=1.0, **ENV_KWARGS)
    cfg = REEConfig.goal_stream(
        body_obs_dim=env.body_obs_dim, world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim, alpha_world=0.9,
        use_difficulty_gated_proposal_entropy=dgpe,
        dgpe_candidate_widen_max=widen, dgpe_temperature_gain_max=gain,
        use_differentiable_cem=True, use_dacc=dacc,
        use_sleep_loop=False, sws_enabled=False, rem_enabled=False,
        use_sleep_aggregation_cluster=False)
    ag = REEAgent(cfg)
    ag.reset()
    rec = []
    o = ag.hippocampal.propose_trajectories

    def w(*a, **k):
        r = o(*a, **k)
        rec.append((len(r), round(ent(r), 6)))
        return r

    ag.hippocampal.propose_trajectories = w
    seen = Counter()
    if ag.stuck_state_detector is not None:
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
    ss, acts, rises, prev = [], [], 0, False
    for _t in range(T):
        a_ = ag.act_with_split_obs(od["body_state"], od["world_state"])
        a_t = a_[0] if isinstance(a_, tuple) else a_
        acts.append(int(torch.argmax(a_t[0]).item()))
        _, h, d, inf, od = env.step(a_t)
        ss.append(ag._last_stuck_score)
        cur = ag.beta_gate.is_elevated
        if cur and not prev:
            rises += 1
        prev = cur
        if d:
            _, od = env.reset()
    duty = sum(1 for v in ss if v >= 0.5) / len(ss)
    print("[%s] props=%d stuck %.4f..%.4f duty=%.3f beta_rises=%d dacc_bundle=%s"
          % (tag, len(rec), min(ss), max(ss), duty, rises,
             getattr(ag, "_dacc_last_bundle", None) is not None), flush=True)
    print("   inputs:", dict(seen), flush=True)
    print("   n_cand:", sorted({r[0] for r in rec}),
          " ent[:6]:", [r[1] for r in rec[:6]], flush=True)
    return acts, [r[1] for r in rec]


run(42, True, 8, 1.0, True, "ARM2_dacc_on")
a1, e1 = run(42, True, 0, 0.0, True, "ARM1_gains_zero")
a2, e2 = run(42, False, 8, 1.0, True, "ARM0_flag_off")
print("PARITY actions_identical=%s entropy_identical=%s" % (a1 == a2, e1 == e2), flush=True)
