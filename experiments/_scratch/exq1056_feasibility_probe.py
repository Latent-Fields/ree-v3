"""Step 2.5a empirical probe for V3-EXQ-1056 (SD-061 / MECH-343 upstream leg).

Throwaway. Confirms at RUNTIME what the source read implies:
  P1 use_differentiable_cem default state (is SD-061's temperature lever live?)
  P2 detector + regulator instantiate; per-axis deficits exposed
  P3 widen=0/gain=0 -> identity gain for every stuck_score (the ARM-1 construction)
  P4 an ecological loop feeds the detector non-None inputs and moves stuck_score
  P5 the proposal-layer candidate set is readable per proposal via a wrapper
"""
from __future__ import annotations
import sys, random
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import numpy as np, torch
from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig

ENV_KWARGS = dict(size=8, num_resources=3, num_hazards=2, use_proxy_fields=True)


def build(seed, dgpe, widen, gain, diff_cem=False, extra=None):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    env = CausalGridWorldV2(seed=seed, **ENV_KWARGS)
    kw = dict(
        body_obs_dim=env.body_obs_dim, world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim, alpha_world=0.9,
        use_difficulty_gated_proposal_entropy=dgpe,
        dgpe_candidate_widen_max=widen, dgpe_temperature_gain_max=gain,
        use_differentiable_cem=diff_cem,
        use_sleep_loop=False, sws_enabled=False, rem_enabled=False,
        use_sleep_aggregation_cluster=False,
    )
    if extra:
        kw.update(extra)
    cfg = REEConfig.goal_stream(**kw)
    a = REEAgent(cfg); a.reset()
    return env, a


def main():
    env, a = build(42, True, 8, 1.0)
    print("P1 hippocampal.use_differentiable_cem =",
          getattr(a.hippocampal.config, "use_differentiable_cem", "MISSING"))
    print("P1 hippocampal.differentiable_cem_temperature =",
          getattr(a.hippocampal.config, "differentiable_cem_temperature", "MISSING"))
    print("P1 hippocampal.config.num_candidates =", a.hippocampal.config.num_candidates)
    print("P2 detector =", type(a.stuck_state_detector).__name__,
          "regulator =", type(a.difficulty_gated_proposal_entropy).__name__)
    print("P2 detector state keys =", sorted(a.stuck_state_detector.get_state().keys()))
    print("P2 goal_state present =", a.goal_state is not None,
          "active =", a.goal_state.is_active() if a.goal_state is not None else None)

    _, a0 = build(42, True, 0, 0.0)
    for s in (0.0, 0.25, 0.5, 1.0):
        print("P3 gain at s=%.2f ->" % s,
              a0.difficulty_gated_proposal_entropy.compute_proposal_gain(s))

    # P4/P5: ecological loop with a wrapped proposer.
    env, ag = build(42, True, 8, 1.0, diff_cem=True)
    rec = []
    orig = ag.hippocampal.propose_trajectories
    def wrapped(*args, **kw):
        out = orig(*args, **kw)
        rec.append((len(out), float(getattr(ag.hippocampal.config,
                                            "differentiable_cem_temperature", -1.0))))
        return out
    ag.hippocampal.propose_trajectories = wrapped
    _, od = env.reset()
    if ag.goal_state is not None:
        ag.goal_state._z_goal = torch.ones(1, ag.goal_state.config.goal_dim) * 0.5
    seen_inputs = {"prox": 0, "margin": 0, "cls": 0, "diff": 0, "sal": 0}
    orig_upd = ag.stuck_state_detector.update
    def upd(**kw):
        for k, n in (("goal_proximity", "prox"), ("score_margin", "margin"),
                     ("committed_action_class", "cls"), ("choice_difficulty", "diff"),
                     ("goal_salience", "sal")):
            if kw.get(k) is not None:
                seen_inputs[n] += 1
        return orig_upd(**kw)
    ag.stuck_state_detector.update = upd
    scores = []
    for t in range(60):
        act = ag.act_with_split_obs(od["body_state"], od["world_state"])
        a_t = act[0] if isinstance(act, tuple) else act
        _, h, d, inf, od = env.step(a_t)
        scores.append(ag._last_stuck_score)
        if d:
            _, od = env.reset()
    print("P4 detector inputs non-None counts over 60 ticks:", seen_inputs)
    print("P4 stuck_score min/max/last = %.4f / %.4f / %.4f" %
          (min(scores), max(scores), scores[-1]))
    print("P4 detector n_ticks =", ag.stuck_state_detector.get_state()["sd061_n_ticks"])
    print("P4 per-axis deficits =", {k: round(v, 4) for k, v in
          ag.stuck_state_detector.get_state().items() if k.startswith("last_deficit")})
    print("P5 proposals recorded =", len(rec), "first 5 (n_cand, temp) =", rec[:5])
    print("P5 distinct candidate counts =", sorted({r[0] for r in rec}))
    print("P5 distinct temps seen INSIDE proposal =", sorted({r[1] for r in rec}))
    print("P5 beta elevated ever =", ag.beta_gate.get_state())


if __name__ == "__main__":
    main()
