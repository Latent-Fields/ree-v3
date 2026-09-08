#!/opt/local/bin/python3
"""Feasibility probe for V3-EXQ-642d's six mandatory repairs.

Uses V3-EXQ-642c's OWN module (and therefore 642a's builders, ENV_KWARGS,
CFG_KWARGS + CALIBRATION_CONFIG, GOAL_PIN and P0 schedule) so the harness is
faithful by construction rather than by re-derivation. The only deltas are the
ones V3-EXQ-642d is specified to introduce.

Questions, from failure_autopsy_V3-EXQ-642c_2026-09-05 recommended_next_experiments:
  (1) action_dim=5 + blocked_agency_noop_class=4 -- does the DV survive it?
  (2) num_hazards > 0 -- does z_harm_a actually become free to vary?
  (4) a SELF-attributable cancellation arm -- does it push motor_agency below
      the 0.5 attribution floor? (the autopsy calls this "the only arm that
      ... exercises the attribution clause at all")

Writes nothing. ~10 min.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from experiments import v3_exq_642c_blocked_agency_headroom_dv_validation as c642  # noqa: E402

base = c642.base
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402

P0_EPISODES = 30          # half of base.P0_WARMUP_EPISODES; enough to reach its regime
STEPS = base.STEPS_PER_EPISODE
MEASURE_STEPS = 900


def _one_hot(idx: int, action_dim: int) -> torch.Tensor:
    a = torch.zeros(1, action_dim)
    a[0, idx] = 1.0
    return a


def _build(action_dim: int, num_hazards: int, noop_class, seed: int):
    """base._build_env / _build_agent, with the 642d deltas applied."""
    env_kwargs = dict(base.ENV_KWARGS)
    env_kwargs["num_hazards"] = num_hazards

    def env_(block: bool):
        return CausalGridWorldV2(
            seed=seed,
            scheduled_action_block_enabled=block,
            scheduled_action_block_interval=base.BLOCK_INTERVAL,
            scheduled_action_block_prob=1.0,
            **env_kwargs,
        )

    e = env_(False)
    cfg_kwargs = dict(base.CFG_KWARGS)  # already carries CALIBRATION_CONFIG
    if noop_class is not None:
        cfg_kwargs["blocked_agency_noop_class"] = noop_class
    cfg = REEConfig.from_dims(
        body_obs_dim=e.body_obs_dim,
        world_obs_dim=e.world_obs_dim,
        action_dim=action_dim,
        **cfg_kwargs,
    )
    return env_, REEAgent(cfg)


def _p0(agent, env_, action_dim, seed, train_self):
    """base._train_world_forward, optionally also training E2's SELF forward."""
    e = env_(False)
    _, od = e.reset()
    base._pin_goal(agent)
    params = (
        list(agent.e2.world_transition.parameters())
        + list(agent.e2.world_action_encoder.parameters())
    )
    if train_self:
        params += (
            list(agent.e2.self_transition.parameters())
            + list(agent.e2.self_action_encoder.parameters())
        )
    opt = torch.optim.Adam(params, lr=3e-3)
    lat = agent.sense(od["body_state"], od["world_state"])
    pw = lat.z_world.detach().clone()
    ps = lat.z_self.detach().clone()
    rng = np.random.RandomState(seed)
    last = 0.0
    for _ep in range(P0_EPISODES):
        for _ in range(STEPS):
            a = _one_hot(int(rng.randint(0, 4)), action_dim)
            _, _h, d, _inf, od = e.step(a)
            lat = agent.sense(od["body_state"], od["world_state"])
            cw = lat.z_world.detach().clone()
            cs = lat.z_self.detach().clone()
            loss = ((agent.e2.world_forward(pw, a) - cw) ** 2).mean()
            if train_self:
                loss = loss + ((agent.e2.predict_next_self(ps, a) - cs) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            last = float(loss.item())
            pw, ps = cw.clone(), cs.clone()
            if d:
                _, od = e.reset()
                base._pin_goal(agent)
                lat = agent.sense(od["body_state"], od["world_state"])
                pw = lat.z_world.detach().clone()
                ps = lat.z_self.detach().clone()
    return last


def _measure(agent, env_, action_dim, noop_class, mode):
    """mode: FREE | EXTERNAL (scheduled block) | SELF (body fails -> executes no-op)."""
    e = env_(mode == "EXTERNAL")
    agent.reset()
    _, od = e.reset()
    base._pin_goal(agent)
    agent.sense(od["body_state"], od["world_state"])
    rng = np.random.RandomState(7)
    zb, om, ma, fl, zh, pm = [], [], [], [], [], []
    n_env_blocked = 0
    n_ext_fire = 0
    for _ in range(MEASURE_STEPS):
        idx = int(rng.randint(0, 4))          # always a real MOVE, never the no-op
        intended = _one_hot(idx, action_dim)
        agent._last_action = intended.clone()
        if mode == "SELF" and noop_class is not None:
            executed = _one_hot(noop_class, action_dim)   # efference says move; body stays
        else:
            executed = intended
        _, _h, d, inf, od = e.step(executed)
        lat = agent.sense(od["body_state"], od["world_state"])
        o = agent.blocked_agency.last_output()
        zb.append(float(o.z_block))
        om.append(float(o.outcome_mismatch))
        ma.append(float(o.motor_agency))
        fl.append(float(o.effective_outcome_mismatch_floor))
        zh.append(
            float(lat.z_harm_a.detach().norm().item())
            if lat.z_harm_a is not None else 0.0
        )
        if inf.get("action_blocked_this_step", False):
            n_env_blocked += 1
        if bool(o.external_block_this_tick):
            n_ext_fire += 1
        if d:
            _, od = e.reset()
            agent.reset()
            base._pin_goal(agent)
            agent.sense(od["body_state"], od["world_state"])
    ma_a = np.array(ma)
    return {
        "z_block_mean": float(np.mean(zb)), "z_block_max": float(np.max(zb)),
        "mismatch_mean": float(np.mean(om)), "mismatch_max": float(np.max(om)),
        "floor_mean": float(np.mean(fl)),
        "motor_mean": float(ma_a.mean()), "motor_min": float(ma_a.min()),
        "motor_frac_below_floor": float((ma_a < 0.5).mean()),
        "z_harm_a_mean": float(np.mean(zh)), "z_harm_a_sd": float(np.std(zh)),
        "n_env_blocked": n_env_blocked, "n_ext_fire": n_ext_fire,
    }


def _row(tag, r):
    print(
        f"{tag:26s} zblk mean={r['z_block_mean']:.4f} max={r['z_block_max']:.4f} | "
        f"mism mean={r['mismatch_mean']:.4f} max={r['mismatch_max']:.4f} "
        f"floor={r['floor_mean']:.4f} | motor mean={r['motor_mean']:.3f} "
        f"min={r['motor_min']:.3f} frac<0.5={r['motor_frac_below_floor']:.3f} | "
        f"zha sd={r['z_harm_a_sd']:.4f} | envblk={r['n_env_blocked']} "
        f"extfire={r['n_ext_fire']}",
        flush=True,
    )


def config(tag, action_dim, num_hazards, noop_class, train_self, modes, seed=42):
    print(f"\n=== {tag} (action_dim={action_dim}, num_hazards={num_hazards}, "
          f"noop_class={noop_class}, self_P0={train_self}) ===", flush=True)
    base._seed_all(seed)
    env_, agent = _build(action_dim, num_hazards, noop_class, seed)
    p0 = _p0(agent, env_, action_dim, seed, train_self)
    print(f"  P0 loss={p0:.3e}", flush=True)
    for m in modes:
        _row(m, _measure(agent, env_, action_dim, noop_class, m))


if __name__ == "__main__":
    # A: faithful 642c control -- must reproduce z_block firing, else the probe is wrong.
    config("A 642c-faithful", 4, 0, None, False, ["FREE", "EXTERNAL"])
    # B: repair (1) alone -- does widening the action space keep the DV alive?
    config("B +action_dim5/noop4", 5, 0, 4, False, ["FREE", "EXTERNAL", "SELF"])
    # C: repairs (1)+(2) -- is z_harm_a free to vary with hazards on?
    config("C +hazards", 5, 2, 4, False, ["FREE", "EXTERNAL", "SELF"])
    # D: repairs (1)+(2)+(4) -- does a trained self-forward let SELF drop motor_agency?
    config("D +selfP0", 5, 2, 4, True, ["FREE", "EXTERNAL", "SELF"])


# ------------------------------------------------------------------ #
# Round 2 (2026-09-08): use_affective_harm_stream is a DEFAULT-OFF flag
# (config.py:174). Round 1 omitted it, so its z_harm_a sd of 0.0000 was a
# harness omission, not a substrate finding. Re-run with it on, and add a
# direct test of whether z_self is action-dependent at all -- if it is not,
# motor_agency (1/(1+||predict_next_self(zs,a) - zs_now||)) cannot respond to
# an execution failure and repair (4) is not instantiable.
# ------------------------------------------------------------------ #

def _build2(action_dim, num_hazards, noop_class, seed, harm_stream):
    env_kwargs = dict(base.ENV_KWARGS)
    env_kwargs["num_hazards"] = num_hazards

    def env_(block: bool):
        return CausalGridWorldV2(
            seed=seed, scheduled_action_block_enabled=block,
            scheduled_action_block_interval=base.BLOCK_INTERVAL,
            scheduled_action_block_prob=1.0, **env_kwargs)

    e = env_(False)
    cfg_kwargs = dict(base.CFG_KWARGS)
    if noop_class is not None:
        cfg_kwargs["blocked_agency_noop_class"] = noop_class
    if harm_stream:
        cfg_kwargs["use_affective_harm_stream"] = True
    cfg = REEConfig.from_dims(
        body_obs_dim=e.body_obs_dim, world_obs_dim=e.world_obs_dim,
        action_dim=action_dim, **cfg_kwargs)
    return env_, REEAgent(cfg)


def zself_action_sensitivity(agent, env_, action_dim, noop_class, prefix=25):
    """Deterministic replay: drive an identical prefix, then branch on the FINAL action
    only, and compare the resulting z_self. (An earlier deepcopy-based form failed --
    the agent holds a module reference and cannot be deep-copied.)

    If ||z_self(move) - z_self(no-op)|| is small relative to ||z_self||, motor_agency
    = 1/(1+||predict_next_self(zs,a) - zs_now||) cannot register an execution failure,
    and repair (4) is not instantiable.
    """
    def replay_then(action_idx):
        base._seed_all(42)
        e = env_(False)
        _, od = e.reset()
        base._pin_goal(agent)
        agent.reset()
        agent.sense(od["body_state"], od["world_state"])
        r = np.random.RandomState(3)
        for _ in range(prefix):
            a = _one_hot(int(r.randint(0, 4)), action_dim)
            _, _h, _d, _i, od = e.step(a)
            agent.sense(od["body_state"], od["world_state"])
        _, _h, _d, _i, od = e.step(_one_hot(action_idx, action_dim))
        return agent.sense(od["body_state"], od["world_state"]).z_self.detach().clone()

    zs_move = replay_then(0)
    zs_stay = replay_then(noop_class)
    zs_other = replay_then(1)
    return (float((zs_move - zs_stay).norm().item()),
            float((zs_move - zs_other).norm().item()),
            float(zs_move.norm().item()))


def round2(seed=42):
    print("\n=== E +harm_stream +selfP0 (action_dim=5, hazards=2, noop=4, "
          "use_affective_harm_stream=True) ===", flush=True)
    base._seed_all(seed)
    env_, agent = _build2(5, 2, 4, seed, harm_stream=True)
    p0 = _p0(agent, env_, 5, seed, train_self=True)
    print(f"  P0 loss={p0:.3e}", flush=True)
    for m in ("FREE", "EXTERNAL", "SELF"):
        _row(m, _measure(agent, env_, 5, 4, m))
    ms, mm, nrm = zself_action_sensitivity(agent, env_, 5, 4)
    print(f"  z_self action-sensitivity: ||move - stay|| = {ms:.6f}   "
          f"||move - other_move|| = {mm:.6f}   ||z_self|| = {nrm:.6f}", flush=True)


if __name__ == "__main__" and "--round2" in sys.argv:
    round2()
