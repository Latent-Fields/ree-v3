"""SD-075 diagnostic spike (NOT an experiment; no queue entry).

QUESTION. The SD-075 live smoke ran an UNTRAINED agent and found
n_events_converged == 0 in BOTH continuity modes under
phasic_burst_warmup_ticks=-1 (derived 30): every phasic event landed inside the
first 30 lifetime ticks, so the convergence-gated read was UNINFORMATIVE.

The SD-075 doc records an UNTESTED hypothesis: a TRAINED agent's surprise stream
may keep producing genuine relative excesses past tick 30. This spike tests it
by warming the agent via SD-074 experiments/_lib/probe_warmup.py and then
re-driving the same phasic read.

DESIGN NOTE -- fresh regulator after warmup. warmup_train() drives select_action(),
which ticks the phasic regulator and so burns lifetime ticks before the read
begins. The regulator carries no state_dict entries, so a cache-HIT agent and a
cache-MISS agent would arrive at the read with DIFFERENT lifetime counters --
the read would then be a function of cache state. We therefore install a FRESH
PhasicSurpriseBurst after warmup, so the measured lifetime is exactly the read
rollout in every cell, hit or miss.

ASCII-only output (CLAUDE.md).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
_REPO_ROOT = _THIS.parent.parent.parent          # ree-v3
_EXP_DIR = _REPO_ROOT / "experiments"
for _p in (str(_REPO_ROOT), str(_EXP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

from ree_core.utils.config import REEConfig  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.regulators.phasic_surprise_burst import (  # noqa: E402
    PhasicSurpriseBurst,
    PhasicSurpriseBurstConfig,
)
from experiments._harness import StepHarness  # noqa: E402
from _lib.probe_warmup import WarmupRecipe, warm_agent  # noqa: E402

# Env + phasic config: byte-identical to V3-EXQ-779b so the read is denominated
# in the same units as the failure record.
ENV_SIZE, ENV_HAZARDS, ENV_RESOURCES = 8, 2, 3
ENV_DRIFT_SOURCES, ENV_DRIFT_POLICY = 3, "random_walk"
PHASIC_SOURCE = "instantaneous_pe"
PHASIC_TRIGGER_RATIO = 1.2
PHASIC_EMA_DECAY = 0.1
PHASIC_TEMP_DELTA = -0.5
PHASIC_DECAY = 0.5
PHASIC_TRIGGER_FLOOR = 1e-6
PHASIC_MIN_T = 0.1
EVENT_LEVEL_FLOOR = 0.05

MIN_EVENT_TICKS = 10          # 779b R1 bar, the thing this spike asks about
STEPS_PER_EPISODE = 300


def _mk_env() -> CausalGridWorldV2:
    return CausalGridWorldV2(
        size=ENV_SIZE,
        num_hazards=ENV_HAZARDS,
        num_resources=ENV_RESOURCES,
        background_drift_enabled=True,
        n_drift_sources=ENV_DRIFT_SOURCES,
        drift_policy=ENV_DRIFT_POLICY,
    )


def _mk_config(env: CausalGridWorldV2) -> REEConfig:
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
    )
    cfg.use_control_vector_logging = True
    cfg.hippocampal.use_action_class_scaffold_candidates = True
    cfg.use_noise_floor = False
    cfg.use_phasic_burst = True
    cfg.phasic_burst_signal_source = PHASIC_SOURCE
    cfg.phasic_burst_trigger_ratio = PHASIC_TRIGGER_RATIO
    cfg.phasic_burst_surprise_ema_decay = PHASIC_EMA_DECAY
    cfg.phasic_burst_temp_delta = PHASIC_TEMP_DELTA
    cfg.phasic_burst_decay = PHASIC_DECAY
    cfg.phasic_burst_trigger_floor = PHASIC_TRIGGER_FLOOR
    cfg.phasic_burst_min_temperature = PHASIC_MIN_T
    cfg.phasic_burst_baseline_continuity = "carry"
    cfg.phasic_burst_warmup_ticks = -1
    return cfg


def _fresh_regulator(agent: REEAgent) -> None:
    """Reinstall a zero-lifetime regulator (see module docstring)."""
    agent.phasic_burst = PhasicSurpriseBurst(
        config=PhasicSurpriseBurstConfig(
            enabled=True,
            surprise_ema_decay=PHASIC_EMA_DECAY,
            trigger_ratio=PHASIC_TRIGGER_RATIO,
            trigger_floor=PHASIC_TRIGGER_FLOOR,
            temp_delta=PHASIC_TEMP_DELTA,
            decay=PHASIC_DECAY,
            min_temperature=PHASIC_MIN_T,
            baseline_continuity="carry",
            warmup_ticks=-1,
        )
    )


def run_cell(seed: int, warmup_eps: int, read_steps: int, read_eps: int) -> dict:
    env = _mk_env()
    cfg = _mk_config(env)
    agent = REEAgent(cfg)

    recipe = WarmupRecipe(num_episodes=int(warmup_eps),
                          steps_per_episode=STEPS_PER_EPISODE)
    env_kwargs = {
        "size": ENV_SIZE, "num_hazards": ENV_HAZARDS,
        "num_resources": ENV_RESOURCES, "background_drift_enabled": True,
        "n_drift_sources": ENV_DRIFT_SOURCES, "drift_policy": ENV_DRIFT_POLICY,
    }
    out = warm_agent(agent, env, seed=seed, recipe=recipe,
                     env_kwargs=env_kwargs,
                     label="sd075spike w%d" % warmup_eps, measure=False)

    _fresh_regulator(agent)

    # Read rollout. train_mode=False: measure the landscape warmup produced,
    # do not keep training inside the read.
    harness = StepHarness(agent, env, train_mode=False, seed=seed)
    burst_max = 0.0
    n_event_window_ticks = 0
    ep_lengths = []
    steps = 0
    agent.eval()
    with torch.no_grad():
        for _ep in range(int(read_eps)):
            _flat, obs_dict = env.reset()
            agent.reset()
            harness.reset()
            ep_len = 0
            for _ in range(STEPS_PER_EPISODE):
                r = harness.step(obs_dict)
                obs_dict = r.next_obs_dict
                steps += 1
                ep_len += 1
                lvl = float(agent.phasic_burst.burst_level)
                burst_max = max(burst_max, lvl)
                if lvl >= EVENT_LEVEL_FLOOR:
                    n_event_window_ticks += 1
                if r.done or steps >= read_steps:
                    break
            ep_lengths.append(ep_len)
            if steps >= read_steps:
                break

    st = agent.phasic_burst.get_state()
    return {
        "seed": seed,
        "warmup_episodes": warmup_eps,
        "warmup_cache_hit": bool(out.cache_hit),
        "lifetime_ticks": int(st["lifetime_ticks"]),
        "lifetime_episodes": int(st["lifetime_episodes"]),
        "warmup_ticks_resolved": int(st["warmup_ticks"]),
        "n_converged_ticks": int(st["n_converged_ticks"]),
        "n_events_converged": int(st["n_events_converged"]),
        "n_events_prewarmup": int(st["n_events_prewarmup"]),
        "n_event_window_ticks": int(n_event_window_ticks),
        "burst_level_max": float(burst_max),
        "surprise_ema": float(st["surprise_ema"]),
        "n_read_env_steps": steps,
        "n_read_episodes": len(ep_lengths),
        "mean_episode_len": (sum(ep_lengths) / len(ep_lengths)) if ep_lengths else 0.0,
        "meets_min_event_ticks_converged": bool(
            int(st["n_events_converged"]) >= MIN_EVENT_TICKS),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 29])
    ap.add_argument("--warmups", type=int, nargs="+", default=[0, 10, 40])
    ap.add_argument("--read-steps", type=int, default=600)
    ap.add_argument("--read-eps", type=int, default=200)
    ap.add_argument("--out", type=str, default="")
    a = ap.parse_args()

    rows = []
    for w in a.warmups:
        for s in a.seeds:
            print("=== warmup=%d seed=%d ===" % (w, s), flush=True)
            r = run_cell(s, w, a.read_steps, a.read_eps)
            rows.append(r)
            print("  -> conv_events=%d prewarm_events=%d conv_ticks=%d "
                  "life_ticks=%d burst_max=%.2f eps=%d mean_ep_len=%.1f"
                  % (r["n_events_converged"], r["n_events_prewarmup"],
                     r["n_converged_ticks"], r["lifetime_ticks"],
                     r["burst_level_max"], r["n_read_episodes"],
                     r["mean_episode_len"]), flush=True)

    print("\n%-8s %-6s %-12s %-12s %-12s %-12s %-10s"
          % ("warmup", "seed", "conv_events", "prewarm_ev", "conv_ticks",
             "life_ticks", "burst_max"))
    for r in rows:
        print("%-8d %-6d %-12d %-12d %-12d %-12d %-10.2f"
              % (r["warmup_episodes"], r["seed"], r["n_events_converged"],
                 r["n_events_prewarmup"], r["n_converged_ticks"],
                 r["lifetime_ticks"], r["burst_level_max"]))
    n_any = sum(1 for r in rows if r["n_events_converged"] > 0)
    n_bar = sum(1 for r in rows if r["meets_min_event_ticks_converged"])
    print("\ncells with n_events_converged > 0: %d/%d" % (n_any, len(rows)))
    print("cells reaching MIN_EVENT_TICKS=%d on converged: %d/%d"
          % (MIN_EVENT_TICKS, n_bar, len(rows)))

    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=2))
        print("wrote %s" % a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
