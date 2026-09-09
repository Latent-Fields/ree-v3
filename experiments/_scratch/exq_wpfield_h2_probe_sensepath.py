"""SENSE-PATH extension of exq_wpfield_h2_probe_dvrange_trained.py (campaign W6-S6 item 1).
NOT an experiment; no manifest, no queue entry.

WHY. The encoder-path probe measures world_encoder(w) only. The REE consumer reads SENSE-time
z_world = (world_encoder(w) + world_topdown(beta_to_split(z_beta))) * prec, temporally smoothed
by alpha_world (x724 config: 0.9) -- the quantity V3-EXQ-1002's adapter read (x737._agent_zworld).
If top-down conditioning or smoothing blurs the pending-waypoint direction, H2 is live at the
SENSE path even though the encoder path preserves it. This probe measures the ON-minus-OFF
direction-decodability lift at sense-time z_world, before and after the SD-070 P0a warmup,
replaying identical stored episodes (agent.reset() per episode; ON and OFF as separate replays
because smoothing carries state across steps).

Ablation is by ZEROING world_state[250:275] in the stored obs dict (field flag stays ON).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments" / "_scratch"))

from exq_wpfield_h2_probe_dvrange import (  # noqa: E402
    FIELD_DIMS, build_env, direction_label, linear_probe,
)
import experiments.v3_exq_1002_zworld_actor_adequacy_oracle_adapter as x1002  # noqa: E402
import experiments.v3_exq_737_ree_latent_policy_head_competence_probe as x737  # noqa: E402
from experiments._lib.capability_eval import RandomPolicy  # noqa: E402
from experiments._lib.zworld_encoder_guard import (  # noqa: E402
    latent_stack_snapshot, latent_stack_weight_delta,
)
from experiments._lib.zworld_p0_warmup import run_zworld_p0  # noqa: E402


def _clone_obs(obs):
    return {k: (v.detach().clone() if torch.is_tensor(v) else v) for k, v in obs.items()}


def collect_episodes(seed, n_episodes, steps, rng):
    env = build_env(seed)
    eps = []
    for _ in range(n_episodes):
        obs = env.reset()
        obs_dict = obs[1] if isinstance(obs, tuple) else obs
        seq, labs = [], []
        for _ in range(steps):
            lab = direction_label(env)
            if isinstance(obs_dict, dict) and "world_state" in obs_dict:
                seq.append(_clone_obs(obs_dict))
                labs.append(int(lab))          # -1 kept in sequence, dropped at scoring
            a = int(rng.integers(0, int(env.action_dim)))
            step = env.step(a)
            obs_dict = step[-1] if isinstance(step[-1], dict) else {}
            if bool(step[2]):
                break
        eps.append((seq, labs))
    return eps


def sense_features(agent, eps, zero_field):
    xs, ys = [], []
    for seq, labs in eps:
        agent.reset()
        for obs, lab in zip(seq, labs):
            o = obs
            if zero_field:
                o = dict(obs)
                w = obs["world_state"].detach().clone()
                w[..., -FIELD_DIMS:] = 0.0
                o["world_state"] = w
            z = x737._agent_zworld(agent, o).reshape(-1).cpu().numpy()
            if lab >= 0:
                xs.append(z)
                ys.append(lab)
    return np.asarray(xs, np.float32), np.asarray(ys, np.int64)


def lift(agent, eps, tr, te, seed):
    Zon, Y = sense_features(agent, eps, zero_field=False)
    Zoff, Y2 = sense_features(agent, eps, zero_field=True)
    assert (Y == Y2).all()
    on = linear_probe(Zon[tr], Y[tr], Zon[te], Y[te], seed=seed)
    off = linear_probe(Zoff[tr], Y[tr], Zoff[te], Y[te], seed=seed)
    return on, off, Zon


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--episodes", type=int, default=12)
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--p0a-episodes", type=int, default=60)
    p.add_argument("--json", action="store_true")
    a = p.parse_args()
    rows = []
    for seed in a.seeds:
        rng = np.random.default_rng(seed)
        eps = collect_episodes(seed, a.episodes, a.steps, rng)
        n_lab = sum(1 for _, labs in eps for l in labs if l >= 0)
        if n_lab < 200:
            print(f"seed {seed}: only {n_lab} usable states -- SKIP", flush=True)
            continue
        perm = np.random.default_rng(seed).permutation(n_lab)
        ntr = int(0.7 * n_lab)
        tr, te = perm[:ntr], perm[ntr:]

        torch.manual_seed(seed)
        np.random.seed(seed)
        env = build_env(seed)
        agent = x1002._make_agent(env)
        u_on, u_off, _ = lift(agent, eps, tr, te, seed)

        before = latent_stack_snapshot(agent)
        p0a = run_zworld_p0(agent, build_env(seed), seed, a.p0a_episodes, a.steps,
                            policy=RandomPolicy(seed), label="h2probe-sense",
                            resource_field_weight=0.0)
        guard = latent_stack_weight_delta(agent, before)
        t_on, t_off, Zt = lift(agent, eps, tr, te, seed)

        row = dict(seed=seed, n_states=n_lab,
                   sense_untrained_on=u_on, sense_untrained_off=u_off,
                   sense_untrained_lift=u_on - u_off,
                   sense_trained_on=t_on, sense_trained_off=t_off,
                   sense_trained_lift=t_on - t_off,
                   p0a_ran=bool(p0a.get("p0a_ran")),
                   world_encoder_max_abs_delta=guard.get("world_encoder_max_abs_delta"))
        rows.append(row)
        print(f"seed {seed}: n={n_lab} | SENSE untrained on={u_on:.3f} off={u_off:.3f} "
              f"lift={u_on-u_off:+.3f} | SENSE TRAINED on={t_on:.3f} off={t_off:.3f} "
              f"lift={t_on-t_off:+.3f} | p0a ran={row['p0a_ran']} "
              f"dW={row['world_encoder_max_abs_delta']}", flush=True)
    if rows:
        print("\n=== SENSE-PATH DV RANGE (mean over seeds) ===")
        for k in ("sense_untrained_lift", "sense_trained_lift", "sense_trained_on",
                  "sense_trained_off"):
            v = [r[k] for r in rows]
            print(f"  {k:22s} mean={np.mean(v):+.4f} min={np.min(v):+.4f} "
                  f"max={np.max(v):+.4f} range={np.max(v)-np.min(v):.4f}")
    if a.json:
        print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
