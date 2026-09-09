"""TRAINED-ENCODER extension of exq_wpfield_h2_probe_dvrange.py (campaign W6-S6 item 1,
chip-20260905-waypoint-consumer-reach-portfolio). NOT an experiment; no manifest, no queue entry.

WHY. The 2026-09-07 record (REE_assembly evidence/planning/exq_wpfield_h2_dv_range_probe_20260907.md
section 7b) left leg H-wpfield-zworld-interface NOT queueable: its corrected criterion ("trained
z_world lift falls materially below the random-projection floor") failed dv_headroom_check because
the only control arm measured -- the UNTRAINED SplitEncoder -- has a seed spread of 0.04 against a
required drop of 0.147. The record's stated remaining work is to MEASURE A TRAINED ENCODER AT PROBE
SCALE so the criterion can be written against a range the configuration can actually produce.

WHAT THIS MEASURES. The same ON-minus-OFF direction-decodability lift as the original probe (raw,
randproj, untrained encoder), plus the lineage's TRAINED z_world: the x1002/x1008 all-ON agent
(x1002._make_agent, built exactly as V3-EXQ-978/1002/1008 built theirs) warmed with the SD-070 P0a
recipe (run_zworld_p0, ZWORLD_P0_EPISODES=60 x 200 steps, random policy, resource_field_weight
0.0 = the 978 OFF arm) on the 1004 bench with the waypoint field ON (world_obs_dim 275). P0a is
the ONLY phase that ever steps the world encoder (P0b/P1 have no latent_stack optimizer group),
so this IS the encoder the REE consumer would read in the queued experiment.

Ablation is by ZEROING the trailing 25 dims (never by turning the env flag off) -- same reason as
the original probe: the flag changes world_obs_dim and therefore the encoder init draw.

Usage: /opt/local/bin/python3 experiments/_scratch/exq_wpfield_h2_probe_dvrange_trained.py \
           --seeds 42 43 44 45 46 --episodes 40 --steps 200 --p0a-episodes 60
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
    FIELD_DIMS, build_env, collect, linear_probe, randproj,
)
import experiments.v3_exq_1002_zworld_actor_adequacy_oracle_adapter as x1002  # noqa: E402
from experiments._lib.capability_eval import RandomPolicy  # noqa: E402
from experiments._lib.zworld_encoder_guard import (  # noqa: E402
    latent_stack_snapshot, latent_stack_weight_delta,
)
from experiments._lib.zworld_p0_warmup import run_zworld_p0  # noqa: E402


def participation_ratio(z: np.ndarray) -> float:
    zc = z - z.mean(0, keepdims=True)
    ev = np.linalg.eigvalsh(np.cov(zc, rowvar=False))
    ev = np.clip(ev, 0.0, None)
    return float(ev.sum() ** 2 / max((ev ** 2).sum(), 1e-12))


def encode_with(agent, X: np.ndarray) -> np.ndarray:
    """Encoder-path z_world (world_encoder only), matching the original probe's untrained column
    so the trained number is comparable to the 09-07 table."""
    with torch.no_grad():
        return agent.latent_stack.split_encoder.world_encoder(torch.as_tensor(X)).numpy()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    p.add_argument("--episodes", type=int, default=12)
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--p0a-episodes", type=int, default=60)
    p.add_argument("--json", action="store_true")
    a = p.parse_args()

    rows = []
    for seed in a.seeds:
        rng = np.random.default_rng(seed)
        X, Y = collect(seed, a.episodes, a.steps, rng)
        if len(X) < 200:
            print(f"seed {seed}: only {len(X)} usable states -- SKIP", flush=True)
            continue
        n = len(X)
        perm = np.random.default_rng(seed).permutation(n)
        ntr = int(0.7 * n)
        tr, te = perm[:ntr], perm[ntr:]
        X_off = X.copy()
        X_off[:, -FIELD_DIMS:] = 0.0

        maj = float(np.bincount(Y[te], minlength=4).max() / len(te))
        raw_on = linear_probe(X[tr], Y[tr], X[te], Y[te], seed=seed)
        raw_off = linear_probe(X_off[tr], Y[tr], X_off[te], Y[te], seed=seed)
        R_on, R_off = randproj(X, seed), randproj(X_off, seed)
        rp_lift = (linear_probe(R_on[tr], Y[tr], R_on[te], Y[te], seed=seed)
                   - linear_probe(R_off[tr], Y[tr], R_off[te], Y[te], seed=seed))

        # The lineage agent, built as 978/1002/1008 built it, on the 1004 bench (275-dim obs).
        torch.manual_seed(seed)
        np.random.seed(seed)
        env = build_env(seed)
        agent = x1002._make_agent(env)
        assert int(env.world_obs_dim) == int(X.shape[1]) == 275, (env.world_obs_dim, X.shape)

        Zu_on, Zu_off = encode_with(agent, X), encode_with(agent, X_off)
        unt_lift = (linear_probe(Zu_on[tr], Y[tr], Zu_on[te], Y[te], seed=seed)
                    - linear_probe(Zu_off[tr], Y[tr], Zu_off[te], Y[te], seed=seed))

        before = latent_stack_snapshot(agent)
        p0a = run_zworld_p0(agent, build_env(seed), seed, a.p0a_episodes, a.steps,
                            policy=RandomPolicy(seed), label="h2probe-trained",
                            resource_field_weight=0.0)
        guard = latent_stack_weight_delta(agent, before)

        Zt_on, Zt_off = encode_with(agent, X), encode_with(agent, X_off)
        tr_on = linear_probe(Zt_on[tr], Y[tr], Zt_on[te], Y[te], seed=seed)
        tr_off = linear_probe(Zt_off[tr], Y[tr], Zt_off[te], Y[te], seed=seed)

        row = dict(
            seed=seed, n_states=n, majority=maj,
            raw_lift=raw_on - raw_off, randproj_lift=rp_lift,
            zworld_untrained_lift=unt_lift,
            zworld_trained_on=tr_on, zworld_trained_off=tr_off,
            zworld_trained_lift=tr_on - tr_off,
            pr_untrained=participation_ratio(Zu_on), pr_trained=participation_ratio(Zt_on),
            p0a_ran=bool(p0a.get("p0a_ran")), p0a_n_steps=p0a.get("p0a_n_steps"),
            p0a_final_loss=p0a.get("p0a_final_loss"),
            p0a_holdout_mean_lift=p0a.get("p0a_holdout_mean_lift"),
            world_encoder_max_abs_delta=guard.get("world_encoder_max_abs_delta"),
        )
        rows.append(row)
        print(f"seed {seed}: n={n} majority={maj:.3f} | RAW lift={row['raw_lift']:+.3f} | "
              f"RANDPROJ lift={rp_lift:+.3f} | ZWORLD untrained lift={unt_lift:+.3f} | "
              f"ZWORLD TRAINED on={tr_on:.3f} off={tr_off:.3f} lift={tr_on-tr_off:+.3f} | "
              f"PR unt={row['pr_untrained']:.2f} tr={row['pr_trained']:.2f} | "
              f"p0a ran={row['p0a_ran']} dW={row['world_encoder_max_abs_delta']}", flush=True)

    if rows:
        print("\n=== DV RANGE (mean over seeds) ===")
        for k in ("raw_lift", "randproj_lift", "zworld_untrained_lift", "zworld_trained_lift",
                  "zworld_trained_on", "zworld_trained_off", "pr_trained"):
            vals = [r[k] for r in rows]
            print(f"  {k:22s} mean={np.mean(vals):+.4f}  min={np.min(vals):+.4f}  "
                  f"max={np.max(vals):+.4f}  range={np.max(vals)-np.min(vals):.4f}")
    if a.json:
        print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
