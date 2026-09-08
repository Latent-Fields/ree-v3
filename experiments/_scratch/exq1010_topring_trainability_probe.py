"""AUTHORING-TIME PROBE for V3-EXQ-1010 (not an experiment; writes no manifest).

QUESTION: at FULL data scale and the real pass count, does the capacity ladder's top rung
(`deep2048x4`, 12.67M action-path params) actually FIT under the run's one fixed protocol
(Adam @ ADAPTER_LR, ADAPTER_BATCH, ADAPTER_PASSES, grad-clip GRAD_CLIP_NORM)?

WHY IT CANNOT BE ANSWERED FROM THE DRY-RUN: the smoke runs 3 passes over a handful of episodes,
so a 12.67M-param net is nonsense there by construction. The divergence it showed is not
evidence about the real run. This probe uses the FULL dataset recipe.

WHY IT IS CHEAP: it probes the `ws250_pca` ANCHOR track, which needs NO warmup at all -- only
the deterministic dataset collection. That is exactly the track GUARD 1 is measured on, so this
measures Guard 1's own subject at authoring time.

Run:  /opt/local/bin/python3 experiments/_scratch/exq1010_topring_trainability_probe.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

import experiments.v3_exq_1002_zworld_actor_adequacy_oracle_adapter as x1002  # noqa: E402
import experiments.v3_exq_1008_zworld_adequacy_portfolio_ws250_rebasis as x1008  # noqa: E402
import experiments.v3_exq_734_env_difficulty_competence_recovery_sweep as x734  # noqa: E402
import experiments.v3_exq_1010_zworld_overcapacity_decoder_sweep as x1010  # noqa: E402

SEED = 42


def main() -> int:
    t0 = time.perf_counter()
    env_kwargs = x734._env_kwargs_for_rung(x1010.RUNG)
    probe_env = x734._make_env(SEED, env_kwargs)
    action_dim = int(probe_env.action_dim)

    torch.manual_seed(SEED)
    print("[probe] collecting the FULL 1002 dataset recipe (seed %d) ..." % SEED, flush=True)
    oracle_eps = x1002._collect_episodes(SEED, env_kwargs, "oracle",
                                         x1010.BC_EPISODES, x1010.STEPS_PER_EPISODE)
    rand_eps = x1002._collect_episodes(SEED, env_kwargs, "random",
                                       x1010.BC_RANDOM_EPISODES, x1010.STEPS_PER_EPISODE)
    tr, te = x1002._split_episodes(oracle_eps)
    _f_tr, y_tr = x1002._rawfield_features(tr)
    _f_te, y_te = x1002._rawfield_features(te)
    w_tr, _ = x1008._world_state_features(tr)
    w_te, _ = x1008._world_state_features(te)
    wr_te, _ = x1008._world_state_features(rand_eps)
    _fr, yr_te = x1002._rawfield_features(rand_eps)
    print("[probe] n_train_rows=%d n_heldout_rows=%d  (1008 measured 5038/4997/4423 train)"
          % (int(w_tr.shape[0]), int(w_te.shape[0])), flush=True)

    W, stats = x1008._world_state_pca_stats(w_tr, x1010.PROJECTION_DIM)
    proj = x1008._LinearProjection(w_tr, W, "pca_32", extra=stats)
    xs_tr, xs_te = proj(w_tr), proj(w_te)
    xsr_te = proj(wr_te)
    in_dim = int(xs_tr.shape[1])

    for rung in ("mlp128", "mlp2048", "deep2048x4"):
        t1 = time.perf_counter()
        x1010.reset_all_rng(SEED)
        net = x1010._make_decoder(rung, in_dim, action_dim)
        rep = x1010._capacity_report(net, rung, in_dim, action_dim)
        st = x1010._train_decoder(net, xs_tr, y_tr, x1010.ADAPTER_PASSES, SEED,
                                  "probe_" + rung, action_dim)
        sc = x1002._score_cell(net, xs_tr, y_tr, xs_te, y_te, xsr_te, yr_te, action_dim,
                               prev_te=x1002._prev_action_vector(te))
        print("[probe] %-12s params=%-9d final_ce=%.4f (uniform=%.4f) diverged=%s "
              "train_agree=%.4f heldout_agree=%.4f  [%.1f s]"
              % (rung, rep["action_path_params"], st["final_ce_loss"], st["uniform_logit_ce"],
                 st["diverged"], sc["oracle_action_agreement_train"] or -1.0,
                 sc["oracle_action_agreement"] or -1.0, time.perf_counter() - t1), flush=True)

    print("[probe] total %.1f s" % (time.perf_counter() - t0), flush=True)
    print("[probe] REFERENCE, V3-EXQ-1008 ws250_pca at mlp128: heldout 0.8771 (seed 42)",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
