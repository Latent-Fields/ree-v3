#!/usr/bin/env python3
"""Cross-process reproducibility probe for REEConfig.stdlib_rng_seed.

WHAT THIS ANSWERS
-----------------
`tests/contracts/test_stdlib_rng_seed_determinism.py` (12 contracts, landed
2026-07-28 in 55c051c52d) pins the stdlib-`random` seeding WITHIN ONE PROCESS.
It does not establish that a run pinning `stdlib_rng_seed` + the env seed +
the driver's torch/numpy seeds is byte-identical across TWO PROCESSES. If it
is not, a FOURTH entropy source is still unfound -- exactly the class of bug
that presents as untraceable non-reproducibility.

This probe runs a real agent+env loop (the `tests/fixtures/tiny_loop.py`
sense -> tick -> generate -> select -> step dance that experiment drivers use)
on a config that ACTUALLY REACHES the stdlib-`random` sites, digests the full
trajectory + sleep metrics + final parameter state, and prints the digest.
Run it twice in two separate interpreters and compare the digests.

TWO ARMS -- the negative control is load-bearing
------------------------------------------------
  --arm pinned    stdlib_rng_seed set, env seed set, torch+numpy seeded.
                  EXPECT: identical digest across processes.
  --arm unpinned  stdlib_rng_seed=None (the "typical driver" state: torch and
                  numpy seeded, stdlib `random` left on OS entropy), env seed
                  and torch/numpy seeds otherwise identical.
                  EXPECT: DIFFERENT digest across processes.

Without the unpinned arm, two matching pinned digests are uninterpretable:
they are equally consistent with "the seeding works" and with "the replay path
never fired". The unpinned arm is what proves the probe is sensitive to the
entropy source it claims to have closed. The per-site draw counters printed in
the summary are the second, direct check on the same thing.

Note the arms are NOT scientifically comparable to each other or to any landed
run -- setting `stdlib_rng_seed` CHANGES RESULTS by design. This is an
infrastructure probe, not evidence for any claim.

USAGE
-----
    python3 scripts/stdlib_rng_cross_process_probe.py --arm pinned --out a.json
    python3 scripts/stdlib_rng_cross_process_probe.py --arm pinned --out b.json

Both processes must be launched with PYTHONHASHSEED=0, OMP_NUM_THREADS=1 and
MKL_NUM_THREADS=1 exported BEFORE interpreter start; the probe refuses to run
otherwise rather than silently producing an uninterpretable divergence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from typing import Any, Dict, List

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402

# NOTE: deliberately NOT tests.fixtures.tiny_loop.step_once. That helper drives
# the decomposed sense -> tick -> generate -> select path, which never reaches
# `REEAgent._do_replay` -- and `_do_replay` is the ONLY waking route to S1.
# Measured: the step_once loop yields ZERO stdlib-random draws on this config,
# so a probe built on it would compare two digests that never touched the RNG
# under test. `act_with_split_obs` is the real driver-level entry point and
# calls `_do_replay` on every e3_quiescent tick.


BASE_SEED = 12345
ENV_SEED = 7
N_EPISODES = 3
N_STEPS = 12


# --------------------------------------------------------------------------- #
# instrumentation                                                             #
# --------------------------------------------------------------------------- #
class CountingRNG:
    """Transparent proxy that counts the stdlib-`random` draws per method.

    Wraps whatever the consumer's `_rng` already is -- either the `random`
    MODULE (the unseeded default) or a `random.Random` instance (seeded).
    Only the three methods the ree_core sites actually call are intercepted;
    everything else delegates untouched.

    Counting is the direct evidence that this config reaches the sites at all.
    A probe that reports 0 draws is measuring nothing, however identical its
    digests look.
    """

    def __init__(self, inner: Any, counts: Dict[str, int], prefix: str) -> None:
        self._inner = inner
        self._counts = counts
        self._prefix = prefix

    def random(self, *a, **kw):
        self._counts[self._prefix + ".random"] = (
            self._counts.get(self._prefix + ".random", 0) + 1
        )
        return self._inner.random(*a, **kw)

    def choice(self, *a, **kw):
        self._counts[self._prefix + ".choice"] = (
            self._counts.get(self._prefix + ".choice", 0) + 1
        )
        return self._inner.choice(*a, **kw)

    def choices(self, *a, **kw):
        self._counts[self._prefix + ".choices"] = (
            self._counts.get(self._prefix + ".choices", 0) + 1
        )
        return self._inner.choices(*a, **kw)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _f(x: Any) -> str:
    """Full-precision float repr. `%.17g` round-trips an IEEE-754 double.

    Digesting formatted floats rather than eyeballing them is the point: a
    divergence in the 15th significant digit is a divergence.
    """
    return "%.17g" % float(x)


def _state_dict_digest(module: torch.nn.Module) -> str:
    """Digest of every parameter/buffer byte, key-sorted.

    Catches divergence that never reaches an action index -- e.g. a sleep
    writeback that stepped different gradients but left the policy's argmax
    unchanged over a short probe.
    """
    h = hashlib.sha256()
    sd = module.state_dict()
    for key in sorted(sd.keys()):
        val = sd[key]
        h.update(key.encode("utf-8"))
        if isinstance(val, torch.Tensor):
            arr = val.detach().cpu().contiguous().numpy()
            h.update(str(arr.dtype).encode("utf-8"))
            h.update(str(arr.shape).encode("utf-8"))
            h.update(arr.tobytes())
        else:
            h.update(repr(val).encode("utf-8"))
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# the run                                                                     #
# --------------------------------------------------------------------------- #
def build(arm: str):
    """Env + agent on a config that REACHES the stdlib-random sites.

    replay_diversity_enabled=True   -> S1, once per replay step (SWS pass).
    enable_sleep_aggregation_cluster() -> use_mech273_self_model + sws/rem,
                                       reaching S3 once the harm replay buffer
                                       is non-empty.
    harm_dim > 0                    -> the waking harm stream that FILLS that
                                       buffer (agent.py gates the append on
                                       `new_latent.z_harm is not None`).
    use_e2_harm_s_forward=True      -> the ARC-033 substrate prerequisite for
                                       Phase E. enable_sleep_aggregation_cluster()
                                       deliberately does NOT bundle it (its own
                                       docstring says so), so without this the
                                       writeback is a no-op and S3 never fires.
    """
    # A typical experiment driver's `_run_seed`: torch + numpy only. The
    # process-global stdlib `random` is deliberately NOT seeded here -- that
    # is the exposure under test, and seeding it would mask the unpinned arm.
    torch.manual_seed(BASE_SEED)
    np.random.seed(BASE_SEED)

    env = CausalGridWorldV2(
        seed=ENV_SEED,
        size=5,
        num_hazards=1,
        num_resources=1,
        use_proxy_fields=True,
    )
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=4,
        self_dim=16,
        world_dim=16,
        # 32, not a smaller value: E2HarmSConfig.z_harm_dim defaults to 32 and
        # offline_gradient_pass slices the waking pairs to that width
        # (`p[0].view(-1)[:z_dim]`). A narrower harm stream slices SHORT rather
        # than erroring at the slice, and the writeback then dies deep inside
        # the transition net on a shape mismatch. Keep these aligned.
        harm_dim=32,
        use_e2_harm_s_forward=True,
        replay_diversity_enabled=True,
        stdlib_rng_seed=(BASE_SEED if arm == "pinned" else None),
    )
    cfg.enable_sleep_aggregation_cluster()
    agent = REEAgent(cfg)
    return env, cfg, agent


def run(arm: str) -> Dict[str, Any]:
    env, cfg, agent = build(arm)

    counts: Dict[str, int] = {}
    # S1 (.random) and S2 (.choice) both live on the hippocampal module's rng.
    agent.hippocampal._rng = CountingRNG(agent.hippocampal._rng, counts, "hippocampal")
    agg = getattr(agent, "sleep_self_model_aggregator", None)
    if agg is not None:
        # S3 (.choices) -- the waking-pair sample in offline_gradient_pass.
        agg._rng = CountingRNG(agg._rng, counts, "self_model")

    actions: List[int] = []
    harms: List[str] = []
    sleep_metrics: List[Dict[str, str]] = []

    for _episode in range(N_EPISODES):
        agent.reset()
        _flat, obs = env.reset()
        for _ in range(N_STEPS):
            body = obs["body_state"]
            world = obs["world_state"]
            if body.dim() == 1:
                body = body.unsqueeze(0)
            if world.dim() == 1:
                world = world.unsqueeze(0)
            with torch.no_grad():
                action = agent.act_with_split_obs(body, world)
            action_idx = int(action.argmax(dim=-1).item())
            _flat, harm_signal, _done, _info, obs = env.step(action_idx)
            agent.update_residue(float(harm_signal))
            actions.append(action_idx)
            harms.append(_f(harm_signal))
        # SleepLoopManager.force_cycle, NOT agent.run_sleep_cycle(). The bare
        # SD-017 surface runs SWS+REM and STOPS -- the Phase-E WRITEBACK that
        # reaches S3 is driven by the phase manager AFTER run_sleep_cycle
        # returns, so a probe calling run_sleep_cycle directly reports zero S3
        # draws (measured). force_cycle is the diagnostic entry point to the
        # full cycle and does not depend on the K-episode counter.
        metrics = agent.sleep_loop.force_cycle(agent)
        sleep_metrics.append({k: _f(v) for k, v in sorted(metrics.items())})

    # ----------------------------------------------------------------- #
    # S2 and S3 at their real call sites.                                #
    #                                                                     #
    # The episode loop above reaches S1 only. S2 needs an all-zero        #
    # retrieval_bias, and S3's writeback early-returns with              #
    # n_regions == 0 unless the aggregator holds posteriors for the       #
    # replayed regions -- which needs the MECH-269 anchor-set substrate   #
    # the sleep cluster deliberately does not bundle (measured on this    #
    # config: mech273_writeback_regions = 0 every cycle). Rather than     #
    # build that substrate, drive both sites directly with the agent's    #
    # OWN seeded consumers and its REAL accumulated waking buffers, so    #
    # all three sites land in the same digest.                            #
    # ----------------------------------------------------------------- #
    from ree_core.sleep.phase_manager import SleepPhase  # noqa: E402
    from ree_core.sleep.routing_gate import RoutedEvent  # noqa: E402

    s2_picks: List[int] = []
    buf = agent.hippocampal._exploration_buffer
    if len(buf) > 0:
        zero_bias = torch.zeros(len(buf))
        for _ in range(12):
            traj = agent.hippocampal._sample_exploration_trajectory(
                retrieval_bias=zero_bias
            )
            s2_picks.append(next(i for i, t in enumerate(buf) if t is traj))

    s3_metrics: Dict[str, str] = {}
    agg = getattr(agent, "sleep_self_model_aggregator", None)
    if agg is not None and getattr(agent, "e2_harm_s", None) is not None:
        regions = [("fast", "0.1"), ("fast", "0.2")]
        for r in regions:
            agg.update(
                RoutedEvent(
                    event=r,
                    anchor_channel=0.6,
                    probe_channel=0.4,
                    phase=SleepPhase.SWS_ANALOG,
                ),
                evidence=2.0,
                domain="self",
            )
        s3_metrics = {
            k: _f(v)
            for k, v in sorted(
                agg.offline_gradient_pass(
                    e2_harm_s=agent.e2_harm_s,
                    replayed_regions=set(regions),
                    n_steps=5,
                    domain="self",
                    use_snapshot=False,
                    harm_replay_buffer=agent._harm_replay_buffer,
                ).items()
            )
        }

    payload = {
        "s2_picks": s2_picks,
        "s3_metrics": s3_metrics,
        "arm": arm,
        "actions": actions,
        "harms": harms,
        "sleep_metrics": sleep_metrics,
        "residue_norm": _f(
            float(torch.linalg.vector_norm(agent.residue_field.field).item())
        )
        if getattr(getattr(agent, "residue_field", None), "field", None) is not None
        else None,
        "state_dict_digest": _state_dict_digest(agent),
        "harm_replay_buffer_len": len(agent._harm_replay_buffer),
        "exploration_buffer_len": len(agent.hippocampal._exploration_buffer),
    }
    # The digest deliberately EXCLUDES `draw_counts` and `env`: a count is
    # evidence the path fired, not part of the result being reproduced, and
    # folding it in would let a count difference masquerade as a result
    # difference.
    payload["digest"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    payload["draw_counts"] = dict(sorted(counts.items()))
    payload["env"] = {
        "python": sys.version.split()[0],
        "platform": f"{platform.system().lower()}-{platform.machine()}",
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pid": os.getpid(),
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
    }
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=("pinned", "unpinned"), required=True)
    ap.add_argument("--out", default=None, help="write the full record as JSON")
    ap.add_argument(
        "--allow-unpinned-env",
        action="store_true",
        help="skip the PYTHONHASHSEED / thread-count precondition check",
    )
    args = ap.parse_args()

    if not args.allow_unpinned_env:
        missing = [
            k
            for k, want in (
                ("PYTHONHASHSEED", "0"),
                ("OMP_NUM_THREADS", "1"),
                ("MKL_NUM_THREADS", "1"),
            )
            if os.environ.get(k) != want
        ]
        if missing:
            print(
                "ERROR: export before interpreter start: "
                + ", ".join(f"{k}=<expected>" for k in missing)
                + "\n  (PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1)"
                + "\n  Re-run with --allow-unpinned-env only if you accept that a"
                " divergence may be thread-count noise, not RNG.",
                file=sys.stderr,
            )
            return 2

    torch.set_num_threads(1)
    rec = run(args.arm)

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(rec, fh, sort_keys=True, indent=2)

    print(f"arm            : {rec['arm']}")
    print(f"digest         : {rec['digest']}")
    print(f"state_dict     : {rec['state_dict_digest']}")
    print(f"draw_counts    : {rec['draw_counts']}")
    print(f"actions[:24]   : {rec['actions'][:24]}")
    print(f"harm_buffer    : {rec['harm_replay_buffer_len']}")
    print(f"explore_buffer : {rec['exploration_buffer_len']}")
    print(f"env            : {rec['env']}")
    if not rec["draw_counts"]:
        print(
            "WARNING: zero stdlib-random draws -- this config does NOT reach the"
            " sites, so matching digests would prove nothing.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
