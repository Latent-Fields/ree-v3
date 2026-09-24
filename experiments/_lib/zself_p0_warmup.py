"""z_self P0 warmup (sd_zself_training_path) for experiment drivers.

WHY THIS EXISTS. Nothing in the substrate trains z_self. The E1/E2 losses read only
DETACHED copies of it (`agent.py:5855`, `:6294`, `:11397`), and no production optimizer
group contains a single z_self-path parameter -- so `split_encoder.self_encoder`, the
DR-13 `self_recurrence` GRU and `self_precision_logit` are a FROZEN RANDOM PROJECTION
for the whole of every run, with no error and no warning. Measured on 3/3 seeds by
V3-EXQ-1078 (`gru_param_max_delta = 0.0`, `latent_stack_tensors_changed = 0/53`) and
root-caused by `REE_assembly/evidence/planning/zself_causal_reach_trace_20260924.md`.

This module is the z_self sibling of `zworld_p0_warmup.run_zworld_p0` (SD-070) and
wraps `ree_core.latent.zself_p0.ZSelfP0Trainer` in the same integration shape: roll out
on a DEDICATED warmup env, buffer body/world observations and the action taken from
each, then run the recipe once.

DEFAULT OFF, BIT-IDENTICAL. `episodes <= 0` -- the default for every caller that does
not ask -- returns immediately without constructing the trainer, without an env reset,
and without touching a parameter or an RNG draw.

RNG NEUTRALITY IS LOAD-BEARING, NOT HYGIENE. `ZSelfP0Trainer.train()` builds its
prediction head with `nn.Linear(...)`, which draws from the GLOBAL torch RNG. Left
alone, merely turning this warmup on would shift every subsequent draw in the run, so an
ON-vs-OFF comparison would confound "z_self is now trained" (the effect under test) with
"the RNG stream moved" (pure noise). `_rng_neutral()` snapshots and restores the global
torch, numpy and python streams across the whole call.

WHAT AN ON ARM CAN AND CANNOT CLAIM. This warmup produces a z_self that carries held-out
body-forward and short-history information and that E1 responds to about 10x more
strongly. It does NOT produce behavioural consequence: the causal-reach trace measured
0/68 E3 ticks and 0/12 whole episodes with any action change under z_self intervention,
because no valuation consumer reads z_self (E3 scores world rollouts only; the
per-candidate E2 self-rollout is computed and discarded; DR-10 is default-off with no
z_self-derived producer). An INV-069 / MECH-113 retest on top of this measures
SELF-STATE QUALITY and E1 USE ONLY. See GFLAG-0481.

Phase ordering. Run this as P0s, BEFORE any phase that fits a predictor on z_self
(E1/E2 P0, the e2 contrastive warmup): a predictor fitted to the untrained projection
would otherwise be regressing on a representation this stage then moves underneath it.

MECH-094: not applicable. Trains on live observations; writes nothing to memory in any
non-waking state.

See `ree_core/latent/zself_p0.py` (the recipe and the measured candidate ranking),
`REE_assembly/docs/architecture/dr13_self_recurrence_temporal_depth.md`,
`REE_assembly/evidence/planning/substrate_queue.json` -> `sd_zself_training_path`.
"""

from __future__ import annotations

import contextlib
import dataclasses
import random
from typing import Any, Dict, Optional

import numpy as np
import torch

from ree_core.latent.zself_p0 import ZSelfP0Config, ZSelfP0Trainer

__all__ = ["run_zself_p0", "resolve_p0s_config"]


@contextlib.contextmanager
def _rng_neutral():
    """Restore the global torch + numpy + python RNG streams on exit.

    Without this, enabling the warmup shifts every downstream draw (the trainer's head
    is built with nn.Linear, which uses the global torch stream), confounding the ON/OFF
    contrast with a pure RNG-stream displacement.
    """
    t_state = torch.get_rng_state()
    np_state = np.random.get_state()
    py_state = random.getstate()
    try:
        yield
    finally:
        torch.set_rng_state(t_state)
        np.random.set_state(np_state)
        random.setstate(py_state)


def resolve_p0s_config(
    seed: int,
    dry_run: bool,
    config: Optional[ZSelfP0Config] = None,
) -> ZSelfP0Config:
    """The ZSelfP0Config a warmup runs with.

    `config=None`: the recipe defaults, seed-stamped. `dry_run` shrinks the update count
    and chunk batch so the smoke path exercises the real training code without spending
    a real run's compute -- it never touches the real-run config.
    """
    cfg = ZSelfP0Config(seed=int(seed)) if config is None else dataclasses.replace(
        config, seed=int(seed))
    if dry_run:
        cfg = dataclasses.replace(cfg, updates=8, batch_size=4, chunk_length=4)
    return cfg


def run_zself_p0(
    agent: Any,
    warmup_env: Any,
    seed: int,
    episodes: int,
    steps_per_episode: int,
    policy: Any,
    label: str = "",
    dry_run: bool = False,
    config: Optional[ZSelfP0Config] = None,
) -> Dict[str, Any]:
    """Run the z_self P0 body-forward-model warmup against `agent`.

    `warmup_env` MUST be a dedicated env instance, not the caller's training env: the
    rollout consumes env RNG, and reusing the training env would shift the layout
    sequence the real phases then see. Build it the same way and with the same seed as
    the training env so the warmup sees the matched state distribution.

    `policy` is any `_lib.capability_eval.Policy` -- typically `RandomPolicy(seed)`. The
    agent is deliberately NOT driven here: the recipe needs only the raw observations
    and the actions taken, and invoking `agent.sense()` / `agent.act()` would mutate
    residue, goal and clock state before the real P0 begins.

    Returns a diagnostic block for the manifest. `p0s_param_delta` is the attributable
    parameter movement (the necessary half of the acceptance criteria) and
    `p0s_holdout_before` / `p0s_holdout_after` are the sufficient half -- effective
    rank, z norm, and held-out R^2 on episodes the objective never saw, against the
    untrained baseline on the SAME episodes. A caller that reports the parameter delta
    WITHOUT the holdout pair cannot tell a trained self-state from a collapsed one: the
    causal-reach trace measured two candidate objectives that move both modules and
    destroy the representation.
    """
    if episodes <= 0:
        return {"p0s_recipe": "zself_p0", "p0s_ran": False, "p0s_reason": "episodes<=0"}

    cfg = resolve_p0s_config(seed, dry_run, config)
    out: Dict[str, Any] = {"p0s_recipe": "zself_p0", "p0s_ran": True}
    out["p0s_config"] = dataclasses.asdict(cfg)

    with _rng_neutral():
        trainer = ZSelfP0Trainer(agent, cfg)

        for ep in range(int(episodes)):
            _flat0, obs_dict = warmup_env.reset()
            policy.reset(warmup_env)

            for _step in range(int(steps_per_episode)):
                action = policy.act(warmup_env, obs_dict)
                trainer.observe(
                    obs_dict["body_state"], obs_dict["world_state"], action)
                with torch.no_grad():
                    _flat, _harm, done, _info, obs_dict = warmup_env.step(action)
                if done:
                    break
            # The final observation carries no action taken FROM it, but it IS the
            # prediction target of the last transition -- record it, then close.
            trainer.observe(obs_dict["body_state"], obs_dict["world_state"], None)
            trainer.end_episode()

            cur = ep + 1
            if cur == 1 or cur % 50 == 0 or cur == int(episodes):
                print(
                    "  [train] %s seed=%d phase=P0s ep %d/%d (z_self body forward model)"
                    % (label or "zself_p0", int(seed), cur, int(episodes)),
                    flush=True,
                )

        out["p0s_n_episodes"] = int(trainer.n_episodes)
        out["p0s_n_buffered_transitions"] = int(trainer.n_buffered)
        out["p0s_used_self_recurrence"] = bool(trainer.uses_self_recurrence())

        # The trainer refuses an undersized buffer rather than producing a
        # confident-looking result on too little data. Surface that as a RECORDED
        # refusal: without it the run would proceed as if z_self had been trained, which
        # is the exact silent failure this module exists to end.
        try:
            stats = trainer.train()
        except ValueError as exc:
            out["p0s_ran"] = False
            out["p0s_reason"] = "trainer_refused_buffer: %s" % (exc,)
            print(
                "  [P0s-REFUSAL] %s seed=%d: %s" % (label or "zself_p0", int(seed), exc),
                flush=True,
            )
            return out

    out["p0s_mean_loss"] = stats.get("mean_loss")
    out["p0s_first10_loss"] = stats.get("first10_loss")
    out["p0s_final_loss"] = stats.get("final_loss")
    out["p0s_n_updates"] = stats.get("n_updates")
    out["p0s_param_delta"] = stats.get("param_delta")
    out["p0s_holdout_before"] = stats.get("holdout_before")
    out["p0s_holdout_after"] = stats.get("holdout_after")
    before = stats.get("holdout_before") or {}
    after = stats.get("holdout_after") or {}
    # Compact top-level deltas for the manifest. None on either side propagates as None
    # (cannot determine), never as 0.0 -- a probe that could not run must not read as a
    # probe that ran and found no gain.
    for key in ("R2_next_body", "R2_action_history", "z_eff_rank", "z_norm_mean"):
        b, a = before.get(key), after.get(key)
        out["p0s_delta_" + key] = (
            (float(a) - float(b)) if (b is not None and a is not None) else None)
    return out
