"""SD-070 z_world encoder warmup for the `_train_all_on_agent` driver family.

WHY THIS EXISTS. The P0/P1 warmup shared by the x728/x734/x737/x742 drivers builds three
optimizer groups -- e2, the lateral-PFC bias head, and the OFC devaluation head -- and NONE
of them covers a single `latent_stack` parameter. So `split_encoder.world_encoder` is never
stepped and z_world stays a FROZEN RANDOM PROJECTION for the whole run, with no error and no
warning. Measured by `_lib/zworld_encoder_guard.py` on two independent drivers:

    V3-EXQ-737a  0 of 61 latent_stack tensors changed (world_encoder 0 of 4) at p0_episodes=200
    V3-EXQ-728   same signature, 3 of 3 seeds, on 728's OWN _train_all_on_agent copy (:522)

Every experiment on this path that assumed a prediction-trained z_world silently measured a
random projection instead. Diagnosis:
`REE_assembly/evidence/planning/zworld_bc_install_failure_V3-EXQ-780_2026-07-19.md` section 6c.

WHAT THE FIX IS NOT. It is NOT "enable the prescribed P0". That is refuted in-corpus: SD-009
event-contrastive CE + SD-018 proximity MSE, online at batch=1, COLLAPSES z_world to
participation ratio ~1.06 (SD-070, measured 2026-07-18). Per the confirmed V3-EXQ-783
adjudication the fix needs two things and SD-070 supplies both: (a) a gradient path that
actually reaches `latent_stack`, and (b) a supervision target the world channel determines.

WHAT THIS MODULE DOES. It wraps `ree_core.latent.zworld_p0.ZWorldP0Trainer` in the exact
integration V3-EXQ-783 already validated (`v3_exq_783_...py:428-512`): roll out, buffer
`world_state` plus the SD-018 proximity target, then run the recipe once. Target behaviour,
from SD-070's own validation: world-path weight-delta > 0, PR retention ~0.63, held-out
grounding lift +0.23..+0.47, with P0/P1 phase separation preserved.

RNG NEUTRALITY IS LOAD-BEARING, NOT HYGIENE. `ZWorldP0Trainer.train()` seeds its own
`torch.Generator` for shuffling and batching, but it constructs its auxiliary heads with
`nn.Linear(...)`, which draws from the GLOBAL torch RNG. Left alone, merely turning this
warmup on would shift every subsequent draw in P0 and P1 -- so an ON-vs-OFF comparison would
confound "the encoder is now trained" (the effect under test) with "the RNG stream moved"
(pure noise). `_rng_neutral()` snapshots and restores the global torch and numpy streams
across the whole call, so this function is a no-op on both. The env rollout is likewise run
on a CALLER-SUPPLIED warmup env, never on the training env, so the training env's own layout
sequence is untouched.

Phased training (unchanged, still mandatory): P0a = this recipe -> P0b = the existing e2
contrastive warmup, now over a MEANINGFUL z_world -> P1 = REINFORCE on stop-gradient latents
with the encoder optimiser not stepped -> P2 = measurement. Ordering matters: e2 regresses on
z_world, so training the encoder AFTER e2 would leave e2 fitted to the random projection.

MECH-094: not applicable. Trains on live observations; writes nothing to memory in any
non-waking state.

See `REE_assembly/docs/architecture/sd_070_zworld_p0_anticollapse_recipe.md`,
`experiments/_lib/zworld_encoder_guard.py` (the detector this is the remedy for),
`REE_assembly/evidence/planning/substrate_queue.json` -> `sd_zworld_warmup_optimizer_group`.
"""

from __future__ import annotations

import contextlib
from typing import Any, Dict, Optional

import numpy as np
import torch

from ree_core.latent.zworld_p0 import ZWorldP0Config, ZWorldP0Trainer

__all__ = ["resource_prox_target", "run_zworld_p0"]


@contextlib.contextmanager
def _rng_neutral():
    """Restore the global torch + numpy RNG streams on exit.

    Without this, enabling the warmup shifts every downstream draw (head construction calls
    nn.Linear, which uses the global torch stream), confounding the ON/OFF contrast with a
    pure-noise offset. See the module docstring.
    """
    torch_state = torch.get_rng_state()
    numpy_state = np.random.get_state()
    try:
        yield
    finally:
        torch.set_rng_state(torch_state)
        np.random.set_state(numpy_state)


def resource_prox_target(obs_dict: Dict[str, Any]) -> Optional[float]:
    """SD-018 resource-proximity regression target, read from `resource_field_view`.

    Vendored from `v3_exq_783_zworld_granularity_training_crossing.py:417` so the driver
    family and the SD-070 validation harness compute the identical target. Returns None when
    the channel is absent; the trainer treats that as an unlabelled sample rather than as a
    zero, which would be a false "resource is maximally far" label.
    """
    rfv = obs_dict.get("resource_field_view")
    if rfv is None:
        return None
    try:
        return float(torch.as_tensor(rfv).max().item())
    except Exception:
        return None


def run_zworld_p0(
    agent: Any,
    warmup_env: Any,
    seed: int,
    episodes: int,
    steps_per_episode: int,
    policy: Any,
    label: str = "",
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Run the SD-070 P0a encoder warmup against `agent.latent_stack`.

    `warmup_env` MUST be a dedicated env instance, not the caller's training env: the rollout
    consumes env RNG, and reusing the training env would shift the layout sequence P0b/P1 then
    see. Build it the same way and with the same seed as the training env so the warmup sees
    the matched state distribution.

    `policy` is any `_lib.capability_eval.Policy` -- typically `RandomPolicy(seed)`. The agent
    is deliberately NOT driven here: the recipe needs only `world_state`, and invoking
    `agent.sense()` would mutate residue / goal / clock state before the real P0 begins.

    Returns a diagnostic block for the manifest. Trains exactly `split_encoder.world_encoder`
    + `world_precision_logit` -- the parameter set the V3-EXQ-783 weight-delta readiness check
    and `zworld_encoder_guard.assert_world_encoder_trained` both watch.
    """
    if episodes <= 0:
        return {"p0a_recipe": "sd070", "p0a_ran": False, "p0a_reason": "episodes<=0"}

    # A dry run buffers a few dozen observations, far too few for the recipe's batch
    # statistics -- the trainer refuses such a buffer BY DESIGN rather than returning a
    # confident-looking result. Scale the batch down explicitly for the smoke path so it still
    # exercises the real training code, and never touch the real-run config.
    cfg = (
        ZWorldP0Config(seed=int(seed), batch_size=8, epochs=2)
        if dry_run else ZWorldP0Config(seed=int(seed))
    )

    out: Dict[str, Any] = {"p0a_recipe": "sd070", "p0a_ran": True}

    with _rng_neutral():
        trainer = ZWorldP0Trainer(agent.latent_stack, cfg)

        for ep in range(int(episodes)):
            _flat0, obs_dict = warmup_env.reset()
            policy.reset(warmup_env)

            for _step in range(int(steps_per_episode)):
                world_obs = obs_dict["world_state"].float()
                trainer.observe(world_obs, resource_prox_target(obs_dict))

                action = policy.act(warmup_env, obs_dict)
                with torch.no_grad():
                    _flat, _harm, done, _info, obs_dict = warmup_env.step(action)
                if done:
                    break

            cur = ep + 1
            if cur == 1 or cur % 50 == 0 or cur == int(episodes):
                print(
                    "  [train] %s seed=%d phase=P0a ep %d/%d (SD-070 z_world encoder)"
                    % (label or "zworld_p0", int(seed), cur, int(episodes)),
                    flush=True,
                )

        out["p0a_n_buffered"] = int(trainer.n_buffered)

        # The trainer refuses an undersized buffer rather than producing a confident-looking
        # result on undefined batch statistics. Surface that as a recorded refusal: the caller
        # still has the guard downstream, which will now correctly report the encoder as
        # untrained instead of the run silently proceeding as if it had been trained.
        try:
            stats = trainer.train()
        except ValueError as exc:
            out["p0a_ran"] = False
            out["p0a_reason"] = "trainer_refused_buffer: %s" % (exc,)
            print(
                "  [P0a-REFUSAL] %s seed=%d: %s" % (label or "zworld_p0", int(seed), exc),
                flush=True,
            )
            return out

    out["p0a_mean_loss"] = stats.get("mean_loss")
    out["p0a_final_loss"] = stats.get("final_loss")
    out["p0a_n_steps"] = stats.get("n_steps")
    out["p0a_variance_term"] = stats.get("variance_term")
    out["p0a_covariance_term"] = stats.get("covariance_term")
    out["p0a_used_proximity_head"] = stats.get("used_proximity_head")
    out["p0a_used_reconstruction_head"] = stats.get("used_reconstruction_head")
    out["p0a_grounding_label_balance"] = stats.get("label_balance")
    # The discriminativeness readout, recorded because the anti-collapse gate can be satisfied
    # VACUOUSLY -- a regulariser can hold the participation ratio up while the encoder learns
    # nothing -- so a PR verdict is not interpretable without it.
    out["p0a_holdout"] = stats.get("holdout")
    ho = stats.get("holdout") or {}
    out["p0a_holdout_mean_lift"] = ho.get("mean_lift")
    return out
