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
import dataclasses
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch

from ree_core.latent.zworld_p0 import ZWorldP0Config, ZWorldP0Trainer

__all__ = ["resource_prox_target", "run_zworld_p0", "resolve_p0a_config", "resolve_target_fn"]


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


def resolve_p0a_config(
    seed: int,
    dry_run: bool,
    resource_field_weight: float = 0.0,
    config: Optional[ZWorldP0Config] = None,
) -> ZWorldP0Config:
    """The ZWorldP0Config a warmup runs with. Factored out so the LEGACY construction stays
    byte-for-byte what every pre-2026-09-09 caller got (contract-tested), while a caller that
    passes `config=` gets its own objective weights / seed-stamped.

    `config=None` (every existing caller): the legacy construction -- seed + resource_field_weight
    only, plus the dry-run batch/epoch shrink. `config=<ZWorldP0Config>`: that config, with `seed`
    overwritten to the warmup seed (the recipe seeds its own shuffling/batching from it) and, under
    `dry_run`, the same batch/epoch shrink applied on top. Passing BOTH a config and a non-zero
    `resource_field_weight` is refused rather than silently resolved in either direction: the
    config carries its own `resource_field_weight`, and two sources for one weight is exactly the
    ambiguity V3-EXQ-978's seam note warns about.
    """
    if config is None:
        return (
            ZWorldP0Config(seed=int(seed), batch_size=8, epochs=2,
                           resource_field_weight=float(resource_field_weight))
            if dry_run else ZWorldP0Config(seed=int(seed),
                                           resource_field_weight=float(resource_field_weight))
        )
    if float(resource_field_weight) != 0.0:
        raise ValueError(
            "run_zworld_p0: pass resource_field_weight EITHER as the kwarg OR inside config=, "
            "not both (config.resource_field_weight=%r, kwarg=%r)"
            % (config.resource_field_weight, resource_field_weight)
        )
    cfg = dataclasses.replace(config, seed=int(seed))
    if dry_run:
        cfg = dataclasses.replace(cfg, batch_size=8, epochs=2)
    return cfg


def resolve_target_fn(
    target_fn: Optional[Callable[[Dict[str, Any]], Optional[float]]] = None,
) -> Callable[[Dict[str, Any]], Optional[float]]:
    """The per-step scalar regression target fed to `trainer.observe(world_obs, target)`.

    `None` (every existing caller) = `resource_prox_target`, the SD-018 resource-proximity target,
    unchanged. A caller may supply any `obs_dict -> Optional[float]` -- this is the seam that makes
    the GOV-MATCHAUX-1 matched arbitrary-auxiliary control constructible WITHOUT a substrate change:
    the same `resource_proximity_head`, the same MSE, the same `proximity_weight`, the same
    cadence and examples, differing only in what the scalar MEANS. Before this seam existed the
    target was a hardcoded call inside the rollout loop (IGW-20260908-233 blocked EXP-1397 on
    exactly that; `experiment_proposals.v1.json` EVB-1712 `gating_reason`). Return `None` from the
    callable for an unlabelled step; the trainer masks it out rather than reading it as zero.
    """
    return resource_prox_target if target_fn is None else target_fn


def run_zworld_p0(
    agent: Any,
    warmup_env: Any,
    seed: int,
    episodes: int,
    steps_per_episode: int,
    policy: Any,
    label: str = "",
    dry_run: bool = False,
    resource_field_weight: float = 0.0,
    config: Optional[ZWorldP0Config] = None,
    target_fn: Optional[Callable[[Dict[str, Any]], Optional[float]]] = None,
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

    `resource_field_weight` is the SD-018 AMEND directional-field leg's P0a loss weight
    (`ZWorldP0Config.resource_field_weight`). DEFAULT 0.0 = leg OFF = bit-identical to the
    pre-2026-09-02 behaviour for every existing caller; nothing changes unless a caller asks.

    Why this parameter exists at all: `ZWorldP0Config.resource_field_weight` defaults to 0.0
    while its scalar sibling `proximity_weight` defaults to 0.5, and this function is the ONLY
    P0a path `allon_training._train_all_on_agent` uses -- which also calls neither
    `agent.compute_resource_proximity_loss` nor `agent.compute_resource_field_loss`. So before
    this seam existed the SD-018 amend's directional head (landed in ree-v3 `028a625`) received
    ZERO gradient steps from any driver in the x734/737/808/948 family: an "ON" arm would have
    differed from its OFF sibling only by an untrained randomly-initialised head that nothing
    reads, i.e. a manipulation that cannot reach the DV. Measured + recorded 2026-09-02 while
    authoring the amend's own owed validation (V3-EXQ-978); see
    `chip-20260902-sd018-p0a-field-weight-seam`.

    Requires `use_resource_field_head=True` on the agent's LatentStackConfig -- the weight
    alone does nothing, because the trainer's leg is gated on the head existing as well.
    `p0a_used_resource_field_head` in the returned block reports whether the leg ACTUALLY ran,
    so a caller that set one half and not the other reads a False rather than assuming.

    `config` / `target_fn` (2026-09-09, V3-EXQ-1017 seam; see `resolve_p0a_config` and
    `resolve_target_fn`): DEFAULT None = the legacy construction and the SD-018 proximity target,
    bit-identical for every existing caller. A driver that needs a P0a objective other than the
    SD-070 default (a generic-only compression, a matched arbitrary-auxiliary control) passes
    them here instead of re-implementing this loop. The resolved config and target name are
    recorded in the returned block (`p0a_config`, `p0a_target`) so the manifest says what
    actually trained.
    """
    if episodes <= 0:
        return {"p0a_recipe": "sd070", "p0a_ran": False, "p0a_reason": "episodes<=0"}

    # A dry run buffers a few dozen observations, far too few for the recipe's batch
    # statistics -- the trainer refuses such a buffer BY DESIGN rather than returning a
    # confident-looking result. Scale the batch down explicitly for the smoke path so it still
    # exercises the real training code, and never touch the real-run config.
    cfg = resolve_p0a_config(seed, dry_run, resource_field_weight, config)
    target = resolve_target_fn(target_fn)

    out: Dict[str, Any] = {"p0a_recipe": "sd070", "p0a_ran": True}
    out["p0a_config"] = dataclasses.asdict(cfg)
    out["p0a_target"] = str(getattr(target, "name", None) or getattr(target, "__name__", "custom"))

    with _rng_neutral():
        trainer = ZWorldP0Trainer(agent.latent_stack, cfg)

        for ep in range(int(episodes)):
            _flat0, obs_dict = warmup_env.reset()
            policy.reset(warmup_env)

            for _step in range(int(steps_per_episode)):
                world_obs = obs_dict["world_state"].float()
                trainer.observe(world_obs, target(obs_dict))

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
    # SD-018 AMEND directional-field leg. `p0a_used_resource_field_head` is the ground truth
    # that the leg RAN (weight > 0 AND the head exists AND world_obs was wide enough), not
    # merely that a weight was passed -- so a half-configured caller reads False here rather
    # than assuming its ON arm was manipulated. The holdout block is the mechanism readout:
    # held-out field MSE against a constant-mean predictor, i.e. it separates a decodable
    # directional field from a fitted mean.
    out["p0a_resource_field_weight"] = float(cfg.resource_field_weight)
    out["p0a_used_resource_field_head"] = stats.get("used_resource_field_head")
    out["p0a_resource_field_holdout"] = stats.get("resource_field_holdout")
    # SD-106 generic bottleneck variance preservation. `p0a_used_preservation_head` is the
    # ground truth that the leg RAN, not merely that a weight was passed, and
    # `p0a_used_world_encoder_skip` that the zero-init bypass module exists at all -- both
    # read False on a half-configured caller rather than letting it assume its ON arm was
    # manipulated. The holdout block is the mechanism readout: held-out R^2 of world_obs from
    # the 32-dim code, directly comparable with the PCA-32 anchor SD-106's acceptance target
    # names. Without these three the flag can read as enabled while the leg is inert.
    out["p0a_preservation_weight"] = float(getattr(cfg, "preservation_weight", 0.0))
    out["p0a_used_preservation_head"] = stats.get("used_preservation_head")
    out["p0a_used_world_encoder_skip"] = stats.get("used_world_encoder_skip")
    out["p0a_preservation_holdout"] = stats.get("preservation_holdout")
    out["p0a_grounding_label_balance"] = stats.get("label_balance")
    # The discriminativeness readout, recorded because the anti-collapse gate can be satisfied
    # VACUOUSLY -- a regulariser can hold the participation ratio up while the encoder learns
    # nothing -- so a PR verdict is not interpretable without it.
    out["p0a_holdout"] = stats.get("holdout")
    ho = stats.get("holdout") or {}
    out["p0a_holdout_mean_lift"] = ho.get("mean_lift")
    return out
