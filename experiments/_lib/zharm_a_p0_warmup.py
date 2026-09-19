"""SD-011 affective-harm-encoder (z_harm_a) P0h warmup for the `_train_all_on_agent` family.

WHY THIS EXISTS. The P0/P1 warmup shared by the x728/x734/x737/x742 drivers builds three
optimizer groups -- e2, the lateral-PFC bias head, and the OFC devaluation head -- and none of
them covers a single `AffectiveHarmEncoder` parameter. So `latent_stack.affective_harm_encoder`
is never stepped and `z_harm_a` stays a FROZEN RANDOM PROJECTION for the whole run, with no
error and no warning. Measured 2026-09-18 by parameter-identity intersection against the
optimizers built at `allon_training.py:541-548`:

    latent_stack param tensors             : 53
    affective_harm_encoder param tensors   : 4
    affective encoder params COVERED       : 0
    AFFECTIVE ENCODER REACHABLE BY TRAINER : False

This is the exact sibling of the V3-EXQ-780 `z_world` defect that `allon_training.py:475-476`
already records for the world stream, and it is remedied here the same way the z_world one was:
an OPT-IN, default-OFF P0 stage that owns its own optimizer group. Diagnosis:
`REE_assembly/evidence/planning/sd086_zharma_readout_precondition_staged_20260918.md` sec 4b.

WHAT SIGNAL THIS TRAINS ON, AND WHY IT IS NOT A FREE CHOICE. The architecture already
specifies it, in two places that agree:

  * `LatentStackConfig.harm_history_len` / `z_harm_a_aux_loss_weight` (SD-011 "second source"):
    when `harm_history_len > 0` the encoder gains a `harm_accum_head` whose job is to predict
    accumulated harm exposure from `z_harm_a`, "forcing the affective encoder to integrate
    temporal harm information that z_harm_s does not receive" (`agent.compute_harm_accum_loss`).
  * `ree_core/predictors/e2_harm_a.py`, "Phased training required", names the missing stage by
    name: "P0: AffectiveHarmEncoder warmup (z_harm_a encoder trains on accumulated-harm /
    harm-surprise supervision per SD-020)".

So this module does NOT pick a new objective. It calls `agent.compute_harm_accum_loss`
unchanged -- which is also where SD-020's precision-weighted prediction-error target lives,
behind the caller's existing `REEConfig.harm_surprise_pe_enabled` flag. The SD-011 EMA target
and the SD-020 surprise target are therefore the SAME loss under a config the caller already
owns, not two rival recipes this module would have had to arbitrate between. The one existing
call site of that loss (`_lib/baselines/exq610_inv074_crystallization_baseline.py:538`) is the
precedent for the optimizer shape used here (Adam, lr 5e-4, grad-norm clip 1.0).

ORDERING: P0h runs AFTER the SD-070 z_world P0a and BEFORE the P0b e2 contrastive warmup. The
two encoder stages are independent (z_harm_a bypasses the world path entirely), but P0b and P1
both drive the FULL agent loop, and `z_harm_a` feeds E3 commit gating and ARC-016 harm-variance
gating on every tick of it. Training the affective encoder afterwards would leave every
selection decision in P0b/P1 taken against the random projection -- the same defect one phase
later, which is the mistake the z_world ordering note already warns about.

DEFAULT `zharm_a_p0_episodes=0` IS EXACTLY THE PRIOR BEHAVIOUR, bit-identical: no optimizer, no
extra tensor, and no RNG draw. Every existing caller is unaffected until it opts in.

RNG NEUTRALITY IS LOAD-BEARING, NOT HYGIENE. The warmup rollout consumes the global numpy
stream through the policy, so leaving it unguarded would shift every subsequent draw in P0b and
P1 -- an ON-vs-OFF comparison would then confound "the encoder is now trained" (the effect under
test) with "the RNG stream moved" (pure noise). This module reuses `zworld_p0_warmup._rng_neutral`
ITSELF rather than copying it, so the two stages cannot drift on the property that makes those
comparisons valid, and rolls out on a CALLER-SUPPLIED warmup env so the training env's own layout
sequence is untouched. `agent._harm_obs_ema` (the SD-020 expected-harm tracker, which
`compute_harm_accum_loss` mutates) is snapshotted and restored for the same reason.

WHAT IT REFUSES, LOUDLY, RATHER THAN NO-OPPING. `compute_harm_accum_loss` returns a zero loss
when `harm_history_len <= 0` or when the aux head did not run -- correct for its existing
per-tick callers, and silently fatal for a warmup, which would report having trained while
stepping on a zero gradient. Every such condition is checked up front and recorded in
`p0h_reason` with `p0h_ran: False`.

ML/AI engineering note (Layer 7). Training online at batch=1 over a temporally correlated
rollout is the standard correlated-update hazard, and the in-corpus precedent to respect is
SD-070: the prescribed z_world P0 collapsed the code to participation ratio ~1.06 exactly that
way. Three things keep it bounded here and none of them is an assumption: the objective is a
1-d regression through a 2-layer head (not a contrastive objective over a 32-d code, which is
what collapsed); gradients are norm-clipped at 1.0 per step; and the last `holdout_episode_frac`
of EPISODES are held out of training entirely and scored before and after, so a model that fit
the training order rather than the signal shows up as a held-out loss that did not fall. The
holdout is by episode, not by tick, because adjacent ticks share an EMA and a tick-level split
would leak.

MECH-094: not applicable. Trains on live observations; writes nothing to memory in any
non-waking state.

See `REE_assembly/docs/architecture/sd_011_dual_nociceptive_streams.md`,
`REE_assembly/docs/architecture/sd_020_harm_surprise_pe.md`,
`experiments/_lib/zworld_p0_warmup.py` (the z_world sibling this mirrors),
`REE_assembly/evidence/planning/substrate_queue.json` -> `sd_zharm_a_warmup_optimizer_group`.
"""

from __future__ import annotations

import contextlib
import dataclasses
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

# Deliberately the SAME context manager object as the z_world stage uses, not a copy: the two
# P0 stages must not be able to drift on RNG neutrality, which is what makes every fixed-seed
# ON/OFF comparison across either of them valid. Pinned by
# tests/contracts/test_zharm_a_p0_warmup.py::test_rng_neutral_is_shared_with_zworld_stage.
from experiments._lib.zworld_p0_warmup import _rng_neutral
from ree_core.latent.stack import LatentState

__all__ = [
    "ZHarmAP0Config",
    "run_zharm_a_p0",
    "affective_encoder_parameters",
    "encoder_weight_snapshot",
    "encoder_weight_delta",
    "recover_effective_target",
    "current_precision_norm",
    "resolve_p0_precision_norm",
]


@dataclass
class ZHarmAP0Config:
    """Objective/optimiser settings for the P0h affective-encoder warmup.

    Defaults follow the one existing `compute_harm_accum_loss` call site
    (`exq610_inv074_crystallization_baseline.py`: Adam at LR_ENC_AUX=5e-4, grad-norm clip 1.0)
    so this stage is not a new recipe with new hyper-parameters, only a new place that recipe
    is reached from. `epochs` and `holdout_episode_frac` have no precedent there because that
    driver trains online inside its own episode loop and never holds anything out; they are
    this module's own, and both are reported in the returned block.
    """

    seed: int = 0
    lr: float = 5e-4
    epochs: int = 4
    max_grad_norm: float = 1.0
    holdout_episode_frac: float = 0.2
    loss_weight_note: str = "weight is LatentStackConfig.z_harm_a_aux_loss_weight (agent-side)"

    # --- 2026-09-19 two-arm re-specification diagnostic (user decision OPTION H) ---------
    # Both default to EXACTLY the prior behaviour. Neither is a new recipe: each isolates one
    # of the two measured causes of the 2026-09-18 readiness failure, so the diagnostic can
    # say which one (if either) is what actually blocks training.
    #
    # ARM F -- `target_source`. "accumulated_harm" (default) is SD-011 as specified: the env's
    # CUMULATIVE EPISODE MEAN, which converges by construction and measured CV 0.076.
    # "harm_exposure" is the PER-TICK scalar the same env emits at `harm_obs[-1]`, measured CV
    # 0.694 -- ~9x the relative dispersion. Changing this changes what z_harm_a MEANS (SD-011's
    # whole point is that the affective stream integrates ACCUMULATED harm, which is what
    # distinguishes it from z_harm_s), so it is a DIAGNOSTIC lever, not a default to flip.
    #
    # ARM E -- `p0_precision_norm`. SD-020's target is scaled by
    # `min(e3.current_precision/500, 3.0)` (ARC-016). At P0 that is 0.004, because
    # `current_precision` sits at its INIT 2.0 while the /500 divisor presupposes the ~95-100
    # of a TRAINED agent -- a ~250x attenuation, and a PHASE-ORDERING contradiction rather than
    # a tuning miss, since P0 runs before the agent has any precision to couple to. Setting
    # this pins the coupling to a chosen value FOR THE DURATION OF THIS STAGE ONLY (the
    # underlying E3 running-variance is snapshotted and restored), which is what "decouple the
    # P0 target from precision" means operationally. None (default) = untouched.
    # Inert unless `harm_surprise_pe_enabled` is on -- that is the only branch reading precision
    # -- and the returned block says so rather than letting a caller assume its arm was applied.
    target_source: str = "accumulated_harm"
    p0_precision_norm: Optional[float] = None
    # DEFAULT-ON FLOOR (user decision OPTION I, 2026-09-19T02:31Z), and the one setting here
    # that changes behaviour for a caller who opts into P0h. 0.1 is mid-plateau on the measured
    # pin sweep -- clear of the 0.004->0.02 cliff below and of the mild decay above -- so it is
    # chosen for the flatness around it, not as an optimum.
    #
    # A FLOOR, NOT A PIN: the applied value is max(agent's own precision_norm, this), so an
    # agent that already sits above it is never dragged DOWN. That matters because the whole
    # finding is that ARC-016's coupling is right at RUNTIME and wrong at P0 -- a pin would
    # discard a genuine trained-agent precision, whereas a floor only rescues the P0 case where
    # precision has not yet had a chance to exist. `p0_precision_norm` (an explicit pin) still
    # overrides it; set this to None to restore the pre-2026-09-19 behaviour exactly.
    p0_precision_norm_floor: Optional[float] = 0.1


def affective_encoder_parameters(agent: Any) -> List[torch.nn.Parameter]:
    """Every parameter the P0h optimizer group covers, in a stable order.

    This is `affective_harm_encoder.parameters()` -- the encoder MLP plus, when
    `harm_history_len > 0`, the `harm_accum_head` that supplies the gradient. Returns an empty
    list when the affective stream is off, which the caller reads as a refusal rather than as
    an empty-but-fine optimizer.
    """
    enc = getattr(getattr(agent, "latent_stack", None), "affective_harm_encoder", None)
    if enc is None:
        return []
    return [p for p in enc.parameters()]


def encoder_weight_snapshot(agent: Any) -> List[torch.Tensor]:
    """Detached clones of the affective-encoder parameters, for the weight-delta readiness
    readout. The readout this feeds is the direct analog of `zworld_encoder_guard`'s
    world-path weight delta: it is what distinguishes "the stage ran" from "the stage moved
    the parameters an experiment's DV depends on"."""
    return [p.detach().clone() for p in affective_encoder_parameters(agent)]


def encoder_weight_delta(
    before: List[torch.Tensor], after: List[torch.Tensor],
) -> Tuple[int, float]:
    """(n_tensors_changed, max_abs_delta) between two snapshots. Length mismatch is a
    programming error, not a measurement, so it raises rather than reporting a zero delta --
    a silent zero here reads identically to the defect this module exists to fix."""
    if len(before) != len(after):
        raise ValueError(
            "encoder_weight_delta: snapshot length mismatch (%d vs %d)"
            % (len(before), len(after))
        )
    n_changed = 0
    max_abs = 0.0
    for b, a in zip(before, after):
        d = (a - b).abs()
        m = float(d.max().item()) if d.numel() else 0.0
        if m > 0.0:
            n_changed += 1
        max_abs = max(max_abs, m)
    return n_changed, max_abs


def current_precision_norm(agent: Any) -> Optional[float]:
    """The ARC-016 factor SD-020's target is multiplied by: `min(current_precision/500, 3.0)`.

    Read back from the agent rather than assumed, because the whole point of arm E is that this
    number is NOT what SD-020's `/500` divisor presupposes: measured 0.004 at P0 (precision at
    its init 2.0) against the ~0.19-1.0 a trained agent's ~95-500 would give.
    """
    e3 = getattr(agent, "e3", None)
    if e3 is None:
        return None
    try:
        return min(float(e3.current_precision) / 500.0, 3.0)
    except Exception:
        return None


def resolve_p0_precision_norm(
    baseline: Optional[float], pin: Optional[float], floor: Optional[float],
) -> Optional[float]:
    """The precision_norm the P0h stage should run at, or None to leave the agent untouched.

    Precedence, and each clause earns its place:
      * an explicit `pin` wins outright -- that is what a diagnostic arm sets;
      * otherwise a `floor` applies only when the agent is BELOW it, so a trained agent's own
        higher precision is never dragged down (a pin would do exactly that, and discarding a
        genuine runtime precision is the opposite of what the SD-020 finding says);
      * otherwise None -- no mutation at all, which keeps the no-op path free of float noise.
    """
    if pin is not None:
        return float(pin)
    if floor is None or baseline is None:
        return None
    return float(floor) if float(baseline) < float(floor) else None


@contextlib.contextmanager
def _pinned_precision_norm(agent: Any, target_norm: Optional[float]):
    """Pin `precision_norm` to `target_norm` for the duration of this stage ONLY.

    `current_precision` is a read-only property over E3's running variance
    (`1/(running_variance + 1e-6)`), so pinning the norm means solving for that variance:
    `running_variance = 1/(500 * target_norm) - 1e-6`. The prior value is snapshotted and
    restored on exit, which is what makes this "during P0 only" by construction rather than by
    convention -- the same discipline the RNG streams and `_harm_obs_ema` already get here.

    Safe precisely because the warmup does NOT drive the agent: E3 is never stepped during P0h,
    so nothing else reads the variance while it is pinned. It would NOT be safe in a stage that
    called `select_action`.

    `target_norm=None` (default) yields without touching anything.
    """
    e3 = getattr(agent, "e3", None)
    if target_norm is None or e3 is None or not hasattr(e3, "_running_variance"):
        yield None
        return
    tn = float(target_norm)
    if tn <= 0.0:
        raise ValueError("p0_precision_norm must be > 0 (got %r)" % (target_norm,))
    prev = e3._running_variance
    e3._running_variance = 1.0 / (500.0 * tn) - 1e-6
    try:
        yield current_precision_norm(agent)
    finally:
        e3._running_variance = prev


def _target_scalar(obs_dict: Dict[str, Any], target_source: str) -> Optional[float]:
    """The per-tick supervision scalar this arm regresses, read from the env's own obs.

    "accumulated_harm" -- SD-011 as specified: the env's cumulative episode mean.
    "harm_exposure"    -- the PER-TICK scalar, taken from `harm_obs[-1]`, which is where
                          CausalGridWorldV2 writes it (harm_obs layout = hazard_field[25] +
                          resource_field[25] + harm_exposure[1]). Read from the SAME obs the
                          stage already consumes, so no env change and no new channel.

    Returns None when the channel is absent, which the caller treats as an unlabelled step
    rather than as a zero -- a false "no harm" label is exactly the kind of quiet corruption
    this whole line of work exists to surface.
    """
    if target_source == "accumulated_harm":
        v = obs_dict.get("accumulated_harm")
        return None if v is None else float(v)
    if target_source == "harm_exposure":
        ho = obs_dict.get("harm_obs")
        if ho is None or ho.reshape(-1).numel() < 1:
            return None
        return float(ho.reshape(-1)[-1].item())
    raise ValueError(
        "unknown target_source %r -- expected 'accumulated_harm' or 'harm_exposure'"
        % (target_source,)
    )


def recover_effective_target(agent: Any, accumulated_harm_target: float,
                             device: Optional[torch.device] = None) -> Optional[float]:
    """The scalar `compute_harm_accum_loss` is ACTUALLY regressing for this sample.

    Recovered from the agent's own loss rather than re-derived here, which is the point: with
    `harm_accum_pred` forced to ZERO the loss is `weight * mse(0, target) = weight * target**2`,
    so `target = sqrt(loss / weight)`. Both target paths are non-negative by construction --
    SD-011's `accumulated_harm` is clipped to [0, 1] by the env, SD-020's is
    `abs(actual - expected) * precision_norm` -- so the square root is exact, not an absolute
    value standing in for a signed quantity.

    WHY NOT read the buffered `accumulated_harm` directly: under SD-020
    (`harm_surprise_pe_enabled`) the head regresses a precision-weighted prediction error, not
    that scalar, and scoring it against the latter would compare it with a quantity it was never
    trained on. Going through the agent is the only way to score BOTH targets on the same
    footing without duplicating the derivation here and letting the copy drift.

    SIDE EFFECT, and it is the caller's to manage: under SD-020 this ADVANCES
    `agent._harm_obs_ema`, exactly as a real training call would -- which is why a readout must
    walk its samples in TEMPORAL order from a defined starting EMA, and restore it afterwards.

    Returns None when the aux-loss weight is non-positive (nothing is being regressed).
    Pinned by contracts C3h (SD-011: recovery reproduces the buffered scalar exactly) and C3i
    (SD-020: recovery reproduces |actual - expected| * precision_norm).
    """
    w = float(getattr(getattr(agent.config, "latent", None),
                      "z_harm_a_aux_loss_weight", 0.1))
    if w <= 0.0:
        return None
    if not bool(getattr(agent.config, "harm_surprise_pe_enabled", False)):
        # Non-PE path: `compute_harm_accum_loss` regresses the scalar it was handed, verbatim
        # (`target_val = accumulated_harm_target`). Returning it directly is exact AND avoids a
        # sign hazard the inversion below cannot: sqrt(loss/weight) recovers |target|, so any
        # target_source that could go negative would be silently rectified. Contract C3h pins
        # the two against each other on the path where both are valid.
        return float(accumulated_harm_target)
    if device is None:
        params = affective_encoder_parameters(agent)
        device = params[0].device if params else torch.device("cpu")
    with torch.no_grad():
        loss = float(agent.compute_harm_accum_loss(
            accumulated_harm_target, _loss_state(torch.zeros(1, 1, device=device), device),
        ))
    return float((max(loss, 0.0) / w) ** 0.5)


def _loss_state(harm_accum_pred: torch.Tensor, device: torch.device) -> LatentState:
    """A LatentState carrying ONLY `harm_accum_pred`.

    `compute_harm_accum_loss` reads exactly that one field off the state; everything else it
    needs it takes from the agent. Building a minimal state lets this stage reuse the canonical
    loss (SD-020 target switch included) WITHOUT calling `agent.sense()`, which would advance
    residue / goal / clock state before the real P0 begins -- the same reason the z_world stage
    does not drive the agent either.

    `harm_accum_pred` is safe to compute from the encoder directly: `LatentStack.encode`
    produces it from the UNBLENDED encoder output and never applies the SD-036 harm-decay blend
    to it (see the blend's own note in `latent/stack.py`), so this value is identical to the one
    a full `sense()` would have returned.
    """
    z = torch.zeros(1, 1, device=device)
    return LatentState(
        z_self=z, z_world=z, z_beta=z, z_theta=z, z_delta=z,
        precision={},
        harm_accum_pred=harm_accum_pred,
    )


def _refuse(reason: str, label: str, seed: int, cfg: Optional[ZHarmAP0Config] = None,
            ) -> Dict[str, Any]:
    out: Dict[str, Any] = {"p0h_recipe": "sd011_harm_accum", "p0h_ran": False,
                           "p0h_reason": reason}
    if cfg is not None:
        out["p0h_config"] = dataclasses.asdict(cfg)
    print(
        "  [P0h-REFUSAL] %s seed=%d: %s" % (label or "zharm_a_p0", int(seed), reason),
        flush=True,
    )
    return out


def run_zharm_a_p0(
    agent: Any,
    warmup_env: Any,
    seed: int,
    episodes: int,
    steps_per_episode: int,
    policy: Any,
    label: str = "",
    dry_run: bool = False,
    config: Optional[ZHarmAP0Config] = None,
) -> Dict[str, Any]:
    """Run the SD-011 P0h affective-harm-encoder warmup against `agent.latent_stack`.

    `warmup_env` MUST be a dedicated env instance, not the caller's training env: the rollout
    consumes env RNG, and reusing the training env would shift the layout sequence P0b/P1 then
    see. Build it the same way, with the same seed AND the same `harm_history_len`, as the
    training env, so the warmup sees the matched state distribution and the same input width.

    `policy` is any `_lib.capability_eval.Policy` -- typically `RandomPolicy(seed)`. The agent
    is deliberately NOT driven here (see `_loss_state`).

    Trains exactly `latent_stack.affective_harm_encoder` -- the 4-tensor parameter set the
    2026-09-18 coverage measurement found in no optimizer group on this path.

    Returns a diagnostic block for the manifest. `p0h_encoder_tensors_changed` /
    `p0h_encoder_max_abs_delta` are the ground truth that the stage MOVED the encoder, not
    merely that it was asked to: a caller that half-configured the agent reads a refusal or a
    zero delta here rather than assuming its ON arm was manipulated.
    """
    cfg = config if config is not None else ZHarmAP0Config(seed=int(seed))
    cfg = dataclasses.replace(cfg, seed=int(seed))
    if dry_run:
        cfg = dataclasses.replace(cfg, epochs=1)

    if int(episodes) <= 0:
        return {"p0h_recipe": "sd011_harm_accum", "p0h_ran": False, "p0h_reason": "episodes<=0"}

    # -- refusals: each of these would otherwise produce a confident-looking zero-gradient run --
    params = affective_encoder_parameters(agent)
    if not params:
        return _refuse(
            "affective_harm_encoder absent -- needs use_affective_harm_stream=True on the "
            "agent's LatentStackConfig", label, seed, cfg,
        )
    enc = agent.latent_stack.affective_harm_encoder
    if getattr(enc, "harm_accum_head", None) is None:
        return _refuse(
            "harm_accum_head absent -- needs LatentStackConfig.harm_history_len > 0; without "
            "it compute_harm_accum_loss returns a zero loss and the encoder would not move",
            label, seed, cfg,
        )
    if int(getattr(getattr(agent.config, "latent", None), "harm_history_len", 0)) <= 0:
        return _refuse(
            "agent.config.latent.harm_history_len <= 0 -- compute_harm_accum_loss short-"
            "circuits to a zero loss", label, seed, cfg,
        )
    if int(getattr(warmup_env, "harm_history_len", 0)) <= 0:
        return _refuse(
            "warmup_env.harm_history_len <= 0 -- the env emits no harm_history/accumulated_harm, "
            "so there is nothing to supervise on", label, seed, cfg,
        )

    device = next(iter(params)).device
    out: Dict[str, Any] = {"p0h_recipe": "sd011_harm_accum", "p0h_ran": True}
    out["p0h_config"] = dataclasses.asdict(cfg)
    out["p0h_n_encoder_tensors"] = len(params)
    out["p0h_harm_surprise_pe_enabled"] = bool(
        getattr(agent.config, "harm_surprise_pe_enabled", False)
    )
    out["p0h_target"] = (
        "sd020_precision_weighted_pe" if out["p0h_harm_surprise_pe_enabled"]
        else ("sd011_accumulated_harm_ema" if cfg.target_source == "accumulated_harm"
              else "sd011_per_tick_harm_exposure")
    )
    out["p0h_target_source"] = str(cfg.target_source)
    out["p0h_precision_norm_requested"] = cfg.p0_precision_norm
    out["p0h_precision_norm_floor"] = cfg.p0_precision_norm_floor
    out["p0h_precision_norm_baseline"] = current_precision_norm(agent)
    _resolved_pnorm = resolve_p0_precision_norm(
        out["p0h_precision_norm_baseline"], cfg.p0_precision_norm, cfg.p0_precision_norm_floor,
    )
    out["p0h_precision_norm_source"] = (
        "explicit_pin" if cfg.p0_precision_norm is not None
        else ("floor" if _resolved_pnorm is not None else "agent_untouched")
    )
    # An override that cannot bite is reported as INERT rather than left to look applied: only
    # the SD-020 branch reads precision at all.
    out["p0h_precision_override_inert"] = bool(
        _resolved_pnorm is not None and not out["p0h_harm_surprise_pe_enabled"]
    )

    before = encoder_weight_snapshot(agent)
    # The SD-020 expected-harm tracker is agent state that compute_harm_accum_loss MUTATES.
    # Restored below for the same reason the RNG streams are: a warmup on a separate env must
    # not leak its own history into the training phases that follow.
    harm_obs_ema_pre = getattr(agent, "_harm_obs_ema", None)

    # (harm_obs_a, harm_history, accumulated_harm) per tick, plus the episode index so the
    # holdout can be cut on an EPISODE boundary (adjacent ticks share an EMA; a tick-level
    # split would leak the target across it).
    buf: List[Tuple[torch.Tensor, Optional[torch.Tensor], float, int]] = []

    with _rng_neutral(), _pinned_precision_norm(agent, _resolved_pnorm) as achieved:
        out["p0h_precision_norm_applied"] = (
            achieved if achieved is not None else out["p0h_precision_norm_baseline"]
        )
        for ep in range(int(episodes)):
            _flat0, obs_dict = warmup_env.reset()
            policy.reset(warmup_env)

            for _step in range(int(steps_per_episode)):
                hoa = obs_dict.get("harm_obs_a")
                hh = obs_dict.get("harm_history")
                accum = _target_scalar(obs_dict, cfg.target_source)
                if hoa is not None and accum is not None:
                    buf.append((
                        hoa.float().unsqueeze(0).to(device),
                        None if hh is None else hh.float().unsqueeze(0).to(device),
                        float(accum),
                        ep,
                    ))

                action = policy.act(warmup_env, obs_dict)
                with torch.no_grad():
                    _flat, _harm, done, _info, obs_dict = warmup_env.step(action)
                if done:
                    break

            cur = ep + 1
            if cur == 1 or cur % 50 == 0 or cur == int(episodes):
                print(
                    "  [train] %s seed=%d phase=P0h ep %d/%d (SD-011 affective harm encoder)"
                    % (label or "zharm_a_p0", int(seed), cur, int(episodes)),
                    flush=True,
                )

        out["p0h_n_buffered"] = len(buf)
        if not buf:
            out["p0h_ran"] = False
            out["p0h_reason"] = (
                "no supervised samples buffered -- env emitted no harm_obs_a/accumulated_harm"
            )
            print(
                "  [P0h-REFUSAL] %s seed=%d: %s"
                % (label or "zharm_a_p0", int(seed), out["p0h_reason"]), flush=True,
            )
            return out

        n_eps_seen = buf[-1][3] + 1
        n_holdout_eps = int(n_eps_seen * float(cfg.holdout_episode_frac))
        # A holdout is only meaningful if something is left to train on. Below 2 episodes it is
        # reported as absent rather than silently taking the whole buffer.
        if n_eps_seen < 2 or n_holdout_eps < 1:
            n_holdout_eps = 0
        first_holdout_ep = n_eps_seen - n_holdout_eps
        train = [s for s in buf if s[3] < first_holdout_ep]
        holdout = [s for s in buf if s[3] >= first_holdout_ep]
        out["p0h_n_train"] = len(train)
        out["p0h_n_holdout"] = len(holdout)
        out["p0h_n_holdout_episodes"] = n_holdout_eps

        def _pred(sample) -> torch.Tensor:
            hoa, hh, _t, _e = sample
            _z, pred = enc(hoa, hh)
            return pred

        def _eval(samples) -> Optional[float]:
            if not samples:
                return None
            ema0 = getattr(agent, "_harm_obs_ema", None)
            total = 0.0
            with torch.no_grad():
                for s in samples:
                    loss = agent.compute_harm_accum_loss(s[2], _loss_state(_pred(s), device))
                    total += float(loss.item())
            if ema0 is not None:
                agent._harm_obs_ema = ema0
            return total / float(len(samples))

        def _effective_targets(samples) -> List[float]:
            """The target `compute_harm_accum_loss` is ACTUALLY regressing, per sample.

            One `recover_effective_target` call per sample -- see that function for how and why
            the target is recovered through the agent instead of re-derived here.

            TEMPORAL ORDER IS LOAD-BEARING: the SD-020 target depends on `agent._harm_obs_ema`,
            which `compute_harm_accum_loss` advances on every call. The walk therefore starts
            from the pre-warmup EMA and runs the WHOLE buffer in order, exactly as a live run
            would, and restores the tracker afterwards.
            """
            if not samples:
                return []
            ema0 = getattr(agent, "_harm_obs_ema", None)
            if harm_obs_ema_pre is not None:
                agent._harm_obs_ema = harm_obs_ema_pre
            targets: List[float] = []
            for s in samples:
                t = recover_effective_target(agent, s[2], device)
                if t is None:
                    if ema0 is not None:
                        agent._harm_obs_ema = ema0
                    return []
                targets.append(t)
            if ema0 is not None:
                agent._harm_obs_ema = ema0
            return targets

        def _head_vs_constant(samples, train_samples, all_samples) -> Dict[str, Any]:
            """Held-out MSE of the aux head against a CONSTANT-MEAN predictor.

            THIS IS THE READOUT THAT MATTERS, and the loss curve is not a substitute for it.
            `harm_accum_head` ends in a Sigmoid, so it starts near 0.5; if the target sits near
            a small constant the loss falls steeply while the encoder learns only that offset.
            Every other number in this block (falling epoch loss, falling held-out loss, moved
            weights) is satisfied by that vacuous fit. `lift = 1 - mse_head/mse_const` is what
            separates it: > 0 means the head beats the constant, <= 0 means it did not, i.e. the
            stage moved the encoder without teaching it anything discriminative.

            Scored against the EFFECTIVE target (see `_effective_targets`), so the SD-011 EMA
            target and the SD-020 precision-weighted PE target are directly comparable: same
            estimator, same split, same baseline, differing only in what the head was asked to
            predict. Before 2026-09-18 this reported `lift: None` on the SD-020 path, which made
            `p0h_readiness_met` False BY CONSTRUCTION there rather than by measurement.
            """
            blk: Dict[str, Any] = {"basis": None, "lift": None, "head_mse": None,
                                   "const_mse": None, "target_mean": None, "target_std": None}
            if not samples:
                return blk
            blk["basis"] = ("sd020_precision_weighted_pe"
                            if out["p0h_harm_surprise_pe_enabled"]
                            else "sd011_accumulated_harm")
            eff = _effective_targets(all_samples)
            if len(eff) != len(all_samples):
                blk["basis"] = "unavailable_zero_aux_loss_weight"
                return blk
            n_train = len(train_samples)
            eff_train = eff[:n_train]
            eff_hold = eff[n_train:]
            if len(eff_hold) != len(samples):
                blk["basis"] = "unavailable_split_mismatch"
                return blk
            tgt = torch.tensor(eff_hold, dtype=torch.float32)
            blk["target_mean"] = float(tgt.mean().item())
            blk["target_std"] = float(tgt.std(unbiased=False).item())
            with torch.no_grad():
                pred = torch.tensor([float(_pred(s).reshape(-1)[0].item()) for s in samples])
            # The constant is fitted on the TRAIN split, not on the holdout: a mean fitted on
            # the holdout itself would be a baseline with information the head never had.
            const = (float(torch.tensor(eff_train).mean().item()) if eff_train
                     else blk["target_mean"])
            blk["const_predictor"] = const
            mse_head = float(((pred - tgt) ** 2).mean().item())
            mse_const = float(((tgt - const) ** 2).mean().item())
            blk["head_mse"] = mse_head
            blk["const_mse"] = mse_const
            blk["lift"] = (1.0 - mse_head / mse_const) if mse_const > 0.0 else None
            blk["train_target_mean"] = (float(torch.tensor(eff_train).mean().item())
                                        if eff_train else None)
            blk["train_target_std"] = (
                float(torch.tensor(eff_train).std(unbiased=False).item())
                if len(eff_train) > 1 else None
            )
            return blk

        out["p0h_holdout_loss_pre"] = _eval(holdout)

        opt = torch.optim.Adam(params, lr=float(cfg.lr))
        n_steps = 0
        epoch_losses: List[float] = []
        for _epoch in range(int(cfg.epochs)):
            # Restore the SD-020 tracker at each epoch boundary so every epoch replays the
            # SAME temporal sequence. Without this the target under harm_surprise_pe_enabled
            # would depend on how many epochs had already run, which is not a property of the
            # data.
            if harm_obs_ema_pre is not None:
                agent._harm_obs_ema = harm_obs_ema_pre
            ep_total = 0.0
            for s in train:
                loss = agent.compute_harm_accum_loss(s[2], _loss_state(_pred(s), device))
                if not loss.requires_grad:
                    continue
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, float(cfg.max_grad_norm))
                opt.step()
                n_steps += 1
                ep_total += float(loss.item())
            if train:
                epoch_losses.append(ep_total / float(len(train)))

        out["p0h_n_steps"] = n_steps
        out["p0h_epoch_mean_losses"] = epoch_losses
        out["p0h_first_epoch_mean_loss"] = epoch_losses[0] if epoch_losses else None
        out["p0h_final_epoch_mean_loss"] = epoch_losses[-1] if epoch_losses else None
        out["p0h_holdout_loss_post"] = _eval(holdout)
        out["p0h_holdout_vs_constant"] = _head_vs_constant(holdout, train, buf)

    if harm_obs_ema_pre is not None:
        agent._harm_obs_ema = harm_obs_ema_pre

    n_changed, max_abs = encoder_weight_delta(before, encoder_weight_snapshot(agent))
    out["p0h_encoder_tensors_changed"] = n_changed
    out["p0h_encoder_max_abs_delta"] = max_abs
    pre, post = out.get("p0h_holdout_loss_pre"), out.get("p0h_holdout_loss_post")
    # The generalisation readout, recorded because the weight delta can be satisfied VACUOUSLY:
    # an encoder can move a long way and fit only the training order. Positive = the held-out
    # episodes got better. None = no holdout was cut, which is NOT the same as zero.
    out["p0h_holdout_loss_drop"] = (
        (pre - post) if (pre is not None and post is not None) else None
    )
    # THE READINESS VERDICT. Both halves are required and they fail for different reasons:
    # a zero weight delta means the gradient path never reached the encoder (the defect this
    # module fixes); a non-positive holdout lift means it did reach it and taught it a
    # constant. Only the pair licenses reading `z_harm_a` as carrying trained signal.
    lift_blk = out.get("p0h_holdout_vs_constant") or {}
    lift = lift_blk.get("lift")
    out["p0h_readiness_met"] = bool(n_changed > 0 and lift is not None and lift > 0.0)
    out["p0h_readiness_basis"] = (
        "encoder_tensors_changed>0 AND holdout_lift_over_constant_mean>0"
    )
    if n_steps > 0 and n_changed == 0:
        print(
            "  [P0h-WARN] %s seed=%d: %d optimizer steps moved NO encoder tensor -- the "
            "gradient path is not reaching the encoder"
            % (label or "zharm_a_p0", int(seed), n_steps),
            flush=True,
        )
    elif lift is not None and lift <= 0.0:
        print(
            "  [P0h-WARN] %s seed=%d: encoder MOVED but held-out lift over a constant-mean "
            "predictor is %.4f (<= 0) -- target mean %.6f std %.6f. The stage trained the "
            "encoder to a CONSTANT, not to the signal; do not read z_harm_a as trained."
            % (label or "zharm_a_p0", int(seed), lift,
               lift_blk.get("target_mean") or 0.0, lift_blk.get("target_std") or 0.0),
            flush=True,
        )
    return out
