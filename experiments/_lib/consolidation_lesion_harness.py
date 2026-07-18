"""
consolidation_lesion_harness.py -- SD-068 experiment-layer instrumentation for the
MECH-120 (SWS denoising) -> MECH-121 (NREM slot-filling) -> MECH-123 (REM precision
recalibration) offline-consolidation pipeline.

WHAT THIS IS FOR
----------------
The live sleep cycle (ree_core/sleep/phase_manager.py SleepLoopManager._run_cycle)
exposes only OPERATIONAL telemetry -- routing counts, replay diversity, update
counts, param-delta. It has NO per-phase OUTPUT-QUALITY readout and NO per-phase
diffuse-damage knob. Without those two, the distinctive MECH-168/INV-047/MECH-169
falsifier cannot be run:

    Under UNIFORM/diffuse damage across the three phases, does functional failure
    emerge STAGED in reverse dependency order (REM >> NREM >> SWS) rather than
    uniform?

This harness supplies both missing pieces at the EXPERIMENT layer (no ree_core
change), following the V3-EXQ-702 injected-content precedent:

  1. INJECT known ground-truth content directly onto each phase's operative
     substrate (context_memory slots, a controlled replay-content target, the E3
     precision reference), rather than depending on the agent to behaviourally
     ENCODE diverse content. The latter STARVES under monostrategy collapse -- the
     confirmed failure_autopsy_V3-EXQ-538a ceiling ("sleep cannot consolidate an
     unencoded representation"). Injected content sidesteps the ceiling: fidelity
     is scored against a KNOWN injection, not against emergent behaviour.

  2. A single UNIFORM diffuse-damage scalar `sigma` applied identically to each
     phase's operative tensor == the MECH-168/169 "diffuse/uniform damage" model.
     (Only the CONSOLIDATION-side functional consequence of uniform damage is
     modelled; the glymphatic / amyloid STRUCTURAL half has no V3 analog and is
     explicitly OUT OF SCOPE.)

THE NON-VACUITY CONTRACT (the whole point)
------------------------------------------
Staged decline under uniform damage is VACUOUS if it is merely feed-forward error
compounding baked into the DAG topology (downstream accumulates upstream error, so
of course it "looks worse" -- that is arithmetic, not a discovered property of the
pipeline). The genuinely falsifiable content is ERROR-PROPAGATION SENSITIVITY:
does a phase AMPLIFY or ATTENUATE upstream corruption?

So the load-bearing readout is a dose-response GAIN, not "downstream looks worse":

  * For each phase, hold its OWN damage at 0 and sweep the UPSTREAM corruption
    magnitude delta; the gain = d(output-error)/d(delta).
      gain < 1  -> ATTENUATING: the phase has genuine corrective capacity. The
                   MECH-168 staging story becomes non-trivial ("the correction
                   needs intact upstream"), NOT topology.
      gain ~ 1  -> PASSTHROUGH: vacuous topology; the claim's staging is NOT
                   supported as a discovered property.
      gain > 1  -> AMPLIFYING: the MECH-094 psychosis polarity.

  * REM specifically is measured TWO ways, because the built substrate splits it:
      (a) the bare linear precision nudge (recalibrate_precision_to, MECH-204) is
          PASSTHROUGH by construction (linear interpolation toward the target);
      (b) the generative re-derivation pass (the hippocampal replay rollout that
          run_rem_attribution_pass performs, Hobson & Friston 2012 "unconstrained
          generative simulation") is where any corrective/amplifying capacity
          could live.
    The falsifier lives in whether (b)'s gain is below (a)'s. If (b) ~ (a) ~
    passthrough, the honest diagnostic outcome is "the V3 REM substrate has no
    corrective capacity, so its staged-first-failure is pure topology" -- a valid,
    informative result, not a harness failure.

    REM (b) READOUT -- ROLLOUT-SEED INJECTION (replaces the retired PROXY).
    ---------------------------------------------------------------------
    The generative pass in run_rem_attribution_pass reads `theta_buffer.recent`
    and re-derives content via `hippocampal.replay(recent) -> e2.rollout_with_world`.
    The RETIRED proxy corrupted the E3 precision REFERENCE and read
    `rem_terrain_variance` back -- but the generative pass never consumes the
    precision reference, so that read was ~null (confirmed in V3-EXQ-778). The
    generative pass consumes the ROLLOUT SEED (the recent z_world), so the clean
    readout (rem_generative_fidelity) injects KNOWN content onto that seed:
      1. capture the seed (theta_buffer.recent[-1]) as the known clean target;
      2. re-derive the clean rollout with a FIXED action sequence -- the same
         e2.rollout_with_world call replay makes, but with actions held constant
         so the seed is the ONLY varying input (replay's per-call random actions
         are exactly why the proxy variance read null: action noise swamped the
         seed effect);
      3. corrupt the seed by `sigma`, re-derive with the SAME fixed actions;
      4. measure the recovered-vs-known-target deviation over the GENERATED
         states, relative to the clean rollout.
    The load-bearing GAIN is then dimensionless and load-bearing rather than null:
      gain = (output relative deviation) / (input relative seed corruption)
        gain < 1 -> ATTENUATING (genuine generative correction; non-vacuous
                    staging -- "the correction needs an intact seed"),
        gain ~ 1 -> PASSTHROUGH (staging is topology),
        gain > 1 -> AMPLIFYING (MECH-094 psychosis polarity).
    The corrupted seed is ALSO pushed into theta_buffer and the real
    run_rem_attribution_pass driven once for liveness telemetry (rem_n_rollouts),
    confirming the injection point is exactly the seed the live REM path reads.

THE NULL-CONTENT CONTROL + CONFOUND REGISTER (V3-EXQ-778b)
----------------------------------------------------------
`run_null_content_control` runs the identical sigma sweep with NO injected known
content (`content_scale=0.0`) while holding the delivered perturbation numerically
identical, and reports a per-phase `null_slope_ratio` = |null slope| / |injected
slope|. This is the analog of the odour-contingency null in Bar et al. 2020 (Curr
Biol, DOI 10.1016/j.cub.2020.01.091), the methodological precedent this harness
follows: their unilateral olfactory stimulation produced NO effect when learning had
occurred without the contextual odour -- the perturbation acts only on injected
content. Without such a null, a readout that moves with sigma on noise alone is
measuring PERTURBATION MAGNITUDE rather than content fidelity, and the
damage-tolerance staging order is an ordering of the three phases' raw noise
sensitivity -- exactly the vacuity this harness claims to escape.

A phase whose ratio exceeds NULL_SLOPE_RATIO_CEILING is CONFOUNDED. Confounded
phases are NAMED in `confounded_phases` and reported alongside the staging order;
they are NEVER silently dropped from it. Standing a-priori expectations, to be
confirmed or refuted by the V3-EXQ-778b measurement rather than assumed:

  * rem (passthrough leg) -- EXPECTED CONFOUNDED. `rem_precision_error` at step=1.0
    returns exactly 1/corrupt_target, so `calibration_error` is a closed-form
    function of the injected corruption with no content term. This leg is already
    documented above as "PASSTHROUGH by construction"; the null control makes that
    structural admission MEASURABLE instead of merely stated.
  * sws -- EXPECTED CONFOUNDED. `_shy` is affine, so shy(clean+noise) - shy(clean) =
    shy_centred(noise) exactly, independent of `clean`. The residual-noise fraction
    is therefore ~sigma^2 whatever the content is.
  * nrem -- AT RISK. The injected "trace" is an isotropic randn offset, which is
    statistically indistinguishable from the corruption it is damaged with, so the
    consolidation pass may close the same fraction of a noise target as of a content
    target.
  * rem (generative leg) -- THE ONE WITH A GENUINE CHANCE. `rollout_with_world` is
    non-linear, so its re-derivation of an in-distribution seed can differ from its
    response to a content-free seed.

If the measurement confirms the sws/rem-passthrough expectations, the V3-EXQ-778
staged order (nrem, rem, sws) rests on legs that are at least partly noise-sensitivity
ordering, and the SD-068 non-vacuity contract must be carried by the REM
passthrough-vs-generative contrast alone. That is a valid and informative diagnostic
outcome -- it scopes the claim honestly rather than withdrawing it.

PREREQUISITE CAVEATS (respected, not lifted)
--------------------------------------------
  * MECH-121 is candidate/substrate_conditional (hold_pending_v3_substrate). This
    harness is REPRESENTATION-LEVEL plumbing instrumentation on injected content,
    NOT MECH-121 behavioural validation. Any run built on it MUST be
    EXPERIMENT_PURPOSE="diagnostic" and MUST NOT tag MECH-121 as promotion
    evidence. The NREM leg is a substrate-plumbing-fidelity readout only.
  * The 120/121/123 <-> substrate mapping is not a clean 1:1 feed-forward chain in
    code (SWS = schema write + SHY; the "NREM slot-filling" content is realised by
    the replay/offline-gradient path; REM = attribution replay + precision nudge).
    The dependency ORDER is nonetheless real (REM requires SWS slots to exist;
    docstrings enforce call ordering), which is what the staging falsifier needs.

MECH-094: this harness runs WEIGHT/state operations offline; it produces no
hypothesis-tagged residue/anchor/memory writes beyond what the phase ops already
do. It does not simulate-then-commit. No new MECH-094 surface.

Design + validity model: REE_assembly/docs/architecture/sd_068_consolidation_lesion_harness.md
Injected-content precedent: ree-v3/experiments/v3_exq_702_gap3b_sleep_cluster_promotion.py
Encoding-starvation ceiling: REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-538a_2026-07-10.md
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch

from ree_core.agent import REEAgent
from ree_core.utils.config import REEConfig

# Phase id -> owning claim. Order IS the forward dependency order.
PHASE_CLAIMS: Dict[str, str] = {
    "sws": "MECH-120",   # SWS denoising / attractor flattening (most upstream)
    "nrem": "MECH-121",  # NREM slot-filling / episodic->schematic transfer (middle)
    "rem": "MECH-123",   # REM precision recalibration (most downstream)
}
# Reverse dependency order = predicted staged-failure order (downstream fails first).
REVERSE_DEPENDENCY_ORDER: Tuple[str, str, str] = ("rem", "nrem", "sws")

# Default dims -- match the V3-EXQ-702 promotion-run cell so the substrate builds
# the same offline-consolidation pathway.
BODY_OBS_DIM = 12
WORLD_OBS_DIM = 250
ACTION_DIM = 4
HARM_OBS_DIM = 51

# A -1.0 sentinel means "readout unavailable on this substrate build" (kept out of
# aggregates by the driver rather than silently scored as a real value).
UNAVAILABLE = -1.0


def _gen(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g


def build_pipeline_agent(
    *,
    seed: int,
    body_obs_dim: int = BODY_OBS_DIM,
    world_obs_dim: int = WORLD_OBS_DIM,
    action_dim: int = ACTION_DIM,
    harm_obs_dim: int = HARM_OBS_DIM,
    shy_decay_rate: float = 0.85,
) -> REEAgent:
    """Build an agent with all three consolidation phases live.

    SWS + REM passes enabled (SD-017), SHY denoising enabled (MECH-120), and the
    unified sleep-aggregation cluster on so the NREM offline-consolidation pathway
    (MECH-121/273) is instantiated. This is the substrate the harness damages and
    reads; the ARM_OFF contrast in a driver rebuilds it with the phases disabled.
    """
    torch.manual_seed(int(seed))
    cfg = REEConfig.from_dims(
        body_obs_dim=body_obs_dim,
        world_obs_dim=world_obs_dim,
        action_dim=action_dim,
        harm_obs_dim=harm_obs_dim,
        shy_enabled=True,
        shy_decay_rate=shy_decay_rate,
        sws_enabled=True,
        rem_enabled=True,
        use_harm_stream=True,
        use_e2_harm_s_forward=True,
        use_sleep_aggregation_cluster=True,
        sleep_loop_episodes_K=1,
    )
    return REEAgent(cfg)


def _warm_encoders(agent: REEAgent, *, n_steps: int, gen: torch.Generator) -> None:
    """Brief waking drive so the encoders + world-experience buffer are non-trivial.

    Mirrors V3-EXQ-702 _drive_waking. This is NOT the content under test (that is
    injected); it only makes the phase operations runnable (buffers non-empty).
    """
    for _ in range(n_steps):
        obs_body = torch.randn(BODY_OBS_DIM, generator=gen)
        obs_world = torch.randn(WORLD_OBS_DIM, generator=gen)
        agent.act_with_split_obs(obs_body=obs_body, obs_world=obs_world)


# --------------------------------------------------------------------------- #
# Uniform diffuse-damage primitive                                            #
# --------------------------------------------------------------------------- #

def diffuse_perturb(
    t: torch.Tensor,
    sigma: float,
    gen: torch.Generator,
    rms_ref: Optional[float] = None,
) -> torch.Tensor:
    """Additive isotropic Gaussian corruption of magnitude `sigma`.

    The single knob applied IDENTICALLY to each phase's operative tensor == the
    MECH-168/169 "uniform/diffuse damage" model. sigma is in units of the tensor's
    own per-element scale (multiplied by its RMS) so a given sigma is comparably
    severe across phases with different natural magnitudes -- this is what makes
    the damage genuinely UNIFORM across phases rather than uniform-in-raw-units.

    `rms_ref` (SD-068 null-content control, V3-EXQ-778b) overrides the scale the
    noise is measured against. The NULL arm scales the injected content to zero, so
    the content's OWN rms goes to zero with it -- which would silently scale the
    damage away too and make the null trivially flat for the wrong reason. Passing
    the UNSCALED content's rms holds the delivered perturbation numerically
    IDENTICAL across the injected and null arms, which is the whole point of the
    control (Bar et al. 2020: the same odour is delivered; only the prior pairing
    is absent). Default None == the tensor's own rms == pre-778b behaviour.
    """
    if sigma <= 0.0:
        return t.clone()
    if rms_ref is None:
        rms = float(t.detach().pow(2).mean().clamp(min=1e-12).sqrt().item())
    else:
        rms = float(rms_ref)
    noise = torch.randn(t.shape, generator=gen) * (sigma * rms)
    return t + noise


def _rms(t: torch.Tensor) -> float:
    """RMS of a tensor, floored -- the damage-scale reference for both arms."""
    return float(t.detach().pow(2).mean().clamp(min=1e-12).sqrt().item())


# --------------------------------------------------------------------------- #
# Phase 1 -- MECH-120 SWS: denoising signal-to-noise                          #
# --------------------------------------------------------------------------- #

def sws_denoising_snr(
    agent: REEAgent,
    *,
    sigma: float,
    gen: torch.Generator,
    n_prototypes: Optional[int] = None,
    content_scale: float = 1.0,
) -> Dict[str, float]:
    """Denoising-SNR of the SWS phase against injected clean prototypes.

    Injected content: a set of distinct clean prototype rows written into
    e1.context_memory (the attractor slots SHY operates on). SWS's job (MECH-120)
    is to flatten dominant attractors and restore SNR while preserving the
    RELATIONAL structure of the traces (the deviation-from-mean that SHY keeps at
    rate `decay`).

    Damage: the operative tensor (context_memory slots) is corrupted by `sigma`
    BEFORE the SHY operation -- the phase's own machinery acting on a diffusely
    damaged store.

    Readout (all fully-verified APIs):
        signal_power = ||decay * (clean - mean(clean))||^2
        noise_power  = ||shy(damaged) - shy(clean)||^2   (residual due to damage)
        denoising_snr_db = 10 log10(signal_power / noise_power)
    Higher = better functional integrity; monotonically decreasing in sigma.

    `content_scale` (SD-068 null-content control, V3-EXQ-778b): scales the INJECTED
    prototypes. 1.0 == the injected-content arm (pre-778b behaviour, bit-identical).
    0.0 == the NULL arm: no known content is planted, so the SHY operation runs on
    noise alone while the delivered damage stays numerically identical (the noise
    scale is taken from the UNSCALED prototypes). signal_power goes to 0 in the null
    arm, so the null arm's error series must be normalised by the INJECTED arm's
    signal_power -- run_null_content_control does exactly that.
    """
    mem = agent.e1.context_memory.memory
    num_slots = int(mem.shape[0])
    dim = int(mem.shape[1])
    k = num_slots if n_prototypes is None else min(int(n_prototypes), num_slots)

    # Inject distinct clean prototypes (structured, well-separated).
    base = torch.randn(num_slots, dim, generator=gen)
    if k > 1:
        # amplify separation between the first k rows so structure is non-trivial
        base[:k] = base[:k] * 2.0
    # Damage scale is always referenced to the UNSCALED content, so the null arm
    # (content_scale=0) receives the SAME absolute perturbation as the injected arm.
    rms_ref = _rms(base)
    clean = base * float(content_scale)
    decay = float(agent.config.shy_decay_rate)

    def _shy(rows: torch.Tensor) -> torch.Tensor:
        mean = rows.mean(dim=0, keepdim=True)
        return mean + (rows - mean) * decay

    clean_out = _shy(clean)
    damaged = diffuse_perturb(clean, sigma, gen, rms_ref=rms_ref)
    damaged_out = _shy(damaged)

    signal_power = float(((clean_out - clean_out.mean(dim=0, keepdim=True)) ** 2).sum().item())
    noise_power = float(((damaged_out - clean_out) ** 2).sum().item())
    if noise_power <= 1e-12:
        snr_db = 120.0  # effectively noiseless (sigma == 0)
    else:
        snr_db = 10.0 * math.log10(max(signal_power, 1e-12) / noise_power)

    # Also drive the REAL substrate op once for liveness telemetry (slot diversity).
    with torch.no_grad():
        agent.e1.context_memory.memory.data = damaged.clone()
    agent.enter_sws_mode()  # runs the live e1.shy_normalise(decay)
    live = agent.run_sws_schema_pass(anchor_weight=1.0)
    agent.exit_sleep_mode()

    return {
        "phase": 0.0,  # sws index
        "denoising_snr_db": float(snr_db),
        "signal_power": signal_power,
        "noise_power": noise_power,
        "content_scale": float(content_scale),
        "sws_slot_diversity": float(live.get("sws_slot_diversity", 0.0)),
        "sws_n_writes": float(live.get("sws_n_writes", 0.0)),
    }


# --------------------------------------------------------------------------- #
# Phase 2 -- MECH-121 NREM: episodic->schematic transfer-fidelity             #
# --------------------------------------------------------------------------- #

def nrem_transfer_fidelity(
    agent: REEAgent,
    *,
    sigma: float,
    gen: torch.Generator,
    n_steps: int = 20,
    lr: float = 1e-3,
    content_scale: float = 1.0,
) -> Dict[str, float]:
    """Transfer-fidelity of the NREM offline-consolidation pass on injected content.

    The MECH-121/273 offline pass is a bounded low-LR gradient step that moves the
    consolidation modules (E1 schematic + E2 forward) toward replayed content. Here
    the replayed content is INJECTED as a known per-parameter target, and fidelity
    is the fraction of the injected-content gap the interleaved consolidation pass
    closes (uses the verified CrossModuleConsolidator API; no E2-forward-signature
    assumptions).

    Damage: the injected content (the "trace being consolidated") is corrupted by
    `sigma`, so the pass consolidates toward a diffusely damaged target. Fidelity is
    scored against the CLEAN target -- damage lowers the achievable gap-closure.

    NOTE (caveat, documented in SD-068): this is a PARAMETER-SPACE proxy for the
    episodic->schematic transfer, and a substrate-plumbing-fidelity readout only --
    it does NOT constitute behavioural validation of MECH-121 (which stays
    hold_pending_v3_substrate).

    `content_scale` (SD-068 null-content control, V3-EXQ-778b): scales the injected
    per-parameter offset (the "trace being consolidated"). 1.0 == the injected-content
    arm (pre-778b behaviour, bit-identical). 0.0 == the NULL arm: the clean target
    collapses onto the CURRENT parameters, so there is no known content to transfer
    and the consolidation pass runs on pure noise -- while the delivered damage stays
    numerically identical (noise scale referenced to the UNSCALED target). gap_before
    goes to 0 in the null arm (so `transfer_fidelity` is UNAVAILABLE there by
    construction); the null arm's error series is instead built from `gap_after`
    normalised by the INJECTED arm's gap_before, which run_null_content_control does.
    """
    from ree_core.sleep.cross_module_consolidation import CrossModuleConsolidator

    e1 = getattr(agent, "e1", None)
    e2 = getattr(agent, "e2", None)
    if e1 is None or e2 is None:
        return {"phase": 1.0, "transfer_fidelity": UNAVAILABLE, "available": 0.0}

    modules: Dict[str, List[torch.nn.Parameter]] = {
        "e1": [p for p in e1.parameters() if p.requires_grad],
        "e2": [p for p in e2.parameters() if p.requires_grad],
    }
    modules = {k: v for k, v in modules.items() if v}
    if not modules:
        return {"phase": 1.0, "transfer_fidelity": UNAVAILABLE, "available": 0.0}

    # Injected known content: a fixed target offset per parameter (the trace to
    # consolidate). Clean target = current params + structured offset.
    clean_targets: Dict[str, List[torch.Tensor]] = {}
    damaged_targets: Dict[str, List[torch.Tensor]] = {}
    for name, params in modules.items():
        cts, dts = [], []
        for p in params:
            offset = torch.randn(p.shape, generator=gen) * 0.1
            # Damage scale referenced to the UNSCALED target so the null arm
            # (content_scale=0) receives the SAME absolute perturbation.
            base_ct = (p.detach() + offset).clone()
            rms_ref = _rms(base_ct)
            ct = (p.detach() + offset * float(content_scale)).clone()
            cts.append(ct)
            dts.append(diffuse_perturb(ct, sigma, gen, rms_ref=rms_ref).detach())
        clean_targets[name] = cts
        damaged_targets[name] = dts

    def _gap(name: str) -> float:
        tot = 0.0
        for p, ct in zip(modules[name], clean_targets[name]):
            tot += float(((p.detach() - ct) ** 2).sum().item())
        return tot

    def _make_loss(name: str) -> Callable[[], torch.Tensor]:
        def _loss() -> torch.Tensor:
            terms = [((p - dt) ** 2).mean() for p, dt in zip(modules[name], damaged_targets[name])]
            return torch.stack(terms).sum() if terms else torch.zeros(())
        return _loss

    gap_before = {name: _gap(name) for name in modules}
    total_before = sum(gap_before.values())

    cons = CrossModuleConsolidator()
    metrics = cons.consolidate(
        module_losses={name: _make_loss(name) for name in modules},
        module_params={name: modules[name] for name in modules},
        n_steps=int(n_steps),
        schedule="interleaved",
        lr=float(lr),
        simulation_mode=False,
    )

    total_after = sum(_gap(name) for name in modules)
    if total_before <= 1e-12:
        fidelity = UNAVAILABLE
    else:
        fidelity = 1.0 - (total_after / total_before)

    return {
        "phase": 1.0,
        "transfer_fidelity": float(fidelity),
        "available": 1.0,
        "content_scale": float(content_scale),
        "gap_before": float(total_before),
        "gap_after": float(total_after),
        "n_updates": float(metrics.get("n_updates", 0.0)),
        "cross_module_replay_share": float(metrics.get("cross_module_replay_share", 0.0)),
    }


# --------------------------------------------------------------------------- #
# Phase 3 -- MECH-123 REM: precision-calibration-error (+ generative gain)    #
# --------------------------------------------------------------------------- #

def rem_precision_error(
    agent: REEAgent,
    *,
    sigma: float,
    gen: torch.Generator,
    target_precision: float = 2.0,
    start_variance: float = 1.0,
    step: float = 1.0,
    run_generative: bool = True,
    content_scale: float = 1.0,
    null_mode: str = "zero_content",
    unpaired_target_precision: Optional[float] = None,
) -> Dict[str, float]:
    """Precision-calibration-error of the REM phase against an injected clean target.

    Injected content: a known target precision `target_precision`. The bare
    recalibration (recalibrate_precision_to, MECH-204) is a linear interpolation
    toward 1/target -- PASSTHROUGH by construction.

    `step` defaults to 1.0 (FULL adoption) so the readout cleanly isolates the
    corruption-passthrough: with step=1.0 the output is exactly 1/corrupt_target,
    so calibration_error is a monotone function of the injected corruption and is
    NOT swamped by the un-reachable-in-one-step residual that a partial step
    (the substrate default 0.1) leaves. This is a measurement choice for the
    passthrough baseline, distinct from the substrate's partial-step default.

    DO NOT SWEEP `step` TO ESCAPE THE NULL-ARM CLAMP -- it is provably inert
    (established 2026-07-18 while auditing the V3-EXQ-778c GOV-FANOUT-1 portfolio,
    BEFORE queuing; the leg was redesigned rather than run). The substrate computes
    rv_after = (1 - step) * rv_before + step * (1 / (corrupt_target + 1e-6))
    (ree_core/predictors/e3_selector.py recalibrate_precision_to). Two consequences:
      1. the `max(1e-3, raw_target)` positivity clamp is applied to corrupt_target
         UPSTREAM of `step`, so `target_clamped` and the null arm's clamp fraction are
         EXACTLY invariant to step; and
      2. both arms' sigma-slopes carry the same linear factor `step`, so it CANCELS in
         null_slope_ratio = |null slope| / |injected slope|.
    Verified numerically over step in {0.1, 0.25, 0.5, 1.0}: the ratio is identical to
    12 significant figures (14569.1833719598-...602, float round-off only) and the clamp
    fraction / distinct-value count are bit-identical. A step ladder therefore CANNOT
    de-rail the null arm and would return the declared null of H-rem-clamp-artifact by
    construction -- a false elimination, not a measurement. The clamp's actual root is
    that MULTIPLICATIVE content scaling sends the null arm's target precision to ZERO,
    so a strictly-positive reference clamps on ~half the draws at ANY floor value. The
    measurement-axis knob that does bite is `null_mode` (below), not `step` or the floor.

    Damage: the precision REFERENCE fed to recalibration is corrupted by `sigma`
    (the corruption channel). Error is scored against the CLEAN target:
        calibration_error = |running_variance_after - 1/target_precision_clean|

    When run_generative=True, also drive the CLEAN generative re-derivation
    readout (rem_generative_fidelity) -- rollout-seed injection: known content is
    injected onto the hippocampal rollout seed (theta_buffer.recent), corrupted by
    `sigma`, re-derived with a FIXED action sequence, and the recovered-vs-known
    deviation is measured. This yields a load-bearing error-propagation GAIN
    (output relative deviation / input relative seed corruption): <1 attenuating,
    ~1 passthrough, >1 amplifying. This REPLACES the retired rem_terrain_variance
    proxy, which read ~null because the generative pass never consumes the E3
    precision reference the proxy corrupted (confirmed in V3-EXQ-778).

    `content_scale` (SD-068 null-content control, V3-EXQ-778b): scales the INJECTED
    known target precision that the recalibration is pointed at. 1.0 == the
    injected-content arm (pre-778b behaviour, bit-identical). 0.0 == the NULL arm:
    the recalibration reference becomes pure jitter with no planted target, while
    the jitter magnitude stays numerically identical (it is always referenced to the
    unscaled `target_precision`). NOTE: this leg is PASSTHROUGH BY CONSTRUCTION
    (step=1.0 makes the output exactly 1/corrupt_target), so it is a priori expected
    to fail the null control -- see the CONFOUND REGISTER in the module docstring.

    `null_mode` (SD-068 GOV-FANOUT-1 leg H-rem-clamp-artifact, V3-EXQ-778d) selects HOW
    the null arm is OPERATIONALISED. This is the measurement-axis knob that actually
    bites (see the `step` warning above for the one that provably does not):

      "zero_content" (DEFAULT, bit-identical to 778b/778c) -- the null target is
        clean_target * content_scale, i.e. ZERO when content_scale is 0. A precision of
        zero is a DEGENERATE POINT of this parameterisation: the corrupt reference is
        then pure jitter centred on zero, so the positivity clamp fires on ~half the
        draws and the readout pins at the saturation constant 998.5009992509989. The
        resulting null slope is identically zero -- degenerate, NOT content-contingent.

      "unpaired_target" -- the null target is an INDEPENDENT positive draw of the same
        magnitude class (`unpaired_target_precision`), UNPAIRED with the clean target
        the error is scored against. This is the faithful analog of the Bar et al. 2020
        odour control that SD-068 follows: "same odour, NO PRIOR PAIRING", not "no
        odour". The reference stays in range so nothing clamps, the delivered
        perturbation is unchanged (jitter is always referenced to the UNSCALED
        clean_target), and the content PAIRING -- the thing whose contribution is under
        test -- is what is removed. A zero-content null removes the pairing AND pushes
        the parameterisation off a cliff; this one removes only the pairing.

    Under "unpaired_target", `content_scale` is not applied to the target (the unpaired
    draw replaces it); it is still reported so the arm is identifiable in the manifest.
    """
    if null_mode not in ("zero_content", "unpaired_target"):
        raise ValueError(
            f"null_mode must be 'zero_content' or 'unpaired_target', got {null_mode!r}"
        )
    if null_mode == "unpaired_target" and unpaired_target_precision is None:
        raise ValueError(
            "null_mode='unpaired_target' requires unpaired_target_precision"
        )
    e3 = getattr(agent, "e3", None)
    if e3 is None:
        return {"phase": 2.0, "calibration_error": UNAVAILABLE, "available": 0.0}

    clean_target = float(target_precision)
    if null_mode == "unpaired_target":
        # The reference points at an INDEPENDENT positive target of the same magnitude
        # class; the error is still scored against clean_target, so the PAIRING is what
        # has been removed. content_scale does not scale the target in this mode.
        injected_target = float(unpaired_target_precision)
    else:
        injected_target = clean_target * float(content_scale)
    # Corrupt the reference precision by sigma (relative), floored positive. The
    # jitter is always scaled by the UNSCALED clean_target so the null arm receives
    # the SAME absolute perturbation as the injected arm.
    if sigma > 0.0:
        jitter = float(torch.randn(1, generator=gen).item()) * sigma * clean_target
        raw_target = injected_target + jitter
    else:
        raw_target = (
            injected_target
            if (null_mode == "unpaired_target" or content_scale != 1.0)
            else clean_target
        )
    corrupt_target = max(1e-3, raw_target)
    # Clamp saturation flag (V3-EXQ-778b): with no injected target the reference can
    # collapse onto the 1e-3 positivity floor, and 1/1e-3 = 1000 then DOMINATES the
    # calibration error. A null-arm slope built on clamped points is off-scale, not a
    # calibrated sensitivity -- so the ratio must be read as "structurally content-free",
    # never as a literal N-fold noise sensitivity. Surfaced so a large ratio cannot be
    # mistaken for a measured magnitude.
    target_clamped = 1.0 if raw_target < 1e-3 else 0.0

    # Inject the known starting variance, then recalibrate toward the corrupt ref.
    e3._running_variance = float(start_variance)
    rv_before, rv_after = e3.recalibrate_precision_to(corrupt_target, step=float(step))
    clean_var = 1.0 / (clean_target + 1e-6)
    calibration_error = abs(float(rv_after) - clean_var)

    # DE-CLAMPED companion readout (SD-068 GOV-FANOUT-1 leg H-rem-genuinely-content-free,
    # V3-EXQ-778e). ADDITIVE: emitted alongside `calibration_error`, never replacing it,
    # so every prior run's scored series is bit-identical.
    #
    # WHY a second readout at all. `calibration_error` is scored in VARIANCE units as
    # |rv_after - 1/clean_target|. At step=1.0 the substrate returns
    # rv_after = 1/(corrupt_target + 1e-6), so when the null arm's reference collapses
    # onto the 1e-3 positivity floor the readout jumps to 1/(1e-3 + 1e-6) = 999.001 and
    # the error pins at the constant 998.5009992509989 -- an identically ZERO sigma-slope
    # that is degenerate, not content-contingent (V3-EXQ-778c: exactly that constant on
    # 5/8 seeds). On the seeds that escape the floor, 1/jitter instead blows the ratio
    # off-scale (1801-9143 on 3/8). Both rails are artifacts of inverting a near-zero
    # precision into variance units.
    #
    # Scoring in PRECISION units against the injected target directly UNDOES that
    # inversion: achieved_precision = 1/rv_after is ~ corrupt_target itself, so the
    # readout is bounded and dimensionless and its sigma-response is graded on both
    # arms. `rem_reference_clamped` (== target_clamped) is carried alongside so a
    # consumer can gate on residual clamp incidence rather than assume it away.
    achieved_precision = 1.0 / max(float(rv_after), 1e-12)
    direct_precision_error = abs(achieved_precision - clean_target) / max(clean_target, 1e-12)

    out: Dict[str, float] = {
        "phase": 2.0,
        "calibration_error": float(calibration_error),
        "available": 1.0,
        "content_scale": float(content_scale),
        "running_variance_before": float(rv_before),
        "running_variance_after": float(rv_after),
        "corrupt_target_precision": float(corrupt_target),
        "clean_target_variance": float(clean_var),
        "target_clamped": float(target_clamped),
        "rem_step": float(step),
        "null_mode_unpaired": 1.0 if null_mode == "unpaired_target" else 0.0,
        "unpaired_target_precision": float(
            unpaired_target_precision if unpaired_target_precision is not None else 0.0
        ),
        "injected_target_precision": float(injected_target),
        "achieved_precision": float(achieved_precision),
        "direct_precision_error": float(direct_precision_error),
        "clean_target_precision": float(clean_target),
        "rem_reference_clamped": float(target_clamped),
    }

    if run_generative:
        gen_out = rem_generative_fidelity(
            agent, sigma=sigma, gen=gen, content_scale=content_scale
        )
        out.update(gen_out)

    return out


# --------------------------------------------------------------------------- #
# Phase 3b -- MECH-123 REM: generative re-derivation fidelity (rollout-seed)   #
# --------------------------------------------------------------------------- #

def rem_generative_fidelity(
    agent: REEAgent,
    *,
    sigma: float,
    gen: torch.Generator,
    n_rollouts: int = 4,
    horizon: Optional[int] = None,
    drive_liveness_pass: bool = True,
    content_scale: float = 1.0,
) -> Dict[str, float]:
    """Clean recovered-vs-known-target fidelity of the REM generative re-derivation.

    This is the rollout-seed-injection readout that REPLACES the retired
    rem_terrain_variance proxy. The live REM pass (run_rem_attribution_pass) reads
    `theta_buffer.recent` and re-derives content through
    `hippocampal.replay(recent) -> e2.rollout_with_world(...)`. The proxy corrupted
    the E3 precision reference, which that generative path never consumes -- so the
    sensitivity read ~null. Here the KNOWN content is injected onto the rollout SEED
    itself (the exact input the generative pass consumes):

      1. Capture the seed the live path would use -- theta_buffer.recent[-1] (an
         in-distribution z_world so the forward model behaves realistically; a
         fresh randn seed is used only if the buffer is empty). This captured seed
         IS the known clean target.
      2. Re-derive the CLEAN rollout with the same call replay makes
         (e2.rollout_with_world), but with a FIXED action sequence -- replay draws
         random actions each call, and that action noise (not the seed) is exactly
         what made the proxy variance read null. Holding actions constant across
         the clean and corrupt re-derivations makes the injected seed corruption
         the ONLY varying input. n_rollouts fixed action draws are averaged to
         remove dependence on any single action sequence.
      3. Corrupt the seed by `sigma` (the SAME diffuse_perturb primitive used for
         the other phases) and re-derive with the SAME fixed actions.
      4. Measure the recovered-vs-known-target deviation over the GENERATED states
         (world_states[1:], i.e. excluding the raw seed passthrough at t=0, so the
         gain reflects the forward-model re-derivation, not the trivial t=0 copy),
         relative to the clean rollout.

    Readout (all fully-verified APIs -- e2.rollout_with_world is exactly the call
    hippocampal.replay makes):
        input_rel_corruption = ||corrupt_seed - clean_seed|| / ||clean_seed||
        output_rel_dev       = ||corrupt_gen - clean_gen|| / ||clean_gen||
        rem_generative_gain  = output_rel_dev / input_rel_corruption
            < 1 -> ATTENUATING (genuine generative correction; non-vacuous)
            ~ 1 -> PASSTHROUGH (staging is topology)
            > 1 -> AMPLIFYING (MECH-094 psychosis polarity)
        rem_generative_fidelity = 1 - output_rel_dev  (1.0 = perfect recovery)

    At sigma == 0 the corrupt seed equals the clean seed, so output_rel_dev == 0,
    the gain is UNAVAILABLE (0/0), and fidelity == 1.0 -- the origin anchor for the
    dose-response slope the driver fits.

    Liveness: the corrupted seed is also pushed into theta_buffer and the real
    run_rem_attribution_pass driven once (drive_liveness_pass=True), confirming the
    injection point is exactly the seed the live REM path reads. Its
    rem_terrain_variance is retained as pure telemetry -- NOT a load-bearing signal.

    `content_scale` (SD-068 null-content control, V3-EXQ-778b): scales the captured
    rollout SEED -- the injected known content. 1.0 == the injected-content arm
    (pre-778b behaviour, bit-identical). 0.0 == the NULL arm: the generative pass is
    re-derived from a CONTENT-FREE (zero) seed, so any remaining sigma-sensitivity is
    the forward model's raw noise response rather than content fidelity. The
    delivered seed corruption stays numerically identical (referenced to the UNSCALED
    captured seed's rms). This leg is the one with a genuine chance of passing the
    null control, because the re-derivation is NON-LINEAR -- see the CONFOUND
    REGISTER in the module docstring.
    """
    e2 = getattr(agent, "e2", None)
    hippo = getattr(agent, "hippocampal", None)
    if e2 is None or hippo is None:
        return {"rem_generative_fidelity": UNAVAILABLE, "generative_available": 0.0}

    world_dim = int(e2.config.world_dim)
    self_dim = int(e2.config.self_dim)
    action_dim = int(e2.config.action_dim)
    H = int(horizon if horizon is not None else getattr(hippo.config, "horizon", 30))

    # Capture the known clean seed the live REM path would consume.
    recent = agent.theta_buffer.recent
    if recent is not None and recent.shape[0] > 0:
        base_seed = recent[-1].detach().clone()  # [batch, world_dim]
    else:
        base_seed = torch.randn(1, world_dim, generator=gen)
    # Damage scale referenced to the UNSCALED captured seed so the null arm
    # (content_scale=0) receives the SAME absolute seed corruption.
    seed_rms_ref = _rms(base_seed)
    clean_seed = base_seed * float(content_scale)
    batch = int(clean_seed.shape[0])
    z_self_init = torch.zeros(batch, self_dim)

    # One corruption realisation at this sigma; averaged over fixed action draws.
    corrupt_seed = diffuse_perturb(clean_seed, sigma, gen, rms_ref=seed_rms_ref).detach()

    def _generated_ws(seed: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        # world_states[1:] -- the re-derived (rolled-out) states, excluding the
        # raw seed passthrough at t=0.
        traj = e2.rollout_with_world(
            z_self_init, seed, actions, compute_action_objects=False
        )
        ws = traj.get_world_state_sequence()  # [batch, H+1, world_dim]
        return ws[:, 1:, :].detach()

    dev_sq = 0.0
    sig_sq = 0.0
    with torch.no_grad():
        for _ in range(max(1, int(n_rollouts))):
            actions = torch.randn(batch, H, action_dim, generator=gen)
            clean_gen = _generated_ws(clean_seed, actions)
            corrupt_gen = _generated_ws(corrupt_seed, actions)
            dev_sq += float(((corrupt_gen - clean_gen) ** 2).sum().item())
            sig_sq += float((clean_gen ** 2).sum().item())

    # Reference the input-corruption fraction to the UNSCALED seed norm: in the null
    # arm the scaled seed is zero, so its own norm is a degenerate denominator. At
    # content_scale=1.0 the two are identical, so the injected arm is unchanged.
    seed_norm = float(base_seed.norm().clamp(min=1e-12).item())
    input_rel = float((corrupt_seed - clean_seed).norm().item()) / seed_norm
    if sig_sq <= 1e-12:
        output_rel = UNAVAILABLE
    else:
        output_rel = math.sqrt(dev_sq) / math.sqrt(sig_sq)

    if sigma <= 0.0 or input_rel <= 1e-9 or output_rel == UNAVAILABLE:
        gain = UNAVAILABLE
    else:
        gain = output_rel / input_rel

    fidelity = (1.0 - output_rel) if output_rel != UNAVAILABLE else UNAVAILABLE

    out: Dict[str, float] = {
        "generative_available": 1.0,
        "rem_gen_content_scale": float(content_scale),
        "rem_gen_input_rel_corruption": float(input_rel),
        "rem_gen_output_rel_dev": float(output_rel) if output_rel != UNAVAILABLE else UNAVAILABLE,
        "rem_generative_gain": float(gain) if gain != UNAVAILABLE else UNAVAILABLE,
        "rem_generative_fidelity": float(fidelity) if fidelity != UNAVAILABLE else UNAVAILABLE,
        "rem_gen_n_rollouts": float(max(1, int(n_rollouts))),
    }

    # Liveness: inject the corrupted seed into the theta_buffer and drive the real
    # REM pass once, so the injection point is provably the seed the live path
    # reads. rem_terrain_variance is retained as telemetry only (not scored).
    if drive_liveness_pass:
        try:
            agent.theta_buffer.update(
                corrupt_seed, torch.zeros(batch, self_dim)
            )
            agent.enter_rem_mode()
            rem = agent.run_rem_attribution_pass()
            agent.exit_sleep_mode()
            out["rem_terrain_variance"] = float(rem.get("rem_terrain_variance", 0.0))
            out["rem_mean_harm_terrain"] = float(rem.get("rem_mean_harm_terrain", 0.0))
            out["rem_n_rollouts"] = float(rem.get("rem_n_rollouts", 0.0))
        except Exception:  # pragma: no cover -- liveness telemetry only, not scored
            out["rem_terrain_variance"] = 0.0
            out["rem_n_rollouts"] = 0.0

    return out


# --------------------------------------------------------------------------- #
# Per-phase integrity at one damage level                                     #
# --------------------------------------------------------------------------- #

def phase_integrity_at_sigma(
    *,
    seed: int,
    sigma: float,
    warm_steps: int = 40,
    content_scale: float = 1.0,
    rem_step: float = 1.0,
) -> Dict[str, Dict[str, float]]:
    """Run all three per-phase readouts at UNIFORM damage `sigma` on one seed.

    Each phase gets a FRESH agent so the phases are measured independently (the
    staging question is about each phase's own transfer function under the same
    diffuse damage, not about a single serial pass -- a serial pass would bake in
    the very error-compounding the non-vacuity contract must avoid).

    `content_scale` selects the arm: 1.0 == injected content (the pre-778b default,
    bit-identical), 0.0 == the SD-068 null-content control arm. The agent build, the
    warm-up drive, the RNG streams, and the delivered damage are IDENTICAL across the
    two arms -- only the injected known content is removed.

    `rem_step` (SD-068 GOV-FANOUT-1 leg H-rem-clamp-artifact, V3-EXQ-778d) is the
    adoption fraction handed to recalibrate_precision_to. 1.0 == FULL adoption == the
    778/778a/778b/778c measurement choice, bit-identical. Values below 1.0 leave the
    output a convex blend of the injected start_variance and the corrupt reference,
    which is the knob the step-ladder leg sweeps to test whether the null arm's
    both-rails degeneracy is a clamp artifact of full adoption.
    """
    results: Dict[str, Dict[str, float]] = {}

    g = _gen(seed * 1009 + 1)
    a_sws = build_pipeline_agent(seed=seed)
    _warm_encoders(a_sws, n_steps=warm_steps, gen=g)
    results["sws"] = sws_denoising_snr(
        a_sws, sigma=sigma, gen=g, content_scale=content_scale
    )

    g = _gen(seed * 1009 + 2)
    a_nrem = build_pipeline_agent(seed=seed)
    _warm_encoders(a_nrem, n_steps=warm_steps, gen=g)
    results["nrem"] = nrem_transfer_fidelity(
        a_nrem, sigma=sigma, gen=g, content_scale=content_scale
    )

    g = _gen(seed * 1009 + 3)
    a_rem = build_pipeline_agent(seed=seed)
    _warm_encoders(a_rem, n_steps=warm_steps, gen=g)
    results["rem"] = rem_precision_error(
        a_rem, sigma=sigma, gen=g, content_scale=content_scale, step=float(rem_step)
    )

    return results


def rem_only_integrity_at_sigma(
    *,
    seed: int,
    sigma: float,
    warm_steps: int = 40,
    content_scale: float = 1.0,
    rem_step: float = 1.0,
    run_generative: bool = True,
    null_mode: str = "zero_content",
    unpaired_target_precision: Optional[float] = None,
) -> Dict[str, Dict[str, float]]:
    """The REM phase readout ONLY, on the SAME RNG stream phase_integrity_at_sigma uses.

    The SD-068 GOV-FANOUT-1 portfolio (V3-EXQ-778d/e/f) is scoped to the rem leg alone
    -- the sws repair is a single unambiguous build and is exempt, and the nrem leg is
    already confirmed content-contingent. Running the full three-phase sweep for those
    legs would triple their compute to recompute two phases nothing reads.

    CRITICALLY, this reproduces `phase_integrity_at_sigma`'s rem cell EXACTLY: the same
    generator seeding (`_gen(seed * 1009 + 3)`), the same fresh agent, the same warm-up.
    It is therefore numerically comparable to the rem cell of V3-EXQ-778/778a/778c --
    which is what lets the portfolio legs carry a within-run replication ANCHOR against
    778c's recorded degeneracy signature rather than merely asserting comparability.

    Returned in the same `{phase: {metric: value}}` shape as phase_integrity_at_sigma so
    it drops straight into run_null_content_control / _common_error_series; the sws and
    nrem keys are simply absent (those phases are not swept).
    """
    g = _gen(seed * 1009 + 3)
    a_rem = build_pipeline_agent(seed=seed)
    _warm_encoders(a_rem, n_steps=warm_steps, gen=g)
    return {
        "rem": rem_precision_error(
            a_rem,
            sigma=sigma,
            gen=g,
            content_scale=content_scale,
            step=float(rem_step),
            run_generative=run_generative,
            null_mode=null_mode,
            unpaired_target_precision=unpaired_target_precision,
        )
    }


def rem_null_slope_ratio(
    *,
    sigmas: List[float],
    injected_pr_by_sigma: Dict[float, Dict[str, Dict[str, float]]],
    null_pr_by_sigma: Dict[float, Dict[str, Dict[str, float]]],
    rem_error_key: str = "calibration_error",
) -> Dict[str, float]:
    """Rem-only null-slope ratio + null-arm degeneracy telemetry.

    A thin rem-scoped wrapper over the same slope machinery run_null_content_control
    uses, for the portfolio legs that sweep the rem phase alone. Returns the ratio, both
    arms' slopes, the clamp fraction, and the NULL-ARM non-degeneracy telemetry
    (`null_series_sd` / `null_series_n_distinct`) that separates "inert on noise"
    (content-contingent) from "saturated constant" (degenerate) -- the ambiguity that
    left V3-EXQ-778c's rem leg unresolved at both rails.
    """
    denom = {
        "rem": float(
            injected_pr_by_sigma[min(sigmas)]["rem"].get("clean_target_variance", 0.0)
        )
    }
    inj_errs = _common_error_series(
        injected_pr_by_sigma, "rem", sigmas, denom, rem_error_key=rem_error_key
    )
    null_errs = _common_error_series(
        null_pr_by_sigma, "rem", sigmas, denom, rem_error_key=rem_error_key
    )
    inj_slope = _slope_of(sigmas, inj_errs)
    null_slope = _slope_of(sigmas, null_errs)

    available = (
        not math.isnan(inj_slope)
        and not math.isnan(null_slope)
        and abs(inj_slope) > NULL_MIN_INJECTED_SLOPE
    )
    ratio = (abs(null_slope) / abs(inj_slope)) if available else UNAVAILABLE

    finite_null = [e for e in null_errs if not math.isnan(e)]
    if len(finite_null) >= 2:
        mn = sum(finite_null) / len(finite_null)
        null_sd = math.sqrt(
            sum((e - mn) ** 2 for e in finite_null) / (len(finite_null) - 1)
        )
    else:
        null_sd = UNAVAILABLE

    clamped = [
        float(null_pr_by_sigma[s]["rem"].get("target_clamped", 0.0)) for s in sigmas
    ]
    return {
        "rem_error_key": rem_error_key,
        "injected_slope": float(inj_slope) if not math.isnan(inj_slope) else UNAVAILABLE,
        "null_slope": float(null_slope) if not math.isnan(null_slope) else UNAVAILABLE,
        "null_slope_ratio": float(ratio) if available else UNAVAILABLE,
        "available": 1.0 if available else 0.0,
        "content_contingent": 1.0
        if (available and ratio <= NULL_SLOPE_RATIO_CEILING)
        else 0.0,
        "null_series_sd": float(null_sd) if null_sd != UNAVAILABLE else UNAVAILABLE,
        "null_series_n_distinct": float(len({round(e, 12) for e in finite_null})),
        "null_series": [float(e) for e in null_errs],
        "injected_series": [float(e) for e in inj_errs],
        "null_target_clamped_frac": float(sum(clamped) / len(clamped)) if clamped else 0.0,
    }


# --------------------------------------------------------------------------- #
# Error-propagation gain (the non-vacuity core)                               #
# --------------------------------------------------------------------------- #

def _lin_slope(xs: List[float], ys: List[float]) -> float:
    """Least-squares slope of ys on xs; 0.0 if degenerate."""
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    return (num / den) if den > 1e-12 else 0.0


def _phase_error_series(pr_by_sigma: Dict[float, Dict[str, Dict[str, float]]], phase: str, sigmas: List[float]) -> List[float]:
    """Per-phase LINEAR relative-error series (HIGHER = WORSE), NaN where unavailable.

    All three phases are expressed as dimensionless linear relative-error fractions
    so the cross-phase tolerance comparison is fair (the log-SNR dB scale, kept as a
    reported diagnostic, is NOT used here because its noiseless sentinel dominates a
    min-max normalisation and spuriously makes SWS "fail first"):
        sws  : noise_power / signal_power        (residual-noise fraction)
        nrem : 1 - transfer_fidelity             (residual-gap fraction)
        rem  : calibration_error / clean_variance (relative calibration error)
    """
    out: List[float] = []
    for s in sigmas:
        row = pr_by_sigma[s][phase]
        if phase == "sws":
            sig = row.get("signal_power", 0.0)
            noi = row.get("noise_power", 0.0)
            out.append((noi / sig) if sig > 1e-12 else float("nan"))
        elif phase == "nrem":
            v = row.get("transfer_fidelity", UNAVAILABLE)
            out.append((1.0 - v) if v != UNAVAILABLE else float("nan"))
        else:  # rem
            err = row.get("calibration_error", UNAVAILABLE)
            cv = row.get("clean_target_variance", 0.0)
            out.append((err / cv) if (err != UNAVAILABLE and cv > 1e-12) else float("nan"))
    return out


def _normalise_degradation(errs: List[float]) -> List[float]:
    """Map an error series to fractional-of-own-range degradation in [0, 1].

    0.0 == the least-damaged observation, 1.0 == the most-damaged. This makes the
    three phases (SNR dB / fidelity / variance units) comparable for a staging-order
    ranking: each is normalised to its OWN intact->collapsed range. Comparing the
    SHAPE / threshold-crossing across phases is meaningful; comparing raw levels is
    not. Returns NaN-preserving list; all-equal or all-NaN -> zeros.
    """
    vals = [e for e in errs if not math.isnan(e)]
    if not vals:
        return [float("nan")] * len(errs)
    lo, hi = min(vals), max(vals)
    span = hi - lo
    if span <= 1e-12:
        return [0.0 if not math.isnan(e) else float("nan") for e in errs]
    return [((e - lo) / span) if not math.isnan(e) else float("nan") for e in errs]


def _threshold_sigma(sigmas: List[float], frac: List[float], level: float = 0.5) -> float:
    """Smallest sigma at which normalised degradation crosses `level` (linear interp).

    This is the phase's DAMAGE TOLERANCE: a lower crossing sigma == "fails at a
    lower damage level" == fails EARLIER in the staged-decline sense. Returns the
    max sigma if the level is never reached; UNAVAILABLE if the series is empty.
    """
    pts = [(s, f) for s, f in zip(sigmas, frac) if not math.isnan(f)]
    if not pts:
        return UNAVAILABLE
    prev_s, prev_f = pts[0]
    for s, f in pts[1:]:
        if f >= level:
            if f == prev_f:
                return float(s)
            t = (level - prev_f) / (f - prev_f)
            return float(prev_s + t * (s - prev_s))
        prev_s, prev_f = s, f
    return float(pts[-1][0])


def error_propagation_gain(
    *,
    seed: int,
    sigmas: List[float],
    warm_steps: int = 40,
    pr_by_sigma: Optional[Dict[float, Dict[str, Dict[str, float]]]] = None,
) -> Dict[str, float]:
    """Staging + non-vacuity readouts over the uniform-damage sigma grid.

    IMPORTANT (substrate finding, honest scoping): the three V3 consolidation
    phases operate on DISJOINT state (context_memory / consolidation params / E3
    precision), so a faithful cross-phase CONTENT-propagation pipe (corrupt one
    phase's output, feed it as the next phase's input) is NOT directly
    instrumentable. This function therefore does NOT claim a cross-phase
    propagation gain. It carries the non-vacuity two honest ways:

      (1) STAGING ORDER via per-phase damage-TOLERANCE. Each phase's error is
          normalised to fractional-of-own-range degradation, then ranked by the
          sigma at which it crosses 50% degradation. Lower crossing sigma == fails
          earlier. This is comparable across phases and matches the clinical
          "which symptom appears at the earliest disease stage" framing.

      (2) REM PASSTHROUGH-vs-GENERATIVE contrast. The bare precision nudge is
          passthrough by construction; the generative re-derivation's fidelity is
          now read cleanly via rollout-seed injection (rem_generative_fidelity),
          NOT the retired rem_terrain_variance proxy. The load-bearing number is
          rem_generative_gain -- the least-squares slope of the generated-rollout
          relative deviation against the injected seed's relative corruption across
          the sigma grid:
            gain < 1 == ATTENUATING  (genuine generative correction; non-vacuous
                        staging -- the REM pass recovers content from a corrupt seed),
            gain ~ 1 == PASSTHROUGH   (staging is topology),
            gain > 1 == AMPLIFYING    (MECH-094 psychosis polarity).
          rem_generative_output_slope (output deviation vs sigma) is kept for the
          existing driver contract but is now fed by the REAL readout, not the
          proxy. rem_generative_attenuates is a convenience verdict (gain < 1).
    """
    if pr_by_sigma is None:
        pr_by_sigma = {}
        for s in sigmas:
            pr_by_sigma[s] = phase_integrity_at_sigma(seed=seed, sigma=s, warm_steps=warm_steps)

    out: Dict[str, float] = {}
    tol: Dict[str, float] = {}
    for phase in ("sws", "nrem", "rem"):
        errs = _phase_error_series(pr_by_sigma, phase, sigmas)
        frac = _normalise_degradation(errs)
        xs = [s for s, f in zip(sigmas, frac) if not math.isnan(f)]
        ys = [f for f in frac if not math.isnan(f)]
        out[f"norm_degradation_slope_{phase}"] = float(_lin_slope(xs, ys))
        tol[phase] = _threshold_sigma(sigmas, frac, level=0.5)
        out[f"tolerance_sigma_{phase}"] = tol[phase]

    # REM passthrough calibration sensitivity vs generative-output sensitivity.
    cal = [pr_by_sigma[s]["rem"].get("calibration_error", float("nan")) for s in sigmas]
    xs = [s for s, c in zip(sigmas, cal) if not math.isnan(c) and c != UNAVAILABLE]
    ys = [c for c in cal if not math.isnan(c) and c != UNAVAILABLE]
    passthrough_slope = float(_lin_slope(xs, ys)) if ys else UNAVAILABLE

    gen_avail = all(pr_by_sigma[s]["rem"].get("generative_available", 0.0) >= 1.0 for s in sigmas)

    # Real generative readout (rollout-seed injection): output relative deviation
    # and input relative seed corruption, per sigma.
    def _rem_val(s: float, key: str) -> float:
        v = pr_by_sigma[s]["rem"].get(key, float("nan"))
        return float("nan") if (isinstance(v, float) and (math.isnan(v) or v == UNAVAILABLE)) else float(v)

    out_dev = [_rem_val(s, "rem_gen_output_rel_dev") for s in sigmas]
    in_cor = [_rem_val(s, "rem_gen_input_rel_corruption") for s in sigmas]

    # rem_generative_output_slope: output deviation vs sigma (existing driver key;
    # now fed by the REAL rollout-seed readout, not the retired terrain-variance).
    oxs = [s for s, d in zip(sigmas, out_dev) if not math.isnan(d)]
    oys = [d for d in out_dev if not math.isnan(d)]
    generative_output_slope = float(_lin_slope(oxs, oys)) if oys else UNAVAILABLE

    # rem_generative_gain: the dimensionless amplify/attenuate factor -- slope of
    # output relative deviation against input relative seed corruption (the origin
    # sigma=0 point (0,0) anchors it).
    gain_pts = [
        (c, d)
        for c, d in zip(in_cor, out_dev)
        if not math.isnan(c) and not math.isnan(d)
    ]
    gxs = [c for c, _ in gain_pts]
    gys = [d for _, d in gain_pts]
    generative_gain = float(_lin_slope(gxs, gys)) if len(gain_pts) >= 2 else UNAVAILABLE

    # Mean of per-sigma point gains (sigma>0 only) as a robustness cross-check.
    per_gain = [_rem_val(s, "rem_generative_gain") for s in sigmas]
    per_gain = [g for g in per_gain if not math.isnan(g)]
    generative_gain_mean = float(sum(per_gain) / len(per_gain)) if per_gain else UNAVAILABLE

    out["rem_passthrough_calibration_slope"] = passthrough_slope
    out["rem_generative_output_slope"] = generative_output_slope
    out["rem_generative_gain"] = generative_gain
    out["rem_generative_gain_mean"] = generative_gain_mean
    out["rem_generative_attenuates"] = (
        1.0 if (generative_gain != UNAVAILABLE and generative_gain < 1.0) else 0.0
    )
    out["rem_generative_available"] = 1.0 if gen_avail else 0.0
    return out


# --------------------------------------------------------------------------- #
# Null-content control (SD-068 non-vacuity floor, V3-EXQ-778b)                #
# --------------------------------------------------------------------------- #

# A phase's readout counts as CONTENT-CONTINGENT when its null-arm sigma-slope is
# at most this fraction of its injected-arm sigma-slope. Pre-registered.
NULL_SLOPE_RATIO_CEILING = 0.25
# Below this injected-arm slope the ratio is not interpretable (0/0) -- the phase is
# reported UNAVAILABLE rather than silently scored as a pass.
NULL_MIN_INJECTED_SLOPE = 1e-9


def _common_error_series(
    pr_by_sigma: Dict[float, Dict[str, Dict[str, float]]],
    phase: str,
    sigmas: List[float],
    denom: Dict[str, float],
    rem_error_key: str = "calibration_error",
) -> List[float]:
    """Per-phase error series in COMMON units across the injected and null arms.

    The injected-arm denominators are supplied by the caller and used for BOTH arms,
    because the null arm's own denominators collapse to zero with the content (no
    injected signal power, no injected consolidation gap). Holding the denominator
    fixed is what makes the two arms' slopes directly comparable -- the ratio then
    answers exactly "how much of this readout's sigma-response survives when the
    content is removed?".

        sws  : noise_power            / injected signal_power
        nrem : gap_after              / injected gap_before
        rem  : calibration_error      / clean_target_variance

    `rem_error_key` (V3-EXQ-778e) selects which rem readout the series is built from.
    "calibration_error" (the DEFAULT) is the variance-units readout every prior run
    scored, kept bit-identical. "direct_precision_error" is the de-clamped
    precision-units companion, which is ALREADY a relative fraction, so its
    denominator is 1.0 rather than clean_target_variance -- dividing an
    already-normalised error by a variance would silently rescale it.
    """
    out: List[float] = []
    for s in sigmas:
        row = pr_by_sigma[s][phase]
        if phase == "sws":
            d = denom.get("sws", 0.0)
            noi = row.get("noise_power", float("nan"))
            out.append((noi / d) if d > 1e-12 else float("nan"))
        elif phase == "nrem":
            d = denom.get("nrem", 0.0)
            ga = row.get("gap_after", float("nan"))
            out.append((ga / d) if d > 1e-12 else float("nan"))
        else:  # rem
            # An already-relative readout carries its own scale; only the
            # variance-units calibration_error needs the injected denominator.
            d = 1.0 if rem_error_key != "calibration_error" else denom.get("rem", 0.0)
            err = row.get(rem_error_key, UNAVAILABLE)
            out.append(
                (err / d) if (err != UNAVAILABLE and d > 1e-12) else float("nan")
            )
    return out


def _injected_denominators(
    pr_by_sigma: Dict[float, Dict[str, Dict[str, float]]], sigmas: List[float]
) -> Dict[str, float]:
    """The injected arm's reference scales, taken at the intact (lowest) sigma."""
    s0 = min(sigmas)
    return {
        "sws": float(pr_by_sigma[s0]["sws"].get("signal_power", 0.0)),
        "nrem": float(pr_by_sigma[s0]["nrem"].get("gap_before", 0.0)),
        "rem": float(pr_by_sigma[s0]["rem"].get("clean_target_variance", 0.0)),
    }


def _slope_of(sigmas: List[float], errs: List[float]) -> float:
    xs = [s for s, e in zip(sigmas, errs) if not math.isnan(e)]
    ys = [e for e in errs if not math.isnan(e)]
    return float(_lin_slope(xs, ys)) if len(ys) >= 2 else float("nan")


def run_null_content_control(
    *,
    seed: int,
    sigmas: Optional[List[float]] = None,
    warm_steps: int = 40,
    injected_pr_by_sigma: Optional[Dict[float, Dict[str, Dict[str, float]]]] = None,
    null_pr_by_sigma: Optional[Dict[float, Dict[str, Dict[str, float]]]] = None,
    rem_error_key: str = "calibration_error",
) -> Dict[str, float]:
    """Zero-injected-content NULL CONTROL for the per-phase damage readouts.

    THE CONTROL THIS IMPLEMENTS (Bar et al. 2020, Curr Biol, DOI
    10.1016/j.cub.2020.01.091 -- the methodological precedent SD-068 follows).
    Bar et al. injected known content, applied a scoped perturbation, and read out at
    the same scope; what made it convincing was the odour-contingency NULL --
    unilateral olfactory stimulation during sleep produced NO memory effect and NO
    oscillatory effect when the learning had happened WITHOUT the contextual odour.
    The perturbation alone does nothing; it acts only on injected content.

    SD-068 had no analog of that null until this function. The risk it closes:

        If a per-phase readout still moves with `sigma` when NO known content has
        been injected, that readout is measuring PERTURBATION MAGNITUDE, not content
        fidelity -- and the damage-tolerance staging order is then an ordering of the
        three phases' raw NOISE SENSITIVITY, not of their functional damage
        tolerance. That is precisely the vacuity SD-068 claims to escape.

    METHOD. The identical sigma sweep is run twice on the identical substrate, seeds,
    warm-up and RNG streams. Only `content_scale` differs (1.0 injected vs 0.0 null).
    The delivered perturbation is held numerically IDENTICAL in both arms (each
    readout references its noise scale to the UNSCALED content -- see
    `diffuse_perturb(rms_ref=...)`), so the null arm is "same odour, no prior
    pairing" rather than "weaker odour". Both arms' error series are expressed in
    COMMON units using the INJECTED arm's denominators, then least-squares fitted
    against sigma.

    RETURNS (per phase p in sws/nrem/rem):
        injected_slope_<p>      -- d(error)/d(sigma) with content injected
        null_slope_<p>          -- d(error)/d(sigma) on noise alone
        null_slope_ratio_<p>    -- |null| / |injected|  (the REPORTED number)
        content_contingent_<p>  -- 1.0 if ratio <= NULL_SLOPE_RATIO_CEILING
        null_control_available_<p>

    The RATIO is reported per phase rather than a bare pass/fail so a PARTIAL null is
    visible: 0.0 == fully content-contingent (the readout is inert on noise), 1.0 ==
    fully confounded (the readout responds to sigma identically with and without
    content), and intermediate values are exactly that -- intermediate.

    A phase whose ratio exceeds the ceiling is CONFOUNDED: its contribution to the
    staging order cannot be distinguished from raw noise sensitivity. Such a phase is
    NAMED in `confounded_phases` and flagged -- never silently dropped from the order.
    """
    if sigmas is None:
        sigmas = [0.0, 0.25, 0.5, 1.0, 2.0]
    sigmas = list(sigmas)

    if injected_pr_by_sigma is None:
        injected_pr_by_sigma = {
            s: phase_integrity_at_sigma(
                seed=seed, sigma=s, warm_steps=warm_steps, content_scale=1.0
            )
            for s in sigmas
        }
    if null_pr_by_sigma is None:
        null_pr_by_sigma = {
            s: phase_integrity_at_sigma(
                seed=seed, sigma=s, warm_steps=warm_steps, content_scale=0.0
            )
            for s in sigmas
        }

    denom = _injected_denominators(injected_pr_by_sigma, sigmas)

    out: Dict[str, float] = {}
    confounded: List[str] = []
    for phase in ("sws", "nrem", "rem"):
        inj_errs = _common_error_series(
            injected_pr_by_sigma, phase, sigmas, denom, rem_error_key=rem_error_key
        )
        null_errs = _common_error_series(
            null_pr_by_sigma, phase, sigmas, denom, rem_error_key=rem_error_key
        )
        inj_slope = _slope_of(sigmas, inj_errs)
        null_slope = _slope_of(sigmas, null_errs)

        available = (
            not math.isnan(inj_slope)
            and not math.isnan(null_slope)
            and abs(inj_slope) > NULL_MIN_INJECTED_SLOPE
        )
        ratio = (abs(null_slope) / abs(inj_slope)) if available else UNAVAILABLE

        out[f"injected_slope_{phase}"] = (
            float(inj_slope) if not math.isnan(inj_slope) else UNAVAILABLE
        )
        out[f"null_slope_{phase}"] = (
            float(null_slope) if not math.isnan(null_slope) else UNAVAILABLE
        )
        out[f"null_slope_ratio_{phase}"] = float(ratio) if available else UNAVAILABLE
        out[f"null_control_available_{phase}"] = 1.0 if available else 0.0

        # NULL-ARM NON-DEGENERACY telemetry (V3-EXQ-778e). A ratio of ~0.0 is
        # ambiguous between "the readout is genuinely inert on noise" (content-
        # contingent, the finding we want) and "the null series is a SATURATED
        # CONSTANT" (degenerate, no finding at all) -- V3-EXQ-778c's rem leg was
        # exactly the latter on 5/8 seeds. Recording the null series' own spread and
        # distinct-value count lets a consumer gate on that difference instead of
        # reading a degenerate zero as a clean pass.
        finite_null = [e for e in null_errs if not math.isnan(e)]
        if len(finite_null) >= 2:
            mn = sum(finite_null) / len(finite_null)
            null_sd = math.sqrt(
                sum((e - mn) ** 2 for e in finite_null) / (len(finite_null) - 1)
            )
        else:
            null_sd = UNAVAILABLE
        out[f"null_series_sd_{phase}"] = float(null_sd) if null_sd != UNAVAILABLE else UNAVAILABLE
        out[f"null_series_n_distinct_{phase}"] = float(
            len({round(e, 12) for e in finite_null})
        )
        contingent = bool(available and ratio <= NULL_SLOPE_RATIO_CEILING)
        out[f"content_contingent_{phase}"] = 1.0 if contingent else 0.0
        # An UNAVAILABLE ratio is NOT a pass -- it is an uninterpretable phase, and
        # it is named alongside the confounded ones so it cannot pass silently.
        if not contingent:
            confounded.append(phase)

    # Clamp saturation in the NULL arm's rem leg: a ratio built on clamped points is
    # off-scale (the 1e-3 positivity floor dominates), so it reads as "structurally
    # content-free", NOT as a literal N-fold noise sensitivity. Reported so a large
    # rem ratio cannot be mistaken for a calibrated magnitude.
    clamped = [
        float(null_pr_by_sigma[s]["rem"].get("target_clamped", 0.0)) for s in sigmas
    ]
    out["null_rem_target_clamped_frac"] = (
        float(sum(clamped) / len(clamped)) if clamped else 0.0
    )
    out["null_slope_ratio_rem_off_scale"] = 1.0 if any(clamped) else 0.0

    out["n_confounded_phases"] = float(len(confounded))
    out["n_content_contingent_phases"] = float(3 - len(confounded))
    out["all_phases_content_contingent"] = 1.0 if not confounded else 0.0
    # Stored as a parallel per-phase flag set; the driver renders the name list.
    out["null_control_seed"] = float(seed)
    return out


def confounded_phase_names(control: Dict[str, float]) -> List[str]:
    """Phases whose staging contribution is confounded by raw noise sensitivity."""
    return [
        p
        for p in ("sws", "nrem", "rem")
        if control.get(f"content_contingent_{p}", 0.0) < 1.0
    ]


# --------------------------------------------------------------------------- #
# Top-level orchestration for a driver                                        #
# --------------------------------------------------------------------------- #

@dataclass
class StagedSweepResult:
    seed: int
    sigmas: List[float]
    integrity: Dict[str, Dict[str, List[float]]] = field(default_factory=dict)
    gains: Dict[str, float] = field(default_factory=dict)
    predicted_order: Tuple[str, str, str] = REVERSE_DEPENDENCY_ORDER
    observed_order: List[str] = field(default_factory=list)
    staging_matches_prediction: bool = False
    # SD-068 null-content control (V3-EXQ-778b); empty unless run_null_control=True.
    null_control: Dict[str, float] = field(default_factory=dict)
    confounded_phases: List[str] = field(default_factory=list)


def run_staged_sweep(
    *,
    seed: int,
    sigmas: Optional[List[float]] = None,
    warm_steps: int = 40,
    run_null_control: bool = False,
) -> StagedSweepResult:
    """Full per-seed staged-damage sweep.

    Produces: per-phase integrity across the sigma grid, the per-phase gains, the
    OBSERVED staged-failure order (phases ranked by gain, most-sensitive first),
    and whether that order matches the predicted reverse-dependency order
    (rem, nrem, sws). The driver combines this with the OFF-arm zero-baseline and
    the passthrough-vs-generative REM contrast to reach a non-vacuous verdict.

    `run_null_control` (V3-EXQ-778b, default False so the V3-EXQ-778 path stays
    bit-identical) additionally runs the zero-injected-content NULL arm and populates
    `null_control` + `confounded_phases`. A phase named in `confounded_phases` has a
    readout that moves with sigma even on noise alone, so its contribution to
    `observed_order` is an ordering of noise sensitivity rather than of functional
    damage tolerance. The confounded phases are REPORTED, never dropped from the
    order -- dropping them would hide the confound rather than surface it.
    """
    if sigmas is None:
        sigmas = [0.0, 0.25, 0.5, 1.0, 2.0]
    res = StagedSweepResult(seed=seed, sigmas=list(sigmas))

    # Compute the sweep ONCE; derive both integrity and gains from it.
    pr_by_sigma: Dict[float, Dict[str, Dict[str, float]]] = {}
    for s in sigmas:
        pr_by_sigma[s] = phase_integrity_at_sigma(seed=seed, sigma=s, warm_steps=warm_steps)

    integ: Dict[str, Dict[str, List[float]]] = {p: {} for p in ("sws", "nrem", "rem")}
    for s in sigmas:
        for phase, row in pr_by_sigma[s].items():
            for k, v in row.items():
                integ[phase].setdefault(k, []).append(float(v))
    res.integrity = integ

    res.gains = error_propagation_gain(
        seed=seed, sigmas=list(sigmas), warm_steps=warm_steps, pr_by_sigma=pr_by_sigma
    )

    # Observed staged order: phases with the LOWEST damage-tolerance (cross 50%
    # degradation at the lowest sigma) fail EARLIEST. Ties broken by the larger
    # normalised-degradation slope (steeper decline). UNAVAILABLE tolerances sort
    # last.
    def _key(p: str) -> Tuple[float, float]:
        tol = res.gains.get(f"tolerance_sigma_{p}", UNAVAILABLE)
        tol_k = float("inf") if tol == UNAVAILABLE else float(tol)
        slope = res.gains.get(f"norm_degradation_slope_{p}", 0.0)
        return (tol_k, -float(slope))

    res.observed_order = sorted(("sws", "nrem", "rem"), key=_key)
    res.staging_matches_prediction = tuple(res.observed_order) == REVERSE_DEPENDENCY_ORDER

    if run_null_control:
        # Reuse the injected sweep already computed above -- only the null arm is new.
        res.null_control = run_null_content_control(
            seed=seed,
            sigmas=list(sigmas),
            warm_steps=warm_steps,
            injected_pr_by_sigma=pr_by_sigma,
        )
        res.confounded_phases = confounded_phase_names(res.null_control)

    return res
