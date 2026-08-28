#!/opt/local/bin/python3
"""
V3-EXQ-862b: Q-040.c -- is dACC behavioural-adjustment magnitude proportional
to V_s precision-modulated forward-PE under MECH-269b=ON?

Supersedes V3-EXQ-862a. See
REE_assembly/evidence/planning/failure_autopsy_dacc-cluster-862a-870a_2026-08-03.md
(confirmed): 862a fixed both prior config bugs (475b's z_harm_a wiring, 862's
dacc_weight=0 consumer gain) and both of its preconditions (P1 aggregate
gate-firing, P2 dACC engagement) PASSed -- but the ON/OFF manipulation itself
STILL never engaged. Code-verified: `vs_gate_held_e1_z_harm_a =
vs_gate_held_e2_z_harm_a = 0` in every ARM_vs_on cell across all 3 seeds
(0/128, 0/200, 0/10 refreshes). `pe_series`/`bias_magnitude_series` were
bit-identical between ARM_vs_on and ARM_vs_off per seed -- the two arms were
functionally IDENTICAL for this DV, because the one stream the DV reads
(z_harm_a) never crossed the gate threshold.

ROOT CAUSE (code-verified, per the autopsy's table): 862a's ARM_vs_on used a
SINGLE GLOBAL gate threshold (`vs_gate_e1_threshold=0.85`,
`vs_gate_e2_threshold=0.85`) carried verbatim from V3-EXQ-490b, which
calibrated it for a DIFFERENT target stream's V_s dynamics. At that
threshold, z_harm_a's own V_s apparently never dips below 0.85 in this
harness -- the ON arm's held-budget landed almost entirely on `z_beta`
instead (119/122, 191/193, 1/1 of the held count), a stream the dACC PE
computation (`dacc.py:211-214`, `pe = ||z_harm_a - z_harm_a_pred||`) does not
read at all. The gate's own `e1_threshold_per_stream`/`e2_threshold_per_stream`
override dicts (`ree_core/regulators/vs_rollout_gate.py:59-60`) exist
precisely to avoid this and were never used by 862/862a.

THE FIX -- empirically calibrated per-stream threshold, not another guess.
This session measured z_harm_a's own empirical V_s distribution directly
(ad hoc calibration probe, ARM_vs_on-equivalent config, 3 seeds x 15 warmup +
5 eval episodes x 100 steps, per-tick `hippocampal.per_stream_vs["z_harm_a"]`
read via a StepHooks callback -- not committed, throwaway):

    n=1192 pooled ticks: min=0.9892 p5=0.9932 p25=0.9961 median=0.9983
    p75=0.9997 p95=1.0000 max=1.0000

z_harm_a lives in a tight, near-ceiling band -- confirming the autopsy's
read exactly (0.85 sits nowhere near this stream's operating range; 490b's
threshold was calibrated for a stream whose V_s spans a much lower range).
Z_HARM_A_E1_THRESHOLD = Z_HARM_A_E2_THRESHOLD = 0.995 sits between p5 (0.9932)
and p25 (0.9961) of the observed distribution -- low enough that a genuine
~15-20% minority of ticks fall below it (producing real, non-degenerate
held-snapshot substitutions), high enough that it does not degenerate into
"held on every tick" (which would erase the ON/OFF contrast the DV depends
on the OTHER direction). VS_GATE_SNAPSHOT_REFRESH_THRESHOLD is raised from
862a's 0.95 to 0.999 (sits above p75=0.9997, near p95=1.0000) so the
gate's documented Schmitt-trigger dead-band property (`vs_rollout_gate.py`
docstring: "snapshot_refresh_threshold sits ABOVE max(e1_threshold,
e2_threshold) by design") actually holds for z_harm_a's threshold -- with
862a's refresh_threshold=0.95 sitting BELOW the new 0.995 gate threshold,
z_harm_a's snapshot would refresh on essentially every tick (min observed
0.9892 > 0.95) including the very tick a hold is triggered, making the
"held" substitution content-identical to the current value and defeating
the manipulation even while the diagnostic counters showed nonzero holds.
snapshot_refresh_threshold is a single GLOBAL scalar on VsRolloutGateConfig
(no per-stream override exists for it, unlike e1/e2), so this change also
affects the other 5 streams' refresh cadence -- deliberately not a concern,
since none of them feed the PE this experiment's DV reads (see MECHANISM
below); their gating fidelity is out of scope for Q-040.c.

WIRING NOTE (the autopsy's routing text names this fix as `hc.
e1_threshold_per_stream = {...}` but that field does not exist on
HippocampalConfig -- confirmed by reading `ree_core/agent.py:2549-2579`,
where `VsRolloutGateConfig` is constructed from `self.hippocampal.config`
attributes and no `*_threshold_per_stream` getattr is ever read there;
`ree_core/utils/config.py:2585-2588`'s own comment confirms this
explicitly: "Per-stream override dicts ... are kept at the gate-config level
... and wired only via the agent constructor" -- but the agent constructor
itself never populates them). The only live wiring path is to set them
directly on the constructed gate's config object, post-`REEAgent(cfg)`:
`agent.vs_rollout_gate.config.e1_threshold_per_stream = {"z_harm_a": ...}`.
`_apply_z_harm_a_gate_calibration()` below does this, called immediately
after agent construction in both the pre-flight probe and every
seed x arm cell (a no-op on the OFF arm, where `agent.vs_rollout_gate` is
None because `use_vs_rollout_gating=False` never instantiates the gate at
all).

PRE-FLIGHT SMOKE GATE (extended over 862a's version, per this autopsy's
explicit routing: "a preflight assertion that the gate actually opens at
least once" / "a stream-specific engagement precondition"). Both of 862a's
gates are UNCHANGED (`PreflightDaccNotFiring` for the 475b defect class,
`PreflightDaccBiasZero` for the 862 defect class) -- neither fix from 862a
regressed. NEW: `PreflightVsGateZHarmANotHeld`, raised if the pilot rollout
(ARM_vs_on, seed=42, 3 warmup + 1 eval x 20 steps -- unchanged scale from
862a) observes ZERO `vs_gate_held_e1_z_harm_a + vs_gate_held_e2_z_harm_a`
after applying the new per-stream calibration. This is the exact defect
class this letter fixes (862a's own precondition never checked the
STREAM-SPECIFIC hold count, only the six-stream aggregate, which is why it
false-positived). Verified this session at preflight scale (3 warmup + 1
eval x 20 steps): seed 42 -> held_e1=3 held_e2=3; seed 7 -> held_e1=11
held_e2=3; seed 19 -> held_e1=0 held_e2=0 (2/3 seeds nonzero even at this
tiny scale -- at the full 200-step eval scale all 3 seeds are expected to
clear the >=2/3 P1' bar comfortably).

MECHANISM. Unchanged from 862a. ree_core/cingulate/dacc.py's
DACCAdaptiveControl.forward() computes a precision-weighted affective-pain
PE (MECH-258, bundle["pe"]) each tick and DACCtoE3Adapter converts the
bundle into a per-candidate E3 score bias (agent._dacc_last_bias). The
"dACC weight-delta" this experiment reads is the MAGNITUDE of that per-step
bias vector (its L2 norm across candidates) -- the behavioural-adjustment
dACC actually applies. The claim under test is architectural: MECH-269b (V_s
gating E1/E2 forward rollouts, ree_core/agent.py, IMPLEMENTED 2026-04-26)
determines whether the z_harm_a forward prediction dACC's PE is computed
against is fresh or stale. If V_s gating is OFF, E2 keeps grounding
harm-rollouts on stale streams, so any PE dACC sees is decoupled from
genuine environmental precision -- the bias magnitude should track it weakly
if at all. If V_s gating is ON *and actually engages the z_harm_a stream
specifically* (862a's gap), PE should track a live precision-modulated
signal, and the bias magnitude should track it detectably.

ARMS. Unchanged from 862a except the calibration applied to ARM_vs_on (see
THE FIX above):
  ARM_vs_off: gap4-475a-conditions substrate + full V_s invalidation circuit
    (use_per_stream_vs/use_event_segmenter/use_invalidation_trigger/
    use_anchor_sets/use_per_region_vs/use_staleness_accumulator/
    use_mech284_hysteresis/use_vs_commit_release all True) but
    use_vs_rollout_gating=False. AFFECTIVE HARM STREAM ON (475b/862a fix).
  ARM_vs_on: identical, use_vs_rollout_gating=True, global gate thresholds
    unchanged from 862a (vs_gate_e1_threshold=vs_gate_e2_threshold=0.85 --
    irrelevant to z_harm_a now that its own per-stream override wins, and
    left as-is for the other 5 streams which this DV does not read), PLUS
    the new z_harm_a-specific per-stream override (0.995/0.995) and the
    raised global snapshot_refresh_threshold (0.999) -- see THE FIX above.

HARD PRECONDITIONS.
  P1 (aggregate gate-firing, RETAINED, informational only -- no longer
     gates the run). agent.vs_rollout_gate diagnostics show >0 total held
     (e1+e2, all six streams) on >=2/3 seeds in ARM_vs_on. Kept for
     continuity with 475b/862/862a manifests; NOT sufficient on its own
     (862a's exact failure mode: this passed while z_harm_a itself never
     held).
  P1' (z_harm_a-specific gate-firing, NEW -- this is the operative
     precondition). agent.vs_rollout_gate diagnostics show
     vs_gate_held_e1_z_harm_a + vs_gate_held_e2_z_harm_a > 0 on >=2/3 seeds
     in ARM_vs_on. If not met, the ON/OFF manipulation never actually
     touched the stream this DV reads and no verdict is possible --
     precisely 862a's confirmed failure mode.
  P2 (dACC engagement, unchanged from 862a): dacc_bias_nonzero_steps > 0 on
     >=2/3 seeds in BOTH arms.
  All three are checked BEFORE the primary DV is read; failing P1' or P2
  self-routes evidence_direction="non_contributory"
  (substrate_not_ready_requeue), never a weakens. P1 alone failing does not
  block (it is diagnostic-only, per above).

PRIMARY DV. Unchanged from 862a. Per (seed, arm), the Spearman rank
correlation (via the degeneracy-safe experiments/_lib/stats.py::spearman)
between the per-eval-step precision-weighted PE (bundle["pe"]) and the
per-step dACC bias magnitude (||agent._dacc_last_bias||_2), over every eval
step where both are defined.

PASS = (>=2/3 ARM_vs_on seeds have a DEFINED rho with |rho| >= 0.3) AND
       (>=2/3 ARM_vs_off seeds have a DEFINED rho with |rho| < 0.15).
A seed with an UNDEFINED (degenerate-input) rho is excluded from both counts,
never coerced to 0.

DECLARED NULL. Unchanged from 862a. A FAIL here (no clean ON/OFF
dissociation, WITH the manipulation confirmed engaged via P1') does NOT
reopen Q-040a/b or goal_pipeline:GAP-4 -- those are independently settled by
the 490 cohort. A FAIL means only that the fine-grained PE-magnitude coupling
this sub-question asks about is not detectable at this harness's
scale/design, GIVEN that the manipulation genuinely reached the stream it
targets (the P1' gate is precisely what makes this reading trustworthy,
where 862a's could not).

GOV-REUSE-1: the decisive readout (dACC bias-magnitude vs precision-weighted
PE correlation, WITH dACC engaged AND the z_harm_a-specific V_s gate
actually holding at least once) is not recorded in any prior manifest.
`reanalysis_query.py query --readout rho_pe_vs_bias_magnitude --claim Q-040`
returns 475b (n_dacc_fires=0 everywhere) and 862a (dACC engaged but
vs_gate_held_e1/e2_z_harm_a=0 everywhere) -- neither satisfies the P1'
precondition this letter adds, so neither is a usable prior reading. Not
recoverable -> proceed to author (this script).

Re-derive brake: 0 prior `substrate_ceiling` autopsies tag Q-040 (per the
standard grep-count method over
REE_assembly/evidence/planning/failure_autopsy_*.json) -- does not fire.
This is a third consecutive non_contributory driver-bug category
(measurement/instrumentation), not a substrate_ceiling; per the autopsy's
own "Granularity-debt recurrence check", three non_contributory reads with
no `weakened` reading is instrumentation debt, not a signal the claim needs
splitting.

SLEEP DRIVER: not applicable (no sleep-cycle flags set).

claim_ids: [Q-040]
experiment_purpose: evidence
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from experiment_protocol import emit_outcome  # noqa: E402
from _lib.stats import spearman  # noqa: E402
from _lib.goal_pipeline_tier1 import (  # noqa: E402
    ArmSpec,
    ENV_FISHTANK_KWARGS,
    EVAL_EPISODES_DEFAULT,
    SEEDS_DEFAULT,
    STEPS_PER_EPISODE_DEFAULT,
    WARMUP_EPISODES_DEFAULT,
    build_config,
    make_env,
    warmup_train,
)
from experiments._metrics import check_degeneracy  # noqa: E402
from experiments._harness import StepHarness, StepHooks  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_862b_q040c_dacc_pe_weight_delta_correlation"
QUEUE_ID = "V3-EXQ-862b"
CLAIM_IDS = ["Q-040"]
EXPERIMENT_PURPOSE = "evidence"
SUPERSEDES_RUN_ID = "v3_exq_862a_q040c_dacc_pe_weight_delta_correlation_20260802T195935Z_v3"
SUPERSEDES_QUEUE_ID = "V3-EXQ-862a"

VS_ON_ARM = "ARM_vs_on"
VS_OFF_ARM = "ARM_vs_off"

# Global gate thresholds -- unchanged from 862a's borrowed-from-490b values.
# Left in place ONLY for the 5 streams this DV does not read; z_harm_a's own
# per-stream override (below) supersedes these for the one stream that
# matters. See module docstring "THE FIX" / "WIRING NOTE".
VS_GATE_E1_THRESHOLD = 0.85
VS_GATE_E2_THRESHOLD = 0.85

# 862b fix: empirically-calibrated per-stream threshold for z_harm_a (see
# module docstring "THE FIX" for the calibration procedure and the observed
# distribution: min=0.9892 p5=0.9932 p25=0.9961 median=0.9983 p75=0.9997
# p95=1.0000 max=1.0000, n=1192 pooled ticks across 3 seeds).
Z_HARM_A_E1_THRESHOLD = 0.995
Z_HARM_A_E2_THRESHOLD = 0.995

# Raised from 862a's 0.95 -- must sit ABOVE Z_HARM_A_E1/E2_THRESHOLD for the
# gate's documented Schmitt-trigger dead-band to hold for z_harm_a (see
# module docstring "THE FIX"). Global scalar (no per-stream override exists
# for this field on VsRolloutGateConfig); affects the other 5 streams'
# refresh cadence too, deliberately out of scope for this DV.
VS_GATE_SNAPSHOT_REFRESH_THRESHOLD = 0.999

# Acceptance thresholds for the primary DV (Spearman rho of |PE| vs dACC
# bias-vector magnitude, per seed). Unchanged from 862a.
CORR_DETECT_THRESH = 0.3   # ON arm: |rho| at/above this counts as "detected"
CORR_NULL_THRESH = 0.15    # OFF arm: |rho| below this counts as "~zero"
SEEDS_PASS_MIN = 2         # of 3, matching the cohort's TIER1_SEEDS_PASS_MIN

# Pre-flight smoke-gate scale -- unchanged from 862a.
PREFLIGHT_WARMUP_EPISODES = 3
PREFLIGHT_EVAL_EPISODES = 1
PREFLIGHT_STEPS_PER_EPISODE = 20
PREFLIGHT_SEED = SEEDS_DEFAULT[0]


class PreflightDaccNotFiring(RuntimeError):
    """Unchanged from 862a. Raised when the pre-flight pilot rollout observes
    zero dACC engagement (the 475b defect class -- z_harm_a structurally
    None because enable_affective_harm_stream was never forwarded)."""


class PreflightDaccBiasZero(RuntimeError):
    """Unchanged from 862a. Raised when dACC fires but its bias magnitude
    never clears the floor (the 862 defect class -- dacc_weight=0.0
    structurally zeroing DACCtoE3Adapter's output regardless of PE)."""


class PreflightVsGateZHarmANotHeld(RuntimeError):
    """NEW in 862b. Raised when the pilot rollout observes dACC engaging
    (bundle populated, bias nonzero -- both prior gates pass) but the
    z_harm_a-specific V_s gate never actually held a stale snapshot
    (vs_gate_held_e1_z_harm_a + vs_gate_held_e2_z_harm_a == 0).

    This is the exact defect class this letter fixes -- 862a's own preflight
    checked only dACC firing and bias magnitude, neither of which depends on
    which STREAM the V_s gate happens to be holding. A run with this gate
    disabled (or with a threshold still miscalibrated for z_harm_a
    specifically) would sail through 862a's preflight exactly as it did
    (n_dacc_fires>0, peak_bias_magnitude>0 -- both stages fire on every
    tick regardless of V_s gating, since the OFF arm's bias/PE computation
    does not depend on the gate at all) and then burn the full 3-seed x
    2-arm budget on a manipulation check that silently never manipulated
    anything, exactly like 862a's confirmed failure mode
    (REE_assembly/evidence/planning/failure_autopsy_dacc-cluster-862a-870a_2026-08-03.md).
    Catches it in the time it takes to run 20 pilot steps instead.
    """


def _build_arm_config(env, gap4_arm: ArmSpec, *, vs_rollout_gating: bool):
    """475a-conditions gap4-operating config + full V_s invalidation circuit.

    Unchanged from 862a except VS_GATE_SNAPSHOT_REFRESH_THRESHOLD's value
    (see module constants above). The z_harm_a per-stream override is NOT
    set here -- it cannot be, because VsRolloutGateConfig (and its
    e1_threshold_per_stream / e2_threshold_per_stream dicts) does not exist
    until REEAgent(cfg) constructs it. See _apply_z_harm_a_gate_calibration()
    below, called on the agent immediately after construction.
    """
    cfg = build_config(env, gap4_arm, enable_affective_harm_stream=True)
    cfg.dacc_weight = 0.5
    cfg.dacc_interaction_weight = 0.5
    hc = cfg.hippocampal
    hc.use_per_stream_vs = True
    hc.use_event_segmenter = True
    hc.use_invalidation_trigger = True
    hc.use_anchor_sets = True
    hc.use_per_region_vs = True
    hc.use_staleness_accumulator = True
    hc.use_mech284_hysteresis = True
    hc.use_vs_commit_release = True
    hc.use_vs_rollout_gating = bool(vs_rollout_gating)
    hc.vs_gate_snapshot_refresh_threshold = VS_GATE_SNAPSHOT_REFRESH_THRESHOLD
    hc.vs_gate_e1_threshold = VS_GATE_E1_THRESHOLD
    hc.vs_gate_e2_threshold = VS_GATE_E2_THRESHOLD
    return cfg


def _apply_z_harm_a_gate_calibration(agent: REEAgent) -> None:
    """862b fix: wire the empirically-calibrated z_harm_a per-stream
    threshold onto the constructed gate's config.

    No-op when agent.vs_rollout_gate is None (the OFF arm, where
    use_vs_rollout_gating=False never instantiates a gate at all -- nothing
    to calibrate). See module docstring "WIRING NOTE" for why this cannot be
    done earlier, at _build_arm_config's REEConfig-construction stage.
    """
    gate = getattr(agent, "vs_rollout_gate", None)
    if gate is None:
        return
    gate.config.e1_threshold_per_stream = {"z_harm_a": Z_HARM_A_E1_THRESHOLD}
    gate.config.e2_threshold_per_stream = {"z_harm_a": Z_HARM_A_E2_THRESHOLD}


def _pe_and_bias_magnitude(agent: REEAgent) -> Tuple[Optional[float], Optional[float]]:
    """Unchanged from 862a. Reads agent._dacc_last_bundle / ._dacc_last_bias
    directly. Returns (None, None) when dACC did not fire this tick."""
    bundle = getattr(agent, "_dacc_last_bundle", None)
    if bundle is None:
        return None, None
    pe = bundle.get("pe")
    bias = getattr(agent, "_dacc_last_bias", None)
    if pe is None or bias is None:
        return None, None
    try:
        bias_mag = float(torch.as_tensor(bias).norm().item())
    except Exception:
        return None, None
    return float(pe), bias_mag


def _preflight_dacc_engagement_check(env_kwargs: Optional[Dict[str, Any]] = None) -> Tuple[int, float, int, int]:
    """Pre-flight smoke gate: assert dACC engages, its bias is nonzero, AND
    the z_harm_a-specific V_s gate actually holds at least once (862b's new
    check -- see PreflightVsGateZHarmANotHeld).

    Builds ARM_vs_on at a fixed pilot seed (unchanged scale from 862a: 3
    warmup episodes, 1 eval episode, 20 steps). Returns
    (n_dacc_fires, peak_bias_magnitude, held_e1_z_harm_a, held_e2_z_harm_a)
    on success (informational).
    """
    import random

    import numpy as np

    seed = PREFLIGHT_SEED
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    kwargs = dict(env_kwargs or ENV_FISHTANK_KWARGS)
    env = make_env(seed, kwargs)
    env._exq_env_kwargs = kwargs
    gap4_arm = ArmSpec(
        "gap4_475a_conditions",
        gap4_operating=True,
        use_gabaergic_decay=True,
        use_pag_freeze_gate=True,
        use_broadcast_override=True,
    )
    cfg = _build_arm_config(env, gap4_arm, vs_rollout_gating=True)
    agent = REEAgent(cfg)
    _apply_z_harm_a_gate_calibration(agent)

    warmup_train(
        agent,
        env,
        num_episodes=PREFLIGHT_WARMUP_EPISODES,
        steps_per_episode=PREFLIGHT_STEPS_PER_EPISODE,
        label="preflight_dacc_check",
        progress_total_episodes=PREFLIGHT_WARMUP_EPISODES + PREFLIGHT_EVAL_EPISODES,
    )

    n_dacc_fires = 0
    peak_bias_mag = 0.0

    def on_post_step(*, agent, **kwargs) -> None:
        nonlocal n_dacc_fires, peak_bias_mag
        pe, bias_mag = _pe_and_bias_magnitude(agent)
        if pe is not None and bias_mag is not None:
            n_dacc_fires += 1
            if bias_mag > peak_bias_mag:
                peak_bias_mag = bias_mag

    hooks = StepHooks(on_post_step=on_post_step)
    harness = StepHarness(agent, env, train_mode=False, hooks=hooks, seed=seed)
    agent.eval()

    for _ep in range(PREFLIGHT_EVAL_EPISODES):
        _, obs_dict = env.reset()
        agent.reset()
        harness.reset()
        for _ in range(PREFLIGHT_STEPS_PER_EPISODE):
            result = harness.step(obs_dict)
            obs_dict = result.next_obs_dict
            if result.done:
                break

    gate = getattr(agent, "vs_rollout_gate", None)
    gate_diag = dict(gate.get_diagnostics()) if gate is not None else {}
    held_e1_z_harm_a = int(gate_diag.get("vs_gate_held_e1_z_harm_a", 0))
    held_e2_z_harm_a = int(gate_diag.get("vs_gate_held_e2_z_harm_a", 0))

    print(
        f"[preflight] dACC engagement check: n_dacc_fires={n_dacc_fires} "
        f"peak_bias_magnitude={peak_bias_mag:.6f} "
        f"held_e1_z_harm_a={held_e1_z_harm_a} held_e2_z_harm_a={held_e2_z_harm_a} "
        f"over {PREFLIGHT_EVAL_EPISODES * PREFLIGHT_STEPS_PER_EPISODE} pilot eval steps",
        flush=True,
    )

    if n_dacc_fires == 0:
        raise PreflightDaccNotFiring(
            "Pre-flight pilot rollout observed ZERO dACC engagement "
            f"(n_dacc_fires=0 over {PREFLIGHT_EVAL_EPISODES * PREFLIGHT_STEPS_PER_EPISODE} "
            "eval steps, ARM_vs_on, seed="
            f"{seed}). agent._dacc_last_bundle never populated -- either "
            "self.dacc is None (use_dacc not set) or z_harm_a is None "
            "(affective harm stream not wired into the config). This is "
            "the exact V3-EXQ-475b failure mode. Refusing to proceed -- fix "
            "the config wiring before re-running."
        )
    if peak_bias_mag <= 1e-6:
        raise PreflightDaccBiasZero(
            "Pre-flight pilot rollout observed dACC FIRING "
            f"(n_dacc_fires={n_dacc_fires}) but the bias magnitude never "
            f"exceeded the 1e-6 floor (peak_bias_magnitude={peak_bias_mag:.6f}). "
            "DACCtoE3Adapter.forward emits torch.zeros_like(mode_ev) "
            "unconditionally whenever config.dacc_weight * drive_gain == "
            "0.0. This is the exact V3-EXQ-862 failure mode. Refusing to "
            "proceed -- fix the config wiring before re-running."
        )
    if held_e1_z_harm_a + held_e2_z_harm_a == 0:
        raise PreflightVsGateZHarmANotHeld(
            "Pre-flight pilot rollout observed dACC firing with nonzero "
            f"bias (n_dacc_fires={n_dacc_fires}, peak_bias_magnitude="
            f"{peak_bias_mag:.6f}) but the z_harm_a-specific V_s gate never "
            "held a stale snapshot (vs_gate_held_e1_z_harm_a="
            f"{held_e1_z_harm_a}, vs_gate_held_e2_z_harm_a={held_e2_z_harm_a}, "
            f"both 0, over {PREFLIGHT_EVAL_EPISODES * PREFLIGHT_STEPS_PER_EPISODE} "
            "eval steps, ARM_vs_on, seed="
            f"{seed}). This is the exact V3-EXQ-862a failure mode "
            "(REE_assembly/evidence/planning/failure_autopsy_dacc-cluster-862a-870a_2026-08-03.md): "
            "dACC's PE-computation and bias-consumption stages both fire "
            "regardless of whether the V_s gate ever actually substitutes a "
            "stale z_harm_a value, so the manipulation can silently never "
            "engage while both of 862a's preflight checks still pass. "
            "Z_HARM_A_E1_THRESHOLD/Z_HARM_A_E2_THRESHOLD "
            f"({Z_HARM_A_E1_THRESHOLD}/{Z_HARM_A_E2_THRESHOLD}) may need "
            "re-calibration against this harness's actual z_harm_a V_s "
            "distribution -- see module docstring 'THE FIX'. Refusing to "
            "proceed -- the full 3-seed x 2-arm design would burn its "
            "whole compute budget on a run where the ON/OFF arms are "
            "functionally identical for this DV, exactly like 862a."
        )
    return n_dacc_fires, peak_bias_mag, held_e1_z_harm_a, held_e2_z_harm_a


def run_seed_arm(
    seed: int,
    *,
    vs_rollout_gating: bool,
    arm_id: str,
    env_kwargs: Optional[Dict[str, Any]] = None,
    warmup_episodes: int = WARMUP_EPISODES_DEFAULT,
    eval_episodes: int = EVAL_EPISODES_DEFAULT,
    steps_per_episode: int = STEPS_PER_EPISODE_DEFAULT,
) -> Dict[str, Any]:
    import random

    import numpy as np

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    kwargs = dict(env_kwargs or ENV_FISHTANK_KWARGS)
    env = make_env(seed, kwargs)
    env._exq_env_kwargs = kwargs
    gap4_arm = ArmSpec(
        "gap4_475a_conditions",
        gap4_operating=True,
        use_gabaergic_decay=True,
        use_pag_freeze_gate=True,
        use_broadcast_override=True,
    )
    cfg = _build_arm_config(env, gap4_arm, vs_rollout_gating=vs_rollout_gating)
    agent = REEAgent(cfg)
    _apply_z_harm_a_gate_calibration(agent)
    label = f"seed={seed} arm={arm_id}"
    print(f"Seed {seed} Condition {arm_id}", flush=True)

    warmup_train(
        agent,
        env,
        num_episodes=warmup_episodes,
        steps_per_episode=steps_per_episode,
        label=label,
        progress_total_episodes=warmup_episodes + eval_episodes,
    )

    pe_series: List[float] = []
    bias_series: List[float] = []
    dacc_nonzero_steps = 0
    total_eval_steps = 0

    def on_post_step(*, agent, **kwargs) -> None:
        nonlocal dacc_nonzero_steps, total_eval_steps
        total_eval_steps += 1
        pe, bias_mag = _pe_and_bias_magnitude(agent)
        if pe is not None and bias_mag is not None:
            pe_series.append(pe)
            bias_series.append(bias_mag)
            if bias_mag > 1e-6:
                dacc_nonzero_steps += 1

    hooks = StepHooks(on_post_step=on_post_step)
    harness = StepHarness(agent, env, train_mode=False, hooks=hooks, seed=seed)
    agent.eval()

    for _ep in range(eval_episodes):
        _, obs_dict = env.reset()
        agent.reset()
        harness.reset()
        for _ in range(steps_per_episode):
            result = harness.step(obs_dict)
            obs_dict = result.next_obs_dict
            if result.done:
                break
        if (_ep + 1) == eval_episodes:
            print(
                f"  [train] {label} ep {warmup_episodes + _ep + 1}/{warmup_episodes + eval_episodes}",
                flush=True,
            )

    vs_gate_total_held = 0
    vs_gate_diag: Dict[str, Any] = {}
    gate = getattr(agent, "vs_rollout_gate", None)
    if gate is not None:
        vs_gate_diag = dict(gate.get_diagnostics())
        vs_gate_total_held = int(vs_gate_diag.get("vs_gate_total_held_e1", 0)) + int(
            vs_gate_diag.get("vs_gate_total_held_e2", 0)
        )
    vs_gate_held_z_harm_a = int(vs_gate_diag.get("vs_gate_held_e1_z_harm_a", 0)) + int(
        vs_gate_diag.get("vs_gate_held_e2_z_harm_a", 0)
    )

    rho = spearman(pe_series, bias_series)

    n_dacc_fires = len(pe_series)
    cell_ok = n_dacc_fires > 0
    print(f"verdict: {'PASS' if cell_ok else 'FAIL'}", flush=True)

    return {
        "seed": int(seed),
        "arm": arm_id,
        "vs_rollout_gating": bool(vs_rollout_gating),
        "total_eval_steps": int(total_eval_steps),
        "n_dacc_fires": n_dacc_fires,
        "dacc_bias_nonzero_steps": int(dacc_nonzero_steps),
        "vs_gate_total_held": int(vs_gate_total_held),
        "vs_gate_held_z_harm_a": int(vs_gate_held_z_harm_a),
        "vs_gate_diagnostics": vs_gate_diag,
        "pe_series": pe_series,
        "bias_magnitude_series": bias_series,
        "rho_pe_vs_bias_magnitude": rho,
    }


def evaluate_q040c(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    on_rows = [r for r in rows if r["arm"] == VS_ON_ARM]
    off_rows = [r for r in rows if r["arm"] == VS_OFF_ARM]

    # P1: aggregate gate-firing precondition (ON arm only), RETAINED from
    # 862a for continuity but no longer gates the run -- see module
    # docstring "HARD PRECONDITIONS". This is exactly the check that
    # false-positived in 862a (passed while z_harm_a itself never held).
    on_gate_fired_seeds = sum(1 for r in on_rows if r["vs_gate_total_held"] > 0)
    p1_pass = on_gate_fired_seeds >= SEEDS_PASS_MIN

    # P1' (NEW, 862b): z_harm_a-SPECIFIC gate-firing precondition. This is
    # the operative precondition -- see PreflightVsGateZHarmANotHeld and
    # module docstring "HARD PRECONDITIONS".
    on_z_harm_a_gate_fired_seeds = sum(1 for r in on_rows if r["vs_gate_held_z_harm_a"] > 0)
    p1_z_harm_a_pass = on_z_harm_a_gate_fired_seeds >= SEEDS_PASS_MIN

    # P2: dACC engagement precondition (both arms). Unchanged from 862a.
    on_dacc_fired_seeds = sum(1 for r in on_rows if r["dacc_bias_nonzero_steps"] > 0)
    off_dacc_fired_seeds = sum(1 for r in off_rows if r["dacc_bias_nonzero_steps"] > 0)
    p2_pass = on_dacc_fired_seeds >= SEEDS_PASS_MIN and off_dacc_fired_seeds >= SEEDS_PASS_MIN

    # Gating precondition uses P1' (z_harm_a-specific), NOT the aggregate P1
    # -- see module docstring "HARD PRECONDITIONS" for why the aggregate is
    # informational-only after 862a's false-positive.
    preconditions_met = bool(p1_z_harm_a_pass and p2_pass)

    on_valid = [r["rho_pe_vs_bias_magnitude"] for r in on_rows if r["rho_pe_vs_bias_magnitude"] is not None]
    off_valid = [r["rho_pe_vs_bias_magnitude"] for r in off_rows if r["rho_pe_vs_bias_magnitude"] is not None]
    on_detect_seeds = sum(1 for rho in on_valid if abs(rho) >= CORR_DETECT_THRESH)
    off_null_seeds = sum(1 for rho in off_valid if abs(rho) < CORR_NULL_THRESH)

    c3_on_detects = len(on_valid) >= SEEDS_PASS_MIN and on_detect_seeds >= SEEDS_PASS_MIN
    c4_off_null = len(off_valid) >= SEEDS_PASS_MIN and off_null_seeds >= SEEDS_PASS_MIN

    verdict_pass = bool(preconditions_met and c3_on_detects and c4_off_null)

    if not preconditions_met:
        self_route = "substrate_not_ready_requeue"
    else:
        self_route = None

    degeneracy = check_degeneracy(
        {
            "dacc_bias_magnitude_series": {
                "groups": [r["bias_magnitude_series"] for r in rows if r["bias_magnitude_series"]],
            },
        }
    )

    return {
        "pass": verdict_pass,
        "preconditions_met": preconditions_met,
        "self_route": self_route,
        "p1_gate_firing_pass": bool(p1_pass),
        "p1_on_gate_fired_seeds": int(on_gate_fired_seeds),
        "p1_z_harm_a_gate_firing_pass": bool(p1_z_harm_a_pass),
        "p1_on_z_harm_a_gate_fired_seeds": int(on_z_harm_a_gate_fired_seeds),
        "p2_dacc_engagement_pass": bool(p2_pass),
        "p2_on_dacc_fired_seeds": int(on_dacc_fired_seeds),
        "p2_off_dacc_fired_seeds": int(off_dacc_fired_seeds),
        "on_valid_rho_seeds": len(on_valid),
        "off_valid_rho_seeds": len(off_valid),
        "on_detect_seeds": int(on_detect_seeds),
        "off_null_seeds": int(off_null_seeds),
        "c3_on_detects_correlation": bool(c3_on_detects),
        "c4_off_correlation_null": bool(c4_off_null),
        "corr_detect_thresh": CORR_DETECT_THRESH,
        "corr_null_thresh": CORR_NULL_THRESH,
        "seeds_pass_min": SEEDS_PASS_MIN,
        **degeneracy,
    }


def _utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def main(dry_run: bool = False) -> Tuple[str, Path] | int:
    seeds = [SEEDS_DEFAULT[0]] if dry_run else SEEDS_DEFAULT
    warmup = 8 if dry_run else WARMUP_EPISODES_DEFAULT
    eval_eps = 2 if dry_run else EVAL_EPISODES_DEFAULT
    steps = 40 if dry_run else STEPS_PER_EPISODE_DEFAULT

    # Pre-flight smoke gate -- runs BEFORE the full seeds x arms design,
    # every invocation (dry-run and real). Raises uncaught on failure (fast
    # ERROR, not a full-budget FAIL). See module docstring,
    # PreflightDaccNotFiring, PreflightDaccBiasZero (862a),
    # PreflightVsGateZHarmANotHeld (862b's new check).
    n_preflight_fires, preflight_peak_bias_mag, preflight_held_e1_z_harm_a, preflight_held_e2_z_harm_a = (
        _preflight_dacc_engagement_check(env_kwargs=ENV_FISHTANK_KWARGS)
    )

    rows: List[Dict[str, Any]] = []
    t0 = time.time()
    for seed in seeds:
        for arm_id, vs_on in ((VS_OFF_ARM, False), (VS_ON_ARM, True)):
            rows.append(
                run_seed_arm(
                    seed,
                    vs_rollout_gating=vs_on,
                    arm_id=arm_id,
                    env_kwargs=ENV_FISHTANK_KWARGS,
                    warmup_episodes=warmup,
                    eval_episodes=eval_eps,
                    steps_per_episode=steps,
                )
            )

    acceptance = evaluate_q040c(rows)
    acceptance["preflight_n_dacc_fires"] = int(n_preflight_fires)
    acceptance["preflight_peak_bias_magnitude"] = float(preflight_peak_bias_mag)
    acceptance["preflight_held_e1_z_harm_a"] = int(preflight_held_e1_z_harm_a)
    acceptance["preflight_held_e2_z_harm_a"] = int(preflight_held_e2_z_harm_a)
    elapsed = time.time() - t0

    if acceptance["self_route"] is not None:
        outcome = "FAIL"
        evidence_direction = "non_contributory"
    else:
        outcome = "PASS" if acceptance["pass"] else "FAIL"
        evidence_direction = "supports" if outcome == "PASS" else "weakens"

    if dry_run:
        print(
            f"[{EXPERIMENT_TYPE}] dry-run outcome={outcome} "
            f"self_route={acceptance['self_route']} pass={acceptance['pass']} "
            f"n_dacc_fires_per_cell={[r['n_dacc_fires'] for r in rows]} "
            f"vs_gate_held_z_harm_a_per_cell={[r['vs_gate_held_z_harm_a'] for r in rows]}",
            flush=True,
        )
        return 0

    per_run_rows = rows

    run_id = f"{EXPERIMENT_TYPE}_{_utc_compact()}_v3"
    out_dir = (
        REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.json"
    manifest = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": _utc_compact(),
        "outcome": outcome,
        "evidence_direction": evidence_direction,
        "acceptance": acceptance,
        "per_run": per_run_rows,
        "supersedes": SUPERSEDES_RUN_ID,
        "supersedes_queue_id": SUPERSEDES_QUEUE_ID,
        "elapsed_seconds": elapsed,
    }
    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=False,
        config=manifest.get("config"),
        seeds=list(seeds),
        script_path=Path(__file__),
    )
    return outcome, out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = main(dry_run=args.dry_run)
    if result == 0:
        sys.exit(0)
    outcome, out_path = result
    emit_outcome(outcome=outcome, manifest_path=out_path, dry_run=bool(args.dry_run))
    sys.exit(0)
