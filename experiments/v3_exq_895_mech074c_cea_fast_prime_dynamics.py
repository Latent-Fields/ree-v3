#!/opt/local/bin/python3
"""V3-EXQ-895: MECH-074c wall-INDEPENDENT functional-signature confirming test.

Confirms provisional claim MECH-074c -- "CeA analogue emits a fast subcortical
priming signal (fast_prime) on low-frequency z_harm_a with an overridable decay
window, distinct from MECH-046's cortical mode-prior write" -- which carries ZERO
indexed EVIDENCE-class experiment despite its substrate being built + IMPLEMENTED
in ree-v3 (SD-035, 2026-04-21). Surfaced by the GOV-CONFIRM-1 evidence-confirmer
detector as IGW workset item IGW-20260807-240 (stable_hash 133da2eb256a).

  Substrate under test:
    ree_core/amygdala/cea.py -- CeAAnalog.tick(z_harm_a) emits fast_prime, a
    scalar candidate-prior pulse, when |LowFreq(z_harm_a)| = ||z_harm_a||_1 / n
    crosses fast_route_threshold; the pulse then decays with half-life
    fast_prime_decay_tau_steps unless cortical_confirmation arrives inside the
    fast_prime_override_window_steps window. agent.py:6528-6536 injects it into
    the SalienceCoordinator via update_signal("cea_fast_prime", value),
    registered on salience_weights["cea_fast_prime"] = 0.5.

  GOV-REUSE-1 (checked 2026-08-08): the ONLY prior run tagging MECH-074c,
    V3-EXQ-473 (v3_exq_473_sd035_cea_mode_prior), is
    experiment_purpose=diagnostic -- excluded from governance confidence, which
    is WHY MECH-074c scores exp_conf 0.0 -- and its substrate_hash is None
    (pre-2026-07-12 recording standard, compatibility UNVERIFIABLE). On the
    fast_prime channel it recorded exactly four numbers: a single fire point
    (fast_prime 0.2999 at |z_harm_a| = 0.9), and a 2-point decay/hold pair
    (c5a 0.8 -> 0.2 over 8 steps; c5b 0.8 -> 0.8 with confirmation). It recorded
    NO onset latency (the claim's first falsifier leg), NO fitted decay tau
    against the claim's pre-registered [3, 5] range (a 2-point threshold test is
    not a tau), and NO selectivity probe at all (the claim's named failure
    signature). No per-tick trace exists, so none of the three is derivable
    post-hoc. Not recoverable -> run.

WHY WALL-INDEPENDENT (the design contract):
  The V3 program is bottlenecked on the "competence wall" -- the integrated agent
  is not behaviourally competent enough to emit committed behaviour worth
  measuring. ANY experiment with a committed-behaviour DV is wall-bound. The
  gated DVs here are FUNCTIONAL-SIGNATURE / TIMING readouts on the fast_prime
  scalar the real CeAAnalog emits, read out action-free: no agent policy, no
  training, no behavioural outcome. They pass or fail independent of the wall
  (precedent: V3-EXQ-762 PASSED confirming the sibling claim MECH-046 on this
  exact substrate the same way; also V3-EXQ-757, V3-EXQ-455/447/448).

===============================================================================
WHAT THIS EXPERIMENT DELIBERATELY DOES NOT TEST -- read before citing it
===============================================================================
MECH-074c's falsifier names THREE legs. This run gates on two of them and
explicitly does NOT gate on the third:

  leg 1  fast_prime emits within 1-2 steps of threshold crossing   -> C1  GATED
  leg 2  decays under cortical non-confirmation within the window  -> C2/C3 GATED
  leg 3  does NOT fire on arousing-but-neutral stimuli             -> NOT GATED

Leg 3 is NOT DESIGNABLE as a CeA-module manipulation, by arithmetic, and forcing
it would produce exactly the pre-determined-delta artifact the DV-symmetry rule
exists to prevent (failure_autopsy_V3-EXQ-604c section 3). CeAAnalog.tick() is a
PURE FUNCTION of the scalar ||z_harm_a||_1 / n (cea.py:309-331) -- it is
therefore INVARIANT under every transformation of its input that preserves that
scalar. So:
  - drive it with a magnitude-MATCHED non-harm stream -> fires IDENTICALLY, at
    every seed, on every substrate, by arithmetic. A guaranteed "fails
    selectivity" reading that measures nothing.
  - drive it with a magnitude-MISMATCHED non-harm stream -> the outcome is fixed
    by the magnitude alone, again independent of whether the claim is true.
Either way the delta is fixed before the run. Selectivity is therefore a
question about the UPSTREAM AffectiveHarmEncoder -- does ||z_harm_a|| track
harm-affective valence rather than generic arousal? -- not about CeAAnalog.

That upstream question IS measurable at representation level, so this run
MEASURES AND RECORDS it (Block C) as a NON-GATING diagnostic with its own
self-route, and reports whether the substrate is ready for a leg-3 confirmer.
Block C is deliberately kept OUT of interpretation.preconditions[] so it cannot
vacate the two clean gated legs (the V3-EXQ-785 whole-run-AND defect); it is
carried at top level under interpretation.selectivity_leg with an explicit
preconditions_scope_note.

Also out of scope: the functional_restatement's fast:slow ratio (~5:1 vs the
cortical AIC/dACC comparator). That needs SD-032c AIC comparator wiring on the
same tick base and is a named successor, not a silent omission.

===============================================================================
METHOD (representation/timing level; NO training, NO phased training, NO policy)
===============================================================================
BLOCKS A + B (GATED, action-free, module-level). Instantiate the REAL CeAAnalog
built exactly as agent.py builds it when use_cea_analog=True (CeAConfig mapped
field-for-field from a REEConfig with use_amygdala_analog=True /
use_cea_analog=True -- honouring the V3-EXQ-688 vacuous-null ARMING CAVEAT: the
CeA is explicitly ON and its z_harm_a input path is explicitly populated, never
left to a default). Drive it per trial with a controlled TEMPORAL protocol:

    N_QUIET_PRE ticks   sub-threshold  lf ~ U(QUIET_LO, QUIET_HI)
    1 onset tick        supra-threshold lf ~ U(FIRE_LO, FIRE_HI)
    N_POST ticks        sub-threshold, cortical_confirmation per ARM

  ARM_CONF_ABSENT   cortical_confirmation=None -- matches agent.py:4514, which
                    hardwires None, so this is the LIVE in-agent regime.
  ARM_CONF_PRESENT  cortical_confirmation=1.0 inside the override window.

  The two arms receive BIT-IDENTICAL drive at matched (seed, trial) -- asserted
  by precondition P3 -- so the C4 contrast is attributable to the cortical
  confirmation input alone.

  Per-seed, per-trial the drive MAGNITUDES are randomised over pre-registered
  ranges. This is load-bearing, not decoration: the CeA is deterministic, so a
  seed sweep at a FIXED magnitude would be bit-identical across seeds and the
  whole seed dimension would be degenerate. Randomised peaks make the measured
  tau an invariance ACROSS genuinely different pulses (the claim's actual
  assertion) rather than one number re-reported N times.

BLOCK C (NON-GATING, agent-driven). Build a real REEAgent + CausalGridWorldV2
and step it through the canonical StepHarness (which is the only path that
supplies obs_harm_a and therefore the only path on which z_harm_a -- and hence
the CeA -- exists at all; agent.act()/sense_flat() pass obs_harm_a=None, so the
CeA never ticks there). Harvest per tick: the CeA's own gate statistic
low_freq_magnitude, fast_prime, the encoder INPUT ||harm_obs_a||_1/n, a generic
activation covariate ||z_world||_1/n, and the env ground truth
info["hazard_field_at_agent"]. Report AUROC of each against a hazard-proximity
label, plus the embodied fire count.

DVs / PRE-REGISTERED PASS (thresholds are module constants below, not post-hoc):
  C1_onset_latency (LOAD-BEARING): across every trial in both arms, the number
      of sim steps from the first tick whose low_freq_magnitude exceeds
      fast_route_threshold to the first tick with fast_prime != 0 is
      <= ONSET_MAX_STEPS. This is leg 1 verbatim ("emits within 1-2 steps").
  C2_decay_tau (LOAD-BEARING): in ARM_CONF_ABSENT, the half-life fitted from the
      post-onset fast_prime trace lies in [TAU_MIN, TAU_MAX] for EVERY trial --
      the claim's own "tau_decay in [3, 5] sim steps".
  C3_decays_within_override_window (LOAD-BEARING): in ARM_CONF_ABSENT,
      fast_prime at the END OF THE OVERRIDE WINDOW is <= WINDOW_MAX_RETAINED of
      its peak, every trial -- leg 2's "decays toward baseline within the
      override window". "End of the window" is read off the module's OWN
      reported override_window_remaining (the last tick at which it is still
      > 0), NOT computed as onset + fast_prime_override_window_steps: the
      counter is decremented at the TOP of a non-firing tick (cea.py:371-372),
      so at onset+window it already reads 0 and the pulse has begun decaying
      even under confirmation. Measuring there would score the first
      POST-window tick as if it were inside the window. Both arms use the same
      definition, so C3 and C4 remain directly comparable.
  C4_overridable (LOAD-BEARING): in ARM_CONF_PRESENT the same retained fraction
      is >= HOLD_MIN_RETAINED every trial, AND the arm means separate by
      >= OVERRIDE_SEPARATION_MIN. This is the "OVERRIDABLE decay window" in the
      claim title, and it is what makes C3 a response to non-confirmation rather
      than an unconditional countdown.
  C5_bounded: over a magnitude sweep spanning into the saturation region,
      max |fast_prime| <= mode_prior_log_odds_max + BOUND_EPS -- the claim's
      "override discipline" (CeA never over-rules cortex via the fast path).
  PASS = C1 AND C2 AND C3 AND C4 AND C5 (a plain AND; recorded explicitly as
  combination_rule so a reader need not open this docstring).

DV-SYMMETRY DECLARATION (mandatory; one line per arm; the 604c net):
  ARM_CONF_ABSENT / ARM_CONF_PRESENT, C1. DV = onset latency in ticks. Symmetry
    group = temporal translation of the drive onset. The manipulation is a step
    change in drive magnitude AT a designated tick, which is NOT invariant under
    temporal translation -- translating the onset translates the measured
    latency 1:1, and a substrate gating on a smoothed/EMA'd or buffered
    projection would return latency > 0 here.
  ARM_CONF_ABSENT, C2/C3. DV = fitted half-life and retained fraction, both
    RATIOS of successive fast_prime values. Symmetry group = uniform positive
    rescaling of fast_prime (both DVs are scale-invariant by construction). The
    manipulation (withholding cortical confirmation) is NOT invariant under it:
    it changes the decay RATE, not the scale. That the DV is scale-invariant is
    the point -- it is why randomising the peak magnitude cannot manufacture the
    result.
  ARM_CONF_PRESENT vs ARM_CONF_ABSENT, C4. DV = retained fraction. Symmetry
    group = uniform rescaling (cancels in the ratio) and permutation of ticks.
    The manipulation (cortical_confirmation 0 vs 1) is invariant under NEITHER:
    it is a temporal-order-dependent modification of the decay recursion
    (cea.py:396-415). Non-invariance verified numerically before authoring.
  C5 sweep. DV = max |fast_prime| over a magnitude sweep. The sweep is
    pre-registered to reach into the clip region, and C5's non-degeneracy flag
    records whether the UNCLIPPED value actually exceeded the cap -- if it did
    not, the bound was never exercised and C5 passed vacuously.

NON-VACUITY / READINESS PRECONDITIONS (same statistics the criteria route on;
below-floor self-routes to substrate_not_ready_requeue, never to a verdict):
  P1a_drive_crosses_threshold  worst-cell MIN of (lf_onset - fast_route_threshold)
      over all trials, floor 0.0 exclusive. Without a crossing there is no onset
      to time and C1 is undefined. Positive control: the onset magnitude is
      drawn from a pre-registered supra-threshold range.
  P1b_quiet_stays_subthreshold worst-cell MAX of lf_quiet over all trials,
      CEILING at fast_route_threshold (direction upper). A quiet phase that
      re-fires would corrupt the decay measurement C2/C3 read.
  P2_fast_prime_peak_supra_floor  worst-cell MIN of the peak fast_prime over all
      trials, floor FP_PEAK_FLOOR. This is the SAME statistic C2 and C3
      normalise by: a half-life and a retained fraction are both undefined on a
      zero pulse. Positive control: supra-threshold drive on an explicitly
      armed CeA.
  P3_arms_receive_identical_drive  worst-cell MAX abs difference between the two
      arms' low_freq_magnitude traces at matched (seed, trial), CEILING at
      DRIVE_MATCH_EPS. Makes C4's separation attributable to
      cortical_confirmation alone.
  Every precondition is SATISFIABLE from the values this script pre-registers
  (checked on paper: QUIET_HI < fast_route_threshold < FIRE_LO by construction,
  and FIRE_LO gives a peak of 0.75*(FIRE_LO - 0.5) > FP_PEAK_FLOOR).
  Every precondition applies to the module arms ONLY, which is every arm this
  run gates on -- so there is no regime-conditioning gap here and no arm is
  scoped out. Block C carries no precondition by design (see above).

experiment_purpose=evidence. This is the GOV-CONFIRM-1 confirmer: V3-EXQ-473 was
diagnostic, which is precisely why MECH-074c has no indexed experimental
evidence.

See REE_assembly/docs/architecture/sd_035_amygdala_analog.md
See REE_assembly/docs/claims/claims.yaml (- id: MECH-074c)
"""

import argparse
import json
import math
import random
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from ree_core.agent import REEAgent  # noqa: E402
from ree_core.amygdala import CeAAnalog, CeAConfig  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402

from experiments._harness import StepHarness  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402

from experiment_protocol import emit_outcome  # noqa: E402


EXPERIMENT_TYPE = "v3_exq_895_mech074c_cea_fast_prime_dynamics"
CLAIM_IDS = ["MECH-074c"]
EXPERIMENT_PURPOSE = "evidence"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"

# -- Seeds / grid ------------------------------------------------------------
# Seed 44 is deliberately absent (recurring reef-config early-death instability,
# CLAUDE.md); 45 is the sanctioned substitute.
SEEDS = [42, 7, 13, 23, 45]
ARM_CONF_ABSENT = "ARM_CONF_ABSENT"
ARM_CONF_PRESENT = "ARM_CONF_PRESENT"
BLOCK_C = "SELECTIVITY_PROBE"
CONDITIONS = [ARM_CONF_ABSENT, ARM_CONF_PRESENT, BLOCK_C]

# One "episode" for progress-instrumentation purposes = one drive trial in the
# module arms, and one env episode in the selectivity probe. Kept equal so the
# queue entry's episodes_per_run is a single well-defined number.
TRIALS_PER_RUN = 12

# -- Drive protocol (Blocks A + B) -------------------------------------------
N_QUIET_PRE = 5          # sub-threshold ticks before onset
N_POST = 14              # sub-threshold ticks after onset (> override window 8)
QUIET_LO, QUIET_HI = 0.05, 0.40    # both < fast_route_threshold (0.5)
FIRE_LO, FIRE_HI = 0.70, 2.20      # both > fast_route_threshold; spans the clip

# -- Pre-registered acceptance thresholds ------------------------------------
ONSET_MAX_STEPS = 2              # claim: "within 1-2 sim steps"
TAU_MIN, TAU_MAX = 3.0, 5.0      # claim: "tau_decay in [3, 5] sim steps"
WINDOW_MAX_RETAINED = 0.50       # claim: decays toward baseline within window
HOLD_MIN_RETAINED = 0.90         # claim: confirmation holds the pulse
OVERRIDE_SEPARATION_MIN = 0.40   # arm means must separate
BOUND_EPS = 1e-6

# -- Pre-registered readiness floors -----------------------------------------
FP_PEAK_FLOOR = 0.05             # a tau/retained-fraction is undefined below this
DRIVE_MATCH_EPS = 1e-9           # arms must receive bit-identical drive

# -- C5 magnitude sweep ------------------------------------------------------
SWEEP_LO, SWEEP_HI, SWEEP_N = 0.0, 2.5, 51

# -- Block C (selectivity probe, NON-GATING) ---------------------------------
PROBE_STEPS_PER_EP = 60
AUROC_READY_FLOOR = 0.60         # reported only; Block C gates nothing


# ===========================================================================
# Helpers
# ===========================================================================

def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _l1_per_dim(t: torch.Tensor) -> float:
    """||t||_1 / n -- the exact statistic CeAAnalog's fast-route gate reads."""
    flat = t.detach().flatten()
    n = int(flat.shape[0])
    if n == 0:
        return 0.0
    return float(torch.linalg.vector_norm(flat, ord=1).item()) / float(n)


def _z_at(mag: float, dim: int) -> torch.Tensor:
    """A z_harm_a vector whose ||z||_1 / n is exactly `mag`."""
    return torch.full((dim,), float(mag), dtype=torch.float32)


def _worst_cell(rows: List[Dict[str, Any]], key: str, mode: str) -> Tuple[float, Any]:
    """Return (extremum, offending_cell_id) -- never a mean.

    A precondition whose `met` quantifies over cells (all/any) must report the
    WORST cell, or the indexer's recompute passes an out-of-band cell that an
    in-band mean masked (the V3-EXQ-779b defect).
    """
    vals = [(float(r[key]), r.get("cell_id")) for r in rows if r.get(key) is not None]
    if not vals:
        return float("nan"), None
    return min(vals) if mode == "min" else max(vals)


def _auroc(scores: List[float], labels: List[bool]) -> float:
    pairs = [(s, l) for s, l in zip(scores, labels)
             if not (isinstance(s, float) and math.isnan(s))]
    pos = [s for s, l in pairs if l]
    neg = [s for s, l in pairs if not l]
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else (0.5 if p == n else 0.0)
    return wins / float(len(pos) * len(neg))


def _fit_half_life(trace: List[float], min_value: float = 1e-4) -> float:
    """Half-life in sim steps, fitted log-linearly over the decaying prefix.

    Uses only ticks whose value stays above `min_value` (below that the module
    zero-snaps at 1e-6 and the ratio is meaningless). Returns inf for a trace
    that does not decay at all (the confirmed-hold arm), which is the correct
    reading rather than a divide-by-zero.
    """
    usable = []
    for v in trace:
        if abs(v) <= min_value:
            break
        usable.append(abs(v))
    if len(usable) < 3:
        return float("nan")
    v0 = usable[0]
    vn = usable[-1]
    k = len(usable) - 1
    if v0 <= 0.0 or vn <= 0.0:
        return float("nan")
    if vn >= v0:
        return float("inf")   # held, not decaying
    return float(k * math.log(2.0) / math.log(v0 / vn))


def _cea_config_from_ree(cfg: REEConfig) -> CeAConfig:
    """Build CeAConfig exactly as agent.py:2091-2101 does when use_cea_analog."""
    return CeAConfig(
        fast_route_threshold=cfg.cea_fast_route_threshold,
        fast_route_input_is_lowfreq=cfg.cea_fast_route_input_is_lowfreq,
        mode_prior_log_odds_max=cfg.cea_mode_prior_log_odds_max,
        mode_prior_gain=cfg.cea_mode_prior_gain,
        pre_softmax_additive=cfg.cea_pre_softmax_additive,
        fast_prime_amplitude=cfg.cea_fast_prime_amplitude,
        fast_prime_decay_tau_steps=cfg.cea_fast_prime_decay_tau_steps,
        fast_prime_override_window_steps=cfg.cea_fast_prime_override_window_steps,
        cortical_confirmation_weight=cfg.cea_cortical_confirmation_weight,
    )


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed,
        size=10,
        num_hazards=2,
        num_resources=3,
        hazard_harm=0.04,
        proximity_harm_scale=0.12,
        proximity_benefit_scale=0.18,
        hazard_field_decay=0.5,
        energy_decay=0.005,
        use_proxy_fields=True,
        resource_respawn_on_consume=True,
        harm_history_len=10,
    )


def _make_config(env: CausalGridWorldV2) -> REEConfig:
    """ARMING CAVEAT (V3-EXQ-688): the CeA is EXPLICITLY on and its z_harm_a
    input path is EXPLICITLY populated -- never left to a default. Note
    use_affective_harm_stream must be set here: the from_dims branch does not
    enable it unconditionally (the V3-EXQ-475b branch-asymmetry class)."""
    return REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=32,
        alpha_world=0.9,
        alpha_self=0.3,
        use_harm_stream=True,
        harm_obs_dim=51,
        use_affective_harm_stream=True,
        harm_obs_a_dim=50,
        harm_history_len=10,
        z_harm_a_dim=16,
        use_amygdala_analog=True,
        use_bla_analog=False,      # isolate the CeA path
        use_cea_analog=True,
        use_salience_coordinator=True,
        salience_apply_to_dacc_bias=False,   # observer mode
    )


def _config_slice(cfg: REEConfig, cea_cfg: CeAConfig) -> Dict[str, Any]:
    """Everything the drive computation reads. No acceptance thresholds."""
    return {
        "z_harm_a_dim": int(cfg.latent.z_harm_a_dim),
        "cea_fast_route_threshold": float(cea_cfg.fast_route_threshold),
        "cea_fast_route_input_is_lowfreq": bool(cea_cfg.fast_route_input_is_lowfreq),
        "cea_mode_prior_log_odds_max": float(cea_cfg.mode_prior_log_odds_max),
        "cea_mode_prior_gain": float(cea_cfg.mode_prior_gain),
        "cea_fast_prime_amplitude": float(cea_cfg.fast_prime_amplitude),
        "cea_fast_prime_decay_tau_steps": int(cea_cfg.fast_prime_decay_tau_steps),
        "cea_fast_prime_override_window_steps": int(
            cea_cfg.fast_prime_override_window_steps),
        "cea_cortical_confirmation_weight": float(cea_cfg.cortical_confirmation_weight),
        "n_quiet_pre": N_QUIET_PRE,
        "n_post": N_POST,
        "quiet_range": [QUIET_LO, QUIET_HI],
        "fire_range": [FIRE_LO, FIRE_HI],
        "trials_per_run": TRIALS_PER_RUN,
    }


# ===========================================================================
# Blocks A + B -- gated, action-free module drive
# ===========================================================================

def _trial_drive(seed: int, trial: int) -> Tuple[List[float], float, float]:
    """Deterministic per-(seed, trial) drive magnitudes.

    Derived from a seed-and-trial-keyed Random so the two ARMS receive
    BIT-IDENTICAL drive at matched (seed, trial) -- the P3 precondition.
    """
    rng = random.Random((seed * 100003) ^ (trial * 7919))
    lf_quiet = rng.uniform(QUIET_LO, QUIET_HI)
    lf_fire = rng.uniform(FIRE_LO, FIRE_HI)
    mags = [lf_quiet] * N_QUIET_PRE + [lf_fire] + [lf_quiet] * N_POST
    return mags, lf_quiet, lf_fire


def _run_module_trial(cea_cfg: CeAConfig, dim: int, seed: int, trial: int,
                      confirm: Optional[float]) -> Dict[str, Any]:
    """One drive trial through a fresh CeAAnalog. Returns the per-trial row."""
    cea = CeAAnalog(cea_cfg)
    mags, lf_quiet, lf_fire = _trial_drive(seed, trial)
    thr = float(cea_cfg.fast_route_threshold)
    window = int(cea_cfg.fast_prime_override_window_steps)

    lf_trace: List[float] = []
    fp_trace: List[float] = []
    fire_trace: List[bool] = []
    window_trace: List[int] = []
    for idx, mag in enumerate(mags):
        # Confirmation is supplied only AFTER onset and only inside the window,
        # which is what "cortical confirmation arriving inside the override
        # window" means. Before onset there is no pulse to confirm.
        cc = None
        if confirm is not None and N_QUIET_PRE < idx <= N_QUIET_PRE + window:
            cc = float(confirm)
        out = cea.tick(_z_at(mag, dim), cortical_confirmation=cc)
        lf_trace.append(float(out.low_freq_magnitude))
        fp_trace.append(float(out.fast_prime))
        fire_trace.append(bool(out.urgency_fire))
        window_trace.append(int(out.override_window_remaining))

    # Onset latency: measured entirely from the module's OWN reported values --
    # first tick whose reported low_freq_magnitude exceeds threshold, to the
    # first tick with a non-zero fast_prime.
    cross_idx = next((i for i, v in enumerate(lf_trace) if v > thr), None)
    emit_idx = next((i for i, v in enumerate(fp_trace) if v != 0.0), None)
    onset_latency = (
        int(emit_idx - cross_idx)
        if (cross_idx is not None and emit_idx is not None and emit_idx >= cross_idx)
        else None
    )

    peak_fp = max((abs(v) for v in fp_trace), default=0.0)
    decay_trace = fp_trace[N_QUIET_PRE + 1:]          # ticks strictly after onset
    half_life = _fit_half_life(decay_trace)

    # End of the override window = the LAST tick at which the module itself
    # still reports the window open (override_window_remaining > 0). Read off
    # the module's own diagnostic rather than computed as onset+window: the
    # counter is decremented at the TOP of a non-firing tick (cea.py:371-372),
    # so at onset+window it already reads 0 and the pulse has begun decaying
    # even under confirmation. Measuring there would score the first
    # post-window tick as if it were inside the window.
    end_idx = next((i for i in range(len(window_trace) - 1, N_QUIET_PRE, -1)
                    if window_trace[i] > 0), None)
    if end_idx is None:
        end_idx = min(N_QUIET_PRE + window, len(fp_trace) - 1)
    fp_at_window_end = abs(fp_trace[end_idx])
    retained = (fp_at_window_end / peak_fp) if peak_fp > 0.0 else float("nan")

    return {
        "cell_id": f"seed{seed}_trial{trial}",
        "seed": int(seed),
        "trial": int(trial),
        "lf_quiet": float(lf_quiet),
        "lf_fire": float(lf_fire),
        "lf_fire_minus_threshold": float(lf_fire - thr),
        "n_fires": int(sum(fire_trace)),
        "onset_latency_steps": onset_latency,
        "peak_fast_prime": float(peak_fp),
        "fitted_half_life_steps": float(half_life),
        "window_end_tick_index": int(end_idx),
        "fast_prime_at_window_end": float(fp_at_window_end),
        "retained_fraction": float(retained),
        "lf_trace": [round(v, 10) for v in lf_trace],
        "fast_prime_trace": [round(v, 10) for v in fp_trace],
        "override_window_remaining_trace": window_trace,
    }


def _run_module_arm(cea_cfg: CeAConfig, dim: int, seed: int, arm: str,
                    config_slice: Dict[str, Any]) -> Dict[str, Any]:
    confirm = None if arm == ARM_CONF_ABSENT else 1.0
    print(f"Seed {seed} Condition {arm}", flush=True)
    with arm_cell(seed, config_slice=config_slice, script_path=Path(__file__),
                  config_slice_declared=True) as cell:
        trials = []
        for t in range(TRIALS_PER_RUN):
            trials.append(_run_module_trial(cea_cfg, dim, seed, t, confirm))
            if (t + 1) % 4 == 0 or (t + 1) == TRIALS_PER_RUN:
                print(f"  [train] cea-drive seed={seed} arm={arm} "
                      f"ep {t + 1}/{TRIALS_PER_RUN}", flush=True)
        row: Dict[str, Any] = {
            "arm": arm,
            "seed": int(seed),
            "cortical_confirmation": confirm,
            "trials": trials,
        }
        cell.stamp(row)
    # Per-run verdict: every trial fired and produced a timeable onset.
    ok = all(r["onset_latency_steps"] is not None and r["n_fires"] == 1
             for r in trials)
    print(f"verdict: {'PASS' if ok else 'FAIL'}", flush=True)
    return row


def _run_c5_sweep(cea_cfg: CeAConfig, dim: int) -> Dict[str, Any]:
    """Magnitude sweep for the override-discipline bound."""
    cap = float(cea_cfg.mode_prior_log_odds_max)
    thr = float(cea_cfg.fast_route_threshold)
    amp_bounded = min(float(cea_cfg.fast_prime_amplitude), cap)
    step = (SWEEP_HI - SWEEP_LO) / float(SWEEP_N - 1)
    observed: List[float] = []
    unclipped: List[float] = []
    for i in range(SWEEP_N):
        mag = SWEEP_LO + i * step
        cea = CeAAnalog(cea_cfg)
        out = cea.tick(_z_at(mag, dim))
        observed.append(abs(float(out.fast_prime)))
        # The value the module WOULD emit without its clip (cea.py:348-360).
        raw = max(0.0, (mag - thr)) / max(1e-6, cap) * amp_bounded
        unclipped.append(raw)
    return {
        "sweep_lo": SWEEP_LO,
        "sweep_hi": SWEEP_HI,
        "sweep_n": SWEEP_N,
        "max_abs_fast_prime_observed": max(observed),
        "max_unclipped_fast_prime": max(unclipped),
        "cap": cap,
        "clip_region_reached": bool(max(unclipped) > cap),
    }


# ===========================================================================
# Block C -- NON-GATING selectivity / substrate-readiness probe
# ===========================================================================

def _run_selectivity_probe(seed: int) -> Dict[str, Any]:
    print(f"Seed {seed} Condition {BLOCK_C}", flush=True)
    torch.manual_seed(seed)
    env = _make_env(seed)
    cfg = _make_config(env)
    agent = REEAgent(cfg)
    harness = StepHarness(agent, env, train_mode=False, seed=seed)

    lf_harm: List[float] = []
    fast_prime: List[float] = []
    lf_input: List[float] = []
    lf_generic: List[float] = []
    hazard: List[float] = []
    n_fires = 0
    n_cea_none = 0

    for ep in range(TRIALS_PER_RUN):
        _flat, obs_dict = env.reset()
        agent.reset()
        harness.reset()
        for _step in range(PROBE_STEPS_PER_EP):
            res = harness.step(obs_dict)
            out = agent._cea_last_output
            if out is None:
                n_cea_none += 1
            else:
                lf_harm.append(float(out.low_freq_magnitude))
                fast_prime.append(float(out.fast_prime))
                n_fires += int(out.urgency_fire)
                ha = obs_dict.get("harm_obs_a")
                lf_input.append(_l1_per_dim(ha) if ha is not None else float("nan"))
                lf_generic.append(
                    _l1_per_dim(res.latent.z_world)
                    if res.latent.z_world is not None else float("nan"))
                hazard.append(float(res.info.get("hazard_field_at_agent", float("nan"))))
            obs_dict = res.next_obs_dict
            if res.done:
                break
        if (ep + 1) % 4 == 0 or (ep + 1) == TRIALS_PER_RUN:
            print(f"  [train] selectivity seed={seed} arm={BLOCK_C} "
                  f"ep {ep + 1}/{TRIALS_PER_RUN}", flush=True)

    valid_hz = [h for h in hazard if not math.isnan(h)]
    med = statistics.median(valid_hz) if valid_hz else float("nan")
    labels = [(h > med) for h in hazard]
    row = {
        "arm": BLOCK_C,
        "seed": int(seed),
        "n_ticks": len(lf_harm),
        "n_cea_output_none": n_cea_none,
        "embodied_fire_count": int(n_fires),
        "cea_fast_route_threshold": float(agent.cea.config.fast_route_threshold),
        "lf_harm_max": max(lf_harm) if lf_harm else float("nan"),
        "lf_harm_mean": statistics.fmean(lf_harm) if lf_harm else float("nan"),
        "fast_prime_max": max(fast_prime) if fast_prime else float("nan"),
        "hazard_label_median": float(med),
        "n_label_pos": int(sum(labels)),
        "n_label_neg": int(len(labels) - sum(labels)),
        "auroc_encoder_input_vs_hazard": _auroc(lf_input, labels),
        "auroc_cea_gate_stat_vs_hazard": _auroc(lf_harm, labels),
        "auroc_generic_activation_vs_hazard": _auroc(lf_generic, labels),
        "z_goal_stream_stats": harness.z_goal_stream_stats(),
    }
    # Non-gating: the verdict line reports whether the probe COLLECTED data,
    # not whether MECH-074c holds. Block C decides nothing.
    print(f"verdict: {'PASS' if row['n_ticks'] > 0 else 'FAIL'}", flush=True)
    return row, agent


# ===========================================================================
# Scoring
# ===========================================================================

def _score(absent_rows: List[Dict], present_rows: List[Dict],
           sweep: Dict[str, Any], cea_cfg: CeAConfig) -> Dict[str, Any]:
    absent = [t for r in absent_rows for t in r["trials"]]
    present = [t for r in present_rows for t in r["trials"]]
    all_trials = absent + present
    cap = float(cea_cfg.mode_prior_log_odds_max)

    # -- C1 onset latency -----------------------------------------------
    lat = [t["onset_latency_steps"] for t in all_trials]
    c1_pass = bool(lat) and all(v is not None and v <= ONSET_MAX_STEPS for v in lat)
    lat_ok = [v for v in lat if v is not None]

    # -- C2 fitted decay tau (ARM_CONF_ABSENT only) ----------------------
    taus = [t["fitted_half_life_steps"] for t in absent]
    c2_pass = bool(taus) and all(
        (not math.isnan(v)) and TAU_MIN <= v <= TAU_MAX for v in taus)

    # -- C3 decays within the override window ----------------------------
    ret_absent = [t["retained_fraction"] for t in absent]
    c3_pass = bool(ret_absent) and all(
        (not math.isnan(v)) and v <= WINDOW_MAX_RETAINED for v in ret_absent)

    # -- C4 overridable ---------------------------------------------------
    ret_present = [t["retained_fraction"] for t in present]
    hold_ok = bool(ret_present) and all(
        (not math.isnan(v)) and v >= HOLD_MIN_RETAINED for v in ret_present)
    mean_present = statistics.fmean(ret_present) if ret_present else float("nan")
    mean_absent = statistics.fmean(ret_absent) if ret_absent else float("nan")
    separation = mean_present - mean_absent
    c4_pass = bool(hold_ok and (not math.isnan(separation))
                   and separation >= OVERRIDE_SEPARATION_MIN)

    # -- C5 bounded --------------------------------------------------------
    c5_pass = bool(sweep["max_abs_fast_prime_observed"] <= cap + BOUND_EPS)

    overall = bool(c1_pass and c2_pass and c3_pass and c4_pass and c5_pass)

    criteria = [
        {"name": "C1_onset_latency", "load_bearing": True, "passed": c1_pass,
         "measured_max": max(lat_ok) if lat_ok else None,
         "threshold": ONSET_MAX_STEPS,
         "desc": "onset latency <= %d sim steps, every trial, both arms"
                 % ONSET_MAX_STEPS},
        {"name": "C2_decay_tau", "load_bearing": True, "passed": c2_pass,
         "measured_min": min(taus) if taus else None,
         "measured_max": max(taus) if taus else None,
         "threshold_low": TAU_MIN, "threshold_high": TAU_MAX,
         "desc": "fitted fast_prime half-life in [%.1f, %.1f] steps under "
                 "cortical non-confirmation, every trial" % (TAU_MIN, TAU_MAX)},
        {"name": "C3_decays_within_override_window", "load_bearing": True,
         "passed": c3_pass,
         "measured_max": max(ret_absent) if ret_absent else None,
         "threshold": WINDOW_MAX_RETAINED,
         "desc": "retained fraction at override-window end <= %.2f, every trial"
                 % WINDOW_MAX_RETAINED},
        {"name": "C4_overridable", "load_bearing": True, "passed": c4_pass,
         "measured_min_retained_present": min(ret_present) if ret_present else None,
         "measured_arm_separation": separation,
         "threshold_hold": HOLD_MIN_RETAINED,
         "threshold_separation": OVERRIDE_SEPARATION_MIN,
         "desc": "cortical confirmation holds the pulse (>= %.2f retained) and "
                 "the arms separate by >= %.2f" % (HOLD_MIN_RETAINED,
                                                   OVERRIDE_SEPARATION_MIN)},
        {"name": "C5_bounded", "load_bearing": True, "passed": c5_pass,
         "measured_max": sweep["max_abs_fast_prime_observed"],
         "threshold": cap + BOUND_EPS,
         "desc": "max |fast_prime| over the magnitude sweep <= "
                 "mode_prior_log_odds_max (CeA never over-rules cortex)"},
    ]

    # -- Non-degeneracy, per criterion ------------------------------------
    fire_spread = (max(t["lf_fire"] for t in all_trials)
                   - min(t["lf_fire"] for t in all_trials)) if all_trials else 0.0
    peak_spread_absent = (max(t["peak_fast_prime"] for t in absent)
                          - min(t["peak_fast_prime"] for t in absent)) if absent else 0.0
    # C4 is degenerate if the two arms were bit-identical -- i.e. the
    # manipulation did nothing at all.
    arms_differ = bool(
        ret_present and ret_absent
        and abs(mean_present - mean_absent) > 1e-9)
    criteria_non_degenerate = {
        "C1_onset_latency": bool(fire_spread > 1e-6 and len(lat_ok) == len(all_trials)),
        "C2_decay_tau": bool(peak_spread_absent > 1e-6),
        "C3_decays_within_override_window": bool(peak_spread_absent > 1e-6),
        "C4_overridable": arms_differ,
        # C5 passes VACUOUSLY if the sweep never reached the clip region.
        "C5_bounded": bool(sweep["clip_region_reached"]),
    }

    return {
        "overall_pass": overall,
        "criteria": criteria,
        "criteria_non_degenerate": criteria_non_degenerate,
        "combination_rule": ("PASS = C1_onset_latency AND C2_decay_tau AND "
                             "C3_decays_within_override_window AND "
                             "C4_overridable AND C5_bounded (plain AND; all five "
                             "load-bearing)"),
        "summary_stats": {
            "onset_latency_values": lat_ok,
            "fitted_half_life_absent": taus,
            "retained_fraction_absent": ret_absent,
            "retained_fraction_present": ret_present,
            "mean_retained_absent": mean_absent,
            "mean_retained_present": mean_present,
            "arm_separation": separation,
            "lf_fire_spread": fire_spread,
            "peak_fast_prime_spread_absent": peak_spread_absent,
        },
    }


def _preconditions(absent_rows: List[Dict], present_rows: List[Dict],
                   cea_cfg: CeAConfig) -> List[Dict[str, Any]]:
    """Readiness preconditions -- MODULE ARMS ONLY (every gated arm).

    Block C is deliberately excluded: the indexer reads this list flat and
    arm-blind and returns precondition_unmet for the WHOLE RUN on the first
    unmet entry, so putting a non-gating probe's readiness here would let it
    vacate the two clean gated legs (the V3-EXQ-785 defect).
    """
    absent = [t for r in absent_rows for t in r["trials"]]
    present = [t for r in present_rows for t in r["trials"]]
    all_trials = absent + present
    thr = float(cea_cfg.fast_route_threshold)

    cross_min, cross_cell = _worst_cell(all_trials, "lf_fire_minus_threshold", "min")
    quiet_max, quiet_cell = _worst_cell(all_trials, "lf_quiet", "max")
    peak_min, peak_cell = _worst_cell(all_trials, "peak_fast_prime", "min")

    # P3: the two arms must receive bit-identical drive at matched (seed, trial).
    by_key = {}
    for t in absent:
        by_key[(t["seed"], t["trial"])] = t["lf_trace"]
    drive_diff = 0.0
    drive_cell = None
    for t in present:
        other = by_key.get((t["seed"], t["trial"]))
        if other is None:
            continue
        d = max((abs(a - b) for a, b in zip(t["lf_trace"], other)), default=0.0)
        if d > drive_diff:
            drive_diff, drive_cell = d, t["cell_id"]

    return [
        {"name": "P1a_drive_crosses_threshold",
         "kind": "readiness",
         "description": "worst-cell (min) margin of the onset drive above "
                        "fast_route_threshold; without a crossing there is no "
                        "onset for C1 to time",
         "control": "onset magnitude drawn from the pre-registered "
                    "supra-threshold range [%.2f, %.2f] vs threshold %.2f"
                    % (FIRE_LO, FIRE_HI, thr),
         "measured": cross_min, "threshold": 0.0, "comparator": ">",
         "direction": "lower", "offending_cell": cross_cell,
         "met": bool(cross_min > 0.0)},
        {"name": "P1b_quiet_stays_subthreshold",
         "kind": "readiness",
         "description": "worst-cell (max) quiet-phase drive; a quiet phase that "
                        "re-fires would corrupt the C2/C3 decay measurement",
         "control": "quiet magnitude drawn from the pre-registered "
                    "sub-threshold range [%.2f, %.2f]" % (QUIET_LO, QUIET_HI),
         "measured": quiet_max, "threshold": thr, "comparator": "<",
         "direction": "upper", "offending_cell": quiet_cell,
         "met": bool(quiet_max < thr)},
        {"name": "P2_fast_prime_peak_supra_floor",
         "kind": "readiness",
         "description": "worst-cell (min) peak fast_prime -- the SAME statistic "
                        "C2 and C3 normalise by; a half-life and a retained "
                        "fraction are both undefined on a zero pulse",
         "control": "supra-threshold drive on an explicitly armed CeAAnalog "
                    "(use_amygdala_analog + use_cea_analog both True)",
         "measured": peak_min, "threshold": FP_PEAK_FLOOR,
         "direction": "lower", "offending_cell": peak_cell,
         "met": bool(peak_min >= FP_PEAK_FLOOR)},
        {"name": "P3_arms_receive_identical_drive",
         "kind": "readiness",
         "description": "worst-cell (max) abs difference between the two arms' "
                        "low_freq_magnitude traces at matched (seed, trial); "
                        "makes C4's separation attributable to "
                        "cortical_confirmation alone",
         "control": "both arms draw their drive from the same "
                    "(seed, trial)-keyed Random in _trial_drive()",
         "measured": drive_diff, "threshold": DRIVE_MATCH_EPS,
         "direction": "upper", "offending_cell": drive_cell,
         "met": bool(drive_diff <= DRIVE_MATCH_EPS)},
    ]


def _selectivity_leg(probe_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """NON-GATING self-route for the untested third falsifier leg."""
    total_fires = sum(int(r["embodied_fire_count"]) for r in probe_rows)
    max_lf = max((r["lf_harm_max"] for r in probe_rows), default=float("nan"))
    thr = probe_rows[0]["cea_fast_route_threshold"] if probe_rows else float("nan")
    aurocs_in = [r["auroc_encoder_input_vs_hazard"] for r in probe_rows]
    aurocs_harm = [r["auroc_cea_gate_stat_vs_hazard"] for r in probe_rows]
    aurocs_gen = [r["auroc_generic_activation_vs_hazard"] for r in probe_rows]

    def _m(vals):
        ok = [v for v in vals if not math.isnan(v)]
        return statistics.fmean(ok) if ok else float("nan")

    measurable = total_fires > 0
    return {
        "leg": "leg3_selectivity_does_not_fire_on_arousing_but_neutral",
        "status": "measurable" if measurable else "substrate_not_ready_requeue",
        "gated": False,
        "why_not_gated": (
            "Two independent reasons, both established before the run. (1) "
            "ARITHMETIC: CeAAnalog.tick() is a pure function of ||z_harm_a||_1/n, "
            "so a magnitude-matched non-harm drive fires IDENTICALLY by "
            "arithmetic and a mismatched one is decided by magnitude alone -- "
            "either way the delta is fixed before the run (the DV-symmetry "
            "artifact class, failure_autopsy_V3-EXQ-604c section 3). Selectivity "
            "is a property of the upstream AffectiveHarmEncoder, not of "
            "CeAAnalog. (2) MEASURED: in the embodied regime the fast route "
            "never fires at all, so there is no fast_prime signal on which any "
            "in-agent selectivity DV could be computed."),
        "embodied_fire_count_total": int(total_fires),
        "embodied_max_lf_harm": max_lf,
        "cea_fast_route_threshold": thr,
        "mean_auroc_encoder_input_vs_hazard": _m(aurocs_in),
        "mean_auroc_cea_gate_stat_vs_hazard": _m(aurocs_harm),
        "mean_auroc_generic_activation_vs_hazard": _m(aurocs_gen),
        "auroc_reference_floor": AUROC_READY_FLOOR,
        "recorded_for_successor": (
            "A leg-3 confirmer needs (a) an AffectiveHarmEncoder trained far "
            "enough that ||z_harm_a||_1/n reaches fast_route_threshold in an "
            "embodied run, and (b) an arousing-but-neutral env condition that "
            "dissociates generic activation from harm-affective valence. The "
            "AUROC triple recorded here (encoder input / CeA gate statistic / "
            "generic activation, all against the same hazard-proximity label) "
            "is the readiness denominator for that successor, so it need not "
            "re-derive it."),
    }


# ===========================================================================
# Main
# ===========================================================================

def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    t0 = time.perf_counter()
    seeds = SEEDS[:2] if dry_run else SEEDS

    probe_env = _make_env(seeds[0])
    cfg = _make_config(probe_env)
    cea_cfg = _cea_config_from_ree(cfg)
    dim = int(cfg.latent.z_harm_a_dim)
    slice_ = _config_slice(cfg, cea_cfg)

    print("=== V3-EXQ-895 MECH-074c fast_prime dynamics ===", flush=True)
    print("CeA: thr=%.3f cap=%.3f amp=%.3f tau=%d window=%d" % (
        cea_cfg.fast_route_threshold, cea_cfg.mode_prior_log_odds_max,
        cea_cfg.fast_prime_amplitude, cea_cfg.fast_prime_decay_tau_steps,
        cea_cfg.fast_prime_override_window_steps), flush=True)

    absent_rows: List[Dict[str, Any]] = []
    present_rows: List[Dict[str, Any]] = []
    for seed in seeds:
        absent_rows.append(_run_module_arm(cea_cfg, dim, seed, ARM_CONF_ABSENT, slice_))
    for seed in seeds:
        present_rows.append(_run_module_arm(cea_cfg, dim, seed, ARM_CONF_PRESENT, slice_))

    sweep = _run_c5_sweep(cea_cfg, dim)

    probe_rows: List[Dict[str, Any]] = []
    last_agent = None
    for seed in seeds:
        row, last_agent = _run_selectivity_probe(seed)
        probe_rows.append(row)

    scored = _score(absent_rows, present_rows, sweep, cea_cfg)
    preconds = _preconditions(absent_rows, present_rows, cea_cfg)
    sel = _selectivity_leg(probe_rows)

    unmet = [p["name"] for p in preconds if not p["met"]]
    if unmet:
        label = "substrate_not_ready_requeue"
        outcome = "FAIL"
        direction = "unknown"
    else:
        label = ("mech074c_fast_prime_dynamics_confirmed"
                 if scored["overall_pass"]
                 else "mech074c_fast_prime_dynamics_not_confirmed")
        outcome = "PASS" if scored["overall_pass"] else "FAIL"
        direction = "supports" if scored["overall_pass"] else "weakens"

    # Non-degeneracy for the scoring net (evidence-purpose runs too). Fed the
    # DRIVE quantities, which genuinely vary across seeds/trials -- deliberately
    # NOT the fitted tau, whose invariance across varied peaks is the PREDICTED
    # RESULT (the claim asserts tau is a constant) and not a degeneracy.
    all_trials = [t for r in (absent_rows + present_rows) for t in r["trials"]]
    fire_vals = [t["lf_fire"] for t in all_trials]
    peak_vals = [t["peak_fast_prime"] for t in all_trials]
    non_degenerate = bool(
        len(set(round(v, 9) for v in fire_vals)) > 1
        and len(set(round(v, 9) for v in peak_vals)) > 1
        and scored["criteria_non_degenerate"]["C4_overridable"])
    degeneracy_reason = None if non_degenerate else (
        "drive magnitudes or fast_prime peaks did not vary across "
        "seeds/trials, or the two confirmation arms were bit-identical -- the "
        "seed dimension carries no information")

    arm_results = absent_rows + present_rows

    manifest: Dict[str, Any] = {
        "run_id": f"{EXPERIMENT_TYPE}_{_utc_stamp()}_v3",
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "queue_id": "V3-EXQ-895",
        "outcome": outcome,
        "timestamp_utc": _utc_stamp(),
        "evidence_direction": direction,
        "evidence_scope_note": (
            "PARTIAL confirmation by design. MECH-074c's falsifier names three "
            "legs; this run gates on leg 1 (onset latency) and leg 2 (decay "
            "under cortical non-confirmation, plus its overridability) and does "
            "NOT gate on leg 3 (selectivity vs arousing-but-neutral input), "
            "which is not designable as a CeA-module manipulation -- see "
            "interpretation.selectivity_leg. A 'supports' here is support for "
            "legs 1 and 2 only. The fast:slow ratio in the claim's "
            "functional_restatement is also out of scope (needs SD-032c AIC "
            "comparator wiring)."),
        "non_degenerate": non_degenerate,
        "degeneracy_reason": degeneracy_reason,
        "overall_pass": scored["overall_pass"],
        "criteria": scored["criteria"],
        "combination_rule": scored["combination_rule"],
        "summary_stats": scored["summary_stats"],
        "c5_magnitude_sweep": sweep,
        "arm_results": arm_results,
        "per_seed_selectivity_probe": probe_rows,
        "interpretation": {
            "label": label,
            "preconditions": preconds,
            "preconditions_scope_note": (
                "These preconditions cover the MODULE arms "
                "(ARM_CONF_ABSENT / ARM_CONF_PRESENT), which are every arm this "
                "run gates on -- no gated arm is scoped out. The SELECTIVITY_PROBE "
                "block is NON-GATING and carries no precondition here on purpose: "
                "the indexer reads this list flat and arm-blind, so a probe "
                "readiness entry would vacate the two clean gated legs at "
                "adjudication time (the V3-EXQ-785 whole-run-AND defect). The "
                "probe's own readiness self-route is at "
                "interpretation.selectivity_leg."),
            "criteria_non_degenerate": scored["criteria_non_degenerate"],
            "selectivity_leg": sel,
        },
        "custom_information": {
            "gov_reuse_1_check": (
                "Decisive readouts sought: fast_prime onset latency, fitted "
                "decay half-life, and a selectivity contrast. Checked "
                "v3_exq_473_sd035_cea_mode_prior (the only prior run tagging "
                "MECH-074c; run_ids ..._20260421T195334Z_v3 and "
                "..._20260421T195533Z_v3). It is experiment_purpose=diagnostic "
                "with substrate_hash None, and recorded no latency, no fitted "
                "tau, and no selectivity probe -- only a single fire point and a "
                "2-point decay/hold pair, with no per-tick trace to derive from. "
                "Not recoverable -> run."),
            "substrate_wiring_note": (
                "agent.act()/sense_flat() pass obs_harm_a=None, so z_harm_a is "
                "never produced and the CeA never ticks on that path. The "
                "selectivity probe therefore drives the agent through the "
                "canonical StepHarness, which supplies obs_harm_a/harm_history "
                "from the env obs_dict. Verified live before authoring: 480/480 "
                "harness ticks produced a CeAOutput, 0/480 via agent.act()."),
            "out_of_scope": [
                "leg 3 selectivity (not designable as a module manipulation; "
                "see interpretation.selectivity_leg)",
                "fast:slow latency ratio vs the cortical AIC/dACC comparator "
                "(needs SD-032c wiring on the same tick base)",
            ],
        },
    }

    elapsed = time.perf_counter() - t0
    print("elapsed_seconds=%.2f outcome=%s" % (elapsed, outcome), flush=True)
    return {
        "manifest": manifest,
        "outcome": outcome,
        "config": {
            "seeds": seeds,
            "conditions": CONDITIONS,
            "trials_per_run": TRIALS_PER_RUN,
            "probe_steps_per_ep": PROBE_STEPS_PER_EP,
            "drive_protocol": slice_,
            "thresholds": {
                "ONSET_MAX_STEPS": ONSET_MAX_STEPS,
                "TAU_MIN": TAU_MIN, "TAU_MAX": TAU_MAX,
                "WINDOW_MAX_RETAINED": WINDOW_MAX_RETAINED,
                "HOLD_MIN_RETAINED": HOLD_MIN_RETAINED,
                "OVERRIDE_SEPARATION_MIN": OVERRIDE_SEPARATION_MIN,
                "FP_PEAK_FLOOR": FP_PEAK_FLOOR,
                "DRIVE_MATCH_EPS": DRIVE_MATCH_EPS,
                "BOUND_EPS": BOUND_EPS,
            },
        },
        "seeds": seeds,
        "started_at": t0,
        "agent": last_agent,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="smoke: 2 seeds, same protocol, manifest relocated")
    args = ap.parse_args()

    result = run_experiment(dry_run=args.dry_run)
    manifest = result["manifest"]

    out_path = write_flat_manifest(
        manifest,
        dry_run=args.dry_run,
        config=result["config"],
        seeds=result["seeds"],
        script_path=Path(__file__),
        started_at=result["started_at"],
        agent=result["agent"],
    )

    print("manifest: %s" % out_path, flush=True)
    print("outcome: %s" % result["outcome"], flush=True)
    print(json.dumps(manifest["summary_stats"], indent=1)[:1200], flush=True)

    _outcome_raw = str(result["outcome"]).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
        dry_run=args.dry_run,
    )
