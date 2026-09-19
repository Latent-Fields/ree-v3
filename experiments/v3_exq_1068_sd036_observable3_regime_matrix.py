"""V3-EXQ-1068: SD-036 observable #3 -- the shared harm-stream REGIME MATRIX.

Scores SD-036's observable #3 (the multi-stream cluster), carries SD-011's only
open measurement (the dissociation under use_gabaergic_decay=True), and emits
MECH-279's pag_n_commits at theta_freeze 2.0 and 0.8. The claim text commissions
exactly this joint run ("RUN THIS AS THE GROUP'S SHARED REGIME MATRIX, not as a
solo experiment").

red-team (fable): see verdict line at the end of this docstring.

WHAT IS SCORED, AND WHAT IS NOT
--------------------------------
  observable #1 -- NOT scored. WITHDRAWN AS WORDED, permanently (measured
    encoder floor harm_norm ~0.509 EXCEEDS the 471-lineage avoid threshold 0.25,
    so no decay rate can resolve the mode lock under that classifier). Mode
    metrics are still RECORDED, non-gating, as V3-EXQ-854 did.
  observable #2 -- NOT scored. Discharged by V3-EXQ-854 (C1 rho = -1.0 across
    the sweep, 3/3 seeds).
  observable #3 -- SCORED. C1 below is load-bearing. This is the whole content
    of the claim.
  observable #4 -- NOT scored. Belongs to MECH-279; ran as V3-EXQ-776.
  MECH-258 -- NOT attached. PARKED by user decision 2026-09-08.

WHY THE V3-EXQ-854 C2 RESULT IS NOT SIMPLY PROMOTED
---------------------------------------------------
SD-036's claim text says observable #3 was "NEVER RUN". That is stale:
V3-EXQ-854 carries a non-load-bearing `C2_multi_stream_cluster` which FAILED,
1 of 3 seeds passing (42: F/T/F, 43: F/T/T, 44: T/T/T), inside a run recorded
PASS / supports. But 854's C2 is a BARE SIGN TEST at a two-point contrast
(tone 0.3 vs 2.0) with no magnitude floor and no per-stream readiness gate, and
a reanalysis of that manifest (this session, 2026-09-19) shows the failure is
not a near-miss on a small effect. Spearman rho(gaba_tone, sustain_ratio) over
the full registered 5-tone sweep, per seed:

    z_harm_a (tau 0.02):  -1.00  -1.00  -1.00   <- clean monotone decrease
    z_harm   (tau 0.05):  +0.10  +0.70  -0.50   <- NO tone ordering
    z_beta   (tau 0.03):  +0.60  +0.20  -0.10   <- NO tone ordering

Only ONE of the three streams responds to gaba_tone at all. Two further facts
from the same reanalysis, both of which this driver is built to handle:
  * the `1.4e-03` spread SD-036 precondition (iii) attributes to z_harm is
    actually z_BETA's. z_harm's sweep spread is 4.8e-02 to 1.5e-01, i.e.
    30-110x larger than the claim states, and entirely unordered.
  * z_beta's sustain ratio is pinned at 0.9896-0.9923 across every tone and
    seed -- the DV has ~0.15% of dynamic range. And 854's readiness control
    gates `harm_a` ONLY, although SD-036 precondition (i) requires a non-zero
    sweep spread on "EVERY stream it scores".

So a three-stream conjunction scored on sustain_ratio alone would record a
FALSIFICATION of SD-036's sole architectural commitment that is substantially a
DV-validity artifact -- the mirror image of the "confident-but-wrong
confirmation" precondition (i) exists to prevent.

THE DESIGN DECISIONS THIS RUN EXECUTES (user, 2026-09-19T23:48:33Z)
-------------------------------------------------------------------
  D0 = A+B. Extend the per-stream readiness gate so it can DISQUALIFY a stream,
       and pre-register `shape_deviation` as the per-stream DV for any stream
       the gate disqualifies -- computed for EVERY stream, symmetrically, so
       there is no post-hoc DV selection.
  D1 = B.   The per-stream bar is MONOTONICITY -- Spearman rho <= -0.9 over the
       full 5-tone sweep -- not a two-point sign test and not a magnitude floor.
       This is the same bar V3-EXQ-854's load-bearing C1 already uses.
  D2 = C.   The tau-ORDERING test is DROPPED from this run. It is ill-posed as
       written: the composed pole is (1 - alpha) * exp(-tau * tone) and SD-036
       never mentions alpha, so the substrate predicts the largest sustain-ratio
       effect on the SMALLEST-tau stream (z_harm_a, alpha 0.2, retained 0.8),
       which is the reverse of what CONFIRMING scores as success. Raised to
       /governance as a flag rather than scored here.
  D3 = B.   NO single-stream isolation arms. The isolation disjunct only
       discriminates once a cluster is observed; joint test first.
  D4 = A.   P0h affective-encoder warmup NOT armed. The regime is STATED:
       harm_history_len=10, limb_damage_enabled=True, p0h_armed=False.

DV-SYMMETRY DECLARATION (mandatory, per arm)
--------------------------------------------
* Primary per-stream DV `<stream>_sustain_ratio` = mean/peak of ||z||. Its
  symmetry group is POSITIVE SCALAR RESCALING of the trajectory. The ARM_ON
  manipulation (gaba_tone) is NOT invariant under that group post-35e8969: it
  changes the leaky-integrator pole, altering trajectory SHAPE, not scale. This
  is MEASURED per stream by the readiness control below, never assumed -- and
  pre-fix the same measurement returned ~8.6e-08 (exact invariance).
* Fallback DV `<stream>_shape_deviation_vs_off` = max deviation between
  peak-normalised trajectories. Same symmetry group (scale-free by
  construction), same non-invariance argument, and it is the readout that was
  exactly zero pre-fix.
* ARM_OFF is the MASTER SWITCH (structural presence/absence of the regulator
  and its recurrence), not a rescale, so it is likewise not invariant.
* The PAG arms' DV `pag_n_commits` is a COUNT of freeze-commit ticks. Its
  symmetry group is permutation of ticks; theta_freeze changes the commit
  predicate itself, not the tick order, so it is not invariant.
* Both DVs are invariant under time-permutation (mean, peak and max are
  symmetric functions). The manipulation is a temporal decay, which changes the
  MULTISET of trajectory values, not merely their order -- not invariant.

THE CONTROL ARM IS `use_gabaergic_decay=False`, NOT `gaba_tone=0.0`
-------------------------------------------------------------------
SD-036 precondition (ii), verbatim: post-fix, "tone 0.0 suspends decay but
leaves the recurrence live, so it is no longer bit-equal to the legacy arm".
Using tone 0.0 as the control would compare two stateful substrates and credit
the recurrence to the regulator.

MECH-279 IS HELD OFF IN THE ARMS SCORING SD-036
-----------------------------------------------
`use_pag_freeze_gate=False` in ARM_OFF and ARM_ON, per the V3-EXQ-854
de-confounding convention SD-036 names explicitly. MECH-279 consumes gaba_tone
as a DIRECT SCALAR (exit_threshold = theta_freeze * gaba_tone) and never through
the decay path, so an active gate would give the tone sweep a second, non-decay
route to the readout. The two PAG arms are therefore SEPARATE arms at the
baseline tone, and their readout is EMIT-ONLY: the claim text says "emit
MECH-279's pag_n_commits at both theta 2.0 and 0.8", so no criterion scores
MECH-279 here and its per-claim direction is recorded "unknown".

SD-011's LEG -- THE REGIME COMPARISON THAT IS ITS ONLY OPEN MEASUREMENT
-----------------------------------------------------------------------
SD-011 what_would_answer: "every result above predates ree-v3 35e8969, which
gave z_harm/z_harm_a temporal state for the first time. The dissociation has
never been measured with use_gabaergic_decay=True. This is the SD-011 leg of
the shared harm-stream regime-matrix run described in SD-036's
what_would_answer -- run it there, do not mint a separate experiment."

So the DVs (stream_corr, autocorr_gap, harm_fwd_r2, and the D3 direction
R2_affective vs R2_sensory) are measured in BOTH regimes -- ARM_OFF is the
validated-regime positive control, ARM_ON at the baseline tone is the
never-measured one.

REGIME DECLARATION (SD-011 preconditions (i)-(iii), all three REQUIRED):
  (i)   harm_history_len = 10 (NOT the default 0). A run at 0 is VACUOUS as an
        SD-011 test regardless of what its stream_corr reads. Supplied by
        `_lib/baselines/sd036_decay.py`; ASSERTED at runtime below.
  (ii)  Encoder floor measured every run and every DV in floor-robust RELATIVE
        form -- never an absolute level. (harm_encoder(zeros) ~0.46,
        affective_harm_encoder(zeros,zeros) ~0.33, resting harm_norm ~0.51.)
  (iii) SOURCING MODE STATED: limb_damage_enabled=True -- the SD-022 BODY path
        (rank 4 free / 5 numerical), NOT the legacy proximity path, whose
        harm_obs_a is numerical rank 2 (causal_grid_world.py:3033-3038 writes
        hazard_at_agent into [:25] and resource_at_agent into [25:], each
        broadcast over 25 dims). On the legacy path the 2026-09-18 rank-2 spike
        found SD-011's C2/C3 "pass for a reason that is not the reason SD-011
        asserts". ASSERTED at runtime below.
  P0h:  NOT armed (D4=A). On SD-011's own target the P0h stage does not clear
        readiness (mean lift -1108.379, 0/3 seeds); only the SD-020 PE target
        clears. Arming it would buy no SD-011 power while breaking comparability
        with V3-EXQ-854, the one landed SD-036 measurement.

SAMPLE-SIZE INTEGRITY ON THE PAG COUNT (load-bearing, easy to get wrong)
------------------------------------------------------------------------
`agent._pag_last_output` is a LATCHED attribute: the PAG gate ticks inside
`select_action`, which RETURNS EARLY at agent.py:7416 on any non-E3 tick
(`if not ticks["e3_tick"] and self._last_action is not None`). So the gate fires
roughly once per `e3_steps_per_tick` env steps, not every step. Reading the
attribute once per env step without clearing would re-record one gate decision
as ~10 independent observations. This driver clears it to None before every
step and counts only fresh non-None reads, emitting `pag_n_latched_ticks`
alongside so the true denominator is auditable.

CRITERIA (pre-registered; thresholds are constants below, never post-hoc)
------------------------------------------------------------------------
  C1 (LOAD-BEARING) -- observable #3, the multi-stream cluster. For every
     stream that CLEARS its per-stream readiness gate, Spearman rho between
     gaba_tone and that stream's scored DV across the 5 tones is <= C1_RHO_MAX.
     CONFIRMING requires ALL readiness-clearing streams to be monotone in the
     same run. PASS needs >= C1_SEEDS_REQUIRED of 3 seeds. A stream that FAILS
     its readiness gate on BOTH DVs is EXCLUDED from scoring and recorded
     vacuous for that stream -- never counted as a failure (SD-036 precondition
     (i)). If FEWER THAN C1_MIN_SCOREABLE_STREAMS clear, the run self-routes
     `observable_3_not_scoreable` and sets non_degenerate=false: that is the
     honest outcome, not a refutation.
  C2 -- SD-011 dissociation under decay. In ARM_ON at the baseline tone:
     stream_corr <= C2_STREAM_CORR_MAX, autocorr_gap >= C2_AUTOCORR_GAP_MIN,
     harm_fwd_r2 >= C2_HARM_FWD_R2_MIN, and the D3 direction does NOT reverse
     (R2_affective <= R2_sensory). >= C2_SEEDS_REQUIRED of 3 seeds.
  C3 -- ON-vs-OFF separation at the baseline tone (floor-robust form): the
     trained ON agent has a LOWER harm_a sustain ratio than the trained OFF
     agent. >= C3_SEEDS_REQUIRED of 3 seeds. Carried forward from V3-EXQ-854 as
     a continuity check; NOT load-bearing.

Overall PASS requires C1. C2/C3 are recorded and reported but do not carry the
verdict, so a null on either is informative rather than fatal.

NULLS, DECLARED
---------------
  * C1 null with every stream readiness-GREEN: the streams do NOT degrade
    together -- SD-036's FALSIFYING first disjunct, and a genuine weakens.
  * C1 not-scoreable (fewer than C1_MIN_SCOREABLE_STREAMS clear readiness on
    either DV): NOT a scientific null and NOT a weakens. The instrument cannot
    express the effect on those streams; routes `observable_3_not_scoreable`.
  * C1 spread below VACUITY_CEILING on the PRIMARY stream: the pre-fix
    signature returning. Routes `substrate_not_ready_requeue`.
  * C2 null: the dissociation does NOT survive the decay regime -- a genuine
    weakens for SD-011, PROVIDED the regime assertions below all held. If
    harm_history_len were 0 the leg would be VACUOUS, not a weakens; the run
    refuses to start in that case rather than reporting one.
  * MECH-279: emit-only. No criterion; direction recorded "unknown".

No sleep machinery is used, so no SLEEP DRIVER line applies.

ASCII-only output (repo rule).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

_THIS = Path(__file__).resolve()
_EXPERIMENTS_DIR = _THIS.parent
_REE_V3_ROOT = _THIS.parents[1]
for _p in (str(_EXPERIMENTS_DIR), str(_REE_V3_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiment_protocol import emit_outcome  # noqa: E402

from _lib.arm_fingerprint import arm_cell  # noqa: E402
from _lib.stats import spearman  # noqa: E402
from _lib.baselines import sd036_decay as B  # noqa: E402
from _lib.manifest_core import stamp_recording_core  # noqa: E402
from _lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402
from pack_writer import write_flat_manifest  # noqa: E402

from ree_core.agent import REEAgent  # noqa: E402

_ZG = ZGoalStreamAccumulator()


EXPERIMENT_PURPOSE = "evidence"
EXPERIMENT_TYPE = "v3_exq_1068_sd036_observable3_regime_matrix"
CLAIM_IDS = ["SD-036", "SD-011", "MECH-279"]
QUEUE_ID = "V3-EXQ-1068"

# --- the registered sweep (identical to V3-EXQ-854, so the two are comparable) ---
TONE_SWEEP: List[float] = [0.3, 0.5, 1.0, 1.5, 2.0]
BASELINE_TONE = 1.0

# The three registered decay streams and their registered taus. The taus are
# RECORDED for provenance only -- D2=C drops the tau-ordering test from this run
# (see docstring: the composed pole includes alpha, which SD-036 never mentions).
STREAMS: List[str] = ["z_harm", "z_harm_a", "z_beta"]
REGISTERED_TAUS: Dict[str, float] = {"z_harm": 0.05, "z_harm_a": 0.02, "z_beta": 0.03}
PRIMARY_STREAM = "z_harm_a"   # SD-036 precondition (iii)

ARM_OFF = "decay_off_legacy"
ARM_ON = "decay_on"
ARM_PAG_HI = "pag_theta_2p0"
ARM_PAG_LO = "pag_theta_0p8"
ARMS = [ARM_OFF, ARM_ON, ARM_PAG_HI, ARM_PAG_LO]
PAG_THETAS: Dict[str, float] = {ARM_PAG_HI: 2.0, ARM_PAG_LO: 0.8}

# --- pre-registered thresholds (NEVER derived from the run's own statistics) ---
# C1: the SAME bar V3-EXQ-854's load-bearing C1 uses (D1=B).
C1_RHO_MAX = -0.9
C1_SEEDS_REQUIRED = 2
# At least this many of the three streams must clear readiness for observable #3
# to be scoreable as a CLUSTER at all. Two is the minimum at which "degrade
# TOGETHER" has any content.
C1_MIN_SCOREABLE_STREAMS = 2

# C2: SD-011's registered bands. stream_corr / autocorr_gap are the EXQ-178b
# thresholds; harm_fwd_r2 is SD-011's own CONFIRMING band ("~0.6, the
# EXQ-178b/EXQ-198 band"), which is STRICTER than 178b's driver-side 0.20.
C2_STREAM_CORR_MAX = 0.85
C2_AUTOCORR_GAP_MIN = 0.10
C2_HARM_FWD_R2_MIN = 0.60
C2_SEEDS_REQUIRED = 2

C3_SEEDS_REQUIRED = 2

# Substrate-readiness floors, carried over from V3-EXQ-854 unchanged.
VACUITY_CEILING = 1e-5
READINESS_SPREAD_FLOOR = 1e-3
READINESS_TAPE_STEPS = 60

# SD-011 forward-model fitting (P1-style: frozen encoder, detached latents).
FWD_FIT_STEPS = 600
FWD_FIT_EPOCHS = 60
FWD_LR = 1e-3
FWD_HIDDEN = 64
AUTOCORR_LAG = 10

# Regime declaration (SD-011 preconditions (i) and (iii)) -- ASSERTED, not assumed.
REQUIRED_HARM_HISTORY_LEN = 10
REQUIRED_LIMB_DAMAGE_ENABLED = True
P0H_ARMED = False


# --------------------------------------------------------------------------
# Regime assertion -- SD-011 (i)/(iii). A violation makes the SD-011 leg
# VACUOUS, so the run refuses to start rather than reporting a false weakens.
# --------------------------------------------------------------------------
def assert_regime() -> Dict[str, Any]:
    hhl = int(getattr(B, "HARM_HISTORY_LEN", 0))
    lde = bool(B.ENV_KWARGS.get("limb_damage_enabled", False))
    if hhl != REQUIRED_HARM_HISTORY_LEN:
        raise SystemExit(
            "REGIME VIOLATION (SD-011 precondition (i)): harm_history_len is "
            f"{hhl}, required {REQUIRED_HARM_HISTORY_LEN}. A run at 0 is VACUOUS "
            "as an SD-011 test and must not be reported as a weakens."
        )
    if lde != REQUIRED_LIMB_DAMAGE_ENABLED:
        raise SystemExit(
            "REGIME VIOLATION (SD-011 precondition (iii)): limb_damage_enabled is "
            f"{lde}, required {REQUIRED_LIMB_DAMAGE_ENABLED} (the SD-022 body path). "
            "The legacy proximity path is numerical rank 2."
        )
    return {
        "harm_history_len": hhl,
        "limb_damage_enabled": lde,
        "p0h_armed": P0H_ARMED,
        "sourcing_mode": "sd022_body_damage",
    }


def _batch(v: Any) -> Optional[torch.Tensor]:
    if v is None:
        return None
    v = v.float()
    return v.unsqueeze(0) if v.dim() == 1 else v


# --------------------------------------------------------------------------
# D0=A -- the PER-STREAM readiness control.
#
# V3-EXQ-854 asserted the tone effect on harm_a ONLY. SD-036 precondition (i)
# requires a non-zero sweep spread on EVERY stream scored, so this replays the
# same fixed tape and measures all three. A stream below the floor on BOTH DVs
# is EXCLUDED from scoring, never counted as a failure.
# --------------------------------------------------------------------------
def _record_tape(seed: int, steps: int) -> List[Dict[str, Any]]:
    """A fixed observation tape, independent of any agent (854 methodology)."""
    env = B.make_env(seed)
    _, od = env.reset()
    rng = np.random.RandomState(seed)
    frames: List[Dict[str, Any]] = []
    for _ in range(steps):
        frames.append(
            dict(
                body=_batch(od["body_state"]),
                world=_batch(od["world_state"]),
                harm=_batch(od.get("harm_obs")),
                harm_a=_batch(od.get("harm_obs_a")),
                hist=_batch(od.get("harm_history")),
            )
        )
        _, od, _, done, _ = env.step(int(rng.randint(env.action_dim)))
        if done:
            _, od = env.reset()
    return frames


def _replay_tape_streams(
    seed: int, frames: List[Dict[str, Any]], tone: float
) -> Dict[str, np.ndarray]:
    """Replay the fixed tape into a FRESH ON agent at `tone`; all three streams."""
    env = B.make_env(seed)
    _, od0 = env.reset()
    agent = B.make_agent(env, od0, use_gabaergic_decay=True, gaba_tone=tone)
    agent.reset()
    agent.eval()
    out: Dict[str, List[float]] = {s: [] for s in STREAMS}
    with torch.no_grad():
        for fr in frames:
            lat = agent.sense(
                fr["body"],
                fr["world"],
                obs_harm=fr["harm"],
                obs_harm_a=fr["harm_a"],
                obs_harm_history=fr["hist"],
            )
            for s in STREAMS:
                z = getattr(lat, s, None)
                out[s].append(float(z.norm()) if z is not None else float("nan"))
    return {s: np.asarray(v, dtype=np.float64) for s, v in out.items()}


def readiness_control(seed: int) -> Dict[str, Any]:
    """P0 POSITIVE CONTROL, per stream.

    Asserts the SAME STATISTIC the load-bearing criterion C1 routes on -- the
    SPREAD across the tone sweep of each stream's scored DV -- on a condition
    where the effect is known to exist for at least the primary stream (fixed
    observation tape, fresh agents).

    Both DVs are measured for EVERY stream, symmetrically (D0=B), so the
    fallback is pre-registered rather than selected after seeing the primary.
    Below the floor on BOTH means the INSTRUMENT cannot express a tone effect on
    that stream -- never "SD-036 is refuted".
    """
    frames = _record_tape(seed, READINESS_TAPE_STEPS)
    per_tone = {t: _replay_tape_streams(seed, frames, t) for t in TONE_SWEEP}
    ref = per_tone[BASELINE_TONE]

    streams: Dict[str, Any] = {}
    for s in STREAMS:
        sustain = [B.sustain_ratio(per_tone[t][s]) for t in TONE_SWEEP]
        # Fallback DV reference is the tone-1.0 trajectory here (no OFF arm
        # exists on the tape); the scored-arm fallback uses the OFF arm.
        shape = [B.shape_deviation(per_tone[t][s], ref[s]) for t in TONE_SWEEP]
        sustain_spread = float(np.nanmax(sustain) - np.nanmin(sustain))
        shape_spread = float(np.nanmax(shape) - np.nanmin(shape))
        streams[s] = {
            "tau_registered": REGISTERED_TAUS[s],
            "sustain_ratios": [float(x) for x in sustain],
            "sustain_spread": sustain_spread,
            "sustain_ready": bool(sustain_spread >= READINESS_SPREAD_FLOOR),
            "shape_deviations": [float(x) for x in shape],
            "shape_spread": shape_spread,
            "shape_ready": bool(shape_spread >= READINESS_SPREAD_FLOOR),
            # Non-gating DV-compression diagnostic: a sustain ratio pinned near
            # its 1.0 ceiling has no room to express a decay effect. Recorded so
            # a reader can see WHY a stream failed readiness (z_beta sat at
            # 0.9896-0.9923 across the whole sweep in V3-EXQ-854).
            "dv_ceiling_headroom": float(1.0 - float(np.nanmax(sustain))),
        }
        streams[s]["ready_any_dv"] = bool(
            streams[s]["sustain_ready"] or streams[s]["shape_ready"]
        )
        streams[s]["scored_dv"] = (
            "sustain_ratio" if streams[s]["sustain_ready"]
            else ("shape_deviation_vs_off" if streams[s]["shape_ready"] else None)
        )
    return {"seed": seed, "tones": list(TONE_SWEEP), "streams": streams}


# --------------------------------------------------------------------------
# SD-011 dissociation DVs. Implementations follow V3-EXQ-178b verbatim
# (stream_corr, autocorr_gap) plus the D3 direction, which 178b did not emit.
# All floor-robust RELATIVE forms -- no absolute harm_norm level anywhere.
# --------------------------------------------------------------------------
def _autocorr(series: np.ndarray, lag: int) -> float:
    arr = np.asarray(series, dtype=np.float64)
    if arr.size <= lag or float(np.nanstd(arr)) < 1e-8:
        return 0.0
    a = arr[:-lag] - arr[:-lag].mean()
    b = arr[lag:] - arr[lag:].mean()
    denom = (np.sqrt((a ** 2).sum()) * np.sqrt((b ** 2).sum())) + 1e-8
    return float(np.dot(a, b) / denom)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 10 or b.size < 10:
        return 0.0
    n = min(a.size, b.size)
    ac = a[:n] - a[:n].mean()
    bc = b[:n] - b[:n].mean()
    denom = (np.sqrt((ac ** 2).sum()) * np.sqrt((bc ** 2).sum())) + 1e-8
    return float(np.dot(ac, bc) / denom)


def _fit_forward_r2(states: np.ndarray, actions: np.ndarray) -> float:
    """R^2 of a small forward model predicting z(t+1) from (z(t), action).

    P1-style by construction: `states` are already-detached numpy latents from a
    FROZEN agent, so no gradient reaches any encoder. Compared against a
    constant-mean predictor, which is what makes it an R^2 rather than a loss.
    """
    if states.shape[0] < 20:
        return float("nan")
    x = torch.tensor(states[:-1], dtype=torch.float32)
    y = torch.tensor(states[1:], dtype=torch.float32)
    a = torch.tensor(actions[:-1], dtype=torch.float32)
    inp = torch.cat([x, a], dim=-1)
    model = torch.nn.Sequential(
        torch.nn.Linear(inp.shape[-1], FWD_HIDDEN),
        torch.nn.ReLU(),
        torch.nn.Linear(FWD_HIDDEN, y.shape[-1]),
    )
    opt = torch.optim.Adam(model.parameters(), lr=FWD_LR)
    for _ in range(FWD_FIT_EPOCHS):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(inp), y)
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = model(inp)
        ss_res = float(((y - pred) ** 2).sum())
        ss_tot = float(((y - y.mean(dim=0, keepdim=True)) ** 2).sum())
    return float(1.0 - ss_res / (ss_tot + 1e-8))


def measure_sd011_dvs(agent: REEAgent, env_seed: int) -> Dict[str, Any]:
    """SD-011's dissociation DVs on a FROZEN agent. No learning of the agent."""
    env = B.make_env(env_seed)
    _, od = env.reset()
    agent.reset()
    agent.eval()
    hs_norms: List[float] = []
    ha_norms: List[float] = []
    hs_vecs: List[np.ndarray] = []
    ha_vecs: List[np.ndarray] = []
    acts: List[np.ndarray] = []
    rng = np.random.RandomState(env_seed + 1000)
    n_actions = env.action_dim
    with torch.no_grad():
        for _ in range(FWD_FIT_STEPS):
            lat = agent.sense(
                _batch(od["body_state"]),
                _batch(od["world_state"]),
                obs_harm=_batch(od.get("harm_obs")),
                obs_harm_a=_batch(od.get("harm_obs_a")),
                obs_harm_history=_batch(od.get("harm_history")),
            )
            zs = getattr(lat, "z_harm", None)
            za = getattr(lat, "z_harm_a", None)
            if zs is None or za is None:
                break
            hs_norms.append(float(zs.norm()))
            ha_norms.append(float(za.norm()))
            hs_vecs.append(zs.detach().cpu().numpy().reshape(-1))
            ha_vecs.append(za.detach().cpu().numpy().reshape(-1))
            ai = int(rng.randint(n_actions))
            oh = np.zeros(n_actions, dtype=np.float64)
            oh[ai] = 1.0
            acts.append(oh)
            _, od, _, done, _ = env.step(ai)
            if done:
                _, od = env.reset()
                agent.reset()

    hs = np.asarray(hs_norms, dtype=np.float64)
    ha = np.asarray(ha_norms, dtype=np.float64)
    stream_corr = _pearson(hs, ha)
    ha_ac = _autocorr(ha, AUTOCORR_LAG)
    hs_ac = _autocorr(hs, AUTOCORR_LAG)
    r2_sensory = _fit_forward_r2(np.asarray(hs_vecs), np.asarray(acts))
    r2_affective = _fit_forward_r2(np.asarray(ha_vecs), np.asarray(acts))
    return {
        "n_steps_measured": int(hs.size),
        "stream_corr": stream_corr,
        "autocorr_gap": float(ha_ac - hs_ac),
        "z_ha_autocorr_lag10": ha_ac,
        "z_hs_autocorr_lag10": hs_ac,
        "harm_fwd_r2": r2_sensory,
        "r2_sensory": r2_sensory,
        "r2_affective": r2_affective,
        # D3 direction: CONFIRMING requires R2_affective NOT to exceed
        # R2_sensory. True here means the reversal RETURNED (a weakens).
        "d3_reversed": bool(
            np.isfinite(r2_affective) and np.isfinite(r2_sensory)
            and r2_affective > r2_sensory
        ),
    }


# --------------------------------------------------------------------------
# MECH-279 -- emit-only pag_n_commits, with the latch cleared per step.
# --------------------------------------------------------------------------
def measure_pag_commits(agent: REEAgent, env_seed: int, steps: int) -> Dict[str, Any]:
    """Count FRESH freeze-commit ticks. See the docstring's sample-size section.

    `_pag_last_output` latches: the gate ticks inside select_action, which
    returns early on a non-E3 tick (agent.py:7416). Clearing before each step
    and counting only non-None reads is what makes n honest.
    """
    env = B.make_env(env_seed)
    _, od = env.reset()
    agent.reset()
    agent.eval()
    n_commits = 0
    n_fresh = 0
    n_latched = 0
    n_active = 0
    from _harness import StepHarness  # noqa: E402

    harness = StepHarness(agent, env, train_mode=False)
    harness.reset()
    with torch.no_grad():
        for _ in range(steps):
            agent._pag_last_output = None   # CLEAR immediately before the call
            result = harness.step(od)
            out = getattr(agent, "_pag_last_output", None)
            if out is None:
                n_latched += 1              # gate did not tick; record NOTHING
            else:
                n_fresh += 1
                if bool(getattr(out, "freeze_commit", False)):
                    n_commits += 1
                if bool(getattr(out, "freeze_active", False)):
                    n_active += 1
            od = result.next_obs_dict
            if result.done:
                _, od = env.reset()
                harness.reset()
    return {
        "pag_n_commits": n_commits,
        "pag_n_freeze_active_ticks": n_active,
        "pag_n_fresh_gate_ticks": n_fresh,
        "pag_n_latched_ticks": n_latched,
        "pag_env_steps": steps,
        "pag_commit_rate_per_gate_tick": (
            float(n_commits) / n_fresh if n_fresh else float("nan")
        ),
    }


# --------------------------------------------------------------------------
# One (arm x seed) cell.
# --------------------------------------------------------------------------
def _build_agent(arm: str, env, od0, tone: float) -> REEAgent:
    """ARM_OFF/ARM_ON use the lineage builder verbatim. The PAG arms override
    the two MECH-279 fields on the returned config rather than editing the
    shared `_lib` baseline module -- editing `_lib` would bust the lineage's
    substrate_hash and refuse every existing sd036_decay baseline."""
    if arm in (ARM_OFF, ARM_ON):
        return B.make_agent(
            env, od0, use_gabaergic_decay=(arm == ARM_ON), gaba_tone=tone
        )
    cfg = B.build_config(env, od0, use_gabaergic_decay=True, gaba_tone=tone)
    cfg.use_pag_freeze_gate = True
    cfg.pag_theta_freeze = float(PAG_THETAS[arm])
    return REEAgent(cfg)


def run_cell(arm: str, seed: int) -> Dict[str, Any]:
    print(f"Seed {seed} Condition {arm}", flush=True)

    use_decay = arm != ARM_OFF
    if arm == ARM_OFF:
        # The OFF arm is the lineage's canonical baseline -> mint it
        # reuse-eligible with the driver EXCLUDED from the hash, so a later
        # sibling with a different driver can match it.
        config_slice = B.off_path_config_slice()
        include_driver = False
    else:
        config_slice = dict(B.off_path_config_slice())
        config_slice["on_arm_flags"] = {
            "use_gabaergic_decay": True,
            "gaba_tone_train": BASELINE_TONE,
            "use_pag_freeze_gate": arm in PAG_THETAS,
            "pag_theta_freeze": PAG_THETAS.get(arm),
        }
        config_slice["tone_sweep"] = list(TONE_SWEEP)
        include_driver = True

    with arm_cell(
        seed,
        config_slice=config_slice,
        script_path=_THIS,
        config_slice_declared=True,
        include_driver_script_in_hash=include_driver,
    ) as cell:
        env = B.make_env(seed)
        _, od0 = env.reset()
        agent = _build_agent(arm, env, od0, BASELINE_TONE)

        floor = B.encoder_floor_norms(agent)
        train_diag = B.train_agent(agent, env, label=f"{arm} seed={seed}", seed=seed)

        # ---- P2: frozen evaluation. No learning, so the SAME trained agent is
        # re-evaluated at several tones without earlier evals changing weights.
        per_tone: List[Dict[str, Any]] = []
        tones = TONE_SWEEP if arm == ARM_ON else [BASELINE_TONE if use_decay else None]
        for tone in tones:
            eval_env = B.make_env(seed)
            if use_decay and agent.gabaergic_decay is not None:
                agent.gabaergic_decay.set_gaba_tone(float(tone))
            traj = B.record_stream_trajectories(agent, eval_env, steps=B.EVAL_STEPS)
            row = {
                "gaba_tone": (float(tone) if tone is not None else None),
                "mode_metrics": B.mode_metrics(traj["modes"]),
            }
            for s in STREAMS:
                row[f"{s}_sustain_ratio"] = B.sustain_ratio(traj[s])
                row[f"{s}_peak"] = float(np.nanmax(traj[s]))
                row[f"{s}_mean"] = float(np.nanmean(traj[s]))
                row[f"{s}_trajectory"] = [float(x) for x in traj[s]]
            per_tone.append(row)

        # ---- SD-011 leg: measured in BOTH regimes (OFF = validated-regime
        # positive control, ON = the never-measured decay regime). Not measured
        # on the PAG arms, whose freeze no-ops would confound the DVs.
        sd011: Optional[Dict[str, Any]] = None
        if arm in (ARM_OFF, ARM_ON):
            if use_decay and agent.gabaergic_decay is not None:
                agent.gabaergic_decay.set_gaba_tone(float(BASELINE_TONE))
            sd011 = measure_sd011_dvs(agent, seed)

        # ---- MECH-279 leg: emit-only, PAG arms only.
        pag: Optional[Dict[str, Any]] = None
        if arm in PAG_THETAS:
            if agent.gabaergic_decay is not None:
                agent.gabaergic_decay.set_gaba_tone(float(BASELINE_TONE))
            pag = measure_pag_commits(agent, seed, steps=B.EVAL_STEPS)
            pag["pag_theta_freeze"] = float(PAG_THETAS[arm])

        row_out: Dict[str, Any] = {
            "arm_id": arm,
            "seed": seed,
            "use_gabaergic_decay": use_decay,
            "use_pag_freeze_gate": arm in PAG_THETAS,
            "per_tone": per_tone,
            "encoder_floor": floor,
            "train_diagnostics": train_diag,
            "sd011_dvs": sd011,
            "mech279_pag": pag,
        }
        cell.stamp(row_out)
        _ZG.observe(agent)

    # Per-cell verdict line (the runner counts these; seeds x conditions).
    if arm == ARM_ON:
        ratios = [r[f"{PRIMARY_STREAM}_sustain_ratio"] for r in per_tone]
        rho = spearman(TONE_SWEEP, ratios)
        spread = float(np.nanmax(ratios) - np.nanmin(ratios))
        row_out["cell_rho_primary"] = rho
        row_out["cell_spread_primary"] = spread
        cell_pass = rho is not None and rho <= C1_RHO_MAX
    else:
        row_out["cell_rho_primary"] = None
        row_out["cell_spread_primary"] = None
        # A control/PAG arm has no dose-response to pass. Its verdict reports
        # whether it produced a usable measurement, so a broken arm is visible
        # in the runner log rather than silently green.
        cell_pass = bool(
            per_tone
            and np.isfinite(per_tone[0][f"{PRIMARY_STREAM}_sustain_ratio"])
            and per_tone[0][f"{PRIMARY_STREAM}_peak"] > 0.0
        )
    row_out["cell_pass"] = bool(cell_pass)
    print(f"verdict: {'PASS' if cell_pass else 'FAIL'}", flush=True)
    return row_out


# --------------------------------------------------------------------------
# Criteria.
# --------------------------------------------------------------------------
def _scored_dv_series(
    on_row: Dict[str, Any], off_row: Optional[Dict[str, Any]], stream: str, dv: str
) -> List[float]:
    """The per-tone series for the stream's scored DV.

    `sustain_ratio` is read directly. `shape_deviation_vs_off` is computed
    against the OFF arm's trajectory -- deliberately NOT against the tone-1.0
    trajectory, because a vs-tone-1.0 reference is V-shaped in tone (zero at the
    reference) and therefore cannot be scored by a MONOTONICITY bar. Against the
    no-decay control it is monotone INCREASING in tone, so C1_RHO_MAX is applied
    to its NEGATION to keep one bar for both DVs. Both reference forms are
    recorded in the manifest so either can be re-derived.
    """
    rows = sorted(on_row["per_tone"], key=lambda r: r["gaba_tone"])
    if dv == "sustain_ratio":
        return [r[f"{stream}_sustain_ratio"] for r in rows]
    ref = None
    if off_row is not None and off_row["per_tone"]:
        ref = np.asarray(off_row["per_tone"][0][f"{stream}_trajectory"], dtype=np.float64)
    if ref is None:
        return [float("nan")] * len(rows)
    # Negated so that "more decay -> larger deviation" reads as a DECREASING
    # series, scored by the same C1_RHO_MAX bar as sustain_ratio.
    return [
        -B.shape_deviation(np.asarray(r[f"{stream}_trajectory"], dtype=np.float64), ref)
        for r in rows
    ]


def evaluate(
    arm_results: List[Dict[str, Any]], readiness: List[Dict[str, Any]]
) -> Dict[str, Any]:
    on_rows = [r for r in arm_results if r["arm_id"] == ARM_ON]
    off_rows = {r["seed"]: r for r in arm_results if r["arm_id"] == ARM_OFF}
    ready_by_seed = {k["seed"]: k for k in readiness}

    # ---- C1: observable #3, per-stream MONOTONICITY over the full sweep ----
    c1_per_seed = []
    for r in on_rows:
        seed = r["seed"]
        rk = ready_by_seed.get(seed, {}).get("streams", {})
        per_stream: Dict[str, Any] = {}
        scoreable: List[str] = []
        for s in STREAMS:
            info = rk.get(s, {})
            dv = info.get("scored_dv")
            if dv is None:
                per_stream[s] = {
                    "scored": False,
                    "reason": "readiness_below_floor_on_both_dvs",
                    "sustain_spread": info.get("sustain_spread"),
                    "shape_spread": info.get("shape_spread"),
                    "dv_ceiling_headroom": info.get("dv_ceiling_headroom"),
                }
                continue
            series = _scored_dv_series(r, off_rows.get(seed), s, dv)
            rho = spearman(TONE_SWEEP, series)
            spread = float(np.nanmax(series) - np.nanmin(series))
            per_stream[s] = {
                "scored": True,
                "scored_dv": dv,
                "measured_rho": rho,
                "threshold_rho": C1_RHO_MAX,
                "measured_spread": spread,
                "series": [float(x) for x in series],
                "monotone": bool(rho is not None and rho <= C1_RHO_MAX),
            }
            scoreable.append(s)
        n_scoreable = len(scoreable)
        # CONFIRMING: every scoreable stream monotone, in one run.
        all_monotone = bool(
            n_scoreable >= C1_MIN_SCOREABLE_STREAMS
            and all(per_stream[s]["monotone"] for s in scoreable)
        )
        c1_per_seed.append(
            {
                "seed": seed,
                "n_scoreable_streams": n_scoreable,
                "scoreable_streams": scoreable,
                "min_scoreable_required": C1_MIN_SCOREABLE_STREAMS,
                "per_stream": per_stream,
                "passed": all_monotone,
            }
        )
    c1_n = sum(1 for s in c1_per_seed if s["passed"])
    c1_passed = c1_n >= C1_SEEDS_REQUIRED
    # Not-scoreable is a DIFFERENT outcome from a failure (D0=A).
    c1_scoreable_seeds = sum(
        1 for s in c1_per_seed if s["n_scoreable_streams"] >= C1_MIN_SCOREABLE_STREAMS
    )
    c1_not_scoreable = c1_scoreable_seeds < C1_SEEDS_REQUIRED

    # ---- C2: SD-011 dissociation under decay (ARM_ON, baseline tone) ----
    c2_per_seed = []
    for r in on_rows:
        d = r.get("sd011_dvs") or {}
        sub = {
            "stream_corr": (d.get("stream_corr"), C2_STREAM_CORR_MAX, "<="),
            "autocorr_gap": (d.get("autocorr_gap"), C2_AUTOCORR_GAP_MIN, ">="),
            "harm_fwd_r2": (d.get("harm_fwd_r2"), C2_HARM_FWD_R2_MIN, ">="),
        }
        checks = {
            "stream_corr_ok": bool(
                d.get("stream_corr") is not None
                and d["stream_corr"] <= C2_STREAM_CORR_MAX
            ),
            "autocorr_gap_ok": bool(
                d.get("autocorr_gap") is not None
                and d["autocorr_gap"] >= C2_AUTOCORR_GAP_MIN
            ),
            "harm_fwd_r2_ok": bool(
                d.get("harm_fwd_r2") is not None
                and np.isfinite(d["harm_fwd_r2"])
                and d["harm_fwd_r2"] >= C2_HARM_FWD_R2_MIN
            ),
            "d3_not_reversed": bool(not d.get("d3_reversed", True)),
        }
        off = off_rows.get(r["seed"], {}).get("sd011_dvs") or {}
        c2_per_seed.append(
            {
                "seed": r["seed"],
                "measured": {k: v[0] for k, v in sub.items()},
                "thresholds": {k: v[1] for k, v in sub.items()},
                "comparators": {k: v[2] for k, v in sub.items()},
                "r2_sensory": d.get("r2_sensory"),
                "r2_affective": d.get("r2_affective"),
                "checks": checks,
                # The validated-regime control, for the across-regime comparison
                # SD-011's CONFIRMING explicitly asks for (autocorr_gap should
                # WIDEN under decay, not collapse).
                "off_regime_reference": {
                    "stream_corr": off.get("stream_corr"),
                    "autocorr_gap": off.get("autocorr_gap"),
                    "harm_fwd_r2": off.get("harm_fwd_r2"),
                    "d3_reversed": off.get("d3_reversed"),
                },
                "autocorr_gap_widened_vs_off": (
                    bool(d.get("autocorr_gap", 0.0) > off.get("autocorr_gap", 0.0))
                    if (d.get("autocorr_gap") is not None
                        and off.get("autocorr_gap") is not None)
                    else None
                ),
                "passed": all(checks.values()),
            }
        )
    c2_n = sum(1 for s in c2_per_seed if s["passed"])
    c2_passed = c2_n >= C2_SEEDS_REQUIRED

    # ---- C3: ON-vs-OFF separation at the baseline tone (854 continuity) ----
    c3_per_seed = []
    for r in on_rows:
        off = off_rows.get(r["seed"])
        by_tone = {t["gaba_tone"]: t for t in r["per_tone"]}
        on_t1 = by_tone.get(BASELINE_TONE)
        if off is None or on_t1 is None or not off["per_tone"]:
            c3_per_seed.append({"seed": r["seed"], "passed": False})
            continue
        on_v = on_t1[f"{PRIMARY_STREAM}_sustain_ratio"]
        off_v = off["per_tone"][0][f"{PRIMARY_STREAM}_sustain_ratio"]
        c3_per_seed.append(
            {
                "seed": r["seed"],
                "measured_on_tone1": on_v,
                "measured_off": off_v,
                "threshold": off_v,   # the bar IS the OFF arm's own value
                "passed": bool(on_v < off_v),
            }
        )
    c3_n = sum(1 for s in c3_per_seed if s["passed"])
    c3_passed = c3_n >= C3_SEEDS_REQUIRED

    # ---- readiness / degeneracy ----
    primary_spreads = [
        ready_by_seed[k]["streams"][PRIMARY_STREAM]["sustain_spread"]
        for k in ready_by_seed
    ]
    min_primary_spread = float(np.nanmin(primary_spreads)) if primary_spreads else 0.0
    primary_ready = min_primary_spread >= READINESS_SPREAD_FLOOR
    vacuous = min_primary_spread <= VACUITY_CEILING

    return {
        "C1_multi_stream_cluster": {
            "load_bearing": True,
            "passed": c1_passed,
            "seeds_passing": c1_n,
            "seeds_required": C1_SEEDS_REQUIRED,
            "threshold_rho": C1_RHO_MAX,
            "min_scoreable_streams": C1_MIN_SCOREABLE_STREAMS,
            "seeds_with_enough_scoreable_streams": c1_scoreable_seeds,
            "not_scoreable": c1_not_scoreable,
            "per_seed": c1_per_seed,
        },
        "C2_sd011_dissociation_under_decay": {
            "load_bearing": False,
            "passed": c2_passed,
            "seeds_passing": c2_n,
            "seeds_required": C2_SEEDS_REQUIRED,
            "thresholds": {
                "stream_corr_max": C2_STREAM_CORR_MAX,
                "autocorr_gap_min": C2_AUTOCORR_GAP_MIN,
                "harm_fwd_r2_min": C2_HARM_FWD_R2_MIN,
            },
            "per_seed": c2_per_seed,
        },
        "C3_on_vs_off_separation": {
            "load_bearing": False,
            "passed": c3_passed,
            "seeds_passing": c3_n,
            "seeds_required": C3_SEEDS_REQUIRED,
            "per_seed": c3_per_seed,
        },
        "combination_rule": (
            "Overall PASS requires C1 ONLY (load-bearing). C1 itself requires, in "
            ">= 2 of 3 seeds, that EVERY stream clearing its per-stream readiness "
            "gate is monotone in tone (Spearman rho <= -0.9) AND that at least 2 "
            "streams clear. C2 and C3 are recorded, not gating."
        ),
        "_readiness": {
            "min_primary_sustain_spread": min_primary_spread,
            "primary_ready": primary_ready,
            "vacuous": vacuous,
        },
    }


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    t0 = time.perf_counter()
    regime = assert_regime()
    seeds = list(B.SEEDS[:1]) if dry_run else list(B.SEEDS)
    arms = [ARM_OFF, ARM_ON] if dry_run else ARMS

    print(f"[regime] {json.dumps(regime)}", flush=True)
    readiness = [readiness_control(s) for s in seeds]
    for k in readiness:
        for s in STREAMS:
            st = k["streams"][s]
            print(
                f"[readiness] seed={k['seed']} {s}: sustain_spread="
                f"{st['sustain_spread']:.3e} shape_spread={st['shape_spread']:.3e} "
                f"scored_dv={st['scored_dv']} headroom={st['dv_ceiling_headroom']:.4f}",
                flush=True,
            )

    arm_results: List[Dict[str, Any]] = []
    for arm in arms:
        for seed in seeds:
            arm_results.append(run_cell(arm, seed))

    acc = evaluate(arm_results, readiness)
    rd = acc.pop("_readiness")
    c1 = acc["C1_multi_stream_cluster"]

    if rd["vacuous"]:
        outcome = "FAIL"
        label = "substrate_not_ready_requeue"
        non_degenerate = False
        degeneracy_reason = (
            f"primary stream {PRIMARY_STREAM} sweep spread "
            f"{rd['min_primary_sustain_spread']:.3e} <= VACUITY_CEILING "
            f"{VACUITY_CEILING:.1e}: the pre-fix feedforward signature has returned."
        )
        direction = "non_contributory"
    elif c1["not_scoreable"]:
        outcome = "FAIL"
        label = "observable_3_not_scoreable"
        non_degenerate = False
        degeneracy_reason = (
            f"fewer than {C1_MIN_SCOREABLE_STREAMS} of {len(STREAMS)} streams cleared "
            "the per-stream readiness gate on either DV, so the multi-stream CLUSTER "
            "has no content at this n. This is an instrument result, NOT a refutation "
            "of SD-036 (precondition (i))."
        )
        direction = "non_contributory"
    else:
        outcome = "PASS" if c1["passed"] else "FAIL"
        label = (
            "multi_stream_cluster_confirmed" if c1["passed"]
            else "streams_degrade_independently"
        )
        non_degenerate = True
        degeneracy_reason = ""
        direction = "supports" if c1["passed"] else "weakens"

    per_claim = {
        "SD-036": direction,
        # SD-011's leg is scored by C2 only when the regime assertions held --
        # and they are asserted at start, so a run that gets here is non-vacuous
        # for SD-011 by construction.
        "SD-011": (
            "unknown" if direction == "non_contributory"
            else ("supports" if acc["C2_sd011_dissociation_under_decay"]["passed"]
                  else "weakens")
        ),
        # EMIT-ONLY. The claim text says "emit MECH-279's pag_n_commits", not
        # score it; no criterion bears on MECH-279, so nothing here may move its
        # confidence in either direction.
        "MECH-279": "unknown",
    }

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    manifest: Dict[str, Any] = {
        "run_id": f"{EXPERIMENT_TYPE}_{ts}_v3",
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": list(CLAIM_IDS),
        "outcome": outcome,
        "timestamp_utc": ts,
        "evidence_direction": direction,
        "evidence_direction_per_claim": per_claim,
        "non_degenerate": non_degenerate,
        "degeneracy_reason": degeneracy_reason,
        "acceptance": acc,
        "arm_results": arm_results,
        "readiness_control": readiness,
        "sd011_regime_declaration": regime,
        "registered_taus": dict(REGISTERED_TAUS),
        "tau_ordering_test": {
            "scored": False,
            "reason": (
                "DROPPED by user decision D2=C (2026-09-19). The clause is ill-posed "
                "as written: the composed pole is (1 - alpha) * exp(-tau * tone) and "
                "SD-036 never mentions alpha, so the substrate predicts the largest "
                "sustain-ratio effect on the SMALLEST-tau stream. Raised to "
                "/governance as a clause-amendment flag instead of scored here."
            ),
        },
        "interpretation": {
            "label": label,
            "preconditions": [
                {
                    "name": f"readiness_sweep_spread_{s}",
                    "description": (
                        f"fixed-tape sweep spread of {s}'s scored DV clears the "
                        "readiness floor (SD-036 precondition (i), EVERY stream)"
                    ),
                    "measured": min(
                        (k["streams"][s]["sustain_spread"] for k in readiness),
                        default=0.0,
                    ),
                    "threshold": READINESS_SPREAD_FLOOR,
                    "direction": "lower",
                    "control": "fixed observation tape, fresh agents, 5-tone sweep",
                    "met": all(k["streams"][s]["ready_any_dv"] for k in readiness),
                }
                for s in STREAMS
            ],
            "criteria_non_degenerate": {
                "C1_multi_stream_cluster": bool(
                    non_degenerate
                    and c1["seeds_with_enough_scoreable_streams"] >= C1_SEEDS_REQUIRED
                ),
                "C2_sd011_dissociation_under_decay": bool(
                    any(
                        (r.get("sd011_dvs") or {}).get("n_steps_measured", 0) > 20
                        for r in arm_results
                        if r["arm_id"] == ARM_ON
                    )
                ),
                "C3_on_vs_off_separation": bool(
                    len([r for r in arm_results if r["arm_id"] == ARM_OFF]) > 0
                ),
            },
        },
        "ethics_preflight": {
            "involves_negative_valence": False,
            "involves_suffering_like_state": False,
            "involves_self_model": False,
            "involves_inescapability_or_helplessness": False,
            "involves_offline_replay_over_harm": False,
            "involves_social_mind_or_language": False,
            "involves_human_data_or_clinical_context": False,
            "decision": "allow",
        },
    }

    # FLAT scalar readout -- the machine-readable projection the indexer reads.
    readout: Dict[str, float] = {
        "c1_seeds_passing": int(c1["seeds_passing"]),
        "c1_seeds_required": int(C1_SEEDS_REQUIRED),
        "c1_passed": int(bool(c1["passed"])),
        "c1_not_scoreable": int(bool(c1["not_scoreable"])),
        "c1_seeds_with_enough_scoreable_streams": int(
            c1["seeds_with_enough_scoreable_streams"]
        ),
        "c2_seeds_passing": int(acc["C2_sd011_dissociation_under_decay"]["seeds_passing"]),
        "c2_passed": int(bool(acc["C2_sd011_dissociation_under_decay"]["passed"])),
        "c3_seeds_passing": int(acc["C3_on_vs_off_separation"]["seeds_passing"]),
        "c3_passed": int(bool(acc["C3_on_vs_off_separation"]["passed"])),
        "min_primary_sustain_spread": float(rd["min_primary_sustain_spread"]),
        "readiness_spread_floor": float(READINESS_SPREAD_FLOOR),
    }
    for s in STREAMS:
        sp = [k["streams"][s]["sustain_spread"] for k in readiness]
        readout[f"readiness_sustain_spread_{s}"] = float(np.nanmin(sp)) if sp else 0.0
        readout[f"readiness_ready_{s}"] = int(
            all(k["streams"][s]["ready_any_dv"] for k in readiness)
        )
    pag_rows = [r for r in arm_results if r["arm_id"] in PAG_THETAS]
    for r in pag_rows:
        p = r.get("mech279_pag") or {}
        key = r["arm_id"]
        readout[f"pag_n_commits_{key}_seed{r['seed']}"] = int(p.get("pag_n_commits", 0))
    # Drop any non-finite value rather than emitting it (a nan IS numeric to the
    # indexer and would pollute a delta; an absent key correctly reads unmeasured).
    manifest["readout"] = {
        k: v for k, v in readout.items()
        if isinstance(v, (int, float)) and np.isfinite(v)
    }

    full_config = {
        "env_kwargs": dict(B.ENV_KWARGS),
        "tone_sweep": list(TONE_SWEEP),
        "baseline_tone": BASELINE_TONE,
        "arms": list(arms),
        "pag_thetas": dict(PAG_THETAS),
        "schedule": {
            "p0_warmup_episodes": B.P0_WARMUP_EPISODES,
            "p1_main_episodes": B.P1_MAIN_EPISODES,
            "steps_per_episode": B.STEPS_PER_EPISODE,
            "eval_steps": B.EVAL_STEPS,
        },
        "thresholds": {
            "C1_RHO_MAX": C1_RHO_MAX,
            "C1_SEEDS_REQUIRED": C1_SEEDS_REQUIRED,
            "C1_MIN_SCOREABLE_STREAMS": C1_MIN_SCOREABLE_STREAMS,
            "C2_STREAM_CORR_MAX": C2_STREAM_CORR_MAX,
            "C2_AUTOCORR_GAP_MIN": C2_AUTOCORR_GAP_MIN,
            "C2_HARM_FWD_R2_MIN": C2_HARM_FWD_R2_MIN,
            "C2_SEEDS_REQUIRED": C2_SEEDS_REQUIRED,
            "C3_SEEDS_REQUIRED": C3_SEEDS_REQUIRED,
            "READINESS_SPREAD_FLOOR": READINESS_SPREAD_FLOOR,
            "VACUITY_CEILING": VACUITY_CEILING,
        },
        "sd011_regime": regime,
        "dry_run": bool(dry_run),
    }
    stamp_recording_core(
        manifest,
        config=full_config,
        seeds=seeds,
        script_path=_THIS,
        started_at=t0,
        z_goal_stream_stats=_ZG.stats(),
    )
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    manifest = run_experiment(dry_run=args.dry_run)
    out_path = write_flat_manifest(
        manifest, EXPERIMENT_TYPE, dry_run=args.dry_run, stamp=False
    )
    print(f"outcome: {manifest['outcome']}", flush=True)
    print(f"manifest: {out_path}", flush=True)

    _raw = str(manifest["outcome"]).upper()
    emit_outcome(
        outcome=_raw if _raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
