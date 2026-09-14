"""V3-EXQ-1037: SD-075 (phasic.ema_episode_continuity) VALIDATION -- new-number, different
mechanism than the braked 779/963 tonic/phasic dissociation lineage.

WHY THIS IS NOT A "V3-EXQ-779c" OR A "V3-EXQ-963c" (read before touching the 779/963 lineage).

MECH-063 sub-claim (ii) carries 5 counted substrate_ceiling autopsies as of this writing
(777a-cluster, 779b, 963, 963a, plus the legacy 164a co-tag) -- the re-derive brake is fired hard
on that claim and this script does NOT touch it: `CLAIM_IDS = ["SD-075"]` only, and the driver
below runs PHASIC-ONLY (no `use_tonic_vigor`, no dual-regulator TONIC x PHASIC grid). It is not a
lettered iteration of the tonic/phasic dissociation probe against the current regulator; it is a
narrower, phasic-only measurement of SD-075's own accounting mechanism, which SD-075's own claim
notes explicitly permit ("a redesign of a DIFFERENT mechanism under a new EXQ number remains
permitted").

SD-075 has ZERO counted substrate_ceiling autopsies of its own (checked this session via the
/queue-experiment Step 2.5b recipe against every failure_autopsy_*.json in
REE_assembly/evidence/planning/) -- it is not braked; MECH-063 is.

SD-075's own claim record (`claims.yaml`) states: "validation_experiment: none queued -- blocked
by the MECH-063 re-derive brake (no V3-EXQ-779c); a successor must be a new-number redesign of a
different mechanism, not a lettered tonic/phasic iteration." The design doc's own Retest section
(`REE_assembly/docs/architecture/sd_075_phasic_ema_episode_continuity.md`) says the same, and adds
the two hard requirements any retest must satisfy: declare `baseline_continuity="carry"` plus a
convergence gate, and consume `n_events_converged` with an UNINFORMATIVE-CELL PATH rather than a
raw MIN. Both are implemented below (see INFORMATIVE-CELL PATH).

RELATIONSHIP TO V3-EXQ-952 (the diagnostic that preceded this). V3-EXQ-952
(`v3_exq_952_sd075_phasic_warmup_rescue_diagnostic`, 2026-08-28) answered the substrate-readiness
question SD-075's implementation note left open (does agent training let a carry+gate cell clear
MIN_EVENT_TICKS on n_events_converged) with `CLAIM_IDS = []` -- deliberately non-contributory,
by design, because a diagnostic's job is readiness, not a claim verdict. It found
phasic_warmup_rescue_confirmed on seeds {11, 23, 29} at warmup_episodes=40. THIS script is the
claim-tagged successor 952 was gating: `EXPERIMENT_PURPOSE = "evidence"`, `CLAIM_IDS = ["SD-075"]`,
and it adds the uninformative-cell consumption path 952 did not need (952's own criterion was a
raw MIN over the warmed arm; that is exactly the pattern the design doc's Retest section forbids
for a claim-tagged run). GOV-REUSE-1 checked THIS session (reanalysis_query.py --readout
n_events_converged / min_warmed_n_events_converged): no recorded manifest -- including 952's own
-- carries `n_events_converged` as a flat `readout` key (952 stores it only inside
`arm_results[]`, which the reuse tool does not index), so the decisive readout is not
post-hoc-derivable from existing manifests; not recoverable, so this experiment runs rather than
reanalyzing. Separately, 952 ran BEFORE SD-104/SD-105 landed (2026-09-04) and did not set
`phasic_burst_refractory_ticks` / `phasic_burst_extinction_level`, so its substrate_hash predates
this run's config regardless -- not a reusable cell for this question even if the readout had been
recorded flat.

WARMUP DOSE (40, matching 952 -- see RE-REVIEW below for why an originally-planned dose of 10 was
dropped). 952 measured, on THIS exact env/phasic config and READ_MAX_ENV_STEPS=600 budget,
n_converged_ticks of 67/87/189 across seeds {23, 11, 29} at warmup_episodes=40 -- tick
accumulation is refractory-independent (ticks occur regardless of whether a tick can FIRE; only
confirmed by simulation this session, see RATE-BASED CRITERION below), so these counts transfer
directly to this driver's refractory-bounded regulator and give real, measured margin above
INFORMATIVE_TICKS_FLOOR=60 on every seed. Reusing 952's own dose does not make this a re-run of
952: 952 ran with the SD-104 knobs at their no-op defaults (before SD-104 existed) and measured a
raw, unbounded event COUNT with `CLAIM_IDS=[]`; this driver measures a refractory-bounded event
RATE with `CLAIM_IDS=["SD-075"]` -- a materially different regulator configuration and a
materially different (claim-tagged) measurement, even at the same warmup exposure.

RED-TEAM (fable): BLOCKING on the first draft, FIXED (see RATE-BASED CRITERION below), CONTESTED
on the re-spawn, both named confirmers addressed (see RE-REVIEW below). First-draft finding,
independently confirmed by direct simulation this session (a
synthetic always-triggering surprise stream at refractory_ticks=29 fired 19 events over 570
converged ticks -- exactly floor((570-1)/30)+1 = 19, the refractory's own tick() logic: a fired
event blocks any further firing for `refractory_ticks` ticks regardless of how strongly the next
tick's surprise clears the trigger threshold): the ORIGINAL design used an absolute-count
criterion (n_events_converged >= MIN_EVENT_TICKS=10) inherited unchanged from the 779b/952
lineage, which measured that count under refractory_ticks=0 (no cap). Under this script's
refractory_ticks=29, the maximum POSSIBLE n_events_converged in N converged ticks is
floor((N-1)/30)+1 -- reaching 10 requires N >= 271 converged ticks, and 952's own warmed cells
(same env/budget, refractory=0) measured only 67-189 converged ticks at READ_MAX_ENV_STEPS=600.
So the count bar was arithmetically unreachable at this budget regardless of whether SD-075's
carry+gate mechanism actually works, and a FAIL would have been misrouted to
`evidence_direction="weakens"` -- an instrument-ceiling artifact read as a claim disconfirmation.
Fix: the load-bearing criterion is now a REFRACTORY-CEILING-RELATIVE RATE, not an absolute count
-- see RATE-BASED CRITERION below, which is reachable at ANY read budget long enough to estimate
a rate at all, independent of READ_MAX_ENV_STEPS.

RATE-BASED CRITERION (the fix). `MAX_EVENT_RATE_CEILING = 1 / (refractory_ticks + 1)` (~=0.0333
at refractory_ticks=29) is the theoretical maximum fraction of converged ticks that can fire an
event, by construction of the refractory gate itself (confirmed by the simulation above: a
stream triggering on EVERY tick saturates exactly at this ceiling). `MIN_EVENT_RATE = 0.5 *
MAX_EVENT_RATE_CEILING` (~=0.0167) is the load-bearing bar: a warmed, informative cell must fire
at least half as often as the refractory bound physically permits. This is reachable at this
script's read budget (READ_MAX_ENV_STEPS=600, unchanged from 952 -- no budget increase needed,
since a RATE estimate does not require the large N an ABSOLUTE count needed) and is
ceiling-relative rather than tied to a read-budget-dependent absolute count, so it does not
recur if a future author changes the read budget or the refractory value. The retained
`n_events_converged` field is now purely descriptive (recorded for comparability with 952), not
gating.

RE-REVIEW (fable, second and final pass per policy -- re-spawn exactly once on a BLOCKING finding
that changed the causal chain, never iterate to CLEAR). Verdict CONTESTED, two named confirmers,
both addressed rather than left open:
  (1) "the row does not record `n_events_refractory_suppressed`, so a `weakens` verdict cannot be
  told apart from refractory MASKING a real effect (surprise detected, but inside a refractory
  window so it does not fire or count)." Confirmed the field exists on `get_state()`
  (`phasic_surprise_burst.py`, tracked since SD-104). FIXED: now recorded per cell below, purely
  descriptive (not gating) -- a reader can see directly whether a `weakens` cell had suppressed
  detections (masking) or genuinely few detections (a real null).
  (2) "the only warmup=10 data point (the design doc's own spike, a different config) put seed 29
  at ~47 converged ticks at this budget -- below INFORMATIVE_TICKS_FLOOR=60, i.e. warmup=10 might
  route to a wasted non_contributory run instead of a real test." FIXED by dropping the
  warmup=10 arm entirely rather than attempting a live pre-verification (a direct
  seed=29/warmup=10 probe cell was attempted this session and abandoned after it ran past 15
  minutes with no output -- a single-cell warmup+read at this scale is expensive to pre-verify
  live, and 952 already supplies the needed measurement at a DIFFERENT, already-informative dose):
  WARMUP_CONDITIONS is now `[0, 40]`, matching 952's own measured-informative dose -- see WARMUP
  DOSE above. This is the disposition, not a deferral: dose reachability at warmup=40 is grounded
  in 952's actual per-seed numbers, not an extrapolation.
  Not fixed, and not claimed clear: the re-review's Family-2 point that a live agent's detection
  rate under the SD-104-bounded regime was not directly measured (only inferred from 952's
  unbounded counts, which are between the bar and its ceiling with ~1.7x/~5x margin) remains an
  assumption this run's OWN R0 precondition and per-cell readout will confirm or refute --
  exactly the kind of thing a claim-tagged run is for, not something owed before queuing it.

SD-104/SD-105 SUBSTRATE-PATH OVERLAP (skill Step 2.5c). One OPEN `degrading`-severity
substrate_queue entry names a file this driver imports at module level:
  `sd_phasic_burst_decay_and_warmup_headroom` (severity degrading, status
  implemented_pending_validation) names `ree_core/regulators/phasic_surprise_burst.py` and
  `ree_core/regulators/selection_entropy_floor.py`. Its failure record: on a WARMED agent under
  the 779/963-lineage's TONIC+PHASIC dissociation config (no refractory/extinction bound), phasic
  event-tick rate reached 0.390-0.884 of ticks (vs 779a's healthy 0.007-0.136) -- the burst does
  not decay fast enough post-trigger on a warmed agent, inflating the event count in a way that is
  a regulator artifact, not a genuine surprise reading. SD-105's leg (tonic-side sustained-entropy
  headroom) does not apply here at all -- this driver never enables `use_tonic_vigor` /
  `use_selection_entropy_floor`; the TONIC axis and the dual-regulator interaction the 963a
  failure record concerns are simply not in this driver's causal path. For the PHASIC leg (SD-104),
  this driver does not merely note the defect and proceed (that is the correct handling for
  cosmetic overlap, but this defect is a direct threat to the decisive readout) -- it DEFENSIVELY
  BOUNDS it: `phasic_burst_refractory_ticks=29`, `phasic_burst_extinction_level=0.05`, the exact
  values SD-104's own positive-control contract (A6, test_sd104_sd105_burst_decay_and_entropy_
  headroom.py) measured as reproducing 779a's healthy 0.007-0.136 event-tick-rate band instead of
  963a's 0.390-0.884. Both knobs are no-op-by-default and additive to SD-075's own two fields;
  Step 2.5a below confirms end-to-end wiring. This is the DIFFERENT-mechanism redesign choice this
  script makes relative to 952 (which ran with the knobs at their no-op defaults, before SD-104
  existed): a claim-tagged run cannot inherit an unbounded-duty-cycle regulator artifact into its
  decisive readout the way a pre-SD-104 diagnostic reasonably could.

STEP 2.5a EMPIRICAL CONFIRMATION (this session, before writing this file). Instantiated
`PhasicSurpriseBurst` directly with `baseline_continuity="carry", warmup_ticks=-1,
refractory_ticks=29, extinction_level=0.05` and read `get_state()`: `warmup_ticks=30`
(DERIVE resolved correctly), `refractory_ticks=29`, `extinction_level=0.05`,
`burst_duty_cycle_bound=0.1667`, `burst_duty_cycle_within_bound=True` at zero lifetime ticks.
Then built a full `REEConfig` -> `REEAgent` with the same seven `phasic_burst_*` fields set via
the config surface (not the regulator constructor directly) and confirmed `agent.phasic_burst`
wires through identically (`refractory_ticks=29`, `extinction_level=0.05`,
`baseline_continuity="carry"`, `warmup_ticks=30`). Doc and runtime agree.

INFORMATIVE-CELL PATH (the design doc's second hard requirement). Per cell, `n_converged_ticks`
is compared against `INFORMATIVE_TICKS_FLOOR = 2 * (refractory_ticks + 1)` (60 at
refractory_ticks=29) -- a floor tied to the REFRACTORY SPAN (the quantity the rate criterion
actually reads), not to `warmup_ticks_resolved` (the EMA-convergence span, which the ORIGINAL
draft used and the red-team correctly flagged as "protects nothing": 952's own worst warmed cell
already cleared 30 converged ticks by a wide margin, so a 30-tick floor is satisfied at real-run
scale before it can ever do any gating work). 60 ticks gives room for up to 2 possible firings at
the refractory ceiling -- enough to distinguish "some firing" from "zero firing" without
requiring the READ_MAX_ENV_STEPS increase an absolute-count criterion would have needed. A cell
with `n_converged_ticks < INFORMATIVE_TICKS_FLOOR` is marked `informative=False` and EXCLUDED
from the load-bearing criterion's MIN rather than folded in as a near-zero rate -- this is the
literal mechanism the design doc requires ("must consume n_events_converged with an
uninformative-cell path rather than feeding a raw MIN"). At this driver's read budget
(READ_MAX_ENV_STEPS=600; 952 measured 67-189 converged ticks per warmed cell at this same budget)
every WARMED cell is expected to be informative in the full run; a short `--dry-run` budget (40
env steps) is expected to exercise the uninformative branch directly, since 40 env steps cannot
accumulate 60 post-convergence ticks -- this is intentional smoke coverage of the branch, not a
design accident.

CLAIM_IDS = ["SD-075"] only. This experiment does not test MECH-063 -- no tonic axis, no
dissociation comparison; it measures whether SD-075's own carry+gate mechanism, on a phasic-only
config with the SD-104 duty-cycle bound applied, produces an INFORMATIVE post-convergence event
RATE that (a) is honestly gated (declares itself uninformative rather than reporting a truncated
near-zero) and (b) clears MIN_EVENT_RATE -- half of the refractory-imposed ceiling -- once a
modest (40-episode) warmup exposure is applied. A FAIL here is evidence AGAINST SD-075's
mechanism (the carry+gate combination does not, in fact, produce usable accounting at this
warmup dose) -- not evidence about MECH-063, which stays untouched and un-retested.

MECH-094: warm_agent's warmup phase is P0-style forward-model de-saturation only (SD-074's own
protocol, matching 952); no head is trained on z_world/z_harm/E3 output, so phased-training does
not apply. The READ phase is train_mode=False (eval only, no gradient step, no memory write).
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

_THIS = Path(__file__).resolve()
_REE_V3 = _THIS.parent.parent
if str(_REE_V3) not in sys.path:
    sys.path.insert(0, str(_REE_V3))

from ree_core.utils.config import REEConfig  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.regulators.phasic_surprise_burst import (  # noqa: E402
    PhasicSurpriseBurst,
    PhasicSurpriseBurstConfig,
)
from experiments._harness import StepHarness  # noqa: E402
from experiments._lib.probe_warmup import WarmupRecipe, warm_agent  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402
from experiment_protocol import emit_outcome  # noqa: E402
from experiments.pack_writer import write_flat_manifest, flat_readout  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_1037_sd075_phasic_ema_episode_continuity_validation"
EXPERIMENT_PURPOSE = "evidence"
CLAIM_IDS: List[str] = ["SD-075"]

# regulator_fires_at_all reuses the exact burst_level computation independently validated live
# in 779/779a/779b/952 (burst_level_max measured 1.00 in every one); not a new or retuned gate.
ANCHOR_REACHABILITY_EXEMPT = (
    "regulator_fires_at_all reuses PhasicSurpriseBurst.burst_level, already confirmed >0.05 "
    "(measured 1.00) in 779/779a/779b/952; not a new or retuned gate."
)

# ---- Pre-registered constants ----
SEEDS = [11, 23, 29]                    # 779b/952's starvation-category seeds (mild/worst/severe).
WARMUP_CONDITIONS = [0, 40]             # 0 = untrained control; 40 = 952's OWN warmed dose --
                                         # switched from an originally-planned 10 after the
                                         # second red-team pass (see RE-REVIEW section below):
                                         # 952 measured n_converged_ticks 67/87/189 at warmup=40
                                         # on this exact 600-step budget (refractory-independent,
                                         # since tick accumulation does not depend on whether a
                                         # tick can FIRE), comfortably clearing
                                         # INFORMATIVE_TICKS_FLOOR=60 with real margin on every
                                         # seed; warmup=10 has no such measured precedent at this
                                         # budget and was assessed, not verified, informative.
WARMED_EPISODES = max(WARMUP_CONDITIONS)
STEPS_PER_EPISODE = 300                 # unchanged from 779/779a/779b/952.

READ_MAX_ENV_STEPS = 600                # unchanged from 952.
READ_MAX_EPISODES = 200

EVENT_LEVEL_FLOOR = 0.05                # unchanged from 779b/952.
# NOT the 779b/952 count bar (MIN_EVENT_TICKS=10, calibrated under refractory_ticks=0) -- see
# module docstring RED-TEAM / RATE-BASED CRITERION. Under refractory bounding, event RATE, not
# an absolute count, is the criterion that stays reachable independent of read budget.

ENV_SIZE = 8
ENV_HAZARDS = 2
ENV_RESOURCES = 3
ENV_DRIFT_SOURCES = 3
ENV_DRIFT_POLICY = "random_walk"

# PHASIC axis -- byte-identical to 779b/952 except the SD-104 duty-cycle bound added below.
PHASIC_SOURCE = "instantaneous_pe"
PHASIC_TRIGGER_RATIO = 1.2
PHASIC_EMA_DECAY = 0.1
PHASIC_TEMP_DELTA = -0.5
PHASIC_DECAY = 0.5
PHASIC_TRIGGER_FLOOR = 1e-6
PHASIC_MIN_T = 0.1
# SD-075 fix, held FIXED across the whole grid (the swept variable is warmup only).
PHASIC_BASELINE_CONTINUITY = "carry"
PHASIC_WARMUP_TICKS = -1                 # DERIVE = ceil(3 / PHASIC_EMA_DECAY) = 30
# SD-104 duty-cycle bound, held FIXED -- defensive against the open degrading substrate_queue
# entry sd_phasic_burst_decay_and_warmup_headroom (see module docstring). Values are SD-104's
# own A6 positive-control measurement reproducing the healthy 779a event-tick-rate band.
PHASIC_REFRACTORY_TICKS = 29
PHASIC_EXTINCTION_LEVEL = 0.05

# RATE-BASED CRITERION constants (module docstring RED-TEAM section). A fired event blocks
# further firing for PHASIC_REFRACTORY_TICKS ticks (phasic_surprise_burst.py tick()), so the
# maximum possible event RATE over any converged window is exactly this ceiling -- confirmed by
# direct simulation this session (an always-triggering synthetic stream saturates at exactly
# this value). An absolute event COUNT bar does not share this property: it depends on the
# read-budget-determined N of converged ticks, which is why the original MIN_EVENT_TICKS=10
# draft (calibrated under refractory_ticks=0, where no such ceiling exists) was unreachable here.
MAX_EVENT_RATE_CEILING = 1.0 / (PHASIC_REFRACTORY_TICKS + 1)   # ~= 0.0333
MIN_EVENT_RATE = 0.5 * MAX_EVENT_RATE_CEILING                  # ~= 0.0167 -- the load-bearing bar
# Informative-cell floor: tied to the REFRACTORY SPAN (what the rate criterion reads), not to
# warmup_ticks_resolved (what the EMA-convergence gate reads) -- see module docstring
# INFORMATIVE-CELL PATH. 2 refractory spans gives room for up to 2 possible firings.
INFORMATIVE_TICKS_FLOOR = 2 * (PHASIC_REFRACTORY_TICKS + 1)    # 60

PROGRESS_EVERY_ENV_STEPS = 100

_ZG = ZGoalStreamAccumulator()


def _mk_env() -> CausalGridWorldV2:
    return CausalGridWorldV2(
        size=ENV_SIZE,
        num_hazards=ENV_HAZARDS,
        num_resources=ENV_RESOURCES,
        background_drift_enabled=True,
        n_drift_sources=ENV_DRIFT_SOURCES,
        drift_policy=ENV_DRIFT_POLICY,
    )


def _mk_config(env: CausalGridWorldV2) -> REEConfig:
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
    )
    cfg.use_control_vector_logging = True
    cfg.hippocampal.use_action_class_scaffold_candidates = True
    cfg.use_noise_floor = False
    cfg.use_phasic_burst = True
    cfg.phasic_burst_signal_source = PHASIC_SOURCE
    cfg.phasic_burst_trigger_ratio = PHASIC_TRIGGER_RATIO
    cfg.phasic_burst_surprise_ema_decay = PHASIC_EMA_DECAY
    cfg.phasic_burst_temp_delta = PHASIC_TEMP_DELTA
    cfg.phasic_burst_decay = PHASIC_DECAY
    cfg.phasic_burst_trigger_floor = PHASIC_TRIGGER_FLOOR
    cfg.phasic_burst_min_temperature = PHASIC_MIN_T
    cfg.phasic_burst_baseline_continuity = PHASIC_BASELINE_CONTINUITY
    cfg.phasic_burst_warmup_ticks = PHASIC_WARMUP_TICKS
    cfg.phasic_burst_refractory_ticks = PHASIC_REFRACTORY_TICKS
    cfg.phasic_burst_extinction_level = PHASIC_EXTINCTION_LEVEL
    return cfg


def _config_slice(warmup_episodes: int) -> Dict[str, Any]:
    """Fingerprint config slice: env + phasic config + the swept warmup exposure."""
    return {
        "env_size": ENV_SIZE,
        "env_hazards": ENV_HAZARDS,
        "env_resources": ENV_RESOURCES,
        "env_drift_sources": ENV_DRIFT_SOURCES,
        "env_drift_policy": ENV_DRIFT_POLICY,
        "phasic_burst_signal_source": PHASIC_SOURCE,
        "phasic_burst_trigger_ratio": PHASIC_TRIGGER_RATIO,
        "phasic_burst_surprise_ema_decay": PHASIC_EMA_DECAY,
        "phasic_burst_temp_delta": PHASIC_TEMP_DELTA,
        "phasic_burst_decay": PHASIC_DECAY,
        "phasic_burst_trigger_floor": PHASIC_TRIGGER_FLOOR,
        "phasic_burst_min_temperature": PHASIC_MIN_T,
        "phasic_burst_baseline_continuity": PHASIC_BASELINE_CONTINUITY,
        "phasic_burst_warmup_ticks": PHASIC_WARMUP_TICKS,
        "phasic_burst_refractory_ticks": PHASIC_REFRACTORY_TICKS,
        "phasic_burst_extinction_level": PHASIC_EXTINCTION_LEVEL,
        "warmup_episodes": warmup_episodes,
        "steps_per_episode": STEPS_PER_EPISODE,
        "read_max_env_steps": READ_MAX_ENV_STEPS,
        "read_max_episodes": READ_MAX_EPISODES,
    }


def _fresh_regulator(agent: REEAgent) -> None:
    """Reinstall a zero-lifetime regulator so the READ phase's convergence-gate
    accounting starts at tick zero regardless of the warmup path taken (cache-hit
    vs cache-miss warmup consume different numbers of regulator ticks otherwise --
    same reasoning as 952's _fresh_regulator)."""
    agent.phasic_burst = PhasicSurpriseBurst(
        config=PhasicSurpriseBurstConfig(
            enabled=True,
            surprise_ema_decay=PHASIC_EMA_DECAY,
            trigger_ratio=PHASIC_TRIGGER_RATIO,
            trigger_floor=PHASIC_TRIGGER_FLOOR,
            temp_delta=PHASIC_TEMP_DELTA,
            decay=PHASIC_DECAY,
            min_temperature=PHASIC_MIN_T,
            baseline_continuity=PHASIC_BASELINE_CONTINUITY,
            warmup_ticks=PHASIC_WARMUP_TICKS,
            refractory_ticks=PHASIC_REFRACTORY_TICKS,
            extinction_level=PHASIC_EXTINCTION_LEVEL,
        )
    )


def _run_cell(
    seed: int, warmup_episodes: int, cell_idx: int, n_cells: int,
) -> Dict[str, Any]:
    print("Seed %d Condition warmup%d" % (seed, warmup_episodes), flush=True)
    with arm_cell(
        seed,
        config_slice=_config_slice(warmup_episodes),
        script_path=_THIS,
        config_slice_declared=True,
    ) as cell:
        env = _mk_env()
        cfg = _mk_config(env)
        agent = REEAgent(cfg)

        recipe = WarmupRecipe(num_episodes=int(warmup_episodes),
                               steps_per_episode=STEPS_PER_EPISODE)
        env_kwargs = {
            "size": ENV_SIZE, "num_hazards": ENV_HAZARDS,
            "num_resources": ENV_RESOURCES, "background_drift_enabled": True,
            "n_drift_sources": ENV_DRIFT_SOURCES, "drift_policy": ENV_DRIFT_POLICY,
        }
        print("  [warmup] seed=%d warmup_episodes=%d starting" % (seed, warmup_episodes),
              flush=True)
        warm_out = warm_agent(
            agent, env, seed=seed, recipe=recipe, env_kwargs=env_kwargs,
            label="v3_exq_1037 cell%d/%d" % (cell_idx + 1, n_cells), measure=False,
        )
        _fresh_regulator(agent)

        harness = StepHarness(agent, env, train_mode=False, seed=seed)
        burst_max = 0.0
        n_event_window_ticks = 0
        ep_lengths: List[int] = []
        steps = 0
        agent.eval()
        with torch.no_grad():
            for ep in range(READ_MAX_EPISODES):
                _flat, obs_dict = env.reset()
                agent.reset()
                harness.reset()
                ep_len = 0
                for _ in range(STEPS_PER_EPISODE):
                    r = harness.step(obs_dict)
                    obs_dict = r.next_obs_dict
                    steps += 1
                    ep_len += 1
                    lvl = float(agent.phasic_burst.burst_level)
                    burst_max = max(burst_max, lvl)
                    if lvl >= EVENT_LEVEL_FLOOR:
                        n_event_window_ticks += 1
                    if (steps == 1) or (steps % PROGRESS_EVERY_ENV_STEPS == 0):
                        print(
                            "  [read] warmup%d seed=%d "
                            "ep %d/%d env-steps (episode %d) event_ticks=%d"
                            % (warmup_episodes, seed, steps, READ_MAX_ENV_STEPS,
                               ep + 1, n_event_window_ticks),
                            flush=True,
                        )
                    if r.done or steps >= READ_MAX_ENV_STEPS:
                        break
                ep_lengths.append(ep_len)
                if steps >= READ_MAX_ENV_STEPS:
                    break

        _ZG.observe(agent)
        st = agent.phasic_burst.get_state()
        warmup_ticks_resolved = int(st["warmup_ticks"])
        n_converged_ticks = int(st["n_converged_ticks"])
        n_events_converged = int(st["n_events_converged"])
        # INFORMATIVE-CELL PATH: the read must expose the CONVERGED regime for at least
        # INFORMATIVE_TICKS_FLOOR ticks (2 refractory spans) before a low/zero rate is trusted
        # as a real measurement rather than a truncated sample. See module docstring.
        informative = bool(n_converged_ticks >= INFORMATIVE_TICKS_FLOOR)
        converged_event_rate = (
            (n_events_converged / n_converged_ticks) if n_converged_ticks > 0 else 0.0
        )
        row: Dict[str, Any] = {
            "seed": seed,
            "warmup_episodes": warmup_episodes,
            "warmup_cache_hit": bool(warm_out.cache_hit),
            "n_read_env_steps": steps,
            "n_read_episodes": len(ep_lengths),
            "mean_episode_len": (statistics.fmean(ep_lengths) if ep_lengths else 0.0),
            "burst_level_max": float(burst_max),
            "n_event_window_ticks": int(n_event_window_ticks),
            "lifetime_ticks": int(st["lifetime_ticks"]),
            "lifetime_episodes": int(st["lifetime_episodes"]),
            "warmup_ticks_resolved": warmup_ticks_resolved,
            "n_events_converged": n_events_converged,
            "n_converged_ticks": n_converged_ticks,
            "converged_event_rate": converged_event_rate,
            "n_events_prewarmup": int(st["n_events_prewarmup"]),
            # RE-REVIEW confirmer (1): descriptive only, not gating -- lets a reader tell a
            # genuinely-low-surprise `weakens` cell apart from one where detections existed
            # but were masked by the refractory window (SD-104's own counter).
            "n_events_refractory_suppressed": int(st["n_events_refractory_suppressed"]),
            "baseline_continuity": str(st["baseline_continuity"]),
            "refractory_ticks": int(st["refractory_ticks"]),
            "extinction_level": float(st["extinction_level"]),
            "burst_duty_cycle_bound": st.get("burst_duty_cycle_bound"),
            "realised_burst_duty_cycle": float(st["realised_burst_duty_cycle"]),
            "burst_duty_cycle_within_bound": st.get("burst_duty_cycle_within_bound"),
            "informative_ticks_floor": INFORMATIVE_TICKS_FLOOR,
            "informative": informative,
            "meets_min_event_rate": bool(
                informative and converged_event_rate >= MIN_EVENT_RATE),
        }
        cell.stamp(row)
    print("verdict: %s (informative=%s)"
          % ("PASS" if row["meets_min_event_rate"] else "FAIL", row["informative"]),
          flush=True)
    return row


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    global SEEDS, WARMUP_CONDITIONS, READ_MAX_ENV_STEPS, READ_MAX_EPISODES
    seeds = SEEDS[:1] if dry_run else SEEDS
    warmups = [0, 2] if dry_run else WARMUP_CONDITIONS
    read_max_env_steps = 40 if dry_run else READ_MAX_ENV_STEPS
    read_max_episodes = 5 if dry_run else READ_MAX_EPISODES
    if dry_run:
        READ_MAX_ENV_STEPS = read_max_env_steps
        READ_MAX_EPISODES = read_max_episodes

    cells = [(w, s) for w in warmups for s in seeds]
    rows: List[Dict[str, Any]] = []
    for idx, (w, s) in enumerate(cells):
        rows.append(_run_cell(s, w, idx, len(cells)))

    warmed_episode_value = max(warmups)

    # ---- Readiness precondition R0: the regulator fires at all (positive control) ----
    max_burst = max((r["burst_level_max"] for r in rows), default=0.0)
    r0_fires = bool(max_burst > EVENT_LEVEL_FLOOR)

    # ---- Load-bearing criterion: informative warmed cells clear MIN_EVENT_RATE ----
    # RATE, not an absolute count -- see module docstring RED-TEAM / RATE-BASED CRITERION.
    warmed_rows = [r for r in rows if r["warmup_episodes"] == warmed_episode_value]
    control_rows = [r for r in rows if r["warmup_episodes"] == 0]
    informative_warmed_rows = [r for r in warmed_rows if r["informative"]]
    uninformative_warmed_rows = [r for r in warmed_rows if not r["informative"]]
    all_warmed_informative = bool(warmed_rows and not uninformative_warmed_rows)

    min_warmed_informative_rate = min(
        (r["converged_event_rate"] for r in informative_warmed_rows), default=None)
    min_control_rate = min((r["converged_event_rate"] for r in control_rows), default=0.0)
    max_control_rate = max((r["converged_event_rate"] for r in control_rows), default=0.0)
    worst_warmed_cell = min(informative_warmed_rows, key=lambda r: r["converged_event_rate"]) \
        if informative_warmed_rows else None

    rescue_confirmed = bool(
        r0_fires and all_warmed_informative
        and min_warmed_informative_rate is not None
        and min_warmed_informative_rate >= MIN_EVENT_RATE
    )
    non_degenerate = bool(
        informative_warmed_rows and control_rows and min_warmed_informative_rate is not None and
        (min_warmed_informative_rate != max_control_rate
         or min_warmed_informative_rate != min_control_rate)
    )

    if not r0_fires:
        outcome = "FAIL"
        label = "substrate_not_ready"
        evidence_direction = "non_contributory"
        degeneracy_reason = "R0 unmet: regulator never fired (burst_level_max <= floor) " \
                             "across the whole grid -- capability failure, unrelated to SD-075."
        non_degenerate = False
    elif not all_warmed_informative:
        outcome = "FAIL"
        label = "phasic_carry_gate_warmed_cell_uninformative"
        evidence_direction = "non_contributory"
        degeneracy_reason = (
            "%d/%d warmed cell(s) had n_converged_ticks below INFORMATIVE_TICKS_FLOOR=%d -- "
            "the read budget did not sustain the converged regime long enough to estimate a "
            "rate; raise READ_MAX_ENV_STEPS/EPISODES rather than feeding a truncated-sample "
            "rate into the MIN."
            % (len(uninformative_warmed_rows), len(warmed_rows), INFORMATIVE_TICKS_FLOOR)
        )
    elif rescue_confirmed:
        outcome = "PASS"
        label = "phasic_carry_gate_achieves_informative_rescue"
        evidence_direction = "supports"
        degeneracy_reason = None
    else:
        outcome = "FAIL"
        label = "phasic_carry_gate_rescue_insufficient"
        evidence_direction = "weakens"
        degeneracy_reason = None

    interpretation: Dict[str, Any] = {
        "label": label,
        "preconditions": [
            {
                "name": "regulator_fires_at_all",
                "kind": "capability",
                "control": "max burst_level_max across the whole grid (positive control)",
                "measured": max_burst,
                "threshold": EVENT_LEVEL_FLOOR,
                "direction": "lower",
                "met": bool(r0_fires),
            },
        ],
        "criteria": [
            {
                "name": "phasic_carry_gate_achieves_informative_rescue",
                "load_bearing": True,
                "passed": bool(rescue_confirmed),
                "all_warmed_cells_informative": all_warmed_informative,
                "measured_min_warmed_informative_converged_event_rate": min_warmed_informative_rate,
                "threshold": MIN_EVENT_RATE,
                "max_event_rate_ceiling": MAX_EVENT_RATE_CEILING,
                "control_converged_event_rate_range": [min_control_rate, max_control_rate],
                "offending_cell": (
                    {"seed": worst_warmed_cell["seed"],
                     "warmup_episodes": worst_warmed_cell["warmup_episodes"],
                     "converged_event_rate": worst_warmed_cell["converged_event_rate"],
                     "n_events_converged": worst_warmed_cell["n_events_converged"],
                     "n_converged_ticks": worst_warmed_cell["n_converged_ticks"]}
                    if worst_warmed_cell else None
                ),
            },
        ],
        "criteria_non_degenerate": {
            "phasic_carry_gate_achieves_informative_rescue": non_degenerate,
        },
        "summary": (
            "EVIDENCE for SD-075 (sd_phasic_ema_episode_continuity), NOT a MECH-063 retest -- "
            "phasic-only, no tonic axis, no dissociation comparison. Tests whether SD-075's "
            "carry-continuity + convergence-gate mechanism, defensively bounded against the "
            "open sd_phasic_burst_decay_and_warmup_headroom duty-cycle defect via SD-104's "
            "refractory_ticks/extinction_level knobs, produces an INFORMATIVE (not a "
            "truncated-sample) post-convergence event RATE that clears MIN_EVENT_RATE (half "
            "the refractory-imposed ceiling) once a 40-episode warmup is applied on the "
            "worst-case starvation-category seeds {11, 23, 29}. "
            "phasic_carry_gate_achieves_informative_rescue = PASS supports SD-075: the "
            "mechanism delivers honest, usable accounting at a modest warmup dose. "
            "phasic_carry_gate_rescue_insufficient = FAIL weakens SD-075: all warmed cells "
            "were informative but the converged rate did not clear the bar -- the mechanism's "
            "accounting is honest but the rescue does not hold at this dose. "
            "phasic_carry_gate_warmed_cell_uninformative = FAIL, non_contributory: the read "
            "budget did not sustain enough post-convergence exposure to trust any reading -- an "
            "instrument-scale finding, not evidence against the mechanism itself."
        ),
    }

    ethics_preflight = {
        "involves_negative_valence": False,
        "involves_suffering_like_state": False,
        "involves_self_model": False,
        "involves_inescapability_or_helplessness": False,
        "involves_offline_replay_over_harm": False,
        "involves_social_mind_or_language": False,
        "involves_human_data_or_clinical_context": False,
        "decision": "allow",
    }

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    manifest: Dict[str, Any] = {
        "schema_version": "v1",
        "run_id": f"{EXPERIMENT_TYPE}_{ts}_v3",
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": ts,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": CLAIM_IDS,
        "outcome": outcome,
        "evidence_direction": evidence_direction,
        "dry_run": dry_run,
        "non_degenerate": non_degenerate,
        "degeneracy_reason": degeneracy_reason,
        "interpretation": interpretation,
        "ethics_preflight": ethics_preflight,
        "arm_results": rows,
        # Flat scalar readout -- the pack's metrics.values source. Every quantitative block
        # this driver emits (arm_results, interpretation.criteria/.preconditions) is a list or
        # a nested dict, so a run with no numeric metrics.values could score with no fail_if
        # stop threshold and no index deltas. These are the pre-registered scalars the
        # load-bearing rescue criterion turns on.
        "readout": flat_readout({
            "C1_phasic_carry_gate_achieves_informative_rescue": rescue_confirmed,
            "n_criteria_passed": int(bool(rescue_confirmed)),
            "n_criteria_total": 1,
            "non_degenerate_flag": non_degenerate,
            "all_warmed_cells_informative_flag": all_warmed_informative,
            # the decisive extremum against its bar (None when no informative warmed cell
            # exists; flat_readout drops non-finite/None entries per its own encoding rule)
            "min_warmed_informative_converged_event_rate": min_warmed_informative_rate,
            "min_event_rate_bar": MIN_EVENT_RATE,
            "max_event_rate_ceiling": MAX_EVENT_RATE_CEILING,
            # the warmup=0 control range non-degeneracy is judged against
            "control_converged_event_rate_min": min_control_rate,
            "control_converged_event_rate_max": max_control_rate,
            # R0 capability precondition
            "r0_regulator_fires_flag": r0_fires,
            "burst_level_max": max_burst,
            "event_level_floor": EVENT_LEVEL_FLOOR,
            "n_preconditions_met": sum(
                1 for pc in interpretation["preconditions"] if pc["met"]),
            "n_preconditions_total": len(interpretation["preconditions"]),
            # cell census
            "n_cells": len(rows),
            "n_warmed_cells": len(warmed_rows),
            "n_informative_warmed_cells": len(informative_warmed_rows),
            "n_control_cells": len(control_rows),
            "n_seeds": len(seeds),
            "warmed_episode_value": warmed_episode_value,
            "n_cells_meeting_min_event_rate": sum(
                1 for r in rows if r["meets_min_event_rate"]),
            # SD-104 duty-cycle bound census (defensive-config confirmation, not a criterion)
            "n_cells_burst_duty_within_bound": sum(
                1 for r in rows if r["burst_duty_cycle_within_bound"]),
        }),
        "seeds": seeds,
        "notes": (
            "Claim-tagged (SD-075) successor to the non-contributory diagnostic V3-EXQ-952 "
            "(v3_exq_952_sd075_phasic_warmup_rescue_diagnostic, 2026-08-28), which established "
            "readiness at warmup_episodes=40 but tagged no claim. This experiment reuses that "
            "same measured-informative dose (40) under a materially different, SD-104-bounded "
            "regulator config and a rate-based (not count-based) criterion, adds the design "
            "doc's required uninformative-cell consumption path, and defensively bounds the "
            "open degrading sd_phasic_burst_decay_and_warmup_headroom duty-cycle defect via "
            "SD-104's refractory_ticks=29/extinction_level=0.05 (952 ran before SD-104 "
            "existed). Two-pass red-team (fable): pass 1 BLOCKING (absolute count bar "
            "unreachable under refractory bounding, fixed to a ceiling-relative rate), pass 2 "
            "CONTESTED (both confirmers addressed: n_events_refractory_suppressed now recorded "
            "per cell; warmup dose switched from an unverified 10 to the 952-measured-informative "
            "40) -- see script docstring RE-REVIEW section for full disposition. "
            "GOV-REUSE-1 checked: reanalysis_query.py --readout n_events_converged / "
            "min_warmed_n_events_converged over the full evidence/experiments corpus found no "
            "manifest -- including 952's own -- carrying n_events_converged as a flat readout "
            "key; not recoverable, so this experiment runs rather than reanalyzing. Supersedes "
            "no prior run (new claim-tagged probe); explicitly NOT a V3-EXQ-779c or -963c, "
            "which the confirmed failure_autopsy_V3-EXQ-779b_2026-07-19 / "
            "failure_autopsy_V3-EXQ-963a_2026-09-02 re-derive brakes refuse for MECH-063 -- "
            "this experiment tests SD-075 alone, phasic-only, no tonic axis."
        ),
    }
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()
    result = run_experiment(dry_run=args.dry_run)
    elapsed = time.perf_counter() - t0

    out_path = write_flat_manifest(
        result,
        dry_run=args.dry_run,
        config={
            "phasic_burst_baseline_continuity": PHASIC_BASELINE_CONTINUITY,
            "phasic_burst_warmup_ticks": PHASIC_WARMUP_TICKS,
            "phasic_burst_refractory_ticks": PHASIC_REFRACTORY_TICKS,
            "phasic_burst_extinction_level": PHASIC_EXTINCTION_LEVEL,
            "warmup_conditions": WARMUP_CONDITIONS,
            "read_max_env_steps": READ_MAX_ENV_STEPS,
            "read_max_episodes": READ_MAX_EPISODES,
            "min_event_rate": MIN_EVENT_RATE,
            "max_event_rate_ceiling": MAX_EVENT_RATE_CEILING,
            "informative_ticks_floor": INFORMATIVE_TICKS_FLOOR,
        },
        seeds=result["seeds"],
        script_path=_THIS,
        elapsed_seconds=elapsed,
        z_goal_stream_stats=_ZG.stats(),
    )
    print(f"outcome: {result['outcome']}")
    print(f"label: {result['interpretation']['label']}")
    print(f"manifest: {out_path}")

    emit_outcome(
        outcome=result["outcome"] if result["outcome"] in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
        dry_run=args.dry_run,
    )
