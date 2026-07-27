#!/opt/local/bin/python3
"""V3-EXQ-828 -- INV-091 cross-stream similarity BAND falsifier, remaining 5 ablations
(complements V3-EXQ-827 / V3-EXQ-827a; IGW-20260726-222 / MECH-245 worktree follow-on).

INV-091 asserts that integration and protected non-equivalence are JOINTLY necessary:
cross-stream similarity has a viable BAND. Too little shared organisation is fragmentation;
too much is representational collapse. FALSIFIER (claims.yaml, verbatim): an ablation series
-- remove event broadcasts / remove mode conditioning / remove commitment landmarks / remove
residue feedback / force lockstep / randomise rates / collapse the harm streams -- showing a
NON-MONOTONIC relation between cross-stream similarity and task-plus-anti-collapse function.
Monotonic improvement with similarity refutes the band one way; monotonic degradation refutes
it the other way. NON-DEGENERACY GUARD: at least one ablation must move the similarity
statistic measurably, or the series tested nothing.

SCOPE -- COMPLEMENTARY TO V3-EXQ-827/827a, NOT A REPLACEMENT. The claim names seven possible
ablations. V3-EXQ-827 (and its redesign V3-EXQ-827a, which superseded 827's confounded
lockstep arm) already exercised TWO of them: remove commitment landmarks (DECOUPLE) and force
lockstep (LOCKSTEP). Both of those runs self-routed `substrate_not_ready_requeue` on C1
non-degeneracy -- most recently (827a) because DECOUPLE and INTACT were statistically
indistinguishable at the move floor, even though LOCKSTEP's redesigned reading correctly
separated in the predicted direction. This run does NOT re-test landmark removal or lockstep
-- it exercises the remaining FIVE: remove event broadcasts, remove mode conditioning, remove
residue feedback, randomise rates, collapse the harm streams. A synthesis of INV-091's
evidence must read this manifest ALONGSIDE 827a's, not in isolation -- together they span all
seven named ablations across two runs, not one.

WHY FIVE NEW ARMS RATHER THAN A BIPOLAR PAIR (C1/C2 REDESIGNED FOR N ARMS). 827/827a
deliberately picked two manipulations pre-registered to move similarity in OPPOSITE directions
from intact, which let C1/C2 be stated as a strict three-point ordering (decouple < intact <
lockstep) and "intact beats both extremes by a margin". The five ablations here do not admit
that same confident pre-registration: "remove X" (broadcasts / mode conditioning / residue
feedback) plausibly reduces shared organisation (fragmentation direction, like landmark
removal), while "randomise rates" could plausibly go either way (it breaks the FIXED 1:3:10
phase relationship the substrate trained under, which most likely reduces coordination, but
unlike lockstep it does not force synchrony) and "collapse the harm streams" is constructed to
raise similarity directly (see DV-SYMMETRY below). Rather than force a directional guess onto
each arm and risk a design that is falsifiable only if the guesses happen to be right, C1/C2
are stated as an N-ARM GENERALISATION of the exact 827/827a shape, pre-registered before this
run (never tuned to its outcome):

  C1 (non-degeneracy, INV-091's own guard, literal wording): at least one of the five ablation
     arms must move mean cross_stream_similarity away from intact by >= SIMILARITY_MOVE_FLOOR
     (checked via _metrics.check_degeneracy over all six arm means for a real spread, AND the
     single largest |arm - intact| move must clear the floor). This is the claim's own "at
     least one ablation must move the statistic" guard, applied literally across the whole
     six-arm set rather than to a single pre-chosen pair. C1 false -> substrate_not_ready_
     requeue. NEVER a verdict on INV-091 itself.
  C2 (the falsifier, band shape): rank all six arms by mean cross_stream_similarity. Let
     sim_min_arm / sim_max_arm be the empirically lowest- and highest-similarity arms (NOT
     assumed in advance -- discovering which arm lands where is what "at least one ablation
     must move the statistic" makes necessary) and peak_arm = argmax(composite) over all six.
     C2 requires peak_arm to be INTERIOR (neither sim_min_arm nor sim_max_arm) AND
     composite(peak_arm) to exceed BOTH composite(sim_min_arm) and composite(sim_max_arm) by
     >= COMPOSITE_MARGIN. This is the literal claim shape -- a non-monotonic, inverted-U
     relation between similarity and task-plus-anti-collapse function -- generalised from "does
     intact beat both extremes" (827/827a, valid only because intact was pre-registered as the
     middle point of a bipolar pair) to "does SOME interior-similarity arm beat both empirical
     extremes" (valid for an arbitrary N-arm similarity spread). A secondary, NON-LOAD-BEARING
     diagnostic reports whether intact SPECIFICALLY beats both extremes (827/827a's own
     formula), for continuity/comparability -- it does not gate the verdict, because intact is
     not guaranteed to be the empirical middle point here the way it was by design in 827/827a.
  Rank-order monotonic_direction (diagnostic, mirrors 827/827a): if composite is non-decreasing
     or non-increasing across the full similarity-sorted rank order, that is recorded as a
     stronger, direction-specific refutation the claim explicitly anticipates.

THRESHOLDS UNCHANGED FROM 827/827a (SIMILARITY_MOVE_FLOOR=0.02, COMPOSITE_MARGIN=0.15) --
these are properties of the DVs (arm-mean similarity units, pooled-SD composite units), not of
which specific manipulations produced them, so there is no principled reason to retune them for
a different manipulation set.

WHY A COMMON TRAINED SUBSTRATE, NOT SIX SEPARATE WARMUPS (unchanged rationale from 827/827a).
All six arms share ONE warmed-up agent per seed (E1/E2 world-forward + E3 harm-eval head
trained once via _lib.goal_pipeline_tier1.warmup_train under the INTACT config); each
manipulation is applied ONLY at eval/recording time, isolating it from "the arms trained
differently" as a confound. Each arm's eval env is a FRESH CausalGridWorldV2 instance built
with the SAME seed, so all six arms see matched episode layouts (a paired design at fixed seed
count). Reuses the Q-081 infra landed 2026-07-22..2026-07-26 (stream_recorder, trace_store,
q081_profile, q081_surrogate, arm_fingerprint) exactly as 827/827a do -- no infra rebuilt.

THE FIVE MANIPULATIONS (each an eval-time-only patch on the shared warmed agent, restored
before the next arm; see the per-arm patch classes / rate sampler below for mechanism):

  BROADCAST_OFF (remove event broadcasts, targets MECH-287 ONLY, not MECH-269). Per
    experiments/_lib/q081_landmark_removal.py's own docstring, event_segmenter.step()'s
    output feeds THREE independent consumers: _boundary_event_queue (MECH-288, always),
    invalidation_trigger.step() -> MECH-287 broadcasts -> apply_invalidation_broadcasts_to_
    regions() (gated on use_invalidation_trigger), and tick_anchor_set() -> MECH-269
    write_anchor (gated on use_anchor_sets, already 827/827a's decouple mechanism). 827/827a's
    landmark scrambler retimes the landmark stream itself and therefore hits all three
    consumers at once -- it cannot isolate MECH-287 alone. This arm instead toggles
    `agent.hippocampal.config.use_invalidation_trigger = False` for the eval window (confirmed
    live at ree_core/agent.py ~4388-4407: the flag gates the ONLY call site of
    `invalidation_trigger.step()`), leaving `use_anchor_sets` True and the landmark/boundary-
    event stream itself completely untouched -- only the broadcast-consumer path is cut.
  MODE_DECORRELATED (remove mode conditioning). `agent.salience.tick(...)` (agent.py ~5959,
    result cached in `self._salience_last_tick`, which experiments/_lib/stream_recorder.py
    reads DIRECTLY to populate the `operating_mode` stream -- confirmed at stream_recorder.py
    ~423-441) is patched so the dacc_bundle / drive_level / per_axis_drive it actually computes
    over are independent per-tick noise, not the real signals SD-032b/SD-012/SD-049 supply.
    The coordinator's own hysteresis / mode-switch / write-gate machinery runs UNCHANGED on
    this substituted input, so every real consumer (MECH-261 write gates, E3 selection) sees a
    genuinely decorrelated operating_mode, not a display-only overwrite. Deliberately NOT
    `use_salience_coordinator=False` -- that nulls the operating_mode stream outright rather
    than decoupling it from its drivers, and SIMILARITY_PAIRS needs the stream present to
    compute the (z_world, operating_mode) pair at all.
  RESIDUE_OFF (remove residue feedback). `agent.update_residue(...)` (agent.py ~8320) is
    patched to force `hypothesis_tag=True` on every call regardless of what the harness passes
    -- the SAME mechanism MECH-094 already uses to block ResidueField.accumulate (confirmed at
    ResidueField's own accumulate() gate; agent.py's ARC-016 comment on update_residue notes
    running-variance tracking and E3 post_action_update run on EVERY call independent of the
    tag). So this arm blocks ONLY the residue write; the E3 post_action_update call and
    running-variance tracking inside the same method are UNCHANGED, cutting the specific
    feedback path (VALENCE_WANTING readout at z_world, SD-039 goal payload) without touching
    anything else update_residue does.
  RATE_RANDOMIZED (randomise rates). Instead of pinning e2_steps_per_tick / e3 base+current
    steps to the fixed configured 1:3:10 (intact) or a synchronised common rate (827a's
    lockstep), this arm RESAMPLES `agent.clock.e2_steps_per_tick` and `agent.clock.
    _e3_base_steps` / `_current_e3_steps` independently, fresh, at the start of EVERY episode,
    from pre-registered ranges (RATE_RANDOM_E2_RANGE, RATE_RANDOM_E3_RANGE -- see constants;
    e1 stays fixed at 1, the sensorium's native rate, unchanged from intact). This breaks the
    FIXED phase relationship the substrate warmed up under episode-to-episode, without forcing
    the streams into either extreme (neither the landmark-removal decoupling nor forced
    lockstep synchrony) -- a genuinely distinct manipulation axis from both of 827/827a's arms.
  HARM_COLLAPSE (collapse the harm streams). `agent.latent_stack.encode(...)`
    (ree_core/latent/stack.py ~1428-1449) computes `z_harm` (SD-010, dim 32 under this
    profile) and `z_harm_a` (SD-011, dim 16) as two independent locals within the SAME call,
    attached to the same returned `LatentState` (a plain, non-frozen @dataclass --
    ree_core/latent/stack.py:734). This arm patches `encode()` to overwrite the returned
    state's `z_harm_a` with a fixed slice of that SAME call's `z_harm` (truncated to
    z_harm_a's native dim), collapsing the affective harm stream onto (a projection of) the
    sensory harm stream at the point of emission -- every downstream consumer (serotonin
    tonic suppression, infralimbic avoidance gate, MECH-219 suffering accumulator, the
    recorder) sees the collapsed value, a genuine representational collapse, not a display
    overwrite. SIMILARITY_PAIRS is extended with a fifth pair, (z_harm, z_harm_a), specifically
    so this manipulation has a channel to move -- see DV-SYMMETRY below for why this one pair's
    movement is BY CONSTRUCTION and must not be over-read as evidence on its own.

CROSS-STREAM SIMILARITY STATISTIC (extends 827/827a's four pairs with a fifth). Mean of
|xcorr| (q081_surrogate.cross_stream_xcorr, reduce="first") over FIVE pairs:
    (z_world, z_self)          E1-rate vs E1-rate, distinct subsystems (world vs self model)
    (z_world, operating_mode)  E1-rate vs E3-rate (MECH-321 salience coordinator)
    (e1_hidden, e3_commitment) E1-rate vs E3-rate (sensorium vs commitment state)
    (z_world, z_goal)          E1-rate vs loop-driven (goal stream; StepHarness calls
                                update_z_goal every tick, so this is genuinely driven, not
                                flat -- see q081_profile.LOOP_DRIVEN_REQUIREMENTS)
    (z_harm, z_harm_a)         E1-rate vs E1-rate, distinct subsystems (sensory vs affective
                                harm) -- NEW pair, added specifically for HARM_COLLAPSE; all
                                six arms use the same five-pair statistic for comparability.
`reduce_stream(..., "first")` takes column 0 of each stream (q081_surrogate.py) -- HARM_
COLLAPSE's truncation-based overwrite (z_harm_a := z_harm[..., :16]) makes column 0 of z_harm_a
LITERALLY EQUAL to column 0 of z_harm under this arm, so the pair's xcorr is expected to read
near its maximum; this is intentional (the arm is CONSTRUCTED to hit the representational-
collapse / high-similarity end of the band, exactly as 827a's lockstep is constructed to hit it
via forced tick synchrony -- a different construction, same intent).
Validated per Q-081's "validate the null before using it" requirement: screen_statistic
confirms this statistic is NOT ruled_out on the intact arm's seed[0] trace while
q081_surrogate.artefactual_rate_statistic on the same arrays IS ruled_out, exactly as in
827/827a (this experiment does not alter q081_surrogate.py or its null-validation logic).

ANTI-COLLAPSE (protected non-equivalence) STATISTIC. Unchanged from 827/827a: 1 -
mean(|cosine(z_self_t, z_world_t)|) over eval steps where both are fresh. Higher = more
protected non-equivalence.

TASK METRIC. Mean per-episode reward (env harm_signal summed per episode), unchanged.

COMPOSITE (task_plus_anticollapse). Unchanged: min(z_task, z_anticollapse), z-scored by
pooling all 18 (arm x seed) cells (6 arms x 3 seeds, vs 827/827a's 9). A sum would let either
axis compensate for the other, which INV-091's own conjunctive wording forbids.

DV-SYMMETRY (per-arm, mandatory declaration -- REE_Working/CLAUDE.md queue-experiment skill
Step 3.5). None of the five arms is a broadcast-scalar / monotone-rescaling / interchangeable-
unit permutation that leaves cross_stream_similarity or the composite invariant BY
CONSTRUCTION for a reason the DV cannot see:
  BROADCAST_OFF:   removes an entire computation path (invalidation_trigger.step() never
                   runs, so per-region V_s resets driven by MECH-287 never fire) -- a genuine
                   change to what is computed, not a relabelling of an existing value.
  MODE_DECORRELATED: replaces salience.tick()'s actual inputs with fresh independent noise
                   each tick -- the coordinator computes a DIFFERENT operating_mode from
                   different inputs every call, not a fixed function of the true one.
  RESIDUE_OFF:     forcing hypothesis_tag=True changes ResidueField.accumulate's control flow
                   (skips the write), which changes what a LATER tick's residue-dependent
                   reads (VALENCE_WANTING, SD-039 goal payload) see -- a genuine feedback-loop
                   removal, not a symmetry.
  RATE_RANDOMIZED: per-episode-varying tick cadence changes WHEN E2/E3 candidates are
                   generated, scored and committed, exactly the same non-symmetry argument
                   827a makes for its own (fixed-target) lockstep redesign -- a resampled
                   target is still a change to what is computed, not a relabelling.
  HARM_COLLAPSE:   NOTED CONSTRUCTION EFFECT, not a loophole -- the (z_harm, z_harm_a) pair's
                   xcorr is EXPECTED to move by construction under this arm (see CROSS-STREAM
                   STATISTIC above), which is the intended mechanism for reaching the
                   high-similarity end of the band, exactly as 827a's forced-lockstep arm is
                   constructed to move ITS statistic. The pair is not invariant under any
                   symmetry -- it is a direct, deliberate overwrite of one of the two streams
                   the DV reads. What is NOT by construction, and is the genuine test, is
                   whether the OTHER four pairs (unrelated to the harm streams) also move, and
                   whether the resulting five-pair MEAN crosses SIMILARITY_MOVE_FLOOR and lands
                   this arm at the correct (high) end of C2's rank ordering -- a single
                   constructed pair contributing 1/5 of the mean cannot by itself satisfy C1's
                   overall-spread check, let alone C2's peak-interiority test.
None of the five arms' DVs is a pure function of a manipulation the DV cannot see.

GOV-REUSE-1: `reanalysis_query.py query --claim INV-091` (2026-07-27) returns exactly the two
manifests V3-EXQ-827 and V3-EXQ-827a (different substrate_hash from each other -- 827a's run
postdates a same-day tightening of q081_landmark_removal.assert_behavioural_reach that folds
into the substrate hash). Neither carries any of the five ablations here, nor the (z_harm,
z_harm_a) similarity pair -- not recoverable; this is the first run for these five ablations.

RE-DERIVE BRAKE (Step 2.5b): INV-091 has exactly ONE prior autopsy
(failure_autopsy_batch-822a-826-817a-827_2026-07-26, section 4) -- below the >=2 threshold, so
the brake does not fire. That autopsy also targeted a `measurement_test_design_defect` (827's
confounded lockstep construction), which is an instrument-defect category and would not count
toward the brake even at a higher tally (the instrument-defect carve-out in the brake rule).
This run is not a re-derive of that autopsy's target -- it exercises five DIFFERENT
manipulations 827/827a never touched.

ethics_preflight:
  involves_negative_valence: false
  involves_suffering_like_state: false
  involves_self_model: false
  involves_inescapability_or_helplessness: false
  involves_offline_replay_over_harm: false
  involves_social_mind_or_language: false
  involves_human_data_or_clinical_context: false
  decision: allow

SLEEP DRIVER: none (use_sleep_loop stays at the Q-081 profile default; the loop never calls
run_sleep_cycle(), so no sleep cycle fires during this run).

MINT (mint-as-you-go). All six arms (intact + five ablations) are emitted reuse-eligible by
default (include_driver_script_in_hash=False, config_slice_declared=True, no
extra_ineligible_reasons): unlike 827/827a's DECOUPLE arm, none of these five ablations
consumes another arm's runtime artifact (no donor-train banking or other cross-arm state
dependency) -- each arm's eval output is a pure function of (config_slice, seed) given the
shared frozen warmed weights and a full RNG reset at cell entry, so all six are genuinely
reusable. No separate baseline-only mint experiment is queued (the in-line mint suffices; no
known parallel consumer, and this run is not Mac-machine-class-locked).

ASCII-only in all runtime strings (repo rule).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from experiment_protocol import emit_outcome  # noqa: E402
from experiments._harness import StepHarness, StepHooks  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._lib.goal_pipeline_tier1 import warmup_train  # noqa: E402
from experiments._lib.q081_profile import (  # noqa: E402
    q081_profile_kwargs,
    q081_substrate_declaration,
)
from experiments._lib import q081_surrogate as surro  # noqa: E402
from experiments._lib.stream_recorder import StreamTraceRecorder  # noqa: E402
from experiments._lib.trace_store import TraceStore  # noqa: E402
from experiments._metrics import check_degeneracy  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_828_inv091_cross_stream_similarity_band_remaining_ablations"
QUEUE_ID = "V3-EXQ-828"
CLAIM_IDS: List[str] = ["INV-091"]
EXPERIMENT_PURPOSE = "evidence"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"

SEEDS: Tuple[int, ...] = (11, 23, 37)
ARMS: Tuple[str, ...] = (
    "intact", "broadcast_off", "mode_decorrelated", "residue_off",
    "rate_randomized", "harm_collapse",
)
OFF_ARM = "intact"

WARMUP_EPISODES = 40
EVAL_EPISODES = 10
STEPS_PER_EPISODE = 150
DRY_WARMUP_EPISODES = 2
DRY_EVAL_EPISODES = 2
DRY_STEPS_PER_EPISODE = 20

HARM_HISTORY_LEN = 10
ENV_KWARGS: Dict[str, Any] = dict(
    size=10, num_hazards=1, num_resources=5, harm_history_len=HARM_HISTORY_LEN,
)

# Pairs spanning different subsystems / native rates; see module docstring. Extends
# 827/827a's four pairs with (z_harm, z_harm_a) for the HARM_COLLAPSE arm.
SIMILARITY_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("z_world", "z_self"),
    ("z_world", "operating_mode"),
    ("e1_hidden", "e3_commitment"),
    ("z_world", "z_goal"),
    ("z_harm", "z_harm_a"),
)
SIMILARITY_STREAM_NAMES: Tuple[str, ...] = tuple(sorted(
    {s for pair in SIMILARITY_PAIRS for s in pair}
))

# Pre-registered thresholds (never derived from the run's own statistics; unchanged from
# 827/827a -- see module docstring "THRESHOLDS UNCHANGED").
SIMILARITY_MOVE_FLOOR = 0.02   # C1: largest |arm - intact| similarity gap must clear this
COMPOSITE_MARGIN = 0.15        # C2: peak-interior arm must beat both extremes by this many
                                # pooled-SD units
MIN_FRESH_COS_SAMPLES = 8
N_SURROGATES = 199
BASE_MAX_LAG_TICKS = 8

# Pre-registered per-episode resampling ranges for RATE_RANDOMIZED (inclusive). e1 stays fixed
# at 1 (the sensorium's native rate, unchanged from intact). Intact's own fixed values (3, 10)
# sit inside both ranges so the resampled distribution genuinely straddles intact, not just one
# side of it.
RATE_RANDOM_E2_RANGE: Tuple[int, int] = (1, 6)
RATE_RANDOM_E3_RANGE: Tuple[int, int] = (3, 18)


def _env_kwargs() -> Dict[str, Any]:
    return dict(ENV_KWARGS)


def _build_cfg(env: CausalGridWorldV2) -> REEConfig:
    """Shared INTACT-rate config for all six arms. use_anchor_sets=True (unchanged from
    827/827a) -- not exercised as a manipulation here, but load-bearing for MECH-269 anchor
    writes to keep behaving exactly as the intact reference arm in 827/827a did."""
    kwargs = dict(q081_profile_kwargs())
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        alpha_world=0.9,
        harm_history_len=HARM_HISTORY_LEN,
        **kwargs,
    )
    cfg.hippocampal.use_anchor_sets = True
    return cfg


def _config_slice(arm: str, warmup_episodes: int, eval_episodes: int, steps: int) -> Dict[str, Any]:
    """Declared config_slice for the arm fingerprint -- ONLY what this cell computes."""
    slice_ = {
        "arm_id": arm,
        "experiment_type": EXPERIMENT_TYPE,
        "env_kwargs": _env_kwargs(),
        "q081_profile": dict(q081_profile_kwargs()),
        "alpha_world": 0.9,
        "use_anchor_sets": True,
        "warmup_episodes": int(warmup_episodes),
        "eval_episodes": int(eval_episodes),
        "steps_per_episode": int(steps),
        "similarity_pairs": list(SIMILARITY_PAIRS),
        "manipulation": arm,
    }
    if arm == "broadcast_off":
        slice_["use_invalidation_trigger_eval"] = False
    elif arm == "mode_decorrelated":
        slice_["salience_tick_inputs"] = "iid_noise_per_tick"
    elif arm == "residue_off":
        slice_["update_residue_hypothesis_tag_forced"] = True
    elif arm == "rate_randomized":
        slice_["rate_random_e2_range"] = list(RATE_RANDOM_E2_RANGE)
        slice_["rate_random_e3_range"] = list(RATE_RANDOM_E3_RANGE)
    elif arm == "harm_collapse":
        slice_["z_harm_a_source"] = "z_harm_truncated_to_z_harm_a_dim"
    return slice_


def _restore_intact_rates(agent: REEAgent) -> None:
    agent.clock.e1_steps_per_tick = 1
    agent.clock.e2_steps_per_tick = 3
    agent.clock._e3_base_steps = 10
    agent.clock._current_e3_steps = 10


class _BroadcastOffPatch:
    """MECH-287 broadcast trigger off for the eval window: toggles
    agent.hippocampal.config.use_invalidation_trigger. Boundary events (MECH-288) and MECH-269
    anchor writes (use_anchor_sets) are completely untouched -- see module docstring
    BROADCAST_OFF."""

    def __init__(self, agent: REEAgent) -> None:
        self._agent = agent
        self._prev: Optional[bool] = None

    def attach(self) -> None:
        self._prev = bool(self._agent.hippocampal.config.use_invalidation_trigger)
        self._agent.hippocampal.config.use_invalidation_trigger = False

    def detach(self) -> None:
        if self._prev is not None:
            self._agent.hippocampal.config.use_invalidation_trigger = self._prev
        self._prev = None


class _ModeDecorrelatedPatch:
    """Decorrelate operating_mode from its true drivers for the eval window: patches
    agent.salience.tick so the dacc_bundle / drive_level / per_axis_drive it actually computes
    over are independent per-tick noise. See module docstring MODE_DECORRELATED."""

    def __init__(self, agent: REEAgent, seed: int) -> None:
        self._agent = agent
        self._rng = np.random.default_rng(seed + 900001)
        self._orig: Optional[Callable[..., Any]] = None

    def attach(self) -> None:
        self._orig = self._agent.salience.tick
        orig = self._orig
        rng = self._rng

        def _patched(dacc_bundle=None, drive_level: float = 0.0, is_offline: bool = False,
                     extra_signals=None, per_axis_drive=None, per_axis_combiner: str = "max"):
            fake_dacc = None
            if dacc_bundle is not None:
                fake_dacc = {
                    "pe": float(rng.standard_normal()),
                    "foraging_value": float(rng.uniform(0.0, 1.0)),
                    "choice_difficulty": float(rng.uniform(0.0, 1.0)),
                }
            fake_drive = float(rng.uniform(0.0, 1.0))
            fake_per_axis = None
            if per_axis_drive is not None:
                try:
                    n = len(per_axis_drive)
                except TypeError:
                    n = 1
                fake_per_axis = rng.uniform(0.0, 1.0, size=n)
            return orig(
                dacc_bundle=fake_dacc, drive_level=fake_drive, is_offline=is_offline,
                extra_signals=extra_signals, per_axis_drive=fake_per_axis,
                per_axis_combiner=per_axis_combiner,
            )

        self._agent.salience.tick = _patched

    def detach(self) -> None:
        if self._orig is not None:
            self._agent.salience.tick = self._orig
        self._orig = None


class _ResidueOffPatch:
    """Forces hypothesis_tag=True on every agent.update_residue call for the eval window --
    reuses MECH-094's own accumulation-block mechanism rather than a bespoke no-op. E3
    post_action_update and running-variance tracking inside update_residue are unaffected.
    See module docstring RESIDUE_OFF."""

    def __init__(self, agent: REEAgent) -> None:
        self._agent = agent
        self._orig: Optional[Callable[..., Any]] = None

    def attach(self) -> None:
        self._orig = self._agent.update_residue
        orig = self._orig

        def _patched(harm_signal, world_delta=None, hypothesis_tag: bool = False, owned: bool = True):
            return orig(harm_signal, world_delta=world_delta, hypothesis_tag=True, owned=owned)

        self._agent.update_residue = _patched

    def detach(self) -> None:
        if self._orig is not None:
            self._agent.update_residue = self._orig
        self._orig = None


class _HarmCollapsePatch:
    """Collapses z_harm_a onto (a truncation of) z_harm at the encoder output for the eval
    window: patches agent.latent_stack.encode so the returned LatentState's z_harm_a is
    overwritten with the SAME call's z_harm, truncated to z_harm_a's native dim. LatentState is
    a plain (non-frozen) dataclass, so every downstream consumer of the returned object sees
    the collapsed value. See module docstring HARM_COLLAPSE."""

    def __init__(self, agent: REEAgent) -> None:
        self._agent = agent
        self._orig: Optional[Callable[..., Any]] = None

    def attach(self) -> None:
        self._orig = self._agent.latent_stack.encode
        orig = self._orig

        def _patched(*args, **kwargs):
            state = orig(*args, **kwargs)
            if state.z_harm is not None and state.z_harm_a is not None:
                dim_a = state.z_harm_a.shape[-1]
                state.z_harm_a = state.z_harm[..., :dim_a].clone()
            return state

        self._agent.latent_stack.encode = _patched

    def detach(self) -> None:
        if self._orig is not None:
            self._agent.latent_stack.encode = self._orig
        self._orig = None


def _resample_rates(agent: REEAgent, rng: np.random.Generator) -> Dict[str, int]:
    """Draw fresh e2/e3 rates for one episode from the pre-registered ranges (module
    docstring RATE_RANDOMIZED). e1 stays fixed at 1."""
    e2 = int(rng.integers(RATE_RANDOM_E2_RANGE[0], RATE_RANDOM_E2_RANGE[1] + 1))
    e3 = int(rng.integers(RATE_RANDOM_E3_RANGE[0], RATE_RANDOM_E3_RANGE[1] + 1))
    agent.clock.e1_steps_per_tick = 1
    agent.clock.e2_steps_per_tick = e2
    agent.clock._e3_base_steps = e3
    agent.clock._current_e3_steps = e3
    return {"e1": 1, "e2": e2, "e3": e3}


def _cross_stream_similarity_stat(arrays) -> float:
    vals: List[float] = []
    for a, b in SIMILARITY_PAIRS:
        try:
            r, _lag = surro.cross_stream_xcorr(arrays, a, b, max_lag_ticks=BASE_MAX_LAG_TICKS, reduce="first")
        except Exception:
            r = float("nan")
        if np.isfinite(r):
            vals.append(float(r))
    return float(np.mean(vals)) if vals else float("nan")


def _anti_collapse_stat(arrays) -> float:
    zs = np.asarray(arrays["z_self"], dtype=np.float64)
    zw = np.asarray(arrays["z_world"], dtype=np.float64)
    fresh = (
        np.asarray(arrays["z_self__fresh"], dtype=bool)
        & np.asarray(arrays["z_world__fresh"], dtype=bool)
        & np.asarray(arrays["z_self__valid"], dtype=bool)
        & np.asarray(arrays["z_world__valid"], dtype=bool)
    )
    idx = np.flatnonzero(fresh)
    if idx.size < MIN_FRESH_COS_SAMPLES:
        return float("nan")
    a = zs[idx]
    b = zw[idx]
    num = np.sum(a * b, axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    den = np.where(den <= 1e-12, np.nan, den)
    cos = num / den
    cos = cos[np.isfinite(cos)]
    if cos.size == 0:
        return float("nan")
    return float(1.0 - np.mean(np.abs(cos)))


def _eval_pass(
    agent: REEAgent,
    env: CausalGridWorldV2,
    arm: str,
    seed: int,
    n_episodes: int,
    steps_per_episode: int,
    store: TraceStore,
    rate_rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """Run n_episodes of eval on `agent`/`env`, recording the Q-081 trace throughout.

    `agent`'s weights are never updated here (train_mode=False -> torch.no_grad via the
    harness). When `rate_rng` is given (RATE_RANDOMIZED only), fresh e2/e3 rates are drawn at
    the start of EVERY episode.
    """
    rec = StreamTraceRecorder(
        agent, run_id=f"{EXPERIMENT_TYPE}_{arm}_seed{seed}_v3",
        store=store, substrate_declaration=q081_substrate_declaration(agent.config),
    )
    acc: Dict[str, Any] = {
        "reward_total": 0.0, "episode_rewards": [], "action_counts": {},
        "state_visitation": {}, "harm_events": 0, "n_steps": 0,
        "episode_lengths": [], "sampled_rates": [],
    }

    def on_post_step(*, agent, latent, action, obs_dict, next_obs_dict,
                      harm_signal, done, ticks, residue_metrics, step, **kw) -> None:
        rec.on_step(extras={"reward": float(harm_signal)})
        aidx = int(action.argmax(dim=-1).item())
        acc["action_counts"][aidx] = acc["action_counts"].get(aidx, 0) + 1
        pos_key = f"{int(getattr(env, 'agent_x', -1))},{int(getattr(env, 'agent_y', -1))}"
        acc["state_visitation"][pos_key] = acc["state_visitation"].get(pos_key, 0) + 1
        if float(harm_signal) < 0.0:
            acc["harm_events"] += 1
        acc["n_steps"] += 1

    hooks = StepHooks(on_post_step=on_post_step)
    harness = StepHarness(agent, env, train_mode=False, hooks=hooks, seed=seed)
    agent.eval()

    for ep in range(int(n_episodes)):
        _flat, obs_dict = env.reset()
        agent.reset()
        harness.reset()
        if rate_rng is not None:
            acc["sampled_rates"].append(_resample_rates(agent, rate_rng))

        ep_reward = 0.0
        ep_len = 0
        for _t in range(int(steps_per_episode)):
            result = harness.step(obs_dict)
            ep_reward += float(result.harm_signal)
            ep_len += 1
            obs_dict = result.next_obs_dict
            if result.done:
                break
        rec.on_episode_end()
        acc["reward_total"] += ep_reward
        acc["episode_rewards"].append(round(ep_reward, 6))
        acc["episode_lengths"].append(ep_len)
        if (ep + 1) % 5 == 0 or (ep + 1) == n_episodes:
            print(f"  [eval] {arm} seed={seed} ep {ep + 1}/{n_episodes}", flush=True)

    n_ep = max(1, int(n_episodes))
    acc["mean_episode_reward"] = float(acc["reward_total"]) / n_ep

    pointer = rec.finalize()
    loaded = store.get(pointer)
    arrays = loaded["arrays"]
    acc["cross_stream_similarity"] = _cross_stream_similarity_stat(arrays)
    acc["anti_collapse"] = _anti_collapse_stat(arrays)
    acc["trace_pointer"] = pointer
    acc["_arrays"] = arrays  # consumed for null validation; stripped before manifest write
    return acc


_PATCH_FACTORIES: Dict[str, Callable[[REEAgent, int], Any]] = {
    "broadcast_off": lambda agent, seed: _BroadcastOffPatch(agent),
    "mode_decorrelated": lambda agent, seed: _ModeDecorrelatedPatch(agent, seed),
    "residue_off": lambda agent, seed: _ResidueOffPatch(agent),
    "harm_collapse": lambda agent, seed: _HarmCollapsePatch(agent),
}


def _run_seed(seed: int, warmup_episodes: int, eval_episodes: int, steps_per_episode: int,
              store: TraceStore, script_path: Path) -> Dict[str, Any]:
    env_warm = CausalGridWorldV2(seed=seed, **_env_kwargs())
    cfg = _build_cfg(env_warm)
    agent = REEAgent(cfg)

    print(f"Seed {seed} Condition warmup_shared", flush=True)
    warmup_train(
        agent, env_warm,
        num_episodes=int(warmup_episodes), steps_per_episode=int(steps_per_episode),
        label=f"warmup_shared seed={seed}", progress_total_episodes=int(warmup_episodes),
    )

    results: Dict[str, Dict[str, Any]] = {}

    for arm in ARMS:
        with arm_cell(seed, config_slice=_config_slice(arm, warmup_episodes, eval_episodes,
                                                         steps_per_episode),
                      script_path=script_path, config_slice_declared=True,
                      include_driver_script_in_hash=False) as cell:
            print(f"Seed {seed} Condition {arm}", flush=True)
            _restore_intact_rates(agent)

            patch = _PATCH_FACTORIES[arm](agent, seed) if arm in _PATCH_FACTORIES else None
            rate_rng = np.random.default_rng(seed + 900002) if arm == "rate_randomized" else None
            if patch is not None:
                patch.attach()
            try:
                row = _eval_pass(
                    agent, CausalGridWorldV2(seed=seed, **_env_kwargs()),
                    arm, seed, eval_episodes, steps_per_episode, store,
                    rate_rng=rate_rng,
                )
            finally:
                if patch is not None:
                    patch.detach()
                _restore_intact_rates(agent)

            cell.stamp(row)
        results[arm] = row

    print("verdict: PASS", flush=True)
    return results


def _null_validation(intact_arrays) -> Dict[str, Any]:
    """Q-081's mandatory 'validate the null before using it' check, run once on the
    intact arm's seed[0] trace. Confirms the real statistic is admissible (varies across
    the surrogate ensemble) while the deliberately artefactual rate statistic is ruled out
    (a pure function of the two streams' update periods, killed by construction)."""
    try:
        real_screen = surro.screen_statistic(
            intact_arrays, list(SIMILARITY_STREAM_NAMES), _cross_stream_similarity_stat,
            n_surrogates=64, seed=1,
        )
    except surro.SurrogateDesignError as exc:
        return {"checked": False, "reason": str(exc)}

    def _artefactual(arrays):
        return surro.artefactual_rate_statistic(arrays, "z_world", "operating_mode")

    try:
        artefact_screen = surro.screen_statistic(
            intact_arrays, ["z_world", "operating_mode"], _artefactual,
            n_surrogates=64, seed=1,
        )
    except surro.SurrogateDesignError as exc:
        return {"checked": False, "reason": str(exc)}

    return {
        "checked": True,
        "real_statistic_verdict": real_screen["verdict"],
        "real_statistic_admissible": real_screen["verdict"] == "admissible",
        "artefactual_statistic_verdict": artefact_screen["verdict"],
        "artefactual_statistic_ruled_out": artefact_screen["verdict"] == "ruled_out",
        "null_validated": (
            real_screen["verdict"] == "admissible"
            and artefact_screen["verdict"] == "ruled_out"
        ),
        "detail": {"real": real_screen, "artefactual": artefact_screen},
    }


def _surrogate_p_values(per_arm_arrays: Dict[str, List[Any]]) -> Dict[str, Any]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for arm, arrays_list in per_arm_arrays.items():
        rows = []
        for i, arrays in enumerate(arrays_list):
            try:
                res = surro.evaluate_against_null(
                    arrays, list(SIMILARITY_STREAM_NAMES), _cross_stream_similarity_stat,
                    n_surrogates=N_SURROGATES, seed=100 + i,
                )
                rows.append({"seed_index": i, "p_value": res["p_value"],
                            "observed": res["observed"], "surrogate_mean": res["surrogate_mean"]})
            except surro.SurrogateDesignError as exc:
                rows.append({"seed_index": i, "error": str(exc)})
        out[arm] = rows
    return out


def main(dry_run: bool) -> Dict[str, Any]:
    t0 = time.perf_counter()
    warmup_episodes = DRY_WARMUP_EPISODES if dry_run else WARMUP_EPISODES
    eval_episodes = DRY_EVAL_EPISODES if dry_run else EVAL_EPISODES
    steps_per_episode = DRY_STEPS_PER_EPISODE if dry_run else STEPS_PER_EPISODE
    script_path = Path(__file__)

    store = TraceStore()
    per_seed: Dict[int, Dict[str, Any]] = {}
    for seed in SEEDS:
        per_seed[seed] = _run_seed(seed, warmup_episodes, eval_episodes, steps_per_episode,
                                   store, script_path)

    # ---- null validation (once, intact seed[0]) ----
    null_report = _null_validation(per_seed[SEEDS[0]]["intact"]["_arrays"])

    # ---- per-arm surrogate p-values (all cells; cheap, array-only) ----
    per_arm_arrays = {arm: [per_seed[s][arm]["_arrays"] for s in SEEDS] for arm in ARMS}
    surrogate_report = _surrogate_p_values(per_arm_arrays)

    # ---- per-arm means ----
    per_arm_similarity = {arm: float(np.mean([per_seed[s][arm]["cross_stream_similarity"]
                                             for s in SEEDS])) for arm in ARMS}
    per_arm_anti_collapse = {arm: float(np.mean([per_seed[s][arm]["anti_collapse"]
                                                for s in SEEDS])) for arm in ARMS}
    per_arm_task = {arm: float(np.mean([per_seed[s][arm]["mean_episode_reward"]
                                       for s in SEEDS])) for arm in ARMS}

    # ---- C1: non-degeneracy (N-arm generalisation; see module docstring) ----
    degeneracy = check_degeneracy({
        "cross_stream_similarity_arm_means": [per_arm_similarity[a] for a in ARMS],
    })
    moves_vs_intact = {a: per_arm_similarity[a] - per_arm_similarity["intact"]
                       for a in ARMS if a != "intact"}
    max_abs_move_arm = max(moves_vs_intact, key=lambda a: abs(moves_vs_intact[a]))
    max_abs_move = abs(moves_vs_intact[max_abs_move_arm])
    c1_pass = bool(degeneracy["non_degenerate"]) and (max_abs_move >= SIMILARITY_MOVE_FLOOR)

    # ---- composite: min(z_task, z_anticollapse), pooled across all 18 cells ----
    task_pool = np.asarray([per_seed[s][a]["mean_episode_reward"] for a in ARMS for s in SEEDS])
    ac_pool = np.asarray([per_seed[s][a]["anti_collapse"] for a in ARMS for s in SEEDS])
    task_mu, task_sd = float(np.nanmean(task_pool)), float(np.nanstd(task_pool, ddof=1) or 1.0)
    ac_mu, ac_sd = float(np.nanmean(ac_pool)), float(np.nanstd(ac_pool, ddof=1) or 1.0)
    task_sd = task_sd if task_sd > 1e-9 else 1.0
    ac_sd = ac_sd if ac_sd > 1e-9 else 1.0

    composite_by_cell: Dict[str, List[float]] = {a: [] for a in ARMS}
    for a in ARMS:
        for s in SEEDS:
            zt = (per_seed[s][a]["mean_episode_reward"] - task_mu) / task_sd
            za = (per_seed[s][a]["anti_collapse"] - ac_mu) / ac_sd
            composite_by_cell[a].append(min(zt, za))
    per_arm_composite = {a: float(np.mean(composite_by_cell[a])) for a in ARMS}

    # ---- C2: band shape (N-arm generalisation of "intact beats both extremes") ----
    ranked_by_similarity = sorted(ARMS, key=lambda a: per_arm_similarity[a])
    sim_min_arm, sim_max_arm = ranked_by_similarity[0], ranked_by_similarity[-1]
    peak_arm = max(ARMS, key=lambda a: per_arm_composite[a])
    peak_is_interior = peak_arm not in (sim_min_arm, sim_max_arm)
    margin_vs_min = per_arm_composite[peak_arm] - per_arm_composite[sim_min_arm]
    margin_vs_max = per_arm_composite[peak_arm] - per_arm_composite[sim_max_arm]
    c2_pass = bool(
        peak_is_interior
        and np.isfinite(margin_vs_min) and margin_vs_min >= COMPOSITE_MARGIN
        and np.isfinite(margin_vs_max) and margin_vs_max >= COMPOSITE_MARGIN
    )

    # Secondary, NON-LOAD-BEARING diagnostic: does intact SPECIFICALLY beat both empirical
    # extremes (827/827a's own formula)? Only meaningful when intact is not itself an extreme.
    intact_beats_both_extremes = None
    intact_vs_min = intact_vs_max = float("nan")
    if sim_min_arm != "intact" and sim_max_arm != "intact":
        intact_vs_min = per_arm_composite["intact"] - per_arm_composite[sim_min_arm]
        intact_vs_max = per_arm_composite["intact"] - per_arm_composite[sim_max_arm]
        intact_beats_both_extremes = bool(
            intact_vs_min >= COMPOSITE_MARGIN and intact_vs_max >= COMPOSITE_MARGIN
        )

    ranked_composites = [per_arm_composite[a] for a in ranked_by_similarity]
    is_nondecreasing = all(
        ranked_composites[i] <= ranked_composites[i + 1] + 1e-9
        for i in range(len(ranked_composites) - 1)
    )
    is_nonincreasing = all(
        ranked_composites[i] >= ranked_composites[i + 1] - 1e-9
        for i in range(len(ranked_composites) - 1)
    )
    monotonic_direction = "none"
    if is_nondecreasing and not is_nonincreasing:
        monotonic_direction = "improves_with_similarity"
    elif is_nonincreasing and not is_nondecreasing:
        monotonic_direction = "degrades_with_similarity"

    if not c1_pass:
        outcome = "FAIL"
        evidence_direction = "unknown"
        label = "substrate_not_ready_requeue"
        note = ("C1 (non-degeneracy) failed: no ablation moved the cross-stream similarity "
                f"statistic measurably away from intact (largest move: {max_abs_move_arm} at "
                f"{moves_vs_intact[max_abs_move_arm]:+.4f}, floor={SIMILARITY_MOVE_FLOOR}; "
                f"per-arm similarity={ {a: round(per_arm_similarity[a], 4) for a in ARMS} }; "
                f"degeneracy_reason={degeneracy['degeneracy_reason']!r}). Per INV-091's own "
                "non-degeneracy guard, the series tested nothing. Not a verdict on INV-091. "
                "Complementary to V3-EXQ-827a (landmark removal / lockstep), which DID clear "
                "this same guard on its two arms -- read the two manifests together.")
    elif c2_pass:
        outcome = "PASS"
        evidence_direction = "supports"
        label = "cross_stream_similarity_band_supported"
        note = ("C1 and C2 both hold: the interior-similarity arm "
                f"'{peak_arm}' (similarity={per_arm_similarity[peak_arm]:.4f}, composite="
                f"{per_arm_composite[peak_arm]:.3f}) exceeds both the empirical low-similarity "
                f"extreme '{sim_min_arm}' ({per_arm_composite[sim_min_arm]:.3f}) and the "
                f"high-similarity extreme '{sim_max_arm}' ({per_arm_composite[sim_max_arm]:.3f}) "
                f"by >= {COMPOSITE_MARGIN} -- a non-monotonic relation across five NEW "
                "ablations (event broadcasts / mode conditioning / residue feedback / "
                "randomised rates / harm-stream collapse). Secondary diagnostic (non-"
                f"load-bearing): intact_beats_both_extremes={intact_beats_both_extremes}. "
                "Complementary to V3-EXQ-827a (landmark removal / lockstep) -- both runs "
                "together span all seven named ablations; read jointly, not in isolation.")
    else:
        outcome = "FAIL"
        evidence_direction = "weakens"
        label = "cross_stream_similarity_band_not_supported"
        note = ("C1 held (at least one ablation moved similarity) but C2 failed: the peak-"
                f"composite arm '{peak_arm}' is "
                f"{'interior' if peak_is_interior else 'AT AN EMPIRICAL EXTREME (' + peak_arm + ' == ' + (sim_min_arm if peak_arm == sim_min_arm else sim_max_arm) + ')'}"
                f", margin_vs_min={margin_vs_min:.3f}, margin_vs_max={margin_vs_max:.3f} "
                f"(threshold {COMPOSITE_MARGIN}); monotonic_direction={monotonic_direction!r}. "
                f"Secondary diagnostic (non-load-bearing): intact_beats_both_extremes="
                f"{intact_beats_both_extremes}. Complementary to V3-EXQ-827a -- see "
                "interpretation.note there for the landmark-removal/lockstep arms.")

    manifest: Dict[str, Any] = {
        "run_id": f"{EXPERIMENT_TYPE}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_v3",
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "outcome": outcome,
        "evidence_direction": evidence_direction,
        "evidence_class": "experimental",
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "queue_id": QUEUE_ID,
        "seeds": list(SEEDS),
        "arms": list(ARMS),
        "similarity_pairs": [list(p) for p in SIMILARITY_PAIRS],
        "thresholds": {
            "similarity_move_floor": SIMILARITY_MOVE_FLOOR,
            "composite_margin": COMPOSITE_MARGIN,
            "base_max_lag_ticks": BASE_MAX_LAG_TICKS,
            "rate_random_e2_range": list(RATE_RANDOM_E2_RANGE),
            "rate_random_e3_range": list(RATE_RANDOM_E3_RANGE),
        },
        "per_arm_cross_stream_similarity": per_arm_similarity,
        "per_arm_anti_collapse": per_arm_anti_collapse,
        "per_arm_task_reward": per_arm_task,
        "per_arm_composite": per_arm_composite,
        "criteria": [
            {"name": "C1_at_least_one_ablation_moves_similarity", "load_bearing": True, "passed": c1_pass},
            {"name": "C2_interior_similarity_arm_beats_both_extremes", "load_bearing": True, "passed": c2_pass},
        ],
        "criteria_non_degenerate": {
            "C1_at_least_one_ablation_moves_similarity": bool(degeneracy["non_degenerate"]),
            "C2_interior_similarity_arm_beats_both_extremes": bool(
                np.isfinite(margin_vs_min) and np.isfinite(margin_vs_max)
            ),
        },
        "band_shape": {
            "ranked_by_similarity": ranked_by_similarity,
            "sim_min_arm": sim_min_arm,
            "sim_max_arm": sim_max_arm,
            "peak_composite_arm": peak_arm,
            "peak_is_interior": peak_is_interior,
            "margin_vs_min": margin_vs_min,
            "margin_vs_max": margin_vs_max,
            "intact_beats_both_extremes": intact_beats_both_extremes,
            "intact_vs_min": intact_vs_min,
            "intact_vs_max": intact_vs_max,
        },
        "monotonic_direction": monotonic_direction,
        "null_validation": null_report,
        "surrogate_p_values": surrogate_report,
        "interpretation": {"label": label, "note": note},
        "arm_results": [
            {
                "arm_id": a, "seed": s,
                "mean_episode_reward": per_seed[s][a]["mean_episode_reward"],
                "cross_stream_similarity": per_seed[s][a]["cross_stream_similarity"],
                "anti_collapse": per_seed[s][a]["anti_collapse"],
                "episode_rewards": per_seed[s][a]["episode_rewards"],
                "n_steps": per_seed[s][a]["n_steps"],
                "trace_pointer": per_seed[s][a]["trace_pointer"],
                "arm_fingerprint": per_seed[s][a]["arm_fingerprint"],
                **({"sampled_rates": per_seed[s][a]["sampled_rates"]}
                   if a == "rate_randomized" else {}),
            }
            for a in ARMS for s in SEEDS
        ],
        "aggregates": {
            "per_level": {
                a: {
                    "cross_stream_similarity_mean": per_arm_similarity[a],
                    "anti_collapse_mean": per_arm_anti_collapse[a],
                    "task_reward_mean": per_arm_task[a],
                    "composite_mean": per_arm_composite[a],
                }
                for a in ARMS
            },
        },
        "dose_key": "cross_stream_similarity_mean",
        "supersedes": None,
        "complements": ["V3-EXQ-827a"],
    }
    manifest.update(degeneracy)

    config_snapshot = {
        "env_kwargs": _env_kwargs(),
        "q081_profile": dict(q081_profile_kwargs()),
        "alpha_world": 0.9,
        "use_anchor_sets": True,
        "warmup_episodes": warmup_episodes,
        "eval_episodes": eval_episodes,
        "steps_per_episode": steps_per_episode,
        "similarity_pairs": [list(p) for p in SIMILARITY_PAIRS],
        "thresholds": manifest["thresholds"],
    }

    out_path = write_flat_manifest(
        manifest, dry_run=dry_run, config=config_snapshot, seeds=list(SEEDS),
        script_path=script_path, started_at=t0,
    )
    manifest["_out_path"] = str(out_path)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    result = main(dry_run=args.dry_run)
    print(f"outcome: {result['outcome']}")
    print(f"interpretation: {result['interpretation']['label']}")

    emit_outcome(
        outcome=result["outcome"],
        manifest_path=result["_out_path"],
        run_id=result["run_id"],
        dry_run=args.dry_run,
    )
