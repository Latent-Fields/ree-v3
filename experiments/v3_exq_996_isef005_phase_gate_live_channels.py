"""
V3-EXQ-996 -- LIVE-CHANNEL redesign of the InfantCurriculumScheduler Phase 0->1
crossing-count validation (successor to V3-EXQ-591h; NEW EXQ NUMBER per
infant_gap14_redesign_staged_20260827.md, not a 591 letter).

SLEEP DRIVER: K=never (SleepLoopManager instantiated with K > total episodes via
the diversity-armed builder; never fires during this readiness probe -- faithful
to the 591b-591h lineage).

red-team (fable): CONTESTED on the FIRST draft (2 should-fix findings); BOTH fixed
in this version -- see RED-TEAM VERDICT section below for the full disposition.

QUESTION
--------
Same question as V3-EXQ-591f/591h: does the V3-EXQ-591f-validated crossing-count
Phase 0->1 advancement criterion (advance only once the cumulative post-ep_min
spike-bar crossings reach 3) discriminate genuine explorers from a false-advancer
WHEN RUN IN LIVE CLOSED LOOP -- i.e. does the scheduler's phase actually reach
the agent's behaviour, so that a live run is evidence about anything at all?

WHY THIS IS A NEW NUMBER, NOT ANOTHER 591 LETTER (per EXQ versioning policy and
the explicit governance instruction below): 591b-591h all wired the SAME single
phase->agent channel -- config.e3.novelty_bonus_weight -- and that channel's
consuming branch (a uniform scalar shift over CEM candidates, argmin-invariant)
was DELETED 2026-05-25 as dead-by-construction. V3-EXQ-591h ran 4.8h and PASSED
5/5 seeds with a unanimous "the gate discriminates" verdict while
per_episode_h_pos was BIT-IDENTICAL between ARM_SPIKE and ARM_CROSSING on 5/5
seeds (0 differing episodes, max |diff| 0.0) -- the manipulation never reached
behaviour. Confirmed: failure_autopsy_V3-EXQ-591h_2026-09-03 (ratified REE_assembly
d51fc6aaa5), recorded on infant_substrate:GAP-14 (governance_2026_09_03). That
autopsy names the fix as a mechanism change (which agent channel the phase
writes to), not a parameter retune of the SAME mechanism -- hence a new EXQ
number, per the autopsy's own instruction (c).

THE FIX -- ONE GENUINELY LIVE CHANNEL (per governance_2026_09_03; NARROWED from
a two-channel draft after red-team review, see RED-TEAM VERDICT below):
  (a) InfantCurriculumScheduler.env_kwargs() is SPREAD INTO the
      CausalGridWorldV2 constructor every episode (harm_gradient_enabled ->
      causal_grid_world.py:2617 harm-gradient reward term consumed by
      update_residue() -> ResidueField.accumulate() -> ARC-046 hazard-avoidance
      term in E3 trajectory scoring; transient_benefit_enabled -> :3077 spawns
      high-salience benefit patches that change resource-approach dynamics).
      This is a GENUINE closed loop: env_kwargs differing by phase changes the
      harm_signal stream the agent observes -> changes update_residue()'s
      accumulation -> changes E3's harm-avoidance scoring on SUBSEQUENT ticks ->
      changes which actions are selected -> changes h_pos. Verified reachable
      by direct construction probe before this script was written (Step 2.5a):
      CausalGridWorldV2(harm_gradient_enabled=True, transient_benefit_enabled=True,
      ...) constructs and steps without error; world_obs_dim is unchanged by
      either flag (250, matching BODY_OBS_DIM/WORLD_OBS_DIM below). Verified
      firing LIVE in this script via a per-cell count of
      `info["transition_type"] == "harm_gradient"` ticks (see
      `n_harm_gradient_ticks` in each cell's row and
      `n_harm_gradient_ticks_by_cell` in the manifest) -- a direct behavioural
      confirmation that channel (a) engaged, not merely that it was wired.

  (b) `InfantCurriculumScheduler.config_overrides()['offline_integration_frequency']`
      is DELIBERATELY NOT WIRED, despite the original governance instruction
      naming it as a required channel alongside env_kwargs(). Step 4.5 red-team
      (fable, an independent model) found that the FIRST DRAFT's workaround for
      this channel -- calling the confirmed-safe half of
      `REEAgent.offline_integration()`, `agent.residue_field.integrate()`, at
      the offline_integration_frequency cadence -- is itself a DEAD CHANNEL:
      read `ree_core/residue/field.py:1120-1141` in full and confirm there is
      no `.backward()`, no optimizer, no `.step()`, and no in-place write
      anywhere in the method or in `RBFLayer`/`neural_field` -- it computes
      `F.mse_loss(...).item()` and discards it. `neural_field`'s parameters are
      grep-confirmed to be referenced NOWHERE else in ree_core/, so nothing
      anywhere ever trains them. The ONLY observable side effect of calling
      `integrate()` is that `torch.randn_like(harm_locations)` (10 draws per
      call) advances the GLOBAL torch RNG stream, which every downstream
      stochastic draw (CEM noise, E3 multinomial, replay) then consumes
      differently. So wiring this "channel" would have reached behaviour only
      by RNG-stream perturbation unrelated to the curriculum's actual content
      -- exactly the 591h failure mode in a second costume: a "live channel"
      that is dead-by-construction, except this time the P6 divergence
      precondition (below) would NOT have caught it, because RNG-stream
      divergence produces real, non-zero |diff| in per_episode_h_pos that has
      NOTHING to do with the curriculum's env_kwargs/gate-decision manipulation
      under test. Leaving it wired but silently vacuous would have been WORSE
      than 591h's failure, not merely a repeat of it.
      DECISION (this version): `agent.config.offline_integration_frequency` is
      never set by this script; `agent.should_integrate()` /
      `agent.residue_field.integrate()` are never called. The channel is
      recorded as a SCOPE LIMITATION with a substrate-defect finding (below),
      not claimed as live.

  SUBSTRATE DEFECTS FOUND (Step 2.5a probe + Step 4.5 red-team; NEITHER fixed
  here -- ree_core edits are out of scope for a /queue-experiment session;
  flagged for /implement-substrate follow-up):
    (i) `REEAgent.offline_integration()` (agent.py ~12053-12058) builds its
        E1-replay batch via `torch.cat([s, w]) for s, w in zip(...)` -- a bare
        `torch.cat` with the DEFAULT dim=0, on tensors of shape [1, self_dim] /
        [1, world_dim]. The CORRECT sibling usage three lines above the
        buffer-append site (agent.py:5775-5777,
        `torch.cat([gated_for_e1.z_self, gated_for_e1.z_world], dim=-1)`) shows
        this is a bug, not a deliberate 1-D-vector convention: with the common
        default self_dim==world_dim (both 32 under REEConfig.from_dims with no
        override, as used by every script in the 591 family including this
        one), `cat(dim=0)` SILENTLY produces a [2, D] tensor instead of the
        intended [1, 2D] feature concatenation; `integrate_experience()`'s
        downstream `torch.stack` + slicing then feeds
        `E1DeepPredictor.context_memory.read()` a wrongly-shaped batch, and it
        crashes (`RuntimeError: mat1 and mat2 shapes cannot be multiplied`)
        once the experience buffer exceeds 10 entries -- deterministically,
        within the first ~11 E1 ticks of any real (non-toy) run. Confirmed live
        via a direct 15-step REEAgent probe before this script was written.
    (ii) `ResidueField.integrate()` (`ree_core/residue/field.py:1120-1141`,
        the E1-replay-free "safe half" of (i)) trains nothing: it computes a
        loss (`F.mse_loss` between `neural_field` and the frozen `rbf_field`
        readout) and discards it via `.item()`, with no backward pass and no
        parameter update anywhere. `neural_field`'s only two call sites in the
        whole codebase are both inside this one method. Confirmed by grep
        (Step 4.5 red-team, corroborated directly against the source before
        landing this version).
  Net: `offline_integration_frequency` has NEVER had a working closed-loop
  behavioural consumer anywhere in ree-v3 -- neither the broken E1-replay half
  nor the safe-but-inert residue half. Grep-confirmed: zero scripts in the
  whole ree-v3/experiments/ corpus call `agent.offline_integration()` or
  `agent.should_integrate()`.

DESIGN -- discriminative pair (EVB-1189 dispatch_mode=discriminative_pair)
--------------------------------------------------------------------------
Unchanged from 591h: two arms, matched seeds, identical in every respect except
the Phase 0->1 gate RULE.

  ARM_SPIKE    (control / ablation) -- legacy single-episode spike gate.
  ARM_CROSSING (primary)            -- phase_0to1_use_crossing_count=True,
                 min 3 crossings.

Each arm runs its OWN InfantCurriculumScheduler instance, so the two arms'
phase timelines (and hence env_kwargs / offline_integration_frequency schedules)
diverge independently based on each arm's own gate.

Seeds 42-46 (faithful to 591b-591h), 160 episodes, 200 steps/episode,
diversity-armed agent build (MECH-313 noise floor + MECH-314 structured
curiosity, SP-CEM main-path default).

THE ORACLE (independent of the criterion under test) -- CHANGED FROM 591h
(Step 4.5 red-team finding). 591h computed the oracle from `h_pos_mean_full_run`
on ARM_SPIKE; that was safe there because env_kwargs was informational-only
(dead channel), so no arm's post-advance episodes ran under different dynamics.
NOW that env_kwargs() is genuinely live (channel a, above), ARM_SPIKE's own
post-advance episodes run under Phase-1 dynamics (harm gradient, transient
benefit), so `h_pos_mean_full_run` would make the "independent" oracle a
function of ARM_SPIKE's OWN gate decision (whether and when it advanced) --
contaminating the very ground truth the gate is being tested against.
THIS VERSION uses `h_pos_mean_pre_ep_min` instead: the mean over episodes
[0, PHASE_EP_MIN[1]=100), which is GUARANTEED Phase-0-only in every arm --
`InfantCurriculumScheduler._try_phase_0_to_1` (infant_curriculum.py:330)
refuses to advance before episode PHASE_EP_MIN[1] under either gate rule, so
this window is identical-dynamics and (given `arm_cell`'s per-seed RNG reset)
provably bit-identical in composition across arms; it cannot be contaminated
by which arm's gate later fires. A seed is an ORACLE genuine explorer iff this
pre-ep_min mean clears ORACLE_H_POS_MEAN_FLOOR = 0.20. Level-only (the
crossing-count half of the 591d/591f oracle is dropped to avoid circularity).
NOTE ON THE FLOOR: 0.20 was originally calibrated (591d/591f) against
FULL-RUN h_pos_mean, not pre-ep_min h_pos_mean -- REFERENCE_591F below (used
only for the structural anchor-reachability check, not as a live label) still
carries the original full-run values, since no pre-ep_min figure was recorded
for those cells. Applying the same numeral to a different (pre-ep_min)
statistic is an approximation, not a re-derivation; the READINESS PRECONDITIONS
(P3/P4 below) are the live-run safety net -- if 0.20 turns out to be poorly
calibrated for the pre-ep_min statistic on this run's seeds (e.g. it fails to
split them into both classes), the run self-routes to
substrate_not_ready_requeue rather than reporting a false verdict; it cannot
silently pass through a miscalibrated floor. REFERENCE_591F freezes the 591f
positive-control cells so assert_anchor_reachable can prove, at setup, that
every readiness predicate this script ships can clear its own gate on a
known-positive control -- this is a STRUCTURAL check on the predicate
machinery, unaffected by which agent channels this script wires.

ACCEPTANCE
----------
LOAD-BEARING C_live_gate_discriminates -- unchanged from 591h: in ARM_CROSSING,
for EVERY seed, the live phase-advance decision agrees with the ORACLE label.

NEW, LOAD-BEARING, EVALUATED FIRST -- P6 raw_trajectory_divergence_present
(the guard whose ABSENCE made 591h vacuous; governance_2026_09_03: "carry a
precondition asserting the two arms' trajectories diverge somewhere BEFORE the
verdict is computed"). 591h HAD a non-degeneracy check (`C_arms_differ`), but it
compared the two arms' ADVANCE DECISIONS (`reached_phase1`, `phase_01_at`), not
their raw behaviour -- and the advance-decision timing genuinely differs between
a single-episode-spike gate and a >=3-crossing gate EVEN WHEN THE UNDERLYING
per-episode h_pos SEQUENCE IS BIT-IDENTICAL (a different threshold applied to the
identical signal still fires at a different episode). That is exactly the 591h
failure mode: `C_arms_differ` was TRUE (decisions differed) while
per_episode_h_pos was IDENTICAL (behaviour never differed). P6 instead counts
episodes where the RAW per_episode_h_pos sequences differ by more than float
noise between ARM_SPIKE and ARM_CROSSING, summed across all seeds. Evaluated
BEFORE gate_green / dropout_bypassed / c_discriminates are allowed to determine
the label: below floor (0 diverging episodes anywhere) means the manipulation
did not reach behaviour in THIS run, and the run self-routes to
substrate_not_ready_requeue / vacuous_design -- never a substrate-verdict label
-- regardless of what the (still-computed, for diagnostic completeness)
advance-decision-level checks say.

READINESS PRECONDITIONS P1-P5 (below-floor -> substrate_not_ready_requeue,
never a verdict label) -- unchanged from 591h, all measured on ARM_SPIKE, the
positive control:
  P1 spike_arm_reproduces_advance       -- COUNT of ARM_SPIKE seeds reaching
       Phase 1 >= 2.
  P2 crossing_counts_reach_gate_minimum -- CHANGED FROM 591h (Step 4.5 red-team
       finding): 591h read this off a POST-HOC recount of ARM_SPIKE's own
       per-episode h_pos values against the spike threshold; now that
       ARM_SPIKE's post-advance dynamics are also live (env_kwargs), that
       recount is exposed to the same contamination concern as the oracle
       (above). This version reads MAX over seeds of ARM_CROSSING's own
       LIVE `scheduler_crossing_count` -- the exact statistic
       InfantCurriculumScheduler itself computes and routes the gate on
       (phase_summary()['phase01_crossing_count']) -- directly, with no
       post-hoc recomputation from a different arm's trajectory.
       >= PHASE_01_CROSSING_COUNT_MIN.
  P3 oracle_genuine_explorers_present   -- COUNT of oracle-genuine seeds >= 1.
  P4 oracle_non_explorers_present       -- COUNT of oracle-non-genuine seeds >= 1.
  P5 live_false_advancer_present        -- COUNT of seeds that are
       oracle-non-genuine AND admitted by the legacy spike gate >= 1.

DV-SYMMETRY INVARIANCE (mandatory declaration, per arm)
-------------------------------------------------------
DV (both arms): per-seed Phase 0->1 advance EPISODE and final phase, PLUS (new
in this script) the raw per-episode h_pos sequence itself.
Manipulation: the gate RULE (spike-count threshold: 1 vs 3), which now ALSO
determines the episode range over which each arm's env_kwargs take their
Phase-1 values (since phase timing differs, the manipulation is not merely a
relabelling of a fixed signal -- it changes which episodes are generated under
which environment dynamics). The DV
is not invariant under any of: a uniform additive constant (env_kwargs is a
discrete on/off switch on reward *terms*, not a scalar added to a selection
score), a monotone rescaling (h_pos is a raw entropy statistic, not a rank), or
a permutation of interchangeable units (episode order is load-bearing -- Phase 1
env_kwargs apply only from the advance episode forward). The one regime where
the manipulation IS invisible to the DV -- both arms staying in Phase 0 for the
whole run (env_kwargs never differ) -- is precisely what P6 exists to detect and
self-route on.

SCOPE LIMITATION (recorded, not silently inherited)
----------------------------------------------------
`config.e3.novelty_bonus_weight` is NOT written by this script at all (unlike
591b-591h, which wrote it every episode to a value nothing consumes). It sits at
its initial-build constant (0.5, from _agent_config_kwargs()) for the entire run
in BOTH arms -- deliberately, since governance_2026_09_03 states the successor
must NOT rely on it.
`config_overrides()['residue_scale_factor']` is NOT applied by this script
(governance named only env_kwargs() and offline_integration_frequency as
required channels; residue_scale_factor has a real consumer elsewhere
(ResidueField.get_coverage_telemetry(residue_scale_factor=...)) but threading it
in is out of the scope governance specified, and is left for a future redesign
pass rather than added speculatively here).
`offline_integration_frequency` is NOT applied by this script AT ALL (narrowed
from the original two-channel governance instruction after Step 4.5 red-team --
see THE FIX above): `agent.offline_integration()`'s E1-replay half is
confirmed-broken (substrate defect (i)); its "safe" residue-only half,
`agent.residue_field.integrate()`, is confirmed to train nothing (substrate
defect (ii)) and its only observable effect would have been to perturb the
global torch RNG stream on a schedule unrelated to the curriculum content --
which the P6 divergence precondition below cannot distinguish from genuine
curriculum-driven divergence. Wiring it would have manufactured a second,
subtler instance of 591h's own failure mode rather than fixing the one 591h
had. Both substrate defects are recorded in the manifest under
`scope_limitation` and `substrate_defect_found` (two entries).

RED-TEAM VERDICT
----------------
red-team (fable, Step 4.5, independent model from the author): CONTESTED on
the first draft -- 2 should-fix findings, both applied in this version:
  F1 (the more serious): the original draft's `offline_integration_frequency`
     workaround (`agent.residue_field.integrate()`) does not train anything
     (see substrate defect (ii) above) -- its only real effect was RNG-stream
     perturbation, a non-curriculum divergence source the P6 precondition
     cannot tell apart from a real one. FIX: the channel is no longer wired at
     all; recorded as a scope limitation + substrate-defect finding instead of
     claimed live.
  F2: the oracle (`h_pos_mean_full_run` on ARM_SPIKE) and the P2 readiness
     check (a post-hoc recount of ARM_SPIKE's post-advance h_pos values) both
     became contaminated by the manipulation once env_kwargs() went live,
     because ARM_SPIKE's own post-advance episodes now run under different
     (Phase-1) dynamics that depend on ARM_SPIKE's own gate decision. FIX:
     oracle switched to `h_pos_mean_pre_ep_min` (provably Phase-0-only,
     uncontaminated in every arm); P2 switched to read ARM_CROSSING's own live
     `scheduler_crossing_count` directly instead of a post-hoc recount.
  Minor/note findings (F3 diagnostic-only, F4 confirms the substrate-defect
  claim, F5 confirms no other vacuity risk, F6 cosmetic) did not require
  script changes beyond F3's cheap sharpening (a live
  `info["transition_type"]=="harm_gradient"` tick counter, added as a
  diagnostic to directly confirm channel (a) fires, not merely that it is
  wired) -- see `n_harm_gradient_ticks` in the per-cell rows and
  `n_harm_gradient_ticks_by_cell` in the manifest.
Full findings text preserved verbatim in the queue entry note.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from experiment_protocol import emit_outcome  # noqa: E402
from infant_curriculum import (  # noqa: E402
    InfantCurriculumScheduler,
    H_POS_FRAC_OF_MAX,
    PHASE_01_CROSSING_COUNT_MIN,
    PHASE_EP_MIN,
)
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402

# Reuse the canonical 591 helpers + constants (DRY -- no copy-drift).
from v3_exq_591_isef005_curriculum_vs_flat_v3 import (  # noqa: E402
    _extract_obs,
    BODY_OBS_DIM,
    WORLD_OBS_DIM,
    GRID_SIZE,
    ACTION_DIM,
)
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._lib.readiness_anchor import assert_anchor_reachable  # noqa: E402
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402

QUEUE_ID = "V3-EXQ-996"
EXPERIMENT_TYPE = "v3_exq_996_isef005_phase_gate_live_channels"
EXPERIMENT_PURPOSE = "diagnostic"

# Claim-free (matches the whole 591 lineage post-correction: 591h's ARC-019 tag
# was removed by governance 2026-09-03; ARC-046/infant_substrate:GAP-14's
# bears_on set does not include this run's mechanism at the requisite
# granularity to be tested here -- this validates the GATE, not the full
# curriculum-vs-flat comparison ARC-046 needs).
CLAIM_IDS: List[str] = []
EVIDENCE_DIRECTION = "non_contributory"

# --- pre-registered constants (NOT derived from this run's own statistics) ---------------
SEEDS = [42, 43, 44, 45, 46]     # the lineage seed draw (591b/591c/591d/591e/591f/591h)
N_EPISODES = 160                 # faithful to 591b-591h
STEPS_PER_EPISODE = 200          # faithful to 591b-591h

H_MAX = math.log(GRID_SIZE ** 2)
PHASE_01_THRESHOLD = H_POS_FRAC_OF_MAX * H_MAX   # the spike bar, ~0.9940 at grid 12

# Readiness precondition thresholds (all pre-registered).
P1_MIN_SPIKE_ADVANCERS = 2
P2_MIN_MAX_CROSSINGS = PHASE_01_CROSSING_COUNT_MIN   # 3 -- SAME statistic the gate routes on
P3_MIN_ORACLE_GENUINE = 1
P4_MIN_ORACLE_NON_GENUINE = 1
P5_MIN_LIVE_FALSE_ADVANCERS = 1
# P6: minimum count of raw per-episode h_pos divergences (across ALL seeds,
# BOTH arms compared pointwise) required to license any verdict at all. "diverge
# somewhere" (governance_2026_09_03) -> floor of 1. A per-episode |diff| above
# this numerical-noise epsilon counts as a genuine divergence, not float jitter.
P6_MIN_DIVERGING_EPISODES = 1
P6_DIVERGENCE_EPSILON = 1e-9

# ORACLE: level-only. Floor unchanged from the 591 lineage.
ORACLE_H_POS_MEAN_FLOOR = 0.20
ORACLE_WINDOW_END = PHASE_EP_MIN[1]   # 100 -- pre-ep_min slice kept as INFORMATIONAL only

# Frozen positive-control reference: the recorded per-seed cells of
# v3_exq_591f_isef005_phase01_gate_criterion_20260615T115131Z_v3 (per_seed_criteria).
# Used ONLY by assert_anchor_reachable at setup, to prove each shipped readiness
# predicate can clear its own gate on a known-good control. Never used as a live
# label, and unaffected by which agent channels THIS script wires (structural
# check on the predicate machinery only).
REFERENCE_591F = [
    {"seed": 42, "h_pos_mean": 0.5621, "n_eligible_ge_threshold": 7,  "spike_advanced": True},
    {"seed": 43, "h_pos_mean": 0.3226, "n_eligible_ge_threshold": 6,  "spike_advanced": True},
    {"seed": 44, "h_pos_mean": 0.8424, "n_eligible_ge_threshold": 36, "spike_advanced": True},
    {"seed": 45, "h_pos_mean": 0.1404, "n_eligible_ge_threshold": 2,  "spike_advanced": True},
    {"seed": 46, "h_pos_mean": 0.0375, "n_eligible_ge_threshold": 0,  "spike_advanced": False},
]
REFERENCE_591F_SOURCE = (
    "v3_exq_591f_isef005_phase01_gate_criterion_20260615T115131Z_v3.per_seed_criteria")

# --- THE SHIPPED PREDICATES (the same callables the live cells are scored with) ----------
def _is_oracle_genuine(cell) -> bool:
    return float(cell["h_pos_mean"]) >= ORACLE_H_POS_MEAN_FLOOR


def _is_oracle_non_genuine(cell) -> bool:
    return not _is_oracle_genuine(cell)


def _meets_crossing_minimum(cell) -> bool:
    return int(cell["n_eligible_ge_threshold"]) >= P2_MIN_MAX_CROSSINGS


def _spike_advanced(cell) -> bool:
    return bool(cell["spike_advanced"])


def _is_live_false_advancer(cell) -> bool:
    """CONJUNCTIVE, as 591f's own `false_advancer` was: oracle-non-genuine AND
    admitted by the legacy spike gate."""
    return _is_oracle_non_genuine(cell) and _spike_advanced(cell)


ARM_SPIKE = "ARM_SPIKE"
ARM_CROSSING = "ARM_CROSSING"
ARMS = [ARM_SPIKE, ARM_CROSSING]

_ZG = ZGoalStreamAccumulator()


def _agent_config_kwargs() -> Dict[str, Any]:
    """The 591c diversity-armed build, as a declared dict (also the fingerprint
    slice). Bit-for-bit the same build the whole 591 lineage uses."""
    return {
        "body_obs_dim": BODY_OBS_DIM,
        "world_obs_dim": WORLD_OBS_DIM,
        "action_dim": ACTION_DIM,
        "z_goal_enabled": True,
        "drive_weight": 2.0,
        "novelty_bonus_weight": 0.5,
        "use_sleep_loop": True,
        "sleep_loop_episodes_K": N_EPISODES + 1,   # K=never
        "use_noise_floor": True,                   # MECH-313
        "use_structured_curiosity": True,          # MECH-314
    }


def _build_diversity_agent() -> REEAgent:
    """Bit-identical build discipline to 591c/591d/591e/591f/591h."""
    cfg = REEConfig.from_dims(**_agent_config_kwargs())
    cfg.latent.alpha_world = 0.9
    cfg.sws_enabled = True
    cfg.rem_enabled = True
    return REEAgent(cfg)


def _config_slice(arm: str) -> Dict[str, Any]:
    """Everything the cell computation reads. The gate rule is INCLUDED -- it is
    what distinguishes the two arms, so omitting it would collapse their
    fingerprints. `live_channels` records the substrate-level design choice
    (this script's whole point) so the fingerprint changes if that choice ever
    does."""
    return {
        "agent": _agent_config_kwargs(),
        "latent_alpha_world": 0.9,
        "sws_enabled": True,
        "rem_enabled": True,
        "env": {
            "size": GRID_SIZE,
            "resource_respawn_on_consume": True,
            "pos_telemetry_enabled": True,
            "traj_telemetry_enabled": True,
        },
        "schedule": {
            "n_episodes": N_EPISODES,
            "steps_per_episode": STEPS_PER_EPISODE,
        },
        "gate": {
            "phase_0to1_use_crossing_count": arm == ARM_CROSSING,
            "phase_0to1_crossing_count_min": PHASE_01_CROSSING_COUNT_MIN,
            "h_pos_frac_of_max": H_POS_FRAC_OF_MAX,
        },
        "live_channels": {
            "env_kwargs_applied": True,
            "offline_integration_frequency_applied": False,  # dead channel; see scope_limitation
            "offline_integration_e1_replay_invoked": False,  # confirmed-broken (substrate defect i)
            "residue_field_integrate_invoked": False,        # confirmed inert (substrate defect ii)
            "novelty_bonus_weight_written": False,           # deliberately NOT written (dead)
        },
    }


def _run_cell(*, arm: str, seed: int, n_episodes: int,
              steps_per_episode: int = STEPS_PER_EPISODE) -> Dict[str, Any]:
    """One (arm x seed) cell. Mirrors the 591h seed runner; the gate rule AND the
    live-channel wiring differ from 591h (see module docstring)."""
    print(f"Seed {seed} Condition {arm}", flush=True)

    with arm_cell(
        seed,
        config_slice=_config_slice(arm),
        script_path=Path(__file__),
        config_slice_declared=True,
        include_driver_script_in_hash=False,   # cross-driver reusable mint (both arms)
    ) as cell:
        torch.manual_seed(seed)
        agent = _build_diversity_agent()
        sched = InfantCurriculumScheduler(
            grid_size=GRID_SIZE,
            phase_0to1_use_crossing_count=(arm == ARM_CROSSING),
            phase_0to1_crossing_count_min=PHASE_01_CROSSING_COUNT_MIN,
        )

        per_ep_h_pos: List[float] = []
        phase_01_at: Optional[int] = None
        phase_12_at: Optional[int] = None
        n_phase1_episodes = 0
        n_harm_gradient_ticks = 0

        for ep in range(n_episodes):
            # LIVE CHANNEL (a): env_kwargs() spread into the env constructor for
            # THIS episode, reflecting the scheduler's phase at episode start.
            # THIS IS THE ONLY LIVE CHANNEL -- offline_integration_frequency is
            # deliberately NOT applied (Step 4.5 red-team finding; see module
            # docstring THE FIX / SCOPE LIMITATION); config.e3.novelty_bonus_weight
            # is likewise deliberately NOT written (dead since 2026-05-25).
            _ek = sched.env_kwargs()
            env = CausalGridWorldV2(
                size=GRID_SIZE,
                seed=seed * n_episodes + ep,
                resource_respawn_on_consume=True,
                pos_telemetry_enabled=True,
                traj_telemetry_enabled=True,
                **_ek,
            )
            if sched.current_phase >= 1:
                n_phase1_episodes += 1

            _flat, obs_dict = env.reset()
            ob, ow = _extract_obs(obs_dict)

            ep_h_pos = -1.0
            ep_benefit_contacts = 0

            for _step in range(steps_per_episode):
                with torch.no_grad():
                    action = agent.act_with_split_obs(obs_body=ob, obs_world=ow)
                ai = int(action.argmax().item()) % ACTION_DIM
                _o, harm_signal, done, info, obs_dict = env.step(ai)
                agent.update_residue(float(harm_signal))
                ob, ow = _extract_obs(obs_dict)
                benefit = float(ob[11].item()) if ob.shape[0] > 11 else 0.0
                energy = float(ob[3].item()) if ob.shape[0] > 3 else 0.5
                drive = max(0.0, min(1.0, 1.0 - energy))
                agent.update_z_goal(benefit_exposure=benefit, drive_level=drive)
                ep_h_pos = float(info.get("pos_entropy", -1.0))
                ep_benefit_contacts += int(
                    float(info.get("transient_benefit_contact_this_tick", 0.0)) > 0.0)
                # Diagnostic-only (Step 4.5 red-team F3): direct live confirmation
                # that LIVE CHANNEL (a) actually fired, not merely that it was
                # wired -- counts ticks where the env's own harm-gradient reward
                # branch was taken this step.
                if str(info.get("transition_type", "")) == "harm_gradient":
                    n_harm_gradient_ticks += 1
                if done:
                    _flat, obs_dict = env.reset()
                    ob, ow = _extract_obs(obs_dict)

            z_norm = agent.goal_state.goal_norm() if agent.goal_state is not None else 0.0
            cov = float(agent.residue_field.get_coverage_telemetry()["residue_coverage_pct"])

            per_ep_h_pos.append(ep_h_pos)

            prev_phase = sched.current_phase
            sched.update(
                ep,
                h_pos=ep_h_pos if ep_h_pos >= 0.0 else None,
                z_goal_norm=z_norm,
                benefit_contacts=ep_benefit_contacts,
                residue_coverage_pct=cov,
            )
            if prev_phase == 0 and sched.current_phase >= 1 and phase_01_at is None:
                phase_01_at = ep
            if prev_phase <= 1 and sched.current_phase >= 2 and phase_12_at is None:
                phase_12_at = ep

            if (ep + 1) % 50 == 0 or (ep + 1) == n_episodes:
                print(
                    f"  [train] {arm} seed={seed} ep {ep + 1}/{n_episodes}"
                    f" phase={sched.current_phase} h_pos={ep_h_pos:.3f}"
                    f" crossings={sched.phase_summary().get('phase01_crossing_count', 0)}"
                    f" env_kwargs={_ek}",
                    flush=True,
                )

        _ZG.observe(agent)

        valid = [h for h in per_ep_h_pos if h >= 0.0]
        pre = [h for h in per_ep_h_pos[:ORACLE_WINDOW_END] if h >= 0.0]
        post = [h for h in per_ep_h_pos[ORACLE_WINDOW_END:] if h >= 0.0]

        row: Dict[str, Any] = {
            "arm_id": arm,
            "seed": seed,
            "final_phase": sched.current_phase,
            "reached_phase1": bool(sched.current_phase >= 1),
            "phase_01_at": phase_01_at,
            "phase_12_at": phase_12_at,
            "n_phase1_episodes": n_phase1_episodes,
            "n_harm_gradient_ticks": n_harm_gradient_ticks,
            # the COUNT statistic the crossing-count gate routes on
            "post_ep_min_crossings": sum(1 for h in post if h >= PHASE_01_THRESHOLD),
            "scheduler_crossing_count": int(
                sched.phase_summary().get("phase01_crossing_count", 0)),
            "h_pos_mean_pre_ep_min": (sum(pre) / len(pre)) if pre else -1.0,
            "h_pos_mean_full_run": (sum(valid) / len(valid)) if valid else -1.0,
            "h_pos_max_full_run": max(valid) if valid else -1.0,
            "n_pre_ep_min_crossings": sum(1 for h in pre if h >= PHASE_01_THRESHOLD),
            "per_episode_h_pos": per_ep_h_pos,
            "final_z_goal_norm": float(z_norm),
            "final_residue_coverage_pct": float(cov),
            "phase_summary": sched.phase_summary(),
        }
        cell.stamp(row)

    verdict = "PASS" if row["reached_phase1"] else "FAIL"
    print(f"verdict: {verdict}", flush=True)
    return row


def _assert_readiness_anchors_reachable() -> List[Dict[str, Any]]:
    """Prove at SETUP that every shipped readiness predicate can clear its own
    gate on the frozen 591f positive control (structural check, independent of
    this script's live-channel wiring). Raises AnchorUnreachable."""
    n = len(REFERENCE_591F)
    return [
        assert_anchor_reachable(
            anchor_name="spike_arm_reproduces_advance",
            reference_cells=REFERENCE_591F, score_fn=_spike_advanced,
            threshold=P1_MIN_SPIKE_ADVANCERS / n, reference_source=REFERENCE_591F_SOURCE),
        assert_anchor_reachable(
            anchor_name="crossing_counts_reach_gate_minimum",
            reference_cells=REFERENCE_591F, score_fn=_meets_crossing_minimum,
            threshold=1.0 / n, reference_source=REFERENCE_591F_SOURCE),
        assert_anchor_reachable(
            anchor_name="oracle_genuine_explorers_present",
            reference_cells=REFERENCE_591F, score_fn=_is_oracle_genuine,
            threshold=P3_MIN_ORACLE_GENUINE / n, reference_source=REFERENCE_591F_SOURCE),
        assert_anchor_reachable(
            anchor_name="oracle_non_explorers_present",
            reference_cells=REFERENCE_591F, score_fn=_is_oracle_non_genuine,
            threshold=P4_MIN_ORACLE_NON_GENUINE / n, reference_source=REFERENCE_591F_SOURCE),
        assert_anchor_reachable(
            anchor_name="live_false_advancer_present",
            reference_cells=REFERENCE_591F, score_fn=_is_live_false_advancer,
            threshold=P5_MIN_LIVE_FALSE_ADVANCERS / n,
            reference_source=REFERENCE_591F_SOURCE),
    ]


def _count_diverging_episodes(spike_h: List[float], cross_h: List[float]) -> int:
    """P6: raw per-episode h_pos divergence count between the two arms' ACTUAL
    trajectories (not their advance decisions). This is the guard 591h lacked."""
    n = min(len(spike_h), len(cross_h))
    return sum(
        1 for i in range(n)
        if abs(spike_h[i] - cross_h[i]) > P6_DIVERGENCE_EPSILON
    )


def run_experiment(*, dry_run: bool = False) -> Dict[str, Any]:
    t0 = time.perf_counter()
    # Setup-time guard -- runs on --dry-run too, before any compute is spent.
    anchor_payloads = _assert_readiness_anchors_reachable()
    print(f"[setup] readiness anchors reachable on {REFERENCE_591F_SOURCE}: "
          f"{[a['anchor_name'] for a in anchor_payloads]}", flush=True)
    seeds = [SEEDS[0], SEEDS[3]] if dry_run else SEEDS   # 42 (explorer) + 45 (false advancer)
    n_episodes = 3 if dry_run else N_EPISODES
    steps_per_episode = 25 if dry_run else STEPS_PER_EPISODE

    arm_results: List[Dict[str, Any]] = []
    for arm in ARMS:
        for seed in seeds:
            arm_results.append(_run_cell(arm=arm, seed=seed, n_episodes=n_episodes,
                                         steps_per_episode=steps_per_episode))

    by_arm: Dict[str, Dict[int, Dict[str, Any]]] = {
        a: {r["seed"]: r for r in arm_results if r["arm_id"] == a} for a in ARMS
    }
    spike = by_arm[ARM_SPIKE]
    cross = by_arm[ARM_CROSSING]

    # --- P6: RAW TRAJECTORY DIVERGENCE (load-bearing, evaluated FIRST) -------------------
    per_seed_diverging_episodes: Dict[int, int] = {
        s: _count_diverging_episodes(spike[s]["per_episode_h_pos"], cross[s]["per_episode_h_pos"])
        for s in seeds
    }
    n_diverging_episodes_total = sum(per_seed_diverging_episodes.values())
    n_seeds_diverging = sum(1 for v in per_seed_diverging_episodes.values() if v > 0)
    raw_trajectories_diverge = n_diverging_episodes_total >= P6_MIN_DIVERGING_EPISODES

    # --- ORACLE labels: level-only, PRE-EP_MIN (uncontaminated by the manipulation) ------
    # Step 4.5 red-team F2: h_pos_mean_full_run on ARM_SPIKE is contaminated once
    # env_kwargs() is genuinely live -- ARM_SPIKE's own post-advance episodes run
    # under different (Phase-1) dynamics that depend on ARM_SPIKE's own gate
    # decision. h_pos_mean_pre_ep_min (episodes [0,100)) is guaranteed Phase-0-only
    # in every arm (PHASE_EP_MIN[1] gates advancement in _try_phase_0_to_1), so it
    # cannot be contaminated by which arm's gate later fires. See module docstring
    # THE ORACLE section for the floor-recalibration caveat this substitution
    # carries (P3/P4 below self-route on a poorly-split floor rather than passing
    # one through silently).
    oracle: Dict[int, bool] = {
        s: _is_oracle_genuine({"h_pos_mean": spike[s]["h_pos_mean_pre_ep_min"]}) for s in seeds
    }
    n_oracle_genuine = sum(1 for v in oracle.values() if v)
    n_oracle_non_genuine = sum(1 for v in oracle.values() if not v)
    live_false_advancers = [
        s_ for s_ in seeds if (not oracle[s_]) and spike[s_]["reached_phase1"]]
    n_live_false_advancers = len(live_false_advancers)

    # --- READINESS PRECONDITIONS (measured on ARM_SPIKE, the positive control) -----------
    n_spike_advancers = sum(1 for s in seeds if spike[s]["reached_phase1"])
    # P2 CHANGED FROM 591h (Step 4.5 red-team F2): read ARM_CROSSING's own LIVE
    # scheduler_crossing_count (the exact statistic the crossing-count gate itself
    # computes and routes on) directly, rather than a post-hoc recount of
    # ARM_SPIKE's h_pos values -- which is exposed to the same contamination the
    # oracle was (ARM_SPIKE's post-advance dynamics are now also live).
    max_crossings = max((cross[s]["scheduler_crossing_count"] for s in seeds), default=0)
    worst_crossing_seed = max(seeds, key=lambda s: cross[s]["scheduler_crossing_count"]) \
        if seeds else None

    preconditions = [
        {
            "name": "raw_trajectory_divergence_present",
            "description": (
                "COUNT of episodes across ALL seeds where ARM_SPIKE and ARM_CROSSING's "
                "RAW per-episode h_pos sequences differ by more than float noise. This "
                "is the LOAD-BEARING non-vacuity guard the V3-EXQ-591h autopsy named "
                "(governance_2026_09_03): a 591h-style run whose two arms' behavioural "
                "trajectories are bit-identical -- even if their ADVANCE DECISIONS differ "
                "-- means the manipulation never reached behaviour, and this run self-"
                "routes to substrate_not_ready_requeue rather than any substrate-verdict "
                "label. Below floor -> the redesigned live channels (env_kwargs / "
                "offline_integration_frequency) still failed to close the loop."),
            "measured": n_diverging_episodes_total,
            "threshold": P6_MIN_DIVERGING_EPISODES,
            "direction": "lower",
            "control": "pointwise comparison of the two arms' own per_episode_h_pos, all seeds",
            "met": bool(raw_trajectories_diverge),
        },
        {
            "name": "spike_arm_reproduces_advance",
            "description": (
                "COUNT of ARM_SPIKE seeds reaching Phase 1. Below floor means the "
                "lineage basis did not reproduce, so nothing here is a verdict."),
            "measured": n_spike_advancers,
            "threshold": P1_MIN_SPIKE_ADVANCERS,
            "direction": "lower",
            "control": "ARM_SPIKE = the legacy single-episode spike gate (591c reproduction)",
            "met": bool(n_spike_advancers >= P1_MIN_SPIKE_ADVANCERS),
        },
        {
            "name": "crossing_counts_reach_gate_minimum",
            "description": (
                "MAX over seeds of ARM_CROSSING's own LIVE scheduler_crossing_count "
                "-- the SAME COUNT statistic the crossing-count gate itself computes "
                "and routes on (phase_summary()['phase01_crossing_count']), read "
                "directly rather than post-hoc-recomputed from a different arm's "
                "trajectory (CHANGED from 591h/the first draft: Step 4.5 red-team F2 -- "
                "a post-hoc recount off ARM_SPIKE's h_pos values is contaminated once "
                "ARM_SPIKE's own post-advance dynamics are also live). Below floor, "
                "the gate could admit NO seed under any outcome, so "
                "C_live_gate_discriminates would be STARVED rather than falsified."),
            "measured": max_crossings,
            "threshold": P2_MIN_MAX_CROSSINGS,
            "direction": "lower",
            "control": f"best-crossing seed in ARM_CROSSING (seed {worst_crossing_seed})",
            "offending_cell": f"seed {worst_crossing_seed}",
            "met": bool(max_crossings >= P2_MIN_MAX_CROSSINGS),
        },
        {
            "name": "oracle_genuine_explorers_present",
            "description": (
                "COUNT of seeds the level oracle (full-run h_pos_mean on ARM_SPIKE) "
                "labels genuine. Below floor -> no positive class to admit."),
            "measured": n_oracle_genuine,
            "threshold": P3_MIN_ORACLE_GENUINE,
            "direction": "lower",
            "control": "full-run h_pos_mean measured on ARM_SPIKE (the control arm)",
            "met": bool(n_oracle_genuine >= P3_MIN_ORACLE_GENUINE),
        },
        {
            "name": "oracle_non_explorers_present",
            "description": (
                "COUNT of seeds the oracle labels NON-genuine. Below floor -> no "
                "negative class to reject, so 'discriminates' is untestable on this "
                "seed draw."),
            "measured": n_oracle_non_genuine,
            "threshold": P4_MIN_ORACLE_NON_GENUINE,
            "direction": "lower",
            "control": "full-run h_pos_mean measured on ARM_SPIKE (the control arm)",
            "met": bool(n_oracle_non_genuine >= P4_MIN_ORACLE_NON_GENUINE),
        },
        {
            "name": "live_false_advancer_present",
            "description": (
                "COUNT of seeds that are oracle-non-genuine AND were admitted by the "
                "legacy spike gate in ARM_SPIKE -- i.e. a live false advancer for the "
                "crossing gate to REJECT. Below floor, the reject leg is never "
                "challenged."),
            "measured": n_live_false_advancers,
            "threshold": P5_MIN_LIVE_FALSE_ADVANCERS,
            "direction": "lower",
            "control": "ARM_SPIKE advance decisions x the oracle label (591f: seed 45)",
            "offending_cell": f"live false advancers: {live_false_advancers}",
            "met": bool(n_live_false_advancers >= P5_MIN_LIVE_FALSE_ADVANCERS),
        },
    ]
    gate_green = all(p["met"] for p in preconditions)

    # --- CRITERIA -------------------------------------------------------------------------
    # F4 GUARD (unchanged from 591h): a missing-telemetry episode advance bypasses
    # the crossing count; detect and refuse to score the criterion if it fired.
    dropout_bypassed = [
        s_ for s_ in seeds
        if cross[s_]["reached_phase1"]
        and cross[s_]["scheduler_crossing_count"] < PHASE_01_CROSSING_COUNT_MIN
    ]
    n_missing_telemetry_episodes = {
        str(s_): sum(1 for h in cross[s_]["per_episode_h_pos"] if h < 0.0) for s_ in seeds
    }

    per_seed_agreement = {
        s: bool(cross[s]["reached_phase1"] == oracle[s]) for s in seeds
    }
    c_discriminates = bool(
        gate_green and not dropout_bypassed and all(per_seed_agreement.values()))

    # Kept as a diagnostic-only comparison (NOT gating -- this is exactly the
    # 591h statistic that was insufficient on its own; P6 above is the
    # load-bearing guard now).
    arms_differ_decision_level = any(
        (spike[s]["reached_phase1"] != cross[s]["reached_phase1"])
        or (spike[s]["phase_01_at"] != cross[s]["phase_01_at"])
        for s in seeds
    )

    criteria = [
        {"name": "C_live_gate_discriminates", "load_bearing": True, "passed": c_discriminates},
        {"name": "C_arms_differ_decision_level", "load_bearing": False,
         "passed": bool(arms_differ_decision_level)},
        {"name": "C_raw_trajectories_diverge", "load_bearing": True,
         "passed": bool(raw_trajectories_diverge)},
    ]
    criteria_non_degenerate = {
        "C_live_gate_discriminates": bool(
            raw_trajectories_diverge and gate_green and n_oracle_genuine >= 1
            and n_oracle_non_genuine >= 1 and not dropout_bypassed),
        "C_arms_differ_decision_level": bool(gate_green),
        "C_raw_trajectories_diverge": True,   # always a genuine measurement, not vacuous
    }

    # --- LABEL, in the mandated order: P6 evaluated FIRST -----------------------
    if not raw_trajectories_diverge:
        label = "substrate_not_ready_requeue"
    elif not gate_green:
        label = "substrate_not_ready_requeue"
    elif dropout_bypassed:
        label = "substrate_not_ready_requeue"
    elif c_discriminates:
        label = "crossing_count_gate_discriminates_live_closed_loop"
    else:
        disagreeing = [s for s in seeds if not per_seed_agreement[s]]
        held_back = [s for s in disagreeing if oracle[s] and not cross[s]["reached_phase1"]]
        label = ("crossing_count_gate_self_defeating_holds_back_genuine_explorers"
                 if held_back else
                 "crossing_count_gate_discrimination_lost_in_closed_loop")

    outcome = "PASS" if (raw_trajectories_diverge and c_discriminates) else "FAIL"

    manifest: Dict[str, Any] = {
        "run_id": f"{EXPERIMENT_TYPE}_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_v3",
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "redesign_of": "V3-EXQ-591h",
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
        "outcome": outcome,
        "status": outcome,
        "claim_ids": CLAIM_IDS,
        "claim_ids_tested": CLAIM_IDS,
        "evidence_class": "simulation",
        "evidence_direction": EVIDENCE_DIRECTION,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "sleep_driver_pattern": "K=never",
        "dry_run": bool(dry_run),
        "arm_results": arm_results,
        "oracle_labels": {str(s): oracle[s] for s in seeds},
        "per_seed_agreement": {str(s): per_seed_agreement[s] for s in seeds},
        "per_seed_diverging_episodes": {str(s): v for s, v in per_seed_diverging_episodes.items()},
        "combination_rule": (
            "C_live_gate_discriminates = raw_trajectories_diverge AND readiness_gate_green "
            "AND for-every-seed(ARM_CROSSING.reached_phase1 == oracle_genuine_explorer) "
            "AND NOT dropout_bypassed. Plain AND. raw_trajectories_diverge is evaluated "
            "FIRST and gates everything else; C_arms_differ_decision_level is diagnostic "
            "only (the 591h-insufficient statistic, retained for comparison)."),
        "interpretation": {
            "label": label,
            "preconditions": preconditions,
            "criteria_non_degenerate": criteria_non_degenerate,
            "criteria": criteria,
            "readiness_anchors": anchor_payloads,
        },
        "metrics": {
            "n_spike_advancers": n_spike_advancers,
            "n_crossing_advancers": sum(1 for s in seeds if cross[s]["reached_phase1"]),
            "max_post_ep_min_crossings_spike": max_crossings,
            "n_oracle_genuine": n_oracle_genuine,
            "n_oracle_non_genuine": n_oracle_non_genuine,
            "n_seeds_agreeing": sum(1 for v in per_seed_agreement.values() if v),
            "arms_differ_decision_level": bool(arms_differ_decision_level),
            "n_diverging_episodes_total": n_diverging_episodes_total,
            "n_seeds_with_diverging_trajectory": n_seeds_diverging,
            "readiness_gate_green": bool(gate_green),
            "n_live_false_advancers": n_live_false_advancers,
            "dropout_bypassed_seeds": dropout_bypassed,
            "n_missing_telemetry_episodes_per_seed": n_missing_telemetry_episodes,
            "n_harm_gradient_ticks_by_cell": {
                f"{a}/{s}": by_arm[a][s]["n_harm_gradient_ticks"]
                for a in ARMS for s in seeds
            },
            "n_phase1_episodes_by_cell": {
                f"{a}/{s}": by_arm[a][s]["n_phase1_episodes"]
                for a in ARMS for s in seeds
            },
        },
        "informational_only": {"ORACLE_WINDOW_END": ORACLE_WINDOW_END},
        "registered_thresholds": {
            "PHASE_01_THRESHOLD": PHASE_01_THRESHOLD,
            "PHASE_01_CROSSING_COUNT_MIN": PHASE_01_CROSSING_COUNT_MIN,
            "ORACLE_H_POS_MEAN_FLOOR": ORACLE_H_POS_MEAN_FLOOR,
            "P1_MIN_SPIKE_ADVANCERS": P1_MIN_SPIKE_ADVANCERS,
            "P2_MIN_MAX_CROSSINGS": P2_MIN_MAX_CROSSINGS,
            "P3_MIN_ORACLE_GENUINE": P3_MIN_ORACLE_GENUINE,
            "P4_MIN_ORACLE_NON_GENUINE": P4_MIN_ORACLE_NON_GENUINE,
            "P5_MIN_LIVE_FALSE_ADVANCERS": P5_MIN_LIVE_FALSE_ADVANCERS,
            "P6_MIN_DIVERGING_EPISODES": P6_MIN_DIVERGING_EPISODES,
            "P6_DIVERGENCE_EPSILON": P6_DIVERGENCE_EPSILON,
        },
        "summary": {
            "scenario": (
                "Live-channel-redesigned closed-loop validation of the V3-EXQ-591f "
                "crossing-count Phase 0->1 curriculum gate: crossing-count gate "
                "(ARM_CROSSING) vs legacy spike gate (ARM_SPIKE), matched seeds, "
                "diversity-armed 591c build, phase reaching behaviour via "
                "env_kwargs() (the ONE live channel; offline_integration_frequency "
                "is deliberately not wired, see substrate_defect_found) instead of "
                "the dead novelty_bonus_weight channel 591b-591h wired."),
            "interpretation": label,
            "pairwise_deltas": {
                str(s): {
                    "spike_phase_01_at": spike[s]["phase_01_at"],
                    "crossing_phase_01_at": cross[s]["phase_01_at"],
                    "spike_final_phase": spike[s]["final_phase"],
                    "crossing_final_phase": cross[s]["final_phase"],
                    "spike_post_ep_min_crossings": spike[s]["post_ep_min_crossings"],
                    "crossing_post_ep_min_crossings": cross[s]["post_ep_min_crossings"],
                    "oracle_genuine": oracle[s],
                    "agrees_with_oracle": per_seed_agreement[s],
                    "n_diverging_episodes": per_seed_diverging_episodes[s],
                } for s in seeds
            },
        },
        "verdict_routing_note": (
            "outcome=FAIL is recorded both when readiness/trajectory-divergence failed "
            "and when the gate was genuinely refuted; the label alone distinguishes "
            "them and NOTHING in the runner or coordinator reads the label string. The "
            "machine-readable distinction lives in interpretation.preconditions[] "
            "instead: every entry carries numeric measured+threshold+direction, so "
            "build_experiment_indexes._precondition_unmet recomputes them and emits "
            "adjudication=precondition_unmet when raw_trajectory_divergence_present is "
            "unmet, which generate_pending_review.py surfaces under 'Diagnostic "
            "adjudication required'."),
        "scope_limitation": (
            "config.e3.novelty_bonus_weight is NOT written (deliberately, unlike "
            "591b-591h -- see module docstring). residue_scale_factor is NOT applied "
            "(out of the scope governance_2026_09_03 specified). "
            "offline_integration_frequency is NOT applied AT ALL (narrowed from the "
            "original two-channel governance instruction after Step 4.5 red-team: "
            "NEITHER agent.offline_integration()'s E1-replay half [confirmed-broken, "
            "substrate_defect_found[0]] NOR its 'safe' residue-only half "
            "[confirmed-inert, substrate_defect_found[1]] is invoked). env_kwargs() "
            "is the ONE live channel this run tests."),
        "substrate_defect_found": [
            {
                "location": "ree_core/agent.py REEAgent.offline_integration() (~line 12053-12058)",
                "description": (
                    "torch.cat([s, w]) for s, w in zip(self._self_experience_buffer, "
                    "self._world_experience_buffer) uses the DEFAULT dim=0 instead of "
                    "dim=-1. Contrast the correct sibling usage 3 lines above the buffer-"
                    "append site (agent.py:5775-5777): "
                    "torch.cat([gated_for_e1.z_self, gated_for_e1.z_world], dim=-1). With "
                    "the common default self_dim==world_dim (both 32 under "
                    "REEConfig.from_dims with no override -- the build every script in "
                    "this lineage uses), dim=0 concatenation silently produces a [2, D] "
                    "tensor instead of the intended [1, 2D] feature vector; "
                    "integrate_experience()'s downstream torch.stack/.unsqueeze(0)/slicing "
                    "then feeds E1DeepPredictor.context_memory.read() a wrongly-shaped "
                    "batch and it crashes (RuntimeError: mat1 and mat2 shapes cannot be "
                    "multiplied) once len(self._world_experience_buffer) > 10 -- "
                    "deterministically, within the first ~11 E1 ticks of any real run. "
                    "Confirmed via a direct 15-step REEAgent probe on 2026-09-03. "
                    "grep-confirmed: zero scripts in the whole ree-v3/experiments/ corpus "
                    "call agent.offline_integration() or agent.should_integrate() before "
                    "this file -- the bug has never been exercised or caught."
                ),
                "workaround_considered_and_rejected": (
                    "An earlier draft of this script called agent.should_integrate() "
                    "(pure arithmetic, not affected by this bug) and, on a True tick, "
                    "invoked only agent.residue_field.integrate() -- the E1-replay-free "
                    "half of offline_integration(). Step 4.5 red-team (fable) found that "
                    "half is ITSELF inert (see substrate_defect_found[1]), so this "
                    "workaround was DROPPED, not kept -- see scope_limitation above."
                ),
                "recommended_fix": (
                    "ree_core/agent.py: change `torch.cat([s, w])` to "
                    "`torch.cat([s, w], dim=-1)` at the offline_integration() list "
                    "comprehension. Out of scope for this /queue-experiment session "
                    "(ree_core edits are not authorised here); route via /implement-substrate."
                ),
            },
            {
                "location": "ree_core/residue/field.py ResidueField.integrate() (lines 1120-1141)",
                "description": (
                    "integrate() computes predictions = self.neural_field(sample_points) "
                    "and loss = F.mse_loss(predictions, targets), then accumulates "
                    "loss.item() -- there is no .backward() call, no optimizer, no "
                    ".step(), and no in-place tensor write anywhere in this method or in "
                    "RBFLayer.forward / the neural_field nn.Sequential it calls. "
                    "grep-confirmed: `neural_field` is referenced NOWHERE else in the "
                    "ree_core/ tree outside field.py, so its parameters are trained by "
                    "NOTHING anywhere in the substrate. The method's only observable side "
                    "effect is that `torch.randn_like(harm_locations)` (10 draws per call, "
                    "one per num_steps) advances the GLOBAL torch RNG stream, which every "
                    "downstream stochastic draw (CEM candidate noise, E3 multinomial "
                    "sampling, replay) subsequently consumes differently. Found by Step 4.5 "
                    "red-team (fable) reviewing the first draft's workaround for "
                    "substrate_defect_found[0]; confirmed directly against source before "
                    "landing this version."
                ),
                "why_this_matters_for_this_experiment": (
                    "Had this half been wired as originally drafted, "
                    "offline_integration_frequency (10 on Phase 0, 20 on Phase 1) would "
                    "have reached behaviour ONLY via RNG-stream perturbation on a schedule "
                    "correlated with the curriculum phase -- a genuine but SPURIOUS source "
                    "of per_episode_h_pos divergence between arms, unrelated to the "
                    "curriculum's actual env_kwargs/gate-decision content, and one the P6 "
                    "raw_trajectory_divergence_present precondition CANNOT distinguish "
                    "from a real one (it only checks that trajectories differ, not why). "
                    "That would have been a subtler repeat of the exact 591h vacuity "
                    "failure this redesign exists to fix -- hence this channel is NOT "
                    "wired at all in this version."
                ),
                "recommended_fix": (
                    "ree_core/residue/field.py: integrate() needs an actual optimizer + "
                    "backward pass on neural_field's parameters (or the method should be "
                    "renamed/documented as a pure diagnostic readout, not 'integration'). "
                    "Out of scope for this /queue-experiment session; route via "
                    "/implement-substrate."
                ),
            },
        ],
    }

    out_path = write_flat_manifest(
        manifest,
        dry_run=dry_run,
        config={
            "agent": _agent_config_kwargs(),
            "env": _config_slice(ARM_SPIKE)["env"],
            "schedule": {"n_episodes": n_episodes, "steps_per_episode": steps_per_episode},
            "arms": {a: _config_slice(a)["gate"] for a in ARMS},
            "live_channels": _config_slice(ARM_SPIKE)["live_channels"],
        },
        seeds=seeds,
        script_path=Path(__file__),
        started_at=t0,
        z_goal_stream_stats=_ZG.stats(),
    )
    return {"outcome": outcome, "manifest_path": out_path, "label": label}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    result = run_experiment(dry_run=args.dry_run)
    out_path = result["manifest_path"]
    print(f"outcome={result['outcome']} label={result['label']}")
    print(f"manifest: {out_path}")

    _outcome_raw = str(result["outcome"]).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
        dry_run=args.dry_run,
    )
