"""
V3-EXQ-932 -- z_goal / residue_wanting -> behaviour observational coupling,
re-instrumented on the V3-EXQ-916a-repaired substrate.

Claims: None (diagnostic / observational; promotes nothing, weights no governance)

EXPERIMENT_PURPOSE = "diagnostic"

WHAT THIS IS (and is NOT). This is a NEW EXQ NUMBER, not a lettered iteration,
because it asks a scientific question no prior run has answered: does REE's
internal goal/wanting signal PREDICT what the organism does next? It combines
two prior pieces that were never brought together:

  1. THE INSTRUMENT -- V3-EXQ-906c (v3_exq_906c_full_stack_observational_fishtank)
     built the observational-coupling machinery: per-step lagged correlations of
     an affect channel[t] against a subsequent behaviour readout[t+1..t+3]
     (`_pearson_r` / `_lagged_pairs` / `_compute_coupling_metrics`). It computed
     `coupling_zgoal_t_to_approach_t1` and `coupling_zgoal_t_to_benefit_t1t3`
     (weak/near-null, r~0.07, n=3785) but NEVER a residue_wanting coupling -- and
     at 906c's authoring time residue_wanting was hard-zero (orphaned writer, see
     below), so even the z_goal couplings were measured on a substrate where the
     benefit/wanting write path was structurally dead.

  2. THE REPAIRED SUBSTRATE -- V3-EXQ-916a fixed that orphaned writer (three
     driver-level changes: tonic_5ht_enabled=True; call update_benefit_salience()
     in the step loops; use_proxy_fields=True + read benefit_exposure from `info`,
     not `obs_dict`). Its landed run confirmed residue_wanting genuinely VARIES
     (chan_max_std_residue_wanting=0.568, chan_nondegen=True) and z_goal too
     (chan_max_std_z_goal=0.078). But 916a is a channel-non-degeneracy telemetry
     showcase -- it records per-step affect but computes NO coupling_* fields at
     all (checked directly).

This driver = 916a's substrate VERBATIM (every ENV_KWARG, config flag, warmup +
eval step loop, writer call, _read_affect, _classify_mode, reef stack are copied
byte-for-byte from 916a and left unchanged -- this is instrumentation added ON
TOP of a proven substrate, never a substrate change) + 906c's coupling instrument
ported onto it, PLUS the residue_wanting -> behaviour couplings that have never
been exercised on live data. It answers organism-review Section 3 Level A
("observational coupling") for residue_wanting for the FIRST time, and re-checks
z_goal now that a matched-substrate comparison against 906c's near-null is
possible.

OBSERVATIONAL, NOT CAUSAL. A lagged correlation between an internal signal and a
subsequent behaviour is consistent with, but does not establish, that the signal
DRIVES the behaviour (common-cause confounds abound -- e.g. proximity to a
resource raises both wanting and approach). The causal-necessity question -- does
zeroing the one channel wanting actually feeds into behavioural choice (the CEM
VALENCE_WANTING scoring weight) change what the organism does -- is the SEPARATE
ablation V3-EXQ-931 (chip-20260812-cem-wanting-weight-causal-ablation), which
this driver deliberately does NOT duplicate. Do not read any coupling here as
causal.

PRE-REGISTERED ANALYSIS (constants below, defined before any run):
  - DV/readout for each coupling is fixed in `_compute_coupling_metrics`.
  - Both lagged PEARSON r (906c's statistic, for direct comparability) AND
    SPEARMAN rho (monotone-robust; a binary behaviour readout vs a continuous
    affect channel is a point-biserial-ish shape where a monotone-but-nonlinear
    association would be understated by Pearson) are reported per coupling.
  - NON-TRIVIAL COUPLING FLOOR: |r| (or |rho|) >= COUPLING_FLOOR_R (0.15) with
    n >= MIN_COUPLING_N (200) pairs. 906c's own near-null r~0.07 is the natural
    "no coupling" reference; 0.15 is ~2x that noise floor. Below it reads as
    "no / near-null observational coupling", NOT as evidence of any effect.
  - NON-DEGENERACY (readiness): residue_wanting MUST actually vary in this run
    (chan_max_std_residue_wanting > STD_FLOOR, the way 916a validated the fix) or
    the wanting couplings are unmeasurable and the run self-routes
    substrate_not_ready_requeue rather than reporting a spurious ~0 coupling.

Provenance of the substrate fix, verbatim from V3-EXQ-916a's own docstring
(reproduced so this file is self-contained about WHY the writer path is live):

BUG FIX vs V3-EXQ-916 (same scientific question, broken instrumentation --
lettered iteration per CLAUDE.md EXQ versioning policy, not a new number).
`failure_autopsy_906b-906c-911-cluster_2026-08-10.md` Finding 2 traced
`residue_wanting` (VALENCE_WANTING, index 0, read correctly by `_read_affect`
below via `evaluate_valence`) reading exact 0.0 across the whole 906-family
driver lineage to an ORPHANED WRITER: `REEAgent.update_benefit_salience()` and
`REEAgent.update_schema_wanting()` (`ree_core/agent.py`) are never called from
inside this driver family's step loop, even though the write path exists and
is exercised correctly by ~20 other experiment scripts (e.g.
`v3_exq_263_mech216_e1_predictive_wanting.py`, `experiments/_harness.py`).
`SD-RESIDUE-VALENCE-BOUND`'s `implementation_note` (2026-08-11) explicitly
descoped this half as an experiment-script gap, not a `ree_core/` gap (no
agent-level step loop exists to wire it into), routing it to
`/queue-experiment`/`/diagnose-errors`. This script is that fix, applied to
916 (the newest fishtank-family driver at authoring time) rather than to an
already-landed/already-scored predecessor.

Three changes vs 916, all additive (916's own PASS gates -- CORE_CHANNELS =
z_harm_a/z_harm_un/drive, block_steps, freeze-not-locked -- are untouched and
do not read residue_wanting or benefit_exposure, so this is NOT a correction
to 916's scientific result; 916 is not superseded):
  1. `tonic_5ht_enabled=True` added to `REEConfig.from_dims(...)` (SD-014/
     MECH-203 serotonin master switch -- default False in 916 and every prior
     664-derived driver, which is WHY update_benefit_salience() no-op'd even on
     scripts that happened to call it). This is a genuine, intentional
     mechanism/behaviour change (MECH-203 valence-weighted replay in
     `agent._do_replay` also activates once this is on) -- exactly why this is
     a new lettered iteration rather than a silent patch to 916.
  2. `agent.update_benefit_salience(benefit_exposure)` called right after the
     existing `agent.update_z_goal(...)` call in both the warmup and eval step
     loops (canonical call site per `_harness.py` / `v3_exq_263`), plus a
     `agent.update_schema_wanting(drive_level=...)` call guarded on
     `config.e1.schema_wanting_enabled` (left at its default False here --
     `_harness.py`'s own "default-off guard preserves the historical no-op
     path" idiom -- so the MECH-216 channel is wired but inert; only the SR-2
     benefit-salience path is actually activated by change 1 above).
  3. SECOND, INDEPENDENT DEFECT FOUND DURING THIS SCRIPT'S OWN CODE-REVIEW /
     SMOKE-TEST PASS (not in the originating failure autopsy): 916's existing
     `benefit_exposure = obs_dict.get("benefit_exposure", 0.0)` line (unchanged
     since 664) reads the WRONG dict -- `CausalGridWorldV2.step()` places
     `benefit_exposure` into `info` (5th env.step() return value), never into
     `obs_dict`. Confirmed directly (probe, not just source-reading):
     `obs_dict.keys()` over a real 916a-config run never contains
     `"benefit_exposure"`. WORSE: even `info["benefit_exposure"]` (and the
     matching `body_state[10:11]` EMA channels) are only populated when
     `CausalGridWorldV2(use_proxy_fields=True)` -- False (the default) in 916
     and therefore in every 664-derived driver, so `benefit_exposure` has been
     structurally 0.0 in this ENTIRE lineage regardless of dict source. This
     ALSO means `agent.update_z_goal(benefit_exposure=...)` -- unchanged,
     pre-existing code, called by 916 and every predecessor -- has never
     received a real value either; the "z_goal active_frac=0.000 -- no writer
     defect" smoke reading this lineage's scripts have always shown is
     therefore NOT distinguishable, from telemetry alone, from this deeper
     defect (both read as "no resource contact"). Fixed here, IN THIS SCRIPT
     ONLY, by (a) `use_proxy_fields=True` in `ENV_KWARGS` (the standard,
     ~20-scripts-wide convention `v3_exq_263`/others already use) and
     (b) reading `info.get("benefit_exposure", 0.0)` instead of
     `obs_dict.get(...)` in both step loops (matching `v3_exq_263`'s own
     convention), with `info: Dict = {}` seeded right after each episode's
     `env.reset()` so the pre-first-`env.step()` read stays a safe 0.0 default
     rather than a `NameError`. NOT retrofitted into 664/906/909/911/912/913/
     916 themselves (already-landed/already-scored predecessors; see CLAUDE.md
     Concurrency Rules / EXQ Versioning -- do not retroactively edit and
     re-score a landed driver). Flagged as a distinct, lineage-wide follow-on
     for governance rather than silently absorbed into this fix's scope.
  `residue_wanting` is also added to the eval loop's `chan_vals` non-degeneracy
  accumulator (reported as `chan_max_std_residue_wanting` /
  `chan_nondegen_residue_wanting`, same convention as excite/dread) so a run
  of this script directly evidences the fix -- NOT load-bearing on PASS/FAIL,
  matching every other non-core channel this showcase reports.

Follow-on to V3-EXQ-664. The 2026-08-10 organism-level review of the V3-EXQ-906
Fishtank lineage (reef_ecology_strategy_affective_occupancy_review_2026-08-10.md,
Section 3a/3e + 2026-08-11 addendum) found that relief (MECH-302) and safety
(MECH-303 contextual / MECH-304 cue-specific) are real, implemented, and already
validated by their own dedicated experiments (V3-EXQ-517c, V3-EXQ-760, V3-EXQ-763)
-- but are never reachable in any Fishtank showcase driver because each sits
behind its own default-False REEConfig flag that 664 (and every 664-derived
driver) never sets. This driver is 664 + those three flags, so relief/safety
become directly observable in the fishtank_viz.html feed for the first time.

664's own defaults are UNCHANGED -- this is a new, separate script, not an edit
to 664's `_make_agent_and_env` (that would silently change 664's own numeric
output for its existing consumers).

New flags vs. 664 (all default False in REEConfig, off in every prior fishtank
driver):
  use_suffering_derivative_comparator=True   MECH-302 relief: non-trainable rolling-
                                              window descent detector on z_harm_a
                                              norm. Fires `_relief_completion_event`,
                                              which reuses the MECH-057a commitment-
                                              release + MECH-094 VALENCE_LIKING tag
                                              write pipeline (agent.py ~5105-5112,
                                              ~6286-6310).
  use_conditioned_safety_store=True          MECH-304 cue-specific safety: EMA
                                              prototype of z_world at MECH-302 event
                                              ticks; cosine similarity to current
                                              z_world -> `_conditioned_safety_signal`
                                              -> commitment release when above
                                              threshold (agent.py ~5116-5126,
                                              ~6312-6341).
  use_contextual_safety_terrain=True         MECH-303 contextual safety: passive RBF
                                              terrain accumulated on quiescent
                                              (low-harm) ticks; `evaluate_safety`
                                              queried each step for telemetry, and
                                              gates a second, independent commitment-
                                              release path when the accumulated
                                              terrain crosses its own release
                                              threshold (agent.py ~5128-5145,
                                              ~6343-6367; ree_core/residue/field.py
                                              `accumulate_safety`/`evaluate_safety`).

TIMING NOTE (load-bearing correctness, not a style choice): `_relief_completion_event`
and `_conditioned_safety_signal` are written by `agent.sense()` but CONSUMED/CLEARED
by `agent.select_action()` -- `_conditioned_safety_signal` unconditionally, every
tick; `_relief_completion_event` only on the tick it fires. Reading either AFTER
select_action() (the pattern 664's `_read_affect` uses for every other channel)
would therefore observe them always-zero. Both are captured in the eval step loop
immediately after `agent.sense()` and BEFORE `agent.select_action()`, then threaded
into `_read_affect` as extra arguments. `evaluate_safety(z_world)` (MECH-303) is a
stateless RBF-terrain query, not a consume-once flag, so it stays inside
`_read_affect` at its usual post-select_action call site.

KNOWN LIMITATION (SD-RESIDUE-VALENCE-BOUND, degrading, substrate_queue.json):
both MECH-302 relief and MECH-304 safety release write into
`residue_field.update_valence(component=VALENCE_LIKING, ...)` -- the same
unbounded accumulator the V3-EXQ-906 lineage found growing 40-110x under
sustained multi-segment exposure with no decay/clamp. This driver is the first
config where that accumulator is fed by MECH-302/304 specifically (664 and its
siblings never fire this path). Not a reason to block this diagnostic (severity
is `degrading`, not `corrupting`, and the showcase's own PASS criteria are not
load-bearing on `liking`'s absolute magnitude) -- noted here so a future reader
of `liking` growth in this run's episode_log does not mistake it for a novel
finding specific to relief/safety.

KNOWN LIMITATION (MECH-303 default threshold unreachable, not a bug): this driver
leaves `contextual_safety_harm_threshold` at its REEConfig default (0.05) and, in
a 4000-step diagnostic run of this driver, `agent.residue_field.total_safety` and
`num_safety_steps` were 0.0 / 0 for the ENTIRE run -- `accumulate_safety()` was
never invoked. This is expected, not a wiring defect: the default 0.05 sits ~11x
below the affective-harm encoder's actual z_harm_a norm baseline (~0.547 safe /
~0.542 unsafe per V3-EXQ-764's calibration measurement, itself an SD-011
limitation -- the encoder barely discriminates hazard density at all), so
`z_harm_a.norm() < 0.05` essentially never holds. Every other experiment that has
exercised this live gate has had to override the threshold explicitly
(V3-EXQ-520's readiness diagnostic used 999 to force accumulation; V3-EXQ-764
calibrated per-seed to 0.55, and even then noted that value barely discriminates
safe from unsafe). MECH-303 itself is real and separately validated at the
representation level by V3-EXQ-760 (which bypasses this gate entirely, calling
`accumulate_safety` directly against ground-truth hazard proximity rather than
the live agent's z_harm_a norm) -- this limitation is about the DEFAULT config
value's reachability in a live agent-path driver, not about the substrate. See
`REE_assembly/evidence/planning/mech303_contextual_safety_threshold_reachability.md`
for the full investigation. A future reader of a flat `safety_terrain_read`
channel in this run's episode_log should not mistake it for a bug in this driver
or in the MECH-303 wiring.

Substrates active (664's affective stack + the new relief/safety register):
  SD-007  reafference                            (perspective correction)
  SD-008  alpha_world=0.9                        (encoder correction)
  SD-010  use_harm_stream=True                   (sensory z_harm_s)
  SD-011  use_affective_harm_stream=True         (affective z_harm_a, suffering)
  SD-019a use_harm_un=True                        (z_harm_un, unpleasantness -- middle tier)
  SD-018  use_resource_proximity_head=True       (resource prox supervision)
  SD-012  z_goal_enabled, drive_weight=2.0       (homeostatic drive + wanting)
  SD-021  harm_descending_mod_enabled=True       (descending modulation)
  SD-054  reef_enabled, hazard_food_attraction=0.7  (reef substrate)
  MECH-090 beta_gate_bistable=True               (commitment latch)
  MECH-320 use_tonic_vigor=True                  (surplus drive-to-act / vigor)
  SD-037  use_broadcast_override=True            (orexin arousal recruitment)
  MECH-279 use_pag_freeze_gate=True (capped)     (committed freeze behaviour)
  MECH-353 use_blocked_agency=True               (blocked-agency / frustration assert pole)
  MECH-307 use_mech307_split_surprise=True + surprise_gated_replay
                                                 (excitement / dread valence channels)
  MECH-302 use_suffering_derivative_comparator=True  (relief-completion event -- NEW)
  MECH-304 use_conditioned_safety_store=True         (cue-specific safety -- NEW)
  MECH-303 use_contextual_safety_terrain=True        (contextual safety terrain -- NEW)
  control_vector_logging                          (reads tonic-vigor v_t)

Environment adds scheduled action blocks (sparse) so the blocked-agency / assert
pole (z_block) actually rises -- the fish tries to move, is externally blocked,
frustration accumulates. Same env config as 664.

Per-step episode_log schema (ADDITIVE to the 664 schema -- the viz reads both):
  relief_event         (0.0/1.0 -- MECH-302 fired this tick)
  safety_cue_signal    (MECH-304 cosine similarity, captured pre-select_action)
  safety_terrain_read  (MECH-303 evaluate_safety() at current z_world)

This is a TELEMETRY SHOWCASE, not a claim test. Its self-reported outcome is a
non-degeneracy check on the CORE affect channels inherited from 664 (a showcase
that emits flat core channels is the failure mode). relief_event / safety_cue_signal
/ safety_terrain_read are reported and their non-degeneracy is recorded in metrics,
but -- like 664's excite/dread/override -- are NOT load-bearing on PASS/FAIL: MECH-302
requires a genuine sustained z_harm_a descent within a 5-tick window and MECH-304
requires a prior MECH-302 firing to seed its EMA prototype, so within a bounded
eval window either can legitimately be sparse even when correctly wired. It carries
NO claim_ids and does NOT weight governance.

Output:
  evidence/experiments/v3_exq_916a_relief_safety_fishtank_showcase/
    v3_exq_916a_relief_safety_fishtank_showcase_<ts>.json          (manifest)
    v3_exq_916a_relief_safety_fishtank_showcase_<ts>_episode_log.json  (fishtank feed)

Estimated runtime: ~55 min on cloud CPU (3 seeds x 50 warmup + 5 eval x 200 steps,
12x12 grid, same as 664 -- the three new flags add negligible per-tick overhead:
a rolling-window comparator tick, an EMA cosine-similarity update, and one RBF
terrain accumulate/query per step).
"""

import sys
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig
from ree_core.residue.field import (
    VALENCE_WANTING, VALENCE_LIKING, VALENCE_SURPRISE,
    VALENCE_POSITIVE_SURPRISE, VALENCE_NEGATIVE_SURPRISE,
)
from experiment_protocol import emit_outcome
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator


EXPERIMENT_TYPE    = "v3_exq_932_zgoal_wanting_coupling_reinstrument"
EXPERIMENT_PURPOSE = "diagnostic"
CLAIM_IDS          = []

# The two readiness anchors (residue_wanting / z_goal non-degenerate) use the
# lineage-standard non-degeneracy predicate `chan_max_std > STD_FLOOR` (1e-4) --
# the SAME definition 664/916 use for every channel, not a hand-narrowed
# predicate. It is reachable by construction: V3-EXQ-916a's landed run measured
# std=0.568 (residue_wanting) and 0.078 (z_goal), both >> 1e-4. So the predicate
# IS the degeneracy definition and cannot be unmeetable -- the assert_anchor_
# reachable machinery would be redundant here (readiness_anchor.py's own EXEMPT
# case: "the predicate IS the degeneracy definition, reachable by construction").
ANCHOR_REACHABILITY_EXEMPT = (
    "Readiness anchors use the lineage-standard chan_max_std > STD_FLOOR (1e-4) "
    "non-degeneracy floor -- the same predicate 664/916 apply to every channel, "
    "not a hand-narrowed one. Reachable by construction: V3-EXQ-916a's landed run "
    "measured residue_wanting std=0.568 and z_goal std=0.078, both >> 1e-4."
)

# --- Pre-registered coupling analysis parameters (defined before any run) ---
# 906c reported a near-null r~0.07 for its z_goal couplings; that is the natural
# "no coupling" reference. A coupling counts as NON-TRIVIAL only if |r| or |rho|
# clears COUPLING_FLOOR_R (~2x 906c's noise floor) AND has at least MIN_COUPLING_N
# paired observations. Below the floor reads as "no / near-null observational
# coupling", never as evidence of an effect. These are diagnostic thresholds for
# a lagged ASSOCIATION; they say nothing about causation.
COUPLING_FLOOR_R = 0.15    # |r|/|rho| a coupling must exceed to be "non-trivial"
MIN_COUPLING_N   = 200     # min paired obs for a coupling to be interpretable
LAG_APPROACH_H   = 1       # affect[t] -> mode[t+1]=="approach"
LAG_BENEFIT_H    = 3       # affect[t] -> resource contact within t+1..t+3
LAG_MOVED_H      = 1       # affect[t] -> locomotion at t+1
LAG_REEFEXIT_H   = 1       # affect[t] -> departure-from-refuge at t+1

HARM_MODE_THRESH    = 0.25
EXPLORE_ERR_THRESH  = 0.10
SHELTER_HARM_THRESH = 0.15   # z_harm_norm floor for shelter mode while in reef
ASSERT_THRESH       = 0.10   # z_block_assert floor for the 'assert' (frustration) mode

ENV_KWARGS = dict(
    size=12,
    num_hazards=4,
    num_resources=5,
    hazard_harm=0.05,
    env_drift_interval=5,
    env_drift_prob=0.1,
    proximity_harm_scale=0.1,
    proximity_benefit_scale=0.05,
    proximity_approach_threshold=0.2,
    hazard_field_decay=0.5,
    resource_respawn_on_consume=True,
    toroidal=False,
    # SD-011 second source: rolling harm-history window
    harm_history_len=10,
    # SD-054: reef enrichment substrate (ARM_1_reef_food config from EXQ-522)
    reef_enabled=True,
    n_reef_patches=3,
    reef_patch_radius=2,
    hazard_food_attraction=0.7,
    # MECH-353 feed: sparse external action blocks so z_block (frustration) rises
    scheduled_action_block_enabled=True,
    scheduled_action_block_interval=10,
    scheduled_action_block_prob=0.4,
    # residue_wanting orphaned-writer fix (916a vs 916): CausalGridWorldV2 only
    # populates `info["benefit_exposure"]` / `info["harm_exposure"]` (and the
    # matching body_state[10:11] EMA channels) when use_proxy_fields=True --
    # False (the default) in 916 and every 664-derived driver, so
    # benefit_exposure was structurally 0.0 REGARDLESS of which dict it was
    # read from. See module docstring point 3.
    use_proxy_fields=True,
)

WARMUP_EPISODES   = 50
EVAL_EPISODES     = 5
STEPS_PER_EPISODE = 200
WORLD_DIM         = 32
SELF_DIM          = 32
HARM_DIM          = 32
HARM_A_DIM        = 16
HARM_HISTORY_LEN  = 10
PAG_MAX_FREEZE    = 8   # MECH-279 cap so the freeze gate never permanently locks

WF_BUF_MAX        = 2000
HARM_EVAL_BUF_MAX = 2000
BATCH_SIZE        = 32
LR_E1             = 1e-4
LR_E2_WF          = 3e-4
LR_E3_HARM        = 1e-3
LR_ENC_AUX        = 5e-4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _action_to_onehot(idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, idx] = 1.0
    return v


def _obs_harm(obs_dict) -> Optional[torch.Tensor]:
    return obs_dict.get("harm_obs")


def _obs_harm_a(obs_dict) -> Optional[torch.Tensor]:
    return obs_dict.get("harm_obs_a")


def _obs_harm_history(obs_dict) -> Optional[torch.Tensor]:
    return obs_dict.get("harm_history")


def _obs_accum(obs_dict) -> float:
    v = obs_dict.get("accumulated_harm")
    return float(v) if v is not None else 0.0


def _obs_resource_prox(obs_dict) -> float:
    rv = obs_dict.get("resource_field_view")
    if rv is None:
        return 0.0
    return float(rv.max().item()) if isinstance(rv, torch.Tensor) else float(np.max(rv))


def _norm(t: Optional[torch.Tensor]) -> Optional[float]:
    if t is None:
        return None
    return float(t.norm().item())


def _make_agent_and_env(seed: int) -> Tuple[REEAgent, CausalGridWorldV2]:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    env = CausalGridWorldV2(seed=seed, **ENV_KWARGS)
    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
        harm_dim=HARM_DIM,
        alpha_world=0.9,
        alpha_self=0.3,
        reafference_action_dim=env.action_dim,
        # SD-010 / SD-011
        use_harm_stream=True,
        z_harm_dim=HARM_DIM,
        use_affective_harm_stream=True,
        z_harm_a_dim=HARM_A_DIM,
        harm_history_len=HARM_HISTORY_LEN,
        # SD-018: resource-prox supervision on z_world
        use_resource_proximity_head=True,
        resource_proximity_weight=0.5,
        # SD-012: homeostatic drive-modulated benefit + goal system
        benefit_eval_enabled=True,
        benefit_weight=1.0,
        z_goal_enabled=True,
        goal_weight=0.5,
        drive_weight=2.0,
        e1_goal_conditioned=True,
        # --- affective register (telemetry, inherited from 664) ---
        use_tonic_vigor=True,             # MECH-320
        use_blocked_agency=True,          # MECH-353
        use_pag_freeze_gate=True,         # MECH-279
        # --- relief / safety register (this experiment's addition) ---
        use_suffering_derivative_comparator=True,   # MECH-302
        use_conditioned_safety_store=True,           # MECH-304
        use_contextual_safety_terrain=True,          # MECH-303
        # --- residue_wanting orphaned-writer fix (916a vs 916, THIS script) ---
        # SD-014/MECH-203 serotonin master switch. Without this,
        # update_benefit_salience() below is a no-op (self.serotonin.enabled
        # gate) regardless of whether it is called -- see module docstring.
        tonic_5ht_enabled=True,
    )
    # Non-from_dims config tweaks (set directly on the dataclass / sub-configs).
    config.e3.commitment_threshold = 0.5           # MECH-090 realistic threshold
    config.heartbeat.beta_gate_bistable = True     # MECH-090 bistable latch
    config.harm_descending_mod_enabled = True      # SD-021
    config.descending_attenuation_factor = 0.5

    # SD-019a: unpleasantness channel (middle harm tier).
    config.latent.use_harm_un = True

    # MECH-279: cap freeze duration so the gate never permanently locks the fish.
    config.pag_max_freeze_duration = PAG_MAX_FREEZE

    # MECH-320 tonic vigor reads its true substrate value (no artificial floor).
    # In this reward-negative reef env the avg-reward-rate EWMA stays negative, so
    # vigor sits low -- a faithful "no surplus drive-to-act under sustained threat"
    # reading. The channel is emitted for the viz but is not load-bearing.

    # SD-037: orexin broadcast-override (arousal recruitment under drive + threat).
    config.use_broadcast_override = True

    # MECH-307 + MECH-205: split-surprise valence channels for excite / dread.
    config.surprise_gated_replay = True
    config.use_mech307_split_surprise = True

    # control-vector telemetry so the tonic-vigor v_t is inspectable each tick.
    config.use_control_vector_logging = True

    # MECH-302/303/304 knobs left at REEConfig defaults (suffering_window_length=5,
    # suffering_drop_threshold=0.10, safety_store_threshold=0.5,
    # contextual_safety_release_threshold=1.0, ...) -- this run's purpose is to
    # confirm the DEFAULT operating point is reachable/observable, not to tune it.

    agent = REEAgent(config)
    return agent, env


# ---------------------------------------------------------------------------
# Affect readout
# ---------------------------------------------------------------------------

def _read_affect(
    agent: REEAgent, latent, obs_body,
    relief_event: bool = False, safety_cue_signal: float = 0.0,
) -> Dict:
    """Read the protoemotional register off the agent + latent after select_action.

    `relief_event` / `safety_cue_signal` are NOT read here -- they must be captured
    by the caller immediately after agent.sense() and before agent.select_action(),
    since select_action() consumes/clears both (see module docstring TIMING NOTE).
    They are threaded through as arguments purely so this function's return dict
    stays the single place step_rec is assembled from.
    """
    z_world = latent.z_world

    # Nociceptive cascade
    z_harm_s  = _norm(latent.z_harm)
    z_harm_un = _norm(getattr(latent, "z_harm_un", None))
    z_harm_a  = _norm(latent.z_harm_a)

    # Drive & wanting
    try:
        drive = float(REEAgent.compute_drive_level(obs_body))
    except Exception:
        drive = None
    z_goal = None
    if getattr(agent, "goal_state", None) is not None:
        try:
            z_goal = float(agent.goal_state.goal_norm())
        except Exception:
            z_goal = None

    # Tonic vigor (MECH-320) via control-vector telemetry
    vigor = None
    cv = getattr(agent, "_last_control_vector", None)
    if isinstance(cv, dict):
        shared = cv.get("shared", {})
        v = shared.get("tonic_vigor_v_t")
        if v is not None:
            vigor = float(v)

    # Orexin override (SD-037)
    override = None
    bo = getattr(agent, "broadcast_override", None)
    if bo is not None and getattr(bo, "override_signal", None) is not None:
        override = float(bo.override_signal)

    # PAG freeze (MECH-279)
    freeze = False
    pag_out = getattr(agent, "_pag_last_output", None)
    if pag_out is not None:
        freeze = bool(getattr(pag_out, "freeze_active", False))

    # Blocked-agency assert pole (MECH-353)
    z_block = 0.0
    ba = getattr(agent, "blocked_agency", None)
    if ba is not None and getattr(ba, "_last_output", None) is not None:
        z_block = float(getattr(ba._last_output, "z_block_assert", 0.0))
    elif getattr(latent, "z_block", None) is not None:
        z_block = float(latent.z_block.abs().mean().item())

    # MECH-307 anticipatory valence (excitement / dread) from residue, plus the
    # unsigned prediction-error magnitude (VALENCE_SURPRISE, index 3) both are split
    # from, and the residue-map wanting/liking channels (indices 0/1) -- all six
    # components are computed every step but were previously read no further than 4/5.
    # NOTE: `residue_wanting` (VALENCE_WANTING) is distinct from `z_goal` above --
    # z_goal is the frontal goal-attractor's own wanting signal (goal_state.goal_norm()),
    # not the hippocampal-map residue channel read here.
    excite, dread, surprise, residue_wanting, liking = None, None, None, None, None
    try:
        val = agent.residue_field.evaluate_valence(z_world)
        if val is not None and val.shape[-1] > VALENCE_NEGATIVE_SURPRISE:
            excite          = float(val[0, VALENCE_POSITIVE_SURPRISE].item())
            dread           = float(val[0, VALENCE_NEGATIVE_SURPRISE].item())
            surprise        = float(val[0, VALENCE_SURPRISE].item())
            residue_wanting = float(val[0, VALENCE_WANTING].item())
            liking          = float(val[0, VALENCE_LIKING].item())
    except Exception:
        excite, dread, surprise, residue_wanting, liking = None, None, None, None, None

    # MECH-303: contextual safety terrain read (stateless RBF query -- safe to
    # call at any point after sense(), unlike relief_event/safety_cue_signal above).
    safety_terrain_read = None
    try:
        rf = getattr(agent, "residue_field", None)
        if rf is not None and getattr(rf, "safety_terrain_enabled", False):
            safety_pred = rf.evaluate_safety(z_world.detach())
            safety_terrain_read = float(safety_pred.mean().item())
    except Exception:
        safety_terrain_read = None

    return {
        "z_harm_s":  z_harm_s,
        "z_harm_un": z_harm_un,
        "z_harm_a":  z_harm_a,
        "drive":     drive,
        "z_goal":    z_goal,
        "vigor":     vigor,
        "override":  override,
        "freeze":    freeze,
        "z_block":   z_block,
        "excite":    excite,
        "dread":     dread,
        "residue_wanting": residue_wanting,
        "liking":    liking,
        "surprise":  surprise,
        # MECH-302/303/304 (this experiment's addition)
        "relief_event":        1.0 if relief_event else 0.0,
        "safety_cue_signal":   float(safety_cue_signal),
        "safety_terrain_read": safety_terrain_read,
    }


def _classify_mode(
    z_harm_norm: float,
    world_change_norm: float,
    harm_signal: float,
    in_reef: bool,
    freeze: bool,
    z_block_assert: float,
) -> str:
    """Behavioural mode with affect precedence: freeze > assert > shelter > avoid > approach > explore > neutral."""
    if freeze:
        return "freeze"
    if z_block_assert is not None and z_block_assert > ASSERT_THRESH:
        return "assert"
    if in_reef and z_harm_norm > SHELTER_HARM_THRESH:
        return "shelter"
    if z_harm_norm > HARM_MODE_THRESH:
        return "avoid"
    if harm_signal > 0.01:
        return "approach"
    if world_change_norm > EXPLORE_ERR_THRESH:
        return "explore"
    return "neutral"


def _get_reef_cells(env: CausalGridWorldV2) -> List[List[int]]:
    raw: Set = getattr(env, "_reef_cells", set())
    return [[int(x), int(y)] for x, y in sorted(raw)]


# ---------------------------------------------------------------------------
# V3-EXQ-932 coupling-metrics helpers.
#
# `_pearson_r` / `_lagged_pairs` / `_contemporaneous_pairs` are ported VERBATIM
# from V3-EXQ-906c (v3_exq_906c_full_stack_observational_fishtank.py) so the
# z_goal re-check here is byte-for-byte the SAME instrument that produced 906c's
# r~0.07 -- the comparison across the substrate fix is otherwise confounded by a
# changed estimator. `_spearman_r` and the residue_wanting couplings are new.
# All degrade gracefully to (0.0, n) on degenerate input rather than raising --
# a diagnostic showcase must not crash the manifest write on a short/edge run.
# ---------------------------------------------------------------------------

def _pearson_r(xs: List[float], ys: List[float]) -> Tuple[float, int]:
    """Pooled Pearson r over paired (x, y). Degrades to (0.0, n) on n<2 or
    zero-variance input (ported verbatim from 906c)."""
    n = len(xs)
    if n < 2:
        return 0.0, n
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if float(np.std(x)) < 1e-12 or float(np.std(y)) < 1e-12:
        return 0.0, n
    r = float(np.corrcoef(x, y)[0, 1])
    return (0.0 if np.isnan(r) else r), n


def _spearman_r(xs: List[float], ys: List[float]) -> Tuple[float, int]:
    """Pooled Spearman rho = Pearson r on rank-transformed inputs (average ranks
    for ties, matching scipy's default). Monotone-robust; reported alongside
    Pearson because a binary behaviour readout vs a continuous affect channel is
    a point-biserial-ish shape where a monotone-but-nonlinear association is
    understated by Pearson. No scipy dependency -- ranks via numpy argsort."""
    n = len(xs)
    if n < 2:
        return 0.0, n
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if float(np.std(x)) < 1e-12 or float(np.std(y)) < 1e-12:
        return 0.0, n

    def _avg_ranks(a: np.ndarray) -> np.ndarray:
        order = np.argsort(a, kind="mergesort")
        ranks = np.empty(len(a), dtype=float)
        ranks[order] = np.arange(1, len(a) + 1, dtype=float)
        # average ranks within tie groups
        sa = a[order]
        i = 0
        while i < len(sa):
            j = i
            while j + 1 < len(sa) and sa[j + 1] == sa[i]:
                j += 1
            if j > i:
                avg = (ranks[order[i]] + ranks[order[j]]) / 2.0
                for k in range(i, j + 1):
                    ranks[order[k]] = avg
            i = j + 1
        return ranks

    rx, ry = _avg_ranks(x), _avg_ranks(y)
    rho = float(np.corrcoef(rx, ry)[0, 1])
    return (0.0 if np.isnan(rho) else rho), n


def _lagged_pairs(
    all_episode_steps: List[List[Dict]],
    x_key: str,
    y_positive_fn: Callable[[Dict], bool],
    horizon: int,
) -> Tuple[List[float], List[float]]:
    """Pool (x_t, y_t) pairs across all episodes: x_t = steps[t][x_key];
    y_t = 1.0 if y_positive_fn(steps[k]) for any k in t+1..min(t+horizon, n-1),
    else 0.0. WITHIN-EPISODE ONLY (no cross-boundary lag -- matches 906c /
    Section 4/12b methodology). Skips t whose x is None/missing, and the last
    step of an episode (no t+1). Ported verbatim from 906c."""
    xs: List[float] = []
    ys: List[float] = []
    for steps in all_episode_steps:
        n = len(steps)
        for t in range(n - 1):
            xv = steps[t].get(x_key)
            if xv is None:
                continue
            window_end = min(t + horizon, n - 1)
            hit = any(y_positive_fn(steps[k]) for k in range(t + 1, window_end + 1))
            xs.append(float(xv))
            ys.append(1.0 if hit else 0.0)
    return xs, ys


def _contemporaneous_pairs(
    all_episode_steps: List[List[Dict]], x_key: str, y_key: str
) -> Tuple[List[float], List[float]]:
    """Pool (x_t, y_t) same-step pairs across all episodes (ported from 906c)."""
    xs: List[float] = []
    ys: List[float] = []
    for steps in all_episode_steps:
        for s in steps:
            xv, yv = s.get(x_key), s.get(y_key)
            if xv is None or yv is None:
                continue
            xs.append(float(xv))
            ys.append(float(yv))
    return xs, ys


def _emit_coupling(
    metrics: Dict[str, float], name: str,
    xs: List[float], ys: List[float],
) -> None:
    """Compute Pearson r + Spearman rho + n for one coupling and write the three
    fields (`coupling_<name>_r`, `_rho`, `_n`) into `metrics`."""
    r, n = _pearson_r(xs, ys)
    rho, _ = _spearman_r(xs, ys)
    metrics[f"coupling_{name}_r"]   = r
    metrics[f"coupling_{name}_rho"] = rho
    metrics[f"coupling_{name}_n"]   = float(n)


def _is_approach(s: Dict) -> bool:
    return s.get("mode") == "approach"


def _has_benefit(s: Dict) -> bool:
    v = s.get("benefit_exposure")
    return v is not None and float(v) > 0.0


def _has_harm_signal_pos(s: Dict) -> bool:
    # 906c's exact "benefit" operational definition for the z_goal->benefit
    # coupling (harm_signal>0), reproduced VERBATIM so the z_goal re-check is
    # directly comparable to 906c's r~0.07. NOTE this is 906c's own naming
    # quirk (it labels harm_signal>0 as "benefit"); the *cleaner* real
    # resource-contact signal is `benefit_exposure` (only live post-916a fix),
    # reported separately below as coupling_zgoal_t_to_benefitexp_*.
    v = s.get("harm_signal")
    return v is not None and float(v) > 0.0


def _has_moved(s: Dict) -> bool:
    return bool(s.get("moved"))


def _is_reef_exit(s: Dict) -> bool:
    return s.get("transition_type") == "reef_exit"


def _compute_coupling_metrics(all_episode_steps: List[List[Dict]]) -> Dict[str, float]:
    """The observational z_goal / residue_wanting -> behaviour couplings.
    Pools across every seed's every episode's steps (passed in flattened).

    z_goal couplings reproduce 906c EXACTLY (mode=="approach" and harm_signal>0)
    for a matched-substrate re-check against 906c's near-null, PLUS an improved
    z_goal->benefit_exposure coupling using 916a's now-live resource-contact
    signal. residue_wanting couplings are NEW (the wanting half never exercised
    on live data): approach, resource acquisition (benefit_exposure), locomotion
    (moved), and departure-from-refuge (reef_exit)."""
    m: Dict[str, float] = {}

    # --- z_goal, 906c-matched definitions (direct comparability) ---
    xs, ys = _lagged_pairs(all_episode_steps, "z_goal", _is_approach, LAG_APPROACH_H)
    _emit_coupling(m, "zgoal_t_to_approach_t1", xs, ys)
    xs, ys = _lagged_pairs(all_episode_steps, "z_goal", _has_harm_signal_pos, LAG_BENEFIT_H)
    _emit_coupling(m, "zgoal_t_to_benefit_t1t3", xs, ys)   # 906c's harm_signal>0 def

    # --- z_goal, improved real-resource-contact definition (916a fix enables) ---
    xs, ys = _lagged_pairs(all_episode_steps, "z_goal", _has_benefit, LAG_BENEFIT_H)
    _emit_coupling(m, "zgoal_t_to_benefitexp_t1t3", xs, ys)

    # --- residue_wanting -> behaviour (NEW; never measured on live data) ---
    xs, ys = _lagged_pairs(all_episode_steps, "residue_wanting", _is_approach, LAG_APPROACH_H)
    _emit_coupling(m, "wanting_t_to_approach_t1", xs, ys)
    xs, ys = _lagged_pairs(all_episode_steps, "residue_wanting", _has_benefit, LAG_BENEFIT_H)
    _emit_coupling(m, "wanting_t_to_benefitexp_t1t3", xs, ys)
    xs, ys = _lagged_pairs(all_episode_steps, "residue_wanting", _has_moved, LAG_MOVED_H)
    _emit_coupling(m, "wanting_t_to_moved_t1", xs, ys)
    xs, ys = _lagged_pairs(all_episode_steps, "residue_wanting", _is_reef_exit, LAG_REEFEXIT_H)
    _emit_coupling(m, "wanting_t_to_reefexit_t1", xs, ys)

    # --- contemporaneous: are the two "wanting" signals even related? ---
    xs, ys = _contemporaneous_pairs(all_episode_steps, "residue_wanting", "z_goal")
    _emit_coupling(m, "wanting_zgoal_contemporaneous", xs, ys)

    return m


# The couplings whose |r|/|rho| >= COUPLING_FLOOR_R with n >= MIN_COUPLING_N are
# reported as "non-trivial"; this is the set scanned for the interpretation label.
COUPLING_NAMES: List[str] = [
    "zgoal_t_to_approach_t1",
    "zgoal_t_to_benefit_t1t3",
    "zgoal_t_to_benefitexp_t1t3",
    "wanting_t_to_approach_t1",
    "wanting_t_to_benefitexp_t1t3",
    "wanting_t_to_moved_t1",
    "wanting_t_to_reefexit_t1",
    "wanting_zgoal_contemporaneous",
]
WANTING_COUPLING_NAMES: List[str] = [c for c in COUPLING_NAMES if c.startswith("wanting_t_")]


# ---------------------------------------------------------------------------
# Phase 0: Warmup training with dual-stream auxiliary losses
# ---------------------------------------------------------------------------

def _warmup_train(
    agent: REEAgent,
    env: CausalGridWorldV2,
    num_episodes: int,
    steps_per_episode: int,
    seed: int,
) -> Dict:
    device     = agent.device
    action_dim = env.action_dim

    e1_optimizer = optim.Adam(agent.e1.parameters(), lr=LR_E1)
    e2_wf_optimizer = optim.Adam(
        list(agent.e2.world_transition.parameters()) +
        list(agent.e2.world_action_encoder.parameters()),
        lr=LR_E2_WF,
    )
    harm_eval_optimizer = optim.Adam(
        agent.e3.harm_eval_head.parameters(), lr=LR_E3_HARM
    )

    aux_params: List[torch.nn.Parameter] = list(agent.latent_stack.parameters())
    aux_optimizer = optim.Adam(aux_params, lr=LR_ENC_AUX)

    wf_buf:        List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    harm_eval_buf: List[Tuple[torch.Tensor, torch.Tensor]]               = []
    reward_log:    List[float] = []

    agent.train()

    for ep in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        # residue_wanting orphaned-writer fix (916a vs 916): benefit_exposure
        # is populated into `info`, not `obs_dict` (env.reset() returns no
        # `info` at all) -- see module docstring. Default {} so the first
        # step's benefit_exposure read (before this episode's first env.step())
        # safely evaluates to 0.0, matching "no exposure recorded yet".
        info: Dict = {}

        z_world_prev: Optional[torch.Tensor] = None
        z_self_prev:  Optional[torch.Tensor] = None
        action_prev:  Optional[torch.Tensor] = None
        ep_reward = 0.0

        for _ in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            obs_h     = _obs_harm(obs_dict)
            obs_h_a   = _obs_harm_a(obs_dict)
            obs_h_h   = _obs_harm_history(obs_dict)
            prox_t    = _obs_resource_prox(obs_dict)
            accum_t   = _obs_accum(obs_dict)

            latent = agent.sense(
                obs_body, obs_world,
                obs_harm=obs_h, obs_harm_a=obs_h_a, obs_harm_history=obs_h_h,
            )
            z_world_curr = latent.z_world.detach()

            aux_terms: List[torch.Tensor] = []
            prox_target_t = torch.tensor([[prox_t]], device=device)
            prox_loss = agent.compute_resource_proximity_loss(prox_target_t, latent)
            if prox_loss is not None and prox_loss.requires_grad:
                aux_terms.append(prox_loss)
            accum_target_t = torch.tensor([[accum_t]], device=device)
            harm_accum_loss = agent.compute_harm_accum_loss(accum_target_t, latent)
            if harm_accum_loss is not None and harm_accum_loss.requires_grad:
                aux_terms.append(harm_accum_loss)
            if aux_terms:
                aux_loss = sum(aux_terms)
                aux_optimizer.zero_grad()
                aux_loss.backward(retain_graph=False)
                torch.nn.utils.clip_grad_norm_(aux_params, 1.0)
                aux_optimizer.step()

            if z_self_prev is not None and action_prev is not None:
                agent.record_transition(z_self_prev, action_prev, latent.z_self.detach())

            latent = agent.sense(
                obs_body, obs_world,
                obs_harm=obs_h, obs_harm_a=obs_h_a, obs_harm_history=obs_h_h,
            )

            ticks    = agent.clock.advance()
            e1_prior = (
                agent._e1_tick(latent) if ticks.get("e1_tick", False)
                else torch.zeros(1, WORLD_DIM, device=device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)

            drive_level      = REEAgent.compute_drive_level(obs_body)
            benefit_exposure = max(0.0, float(info.get("benefit_exposure", 0.0)))
            agent.update_z_goal(benefit_exposure=benefit_exposure, drive_level=drive_level)

            # residue_wanting orphaned-writer fix (916a vs 916) -- canonical
            # call site per _harness.py / v3_exq_263. update_benefit_salience
            # no-ops internally unless serotonin is enabled (tonic_5ht_enabled
            # above); update_schema_wanting no-ops unless schema_wanting_enabled
            # is set (left at its default False here, matching _harness.py's
            # own "default-off guard" idiom -- MECH-216 wired but inert).
            if benefit_exposure > 0:
                agent.update_benefit_salience(benefit_exposure)
            if bool(getattr(agent.config.e1, "schema_wanting_enabled", False)):
                agent.update_schema_wanting(drive_level=drive_level)

            action = agent.select_action(candidates, ticks, temperature=1.0)
            if action is None:
                action = _action_to_onehot(random.randint(0, action_dim - 1), action_dim, device)
                agent._last_action = action

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)
            ep_reward += float(harm_signal)

            # Populate residue (incl. MECH-307 split-surprise valence) so excite /
            # dread are non-degenerate at eval time. Also feeds MECH-302/303/304
            # accumulation state during warmup (residue update, safety terrain).
            agent.update_residue(float(harm_signal))

            if z_world_prev is not None and action_prev is not None:
                wf_buf.append((z_world_prev.cpu(), action_prev.cpu(), z_world_curr.cpu()))
                if len(wf_buf) > WF_BUF_MAX:
                    wf_buf = wf_buf[-WF_BUF_MAX:]

            harm_target = abs(float(harm_signal)) if float(harm_signal) < 0 else 0.0
            harm_eval_buf.append((z_world_curr.cpu(), torch.tensor([harm_target])))
            if len(harm_eval_buf) > HARM_EVAL_BUF_MAX:
                harm_eval_buf = harm_eval_buf[-HARM_EVAL_BUF_MAX:]

            if len(wf_buf) >= BATCH_SIZE:
                idxs  = torch.randperm(len(wf_buf))[:BATCH_SIZE].tolist()
                zw_b  = torch.cat([wf_buf[i][0] for i in idxs]).to(device)
                a_b   = torch.cat([wf_buf[i][1] for i in idxs]).to(device)
                zw1_b = torch.cat([wf_buf[i][2] for i in idxs]).to(device)
                wf_pred = agent.e2.world_forward(zw_b, a_b)
                wf_loss = F.mse_loss(wf_pred, zw1_b)
                if wf_loss.requires_grad:
                    e2_wf_optimizer.zero_grad()
                    wf_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(agent.e2.world_transition.parameters()) +
                        list(agent.e2.world_action_encoder.parameters()), 1.0,
                    )
                    e2_wf_optimizer.step()
                with torch.no_grad():
                    agent.e3.update_running_variance((wf_pred.detach() - zw1_b).detach())

            if len(harm_eval_buf) >= BATCH_SIZE:
                idxs = torch.randperm(len(harm_eval_buf))[:BATCH_SIZE].tolist()
                zw_b = torch.cat([harm_eval_buf[i][0] for i in idxs]).to(device)
                ht_b = torch.cat([harm_eval_buf[i][1] for i in idxs]).to(device)
                hp   = agent.e3.harm_eval(zw_b)
                he_loss = F.mse_loss(hp.squeeze(), ht_b.squeeze())
                if he_loss.requires_grad:
                    harm_eval_optimizer.zero_grad()
                    he_loss.backward()
                    harm_eval_optimizer.step()

            if len(agent._world_experience_buffer) >= 2:
                e1_loss = agent.compute_prediction_loss()
                if e1_loss.requires_grad:
                    e1_optimizer.zero_grad()
                    e1_loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.e1.parameters(), 1.0)
                    e1_optimizer.step()

            z_world_prev = z_world_curr
            z_self_prev  = latent.z_self.detach()
            action_prev  = action.detach()
            if done:
                break

        reward_log.append(ep_reward)

        if (ep + 1) % 10 == 0 or ep == num_episodes - 1:
            print(
                f"  [warmup] seed={seed} ep {ep+1}/{num_episodes}"
                f"  rv={agent.e3._running_variance:.4f}  ep_reward={ep_reward:.4f}",
                flush=True,
            )

    first10 = float(np.mean(reward_log[:10]))  if len(reward_log) >= 10 else float(np.mean(reward_log))
    last10  = float(np.mean(reward_log[-10:])) if len(reward_log) >= 10 else float(np.mean(reward_log))

    return {
        "final_running_variance": agent.e3._running_variance,
        "warmup_first10_reward":  first10,
        "warmup_last10_reward":   last10,
    }


# ---------------------------------------------------------------------------
# Phase 1: Evaluation with affect recording for the fishtank feed
# ---------------------------------------------------------------------------

def _eval_agent(
    agent: REEAgent,
    env: CausalGridWorldV2,
    num_episodes: int,
    steps_per_episode: int,
    seed: int,
) -> Dict:
    action_dim    = env.action_dim
    device        = agent.device
    episode_rewards: List[float] = []
    episode_harms:   List[float] = []
    n_cands_log:     List[int]   = []
    episodes_log:    List[Dict]  = []

    # Per-channel non-degeneracy accumulators (across all eval steps). Core
    # channels (664-inherited) are load-bearing for PASS; relief/safety channels
    # are reported here for the same non-degeneracy statistics but are NOT
    # load-bearing (see module docstring).
    chan_vals: Dict[str, List[float]] = {
        k: [] for k in ["z_harm_s", "z_harm_un", "z_harm_a", "drive", "z_goal",
                        "vigor", "override", "z_block", "excite", "dread",
                        "safety_cue_signal", "safety_terrain_read",
                        # residue_wanting orphaned-writer fix (916a vs 916) --
                        # reported here, NOT load-bearing (same convention as
                        # excite/dread), so a run of this script directly
                        # evidences the fix.
                        "residue_wanting"]
    }
    freeze_fires = 0
    block_steps  = 0
    relief_fires = 0

    agent.eval()

    for ep_idx in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        # residue_wanting orphaned-writer fix (916a vs 916) -- see the matching
        # comment in _warmup_train above.
        info: Dict = {}

        z_self_prev:  Optional[torch.Tensor] = None
        z_world_prev: Optional[torch.Tensor] = None
        action_prev:  Optional[torch.Tensor] = None
        ep_reward = 0.0
        ep_harm   = 0.0

        ep_steps: List[Dict] = []
        initial_hazards   = [list(h) for h in env.hazards]
        initial_resources = [list(r) for r in env.resources]
        current_hazards   = [list(h) for h in env.hazards]
        current_resources = [list(r) for r in env.resources]

        reef_cells     = _get_reef_cells(env)
        reef_cells_set = getattr(env, "_reef_cells", set())
        prev_in_reef   = False
        pos_prev: Optional[Tuple[int, int]] = None   # V3-EXQ-932: for per-step `moved`

        for step_idx in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            obs_h     = _obs_harm(obs_dict)
            obs_h_a   = _obs_harm_a(obs_dict)
            obs_h_h   = _obs_harm_history(obs_dict)
            with torch.no_grad():
                latent = agent.sense(
                    obs_body, obs_world,
                    obs_harm=obs_h, obs_harm_a=obs_h_a, obs_harm_history=obs_h_h,
                )
                # MECH-302/MECH-304: capture BEFORE select_action(). Both signals
                # are consumed/cleared inside select_action() (relief conditionally
                # on having fired, safety unconditionally every tick) -- reading
                # them after select_action(), like every other channel in
                # _read_affect, would always observe a cleared/zeroed value. See
                # module docstring TIMING NOTE and agent.py select_action()'s
                # MECH-302 / MECH-304 blocks.
                relief_event      = bool(agent._relief_completion_event)
                safety_cue_signal = float(agent._conditioned_safety_signal)
                if z_self_prev is not None and action_prev is not None:
                    agent.record_transition(z_self_prev, action_prev, latent.z_self.detach())
                ticks    = agent.clock.advance()
                e1_prior = (
                    agent._e1_tick(latent) if ticks.get("e1_tick", False)
                    else torch.zeros(1, WORLD_DIM, device=device)
                )
                candidates = agent.generate_trajectories(latent, e1_prior, ticks)

                drive_level      = REEAgent.compute_drive_level(obs_body)
                benefit_exposure = max(0.0, float(info.get("benefit_exposure", 0.0)))
                agent.update_z_goal(benefit_exposure=benefit_exposure, drive_level=drive_level)

                # residue_wanting orphaned-writer fix (916a vs 916) -- see the
                # matching comment in _warmup_train above.
                if benefit_exposure > 0:
                    agent.update_benefit_salience(benefit_exposure)
                if bool(getattr(agent.config.e1, "schema_wanting_enabled", False)):
                    agent.update_schema_wanting(drive_level=drive_level)

                if ticks.get("e3_tick", True):
                    n_cands_log.append(len(candidates))
                action = agent.select_action(candidates, ticks, temperature=1.0)
                if action is None:
                    action = _action_to_onehot(random.randint(0, action_dim - 1), action_dim, device)
                    agent._last_action = action

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)

            # Keep residue (incl. valence) populated during eval too.
            with torch.no_grad():
                agent.update_residue(float(harm_signal))

            if info.get("env_drift_occurred", False):
                current_hazards   = [list(h) for h in env.hazards]
                current_resources = [list(r) for r in env.resources]

            agent_pos = (int(env.agent_x), int(env.agent_y))
            in_reef   = agent_pos in reef_cells_set
            blocked   = bool(info.get("action_blocked_this_step", False))
            if blocked:
                block_steps += 1

            # V3-EXQ-932 coupling instrumentation (behaviour readouts). All read
            # from the POST-step `info` / position -- i.e. the outcome of the
            # action just taken at this step, which is the behaviour the lagged
            # couplings predict FROM the affect signals read at earlier steps.
            #   benefit_exposure_realized: genuine resource contact this step
            #     (real only because use_proxy_fields=True on 916a's env; 0.0
            #     structurally in the pre-916a lineage). The "resource-seeking /
            #     target-acquisition" behavioural readout.
            #   moved: locomotion (position changed since the previous step).
            benefit_exposure_realized = max(0.0, float(info.get("benefit_exposure", 0.0)))
            moved = bool(pos_prev is not None and list(agent_pos) != list(pos_prev))

            # Affect readout (after select_action -> module caches are current,
            # EXCEPT relief_event/safety_cue_signal which were captured above,
            # pre-select_action -- see TIMING NOTE).
            affect = _read_affect(
                agent, latent, obs_body,
                relief_event=relief_event, safety_cue_signal=safety_cue_signal,
            )
            if affect["freeze"]:
                freeze_fires += 1
            if relief_event:
                relief_fires += 1
            for k, lst in chan_vals.items():
                v = affect.get(k)
                if isinstance(v, (int, float)) and v is not None:
                    lst.append(float(v))

            z_harm_s = affect["z_harm_s"] if affect["z_harm_s"] is not None else 0.0
            z_beta_val = (
                float(latent.z_beta.mean().item()) if latent.z_beta is not None else 0.0
            )
            world_change_norm = (
                float((latent.z_world - z_world_prev).norm().item())
                if z_world_prev is not None else 0.0
            )

            mode = _classify_mode(
                z_harm_s, world_change_norm, float(harm_signal),
                in_reef, affect["freeze"], affect["z_block"],
            )

            # Transition label: action_blocked > reef edge > env transition_type.
            if blocked:
                step_transition = "action_blocked"
            elif in_reef and not prev_in_reef:
                step_transition = "reef_entry"
            elif not in_reef and prev_in_reef:
                step_transition = "reef_exit"
            else:
                step_transition = info.get("transition_type", "none")

            step_rec = {
                "t":                 step_idx,
                "pos":               list(agent_pos),
                "action":            int(action.argmax(dim=-1).item()),
                "harm_signal":       float(harm_signal),
                # legacy + cascade
                "z_harm_norm":       z_harm_s,
                "z_harm_s":          affect["z_harm_s"],
                "z_harm_un":         affect["z_harm_un"],
                "z_harm_a":          affect["z_harm_a"],
                "z_world_norm":      float(latent.z_world.norm().item()),
                "z_beta_val":        z_beta_val,
                "world_change_norm": world_change_norm,
                # affect register
                "drive":             affect["drive"],
                "z_goal":            affect["z_goal"],
                "vigor":             affect["vigor"],
                "override":          affect["override"],
                "z_block":           affect["z_block"],
                "freeze":            affect["freeze"],
                "excite":            affect["excite"],
                "dread":             affect["dread"],
                "surprise":          affect["surprise"],
                "residue_wanting":   affect["residue_wanting"],
                "liking":            affect["liking"],
                # relief / safety register (this experiment's addition)
                "relief_event":        affect["relief_event"],
                "safety_cue_signal":   affect["safety_cue_signal"],
                "safety_terrain_read": affect["safety_terrain_read"],
                # behaviour
                "mode":              mode,
                "transition_type":   step_transition,
                "health":            float(info.get("health", 1.0)),
                "energy":            float(info.get("energy", 1.0)),
                "harm_event":        float(harm_signal) < 0,
                "n_cands":           len(candidates),
                "hazards":           [list(h) for h in current_hazards],
                "resources":         [list(r) for r in current_resources],
                "in_reef":           in_reef,
                # V3-EXQ-932 behaviour readouts for the coupling analysis
                "benefit_exposure":  benefit_exposure_realized,
                "moved":             moved,
            }
            ep_steps.append(step_rec)

            prev_in_reef = in_reef
            pos_prev     = agent_pos
            z_self_prev  = latent.z_self.detach()
            z_world_prev = latent.z_world.detach()
            action_prev  = action.detach()

            ep_reward += float(harm_signal)
            if float(harm_signal) < 0:
                ep_harm += abs(float(harm_signal))
            if done:
                break

        episode_rewards.append(ep_reward)
        episode_harms.append(ep_harm)
        episodes_log.append({
            "ep":                ep_idx,
            "initial_hazards":   initial_hazards,
            "initial_resources": initial_resources,
            "reef_cells":        reef_cells,
            "steps":             ep_steps,
        })

        print(
            f"  [eval] seed={seed} ep {ep_idx+1}/{num_episodes}"
            f"  reward={ep_reward:.4f}  harm={ep_harm:.4f}  steps={len(ep_steps)}",
            flush=True,
        )

    # Per-channel std (non-degeneracy signal).
    chan_std = {k: (float(np.std(v)) if len(v) >= 2 else 0.0) for k, v in chan_vals.items()}
    chan_mean = {k: (float(np.mean(v)) if v else 0.0) for k, v in chan_vals.items()}

    return {
        "mean_reward":  float(np.mean(episode_rewards)),
        "mean_harm":    float(np.mean(episode_harms)),
        "mean_n_cands": float(np.mean(n_cands_log)) if n_cands_log else 0.0,
        "episodes":     episodes_log,
        "chan_std":     chan_std,
        "chan_mean":    chan_mean,
        "freeze_fires": freeze_fires,
        "block_steps":  block_steps,
        "relief_fires": relief_fires,
        "eval_steps":   int(sum(len(e["steps"]) for e in episodes_log)),
    }


# ---------------------------------------------------------------------------
# Per-seed run
# ---------------------------------------------------------------------------

def run_seed(seed: int, dry_run: bool = False) -> Dict:
    warmup_eps = 3 if dry_run else WARMUP_EPISODES
    eval_eps   = 2 if dry_run else EVAL_EPISODES
    steps      = 30 if dry_run else STEPS_PER_EPISODE

    print(f"\nSeed {seed} Condition zgoal_wanting_coupling", flush=True)
    print(
        f"[EXQ-932] Seed {seed}  warmup={warmup_eps}  eval={eval_eps}"
        f"  steps/ep={steps}  dry_run={dry_run}",
        flush=True,
    )

    agent, env = _make_agent_and_env(seed)
    print(
        f"[EXQ-932] Seed {seed} -- world_obs_dim={env.world_obs_dim}"
        f"  body_obs_dim={env.body_obs_dim}  affective+relief/safety stack ON",
        flush=True,
    )

    warmup = _warmup_train(agent, env, warmup_eps, steps, seed)
    ree    = _eval_agent(agent, env, eval_eps, steps, seed)

    # z_goal liveness (Experimental Recording Standard) -- read counters off this
    # seed's agent now, before it goes out of scope at the end of run_seed().
    _ZG.observe(agent)

    print(
        f"[EXQ-932] Seed {seed} channel std: "
        + "  ".join(f"{k}={ree['chan_std'][k]:.4f}" for k in
                    ["z_harm_a", "z_harm_un", "drive", "vigor", "z_block",
                     "excite", "dread", "safety_cue_signal", "safety_terrain_read",
                     "residue_wanting"]),
        flush=True,
    )
    print(
        f"[EXQ-932] Seed {seed} freeze_fires={ree['freeze_fires']}"
        f"  block_steps={ree['block_steps']}  relief_fires={ree['relief_fires']}"
        f"  eval_steps={ree['eval_steps']}",
        flush=True,
    )

    # Per-seed verdict for runner progress (one per seed x condition run). Same
    # core-channel gate as 664 -- relief/safety are reported, not load-bearing.
    seed_core_ok = all(ree["chan_std"].get(k, 0.0) > STD_FLOOR for k in CORE_CHANNELS)
    seed_pass = bool(seed_core_ok and ree["block_steps"] > 0)
    print(f"verdict: {'PASS' if seed_pass else 'FAIL'}", flush=True)

    return {
        "seed":                  seed,
        "warmup_first10_reward": warmup["warmup_first10_reward"],
        "warmup_last10_reward":  warmup["warmup_last10_reward"],
        "warmup_final_rv":       warmup["final_running_variance"],
        "eval_mean_reward":      ree["mean_reward"],
        "eval_mean_harm":        ree["mean_harm"],
        "eval_mean_n_cands":     ree["mean_n_cands"],
        "chan_std":              ree["chan_std"],
        "chan_mean":             ree["chan_mean"],
        "freeze_fires":          ree["freeze_fires"],
        "block_steps":           ree["block_steps"],
        "relief_fires":          ree["relief_fires"],
        "eval_steps":            ree["eval_steps"],
        "episodes":              ree["episodes"],
    }


# ---------------------------------------------------------------------------
# Aggregate + output
# ---------------------------------------------------------------------------

# Core channels whose non-degeneracy defines a successful showcase -- identical
# to 664's gate. relief_event / safety_cue_signal / safety_terrain_read are
# reported (chan_nondegen / relief_fires below) but deliberately NOT load-bearing:
# MECH-302 needs a genuine sustained z_harm_a descent within a 5-tick window and
# MECH-304 needs a prior MECH-302 firing to seed its EMA prototype, so either can
# be legitimately sparse within a bounded eval window even when correctly wired
# (same reasoning 664 applies to excite/dread/override).
CORE_CHANNELS = ["z_harm_a", "z_harm_un", "drive"]
STD_FLOOR     = 1e-4

# z_goal liveness accumulator (Experimental Recording Standard). Agents are built
# fresh inside run_seed(); observe() reads live counters off each agent right
# after it finishes stepping, rather than keeping every seed's agent alive.
_ZG = ZGoalStreamAccumulator()


def run(seeds=None, dry_run: bool = False) -> dict:
    if seeds is None:
        seeds = [0, 1, 2]

    print(
        f"[V3-EXQ-932] z_goal / residue_wanting -> behaviour observational coupling\n"
        f"  Seeds: {seeds}\n"
        f"  Warmup: {WARMUP_EPISODES} eps  Eval: {EVAL_EPISODES} eps"
        f"  Steps/ep: {STEPS_PER_EPISODE}\n"
        f"  Substrate: V3-EXQ-916a verbatim (residue_wanting writer fix live)\n"
        f"  Instrument: V3-EXQ-906c coupling machinery + NEW residue_wanting couplings\n"
        f"  Floor: |r|/|rho| >= {COUPLING_FLOOR_R}, n >= {MIN_COUPLING_N}  (906c near-null r~0.07)\n"
        f"  OBSERVATIONAL not causal (causal = V3-EXQ-931 CEM wanting_weight ablation)\n"
        f"  Output: REE_assembly/evidence/experiments/{EXPERIMENT_TYPE}/",
        flush=True,
    )

    seed_results = [run_seed(s, dry_run=dry_run) for s in seeds]

    # Aggregate per-channel: a channel is non-degenerate if it varies on >=1 seed.
    chan_keys = list(seed_results[0]["chan_std"].keys())
    chan_max_std = {k: max(r["chan_std"].get(k, 0.0) for r in seed_results) for k in chan_keys}
    chan_nondegen = {k: bool(chan_max_std[k] > STD_FLOOR) for k in chan_keys}
    total_freeze = sum(r["freeze_fires"] for r in seed_results)
    total_block  = sum(r["block_steps"] for r in seed_results)
    total_relief = sum(r["relief_fires"] for r in seed_results)
    total_steps  = sum(r["eval_steps"] for r in seed_results)

    core_ok   = all(chan_nondegen.get(k, False) for k in CORE_CHANNELS)
    block_ok  = total_block > 0
    # freeze never permanently locks: if it ever fired, it must not be every step.
    freeze_not_locked = (total_freeze == 0) or (total_freeze < total_steps)

    # --- V3-EXQ-932: the coupling measurement (the actual scientific payload) ---
    # Pool every seed's every episode's steps, flattened, exactly as 906c does.
    all_episode_steps: List[List[Dict]] = [
        ep["steps"] for r in seed_results for ep in r.get("episodes", []) if ep.get("steps")
    ]
    coupling_metrics = _compute_coupling_metrics(all_episode_steps)

    # Readiness (the measurement's premise): the signals being correlated MUST
    # actually vary this run, or a coupling of ~0 is uninformative, not a null.
    wanting_nondegen = bool(chan_nondegen.get("residue_wanting", False))
    zgoal_nondegen   = bool(chan_nondegen.get("z_goal", False))

    # Per-coupling adequacy: enough paired observations to interpret at all.
    coupling_n_ok = {
        c: bool(coupling_metrics.get(f"coupling_{c}_n", 0.0) >= MIN_COUPLING_N)
        for c in COUPLING_NAMES
    }
    all_couplings_powered = all(coupling_n_ok.values())

    # Non-trivial coupling detection (|r| OR |rho| over the pre-registered floor,
    # with an adequate n). Diagnostic ASSOCIATION only -- NOT causal.
    def _nontrivial(c: str) -> bool:
        r   = abs(float(coupling_metrics.get(f"coupling_{c}_r", 0.0)))
        rho = abs(float(coupling_metrics.get(f"coupling_{c}_rho", 0.0)))
        return bool(coupling_n_ok[c] and max(r, rho) >= COUPLING_FLOOR_R)

    nontrivial = {c: _nontrivial(c) for c in COUPLING_NAMES}
    any_wanting_coupling = any(nontrivial[c] for c in WANTING_COUPLING_NAMES)
    any_zgoal_coupling   = any(nontrivial[c] for c in COUPLING_NAMES
                              if c.startswith("zgoal_"))

    # PASS = the instrument produced a VALID, well-powered measurement (the
    # signals varied and there were enough pairs). Whether a coupling was FOUND
    # is REPORTED, not gated -- a well-measured null is a valid diagnostic result.
    # If residue_wanting is flat, the wanting couplings are unmeasurable and the
    # run self-routes substrate_not_ready_requeue (NOT a substrate verdict label).
    measurement_valid = bool(wanting_nondegen and zgoal_nondegen and core_ok
                             and all_couplings_powered)
    passed  = measurement_valid
    outcome = "PASS" if passed else "FAIL"

    if not wanting_nondegen:
        interp_label = "substrate_not_ready_requeue"
    elif not all_couplings_powered:
        interp_label = "coupling_underpowered_requeue"
    elif any_wanting_coupling:
        interp_label = "wanting_behaviour_coupling_detected"
    else:
        interp_label = "wanting_behaviour_coupling_null"

    metrics: Dict = {"n_seeds": float(len(seeds))}
    for r in seed_results:
        s = r["seed"]
        for key in ("warmup_first10_reward", "warmup_last10_reward", "warmup_final_rv",
                    "eval_mean_reward", "eval_mean_harm", "eval_mean_n_cands",
                    "freeze_fires", "block_steps", "relief_fires", "eval_steps"):
            metrics[f"seed{s}_{key}"] = float(r[key])
    for k in chan_keys:
        metrics[f"chan_max_std_{k}"] = float(chan_max_std[k])
        metrics[f"chan_nondegen_{k}"] = 1.0 if chan_nondegen[k] else 0.0
    metrics["total_freeze_fires"] = float(total_freeze)
    metrics["total_block_steps"]  = float(total_block)
    metrics["total_relief_fires"] = float(total_relief)
    metrics["total_eval_steps"]   = float(total_steps)

    # --- V3-EXQ-932 coupling payload (r, rho, n per coupling) + detection flags ---
    metrics.update({k: float(v) for k, v in coupling_metrics.items()})
    metrics["coupling_floor_r"]        = float(COUPLING_FLOOR_R)
    metrics["min_coupling_n"]          = float(MIN_COUPLING_N)
    metrics["residue_wanting_nondegen"] = 1.0 if wanting_nondegen else 0.0
    metrics["z_goal_nondegen"]          = 1.0 if zgoal_nondegen else 0.0
    metrics["n_coupling_pairs_total"]   = float(sum(len(s) for s in all_episode_steps))
    for c in COUPLING_NAMES:
        metrics[f"coupling_{c}_nontrivial"] = 1.0 if nontrivial[c] else 0.0
    metrics["any_wanting_coupling_nontrivial"] = 1.0 if any_wanting_coupling else 0.0
    metrics["any_zgoal_coupling_nontrivial"]   = 1.0 if any_zgoal_coupling else 0.0

    # Worst (smallest-n) coupling, for the powered-enough readiness precondition
    # -- report the WORST CELL, not the mean (CLAUDE.md / queue-experiment rule).
    _min_n_name = min(COUPLING_NAMES,
                      key=lambda c: coupling_metrics.get(f"coupling_{c}_n", 0.0))
    _min_n = float(coupling_metrics.get(f"coupling_{_min_n_name}_n", 0.0))

    interpretation = {
        "label": interp_label,
        # Readiness-kind preconditions: the measurement's premises, each a
        # measured quantity the indexer recomputes met from (floor direction).
        "preconditions": [
            {
                "name": "residue_wanting_nondegenerate",
                "description": ("residue_wanting must actually vary this run (the way "
                                "916a validated the writer fix, chan_max_std~0.57) or the "
                                "wanting->behaviour couplings are unmeasurable; below-floor "
                                "self-routes substrate_not_ready_requeue, NOT a null/ceiling"),
                "measured": float(chan_max_std.get("residue_wanting", 0.0)),
                "threshold": float(STD_FLOOR),
                "direction": "lower",   # FLOOR: met when std >= STD_FLOOR
                "control": "V3-EXQ-916a landed run: chan_max_std_residue_wanting=0.568, nondegen=True",
                "met": wanting_nondegen,
            },
            {
                "name": "z_goal_nondegenerate",
                "description": ("z_goal must vary for its couplings and the matched "
                                "re-check vs 906c's near-null to be meaningful"),
                "measured": float(chan_max_std.get("z_goal", 0.0)),
                "threshold": float(STD_FLOOR),
                "direction": "lower",   # FLOOR: met when std >= STD_FLOOR
                "control": "V3-EXQ-916a landed run: chan_max_std_z_goal=0.078, nondegen=True",
                "met": zgoal_nondegen,
            },
            {
                "name": "couplings_adequately_powered",
                "description": ("every coupling has >= MIN_COUPLING_N paired obs; measured "
                                "is the WORST (smallest-n) coupling"),
                "measured": _min_n,
                "threshold": float(MIN_COUPLING_N),
                "direction": "lower",   # FLOOR: met when worst n >= MIN_COUPLING_N
                "offending_cell": _min_n_name,
                "met": all_couplings_powered,
            },
        ],
        # Per-coupling non-degeneracy: did each coupling have enough pairs to be
        # interpretable (not a trivially-tiny n)?
        "criteria_non_degenerate": {
            **{f"coupling_{c}_powered": coupling_n_ok[c] for c in COUPLING_NAMES},
            "residue_wanting_varies": wanting_nondegen,
            "z_goal_varies": zgoal_nondegen,
        },
        # Load-bearing criteria: the measurement's validity (NOT whether a
        # coupling was found -- a well-powered null is a valid diagnostic result).
        "criteria": [
            {"name": "residue_wanting_measurable", "load_bearing": True, "passed": wanting_nondegen},
            {"name": "z_goal_measurable",          "load_bearing": True, "passed": zgoal_nondegen},
            {"name": "couplings_adequately_powered", "load_bearing": True, "passed": all_couplings_powered},
        ],
        "combination_rule": ("PASS (measurement_valid) = residue_wanting varies AND z_goal "
                             "varies AND core channels non-degenerate AND every coupling has "
                             ">= MIN_COUPLING_N pairs. This gates MEASUREMENT VALIDITY, not "
                             "coupling detection: whether any coupling clears the |r|/|rho| "
                             "floor is REPORTED (any_wanting_coupling_nontrivial / "
                             "any_zgoal_coupling_nontrivial), never gated."),
        "coupling_summary": {
            "floor_r": float(COUPLING_FLOOR_R),
            "min_n": float(MIN_COUPLING_N),
            "reference_906c_near_null_r": 0.07,
            "nontrivial_couplings": [c for c in COUPLING_NAMES if nontrivial[c]],
            "any_wanting_coupling_nontrivial": any_wanting_coupling,
            "any_zgoal_coupling_nontrivial": any_zgoal_coupling,
        },
        "note": ("OBSERVATIONAL / DIAGNOSTIC -- promotes nothing, weights no governance "
                 "(claim_ids=[], experiment_purpose=diagnostic). Answers organism-review "
                 "Section 3 Level A (observational coupling) for residue_wanting for the "
                 "FIRST time, on V3-EXQ-916a's repaired substrate (residue_wanting now "
                 "varies; it was hard-0 across the whole 906-family lineage), and re-checks "
                 "z_goal against 906c's near-null (r~0.07, n=3785) using the byte-identical "
                 "906c instrument on the fixed substrate. A lagged correlation is consistent "
                 "with but does NOT establish that the signal DRIVES behaviour (common-cause "
                 "confounds: proximity to a resource raises both wanting and approach). The "
                 "causal-necessity test is the SEPARATE ablation V3-EXQ-931 (CEM "
                 "wanting_weight); this must not be read as causal. PASS means the instrument "
                 "produced a valid, well-powered measurement, NOT that a coupling was found; "
                 "a well-measured null (wanting_behaviour_coupling_null) is the correct "
                 "reading if residue_wanting varied but predicts nothing above the floor."),
    }

    def _crow(c: str) -> str:
        r   = float(coupling_metrics.get(f"coupling_{c}_r", 0.0))
        rho = float(coupling_metrics.get(f"coupling_{c}_rho", 0.0))
        n   = int(coupling_metrics.get(f"coupling_{c}_n", 0.0))
        tag = "NON-TRIVIAL" if nontrivial[c] else ("underpowered" if not coupling_n_ok[c] else "near-null")
        return f"| `{c}` | {r:+.4f} | {rho:+.4f} | {n} | {tag} |\n"

    zgoal_rows   = "".join(_crow(c) for c in COUPLING_NAMES if c.startswith("zgoal_"))
    wanting_rows = "".join(_crow(c) for c in COUPLING_NAMES
                          if c.startswith("wanting_t_") or c == "wanting_zgoal_contemporaneous")

    summary_markdown = f"""# V3-EXQ-932 -- z_goal / residue_wanting -> behaviour observational coupling

**Status:** {outcome}  (diagnostic / observational -- claim_ids=[], weights no governance)
**Interpretation label:** `{interp_label}`
**Substrate:** V3-EXQ-916a verbatim (residue_wanting writer fix; use_proxy_fields=True,
tonic_5ht_enabled=True, update_benefit_salience() wired). Coupling instrument ported
byte-for-byte from V3-EXQ-906c so the z_goal re-check is directly comparable to 906c's
near-null (r~0.07, n=3785), now on the fixed substrate.

**OBSERVATIONAL, NOT CAUSAL.** A lagged correlation is consistent with but does not
establish that the internal signal DRIVES behaviour (common-cause confounds abound). The
causal-necessity test is the separate ablation V3-EXQ-931 (CEM wanting_weight); this run
does not duplicate it and must not be read as causal.

**Pre-registered non-trivial-coupling floor:** |r| or |rho| >= {COUPLING_FLOOR_R} with
n >= {MIN_COUPLING_N} (906c's r~0.07 is the "no coupling" reference).

**Readiness (measurement premise):**
- residue_wanting varies: {"YES" if wanting_nondegen else "NO -> substrate_not_ready_requeue"}  (max std {chan_max_std.get("residue_wanting", 0.0):.5f})
- z_goal varies: {"YES" if zgoal_nondegen else "NO"}  (max std {chan_max_std.get("z_goal", 0.0):.5f})
- all couplings powered (n >= {MIN_COUPLING_N}): {"YES" if all_couplings_powered else "NO (worst n=" + str(int(_min_n)) + " @ " + _min_n_name + ")"}
- total pooled steps: {sum(len(s) for s in all_episode_steps)}

## z_goal -> behaviour couplings (906c-matched re-check + improved benefit_exposure)
| coupling | Pearson r | Spearman rho | n | reading |
|----------|-----------|--------------|---|---------|
{zgoal_rows}
## residue_wanting -> behaviour couplings (NEW -- first time on live data)
| coupling | Pearson r | Spearman rho | n | reading |
|----------|-----------|--------------|---|---------|
{wanting_rows}
`*_benefit_t1t3` uses 906c's exact harm_signal>0 definition (for comparability);
`*_benefitexp_t1t3` uses 916a's now-live real resource-contact signal (benefit_exposure).
`wanting_zgoal_contemporaneous` cross-checks whether the two "wanting" signals (the
hippocampal-map residue channel vs the frontal goal-attractor) even track each other.

**Channel non-degeneracy (max std across seeds):**
{chr(10).join(f'- {k}: {chan_max_std[k]:.5f}  ({"varies" if chan_nondegen[k] else "FLAT"})' for k in chan_keys)}
"""

    config_snapshot = {
        "env_kwargs": ENV_KWARGS,
        "warmup_episodes": WARMUP_EPISODES,
        "eval_episodes": EVAL_EPISODES,
        "steps_per_episode": STEPS_PER_EPISODE,
        "coupling_floor_r": COUPLING_FLOOR_R,
        "min_coupling_n": MIN_COUPLING_N,
        "lag_horizons": {
            "approach": LAG_APPROACH_H, "benefit": LAG_BENEFIT_H,
            "moved": LAG_MOVED_H, "reef_exit": LAG_REEFEXIT_H,
        },
        "std_floor": STD_FLOOR,
        "core_channels": CORE_CHANNELS,
        "seeds": list(seeds),
    }

    episode_log = {
        "experiment_type": EXPERIMENT_TYPE,
        "env_config":      ENV_KWARGS,
        "phase":           "zgoal_wanting_coupling",
        "toroidal":        ENV_KWARGS.get("toroidal", False),
        "seeds": [
            {"seed": r["seed"], "episodes": r.get("episodes", [])}
            for r in seed_results
        ],
    }

    return {
        "status":             outcome,
        "outcome":            outcome,
        "metrics":            metrics,
        "summary_markdown":   summary_markdown,
        "claim_ids":          CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "evidence_direction": "non_contributory",
        "experiment_type":    EXPERIMENT_TYPE,
        "interpretation":     interpretation,
        "config":             config_snapshot,
        "seeds":              list(seeds),
        "episode_log":        episode_log,
    }


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",   type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    result = run(seeds=args.seeds, dry_run=args.dry_run)

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"]      = ts
    result["timestamp_utc"]      = ts
    result["run_id"]             = f"{EXPERIMENT_TYPE}_{ts}_v3"
    result["architecture_epoch"] = "ree_hybrid_guardrails_v1"
    result["experiment_purpose"] = EXPERIMENT_PURPOSE
    result["claim_ids"]          = CLAIM_IDS

    out_dir = (
        Path(__file__).resolve().parents[2]
        / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    episode_log = result.pop("episode_log", None)
    if episode_log is not None:
        episode_log["run_id"] = result["run_id"]
        log_path = out_dir / f"{EXPERIMENT_TYPE}_{ts}_episode_log.json"
        log_path.write_text(json.dumps(episode_log, indent=2) + "\n", encoding="utf-8")
        print(f"Episode log written to: {log_path}", flush=True)
        # Declared companion path, relative to write_flat_manifest's out_dir
        # (out_dir.parent below) -- NOT out_dir itself. experiment_runner.py
        # _collect_companion_files resolves a declared relative entry against
        # the MANIFEST's directory (evidence/experiments/), one level above
        # where the episode_log actually lands (evidence/experiments/
        # {EXPERIMENT_TYPE}/), so the prefix is required or the runner's
        # Phase-3 sidefile sync silently finds nothing.
        result["companion_files"] = [f"{EXPERIMENT_TYPE}/{log_path.name}"]

    out_path = write_flat_manifest(
        result,
        out_dir.parent,
        dry_run=args.dry_run,
        config=result.get("config"),
        seeds=result.get("seeds"),
        script_path=Path(__file__),
        z_goal_stream_stats=_ZG.stats(),
    )

    print(f"\nResult written to: {out_path}", flush=True)
    print(f"Status: {result['status']}", flush=True)
    print(f"final_outcome: {result['outcome']}", flush=True)

    _outcome_raw = str(result.get("outcome", "FAIL")).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
        dry_run=bool(args.dry_run),
    )
