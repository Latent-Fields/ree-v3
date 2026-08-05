#!/opt/local/bin/python3
"""V3-EXQ-890 -- MECH-471 CHEAP MANY-SEED ACQUISITION-RELIABILITY DIAGNOSTIC PROBE.

WHY THIS EXISTS. Two full-budget MECH-471 attempts (V3-EXQ-875, ~19-20h; V3-EXQ-875a,
corrected requeue, same order) each measured a stable, seed-dependent bimodal split in
survival-competence acquisition on the all-ON REE stack: seed 43 clears the survival floor
strongly; seeds 42/45 stay near the random-walk baseline regardless of budget. On 2026-08-05
a *different* claim's experiment (V3-EXQ-882a, MECH-472, sharing the identical acquisition
recipe and env) independently reproduced the same split at n=8 (seeds 43/50 solve; 42, 45,
46, 47, 48, 49 never approach the floor across a 10x exposure sweep with no improving trend).
Neither prior run was DESIGNED to discriminate WHY the split exists. The hypothesis-space
registry (REE_assembly/evidence/planning/hypothesis_space_registry.v1.json,
qid=mech471_competence_acquisition_reliability, registered 2026-08-05T06:20:43Z) names three
live hypotheses and recommends exactly this instrument:

  H-exploration-init-variance      -- failing seeds get stuck EARLY in a bad local optimum
                                       and never escape, regardless of remaining budget.
  H-hazard-layout-difficulty-variance -- per-seed hazard/reef layout randomization makes some
                                       seeds' survival task genuinely harder (task-difficulty
                                       variance, not learning variance).
  H-bias-head-ofc-interaction      -- the shared stack's other drives (bias_head lPFC action
                                       bias, OFC devaluation bias) interact adversely with
                                       survival acquisition in a seed-dependent way.

This probe is a `complex (probe-gated) / puzzle (known rules)` diagnostic spike, not a
`complicated (buildable)` build and not an `evidence`-purpose test of MECH-471's own
registered falsifier (the local-update interference test, V3-EXQ-875/875a). Its deliverable
is a FACT (which explanation the bimodal split's shape favours), not a shipped feature, and
it does NOT itself confirm or refute MECH-471 -- `experiment_purpose = "diagnostic"`,
excluded from claim confidence/conflict scoring by construction; `evidence_direction` is
"unknown" in every branch below.

RE-DERIVE BRAKE (Step 2.5b): does not fire. MECH-471's own autopsy count is 0
`substrate_ceiling`-category autopsies (875 = precondition_unmet, 875a =
competence_implementation_gap; per WORKSPACE_STATE.md 2026-08-05T16:49Z). This is also not a
same-question re-test of 875/875a's own falsifier -- it is a genuinely new instrument aimed at
a different (dependency) question the two prior runs were never designed to answer.

GOV-FANOUT-1 (portfolio-vs-single-probe): does not apply. The brake has not fired and no
upstream `/failure-autopsy` emitted a formal `fanout_recommendation` for this qid -- the
hypothesis-space registry's own `decision.live_gate` names exactly ONE instrument ("a cheap,
many-seed diagnostic probe ... to characterize the seed-success-rate distribution and look
for an early-death or hazard-layout correlation"), not a K-leg portfolio, so a single script
is the correct scope.

GOV-REUSE-1 (Step 2.4, existing-evidence check). Decisive readouts for this probe: (a) a
per-seed acquisition-EPISODE-resolution (not per-50/100/250/500-exposure-restart) survival
trajectory fine enough to test whether divergence is already present at the EARLIEST
episodes; (b) per-seed random-walk-vs-acquired-competence correlation (H2); (c) per-episode
bias_head/OFC-devaluation-head parameter-norm introspection (H3). Checked against V3-EXQ-875a
(3 seeds, no fine trajectory, no H2/H3 introspection) and V3-EXQ-882a (8 seeds; DOES already
carry a coarse 4-point [50,100,250,500]-episode in-context dose-response per seed plus
per-seed random_walk_anchors -- genuinely informative for H1/H2 at that resolution, and
already used that way in failure_autopsy_V3-EXQ-882a_2026-08-05.md Section 5/7 as
"informative context, not a formal resolution"). Neither manifest carries readout (a) at
<50-episode resolution or readout (c) at all -- NOT RECOVERABLE for those two; must run. This
probe does not attempt to reuse 882a's cells (different checkpoint schedule, no clean cache
hit for the decisive fine-resolution readout) and mints its own arms from scratch.

SUBSTRATE (Step 2.5/2.5a): TESTABLE NOW, complicated (buildable) status confirmed elsewhere,
not gated here. The SD-070 z_world P0a warmup, SD-056 e2 P0b warmup, and the two-head
REINFORCE bias substrate (lateral-PFC action bias + OFC devaluation bias) are the SAME
substrate V3-EXQ-728/875a/882a already exercised to completion (non-degenerate results,
even where inconclusive). No new architecture. Reused verbatim: `experiments._lib.
allon_training` recipe constants/losses/optimizers (vendored copy below, see "WHY VENDORED"),
`experiments._lib.zworld_p0_warmup.run_zworld_p0`, `experiments.v3_exq_724_...` env kwargs
and config-assembly helpers.

ENV: the SAME "corner" reef-geometry survival/avoidance env as V3-EXQ-875 EXP-0399's variant
"A" / V3-EXQ-882a's `_env_kwargs()` -- 3 reef safe-zone patches at fixed grid corners (legacy
`_place_reef_patches`), scattered point hazards, `hazard_food_attraction=0.0` decoupling
foraging from the survival DV. Reused verbatim, not a new configuration.

DESIGN. Runs ONLY the acquisition stage per seed -- SD-056 e2 P0 warmup, then SD-070 z_world
P0a warmup, then a SINGLE CONTINUOUS P1 REINFORCE run (no restart, no Phase-2 targeted
update, no held-out-context evaluation). Every seed gets exactly ONE
`_train_all_on_agent_traced()` call (below), so REINFORCE's outcome-buffer/EMA-baseline and
the e2/lPFC/OFC-devaluation Adam optimizers carry continuous state across the FULL P1 budget
-- bit-identical training semantics to V3-EXQ-875a/882a's own single-call convention (no
chunked-restart artefact).

  P0b (e2 warmup, P0_WARMUP_EPISODES=60) -> P0a (SD-070 z_world warmup on a DEDICATED env,
  ZWORLD_P0_EPISODES=60; ordering inside `_train_all_on_agent`/`_train_all_on_agent_traced`
  runs P0a before P0b internally, matching 875a/882a) -> P1 REINFORCE for
  P1_TOTAL_EPISODES=160 continuous episodes (CHECKPOINT_INTERVAL=20, N_CHECKPOINTS=8).

TRAJECTORY, WITHOUT EXTRA EVAL CALLS (WHY VENDORED). `experiments._lib.allon_training.
_train_all_on_agent` returns only aggregate counts (n_p0_ticks/n_p1_ticks/
n_e2_train_steps/zworld_p0), not a per-episode record -- adequate for the dose-response-by-
restart designs (875a/882a) but not for THIS probe's job (characterizing divergence TIMING
within one continuous run). Injecting separate eval() calls mid-training would consume
additional torch RNG draws (eval also samples through `agent.select_action`), perturbing the
very training trajectory being measured. So `_train_all_on_agent_traced()` below is a
deliberately VENDORED, COMPUTATION-IDENTICAL copy of the shared function (same precedent as
V3-EXQ-728b's independent copy, documented in allon_training.py's own "DELIBERATELY NOT
COLLAPSED" section: "code that is specific to one driver ... stays in that driver"). The ONLY
additions, both marked "TRACED ADDITION" inline, are READ-ONLY bookkeeping appended after
values the original loop already computes -- no RNG draw, no optimizer/loss/env-stepping
change:
  1. a per-P1-episode `trajectory[]` entry (ticks survived this episode, episode reward,
     whether the episode ended in death before the step cap);
  2. per-P1-episode `lpfc_bias_head_norm` / `ofc_deval_head_norm` (sum of `.norm()` over
     `agent.lateral_pfc.bias_head_parameters()` / `agent.ofc.devaluation_bias_head_
     parameters()`) -- pure parameter introspection, zero extra env steps, the H3 signal.
This script's `arm_cell(..., include_driver_script_in_hash=True)` (the default) correctly
folds this vendored copy into THIS experiment's OWN substrate hash as "the calling script's
own content" -- there is no cross-driver baseline-reuse intent here (this is a bespoke
characterization probe, not a lineage baseline mint), so the under-inclusion hazard that
motivated moving `_train_all_on_agent` INTO `experiments/_lib/` in the first place
(allon_training.py's own docstring) does not apply to this vendoring.

CHECKPOINTS ARE TRAILING-WINDOW MEANS OF RAW TRAINING-EPISODE TICKS, NOT SEPARATE EVALS.
Checkpoint k (k=1..N_CHECKPOINTS, at cumulative P1 episode k*CHECKPOINT_INTERVAL) reports the
mean `tick_in_ep` over the trailing TRAILING_WINDOW P1 episodes ending there. This is a
deliberate choice for an ACQUISITION-only probe (unlike 882a's in-context-vs-held-out design,
generalization is out of scope here): raw training-episode ticks are read off the SAME
layout-drawing process the agent is actually learning under, which is the right lens for
"is this seed learning, and when." A single random-walk anchor (RandomPolicy, same seed, one
evaluate_seed call, no training-loop interaction) is computed once per seed as the
task-difficulty-only reference for H2.

PRIMARY READOUT (informational, per-seed): final_windowed_survival_ticks (mean tick_in_ep
over the LAST TRAILING_WINDOW P1 episodes) vs SURVIVAL_FLOOR_FRAC*steps AND a margin over the
seed's own random-walk anchor -- "cleared" / "not cleared", mirroring 875a/882a's own
acquisition-floor convention exactly (SURVIVAL_FLOOR_FRAC=0.6, RANDOM_MARGIN_FRAC=0.15).

DIAGNOSTIC ADJUDICATION (this run's own PASS/FAIL is about whether the PROBE achieved a
usable discriminating sample, NOT about which hypothesis wins -- the hypothesis reading is a
separate, explicitly-informational `hypothesis_signal` block):
  combination_rule (sequential precondition chain, not a flat AND):
    1. `final_checkpoint_activity_nondegenerate` (readiness; measured via check_degeneracy,
       floor=TRIVIAL_ACTIVITY_FLOOR_FRAC*steps): if EVERY seed is pinned near-zero at the
       final checkpoint, nothing learned across the whole probe -> a genuine substrate/
       instrumentation non-readiness, NOT a scientific finding about MECH-471's hypotheses.
       unmet -> outcome=FAIL, label=substrate_not_ready_requeue, non_degenerate=False.
       WOULD MEAN: the acquisition recipe itself is broken or mis-wired on this substrate
       snapshot (re-check the SD-070/SD-056 wiring before trusting anything else here).
       WOULD NOT MEAN: anything about H1/H2/H3.
    2. `bimodal_split_present_for_discrimination` (load_bearing; measured =
       min(n_cleared, n_not_cleared), threshold=MIN_SPLIT_GROUP_SIZE): THIS IS NOT A
       SUBSTRATE-READINESS GATE -- it is the target phenomenon itself (does a bimodal split
       exist in this n=16 sample at all). Routing an unmet reading here to
       substrate_not_ready_requeue would be a category error (a below-floor LEARNED quantity
       indicating an untrained substrate is not the same thing as "the substrate learned
       fine, and this sample happens not to split"), so it gets its own label instead.
       unmet -> outcome=FAIL, label=insufficient_bimodal_split_for_discrimination,
       non_degenerate=False.
       WOULD MEAN: at n=16 and this budget, the split from 875a/882a either did not
       reproduce, or reproduced too lopsidedly (e.g. 15/16 or 1/16) to support a group
       comparison -- itself a real, reportable finding (informs whether the true failure
       rate is much higher/lower than the ~25-33% seen at n=3/n=8), just not a discriminating
       one for H1 vs H2 vs H3.
       WOULD NOT MEAN: the probe or substrate is broken.
    3. else -> outcome=PASS, non_degenerate=True, interpretation.label names the divergence-
       timing finding from the `hypothesis_signal.h1_early_divergence` block (see below) --
       PASS here means "the probe produced a usable bimodal sample," not "H1 confirmed."

HYPOTHESIS SIGNAL (informational; each sub-block declares what it would and would not mean):
  h1_early_divergence -- group_gap[checkpoint] = mean(cleared group) - mean(not-cleared group)
    at each checkpoint. early_frac = group_gap[1] / group_gap[N_CHECKPOINTS] (guarded: if
    |group_gap[N_CHECKPOINTS]| < MIN_FINAL_GROUP_GAP_FRAC*steps, reported as
    divergence_timing_ambiguous / early_frac=None -- too small a final gap to read a ratio
    off). early_frac >= EARLY_DIVERGENCE_FRAC_FLOOR (0.7): "early_divergence_supports_h1_
    framing" -- WOULD MEAN most of the eventual gap is already present after the very first
    CHECKPOINT_INTERVAL episodes and does not need the rest of the budget to appear, matching
    H1's "stuck early, never escapes" prediction; WOULD NOT MEAN H2/H3 are ruled out (a hard
    early hazard-layout draw could ALSO show up this way; see h2 below). early_frac <=
    LATE_DIVERGENCE_FRAC_CEIL (0.3): "late_or_gradual_divergence_disfavors_h1_framing" --
    WOULD MEAN the eventual failing seeds looked comparable to the succeeding seeds early on
    and only fell behind later/gradually, which the H1 label's own "early, unescapable"
    framing does not predict (this is the reading V3-EXQ-882a's own flat 10x-exposure trend
    already weighed toward, per failure_autopsy_V3-EXQ-882a_2026-08-05.md Section 5, but
    could not test directly at episode resolution); WOULD NOT MEAN H1 is fully eliminated
    (late-onset entrapment in a different local optimum is not what the registered H1 label
    describes, but is not excluded by this probe's design either -- report as a residual
    open question if this reading recurs). Otherwise: "divergence_timing_ambiguous".
  h2_hazard_layout_difficulty -- group-mean random_walk_survival_ticks for the cleared vs
    not-cleared groups; diff_frac = (mean(cleared) - mean(not_cleared)) / steps.
    diff_frac >= H2_NOTABLE_DIFF_FRAC (0.10): "difficulty_correlate_plausible" -- WOULD MEAN
    the seeds that go on to fail acquisition ALSO drew a harder (lower random-walk survival)
    task before any learning happened, consistent with H2's task-difficulty-variance framing;
    WOULD NOT MEAN acquisition/exploration variance (H1) or drive interaction (H3) played no
    role -- a genuine difficulty differential and a learning-dynamics problem are not mutually
    exclusive. Else: "difficulty_correlate_not_evident" -- WOULD MEAN the seeds that fail
    acquisition were not measurably harder BEFORE training under a policy that has learned
    nothing at all, which argues against pure task-difficulty as the sole explanation (though
    the untrained random-walk anchor may not capture every way a layout could be "hard" for a
    partially-trained policy specifically); WOULD NOT MEAN H2 is formally eliminated.
    Exploratory / non-load-bearing: this is a group-mean comparison at n<=16 per group, not a
    powered statistical test, and does not gate `outcome`.
  h3_drive_arbitration_exploratory -- per-checkpoint group-mean lpfc_bias_head_norm /
    ofc_deval_head_norm for the cleared vs not-cleared groups, reported as raw trajectories
    with NO pre-registered threshold (explicitly the weakest-power, most exploratory of the
    three -- "if that's cheaply available from existing agent introspection" per the
    instrument's own brief). A visibly diverging or degenerate (e.g. saturating, oscillating)
    norm trajectory in the not-cleared group relative to the cleared group would be
    SUGGESTIVE, not adjudicating, corroboration for H3; similar trajectories across both
    groups would argue (not prove) against H3 mattering here. No routing decision reads this
    block; it is recorded for a future discriminating design to build on if it looks
    promising.

evidence_direction_per_claim is NOT populated -- claim_ids has exactly one entry (MECH-471),
and this diagnostic's evidence_direction is "unknown" in every branch regardless (it does not
test MECH-471's own registered falsifier).

ethics_preflight:
  involves_negative_valence: false
  involves_suffering_like_state: false
  involves_self_model: false
  involves_inescapability_or_helplessness: false
  involves_offline_replay_over_harm: false
  involves_social_mind_or_language: false
  involves_human_data_or_clinical_context: false
  decision: allow

SLEEP DRIVER: none (no sleep loop; use_sleep_loop / sws_enabled / rem_enabled all OFF).

This module is ASCII-only in all runtime strings.
"""

from __future__ import annotations

import argparse
import random
import statistics
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment_protocol import emit_outcome  # noqa: E402
from experiments._lib import allon_training as _allon  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._lib.capability_eval import RandomPolicy, evaluate_seed  # noqa: E402
from experiments._lib.manifest_core import stamp_recording_core  # noqa: E402
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402
from experiments._lib.zworld_p0_warmup import run_zworld_p0  # noqa: E402
from experiments._metrics import check_degeneracy  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
import experiments.v3_exq_724_competence_localization_diagnostic as x724  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_890_mech471_acquisition_reliability_probe"
QUEUE_ID = "V3-EXQ-890"
CLAIM_IDS: List[str] = ["MECH-471"]
EXPERIMENT_PURPOSE = "diagnostic"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"

# validate_experiments.py anchor-reachability lint: no defect on either flagged precondition.
# `final_checkpoint_activity_nondegenerate` IS the degeneracy definition itself (any seed with
# minimal random movement clears TRIVIAL_ACTIVITY_FLOOR_FRAC*steps -- reachable by construction
# under any functioning substrate; a replay against a positive control would be tautological).
# `bimodal_split_present_for_discrimination` does not route to a substrate-verdict label at all
# (see combination_rule / module docstring): its unmet branch is
# insufficient_bimodal_split_for_discrimination, a bespoke non-substrate label chosen
# specifically to avoid the mislabeling this lint guards against -- it is the target
# phenomenon under characterization, not a readiness anchor, so it has no known-positive
# control to reproduce.
ANCHOR_REACHABILITY_EXEMPT = (
    "both flagged preconditions are non-substrate-verdict checks: "
    "final_checkpoint_activity_nondegenerate is the degeneracy definition (reachable by "
    "construction), and bimodal_split_present_for_discrimination routes to the bespoke "
    "label insufficient_bimodal_split_for_discrimination, never to a substrate-verdict "
    "label -- see module docstring 'combination_rule'."
)

# --- Budget (real run) -- deliberately CHEAPER than V3-EXQ-875a/882a: single continuous P1
# run per seed (no Phase-2 targeted update, no held-out eval, no per-exposure-level restart).
# 16 seeds continues 875a's [42,43,45] and 882a's [42,43,45,46,47,48,49,50] (8 of the 16) for
# direct per-seed comparability, plus 8 genuinely new seeds for unbiased distributional
# coverage. Seed 44 skipped per repo convention (recurring early-death instability). --------
SEEDS: List[int] = [42, 43, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58]
ZWORLD_P0_EPISODES = 60           # SD-070 validated operating point (728/734/742/875a/882a).
P0_WARMUP_EPISODES = 60           # SD-056 e2 forward-model warmup, ONCE per seed (A0 recipe).
P1_TOTAL_EPISODES = 160           # single continuous REINFORCE run (no restart).
CHECKPOINT_INTERVAL = 20          # cumulative P1 episodes per checkpoint (8 checkpoints).
N_CHECKPOINTS = P1_TOTAL_EPISODES // CHECKPOINT_INTERVAL
TRAILING_WINDOW = 10              # trailing-window size for each checkpoint's smoothed mean.
STEPS_PER_EPISODE = 150           # matches V3-EXQ-882a's env-variant-A recipe.
RANDOM_WALK_EVAL_EPISODES = 8     # random-walk anchor: one evaluate_seed call, no training.

# --- Dry-run budget ---------------------------------------------------------------------
DRY_SEEDS: List[int] = [42, 43]
DRY_ZWORLD_P0 = 2
DRY_P0 = 2
DRY_CHECKPOINT_INTERVAL = 4
DRY_N_CHECKPOINTS = 2
DRY_P1_TOTAL = DRY_CHECKPOINT_INTERVAL * DRY_N_CHECKPOINTS
DRY_TRAILING_WINDOW = 2
DRY_STEPS = 15
DRY_RANDOM_WALK_EVAL = 2

# --- Pre-registered thresholds (declared as fractions of STEPS_PER_EPISODE; never derived
# from the run's own statistics) ----------------------------------------------------------
SURVIVAL_FLOOR_FRAC = 0.6              # acquired competence must survive >= 60% of the
                                        # episode at the final checkpoint (mirrors 875a/882a).
RANDOM_MARGIN_FRAC = 0.15              # ... AND beat the seed's own random-walk anchor by
                                        # >= 15% of the episode length.
TRIVIAL_ACTIVITY_FLOOR_FRAC = 0.03     # readiness floor (Step 1 of combination_rule): if
                                        # EVERY seed's final checkpoint is at or below this,
                                        # nothing learned across the whole probe.
MIN_SPLIT_GROUP_SIZE = 2               # need >= 2 cleared AND >= 2 not-cleared for a genuine
                                        # bimodal sample (Step 2 of combination_rule).
EARLY_DIVERGENCE_FRAC_FLOOR = 0.7      # >= 70% of the final group gap already present at
                                        # checkpoint 1 -> "early divergence" (H1 framing).
LATE_DIVERGENCE_FRAC_CEIL = 0.3        # <= 30% of the final group gap present at checkpoint 1
                                        # -> "late/gradual divergence" (disfavors H1 framing).
MIN_FINAL_GROUP_GAP_FRAC = 0.10        # final-checkpoint group gap must clear this (as a
                                        # fraction of steps) before an early/final ratio is
                                        # read at all -- guards a near-zero-denominator ratio.
H2_NOTABLE_DIFF_FRAC = 0.10            # exploratory-only: group-mean random-walk survival
                                        # difference (as a fraction of steps) reported as
                                        # "difficulty_correlate_plausible" above this.


def _env_kwargs() -> Dict[str, Any]:
    """The SAME "corner" reef-geometry survival/avoidance env as V3-EXQ-875 EXP-0399's
    variant "A" / V3-EXQ-882a's `_env_kwargs()`. Reused verbatim; not a new configuration."""
    kw = dict(x724.ENV_KWARGS)
    kw["hazard_food_attraction"] = 0.0
    kw["reef_enabled"] = True
    kw["n_reef_patches"] = 3
    kw["reef_bipartite_layout"] = False   # legacy corner-pattern reef placement
    return kw


def _make_env(seed: int, env_kwargs: Dict[str, Any]) -> CausalGridWorldV2:
    return CausalGridWorldV2(seed=seed, **env_kwargs)


def _make_agent(env: CausalGridWorldV2) -> REEAgent:
    """The all-ON REE stack (724/734/742/875a/882a recipe). No new architecture."""
    kwargs = x724._base_config_kwargs(env)
    kwargs.update(x724._all_on_extra_kwargs())
    cfg = REEConfig.from_dims(**kwargs)
    return REEAgent(cfg)


def _eval_random(env_kwargs: Dict[str, Any], seed: int, eval_eps: int, steps: int) -> Dict[str, Any]:
    env = _make_env(seed, env_kwargs)
    return evaluate_seed(RandomPolicy(seed), env, eval_eps, steps)


# ---------------------------------------------------------------------------------------
# VENDORED, COMPUTATION-IDENTICAL copy of experiments._lib.allon_training._train_all_on_agent.
# See module docstring "TRAJECTORY, WITHOUT EXTRA EVAL CALLS (WHY VENDORED)" for the full
# justification. Every line of computation below (optimizers, losses, env-stepping order,
# REINFORCE update) is copied verbatim from allon_training.py; lines marked
# "TRACED ADDITION" are the only additions, and are all read-only bookkeeping appended after
# values the original loop already computed -- no RNG draw, no computation change.
# ---------------------------------------------------------------------------------------
def _train_all_on_agent_traced(
    agent: REEAgent,
    train_env: CausalGridWorldV2,
    seed: int,
    p0_episodes: int,
    p1_episodes: int,
    steps_per_episode: int,
    rung_id: str,
    total_denominator: int,
    zworld_p0_episodes: int = 0,
    zworld_p0_env: Optional[CausalGridWorldV2] = None,
) -> Dict[str, Any]:
    env = train_env

    zworld_p0_stats: Dict[str, Any] = {"p0a_recipe": "sd070", "p0a_ran": False}
    if zworld_p0_episodes > 0:
        if zworld_p0_env is None:
            raise ValueError(
                "zworld_p0_episodes=%d requires zworld_p0_env: the warmup rollout consumes "
                "env RNG, so reusing train_env would shift the layout sequence P0b/P1 then "
                "see. Build a dedicated env with the same seed and kwargs."
                % (zworld_p0_episodes,)
            )
        zworld_p0_stats = run_zworld_p0(
            agent, zworld_p0_env, seed, zworld_p0_episodes, steps_per_episode,
            policy=RandomPolicy(seed), label=f"ree_allon rung={rung_id}",
        )
    has_ofc = getattr(agent, "ofc", None) is not None
    has_lpfc = getattr(agent, "lateral_pfc", None) is not None

    e2_opt = torch.optim.Adam(agent.e2.parameters(), lr=_allon.E2_CONTRASTIVE_LR)
    bias_opt = (
        torch.optim.Adam(list(agent.lateral_pfc.bias_head_parameters()), lr=_allon.LR_LPFC_BIAS)
        if has_lpfc else None
    )
    ofc_deval_opt = (
        torch.optim.Adam(list(agent.ofc.devaluation_bias_head_parameters()), lr=_allon.LR_OFC_DEVAL)
        if has_ofc else None
    )
    transition_buffer: Deque[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ] = deque(maxlen=_allon.TRANSITION_BUFFER_MAX)
    sample_rng = random.Random(seed)

    total_train_eps = p0_episodes + p1_episodes
    p1_start = p0_episodes
    n_p0_ticks = 0
    n_p1_ticks = 0
    n_e2_train_steps = 0

    reinforce_baseline = 0.0
    outcome_buf: List[Tuple[torch.Tensor, int, float]] = []
    trajectory: List[Dict[str, Any]] = []   # TRACED ADDITION

    for ep in range(total_train_eps):
        is_p1 = (ep >= p1_start)
        is_p0 = not is_p1
        phase_label = "P1" if is_p1 else "P0"

        _flat, obs_dict = env.reset()
        agent.reset()

        z_self_prev: Optional[torch.Tensor] = None
        action_prev: Optional[torch.Tensor] = None
        pending_capture: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        tick_in_ep = 0
        done = False   # TRACED ADDITION: defensive init only (guards a stale read from the
                        # PREVIOUS episode if this episode's inner loop breaks before its
                        # first env.step(); never read by the original computation below).

        ep_reward = 0.0
        ep_buf: List[Tuple[torch.Tensor, int]] = []

        for _step in range(steps_per_episode):
            body = obs_dict["body_state"].float()
            world = obs_dict["world_state"].float()
            if body.dim() == 1:
                body = body.unsqueeze(0)
            if world.dim() == 1:
                world = world.unsqueeze(0)

            latent = agent.sense(
                obs_body=body, obs_world=world,
                obs_harm=_allon._obs_harm(obs_dict),
                obs_harm_a=_allon._obs_harm_a(obs_dict),
                obs_harm_history=_allon._obs_harm_history(obs_dict),
            )

            if pending_capture is not None:
                z0_prev, a_prev = pending_capture
                z1_obs = latent.z_world.detach().reshape(-1).clone()
                if (
                    torch.isfinite(z0_prev).all()
                    and torch.isfinite(a_prev).all()
                    and torch.isfinite(z1_obs).all()
                ):
                    transition_buffer.append((z0_prev, a_prev, z1_obs))
                pending_capture = None

            if z_self_prev is not None and action_prev is not None:
                agent.record_transition(z_self_prev, action_prev, latent.z_self.detach())

            ticks = agent.clock.advance()
            wdim = latent.z_world.shape[-1]
            e1_prior = (
                agent._e1_tick(latent) if ticks.get("e1_tick", False)
                else torch.zeros(1, wdim, device=agent.device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)

            p1_snap_summaries: Optional[torch.Tensor] = None
            if is_p1 and has_lpfc and candidates and len(candidates) >= 2:
                cs = _allon._consumed_summaries(agent, candidates)
                if cs is not None and torch.isfinite(cs).all():
                    p1_snap_summaries = cs.clone()

            action = agent.select_action(candidates, ticks)
            if action is None:
                idx = int(np.random.randint(0, env.action_dim))
                action = torch.zeros(1, env.action_dim, device=agent.device)
                action[0, idx] = 1.0
                agent._last_action = action
            if not torch.isfinite(action).all():
                break

            committed_class = int(action[0].argmax().item())

            if is_p1 and p1_snap_summaries is not None:
                sel = 0
                for ci, c in enumerate(candidates):
                    if (
                        getattr(c, "actions", None) is not None
                        and c.actions.shape[1] >= 1
                        and int(c.actions[:, 0, :].argmax(-1).reshape(-1)[0].item())
                        == committed_class
                    ):
                        sel = min(ci, p1_snap_summaries.shape[0] - 1)
                        break
                ep_buf.append((p1_snap_summaries, sel))

            if is_p1:
                n_p1_ticks += 1
            else:
                n_p0_ticks += 1

            if torch.isfinite(latent.z_world).all() and torch.isfinite(action).all():
                pending_capture = (
                    latent.z_world.detach().reshape(-1).clone(),
                    action.detach().reshape(-1).clone(),
                )

            train_e2_now = is_p0 or (is_p1 and _allon.E2_TRAIN_IN_P1)
            if train_e2_now and (tick_in_ep % _allon.E2_TRAIN_EVERY_K_TICKS == 0):
                if _allon._e2_contrastive_step(agent, transition_buffer, e2_opt, sample_rng) is not None:
                    n_e2_train_steps += 1

            _flat, _harm_signal, done, info, obs_dict = env.step(action)
            harm_signal = float(_harm_signal)
            if is_p1:
                ep_reward += harm_signal

            with torch.no_grad():
                agent.update_residue(
                    harm_signal=harm_signal, world_delta=None,
                    hypothesis_tag=False, owned=True,
                )
            if agent.goal_state is not None:
                benefit_exposure = float(info.get("benefit_exposure", 0.0))
                energy = float(body[0, 3].item())
                agent.update_z_goal(
                    benefit_exposure=benefit_exposure,
                    drive_level=max(0.0, 1.0 - energy),
                )

            z_self_prev = latent.z_self.detach()
            action_prev = action.detach()
            tick_in_ep += 1
            if done:
                break

        if is_p1 and (has_lpfc or has_ofc):
            reinforce_baseline = (
                _allon.EMA_DECAY * reinforce_baseline + (1.0 - _allon.EMA_DECAY) * ep_reward
            )
            for cand_features, sel in ep_buf:
                outcome_buf.append((cand_features, sel, ep_reward))
            if len(outcome_buf) > _allon.OUTCOME_BUF_MAX:
                outcome_buf = outcome_buf[-_allon.OUTCOME_BUF_MAX:]
            if has_lpfc and bias_opt is not None:
                l_loss = _allon._lpfc_reinforce_loss(agent, outcome_buf, reinforce_baseline, agent.device)
                if l_loss.requires_grad:
                    bias_opt.zero_grad()
                    l_loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.lateral_pfc.bias_head_parameters(), 1.0)
                    bias_opt.step()
            if has_ofc and ofc_deval_opt is not None:
                ofc_loss = _allon._ofc_deval_reinforce_loss(agent, outcome_buf, reinforce_baseline, agent.device)
                if ofc_loss.requires_grad:
                    ofc_deval_opt.zero_grad()
                    ofc_loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.ofc.devaluation_bias_head_parameters(), 1.0)
                    ofc_deval_opt.step()

        # TRACED ADDITION: per-P1-episode trajectory + weight-norm bookkeeping. Read-only;
        # appended after every value it reads has already been computed by the code above.
        if is_p1:
            lpfc_norm = (
                float(sum(p.detach().norm().item() for p in agent.lateral_pfc.bias_head_parameters()))
                if has_lpfc else None
            )
            ofc_norm = (
                float(sum(p.detach().norm().item() for p in agent.ofc.devaluation_bias_head_parameters()))
                if has_ofc else None
            )
            trajectory.append({
                "p1_episode_index": int(ep - p1_start + 1),
                "tick_in_ep": int(tick_in_ep),
                "ep_reward": round(float(ep_reward), 6),
                "died_before_step_cap": bool(done and tick_in_ep < steps_per_episode),
                "lpfc_bias_head_norm": (round(lpfc_norm, 6) if lpfc_norm is not None else None),
                "ofc_deval_head_norm": (round(ofc_norm, 6) if ofc_norm is not None else None),
            })

        cur = ep + 1
        if cur % 50 == 0 or cur == total_train_eps or phase_label == "P1":
            print(
                f"  [train] ree_allon rung={rung_id} seed={seed} phase={phase_label} "
                f"ep {cur}/{total_denominator}",
                flush=True,
            )

    return {
        "n_p0_ticks": int(n_p0_ticks),
        "n_p1_ticks": int(n_p1_ticks),
        "n_e2_train_steps": int(n_e2_train_steps),
        "zworld_p0": zworld_p0_stats,
        "trajectory": trajectory,   # TRACED ADDITION
    }


def _windowed_mean(trajectory: List[Dict[str, Any]], key: str, end_ep: int, window: int) -> float:
    vals = [
        float(row[key]) for row in trajectory
        if end_ep - window < row["p1_episode_index"] <= end_ep and row[key] is not None
    ]
    return float(statistics.fmean(vals)) if vals else 0.0


def _run_seed(
    seed: int, kw: Dict[str, Any], p0: int, zworld_p0: int, p1_total: int,
    checkpoint_interval: int, n_checkpoints: int, trailing_window: int,
    random_walk_eval_eps: int, steps: int, zg: ZGoalStreamAccumulator,
) -> Dict[str, Any]:
    env_train = _make_env(seed, kw)
    agent = _make_agent(env_train)
    zworld_p0_env = _make_env(seed, kw) if zworld_p0 > 0 else None

    train_result = _train_all_on_agent_traced(
        agent, env_train, seed, p0_episodes=p0, p1_episodes=p1_total,
        steps_per_episode=steps, rung_id="acquisition_probe",
        total_denominator=(p0 + p1_total), zworld_p0_episodes=zworld_p0,
        zworld_p0_env=zworld_p0_env,
    )
    zg.observe(agent)
    trajectory = train_result["trajectory"]

    checkpoints: List[Dict[str, Any]] = []
    for k in range(1, n_checkpoints + 1):
        end_ep = k * checkpoint_interval
        checkpoints.append({
            "checkpoint": k,
            "cumulative_p1_episodes": end_ep,
            "windowed_survival_ticks": round(_windowed_mean(trajectory, "tick_in_ep", end_ep, trailing_window), 4),
            "windowed_ep_reward": round(_windowed_mean(trajectory, "ep_reward", end_ep, trailing_window), 6),
            "windowed_lpfc_bias_head_norm": round(_windowed_mean(trajectory, "lpfc_bias_head_norm", end_ep, trailing_window), 6),
            "windowed_ofc_deval_head_norm": round(_windowed_mean(trajectory, "ofc_deval_head_norm", end_ep, trailing_window), 6),
        })

    random_walk = _eval_random(kw, seed, random_walk_eval_eps, steps)
    final_survival = checkpoints[-1]["windowed_survival_ticks"] if checkpoints else 0.0
    rw_survival = float(random_walk["survival_horizon"])

    return {
        "seed": int(seed),
        "trajectory": trajectory,
        "checkpoints": checkpoints,
        "final_windowed_survival_ticks": round(final_survival, 4),
        "random_walk_survival_ticks": round(rw_survival, 4),
        "random_margin": round(final_survival - rw_survival, 4),
        "n_p0_ticks": train_result["n_p0_ticks"],
        "n_p1_ticks": train_result["n_p1_ticks"],
        "n_e2_train_steps": train_result["n_e2_train_steps"],
        "zworld_p0_stats": train_result["zworld_p0"],
    }


def run_experiment(
    seeds: List[int], p0: int, zworld_p0: int, p1_total: int, checkpoint_interval: int,
    n_checkpoints: int, trailing_window: int, random_walk_eval_eps: int, steps: int,
) -> Dict[str, Any]:
    kw = _env_kwargs()
    steps_f = float(steps)
    survival_floor = SURVIVAL_FLOOR_FRAC * steps_f
    random_margin_floor = RANDOM_MARGIN_FRAC * steps_f
    trivial_activity_floor = TRIVIAL_ACTIVITY_FLOOR_FRAC * steps_f
    min_final_group_gap = MIN_FINAL_GROUP_GAP_FRAC * steps_f

    print(
        f"MECH-471 acquisition-reliability probe ({len(seeds)} seeds; zworld_p0={zworld_p0} "
        f"P0={p0} P1={p1_total} checkpoints={n_checkpoints}x{checkpoint_interval} steps={steps})",
        flush=True,
    )

    zg = ZGoalStreamAccumulator()
    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        print(f"Seed {seed} Condition acquisition_probe", flush=True)
        slice_cfg = {
            "kind": "mech471_acquisition_reliability_probe",
            "env_kwargs": dict(kw),
            "zworld_p0_episodes": int(zworld_p0),
            "p0_warmup_episodes": int(p0),
            "p1_total_episodes": int(p1_total),
            "checkpoint_interval": int(checkpoint_interval),
            "n_checkpoints": int(n_checkpoints),
            "trailing_window": int(trailing_window),
            "steps_per_episode": int(steps),
        }
        with arm_cell(seed, config_slice=slice_cfg, script_path=Path(__file__), config_slice_declared=True) as cell:
            row = _run_seed(
                seed, kw, p0, zworld_p0, p1_total, checkpoint_interval, n_checkpoints,
                trailing_window, random_walk_eval_eps, steps, zg,
            )
            cell.stamp(row)
        rows.append(row)

        final = row["final_windowed_survival_ticks"]
        margin = row["random_margin"]
        cleared = bool(final >= survival_floor and margin >= random_margin_floor)
        row["cleared_floor"] = cleared
        print(
            f"verdict: {'PASS' if cleared else 'FAIL'} (seed={seed} "
            f"final_survival={final} random_walk={row['random_walk_survival_ticks']} "
            f"margin={margin} cleared={cleared})",
            flush=True,
        )

    final_values = [r["final_windowed_survival_ticks"] for r in rows]
    cleared_rows = [r for r in rows if r["cleared_floor"]]
    not_cleared_rows = [r for r in rows if not r["cleared_floor"]]
    n_cleared = len(cleared_rows)
    n_not_cleared = len(not_cleared_rows)

    degeneracy = check_degeneracy({
        "final_windowed_survival_ticks_all_seeds": {
            "values": final_values, "floor": trivial_activity_floor,
        },
    })
    activity_ok = bool(degeneracy["non_degenerate"])
    bimodal_split_ok = bool(n_cleared >= MIN_SPLIT_GROUP_SIZE and n_not_cleared >= MIN_SPLIT_GROUP_SIZE)

    degeneracy_reason = degeneracy["degeneracy_reason"]
    if activity_ok and not bimodal_split_ok and not degeneracy_reason:
        degeneracy_reason = (
            "bimodal split absent for discrimination: %d cleared / %d not-cleared of %d seeds "
            "(need >=%d on each side)" % (n_cleared, n_not_cleared, len(rows), MIN_SPLIT_GROUP_SIZE)
        )

    # ---- hypothesis_signal (informational; computed only when a bimodal sample exists) ----
    hypothesis_signal: Dict[str, Any] = {}
    if bimodal_split_ok:
        def _group_checkpoint_mean(group_rows: List[Dict[str, Any]], k: int) -> float:
            vals = [r["checkpoints"][k - 1]["windowed_survival_ticks"] for r in group_rows]
            return float(statistics.fmean(vals)) if vals else 0.0

        group_gap = {
            k: round(_group_checkpoint_mean(cleared_rows, k) - _group_checkpoint_mean(not_cleared_rows, k), 4)
            for k in range(1, n_checkpoints + 1)
        }
        early_gap = group_gap[1]
        final_gap = group_gap[n_checkpoints]

        if abs(final_gap) < min_final_group_gap:
            divergence_label = "divergence_timing_ambiguous"
            early_frac: Optional[float] = None
        else:
            early_frac = round(early_gap / final_gap, 4)
            if (early_gap >= 0) != (final_gap >= 0):
                divergence_label = "divergence_timing_ambiguous"
            elif early_frac >= EARLY_DIVERGENCE_FRAC_FLOOR:
                divergence_label = "early_divergence_supports_h1_framing"
            elif early_frac <= LATE_DIVERGENCE_FRAC_CEIL:
                divergence_label = "late_or_gradual_divergence_disfavors_h1_framing"
            else:
                divergence_label = "divergence_timing_ambiguous"

        hypothesis_signal["h1_early_divergence"] = {
            "label": divergence_label,
            "group_gap_by_checkpoint": group_gap,
            "early_gap": early_gap,
            "final_gap": final_gap,
            "early_frac": early_frac,
            "min_final_group_gap_ticks": round(min_final_group_gap, 4),
        }

        rw_cleared = [r["random_walk_survival_ticks"] for r in cleared_rows]
        rw_not_cleared = [r["random_walk_survival_ticks"] for r in not_cleared_rows]
        rw_cleared_mean = float(statistics.fmean(rw_cleared)) if rw_cleared else 0.0
        rw_not_cleared_mean = float(statistics.fmean(rw_not_cleared)) if rw_not_cleared else 0.0
        h2_diff = rw_cleared_mean - rw_not_cleared_mean
        h2_diff_frac = round(h2_diff / steps_f, 4)
        h2_label = (
            "difficulty_correlate_plausible" if h2_diff_frac >= H2_NOTABLE_DIFF_FRAC
            else "difficulty_correlate_not_evident"
        )
        hypothesis_signal["h2_hazard_layout_difficulty"] = {
            "label": h2_label,
            "random_walk_survival_mean_cleared_group": round(rw_cleared_mean, 4),
            "random_walk_survival_mean_not_cleared_group": round(rw_not_cleared_mean, 4),
            "diff_ticks": round(h2_diff, 4),
            "diff_frac": h2_diff_frac,
        }

        h3_by_checkpoint = []
        for k in range(1, n_checkpoints + 1):
            lpfc_cleared = [r["checkpoints"][k - 1]["windowed_lpfc_bias_head_norm"] for r in cleared_rows]
            lpfc_not_cleared = [r["checkpoints"][k - 1]["windowed_lpfc_bias_head_norm"] for r in not_cleared_rows]
            ofc_cleared = [r["checkpoints"][k - 1]["windowed_ofc_deval_head_norm"] for r in cleared_rows]
            ofc_not_cleared = [r["checkpoints"][k - 1]["windowed_ofc_deval_head_norm"] for r in not_cleared_rows]
            h3_by_checkpoint.append({
                "checkpoint": k,
                "lpfc_bias_head_norm_mean_cleared_group": round(statistics.fmean(lpfc_cleared), 6) if lpfc_cleared else None,
                "lpfc_bias_head_norm_mean_not_cleared_group": round(statistics.fmean(lpfc_not_cleared), 6) if lpfc_not_cleared else None,
                "ofc_deval_head_norm_mean_cleared_group": round(statistics.fmean(ofc_cleared), 6) if ofc_cleared else None,
                "ofc_deval_head_norm_mean_not_cleared_group": round(statistics.fmean(ofc_not_cleared), 6) if ofc_not_cleared else None,
            })
        hypothesis_signal["h3_drive_arbitration_exploratory"] = {
            "note": "exploratory only; no pre-registered threshold; does not gate outcome.",
            "by_checkpoint": h3_by_checkpoint,
        }

    non_degenerate = bool(activity_ok and bimodal_split_ok)

    if not activity_ok:
        outcome, direction, label = "FAIL", "unknown", "substrate_not_ready_requeue"
    elif not bimodal_split_ok:
        outcome, direction, label = "FAIL", "unknown", "insufficient_bimodal_split_for_discrimination"
    else:
        outcome, direction = "PASS", "unknown"
        label = hypothesis_signal["h1_early_divergence"]["label"]

    return {
        "outcome": outcome,
        "evidence_direction": direction,
        "interpretation_label": label,
        "combination_rule": (
            "sequential precondition chain: (1) final_checkpoint_activity_nondegenerate "
            "(substrate readiness; unmet -> substrate_not_ready_requeue) -> (2) "
            "bimodal_split_present_for_discrimination (the target phenomenon itself, NOT a "
            "substrate-readiness gate; unmet -> insufficient_bimodal_split_for_discrimination) "
            "-> (3) else PASS, label = h1_early_divergence's own divergence-timing finding."
        ),
        "interpretation": {
            "label": label,
            "preconditions": [
                {
                    "name": "final_checkpoint_activity_nondegenerate",
                    "description": "not every seed is pinned near-zero at the final checkpoint (trivial floor-artefact / substrate-readiness check)",
                    "measured": round(max(final_values), 4) if final_values else 0.0,
                    "threshold": round(trivial_activity_floor, 4),
                    "direction": "lower",
                    "control": "cross-seed maximum of final_windowed_survival_ticks",
                    "met": activity_ok,
                },
                {
                    "name": "bimodal_split_present_for_discrimination",
                    "description": "at least MIN_SPLIT_GROUP_SIZE seeds clear the acquisition floor AND at least MIN_SPLIT_GROUP_SIZE do not, at the final checkpoint -- the target phenomenon under characterization, NOT a substrate-readiness control",
                    "measured": min(n_cleared, n_not_cleared),
                    "threshold": MIN_SPLIT_GROUP_SIZE,
                    "direction": "lower",
                    "control": "n/a -- see description",
                    "met": bimodal_split_ok,
                },
            ],
            "criteria_non_degenerate": {
                "bimodal_split_present_for_discrimination": bimodal_split_ok,
                "final_checkpoint_activity_nondegenerate": activity_ok,
            },
        },
        "criteria": [
            {"name": "bimodal_split_present_for_discrimination", "load_bearing": True, "passed": bimodal_split_ok},
        ],
        "readiness": {
            "activity_ok": activity_ok,
            "bimodal_split_ok": bimodal_split_ok,
            "n_seeds": len(rows),
            "n_cleared": n_cleared,
            "n_not_cleared": n_not_cleared,
            "cleared_seeds": sorted(int(r["seed"]) for r in cleared_rows),
            "not_cleared_seeds": sorted(int(r["seed"]) for r in not_cleared_rows),
            "survival_floor_ticks": round(survival_floor, 4),
            "random_margin_floor_ticks": round(random_margin_floor, 4),
            "trivial_activity_floor_ticks": round(trivial_activity_floor, 4),
        },
        "hypothesis_signal": hypothesis_signal,
        "arm_results": rows,
        "non_degenerate": non_degenerate,
        "degeneracy_reason": degeneracy_reason,
        "degenerate_metrics": degeneracy["degenerate_metrics"],
        "z_goal_stream_stats": zg.stats(),
    }


def _build_manifest(result: Dict[str, Any], timestamp_utc: str, dry_run: bool,
                    cfg: Dict[str, Any]) -> Dict[str, Any]:
    run_id = f"{EXPERIMENT_TYPE}_{timestamp_utc}_v3"
    return {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "queue_id": QUEUE_ID,
        "timestamp_utc": timestamp_utc,
        "dry_run": bool(dry_run),
        "outcome": result["outcome"],
        "evidence_direction": result["evidence_direction"],
        "interpretation_label": result["interpretation_label"],
        "combination_rule": result["combination_rule"],
        "interpretation": result["interpretation"],
        "criteria": result["criteria"],
        "readiness": result["readiness"],
        "hypothesis_signal": result["hypothesis_signal"],
        "arm_results": result["arm_results"],
        "non_degenerate": result["non_degenerate"],
        "degeneracy_reason": result["degeneracy_reason"],
        "degenerate_metrics": result["degenerate_metrics"],
        "sleep_driver_pattern": "none",
        "config": cfg,
        "load_bearing_dv": (
            "bimodal_split_present_for_discrimination = whether >=MIN_SPLIT_GROUP_SIZE seeds "
            "clear the acquisition floor (final_windowed_survival_ticks >= survival_floor AND "
            "margin over the seed's own random-walk anchor >= random_margin_floor) AND "
            ">=MIN_SPLIT_GROUP_SIZE do not, at n=16. This is a DIAGNOSTIC readout about "
            "whether the known bimodal split (875a n=3, 882a n=8) reproduces at a larger "
            "sample under a cheap single-continuous-run acquisition budget -- it does not "
            "itself confirm or refute MECH-471's own registered falsifier (the local-update "
            "interference test). The substantive hypothesis reading (H1/H2/H3) is carried in "
            "hypothesis_signal and is explicitly informational, non-adjudicating."
        ),
        "notes": (
            "V3-EXQ-890: cheap many-seed acquisition-only diagnostic probe for the "
            "hypothesis-space registry question mech471_competence_acquisition_reliability "
            "(registered 2026-08-05T06:20:43Z, live_gate recommended this exact instrument, "
            "priority raised same day per failure_autopsy_V3-EXQ-882a_2026-08-05.md Section "
            "7/8). Reuses the V3-EXQ-875/882a 'corner' env variant A and acquisition recipe "
            "verbatim; skips Phase-2/held-out entirely; single continuous P1 run per seed "
            "(no chunked restart) via a vendored, computation-identical copy of "
            "experiments._lib.allon_training._train_all_on_agent with read-only per-episode "
            "trajectory + bias/OFC-devaluation weight-norm bookkeeping added (see module "
            "docstring 'WHY VENDORED'). 16 seeds (8 overlapping 882a's for direct "
            "comparison, 8 new for unbiased distributional coverage)."
        ),
    }


def main() -> Tuple[Optional[str], Optional[str], bool]:
    parser = argparse.ArgumentParser(
        description="V3-EXQ-890 MECH-471 cheap many-seed acquisition-reliability diagnostic probe"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--zworld-p0", type=int, default=None)
    parser.add_argument("--p0", type=int, default=None)
    parser.add_argument("--p1-total", type=int, default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=None)
    parser.add_argument("--trailing-window", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()
    started = datetime.now(timezone.utc)
    t0 = time.perf_counter()

    if args.dry_run:
        seeds = list(DRY_SEEDS)
        zworld_p0 = DRY_ZWORLD_P0
        p0 = DRY_P0
        p1_total = DRY_P1_TOTAL
        checkpoint_interval = DRY_CHECKPOINT_INTERVAL
        trailing_window = DRY_TRAILING_WINDOW
        steps = DRY_STEPS
        random_walk_eval_eps = DRY_RANDOM_WALK_EVAL
    else:
        seeds = list(SEEDS)
        zworld_p0 = ZWORLD_P0_EPISODES
        p0 = P0_WARMUP_EPISODES
        p1_total = P1_TOTAL_EPISODES
        checkpoint_interval = CHECKPOINT_INTERVAL
        trailing_window = TRAILING_WINDOW
        steps = STEPS_PER_EPISODE
        random_walk_eval_eps = RANDOM_WALK_EVAL_EPISODES

    if args.seeds:
        seeds = [int(s) for s in args.seeds]
    if args.zworld_p0 is not None:
        zworld_p0 = int(args.zworld_p0)
    if args.p0 is not None:
        p0 = int(args.p0)
    if args.p1_total is not None:
        p1_total = int(args.p1_total)
    if args.checkpoint_interval is not None:
        checkpoint_interval = int(args.checkpoint_interval)
    if args.trailing_window is not None:
        trailing_window = int(args.trailing_window)
    if args.steps is not None:
        steps = int(args.steps)

    if p1_total % checkpoint_interval != 0:
        raise ValueError("p1_total must be an exact multiple of checkpoint_interval")
    n_checkpoints = p1_total // checkpoint_interval

    result = run_experiment(
        seeds=seeds, p0=p0, zworld_p0=zworld_p0, p1_total=p1_total,
        checkpoint_interval=checkpoint_interval, n_checkpoints=n_checkpoints,
        trailing_window=trailing_window, random_walk_eval_eps=random_walk_eval_eps, steps=steps,
    )

    timestamp_utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    cfg = {
        "seeds": seeds,
        "zworld_p0_episodes": zworld_p0,
        "p0_warmup_episodes": p0,
        "p1_total_episodes": p1_total,
        "checkpoint_interval": checkpoint_interval,
        "n_checkpoints": n_checkpoints,
        "trailing_window": trailing_window,
        "random_walk_eval_episodes": random_walk_eval_eps,
        "steps_per_episode": steps,
        "env_kwargs": _env_kwargs(),
        "survival_floor_frac": SURVIVAL_FLOOR_FRAC,
        "random_margin_frac": RANDOM_MARGIN_FRAC,
        "trivial_activity_floor_frac": TRIVIAL_ACTIVITY_FLOOR_FRAC,
        "min_split_group_size": MIN_SPLIT_GROUP_SIZE,
        "early_divergence_frac_floor": EARLY_DIVERGENCE_FRAC_FLOOR,
        "late_divergence_frac_ceil": LATE_DIVERGENCE_FRAC_CEIL,
        "min_final_group_gap_frac": MIN_FINAL_GROUP_GAP_FRAC,
        "h2_notable_diff_frac": H2_NOTABLE_DIFF_FRAC,
    }
    manifest = _build_manifest(result, timestamp_utc, dry_run=bool(args.dry_run), cfg=cfg)
    stamp_recording_core(
        manifest, config=cfg, seeds=seeds, script_path=Path(__file__), started_at=t0,
        z_goal_stream_stats=result.get("z_goal_stream_stats"),
    )

    out_dir = Path(args.out_dir) if args.out_dir is not None else (
        REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    )
    out_path = write_flat_manifest(
        manifest, out_dir, dry_run=args.dry_run, config=cfg, seeds=seeds,
        script_path=Path(__file__),
        elapsed_seconds=(datetime.now(timezone.utc) - started).total_seconds(),
        stamp=False,  # already stamped above (need z_goal_stream_stats folded in first)
        z_goal_stream_stats=result.get("z_goal_stream_stats"),
    )

    print(f"manifest: {out_path}", flush=True)
    print(
        f"outcome: {result['outcome']} label={result['interpretation_label']} "
        f"non_degenerate={result['non_degenerate']} "
        f"n_cleared={result['readiness']['n_cleared']} "
        f"n_not_cleared={result['readiness']['n_not_cleared']}",
        flush=True,
    )
    if result["hypothesis_signal"]:
        print(
            f"  h1={result['hypothesis_signal']['h1_early_divergence']['label']} "
            f"h2={result['hypothesis_signal']['h2_hazard_layout_difficulty']['label']}",
            flush=True,
        )

    if args.dry_run:
        try:
            out_path.unlink()
        except FileNotFoundError:
            pass

    outcome_norm = str(result["outcome"]).upper()
    outcome_emit = outcome_norm if outcome_norm in ("PASS", "FAIL") else "FAIL"
    manifest_for_sentinel = str(out_path) if not args.dry_run else None
    return outcome_emit, manifest_for_sentinel, bool(args.dry_run)


if __name__ == "__main__":
    _outcome, _manifest_path, _dry_run = main()
    if _outcome is not None:
        emit_outcome(outcome=_outcome, manifest_path=_manifest_path, dry_run=_dry_run)
