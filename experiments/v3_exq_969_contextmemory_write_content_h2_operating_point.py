#!/opt/local/bin/python3
"""
V3-EXQ-969 -- ContextMemory write-address selection: H2-operating-point leg of
the `contextmemory_write_content_discrimination` fan-out portfolio.

Chip: chip-20260830-ctxmem-write-content-h1h4-portfolio

experiment_purpose: diagnostic

hypothesis_space qid: contextmemory_write_content_discrimination
hypothesis_space hid: H2-operating-point

NULL HYPOTHESIS (this leg tests, and only this leg): w=0.5 is the wrong
operating point; NO weight in the tested neighborhood yields content
discrimination in the correct direction at a Bonferroni-corrected
significance level; 956's wrong-direction move is real. This experiment does
NOT try to explain WHY the loss fails (that is H1/H3/H4's job) -- it only
establishes WHETHER any weight in a reasonable neighborhood of the one
V3-EXQ-956 tested does better.

WHY THIS RUN EXISTS. V3-EXQ-956 (diagnostic, claim-free) trained
`write_addr_tagger` at ONE operating point
(`contextmemory_write_addressing_loss_weight=0.5`, `write_gumbel_anneal_steps`
left at its 2000 default, 100 episodes x 150 steps, 5 seeds
[42,7,13,100,200]) and measured mean 2-cluster probe Jaccard
TRAINED=0.667 vs UNTRAINED=0.400 -- i.e. it MOVED IN THE WRONG DIRECTION
(worse, not better). 956's own autopsy-strength calibration said that
wrong-direction move is only "suggestive, not established" (exact paired
sign-flip/permutation p ~ 0.24-0.28 on 5 bimodal per-seed Jaccards; a
40-episode moderate-scale pre-check during 956's authoring showed the SAME
wrong-direction pattern, mean Jaccard 0.4 untrained -> 0.9 trained). This
experiment is the FIRST leg of the frozen `contextmemory_write_content_
discrimination` fan-out portfolio and asks whether a DIFFERENT operating
weight escapes that wrong-direction result.

WHY THE ANNEAL-SCHEDULE AXIS WAS REMOVED (design-time finding, not a run
result). An earlier design of this script varied BOTH weight AND the Gumbel
anneal schedule (`contextmemory_write_gumbel_anneal_steps`) in Phase B. That
second axis was removed before any cell ran, because it has no causal path
to the trained loss: `write_addr_tagger`'s ONLY gradient source is
`ContextMemory.compute_write_addressing_loss(states)`
(ree_core/predictors/e1_deep.py), a pairwise-cosine loss over
`softmax(-tagger(states))` that never reads the anneal temperature or step
counter. The anneal schedule only affects `_select_write_slot_gumbel`'s
WRITE-TIME slot routing, inside `write()`'s own `torch.no_grad()` block --
and the post-training synthetic probe (`_run_content_probe` below) runs in
`eval()` mode, which is deterministic argmin over write_addr_tagger's scores
and also never reads tau. So varying the anneal schedule cannot change what
the tagger LEARNS, and cannot change the deterministic post-training probe
either -- it is not a real second manipulation axis, only an RNG replicate
of the same weight dressed up as a design variable. The compute that would
have gone to a 2nd anneal setting is spent on 3 more seeds per weight
instead (see PHASE B below): same total cell count, genuine added power on
the axis that actually has a causal path to the DV.

DESIGN -- TWO PHASES IN ONE SCRIPT.

PHASE A -- seed-power check at 956's own operating point (w=0.5, the fixed
`ANNEAL_DEFAULT`=2000-step anneal, unchanged throughout this script),
MODERATE scale (40 episodes -- the same "moderate-scale" length 956's own
docstring used for its own pre-check; a deliberate cost/power tradeoff, NOT
toy-scale). Arms: GUMBEL_UNTRAINED (weight=0.0) and GUMBEL_TRAINED_W0P5
(weight=0.5) -- same two arms as 956's own GUMBEL_UNTRAINED/GUMBEL_TRAINED,
at the SAME weight, but with 10 NEW seeds ([11, 23, 31, 47, 53, 61, 79, 83,
97, 101] -- distinct from 956's [42,7,13,100,200], deliberately avoiding 44
per this repo's CLAUDE.md seed-44-instability note even though this driver's
env is CausalGridWorldV2, not a reef config, so the substitute costs
nothing) so this ADDS statistical power rather than repeating 956's own 5
seeds. Seed IS the pairing key here: seed S's GUMBEL_UNTRAINED cell and seed
S's GUMBEL_TRAINED_W0P5 cell are a natural pair, so the per-seed sign of
(trained_jaccard[S] - untrained_jaccard[S]) is reported as a descriptive
(non-gating) count of how many of the 10 seeds move in the wrong direction
(positive delta) vs the correct one (negative delta), matching 956's own
finding. Phase A's own PAIRED significance test (see ACCEPTANCE CRITERIA
below) uses this same pairing.

PHASE B -- weight sweep, MODERATE scale (40 episodes), all at the fixed
ANNEAL_DEFAULT=2000 anneal (see "WHY THE ANNEAL-SCHEDULE AXIS WAS REMOVED"
above -- no anneal variable anywhere in this script). Weights {0.1, 2.0,
8.0} (0.5 is Phase A's job, not re-tested here), 6 seeds each ([42, 7, 13,
100, 200, 300] -- 956's own 5 plus one new one, 300) = 18 GUMBEL_TRAINED
cells, same total cell count as the earlier weight x anneal design this
replaces (3 weights x 2 anneal settings x 3 seeds = 18), now spent entirely
on seeds. Phase B's untrained comparison baseline is Phase A's own
GUMBEL_UNTRAINED n=10 pool (same episode count, same env/config, different
seeds) -- reused directly as a Python value already computed earlier in the
same run() call, NOT a fresh arm and NOT the arm_reuse_fingerprint machinery
(no try_reuse_cell call here -- see GOV-REUSE-1 note below for why).

GOV-REUSE-1 (Step 2.4, existing-evidence check): the decisive readout (mean
2-cluster Jaccard at weight in {0.1,2.0,8.0}, or at w=0.5 with n=10 fresh
seeds) is not recorded anywhere -- 956 measured only w=0.5/anneal=2000/n=5/
100-episodes. Not recoverable; proceeding to run. Also note: 956's own cells
were minted WITHOUT include_driver_script_in_hash=False, so this script's
Phase A w=0.5 cells CANNOT arm-reuse 956's cells even though the config
nominally overlaps (fingerprint refuses on differing script_path in the
hash, and also on differing training_episodes: 40 here vs 100 there) -- this
is expected and fine, no try_reuse_cell is attempted; every cell in both
phases runs fresh.

ACCEPTANCE CRITERIA.

P0 READINESS (writepath genuinely engaged in every one of the 38 cells --
20 Phase A + 18 Phase B -- exactly 956's own gate, scaled): every cell must
record n_write_calls >= WRITE_CALLS_FLOOR_MODERATE (60.0). Derivation: 40
episodes is 40% of 956's 100-episode schedule; 956's own P0 floor was 200 at
100 episodes, so a directly proportional floor would be ~80 -- 60 is used
instead to leave margin for episode-to-episode variance in a shorter
schedule while still being a genuine non-trivial floor (not merely a nominal
one). On precondition failure, the WHOLE run self-routes to
`substrate_not_ready_requeue`, exactly 956's own P0NotReady handling.

C2-STYLE READINESS (gates the interpretability of EVERY Phase A / Phase B
directional comparison, since they all share ONE untrained baseline):
mean_jaccard(GUMBEL_UNTRAINED, Phase A, n=10) must be >= C2_JACCARD_MARGIN
(0.25) -- otherwise the baseline is already close to the Jaccard=0 floor and
"materially lower" becomes structurally unsatisfiable regardless of what any
operating point does, which would silently misread a floor effect as "no
operating point works." Below-floor reads as `precondition_unmet`, never a
plain FAIL of the load-bearing criterion.

LOAD-BEARING CRITERION -- H2_any_operating_point_improves_content_
discrimination, and WHY IT IS A PERMUTATION TEST, NOT A RAW MEAN-MARGIN
CHECK. The DV (post-training synthetic-probe Jaccard) is near-binary per
seed (956's real data for its untrained arm: [0.0, 1.0, 1.0, 0.0, 0.0]) --
with a DV shaped like that, a raw "mean_trained <= mean_untrained - 0.25"
test, run independently across multiple configs with an OR combination and
no multiplicity correction, has a materially high chance of a spurious PASS
from noise alone (measured design-time risk: 25-68% false-positive rate
under the null, across the configs and n this script uses). So the
load-bearing criterion is instead: PASS iff the C2-style readiness
precondition above is met AND at least one of the N_CONFIGS_TESTED=4 tested
configs (Phase A's w=0.5/n=10, or any of Phase B's 3 weight configs) ALL OF
(a) achieves an observed_diff (mean_jaccard_untrained_pool -
mean_jaccard_trained_config) >= C2_JACCARD_MARGIN (0.25, the original
real-effect-size floor -- kept, not dropped), (b) clears an EXACT
permutation-test p_value < ALPHA_CORRECTED (0.05 / N_CONFIGS_TESTED =
0.0125, Bonferroni-corrected across the 4 configs actually tested), and (c)
that config's write_addr_tagger genuinely received gradient (state_dict
moved from init across every one of that config's seeds -- see
criteria_non_degenerate below; a "PASS" whose winning config's tagger never
moved is a bug reading, not a real discovery). Phase A's test is a PAIRED
exact sign-flip permutation test over its 10 seed-matched deltas (seed IS
the pairing key -- see PHASE A above); Phase B's 3 weight configs each use
an UNPAIRED exact two-sample permutation test against Phase A's n=10
untrained pool (Phase B's 6 seeds are not the same seeds as Phase A's, so
there is no natural pairing). Both are exact enumeration (2**10 = 1024
sign-flips for Phase A; C(16,6) = 8008 relabelings for each Phase B config),
not Monte Carlo, at this script's actual sample sizes -- see
`_permutation_test_pvalue` / `_paired_sign_flip_pvalue` docstrings for the
Monte Carlo fallback this script does not currently exercise. This is the
leg's actual null-test: PASS = an operating point was found that improves
content discrimination in the correct direction, with an effect size that
clears the floor AND is unlikely to be noise at the corrected significance
level, AND whose tagger demonstrably trained; FAIL (with the readiness
precondition met) = the null holds -- no operating point in this
neighborhood escapes 956's wrong-direction result at this power. This is a
genuinely open, falsifiable question and a FAIL here is a real,
scientifically valid, expected-possible outcome, not a bug. The OLD raw
mean-margin-only reading (no significance test, no multiplicity correction)
is still recorded per config as a separate, explicitly non-gating
descriptive field (`raw_margin_reading`, `load_bearing: false`) so a reader
can see both readings side by side.

NON-GATING DESCRIPTIVE CRITERIA: Phase A's paired-sign count (how many of 10
seeds show the wrong-direction move, matching 956's own finding); a
per-config table of (weight, mean_jaccard_trained,
delta_from_untrained_baseline, observed_diff, p_value, permutation_method,
cleared_margin, cleared_significance, tagger_moved, counts_as_discovery,
raw_margin_reading) across all 4 tested configs; and the same before/after
write_addr_tagger state_dict movement negative controls 956 used
(GUMBEL_UNTRAINED expected frozen, every trained config expected to move),
reported in aggregate across all trained cells and per-config.

criteria_non_degenerate: H2_any_operating_point_improves_content_
discrimination reads True only if BOTH phases ran with their full seed
counts (no early truncation) AND the C2-style readiness precondition was met
AND -- if the load-bearing criterion PASSed -- the winning config's
write_addr_tagger actually moved (a PASS whose winning config's tagger never
received gradient is a bug reading and must not read as non_degenerate:
true). Never True if the whole run self-routed to substrate_not_ready_
requeue (that path returns criteria_non_degenerate={} entirely, matching
956's own P0-failure shape).

red-team (fable), 2026-09-02: CONTESTED -> two findings.
  G1 (criterion cannot discriminate by construction, DISMISSED with this
  caveat rather than fixed): at this DV's near-binary per-seed shape, the
  Bonferroni-corrected permutation test (alpha=0.0125 across 4 configs) is
  arithmetically unreachable for a baseline mean_jaccard_untrained in
  roughly [0.25, 0.65) even for a perfect-separation trained result --
  956's own measured untrained baseline (0.4) sits inside that zone. A FAIL
  landing there would be indistinguishable, in its `null_holds` label alone,
  from a genuine null. NOT fixed here: correcting the attainability gap
  requires re-deriving the seed/config compute budget (materially more
  seeds per config), which is a bigger scientific-design change than this
  pass's remit, and this is leg 1 of 4 in the portfolio -- if H2 comes back
  a bare FAIL, `H2_per_config_table`'s recorded per-config `observed_diff`
  and `p_value` (kept regardless of outcome) let a human reviewer check
  post-hoc whether the observed baseline sat in the unattainable zone
  before reading the null as informative; this is disclosed explicitly so
  that check is not skipped. G2 (a gate certifies its own subject) --
  FIXED: `criteria_non_degenerate` checked only `winning_config_tagger_
  moved`, vacuously True on every FAIL (no winning config to invalidate),
  so a silent wiring failure (tagger never receiving gradient in ANY
  trained cell) could present as a valid null. Fixed by also requiring the
  aggregate positive control (`n_trained_moved == len(trained_rows)`) --
  see the `criteria_non_degenerate` dict construction below.

SUBSTRATE PROPERTIES held constant across all arms (not part of the
manipulation) -- identical to V3-EXQ-956's own list (itself identical to
V3-EXQ-943's), reused verbatim so this experiment's arms are the same
measurement family up to RNG/weight:
  - contextmemory_gated_content_write=True (436d write-path repair).
  - sd016_writepath_mode="sense_only".
  - alpha_world=0.9 (SD-008).
  - use_noise_floor=True (MECH-313/ARC-065).
  - context_memory.memory.requires_grad_(False) after construction (436e/f
    Adam-drift neutralization). Does NOT freeze write_addr_tagger -- that
    module's parameters are deliberately left trainable; only self.memory
    (the CONTENT tensor mutated by write()'s own .data write) is frozen
    against optimizer drift.
  - use_per_stream_vs=True, use_anchor_sets=True, use_sd039_anchor_payload=True.
  - use_salience_coordinator left at its REEConfig default (False).
  - use_lateral_pfc_analog left at its REEConfig default (False). See
    "Step 2.5c" below.
  - sd016_enabled left at its REEConfig default (False) -- gates only the
    cue-tagger/context-divergence-loss apparatus, irrelevant here.
  - contextmemory_write_gumbel_anneal_steps=ANNEAL_DEFAULT (2000) for every
    single cell in both phases -- see "WHY THE ANNEAL-SCHEDULE AXIS WAS
    REMOVED" above; this is a held-constant substrate property here, not a
    manipulation.

No sleep. sleep_driver_pattern: "N/A (no sleep loop)". Identical rationale
to V3-EXQ-943/956: SD-016 sleep-cycle consolidation is an orthogonal
question from whether the write-ADDRESS mechanism itself achieves
content-conditioning at some operating point.

Step 2.5c substrate-path overlap (re-checked against the CURRENT
substrate_queue.json, 2026-09-01, not merely cited from V3-EXQ-956): open
entries naming `ree_core/agent.py` at whole-file granularity with status
containing "_pending_validation" are `mode-governance-engagement`
(severity=corrupting) and `SD-082` (severity=corrupting, whole-file agent.py
+ `ree_core/pfc/lateral_pfc_analog.py::compute_bias`). Both removed by
construction, identical disposition to 956's own mode-governance-engagement
check: this driver never sets use_salience_coordinator=True (REEConfig's own
False default is used throughout, gating mode-governance-engagement's
SalienceCoordinator) and never sets use_lateral_pfc_analog=True (REEConfig's
own False default is used throughout, gating SD-082's LateralPFCAnalog
construction in ree_core/agent.py -- confirmed at
`getattr(config, "use_lateral_pfc_analog", False)`, ree_core/agent.py:898).
`modulatory-bias-selection-authority` also names agent.py but carries status
"implemented" (not "_pending_validation") -- closed, non-blocking.
`contextmemory-write-path-addressing-degeneracy` itself (e1_deep.py) is the
entry under test here, not an unrelated one. No other open `corrupting`
entry's substrate_paths overlaps any module this driver imports or
exercises.

Step 2.5b re-derive brake: N/A. claim_ids is empty (see below).

claim_ids: [] -- this experiment tests substrate readiness for the
write-address FIX itself (specifically: does ANY weight operating point of
the redesigned pairwise-diversity addressing loss produce measurable
content-discrimination), not a claim hypothesis (SD-017 /
ARC-045 / MECH-166, the sleep-differentiation claims this substrate_queue
entry unblocks once validated). Diagnostic purpose experiments with
claim_ids=[] are excluded from governance confidence scoring by design.

Does NOT flip substrate_queue status regardless of outcome -- that stays a
human/governance disposition. This script only appends a validation_record
once it has actually run.

ethics_preflight:
  involves_negative_valence: false
  involves_suffering_like_state: false
  involves_self_model: false
  involves_inescapability_or_helplessness: false
  involves_offline_replay_over_harm: false
  involves_social_mind_or_language: false
  involves_human_data_or_clinical_context: false
  decision: allow
"""

import argparse
import hashlib
import itertools
import math
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig
from experiment_protocol import emit_outcome  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402
from experiments._metrics import p0_readiness_gate, P0NotReady  # noqa: E402


EXPERIMENT_TYPE = "v3_exq_969_contextmemory_write_content_h2_operating_point"
QUEUE_ID = "V3-EXQ-969"
CLAIM_IDS: List[str] = []
EXPERIMENT_PURPOSE = "diagnostic"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"

# --- Anneal schedule (FIXED, not a manipulated axis) ---------------------- #
# The anneal-schedule axis was investigated at design time and dropped --
# write_addr_tagger's only gradient source (ContextMemory.compute_write_
# addressing_loss) never reads the anneal temperature/step counter, and the
# post-training probe is deterministic argmin (also reads no tau). See
# module docstring "WHY THE ANNEAL-SCHEDULE AXIS WAS REMOVED". Every cell in
# both phases uses this one fixed value -- there is no anneal sweep variable
# anywhere in this script.
ANNEAL_DEFAULT = 2000  # V3-EXQ-956's own default (ContextMemory's own default)

# --- Phase A: seed-power check at 956's own operating point -------------- #
# (label, write_selection, weight, anneal_steps)
PHASE_A_ARM_SPECS: List[Tuple[str, str, float, int]] = [
    ("GUMBEL_UNTRAINED", "gumbel_learned", 0.0, ANNEAL_DEFAULT),
    ("GUMBEL_TRAINED_W0P5", "gumbel_learned", 0.5, ANNEAL_DEFAULT),
]
PHASE_A_SEEDS: List[int] = [11, 23, 31, 47, 53, 61, 79, 83, 97, 101]

# --- Phase B: weight sweep (anneal axis removed -- see module docstring) - #
PHASE_B_WEIGHTS: List[float] = [0.1, 2.0, 8.0]
PHASE_B_SEEDS: List[int] = [42, 7, 13, 100, 200, 300]  # 956's own 5 + one new


def _weight_label(w: float) -> str:
    # 0.1 -> "W0P1", 2.0 -> "W2P0", 8.0 -> "W8P0"
    s = f"{w:.1f}".replace(".", "P")
    return f"W{s}"


PHASE_B_ARM_SPECS: List[Tuple[str, str, float, int]] = [
    (f"GUMBEL_TRAINED_{_weight_label(w)}", "gumbel_learned", w, ANNEAL_DEFAULT)
    for w in PHASE_B_WEIGHTS
]

ALL_ARM_SPECS: List[Tuple[str, str, float, int]] = PHASE_A_ARM_SPECS + PHASE_B_ARM_SPECS
ALL_ARM_NAMES: List[str] = [a[0] for a in ALL_ARM_SPECS]

# --- F1 fix: exact-permutation-test / multiplicity-correction constants --- #
N_CONFIGS_TESTED = 4  # Phase A's w=0.5 + Phase B's 3 weight configs
ALPHA = 0.05
ALPHA_CORRECTED = ALPHA / N_CONFIGS_TESTED  # Bonferroni-corrected, 0.0125

# Substrate properties held constant across all arms (not the manipulation).
CONTEXTMEMORY_GATED_CONTENT_WRITE = True
SD016_WRITEPATH_MODE = "sense_only"
ALPHA_WORLD = 0.9
USE_NOISE_FLOOR = True
NOISE_FLOOR_ALPHA = 0.1
NOISE_FLOOR_MIN_TEMPERATURE = 1.0
BASELINE_TEMPERATURE = 1.0

TRAINING_EPISODES = 40  # moderate scale -- see module docstring
STEPS_PER_EPISODE = 150
CONTEXT_SWITCH_EVERY = 5
MAX_HARM_BUF = 4000

NUM_SLOTS = 16  # ContextMemory default
LATENT_DIM = 64  # self_dim(32) + world_dim(32) -- must match the contract
                 # test's own LATENT_DIM for the post-training probe to be a
                 # like-for-like measurement.

# P0 readiness floor -- scaled from V3-EXQ-956's 200-at-100-episodes floor
# for this run's 40-episode schedule. See module docstring "P0 READINESS"
# for the derivation (proportional would be ~80; 60 leaves margin).
WRITE_CALLS_FLOOR_MODERATE = 60.0

# C2-style content-discrimination readiness + load-bearing margin (shared).
C2_JACCARD_MARGIN = 0.25

# Post-training synthetic 2-cluster probe (reuses the contract test's own
# instrument -- test_contextmemory_write_address_selection.py's _stream).
PROBE_N = 1500
PROBE_JITTER = 0.0078
PROBE_CLUSTERS = 2

TAGGER_MOVE_EPS = 1e-8  # threshold for "did write_addr_tagger's state_dict move"


# ------------------------------------------------------------------ #
# Env / agent helpers (env params reused verbatim from V3-EXQ-943/956)     #
# ------------------------------------------------------------------ #

def _make_env_safe(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed,
        size=10,
        num_hazards=1,
        num_resources=4,
        hazard_harm=0.02,
        env_drift_interval=50,
        env_drift_prob=0.05,
        proximity_harm_scale=0.10,
        proximity_benefit_scale=0.18,
        proximity_approach_threshold=0.15,
        hazard_field_decay=0.5,
        energy_decay=0.005,
        use_proxy_fields=True,
        resource_respawn_on_consume=True,
    )


def _make_env_dangerous(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed + 1000,
        size=10,
        num_hazards=8,
        num_resources=4,
        hazard_harm=0.05,
        env_drift_interval=50,
        env_drift_prob=0.05,
        proximity_harm_scale=0.15,
        proximity_benefit_scale=0.18,
        proximity_approach_threshold=0.15,
        hazard_field_decay=0.5,
        energy_decay=0.005,
        use_proxy_fields=True,
        resource_respawn_on_consume=True,
    )


def _make_agent(env: CausalGridWorldV2, write_selection: str,
                 write_addressing_loss_weight: float,
                 gumbel_anneal_steps: int) -> REEAgent:
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=32,
        alpha_world=ALPHA_WORLD,
        alpha_self=0.3,
        reafference_action_dim=0,
        novelty_bonus_weight=0.0,
        sd016_writepath_mode=SD016_WRITEPATH_MODE,
        use_per_stream_vs=True,
        use_anchor_sets=True,
        use_sd039_anchor_payload=True,
        contextmemory_gated_content_write=CONTEXTMEMORY_GATED_CONTENT_WRITE,
        contextmemory_write_selection=write_selection,
        contextmemory_write_addressing_loss_weight=write_addressing_loss_weight,
        contextmemory_write_gumbel_anneal_steps=gumbel_anneal_steps,
        use_noise_floor=USE_NOISE_FLOOR,
        noise_floor_alpha=NOISE_FLOOR_ALPHA,
        noise_floor_min_temperature=NOISE_FLOOR_MIN_TEMPERATURE,
    )
    agent = REEAgent(cfg)
    assert agent.noise_floor is not None, (
        "use_noise_floor=True did not construct agent.noise_floor -- "
        "REEConfig/REEAgent wiring regression."
    )
    assert agent.e1.context_memory.num_slots == NUM_SLOTS, (
        f"ContextMemory num_slots changed from the assumed {NUM_SLOTS} -- "
        "update NUM_SLOTS before trusting occupancy-floor thresholds."
    )
    total_latent_dim = agent.e1.config.self_dim + agent.e1.config.world_dim
    assert total_latent_dim == LATENT_DIM, (
        f"self_dim+world_dim={total_latent_dim} != LATENT_DIM={LATENT_DIM} -- "
        "the post-training synthetic probe would no longer match the "
        "contract test's own instrument; update LATENT_DIM before trusting "
        "the content-discrimination measurement."
    )
    assert agent.e1.context_memory.write_addr_tagger is not None, (
        "write_selection='gumbel_learned' did not construct write_addr_tagger "
        "-- config wiring regression."
    )
    assert agent.e1.context_memory.write_gumbel_anneal_steps == gumbel_anneal_steps, (
        f"ContextMemory.write_gumbel_anneal_steps="
        f"{agent.e1.context_memory.write_gumbel_anneal_steps} != requested "
        f"{gumbel_anneal_steps} -- contextmemory_write_gumbel_anneal_steps "
        "wiring regression."
    )
    # 436e/f Adam-drift neutralization (see module docstring). write() still
    # mutates memory.data directly under its own no_grad block regardless.
    # write_addr_tagger is deliberately NOT frozen -- it is this mechanism's
    # own trainable parameter, the entire subject of the manipulation.
    agent.e1.context_memory.memory.requires_grad_(False)
    return agent


def _action_onehot(a_idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, a_idx] = 1.0
    return v


def _effective_temperature(agent: REEAgent) -> float:
    if agent.noise_floor is not None:
        return agent.noise_floor.compute_effective_temperature(
            baseline_temperature=BASELINE_TEMPERATURE, simulation_mode=False,
        )
    return BASELINE_TEMPERATURE


def _select_action_baseline(agent: REEAgent, z_world: torch.Tensor,
                             num_actions: int) -> int:
    """Temperature-graded softmax sample over predicted harm (low harm -> high
    selection probability), using the MECH-313 noise-floor effective
    temperature. Reused verbatim from the 436/943/956 lineage's baseline
    policy.
    """
    with torch.no_grad():
        harms: List[float] = []
        for a in range(num_actions):
            a_oh = _action_onehot(a, num_actions, z_world.device)
            zw_next = agent.e2.world_forward(z_world, a_oh)
            h = agent.e3.harm_eval(zw_next).mean().item()
            harms.append(h)
        eff_t = _effective_temperature(agent)
        harms_t = torch.tensor(harms, dtype=torch.float32)
        probs = F.softmax(-harms_t / eff_t, dim=0)
        best_a = int(torch.multinomial(probs, 1).item())
    return best_a


# ------------------------------------------------------------------ #
# Write-address instrumentation -- INSTRUMENTATION FIELDS ONLY, no          #
# re-derivation of the selection expression (architecture doc mandate).    #
# ------------------------------------------------------------------ #

class _WriteSequenceTracker:
    """Records the ordered sequence of slots ContextMemory.write() actually
    mutated, by polling ContextMemory.last_write_index / .slot_write_counts
    (the authoritative instrumentation, maintained in every selection mode)
    after every agent.sense() call -- never by recomputing write()'s own
    scoring expression. Identical to V3-EXQ-943/956's tracker.
    """

    def __init__(self) -> None:
        self.sequence: List[int] = []
        self._last_total = 0

    def poll(self, context_memory) -> None:
        total = int(context_memory.slot_write_counts.sum().item())
        if total > self._last_total:
            n_new = total - self._last_total
            idx = context_memory.last_write_index
            if idx is not None:
                assert n_new == 1, (
                    f"expected at most one write() per sense() call, got "
                    f"{n_new} new writes in one poll -- polling cadence "
                    f"assumption violated, sequence would be ambiguous"
                )
                self.sequence.append(int(idx))
            self._last_total = total


def _compute_deterministic_dvs(sequence: List[int], num_slots: int) -> Dict[str, Any]:
    """The deterministic occupancy-family columns V3-EXQ-943 established as
    robust -- NOT occ_cos, NOT the synthetic-probe Jaccard (that is a SEPARATE
    post-training measurement, see _run_content_probe below).
    """
    n_write_calls = len(sequence)
    if n_write_calls == 0:
        return {
            "n_write_calls": 0,
            "n_occupied_slots": 0,
            "entropy_bits": 0.0,
            "self_repeat_rate": None,
            "round_robin_agreement": None,
            "slot_write_counts": [0] * num_slots,
        }

    counts = [0] * num_slots
    for s in sequence:
        counts[s] += 1
    n_occupied = sum(1 for c in counts if c > 0)

    entropy = 0.0
    for c in counts:
        if c > 0:
            p = c / n_write_calls
            entropy -= p * math.log2(p)

    if n_write_calls >= 2:
        repeats = sum(
            1 for i in range(1, n_write_calls) if sequence[i] == sequence[i - 1]
        )
        self_repeat_rate = repeats / (n_write_calls - 1)
    else:
        self_repeat_rate = None

    last_write_tick = [-1] * num_slots
    agree = 0
    for tick, s in enumerate(sequence):
        lru = min(range(num_slots), key=lambda i: last_write_tick[i])
        if s == lru:
            agree += 1
        last_write_tick[s] = tick
    round_robin_agreement = agree / n_write_calls

    return {
        "n_write_calls": n_write_calls,
        "n_occupied_slots": n_occupied,
        "entropy_bits": entropy,
        "self_repeat_rate": self_repeat_rate,
        "round_robin_agreement": round_robin_agreement,
        "slot_write_counts": counts,
    }


# ------------------------------------------------------------------ #
# Post-training content-discrimination probe. Reuses the CONTRACT TEST'S   #
# OWN 2-cluster synthetic instrument (test_contextmemory_write_address_    #
# selection.py's _stream/_jaccard) so the measurement is directly          #
# comparable across arms/phases and to V3-EXQ-956's own numbers.           #
# ------------------------------------------------------------------ #

def _probe_stream(seed: int, n: int, latent_dim: int, jitter: float, clusters: int):
    gen = torch.Generator().manual_seed(seed)
    bases = [torch.randn(1, latent_dim, generator=gen) * 0.078 for _ in range(clusters)]
    return [
        (i % clusters, bases[i % clusters] + torch.randn(1, latent_dim, generator=gen) * jitter)
        for i in range(n)
    ]


def _jaccard(per_cluster: Dict[int, Set[int]]) -> float:
    a, b = per_cluster.get(0, set()), per_cluster.get(1, set())
    return len(a & b) / max(len(a | b), 1)


# ------------------------------------------------------------------ #
# F1 fix: exact permutation-test significance (replaces the raw           #
# mean-margin-only PASS criterion). See module docstring "LOAD-BEARING     #
# CRITERION ... WHY IT IS A PERMUTATION TEST".                             #
# ------------------------------------------------------------------ #

_MC_RESAMPLES = 20000
_EXACT_MAX_TOTAL_N = 20  # C(20,10) ~= 184756 -- a few seconds, pure python


def _permutation_test_pvalue(
    untrained_vals: List[float], trained_vals: List[float],
) -> Tuple[float, str, int]:
    """Exact two-sample permutation test (one-sided: is trained's mean lower
    than untrained's by chance alone under the null of no true difference?).
    Pools both samples, enumerates every way to split the pooled values into
    groups of size len(untrained_vals)/len(trained_vals), and computes the
    fraction of splits whose (group-labeled-"untrained" mean - group-labeled-
    "trained" mean) is >= the actually observed difference
    (mean(untrained_vals) - mean(trained_vals); positive = correct
    direction, i.e. trained lower than untrained). Exact enumeration is used
    when the number of combinations is small enough to be fast in pure
    Python (n_untrained + n_trained <= _EXACT_MAX_TOTAL_N=20, i.e.
    C(20,10) ~= 184756 -- a few seconds at most); for anything larger, falls
    back to a large Monte Carlo resample (_MC_RESAMPLES=20000 draws) using a
    FIXED, seeded random.Random (seed derived deterministically from the
    sample sizes, via sha256 -- never Python's unseeded global RNG state) so
    the p-value is REPRODUCIBLE across reruns of this experiment script; an
    unseeded MC p-value would itself be a source of run-to-run noise in a
    supposedly pre-registered test.

    Returns (p_value, method, n_combinations_or_resamples) -- method is
    "exact" or "monte_carlo", and the count makes the result auditable
    (an exact p-value backed by only a handful of combinations is weaker
    evidence than one backed by thousands).
    """
    n1, n2 = len(untrained_vals), len(trained_vals)
    n = n1 + n2
    if n1 == 0 or n2 == 0:
        return float("nan"), "degenerate", 0
    pooled = list(untrained_vals) + list(trained_vals)
    observed_diff = (sum(untrained_vals) / n1) - (sum(trained_vals) / n2)

    if n <= _EXACT_MAX_TOTAL_N:
        n_combos = math.comb(n, n1)
        n_at_least_as_extreme = 0
        for idx_combo in itertools.combinations(range(n), n1):
            idx_set = set(idx_combo)
            group_untrained = [pooled[i] for i in idx_set]
            group_trained = [pooled[i] for i in range(n) if i not in idx_set]
            diff = (sum(group_untrained) / n1) - (sum(group_trained) / n2)
            if diff >= observed_diff - 1e-12:
                n_at_least_as_extreme += 1
        p_value = n_at_least_as_extreme / n_combos
        return p_value, "exact", n_combos

    # Monte Carlo fallback -- fixed, seeded so the p-value is reproducible.
    seed_material = f"permutation_test:{n1}:{n2}"
    rng_seed = int(hashlib.sha256(seed_material.encode("ascii")).hexdigest(), 16) % (2 ** 32)
    rng = random.Random(rng_seed)
    n_at_least_as_extreme = 0
    for _ in range(_MC_RESAMPLES):
        shuffled = pooled[:]
        rng.shuffle(shuffled)
        group_untrained = shuffled[:n1]
        group_trained = shuffled[n1:]
        diff = (sum(group_untrained) / n1) - (sum(group_trained) / n2)
        if diff >= observed_diff - 1e-12:
            n_at_least_as_extreme += 1
    p_value = n_at_least_as_extreme / _MC_RESAMPLES
    return p_value, "monte_carlo", _MC_RESAMPLES


def _paired_sign_flip_pvalue(paired_deltas: List[float]) -> Tuple[float, str, int]:
    """Exact one-sided PAIRED sign-flip permutation test.

    paired_deltas[i] = untrained_jaccard[seed_i] - trained_jaccard[seed_i]
    (positive = correct direction, i.e. untrained higher than trained at
    that seed -- the same sign convention as _permutation_test_pvalue's
    observed_diff). Under the null of "no true paired difference", each
    pair's sign is exchangeable (equally likely +/-) independent of its
    magnitude, so this enumerates all 2**n sign-flip assignments of the
    pair magnitudes and reports the fraction whose signed sum is >= the
    actually observed signed sum. Always exact (this script only ever calls
    it at n=10, 2**10=1024 combinations -- fast; there is no Monte Carlo
    fallback because Phase A's seed count is fixed and small by design).

    Returns (p_value, method, n_combinations) -- method is always
    "exact_sign_flip" here, kept as a string for the same auditability
    reason as _permutation_test_pvalue's return shape.
    """
    n = len(paired_deltas)
    if n == 0:
        return float("nan"), "degenerate", 0
    observed_sum = sum(paired_deltas)
    abs_deltas = [abs(d) for d in paired_deltas]
    n_combos = 2 ** n
    n_at_least_as_extreme = 0
    for signs in itertools.product((1, -1), repeat=n):
        permuted_sum = sum(s * a for s, a in zip(signs, abs_deltas))
        if permuted_sum >= observed_sum - 1e-12:
            n_at_least_as_extreme += 1
    p_value = n_at_least_as_extreme / n_combos
    return p_value, "exact_sign_flip", n_combos


def _run_content_probe(agent: REEAgent, seed: int) -> Tuple[float, Dict[int, Set[int]]]:
    """Post-training, EVAL-MODE probe. eval() makes gumbel_learned selection
    deterministic argmin on write_addr_tagger's own (possibly-trained) scores
    -- exactly the isolation the architecture doc specifies, so noise never
    contaminates what this measures. Uses a probe-specific seed offset so
    this measurement's RNG stream never collides with the training seed's
    own consumption earlier in the same cell.
    """
    agent.eval()
    context_memory = agent.e1.context_memory
    per_cluster: Dict[int, Set[int]] = {}
    with torch.no_grad():
        for cid, state in _probe_stream(
            seed + 500_000, PROBE_N, LATENT_DIM, PROBE_JITTER, PROBE_CLUSTERS
        ):
            context_memory.write(state)
            per_cluster.setdefault(cid, set()).add(int(context_memory.last_write_index))
    return _jaccard(per_cluster), per_cluster


# ------------------------------------------------------------------ #
# Episode / cell runners                                                   #
# ------------------------------------------------------------------ #

def _run_episode(
    agent: REEAgent,
    env: CausalGridWorldV2,
    steps: int,
    optimizer,
    harm_eval_opt,
    harm_buf_pos: List[torch.Tensor],
    harm_buf_neg: List[torch.Tensor],
    tracker: _WriteSequenceTracker,
) -> None:
    """Run a single training episode: sense -> E1 tick -> baseline action
    selection -> step -> prediction-loss training (which now includes the
    write-addressing loss when weight>0, wired inside
    agent.compute_prediction_loss() -- see ree_core/agent.py) -> harm_eval
    training.

    Reuses V3-EXQ-956's episode runner verbatim, INCLUDING its `_e1_tick`
    call: `compute_prediction_loss()` returns an unconditional `zero_loss`
    whenever `self._world_experience_buffer` has fewer than 2 entries, and
    that buffer is populated ONLY inside `_e1_tick` -- which itself only
    runs from `act()`/`act_with_split_obs()`, neither of which this driver
    (or 943's/956's) ever calls, since all three use a hand-rolled baseline
    action-selection policy instead of the agent's own select_action. Without
    the explicit call below, the addressing loss (gated on `weight>0 AND
    write_selection=="gumbel_learned"`, added to `loss` INSIDE
    compute_prediction_loss) is silently never added to the backward graph.
    """
    _, obs_dict = env.reset()
    agent.reset()
    agent.e1.reset_hidden_state()

    for _step in range(steps):
        obs_body = obs_dict["body_state"]
        obs_world = obs_dict["world_state"]
        obs_harm = obs_dict.get("harm_obs", None)

        latent = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
        ticks = agent.clock.advance()
        if ticks.get("e1_tick", False):
            agent._e1_tick(latent)
        tracker.poll(agent.e1.context_memory)  # exact -- see class docstring

        z_world = latent.z_world.detach().clone()
        action_idx = _select_action_baseline(agent, z_world, env.action_dim)
        action_oh = _action_onehot(action_idx, env.action_dim, agent.device)
        agent._last_action = action_oh

        _, harm_signal, done, _info, obs_dict = env.step(action_oh)
        is_harm = float(harm_signal) < 0

        e1_loss = agent.compute_prediction_loss()
        e2_loss = agent.compute_e2_loss()
        total = e1_loss + e2_loss
        if total.requires_grad:
            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
            optimizer.step()

        if is_harm:
            harm_buf_pos.append(z_world)
        else:
            harm_buf_neg.append(z_world)
        if len(harm_buf_pos) > MAX_HARM_BUF:
            del harm_buf_pos[:-MAX_HARM_BUF]
        if len(harm_buf_neg) > MAX_HARM_BUF:
            del harm_buf_neg[:-MAX_HARM_BUF]

        if len(harm_buf_pos) >= 4 and len(harm_buf_neg) >= 4:
            k_pos = min(16, len(harm_buf_pos))
            k_neg = min(16, len(harm_buf_neg))
            pos_idx = torch.randperm(len(harm_buf_pos))[:k_pos].tolist()
            neg_idx = torch.randperm(len(harm_buf_neg))[:k_neg].tolist()
            zw_pos = torch.cat([harm_buf_pos[i] for i in pos_idx], dim=0)
            zw_neg = torch.cat([harm_buf_neg[i] for i in neg_idx], dim=0)
            zw_b = torch.cat([zw_pos, zw_neg], dim=0)
            target_t = torch.cat([
                torch.ones(k_pos, 1, device=agent.device),
                torch.zeros(k_neg, 1, device=agent.device),
            ], dim=0)
            pred = agent.e3.harm_eval_head(zw_b)
            h_loss = F.binary_cross_entropy_with_logits(pred, target_t)
            harm_eval_opt.zero_grad()
            h_loss.backward()
            harm_eval_opt.step()

        if done:
            break


def _run_cell(phase: str, arm_name: str, write_selection: str,
              write_addressing_loss_weight: float, gumbel_anneal_steps: int,
              seed: int, base_config_slice: Dict[str, Any],
              zg: ZGoalStreamAccumulator, training_episodes: int,
              steps_per_episode: int) -> Dict[str, Any]:
    cell_config_slice = {
        **base_config_slice,
        "phase": phase,
        "arm": arm_name,
        "contextmemory_write_selection": write_selection,
        "contextmemory_write_addressing_loss_weight": write_addressing_loss_weight,
        "contextmemory_write_gumbel_anneal_steps": gumbel_anneal_steps,
    }
    with arm_cell(seed, config_slice=cell_config_slice, script_path=Path(__file__)) as cell:
        env_safe = _make_env_safe(seed)
        env_dang = _make_env_dangerous(seed)
        agent = _make_agent(env_safe, write_selection, write_addressing_loss_weight,
                             gumbel_anneal_steps)

        tagger_init_state = {
            k: v.clone()
            for k, v in agent.e1.context_memory.write_addr_tagger.state_dict().items()
        }

        standard_params = [
            p for n, p in agent.named_parameters()
            if "harm_eval_head" not in n and "context_memory.memory" not in n
        ]
        harm_eval_params = list(agent.e3.harm_eval_head.parameters())
        optimizer = optim.Adam(standard_params, lr=1e-3)
        harm_eval_opt = optim.Adam(harm_eval_params, lr=1e-4)

        harm_buf_pos: List[torch.Tensor] = []
        harm_buf_neg: List[torch.Tensor] = []
        tracker = _WriteSequenceTracker()

        agent.train()
        print(f"Seed {seed} Condition {arm_name}", flush=True)
        for ep in range(training_episodes):
            block = ep // CONTEXT_SWITCH_EVERY
            is_safe_ep = (block % 2 == 0)
            env = env_safe if is_safe_ep else env_dang
            _run_episode(
                agent, env, steps_per_episode, optimizer, harm_eval_opt,
                harm_buf_pos, harm_buf_neg, tracker,
            )
            if (ep + 1) % 20 == 0 or (ep + 1) == training_episodes:
                print(
                    f"  [train] phase={phase} arm={arm_name} seed={seed} "
                    f"ep {ep + 1}/{training_episodes} "
                    f"n_writes={len(tracker.sequence)}",
                    flush=True,
                )

        zg.observe(agent)
        dvs = _compute_deterministic_dvs(tracker.sequence, NUM_SLOTS)

        final_state = agent.e1.context_memory.write_addr_tagger.state_dict()
        max_abs_diff = 0.0
        for k, v0 in tagger_init_state.items():
            d = (final_state[k] - v0).abs().max().item()
            max_abs_diff = max(max_abs_diff, d)
        tagger_params_moved = max_abs_diff > TAGGER_MOVE_EPS

        probe_jaccard, probe_per_cluster = _run_content_probe(agent, seed)

        row: Dict[str, Any] = {
            "phase": phase,
            "arm": arm_name,
            "seed": seed,
            "write_selection": write_selection,
            "contextmemory_write_addressing_loss_weight": write_addressing_loss_weight,
            "contextmemory_write_gumbel_anneal_steps": gumbel_anneal_steps,
            **dvs,
            "tagger_params_moved": tagger_params_moved,
            "tagger_max_abs_param_diff": max_abs_diff,
            "probe_2cluster_jaccard": probe_jaccard,
            "probe_2cluster_occupied": {
                str(cid): sorted(slots) for cid, slots in probe_per_cluster.items()
            },
        }
        cell.stamp(row)

    print(
        f"verdict: {'PASS' if dvs['n_occupied_slots'] >= 2 else 'FAIL'} "
        f"(occupancy; probe_jaccard={probe_jaccard:.3f})",
        flush=True,
    )
    return row


# ------------------------------------------------------------------ #
# Top-level run                                                            #
# ------------------------------------------------------------------ #

def run(dry_run: bool = False) -> Tuple[Dict[str, Any], ZGoalStreamAccumulator]:
    zg = ZGoalStreamAccumulator()

    training_episodes = 2 if dry_run else TRAINING_EPISODES
    steps_per_episode = 5 if dry_run else STEPS_PER_EPISODE
    phase_a_seeds = PHASE_A_SEEDS[:1] if dry_run else PHASE_A_SEEDS
    phase_b_seeds = PHASE_B_SEEDS[:1] if dry_run else PHASE_B_SEEDS

    full_config_slice: Dict[str, Any] = {
        "phase_a_arms": [a[0] for a in PHASE_A_ARM_SPECS],
        "phase_b_arms": [a[0] for a in PHASE_B_ARM_SPECS],
        "phase_a_seeds": phase_a_seeds,
        "phase_b_seeds": phase_b_seeds,
        "training_episodes": training_episodes,
        "steps_per_episode": steps_per_episode,
        "context_switch_every": CONTEXT_SWITCH_EVERY,
        "contextmemory_gated_content_write": CONTEXTMEMORY_GATED_CONTENT_WRITE,
        "sd016_writepath_mode": SD016_WRITEPATH_MODE,
        "alpha_world": ALPHA_WORLD,
        "use_noise_floor": USE_NOISE_FLOOR,
        "num_slots": NUM_SLOTS,
        "probe_n": PROBE_N,
        "probe_jitter": PROBE_JITTER,
        "probe_clusters": PROBE_CLUSTERS,
    }

    arm_results: List[Dict[str, Any]] = []
    for arm_name, wsel, waw, asteps in PHASE_A_ARM_SPECS:
        for seed in phase_a_seeds:
            row = _run_cell(
                "A", arm_name, wsel, waw, asteps, seed, full_config_slice, zg,
                training_episodes, steps_per_episode,
            )
            arm_results.append(row)
    for arm_name, wsel, waw, asteps in PHASE_B_ARM_SPECS:
        for seed in phase_b_seeds:
            row = _run_cell(
                "B", arm_name, wsel, waw, asteps, seed, full_config_slice, zg,
                training_episodes, steps_per_episode,
            )
            arm_results.append(row)

    by_arm: Dict[str, List[Dict[str, Any]]] = {name: [] for name in ALL_ARM_NAMES}
    for row in arm_results:
        by_arm[row["arm"]].append(row)

    # --- P0 readiness: writepath genuinely engaged in every cell (both phases) ---
    write_call_counts = [row["n_write_calls"] for row in arm_results]
    min_write_calls = min(write_call_counts) if write_call_counts else 0.0
    try:
        preconditions = p0_readiness_gate([
            {
                "name": "writepath_engaged_every_cell",
                "measured": float(min_write_calls),
                "threshold": WRITE_CALLS_FLOOR_MODERATE,
                "direction": "lower",
            },
        ])
    except P0NotReady as e:
        status = "FAIL"
        label = "substrate_not_ready_requeue"
        result: Dict[str, Any] = {
            "outcome": status,
            "status": status,
            "claim_ids": CLAIM_IDS,
            "experiment_type": EXPERIMENT_TYPE,
            "experiment_purpose": EXPERIMENT_PURPOSE,
            "evidence_direction": "non_contributory",
            "sleep_driver_pattern": "N/A (no sleep loop)",
            "arm_results": arm_results,
            "interpretation": {
                "label": label,
                "preconditions": e.preconditions,
                "criteria_non_degenerate": {},
            },
            "fatal_error_count": 0,
        }
        return result, zg

    # --- Mean Jaccard helper ---
    def _mean_probe_jaccard(arm_name: str) -> float:
        vals = [row["probe_2cluster_jaccard"] for row in by_arm[arm_name]]
        return sum(vals) / len(vals) if vals else float("nan")

    mean_jaccard_untrained = _mean_probe_jaccard("GUMBEL_UNTRAINED")

    # C2-style readiness (module docstring "C2-STYLE READINESS"): the
    # untrained baseline must have enough room below it for "materially
    # lower" to be satisfiable at all.
    c2_headroom_met = mean_jaccard_untrained >= C2_JACCARD_MARGIN

    # --- Phase A paired-sign descriptive count (seed IS the pairing key) ---
    # Computed BEFORE the tested-configs loop below because Phase A's own
    # significance test (F1 fix) reuses this same per-seed pairing.
    untrained_by_seed = {r["seed"]: r["probe_2cluster_jaccard"] for r in by_arm["GUMBEL_UNTRAINED"]}
    trained_w0p5_by_seed = {r["seed"]: r["probe_2cluster_jaccard"] for r in by_arm["GUMBEL_TRAINED_W0P5"]}
    paired_deltas: List[Dict[str, Any]] = []
    n_wrong_direction = 0
    n_correct_direction = 0
    n_tied = 0
    for seed in phase_a_seeds:
        d = trained_w0p5_by_seed[seed] - untrained_by_seed[seed]
        paired_deltas.append({"seed": seed, "delta": d})
        if d > 0:
            n_wrong_direction += 1
        elif d < 0:
            n_correct_direction += 1
        else:
            n_tied += 1

    # --- F1 fix: the 4 tested configs (Phase A w=0.5 + Phase B's 3 weights), ---
    # --- each scored by an EXACT permutation test, Bonferroni-corrected.    ---
    untrained_jaccards = [row["probe_2cluster_jaccard"] for row in by_arm["GUMBEL_UNTRAINED"]]
    tested_configs: List[Dict[str, Any]] = []
    for arm_name, _wsel, weight, asteps in ALL_ARM_SPECS:
        if arm_name == "GUMBEL_UNTRAINED":
            continue  # this IS the baseline, not a tested config
        rows = by_arm[arm_name]
        trained_vals = [row["probe_2cluster_jaccard"] for row in rows]
        mean_j = _mean_probe_jaccard(arm_name)
        delta = mean_j - mean_jaccard_untrained
        raw_directional_pass = mean_j <= (mean_jaccard_untrained - C2_JACCARD_MARGIN)

        if arm_name == "GUMBEL_TRAINED_W0P5":
            # Phase A: PAIRED exact sign-flip test -- seed IS the pairing key
            # (see PHASE A in the module docstring). Sign convention matches
            # _permutation_test_pvalue: untrained - trained, positive =
            # correct direction.
            phase_a_paired = [
                untrained_by_seed[seed] - trained_w0p5_by_seed[seed] for seed in phase_a_seeds
            ]
            observed_diff = sum(phase_a_paired) / len(phase_a_paired)
            p_value, perm_method, n_perms = _paired_sign_flip_pvalue(phase_a_paired)
        else:
            # Phase B: UNPAIRED exact two-sample test against Phase A's
            # n=10 untrained pool (Phase B's 6 seeds do not match Phase A's
            # seeds, so there is no natural pairing).
            observed_diff = (
                (sum(untrained_jaccards) / len(untrained_jaccards)) - mean_j
                if untrained_jaccards else float("nan")
            )
            p_value, perm_method, n_perms = _permutation_test_pvalue(untrained_jaccards, trained_vals)

        cleared_margin = observed_diff >= C2_JACCARD_MARGIN
        cleared_significance = p_value < ALPHA_CORRECTED
        tagger_moved = bool(rows) and all(row["tagger_params_moved"] for row in rows)
        counts_as_discovery = cleared_margin and cleared_significance and tagger_moved

        tested_configs.append({
            "config_label": arm_name,
            "phase": "A" if arm_name == "GUMBEL_TRAINED_W0P5" else "B",
            "weight": weight,
            "anneal_steps": asteps,
            "n_seeds": len(rows),
            "mean_jaccard_trained": mean_j,
            "delta_from_untrained_baseline": delta,
            "observed_diff": observed_diff,
            "p_value": p_value,
            "permutation_method": perm_method,
            "n_permutations_or_resamples": n_perms,
            "alpha_corrected": ALPHA_CORRECTED,
            "cleared_margin": cleared_margin,
            "cleared_significance": cleared_significance,
            "tagger_moved": tagger_moved,
            "counts_as_discovery": counts_as_discovery,
            "raw_margin_reading": {
                "load_bearing": False,
                "directional_pass": raw_directional_pass,
                "note": (
                    "OLD raw mean-margin-only reading (no significance "
                    "test, no multiplicity correction) -- kept as a "
                    "non-gating descriptive field for side-by-side "
                    "comparison against the corrected reading. NOT the "
                    "load-bearing criterion; see counts_as_discovery."
                ),
            },
        })

    assert len(tested_configs) == N_CONFIGS_TESTED, (
        f"expected N_CONFIGS_TESTED={N_CONFIGS_TESTED} tested configs, got "
        f"{len(tested_configs)} -- N_CONFIGS_TESTED (and therefore "
        "ALPHA_CORRECTED, the Bonferroni denominator) must match the "
        "number of configs actually scored here."
    )

    winning_configs = [c for c in tested_configs if c["counts_as_discovery"]]
    any_discovery = len(winning_configs) > 0
    overall_pass = c2_headroom_met and any_discovery
    # F3 fix: a PASS whose winning config's tagger never moved is a bug
    # reading. No winning config (FAIL, or precondition unmet) leaves this
    # True by vacuous truth -- there is nothing to invalidate.
    winning_config_tagger_moved = all(c["tagger_moved"] for c in winning_configs) if winning_configs else True

    # --- Descriptive, non-gating negative controls (tagger movement, ---
    # --- aggregated across ALL trained cells in both phases). ---
    n_untrained_frozen = sum(
        1 for row in by_arm["GUMBEL_UNTRAINED"] if row["tagger_params_moved"] is False
    )
    trained_arm_names = [a[0] for a in ALL_ARM_SPECS if a[0] != "GUMBEL_UNTRAINED"]
    trained_rows = [row for name in trained_arm_names for row in by_arm[name]]
    n_trained_moved = sum(1 for row in trained_rows if row["tagger_params_moved"] is True)

    criteria = [
        {
            "name": "H2_readiness_untrained_baseline_headroom",
            "load_bearing": False,
            "kind": "readiness",
            "passed": c2_headroom_met,
            "measured": mean_jaccard_untrained,
            "threshold": C2_JACCARD_MARGIN,
            "direction": "lower",
            "note": (
                "Precondition shared by every Phase A/B directional "
                "comparison: mean_jaccard(GUMBEL_UNTRAINED, Phase A, n=10) "
                "must be >= 0.25 for 'materially lower' to be satisfiable at "
                "all. If unmet, the load-bearing criterion below reads "
                "precondition_unmet rather than a plain FAIL -- a floor "
                "effect is not evidence that no operating point works."
            ),
        },
        {
            "name": "H2_any_operating_point_improves_content_discrimination",
            "load_bearing": True,
            "passed": overall_pass,
            "precondition_unmet": not c2_headroom_met,
            "any_discovery": any_discovery,
            "mean_jaccard_untrained_baseline": mean_jaccard_untrained,
            "required_margin": C2_JACCARD_MARGIN,
            "alpha_corrected": ALPHA_CORRECTED,
            "n_configs_tested": len(tested_configs),
            "n_configs_cleared_margin": sum(1 for c in tested_configs if c["cleared_margin"]),
            "n_configs_cleared_significance": sum(1 for c in tested_configs if c["cleared_significance"]),
            "n_configs_counts_as_discovery": len(winning_configs),
            "winning_config_labels": [c["config_label"] for c in winning_configs],
            "winning_config_tagger_moved": winning_config_tagger_moved,
            "note": (
                "PASS iff the readiness precondition above is met AND at "
                "least one of the N_CONFIGS_TESTED=4 tested configs (Phase "
                "A w=0.5/n=10, or any of Phase B's 3 weight configs) "
                "counts_as_discovery: observed_diff (mean_jaccard_untrained "
                "- mean_jaccard_trained) >= 0.25 (the real-effect-size "
                "floor) AND an EXACT permutation-test p_value < "
                "alpha_corrected (0.05/4=0.0125, Bonferroni-corrected "
                "across the 4 configs tested) AND that config's "
                "write_addr_tagger genuinely moved from init (see F3/"
                "tagger_moved). This replaces a raw mean-margin-only OR "
                "test across configs, which at this DV's near-binary "
                "per-seed shape had a measured 25-68% false-positive rate "
                "under the null with no multiplicity correction -- see "
                "raw_margin_reading on each config in H2_per_config_table "
                "for the old (non-gating) reading, kept for comparison. "
                "This is H2's actual null-test: PASS = an operating point "
                "was found with the effect size, significance, AND "
                "engagement to count as a real discovery; FAIL "
                "(precondition met) = the null holds -- no operating point "
                "in this neighborhood escapes 956's wrong-direction result "
                "at this power. A FAIL here is a real, scientifically "
                "valid, expected-possible outcome, not a bug. "
                "precondition_unmet=True means the comparison could not be "
                "made at all, not that every operating point was tried and "
                "failed."
            ),
        },
        {
            "name": "H2_per_config_table",
            "load_bearing": False,
            "passed": True,
            "kind": "descriptive",
            "configs": tested_configs,
            "note": (
                "Descriptive table of all 4 tested weight configs' mean "
                "Jaccard, permutation-test result (observed_diff, p_value, "
                "method, cleared_margin, cleared_significance, "
                "tagger_moved, counts_as_discovery), and the OLD "
                "non-gating raw_margin_reading for comparison. Always "
                "'passed' -- this is a measurement, not a criterion; "
                "H2_any_operating_point_improves_content_discrimination "
                "above is the actual gate."
            ),
        },
        {
            "name": "H2_phaseA_paired_sign_count",
            "load_bearing": False,
            "passed": True,
            "kind": "descriptive",
            "n_seeds_total": len(phase_a_seeds),
            "n_seeds_wrong_direction": n_wrong_direction,
            "n_seeds_correct_direction": n_correct_direction,
            "n_seeds_tied": n_tied,
            "per_seed_deltas": paired_deltas,
            "note": (
                "Seed IS the pairing key: seed S's GUMBEL_TRAINED_W0P5 cell "
                "minus seed S's GUMBEL_UNTRAINED cell, per-seed Jaccard "
                "delta. Positive delta = wrong direction (matches 956's own "
                "finding); negative delta = correct direction. Purely "
                "descriptive -- not itself a pass/fail criterion; a bigger-n "
                "read on whether 956's wrong-direction move at w=0.5 "
                "replicates."
            ),
        },
        {
            "name": "GUMBEL_UNTRAINED_tagger_received_zero_gradient",
            "load_bearing": False,
            "passed": n_untrained_frozen == len(by_arm["GUMBEL_UNTRAINED"]),
            "n_seeds_frozen": n_untrained_frozen,
            "n_seeds_total": len(by_arm["GUMBEL_UNTRAINED"]),
            "note": (
                "Mechanical negative control: weight=0.0 should leave "
                "write_addr_tagger's state_dict byte-identical to its "
                "random init across the whole real training run."
            ),
        },
        {
            "name": "trained_configs_tagger_received_gradient",
            "load_bearing": False,
            "passed": n_trained_moved == len(trained_rows),
            "n_cells_moved": n_trained_moved,
            "n_cells_total": len(trained_rows),
            "note": (
                "Mechanical positive control, aggregated across ALL trained "
                "cells in both phases: every weight>0 cell should move "
                "write_addr_tagger's parameters away from init over the "
                "real training run, regardless of whether that movement "
                "achieves content-discrimination."
            ),
        },
    ]
    criteria_non_degenerate = {
        "H2_any_operating_point_improves_content_discrimination": (
            len(phase_a_seeds) >= 2
            and len(phase_b_seeds) >= 2
            and c2_headroom_met
            # F3 fix: a PASS whose winning config's tagger never received
            # gradient is a bug reading, not a real discovery -- must not
            # read as non_degenerate: true. Vacuously True when there is no
            # winning config (a FAIL is a well-formed, non-degenerate run).
            and winning_config_tagger_moved
            # Finding G2 fix (red-team, fable, 2026-09-02): winning_config_
            # tagger_moved is vacuously True on the FAIL branch (no winning
            # config to invalidate), so it alone cannot catch a silent
            # wiring failure where write_addr_tagger never received
            # gradient in ANY trained cell (e.g. the addressing-loss state
            # buffer never populated). Require the aggregate positive
            # control too: a FAIL only reads as non-degenerate if every
            # trained cell's tagger genuinely moved from init.
            and (n_trained_moved == len(trained_rows))
        ),
    }

    if overall_pass:
        label = "h2_operating_point_found_content_discrimination_improves"
    elif not c2_headroom_met:
        label = "h2_readiness_precondition_unmet_untrained_baseline_floor_effect"
    else:
        label = "h2_no_operating_point_improves_content_discrimination_null_holds"

    evidence_direction = "non_contributory"
    status = "PASS" if overall_pass else "FAIL"

    metrics: Dict[str, Any] = {}
    for arm_name in ALL_ARM_NAMES:
        rows = by_arm[arm_name]
        metrics[f"n_occupied_slots_per_seed_{arm_name}"] = [r["n_occupied_slots"] for r in rows]
        metrics[f"n_write_calls_per_seed_{arm_name}"] = [r["n_write_calls"] for r in rows]
        metrics[f"probe_2cluster_jaccard_per_seed_{arm_name}"] = [r["probe_2cluster_jaccard"] for r in rows]
        metrics[f"mean_probe_2cluster_jaccard_{arm_name}"] = _mean_probe_jaccard(arm_name)

    config_table_md = "".join(
        f"| {c['config_label']} | {c['weight']} | "
        f"{c['mean_jaccard_trained']:.3f} | {c['delta_from_untrained_baseline']:+.3f} | "
        f"{c['p_value']:.4f} ({c['permutation_method']}) | "
        f"{'YES' if c['counts_as_discovery'] else 'no'} |\n"
        for c in tested_configs
    )
    summary_markdown = (
        f"# {QUEUE_ID} -- ContextMemory gumbel_learned write-address selection, "
        f"H2-operating-point sweep\n\n"
        f"**Status:** {status}  **Label:** {label}\n"
        f"**Purpose:** diagnostic (claim_ids=[]; validates substrate_queue "
        f"contextmemory-write-path-addressing-degeneracy, H2 leg)\n\n"
        f"Untrained baseline (Phase A, n={len(phase_a_seeds)}): "
        f"mean_jaccard={mean_jaccard_untrained:.3f} "
        f"(readiness {'PASS' if c2_headroom_met else 'FAIL'})\n\n"
        f"| Config | weight | mean Jaccard | delta vs baseline | p-value (method) | counts as discovery |\n"
        f"|---|---|---|---|---|---|\n"
        f"{config_table_md}\n"
        f"H2_any_operating_point_improves_content_discrimination: "
        f"{'PASS' if overall_pass else 'FAIL'} "
        f"(alpha_corrected={ALPHA_CORRECTED:.4f} across N_CONFIGS_TESTED={N_CONFIGS_TESTED})\n"
        f"Phase A paired sign (n={len(phase_a_seeds)}): "
        f"{n_wrong_direction} wrong-direction, {n_correct_direction} correct-direction, "
        f"{n_tied} tied\n"
    )

    result = {
        "outcome": status,
        "status": status,
        "claim_ids": CLAIM_IDS,
        "experiment_type": EXPERIMENT_TYPE,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "evidence_direction": evidence_direction,
        "sleep_driver_pattern": "N/A (no sleep loop)",
        "metrics": metrics,
        "arm_results": arm_results,
        "summary_markdown": summary_markdown,
        "interpretation": {
            "label": label,
            "preconditions": preconditions,
            "criteria_non_degenerate": criteria_non_degenerate,
            "criteria": criteria,
        },
        "fatal_error_count": 0,
    }
    return result, zg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()
    result, zg_accumulator = run(dry_run=args.dry_run)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["timestamp_utc"] = ts
    result["run_id"] = f"{EXPERIMENT_TYPE}_{ts}_v3"
    result["architecture_epoch"] = ARCHITECTURE_EPOCH

    all_seeds = sorted(set(PHASE_A_SEEDS) | set(PHASE_B_SEEDS))
    full_config = {
        "phase_a_arms": [a[0] for a in PHASE_A_ARM_SPECS],
        "phase_b_arms": [a[0] for a in PHASE_B_ARM_SPECS],
        "phase_a_seeds": PHASE_A_SEEDS,
        "phase_b_seeds": PHASE_B_SEEDS,
        "training_episodes": TRAINING_EPISODES,
        "steps_per_episode": STEPS_PER_EPISODE,
        "context_switch_every": CONTEXT_SWITCH_EVERY,
        "contextmemory_gated_content_write": CONTEXTMEMORY_GATED_CONTENT_WRITE,
        "sd016_writepath_mode": SD016_WRITEPATH_MODE,
        "alpha_world": ALPHA_WORLD,
        "use_noise_floor": USE_NOISE_FLOOR,
        "num_slots": NUM_SLOTS,
        "anneal_default": ANNEAL_DEFAULT,
        "phase_b_weights": PHASE_B_WEIGHTS,
        "c2_jaccard_margin": C2_JACCARD_MARGIN,
        "write_calls_floor_moderate": WRITE_CALLS_FLOOR_MODERATE,
        "probe_n": PROBE_N,
        "probe_jitter": PROBE_JITTER,
        "probe_clusters": PROBE_CLUSTERS,
        "latent_dim": LATENT_DIM,
        "n_configs_tested": N_CONFIGS_TESTED,
        "alpha": ALPHA,
        "alpha_corrected": ALPHA_CORRECTED,
    }

    out_path = write_flat_manifest(
        result,
        dry_run=args.dry_run,
        config=full_config,
        seeds=all_seeds,
        script_path=__file__,
        started_at=t0,
        z_goal_stream_stats=zg_accumulator.stats(),
    )

    print(f"\nResult written to: {out_path}", flush=True)
    print(f"Status: {result['status']}", flush=True)

    emit_outcome(
        outcome=result["status"] if result["status"] in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
        dry_run=args.dry_run,
    )
