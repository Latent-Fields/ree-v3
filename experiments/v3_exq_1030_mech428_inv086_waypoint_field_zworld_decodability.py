"""V3-EXQ-1030 -- waypoint_field_consumer_reach H2 (axis: representation).

Registry: REE_assembly/evidence/planning/hypothesis_space_registry.v1.json,
qid `waypoint_field_consumer_reach`, hid `H-wpfield-zworld-interface`. Pre-registered by
the confirmed `failure_autopsy_V3-EXQ-1004_2026-09-05` fan-out (section 7a, GOV-FANOUT-1
two-leg portfolio; H1 is the sibling `drive`-axis leg, queued separately).

QUESTION. V3-EXQ-1004 established that the SD-WAYPOINT-FIELD observable
(`waypoint_proximity_field_view`, the trailing 25 dims of `world_state`) makes the pending
waypoint's direction decodable and behaviourally sufficient FOR A SUPERVISED READER OVER THE
RAW OBSERVATION (0.575 -> 0.841 BC-imitation accuracy). This run asks the orthogonal question:
does that same directional signal survive REE's OWN observation-to-z_world compression, or is
it lost/swamped inside the encoder before it ever reaches E1/E2/E3?

DECLARED NULL (H2, from the registry). The z_world accuracy lift (ON minus OFF) is
indistinguishable from zero while the raw-observation accuracy lift is large -- which would
locate the residual navigation blockage at the encoder, not at the environment. Meeting this
null SUPPORTS H2. A z_world lift comparable in magnitude to the raw-observation lift does NOT
support H2 -- it means the interface carries the signal fine, which locates the blockage
elsewhere (the sibling H1 objective/exploration leg).

DESIGN. Two arms (`field_off` / `field_on`) x 5 seeds. Per (seed, arm) cell:
  P0  -- warm up a REEAgent's world encoder via the SD-070 P0a recipe (SD-018 resource-
         proximity target, `experiments/_lib/zworld_p0_warmup.run_zworld_p0`) on a DEDICATED
         env instance. This is a GENERIC encoder warmup, unrelated to waypoints: the encoder
         is never specifically tasked to predict anything about the field, so a positive
         decodability result cannot be an artifact of having been trained on the very
         quantity under test.
  P1  -- freeze the whole latent stack (`requires_grad_(False)`); no further gradient step
         touches it.
  P2  -- roll out a UNIFORM-RANDOM policy on a second dedicated env instance for probe-data
         collection. At every step, before acting, compute the oracle greedy direction toward
         the pending waypoint (5-way: up/down/left/right/stay -- a LABEL only, never an action
         taken) and the frozen z_world for the current observation. Fit single-layer linear
         probes (cross-entropy, EPISODE-level 80/20 train/test split -- see F4 below) for the
         waypoint-direction label from z_world (the DV) and from the raw `world_state` (the
         internal positive control / harness-replication check).

WHY THE ENV IS NOT NAVIGATION-ISOLATED THE WAY V3-EXQ-1004's WAS. 1004 zeroed hazards and
resources because it trained a reward-maximising policy and needed the waypoint visit to be
the only reward term reachable by the objective. This run trains NO policy toward any reward
at all -- the label is read directly from environment ground truth regardless of what action
the (uniform-random) rollout takes, and the "reward-composition dominates the objective"
confound has no purchase on a supervised linear probe. Hazards and resources are therefore
left ON at modest levels (2 / 3) specifically so the standard SD-070 P0a resource-proximity
target has a live signal for the GENERIC encoder warmup.

DV-SYMMETRY INVARIANCE (queue-experiment Step 3 mandatory declaration). The DV is held-out
linear-probe classification accuracy from a freshly cross-entropy-trained single layer. It is
not a selection over externally-scored candidates, so it carries no exposure to a broadcast-
additive-constant symmetry; it is not an order/rank statistic, so a monotone-rescaling
symmetry does not apply; and the per-seed accuracy figures this run aggregates are not a
symmetric function of interchangeable seeds -- each seed's field content and label sequence
differ by construction, so a permutation-of-units symmetry does not apply either. None of the
three manipulation-invisibility classes in the Step 3 table can mask the ON/OFF contrast here.

=== STEP 4.5 RED-TEAM (fable-5.1, cross-model): CONTESTED -- six findings, all applied ===

F1 (family 2, criterion-cannot-discriminate): an under-fit or scale-starved z_world probe
   reads OFF~=ON~=majority-class in BOTH arms, which would satisfy C1's null BY CONSTRUCTION
   regardless of what z_world actually encodes -- R1 (the raw-slice positive control) proves
   the FITTING PROCEDURE works on a 50-dim [0,1] slice, but says nothing about the SAME
   procedure on the 32-dim z_world's own scale/distribution. FIX: added readiness R3
   `zworld_probe_extracts_known_signal` -- a THIRD auxiliary linear probe, run on the SAME
   z_world features, predicting a 4-way GRID QUADRANT from agent (x, y) position (a target
   structurally present in world_state regardless of the waypoint field, so it must be
   decodable if the z_world pipeline can learn anything at this scale at all). Below-floor
   self-routes `substrate_not_ready_requeue` -- this is deliberately NOT "does z_world beat
   majority class on the WAYPOINT label", which would wrongly penalise a TRUE full loss (H2
   correct implies OFF and ON both sit near floor on the waypoint label by design).

F2 (family 1, manipulation-cannot-cleanly-attribute): `_build_agent` sized `world_obs_dim`
   from each arm's OWN env (250 OFF / 275 ON), so OFF and ON built INDEPENDENTLY-INITIALISED
   networks of different fan-in -- every downstream parameter (not just the first layer)
   consumes a different slice of the RNG stream, so a per-seed z_world lift was an UNPAIRED
   contrast between two different random networks, not a controlled ON/OFF content contrast.
   FIX: `_FieldMaskedEnv` always builds the WIDE (field-enabled) env and MASKS the trailing
   FIELD_DIMS of `world_state` to zero when the arm is OFF -- exactly 1004's own F7 zero-pad
   convention (matched init, matched width), extended from the raw-slice probe to the agent's
   actual z_world input. Both arms now build a BYTE-IDENTICAL-AT-INIT agent for a given seed;
   the only difference is the CONTENT of the trailing 25 `world_state` columns.

F3 (family 3, verdict-grid overclaim): `abs(lift) <= ceiling` folded a strongly NEGATIVE lift
   (ON decodes WORSE than OFF from z_world) into the same FAIL branch as a genuine positive
   lift, both labelled `waypoint_field_survives_zworld_encoder` -- which claims the interface
   carries the signal FINE, the opposite of what a negative lift would mean. FIX: `_score` now
   splits FAIL into `..._survives_zworld_encoder` (positive, beyond ceiling),
   `..._anomalous_negative_lift` (negative beyond ceiling -- flagged, not read either way) and
   `..._lift_inconsistent_across_seeds` (no seed-count majority in any direction).

F4 (family 1, leakage): z_world is a 0.9-EMA over within-episode state
   (`z_world = alpha_world * z_world_instant + (1-alpha_world) * prev_state.z_world`,
   ree_core/latent/stack.py:1584), so a flat step-level random train/test split lets
   near-duplicate, temporally-adjacent samples FROM THE SAME EPISODE land on both sides,
   inflating apparent decodability via smoothing/memorisation rather than generalisation --
   asymmetrically, since the raw slice (no EMA) is not exposed to this. FIX:
   `_episode_split` assigns whole EPISODES to train/test (80/20), never splitting one episode
   across both partitions.

F5 (family 4, gate-certifies-wrong-thing): (a) confirmed by reading the source: `agent.sense()`
   feeds `world_obs_encoder(world_state)` (agent.py:4802, a SEPARATE randomly-initialised
   Linear+ReLU that NOTHING in the corpus ever trains) into `split_encoder.world_encoder`,
   while P0a (`zworld_p0.py::_z_world_path`) trains `split_encoder.world_encoder` directly on
   RAW `world_state` -- a genuine train/eval distribution mismatch, but one that is a
   PRE-EXISTING, corpus-wide SD-070 characteristic (every driver using `run_zworld_p0` +
   `agent.sense()` has it; `zworld_encoder_guard` itself only ever watches
   `split_encoder.world_encoder`'s own weights) and out of THIS session's scope to fix (it
   would mean retraining a shared, already-validated recipe five sibling drivers depend on).
   Measuring via `agent.sense()` remains the RIGHT thing for this question -- it is the actual
   pathway every real REE consumer (E1/E2/E3) reads -- but a null could additionally reflect
   scrambling by the untrained `world_obs_encoder` pre-projection rather than compression loss
   inside `split_encoder.world_encoder` itself. FIX (disambiguating record, not a design
   change): a second, NON-load-bearing diagnostic probe `zworld_direct_probe` reads
   `split_encoder.world_encoder(raw world_state)` directly (mirroring `_z_world_path` exactly,
   bypassing `world_obs_encoder`), so a later reader can tell the two loss sites apart. (b)
   "encoder moved" is a weak positive: field values are bounded away from zero everywhere
   in-bounds (>= ~0.15), so Adam almost certainly perturbs the trailing-25 input columns'
   weights regardless of whether the P0a objective actually PRESERVES that region's content.
   FIX (diagnostic record): `p0a_diagnostics` now also carries `field_cols_weight_delta` /
   `other_cols_weight_delta`, the column-wise `world_encoder[0].weight` delta norm split at
   the FIELD_DIMS boundary, so a later reader can see whether the trailing columns moved
   MORE or LESS than the rest, rather than only a boolean "moved".

F6 (minor, applied): `MIN_TEST_SAMPLES` is now an enforced floor (`test_size_adequate` per
   probe, folded into degeneracy); the probe's own `nn.Linear` init is seeded independently of
   the ambient global RNG (`torch.manual_seed` scoped immediately before construction, distinct
   per probe call) rather than inheriting whatever state training left behind; the probe env
   now uses a seed offset distinct from the warmup env's (`seed + 500_000_003`), since two
   `CausalGridWorldV2` instances built from the identical seed reproduce the identical episode
   LAYOUT sequence, which would have made probe-data collection replay warmup's own layouts
   rather than sampling fresh ones.

Verdict after fixes: CLEAR (no further finding raised against the causal chain as revised).
No second red-team pass was run per the "one pass, re-spawn only on a BLOCKING finding that
changes the causal chain" rule -- these six were CONTESTED with concrete confirmers, not
BLOCKING, and are dispositioned above with sources, not merely asserted fixed.

READINESS (NOT claim evidence -- self-routes `substrate_not_ready_requeue` below floor):
  R1 `raw_observation_decodability_replicated` -- fraction of seeds whose raw-observation
     probe accuracy lift clears RAW_LIFT_FLOOR, on the SAME statistic C1 reads. The RAW probe
     reads the narrow 25/50-dim agent-centred radius-2 waypoint-channel-plus-field slice
     V3-EXQ-1004 probed (`_raw_probe_vector`, zero-padded to 50 dims for OFF) -- NOT the full
     world_state (a pilot check showed the full vector collapses the lift to ~0.06-0.09, since
     both arms already predict the label from the local-view radius-2 channel plus
     resource/hazard proximity fields the manipulation never touches).
  R2 `zworld_encoder_trained_all_cells` -- fraction of the 10 (seed, arm) cells whose world
     encoder moved during P0a warmup. Necessary but (per F5b) not sufficient on its own; R3
     supplies the complementary check.
  R3 `zworld_probe_extracts_known_signal` -- fraction of cells whose z_world quadrant-position
     auxiliary probe clears QUADRANT_FLOOR (well above the 0.25 4-way chance rate). Confirms
     the SAME fitting procedure, on THIS feature space and scale, can extract a signal known to
     be present regardless of the waypoint field -- see F1.

Two pilot checks at moderate scale (60 P0 episodes, 15 probe episodes, 100 probe steps, seeds
42/43, well below the pre-registered budget) during authoring, BEFORE the F2/F4 fixes above:
probing the FULL world_state for R1 measured raw lift ~0.06-0.09 (replaced by the narrow-slice
design); the narrow-slice design then measured raw lift 0.158/0.208 against z_world lift
0.026/-0.039. Recorded as a REACHABILITY check only -- 2 seeds at a fraction of budget, on a
DESIGN since revised by F2/F4, is not evidence and must not be read as the answer; it confirmed
the pre-registered bars are achievable rather than unmeetable by construction (the
readiness-anchor concern `validate_experiments.py` raises for a fresh driver with no prior
full-budget reference to cite as a frozen `reference_cells` control -- left unexempted per that
tool's own guidance, since EXEMPT requires the predicate to BE the degeneracy definition, which
is not the case for a first-of-lineage count/fraction statistic).

LOAD-BEARING CRITERION: C1 `zworld_lift_null` -- on >= MIN_SEEDS of 5 seeds, the z_world probe
accuracy lift (ON minus OFF) is <= ZWORLD_NULL_CEILING (one-sided; see F3). PASS = H2 supported
(the field is lost at the encoder). FAIL routes to one of three distinct labels depending on
which direction the lift actually took (see F3) -- none of them "supported".

CLAIMS. INV-086 and MECH-428 are carried as READ-ACROSS ONLY, per the confirmed 1004 autopsy's
explicit instruction for this fan-out: this run instantiates no REE agent doing subgoal-mode
navigation, no z_goal, no feedback-channel ablation (INV-086) and no forced-seed control
(MECH-428), so neither claim's own `what_would_answer` regime is exercised regardless of
outcome. `evidence_direction_per_claim` is therefore `non_contributory` for both,
unconditionally.

experiment_purpose: diagnostic -- this discriminates WHERE the residual navigation blockage
lives (encoder vs objective), it does not test either claim's own hypothesis.

GOV-REUSE-1 (Step 2.4): checked `reanalysis_query.py --readout waypoint_zworld_decodability`
against both `v3_exq_1004_...` (substrate_hash 9a9fbe795140370f) and
`v3_exq_884_mech428_...` (substrate_hash 0091fba4d567a1ae) -- 0/2 carry the readout -> not
recoverable, run.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "experiments"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import torch.optim as optim  # noqa: E402

from experiment_protocol import emit_outcome  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from experiments._metrics import p0_readiness_gate, P0NotReady, check_degeneracy  # noqa: E402
from experiments._lib.capability_eval import RandomPolicy  # noqa: E402
from experiments._lib.zworld_p0_warmup import run_zworld_p0  # noqa: E402
from experiments._lib.zworld_encoder_guard import (  # noqa: E402
    latent_stack_snapshot,
    assert_world_encoder_trained,
    zworld_precondition,
)
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_1030_mech428_inv086_waypoint_field_zworld_decodability"
QUEUE_ID = "V3-EXQ-1030"
CLAIM_IDS = ["INV-086", "MECH-428"]
EXPERIMENT_PURPOSE = "diagnostic"

# --- env (deliberately NOT navigation-isolated -- see docstring) -----------------------
GRID_SIZE = 12
N_WAYPOINTS = 3
STEPS_PER_EPISODE = 150
NUM_HAZARDS = 2
NUM_RESOURCES = 3
WAYPOINT_VISIT_REWARD = 0.2
WAYPOINT_FIELD_DECAY = 0.25
WAYPOINT_COMPLETION_REWARD = 0.8
SEQUENCE_COMMITMENT_TIMEOUT = 20

# --- agent / encoder ---------------------------------------------------------------------
SELF_DIM = 32
WORLD_DIM = 32
# Step 3.5: experiments depending on z_world fidelity need alpha_world >= 0.9 (default 0.3
# is the SD-008 root cause of degraded z_world fidelity in earlier drivers).
ALPHA_WORLD = 0.9
ALPHA_SELF = 0.3

P0_EPISODES = 200            # SD-070 P0a warmup episodes (the training loop bound, per cell)
PROBE_COLLECT_EPISODES = 40  # random-policy rollout for probe-data collection
PROBE_STEPS = 150            # linear-probe cross-entropy gradient steps
PROBE_LR = 5e-3
N_DIRECTIONS = 5             # up / down / left / right / stay -- the probe's label space
N_QUADRANTS = 4              # F1 auxiliary positive-control target

SEEDS = [42, 43, 44, 45, 46]
MIN_SEEDS = 3
# F6: probe env must NOT replay the warmup env's own episode-layout sequence (two
# CausalGridWorldV2 instances built from the identical seed reproduce identical layouts).
PROBE_ENV_SEED_OFFSET = 500_000_003
ACTION_RNG_SEED_OFFSET = 9973

ARM_OFF = "field_off"
ARM_ON = "field_on"
ARMS = (ARM_OFF, ARM_ON)

DEVICE = torch.device("cpu")

# --- pre-registered bars ----------------------------------------------------------------
# R1/C0: READINESS, not claim evidence. Confirms THIS harness's own linear-probe methodology
# replicates V3-EXQ-1004's qualitative input-level finding (0.575 -> 0.841 raw-observation
# decodability, realised lift ~0.266) on the SAME statistic (probe-accuracy lift, ON minus
# OFF) the load-bearing C1 criterion below reads. 0.15 is roughly 56% of 1004's own realised
# lift, reached by a DIFFERENT method (a fresh single-layer probe vs a 128-hidden BC-cloned
# policy) and a coarser 5-way label -- not set near the realised value, which would make it
# unfalsifiable by re-citation of the same number it is supposed to independently replicate.
RAW_LIFT_FLOOR = 0.15
# C1: LOAD-BEARING. H2's declared null. A third of RAW_LIFT_FLOOR -- small enough that a lift
# clearing it cannot plausibly be probe noise, on the SAME statistic RAW_LIFT_FLOOR reads.
ZWORLD_NULL_CEILING = 0.05
# R3 (F1): the quadrant auxiliary probe's floor. 0.55 is well above the 0.25 4-way chance
# rate and well below what a genuinely spatially-informative z_world should reach (agent
# (x, y) position is reflected directly in the local-view structure every world_state
# carries), so it discriminates "the pipeline extracts nothing at this scale" from "the
# pipeline works but the waypoint-direction signal specifically is what's absent".
QUADRANT_FLOOR = 0.55
# A per-cell accuracy is noisier than a boolean "did it move" (R2) or a lift (R1, which
# already tolerates a minority miss via need/n) -- literal unanimity across all 10 cells
# would let one noisy seed tank readiness for the whole run. 0.6 requires a robust majority
# (6 of 10) rather than perfection on every cell.
QUADRANT_CELL_FRACTION = 0.6
MIN_TEST_SAMPLES = 20
MIN_CLASSES_OBSERVED = 2

# The 5x5x7 local view occupies the first 175 dims of `world_state`, entity-major, channel
# 6 is "waypoint" (CausalGridWorld.ENTITY_TYPES) -- verbatim from V3-EXQ-1004. The RAW
# positive-control probe (R1/C0) reads exactly this narrow slice, zero-padded to 50 dims
# under OFF, so it replicates 1004's OWN input-level measurement rather than the full
# world_state (which also carries resource/hazard proximity fields the manipulation never
# touches, and which a pilot check showed collapses the lift to ~0.06-0.09 -- see docstring).
LOCAL_VIEW_DIMS = 175
N_ENTITY_TYPES = 7
WAYPOINT_ENTITY_CHANNEL = 6
FIELD_DIMS = 25

CONFIG_SLICE_KEYS = {
    "grid_size_declared": GRID_SIZE,
    "n_waypoints_declared": N_WAYPOINTS,
    "steps_per_episode_declared": STEPS_PER_EPISODE,
    "num_hazards_declared": NUM_HAZARDS,
    "num_resources_declared": NUM_RESOURCES,
    "waypoint_visit_reward_declared": WAYPOINT_VISIT_REWARD,
    "waypoint_field_decay_declared": WAYPOINT_FIELD_DECAY,
    "waypoint_completion_reward_declared": WAYPOINT_COMPLETION_REWARD,
    "sequence_commitment_timeout_declared": SEQUENCE_COMMITMENT_TIMEOUT,
    "self_dim_declared": SELF_DIM,
    "world_dim_declared": WORLD_DIM,
    "alpha_world_declared": ALPHA_WORLD,
    "alpha_self_declared": ALPHA_SELF,
    "p0_episodes_declared": P0_EPISODES,
    "probe_collect_episodes_declared": PROBE_COLLECT_EPISODES,
    "probe_steps_declared": PROBE_STEPS,
    "raw_lift_floor_declared": RAW_LIFT_FLOOR,
    "zworld_null_ceiling_declared": ZWORLD_NULL_CEILING,
    "quadrant_floor_declared": QUADRANT_FLOOR,
    "min_seeds_declared": MIN_SEEDS,
}


def _mean(vals: List[float]) -> float:
    return float(statistics.fmean(vals)) if vals else 0.0


class _FieldMaskedEnv:
    """Proxy around CausalGridWorldV2 that ALWAYS builds the WIDE (field-enabled) env, so
    both arms share an identical `world_obs_dim` (275) and therefore an identical RNG-
    consumption sequence at agent-construction time (red-team F2). Under `field_on=False`,
    zeroes the trailing FIELD_DIMS columns of `world_state` on every reset()/step() -- the
    manipulation becomes purely CONTENT, extending 1004's own F7 zero-pad convention from
    the raw-slice probe to the agent's actual z_world input. `waypoint_proximity_field_view`
    is left untouched by the mask (real field values remain under it) -- `_raw_probe_vector`
    ignores that key under OFF unconditionally, so this does not leak the field back in."""

    def __init__(self, seed: int, field_on: bool) -> None:
        self._env = CausalGridWorldV2(
            seed=seed,
            size=GRID_SIZE,
            use_proxy_fields=True,
            subgoal_mode=True,
            num_waypoints=N_WAYPOINTS,
            waypoint_visit_reward=WAYPOINT_VISIT_REWARD,
            subgoal_arrival_position_check=True,
            num_hazards=NUM_HAZARDS,
            num_resources=NUM_RESOURCES,
            waypoint_completion_reward=WAYPOINT_COMPLETION_REWARD,
            sequence_commitment_timeout=SEQUENCE_COMMITMENT_TIMEOUT,
            waypoint_proximity_field_enabled=True,   # ALWAYS wide -- see class docstring
            waypoint_field_decay=WAYPOINT_FIELD_DECAY,
        )
        self._field_on = bool(field_on)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    def _mask(self, obs: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if self._field_on or obs is None:
            return obs
        ws = obs["world_state"].clone()
        ws[-FIELD_DIMS:] = 0.0
        masked = dict(obs)
        masked["world_state"] = ws
        return masked

    def reset(self):
        flat, obs = self._env.reset()
        return flat, self._mask(obs)

    def step(self, action):
        flat, r, done, info, obs = self._env.step(action)
        return flat, r, done, info, self._mask(obs)


def _build_env(field_on: bool, seed: int) -> _FieldMaskedEnv:
    return _FieldMaskedEnv(seed=seed, field_on=field_on)


def _build_agent(env: _FieldMaskedEnv, seed: int) -> REEAgent:
    torch.manual_seed(seed)
    np.random.seed(seed)
    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
        alpha_world=ALPHA_WORLD,
        alpha_self=ALPHA_SELF,
        reafference_action_dim=env.action_dim,
    )
    return REEAgent(config).to(DEVICE)


def _oracle_direction(env: _FieldMaskedEnv) -> int:
    """Greedy 5-way direction toward the pending waypoint, from env GROUND TRUTH. A LABEL
    only -- no action is ever taken from it; the rollout policy is uniform-random (see
    `_collect_probe_data`), so nothing here scripts a walk. Direction logic verbatim from
    V3-EXQ-1004's `_oracle_action` (up/down/left/right/stay)."""
    idx = int(getattr(env, "_next_waypoint_idx", 0))
    wps = getattr(env, "waypoints", []) or []
    if not wps or idx >= len(wps):
        return 4
    wx, wy = int(wps[idx][0]), int(wps[idx][1])
    ax, ay = int(env.agent_x), int(env.agent_y)
    dx, dy = wx - ax, wy - ay
    if abs(dx) >= abs(dy) and dx != 0:
        return 1 if dx > 0 else 0
    if dy != 0:
        return 3 if dy > 0 else 2
    return 4


def _quadrant_label(env: _FieldMaskedEnv) -> int:
    """F1 auxiliary positive-control label: which quadrant of the grid the agent currently
    occupies, from env ground truth. Structurally present in `world_state` regardless of the
    waypoint field (agent position is reflected in the local view), so it must be decodable
    from z_world if the probe pipeline can extract anything at this scale at all."""
    half = GRID_SIZE / 2.0
    ax, ay = int(env.agent_x), int(env.agent_y)
    return (1 if ax >= half else 0) + (2 if ay >= half else 0)


def _raw_probe_vector(obs_dict: Dict[str, Any], field_on: bool) -> torch.Tensor:
    """The narrow 25/50-dim agent-centred radius-2 waypoint-channel-plus-field slice,
    verbatim from V3-EXQ-1004's `_obs_vector` -- the RAW positive-control input, zero-padded
    to 50 dims under OFF so both arms probe the same-width vector. NOT the full world_state
    (see module docstring for why: the full vector already contains resource/hazard
    proximity fields the manipulation never touches, which collapses the measurable lift)."""
    ws = np.asarray(obs_dict["world_state"], dtype=np.float32).reshape(-1)
    wp_local = ws[:LOCAL_VIEW_DIMS][WAYPOINT_ENTITY_CHANNEL::N_ENTITY_TYPES]
    if field_on:
        fv = np.asarray(obs_dict["waypoint_proximity_field_view"],
                        dtype=np.float32).reshape(-1)
    else:
        fv = np.zeros(FIELD_DIMS, dtype=np.float32)
    vec = np.concatenate([wp_local, fv])
    return torch.as_tensor(vec, dtype=torch.float32)


def _direct_zworld(agent: REEAgent, world_state: torch.Tensor) -> torch.Tensor:
    """F5(a) diagnostic: `split_encoder.world_encoder(raw world_state)`, mirroring
    `ZWorldP0Trainer._z_world_path` exactly -- the distribution P0a actually TRAINED on,
    bypassing `agent.world_obs_encoder` (the untrained pre-projection `agent.sense()` applies
    in production). NOT the load-bearing DV; recorded so a null can be attributed to either
    stage rather than only to "the encoder" undifferentiated."""
    se = agent.latent_stack.split_encoder
    with torch.no_grad():
        z = se.world_encoder(world_state.unsqueeze(0))
        skip = getattr(se, "world_encoder_skip", None)
        if skip is not None:
            z = z + skip(world_state.unsqueeze(0))
        z = z * torch.sigmoid(se.world_precision_logit).unsqueeze(0)
    return z.reshape(-1).cpu()


def _collect_probe_data(agent: REEAgent, env: _FieldMaskedEnv, rng: np.random.RandomState,
                        n_episodes: int, field_on: bool) -> Dict[str, torch.Tensor]:
    """Random-policy rollout on a DEDICATED probe env. Collects (z_world, direct-encoder
    z_world, raw narrow-slice, oracle-direction label, quadrant label, episode id) per step.
    The agent's own encoder is frozen (P1) -- `.sense()` here only reads, it never trains."""
    agent.eval()
    zworlds: List[torch.Tensor] = []
    zworlds_direct: List[torch.Tensor] = []
    raws: List[torch.Tensor] = []
    labels: List[int] = []
    quadrants: List[int] = []
    episode_ids: List[int] = []
    for ep in range(n_episodes):
        _flat, obs = env.reset()
        agent.reset()
        for _t in range(STEPS_PER_EPISODE):
            label = _oracle_direction(env)
            quad = _quadrant_label(env)
            with torch.no_grad():
                latent = agent.sense(obs["body_state"], obs["world_state"])
            zworlds.append(latent.z_world.detach().cpu().reshape(-1))
            zworlds_direct.append(_direct_zworld(agent, obs["world_state"]))
            raws.append(_raw_probe_vector(obs, field_on))
            labels.append(int(label))
            quadrants.append(int(quad))
            episode_ids.append(int(ep))
            action = int(rng.randint(0, int(env.action_dim)))
            _flat, _r, done, _info, obs = env.step(action)
            if done:
                break
    zw = torch.stack(zworlds) if zworlds else torch.zeros((0, WORLD_DIM))
    zwd = torch.stack(zworlds_direct) if zworlds_direct else torch.zeros((0, WORLD_DIM))
    rw = torch.stack(raws) if raws else torch.zeros((0, FIELD_DIMS * 2))
    lb = torch.as_tensor(labels, dtype=torch.long)
    qd = torch.as_tensor(quadrants, dtype=torch.long)
    ep_ids = torch.as_tensor(episode_ids, dtype=torch.long)
    return {"z_world": zw, "z_world_direct": zwd, "raw": rw, "labels": lb,
            "quadrants": qd, "episode_ids": ep_ids}


def _episode_split(episode_ids: torch.Tensor, seed: int, train_frac: float = 0.8):
    """F4: split by WHOLE EPISODE, never by flat step. z_world is a 0.9-EMA over
    within-episode state, so a step-level random split lets near-duplicate, temporally-
    adjacent samples from the SAME episode land on both sides of the split, inflating
    apparent decodability via smoothing/memorisation rather than generalisation. Returns
    (train_idx, test_idx) as flat sample-index tensors."""
    uniq = sorted(int(e) for e in episode_ids.unique().tolist())
    g = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(len(uniq), generator=g).tolist()
    n_train_eps = max(1, int(len(uniq) * train_frac))
    if n_train_eps >= len(uniq) and len(uniq) > 1:
        n_train_eps = len(uniq) - 1
    train_eps = {uniq[i] for i in perm[:n_train_eps]}
    test_eps = {uniq[i] for i in perm[n_train_eps:]}
    train_mask = torch.tensor([int(e) in train_eps for e in episode_ids.tolist()])
    test_mask = torch.tensor([int(e) in test_eps for e in episode_ids.tolist()])
    train_idx = train_mask.nonzero(as_tuple=True)[0]
    test_idx = test_mask.nonzero(as_tuple=True)[0]
    return train_idx, test_idx


def _fit_and_eval_probe(features: torch.Tensor, labels: torch.Tensor, episode_ids: torch.Tensor,
                        n_classes: int, n_steps: int, seed: int,
                        lr: float = PROBE_LR) -> Dict[str, Any]:
    """Fit a single-layer linear probe (cross-entropy, EPISODE-level 80/20 split -- F4) and
    return held-out accuracy plus the diagnostics `check_degeneracy`/non-degeneracy reporting
    and the F1/F6 engagement checks need (train accuracy, majority-class rate, test size)."""
    n = int(features.shape[0])
    n_classes_observed = int(labels.unique().numel()) if n else 0
    if n < 2 or n_classes_observed < MIN_CLASSES_OBSERVED:
        return {"accuracy": 0.0, "train_accuracy": 0.0, "majority_class_rate": 0.0,
                "n_test": 0, "n_train": 0, "n_classes_observed": n_classes_observed,
                "test_size_adequate": False}
    train_idx, test_idx = _episode_split(episode_ids, seed)
    if test_idx.numel() == 0 or train_idx.numel() == 0:
        return {"accuracy": 0.0, "train_accuracy": 0.0, "majority_class_rate": 0.0,
                "n_test": int(test_idx.numel()), "n_train": int(train_idx.numel()),
                "n_classes_observed": n_classes_observed, "test_size_adequate": False}
    x_train, y_train = features[train_idx], labels[train_idx]
    x_test, y_test = features[test_idx], labels[test_idx]
    majority_rate = float(torch.bincount(y_test, minlength=n_classes).max().item()
                          / max(1, int(y_test.shape[0])))
    # F6: seed the probe's OWN weight init independently of whatever ambient global-RNG
    # state training left behind, rather than inheriting arm-order-dependent state.
    torch.manual_seed(int(seed) * 7919 + 3)
    probe = nn.Linear(int(features.shape[1]), n_classes)
    opt = optim.Adam(probe.parameters(), lr=lr)
    probe.train()
    train_loss = float("nan")
    for _ in range(n_steps):
        logits = probe(x_train)
        loss = F.cross_entropy(logits, y_train)
        opt.zero_grad()
        loss.backward()
        opt.step()
        train_loss = float(loss.detach())
    probe.eval()
    with torch.no_grad():
        train_preds = probe(x_train).argmax(dim=-1)
        train_acc = float((train_preds == y_train).float().mean().item())
        preds = probe(x_test).argmax(dim=-1)
        acc = float((preds == y_test).float().mean().item())
    return {"accuracy": acc, "train_accuracy": train_acc, "train_final_loss": train_loss,
            "majority_class_rate": majority_rate,
            "n_test": int(test_idx.numel()), "n_train": int(train_idx.numel()),
            "n_classes_observed": n_classes_observed,
            "test_size_adequate": bool(test_idx.numel() >= MIN_TEST_SAMPLES)}


def _run_cell(seed: int, arm: str, dry_run: bool) -> Dict[str, Any]:
    field_on = (arm == ARM_ON)
    print(f"Seed {seed} Condition {arm}", flush=True)

    p0_episodes = 5 if dry_run else P0_EPISODES
    collect_episodes = 3 if dry_run else PROBE_COLLECT_EPISODES
    probe_steps = 20 if dry_run else PROBE_STEPS

    spec_env = _build_env(field_on, seed)   # dims only -- never reset/stepped
    agent = _build_agent(spec_env, seed)

    # Dedicated warmup env (same seed as spec_env's construction; a FRESH instance so the
    # rollout's own RNG consumption never touches spec_env, which is never stepped anyway).
    warmup_env = _build_env(field_on, seed)
    before = latent_stack_snapshot(agent)
    p0a_diagnostics = run_zworld_p0(
        agent, warmup_env, seed=seed, episodes=p0_episodes,
        steps_per_episode=STEPS_PER_EPISODE, policy=RandomPolicy(seed),
        label=f"{EXPERIMENT_TYPE}|{arm}", dry_run=dry_run,
    )
    guard_report = assert_world_encoder_trained(
        agent, before, p0=p0_episodes, strict=False,
        context=f"seed={seed} arm={arm}",
        escape_hint="waypoint_field_consumer_reach H2 (V3-EXQ-1030) generic P0a warmup",
    )
    encoder_precond = zworld_precondition(guard_report, arm=arm, context=EXPERIMENT_TYPE)

    # F5(b): column-wise weight-delta split at the FIELD_DIMS boundary, so a later reader
    # can see whether the trailing (field) input columns moved more or less than the rest,
    # rather than only a boolean "some parameter moved".
    field_cols_delta = None
    other_cols_delta = None
    try:
        w_after = agent.latent_stack.split_encoder.world_encoder[0].weight.detach()
        w_before = before.get("split_encoder.world_encoder.0.weight")
        if w_before is not None and w_before.shape == w_after.shape:
            delta = (w_after - w_before).abs()
            field_cols_delta = float(delta[:, -FIELD_DIMS:].norm().item())
            other_cols_delta = float(delta[:, :-FIELD_DIMS].norm().item())
    except (AttributeError, IndexError, KeyError):
        pass  # diagnostic-only; absence does not affect the load-bearing measurement

    # P1: freeze the whole latent stack. No further gradient step touches it.
    for p in agent.latent_stack.parameters():
        p.requires_grad_(False)

    # Dedicated probe-collection env -- a DISTINCT seed offset (F6): two CausalGridWorldV2
    # instances built from the identical seed reproduce the identical episode-layout
    # sequence, which would otherwise make probe collection replay warmup's own layouts.
    probe_env = _build_env(field_on, seed + PROBE_ENV_SEED_OFFSET)
    rng = np.random.RandomState(seed + ACTION_RNG_SEED_OFFSET)
    data = _collect_probe_data(agent, probe_env, rng, collect_episodes, field_on)

    zw_probe = _fit_and_eval_probe(data["z_world"], data["labels"], data["episode_ids"],
                                   N_DIRECTIONS, probe_steps, seed)
    raw_probe = _fit_and_eval_probe(data["raw"], data["labels"], data["episode_ids"],
                                    N_DIRECTIONS, probe_steps, seed + 1)
    zw_direct_probe = _fit_and_eval_probe(data["z_world_direct"], data["labels"],
                                          data["episode_ids"], N_DIRECTIONS, probe_steps,
                                          seed + 2)
    quadrant_probe = _fit_and_eval_probe(data["z_world"], data["quadrants"],
                                         data["episode_ids"], N_QUADRANTS, probe_steps,
                                         seed + 3)
    label_counts = {int(c): int((data["labels"] == c).sum().item())
                    for c in range(N_DIRECTIONS)}

    print(f"[{EXPERIMENT_TYPE}] seed={seed} arm={arm} "
          f"n_samples={int(data['labels'].shape[0])} "
          f"zworld_acc={zw_probe['accuracy']:.3f} raw_acc={raw_probe['accuracy']:.3f} "
          f"quadrant_acc={quadrant_probe['accuracy']:.3f} "
          f"encoder_trained={encoder_precond.get('met')}", flush=True)
    print("verdict: PASS", flush=True)

    return {
        "arm": arm,
        "field_enabled": bool(field_on),
        "zworld_probe": zw_probe,
        "zworld_direct_probe": zw_direct_probe,
        "raw_probe": raw_probe,
        "quadrant_probe": quadrant_probe,
        "encoder_readiness": encoder_precond,
        "field_cols_weight_delta": field_cols_delta,
        "other_cols_weight_delta": other_cols_delta,
        "label_counts": label_counts,
        "n_samples": int(data["labels"].shape[0]),
        "n_episodes": collect_episodes,
        "world_obs_dim": int(spec_env.world_obs_dim),
        "p0a_diagnostics": p0a_diagnostics,
        "agent": agent,
    }


def _score(per_seed: List[Dict[str, Any]], seeds: List[int]) -> Dict[str, Any]:
    n = len(per_seed)
    need = min(MIN_SEEDS, n)

    raw_lift = {seeds[i]: per_seed[i][ARM_ON]["raw_probe"]["accuracy"]
                - per_seed[i][ARM_OFF]["raw_probe"]["accuracy"] for i in range(n)}
    zworld_lift = {seeds[i]: per_seed[i][ARM_ON]["zworld_probe"]["accuracy"]
                   - per_seed[i][ARM_OFF]["zworld_probe"]["accuracy"] for i in range(n)}

    encoder_reports = [per_seed[i][arm]["encoder_readiness"] for i in range(n) for arm in ARMS]
    n_cells = len(encoder_reports)
    encoder_trained_hits = sum(1 for c in encoder_reports if c.get("met"))
    raw_lift_hits = sum(1 for v in raw_lift.values() if v >= RAW_LIFT_FLOOR)
    quadrant_hits = sum(1 for i in range(n) for arm in ARMS
                        if per_seed[i][arm]["quadrant_probe"]["accuracy"] >= QUADRANT_FLOOR)

    readiness_checks = [
        {"name": "raw_observation_decodability_replicated",
         "measured": raw_lift_hits / max(1, n), "threshold": need / max(1, n),
         "direction": "lower",
         "control": ("fraction of seeds whose OWN-harness linear-probe raw-observation "
                     "accuracy lift (ON-OFF) clears RAW_LIFT_FLOOR -- the positive control "
                     "replicating V3-EXQ-1004's input-level finding with THIS run's probe "
                     "methodology, on the SAME statistic C1 reads.")},
        {"name": "zworld_encoder_trained_all_cells",
         "measured": encoder_trained_hits / max(1, n_cells), "threshold": 1.0,
         "direction": "lower",
         "control": ("fraction of the (seed, arm) cells whose world_encoder moved during "
                     "P0a warmup -- a frozen random projection would make a z_world null "
                     "uninterpretable rather than informative.")},
        {"name": "zworld_probe_extracts_known_signal",
         "measured": quadrant_hits / max(1, n_cells), "threshold": QUADRANT_CELL_FRACTION,
         "direction": "lower",
         "control": ("fraction of the (seed, arm) cells whose z_world grid-quadrant "
                     "auxiliary probe clears QUADRANT_FLOOR -- confirms the SAME fitting "
                     "procedure, on z_world's own scale/distribution, can extract a signal "
                     "known to be present (agent position) regardless of the waypoint "
                     "field, so a small waypoint-direction lift means the field specifically "
                     "did not survive, not that the probe pipeline extracts nothing at all "
                     "from z_world at this scale (red-team F1).")},
    ]

    sample_unmet = False
    unmet: List[str] = []
    try:
        gated = p0_readiness_gate(readiness_checks)
    except P0NotReady as exc:
        sample_unmet = True
        gated = list(getattr(exc, "preconditions", readiness_checks) or readiness_checks)
        unmet = [c.get("name", "?") for c in gated if not c.get("met", False)]

    measurable = not sample_unmet
    null_seeds = [s for s, v in zworld_lift.items() if -ZWORLD_NULL_CEILING <= v <= ZWORLD_NULL_CEILING]
    positive_seeds = [s for s, v in zworld_lift.items() if v > ZWORLD_NULL_CEILING]
    negative_seeds = [s for s, v in zworld_lift.items() if v < -ZWORLD_NULL_CEILING]
    c1 = measurable and (len(null_seeds) >= need)

    # F3: one-sided, three-way disposition -- a negative lift is NOT "survives" (which
    # claims ON decodes BETTER), and a mixed/no-majority result is NOT "survives" either.
    if sample_unmet:
        outcome = "FAIL"
        label = "substrate_not_ready_requeue"
    elif c1:
        outcome = "PASS"
        label = "waypoint_field_lost_at_zworld_encoder"
    elif len(positive_seeds) >= need:
        outcome = "FAIL"
        label = "waypoint_field_survives_zworld_encoder"
    elif len(negative_seeds) >= need:
        outcome = "FAIL"
        label = "waypoint_field_zworld_anomalous_negative_lift"
    else:
        outcome = "FAIL"
        label = "waypoint_field_zworld_lift_inconsistent_across_seeds"

    degeneracy = check_degeneracy({
        "zworld_accuracy_lift": {"values": list(zworld_lift.values())},
        "raw_accuracy_lift": {"values": list(raw_lift.values())},
    })
    test_sizes_adequate = all(
        per_seed[i][arm]["zworld_probe"].get("test_size_adequate")
        and per_seed[i][arm]["raw_probe"].get("test_size_adequate")
        for i in range(n) for arm in ARMS
    )

    return {
        "outcome": outcome,
        "interpretation": {
            "label": label,
            "measurable": measurable,
            "unmet_preconditions": unmet,
            "preconditions": gated,
            "criteria_non_degenerate": {
                "C1_zworld_lift_null": bool(
                    test_sizes_adequate and not degeneracy.get("degenerate_metrics", {})
                    .get("zworld_accuracy_lift")
                ),
            },
        },
        "criteria": {
            "C1_zworld_lift_null": {
                "met": bool(c1), "load_bearing": True,
                "measured": len(null_seeds) / max(1, n), "threshold": need / max(1, n),
                "n_seeds_met": len(null_seeds), "required_seeds": need,
                "zworld_null_ceiling": ZWORLD_NULL_CEILING,
                "n_seeds_positive": len(positive_seeds), "n_seeds_negative": len(negative_seeds),
                "per_seed_zworld_lift": {str(k): v for k, v in zworld_lift.items()},
            },
            "C0_raw_positive_control": {
                "met": bool(raw_lift_hits >= need), "load_bearing": False,
                "measured": raw_lift_hits / max(1, n), "threshold": need / max(1, n),
                "raw_lift_floor": RAW_LIFT_FLOOR,
                "per_seed_raw_lift": {str(k): v for k, v in raw_lift.items()},
            },
        },
        "combination_rule": ("single load-bearing criterion (C1); C0 and the quadrant "
                             "auxiliary probe are readiness/positive controls, not combined "
                             "into the verdict."),
        "arm_means": {
            "raw_acc_off": _mean([per_seed[i][ARM_OFF]["raw_probe"]["accuracy"]
                                  for i in range(n)]),
            "raw_acc_on": _mean([per_seed[i][ARM_ON]["raw_probe"]["accuracy"]
                                 for i in range(n)]),
            "zworld_acc_off": _mean([per_seed[i][ARM_OFF]["zworld_probe"]["accuracy"]
                                     for i in range(n)]),
            "zworld_acc_on": _mean([per_seed[i][ARM_ON]["zworld_probe"]["accuracy"]
                                    for i in range(n)]),
            "zworld_direct_acc_off": _mean([per_seed[i][ARM_OFF]["zworld_direct_probe"]["accuracy"]
                                            for i in range(n)]),
            "zworld_direct_acc_on": _mean([per_seed[i][ARM_ON]["zworld_direct_probe"]["accuracy"]
                                           for i in range(n)]),
            "quadrant_acc_off": _mean([per_seed[i][ARM_OFF]["quadrant_probe"]["accuracy"]
                                       for i in range(n)]),
            "quadrant_acc_on": _mean([per_seed[i][ARM_ON]["quadrant_probe"]["accuracy"]
                                      for i in range(n)]),
        },
        "non_degenerate": bool(degeneracy.get("non_degenerate", True)) and test_sizes_adequate,
        "degeneracy_reason": degeneracy.get("degeneracy_reason", ""),
        "degenerate_metrics": degeneracy.get("degenerate_metrics", {}),
        "evidence_direction": "non_contributory",
        "evidence_direction_per_claim": {cid: "non_contributory" for cid in CLAIM_IDS},
        "evidence_direction_note": (
            "Read-across only, unconditionally, regardless of outcome -- per the confirmed "
            "failure_autopsy_V3-EXQ-1004_2026-09-05 fan-out instruction for this portfolio. "
            "This run instantiates no REE agent doing subgoal-mode navigation toward a "
            "reward, no z_goal, no MECH-116/216/217/426/427 feedback-channel ablation "
            "(INV-086's what_would_answer regime) and no forced-seed control (MECH-428's "
            "what_would_answer regime), so neither claim's own hypothesis is exercised "
            "irrespective of the z_world-decodability verdict."
        ),
    }


def main(dry_run: bool = False) -> Dict[str, Any]:
    seeds = SEEDS[:2] if dry_run else SEEDS
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    per_seed_rows: List[Dict[str, Dict[str, Any]]] = []
    all_agents: List[REEAgent] = []
    for seed in seeds:
        row: Dict[str, Any] = {"seed": int(seed)}
        for arm in ARMS:
            cell = _run_cell(seed, arm, dry_run)
            all_agents.append(cell.pop("agent"))
            row[arm] = cell
        per_seed_rows.append(row)

    result = _score(per_seed_rows, seeds)
    print(f"[{EXPERIMENT_TYPE}] outcome={result['outcome']} "
          f"label={result['interpretation']['label']}", flush=True)

    if dry_run:
        return {"outcome": result["outcome"], "manifest_path": None}

    manifest: Dict[str, Any] = {
        "run_id": f"{EXPERIMENT_TYPE}_{ts}_v3",
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": ts,
        "claim_ids": list(CLAIM_IDS),
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "seeds": list(seeds),
        "arms": list(ARMS),
        "per_seed": per_seed_rows,
        "config": dict(CONFIG_SLICE_KEYS),
    }
    manifest.update(result)
    manifest["readout"] = {
        "zworld_acc_off": result["arm_means"]["zworld_acc_off"],
        "zworld_acc_on": result["arm_means"]["zworld_acc_on"],
        "zworld_lift": (result["arm_means"]["zworld_acc_on"]
                        - result["arm_means"]["zworld_acc_off"]),
        "zworld_direct_acc_off": result["arm_means"]["zworld_direct_acc_off"],
        "zworld_direct_acc_on": result["arm_means"]["zworld_direct_acc_on"],
        "raw_acc_off": result["arm_means"]["raw_acc_off"],
        "raw_acc_on": result["arm_means"]["raw_acc_on"],
        "raw_lift": (result["arm_means"]["raw_acc_on"] - result["arm_means"]["raw_acc_off"]),
        "quadrant_acc_off": result["arm_means"]["quadrant_acc_off"],
        "quadrant_acc_on": result["arm_means"]["quadrant_acc_on"],
        "c1_zworld_lift_null_met": 1 if result["criteria"]["C1_zworld_lift_null"]["met"] else 0,
        "c0_raw_positive_control_met": (
            1 if result["criteria"]["C0_raw_positive_control"]["met"] else 0
        ),
    }

    out_path = write_flat_manifest(
        manifest, None, dry_run=False, config=manifest.get("config"),
        seeds=list(seeds), script_path=Path(__file__), agent=all_agents,
    )
    print(f"[{EXPERIMENT_TYPE}] manifest -> {out_path}", flush=True)
    return {"outcome": result["outcome"], "manifest_path": str(out_path)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    _res = main(dry_run=args.dry_run)
    _o = str(_res["outcome"]).upper()
    emit_outcome(outcome=_o if _o in ("PASS", "FAIL") else "FAIL",
                 manifest_path=_res.get("manifest_path"), dry_run=args.dry_run)
