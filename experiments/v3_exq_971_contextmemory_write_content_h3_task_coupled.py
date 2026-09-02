#!/opt/local/bin/python3
"""
V3-EXQ-971 -- ContextMemory write-address content-discrimination: H3
("task-pressure-required") leg of the frozen hypothesis-space question
`contextmemory_write_content_discrimination`.

Chip: chip-20260830-ctxmem-write-content-h1h4-portfolio

experiment_purpose: diagnostic

hypothesis_space: qid=contextmemory_write_content_discrimination,
hid=H3-task-pressure-required, axis=drive.

H3 ("task-pressure-required"): "No standalone write-side auxiliary loss
suffices; content conditioning needs the READ path's task gradient through
write selection." Design per the frozen registry entry: "Couple the SD-016
read-path retrieval loss through write selection (straight-through) so
end-task pressure reaches the tagger. Null: coupled training still fails the
margin. If confirmed, the frame 'train the tagger with a standalone
auxiliary loss' is wrong."

WHY THIS IS NOT THE SAME LEG AS H1 (read this before touching either script).
H1 (V3-EXQ-970, sibling experiment, same chip) trains write_addr_tagger with
a hand-designed CONTENT-DIVERGENCE objective applied directly to the
tagger's OWN output distribution (a pairwise cosine-dissimilarity loss over
per-example selection probabilities) -- no downstream task is ever consulted,
the loss only asks "do different states get different selection
distributions." H3 instead routes gradient through a REAL downstream task
head (agent.e3.harm_eval_head, already trained elsewhere in this same
harness) and a REAL task label (is_harm, from env.step()'s own harm_signal),
via a straight-through soft-read that simulates "what would a read via this
soft write-address look like." The loss asks "does the CONTENT reachable
through this address help predict something the agent actually needs to
predict" -- task pressure, not a designed auxiliary. MoE importance loss
(956's first attempt) and H1's pairwise-divergence loss are BOTH members of
the "standalone auxiliary loss" family H3's own hypothesis is skeptical of;
H3 tests whether abandoning that whole family in favour of genuine task
gradient is what content-discrimination actually needs.

THE COUPLING MECHANISM (the core engineering of this driver). At each
training step, after computing the SAME write-stream state tensor write()
itself is called with (state = torch.cat([latent.z_self.detach(),
latent.z_world.detach()], dim=-1), confirmed via
`grep -n "context_memory.write(" ree_core/agent.py`):

    scores = agent.e1.context_memory.write_addr_tagger(state)   # (1, num_slots)
    soft_probs = F.softmax(-scores / TAU_ST, dim=-1)             # straight-
                                                                  # through SOFT
                                                                  # address dist,
                                                                  # "lower score
                                                                  # wins" (same
                                                                  # convention
                                                                  # write() and
                                                                  # compute_write_
                                                                  # addressing_loss
                                                                  # both use)
    soft_read_context = soft_probs @ agent.e1.context_memory.memory  # (1, memory_dim)
    harm_pred = agent.e3.harm_eval_head(bridged)                 # see DESIGN
                                                                  # DEVIATION below
    h3_loss = F.binary_cross_entropy_with_logits(harm_pred, is_harm_label)

`soft_probs` is a smooth relaxation of the hard write-address selection
write() itself performs; `soft_read_context` simulates "what would a read
via this address look like"; `h3_loss` measures whether that simulated read
carries task-relevant (harm-predictive) information. `LAMBDA_H3 * h3_loss`
is added to `total = e1_loss + e2_loss [+ LAMBDA_H3 * h3_loss]` before
`optimizer.step()`, using the SAME `optimizer`/`standard_params` group 956
already built (`[p for n, p in agent.named_parameters() if "harm_eval_head"
not in n and "context_memory.memory" not in n]`), which EXCLUDES
harm_eval_head's own parameters -- load-bearing: h3_loss's backward() DOES
populate harm_eval_head.grad (it is part of the graph), but harm_eval_opt
(the SEPARATE optimizer 956's own harness already uses to train
harm_eval_head on the real z_world task) calls harm_eval_opt.zero_grad()
immediately before its OWN h_loss.backward() every time it fires, so that
stale gradient is discarded before it could ever be applied -- harm_eval_head
is updated ONLY by harm_eval_opt.step(), exactly as in 956, so it stays
exactly as good/bad as 956's own harness already trains it.
`is_harm_label` is is_harm = float(harm_signal) < 0, the SAME boolean 956's
own harness already computes at every step from env.step()'s return --
reused directly here, never re-derived.

DESIGN DEVIATION FROM THE LITERAL COUPLING SKETCH ABOVE, recorded here
rather than silently fixed, per this file's own family's standard (see
V3-EXQ-956's own "CORRECTION TO AN EARLIER ASSUMPTION" precedent). Read
literally, `harm_eval_head(soft_read_context)` is a shape-mismatch crash:
soft_read_context has shape (1, memory_dim) with memory_dim=128
(E1Config.hidden_dim, ContextMemory's own memory_dim -- confirmed via
E1DeepPredictor.__init__'s `ContextMemory(..., memory_dim=self.config.
hidden_dim, ...)`), while harm_eval_head is `nn.Sequential(nn.Linear(
world_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1), nn.Sigmoid())`
(ree_core/predictors/e3_selector.py:337-342) and therefore requires a
world_dim=32-wide input. There is no existing substrate component that maps
memory_dim (128) -> world_dim (32) directly: ContextMemory's own
output_proj maps memory_dim -> latent_dim (64, the FULL self+world
concatenation, not decomposed into interpretable self/world halves by
construction) and is used by read(); world_query_proj maps the opposite
direction (world_dim -> memory_dim) and is used by the SD-016 read path.
Neither has the right shape for this interface, and slicing output_proj's
64-wide output in half would be an unprincipled assumption (output_proj's
weight matrix has no structural reason to decompose its OUTPUT along the
same self/world boundary as write_addr_tagger's INPUT).

The fix: `read_bridge`, a dedicated `nn.Linear(memory_dim, world_dim)`
constructed fresh per cell (see _run_cell), whose SOLE job is
dimensionality-matching between ContextMemory's memory_dim and
harm_eval_head's world_dim input contract -- a linear reshaping layer,
structurally the same class of fix as ContextMemory's own existing
output_proj/world_query_proj bridging layers, and carrying no
content-discrimination semantics of its own. Its parameters are included in
the SAME optimizer group as write_addr_tagger (`standard_params`), so
gradient can flow end-to-end from h3_loss into write_addr_tagger via
soft_probs (the object under test) AND into read_bridge itself (interface
plumbing), but -- exactly as intended -- NEVER into harm_eval_head's own
weights (excluded, as above) or into context_memory.memory content (frozen,
see below). `bridged = read_bridge(soft_read_context)` in the sketch above
stands in for the literal `harm_eval_head(soft_read_context)` line; this is
the only deviation from the pseudocode this experiment was designed against.

WHY THIS NEVER TOUCHES write()'s OWN torch.no_grad() BLOCK OR ITS .data
MUTATION (mandatory safety justification -- e1_deep.py's own
compute_write_addressing_loss() docstring explicitly warns about this exact
class of bug): "write() runs entirely under torch.no_grad() and its content
update (self.memory.data[idx] = ...) is a raw .data write with no autograd
version tracking. A soft selection tensor retained FROM inside write() and
consumed by .backward() at some LATER point ... would silently differentiate
through the CURRENT memory values instead of the ones actually seen at that
forward pass -- wrong gradients with no error raised." This driver's h3_loss
computation NEVER calls write(), never reads any tensor produced inside
write()'s no_grad block, and never retains anything across write() calls: it
is an entirely SEPARATE, fresh differentiable forward pass built from
write_addr_tagger(h3_state) [a fresh call, mirroring the identical pattern
compute_write_addressing_loss() itself already uses -- a fresh
write_addr_tagger(states) call, discarded after backward()] and
context_memory.memory read directly as a live nn.Parameter (not via any
value cached from inside write()). Because agent.e1.context_memory.memory is
constructed with requires_grad_(False) (see "436e/f Adam-drift
neutralization" below), reading it in this fresh expression carries NO
gradient into memory CONTENT regardless of whether .detach() is called on it
-- gradient can only flow into whichever OTHER tensor multiplies it, i.e.
soft_probs (and, downstream, read_bridge/harm_eval_head) -- so even a
concurrent write() call mutating self.memory.data on a later step cannot
retroactively corrupt an already-completed backward() from an earlier step;
each step's h3_loss graph is discarded (default retain_graph=False) the
moment that step's total.backward() returns, before the next write() call
even happens.

ARMS (2, all seeds x both arms):
  H3_UNTRAINED  LAMBDA_H3=0.0 -- h3_loss is still COMPUTED every step (for
                logging/measurement only, via a SEPARATE .detach()'d /
                torch.no_grad() forward pass that never touches the graph
                feeding optimizer.step() -- structured as
                `if LAMBDA_H3 > 0.0: total = total + LAMBDA_H3 * h3_loss`,
                never an always-added zero-weighted term, so there is no
                possibility of accidental leakage into the backward graph)
                but NEVER added to `total`, so write_addr_tagger and
                read_bridge receive literally zero gradient from it. This is
                the clean negative control: BOTH arms also set
                contextmemory_write_addressing_loss_weight=0.0 (the
                substrate's OWN V3-EXQ-956 pairwise-diversity write-
                addressing loss is disabled entirely in this experiment --
                H3 isolates its OWN coupling mechanism alone, not combined
                with H1/956's loss), so ANY movement of write_addr_tagger in
                either arm can only come from H3's own h3_loss.
  H3_TRAINED    LAMBDA_H3=1.0 -- h3_loss genuinely added to `total` every
                step; training signal ON.

TAU_ST (straight-through temperature) = 1.0: an undramatic first choice.
No operating point has been tuned for this novel mechanism -- H3's null test
is about whether the coupling MECHANISM works at all, not about tuning it.
LAMBDA_H3 = 1.0 for H3_TRAINED: no established precedent exists for this
novel loss family (unlike H1, which borrows V3-EXQ-907's LAMBDA_DIVERSIFY=0.5
precedent for a structurally analogous pairwise loss). 1.0 matches
harm_eval_head's own unscaled BCE convention (both h3_loss and harm_eval_
opt's h_loss operate on a state -> harm-probability mapping task via the
same head family, so they share a comparable natural scale) -- stated as
reasoning, not claimed precision.

SCHEDULE: TRAINING_EPISODES=100, STEPS_PER_EPISODE=150 -- 956's own full
real (non-toy) schedule, matching V3-EXQ-907's precedent for the same
symmetry-breaking-difficulty class of loss. No synthetic-stream regime (H1's
own leg has one; H3's own frozen registry design is specifically about
END-TASK pressure, which has no synthetic-stream analog -- there is no
"task" to be under pressure from in a synthetic 2-cluster stream). Alternating
safe/dangerous context every CONTEXT_SWITCH_EVERY=5 episodes, identical to
956/943.

SEEDS: [42, 7, 13, 100, 200] -- identical to 956/943, for read-across
comparability.

MECHANICAL NEGATIVE/POSITIVE CONTROL (mirrors 956's own tagger-movement
check): before/after write_addr_tagger.state_dict() comparison across the
whole real training run. H3_UNTRAINED is expected to show ZERO movement
(LAMBDA_H3=0.0 -> h3_loss's term is structurally never added to the
backward graph -> zero gradient ever reaches the tagger, and
contextmemory_write_addressing_loss_weight=0.0 rules out the substrate's own
mechanism as an alternate source of movement). H3_TRAINED is expected to
show nonzero movement, confirming the coupling actually delivers gradient
into the tagger.

POST-TRAINING CONTENT-DISCRIMINATION PROBE: identical instrument to 956 --
after real training completes, each cell's REAL (possibly-trained)
agent.e1.context_memory is switched to agent.eval() (deterministic argmin on
write_addr_tagger's own scores, isolating what the tagger has learned from
Gumbel-noise contamination) and fed a fresh 2-cluster synthetic stream
(n=1500, jitter=0.0078) directly via context_memory.write(state). Per-cluster
occupied-slot sets give a Jaccard exactly as in 956/the contract test.

ACCEPTANCE CRITERIA:
  P0 READINESS: min(n_write_calls across all 10 cells) >= WRITE_CALLS_FLOOR
  (200.0, 956's own floor) -- confirms the writepath genuinely engaged
  during real training on THIS run. On failure, the whole run self-routes to
  `substrate_not_ready_requeue` (956's own P0NotReady block, copied
  verbatim) rather than any substrate verdict.
  H3_READINESS (precondition for the load-bearing criterion's
  interpretability, non-gating on its own): mean_jaccard(H3_UNTRAINED) >=
  0.25 -- otherwise the untrained baseline is already close enough to the
  Jaccard=0 floor that "materially lower" becomes structurally unsatisfiable
  regardless of what training does (956's own C2 READINESS rationale,
  reused verbatim for this leg).
  LOAD-BEARING -- H3_task_coupled_objective_succeeds: PASS iff H3_READINESS
  is met AND mean_jaccard(H3_TRAINED) <= mean_jaccard(H3_UNTRAINED) - 0.25.
  This is the genuinely open question this leg exists to answer; a FAIL here
  (readiness met, margin not cleared) is H3's own registered null -- "coupled
  training still fails the margin" -- a real, scientifically valid outcome,
  not a bug.
  Mechanical negative/positive control criteria (non-gating, descriptive):
  H3_UNTRAINED_tagger_received_zero_gradient (state_dict frozen across all
  5 seeds) and H3_TRAINED_tagger_received_gradient (state_dict moved across
  all 5 seeds).
  Also recorded, purely descriptively (non-gating): mean h3_loss (the
  harm-prediction BCE via the soft-read path) at the START (first episode's
  steps) vs END (last episode's steps) of training for BOTH arms, so a
  reader can see whether the coupling loss itself decreased for H3_TRAINED
  independent of whether Jaccard moved -- "the coupling trained the tagger
  to help harm-prediction but that did not translate into content-
  DISCRIMINATION as measured by the synthetic probe" is a genuinely
  different, more informative FAIL than "nothing happened."
  criteria_non_degenerate: True for H3_task_coupled_objective_succeeds iff
  len(seeds) >= 2 AND H3_READINESS held.

claim_ids: [] -- this experiment tests whether a task-coupled (not
standalone) auxiliary loss suffices for write-address content-
discrimination under substrate_queue's `contextmemory-write-path-
addressing-degeneracy` entry, not a claim hypothesis. Diagnostic-purpose
experiments with claim_ids=[] are excluded from governance confidence
scoring by design. evidence_direction: "non_contributory".

Does NOT flip substrate_queue status regardless of outcome -- that stays a
human/governance disposition, identical to 956's own convention.

sleep_driver_pattern: "N/A (no sleep loop)" -- SD-016 sleep-cycle
consolidation is orthogonal to whether the write-ADDRESS mechanism achieves
content-conditioning under genuine task pressure.

Step 2.5b re-derive brake: N/A. claim_ids is empty (see above).

red-team (fable), 2026-09-02: CONTESTED -> three findings.
  G1 (manipulation cannot reach the DV at full strength) -- FIXED:
  `agent.e3.harm_eval_head` already ends in its own `nn.Sigmoid()`
  (e3_selector.py, "harm in [0, 1]"), so feeding its output through
  `F.binary_cross_entropy_with_logits` (both here and in the pre-existing
  `harm_eval_opt` loop, inherited from V3-EXQ-956) was a double sigmoid:
  the effective probability is confined to sigmoid((0,1)) = (0.5, 0.731)
  with no attainable stationary point, so the loss perpetually saturates
  and its gradient into `read_bridge`/`soft_probs`/`write_addr_tagger`
  collapses. In 956 harm_eval_head was scaffolding so this was inert; here
  it is H3's own gradient-transmission line, so the inherited wart became
  load-bearing. Fixed at both call sites (`h3_loss` and `harm_eval_opt`'s
  `h_loss`) to plain `F.binary_cross_entropy` on the head's already-
  probabilistic output.
  G2 (verdict grid over-claims on FAIL, DISMISSED with this caveat rather
  than fixed): a FAIL is labeled `h3_task_coupled_objective_fails_margin_
  null_confirmed`, but nothing in this design separates "H3 itself is
  false" from "`read_bridge` (a full trainable linear layer in the same
  optimizer group) found a shortcut that reduces h3_loss without engaging
  write_addr_tagger meaningfully" or "context_memory.memory (frozen,
  written by the very address path under study) carries no harm-relevant
  content to route regardless of H3's truth". NOT fixed here: isolating
  this would need a third arm (a detached-scores control, `scores.
  detach()` in the soft-read, to see whether h3_loss drops identically
  without any path into the tagger) -- a bigger design change than this
  pass's remit, and only the FAIL direction is affected (a PASS -- Jaccard
  clearing the margin with the readiness precondition met -- remains fully
  attributable to H3, since read_bridge alone has no mechanism to move a
  DOWNSTREAM synthetic-probe Jaccard it never touches). Disclosed so a FAIL
  outcome here is read as "H3 not confirmed by this design", not "H3
  refuted".
  G3 (gate certifies its own subject, noted, non-blocking): the mechanical
  positive control's threshold (`TAGGER_MOVE_EPS=1e-8`) is loose relative
  to Adam's own step size, so "moved" certifies graph connectivity rather
  than usable signal -- true, but the negative control (Gumbel noise cannot
  move params, since sampling lives inside write()'s own no_grad block) is
  sound, and both controls are already non-gating/descriptive only.

Step 2.4 GOV-REUSE-1 (existing-evidence check): the decisive readout (mean
2-cluster content-discrimination Jaccard for a task-COUPLED gumbel_learned
write_addr_tagger, trained via a straight-through soft-read routed through
agent.e3.harm_eval_head, under real agent training dynamics) is not recorded
anywhere. V3-EXQ-956 recorded the standalone-loss-OFF (GUMBEL_UNTRAINED) and
substrate-own-loss-ON (GUMBEL_TRAINED, the pairwise-diversity loss)
conditions; V3-EXQ-970 (H1, sibling, same chip) records a different
standalone (content-divergence) auxiliary. Neither exercises this driver's
task-coupled mechanism, which did not exist in the substrate before this
chip. Not recoverable; proceeding to run.

Step 2.5c substrate-path overlap (re-checked against the CURRENT
substrate_queue.json, 2026-09-01, not merely cited from 956/970):
  - `contextmemory-write-path-addressing-degeneracy` (ree_core/predictors/
    e1_deep.py::ContextMemory.write, severity=corrupting, status=
    implemented_pending_validation): the entry under test here, not an
    unrelated one -- identical disposition to 956.
  - `mode-governance-engagement` (severity=corrupting, status=
    implemented_pending_validation, substrate_paths include ree_core/
    agent.py whole-file): removed by construction, identical disposition to
    956/970 -- this driver never sets use_salience_coordinator=True (REEConfig's
    own False default is used throughout) and never reads
    agent.operating_mode or any SalienceCoordinator output.
  - `SD-082` (severity=corrupting, status=implemented_pending_validation,
    substrate_paths=['ree_core/pfc/lateral_pfc_analog.py::compute_bias',
    'ree_core/agent.py']): removed by construction -- this driver never sets
    use_lateral_pfc_analog=True.
  - `SD-e1-rollout-consistency-training` (severity=corrupting, status=
    item1_validated_item2_pending, substrate_paths=['ree_core/predictors/
    e1_deep.py::forward', 'ree_core/predictors/e1_deep.py::
    predict_long_horizon']): item1 (forward) is validated and unavoidably
    exercised (agent.sense()/_e1_tick() call E1DeepPredictor.forward());
    item2 (predict_long_horizon) is the pending half, and this driver never
    calls predict_long_horizon (no rollout/horizon planning is performed
    anywhere in this harness) -- removed by construction.
  - Re-checked specifically for THIS driver's additional surface
    (agent.e3.harm_eval_head, ree_core/predictors/e3_selector.py): no open
    `corrupting` entry with status containing "pending_validation" lists any
    path under ree_core/predictors/e3_selector.py. The one entry naming
    e3_selector.py at all
    (['ree_core/predictors/e2_fast.py::rollout_with_world',
    'ree_core/predictors/e3_selector.py::compute_reality_cost',
    'ree_core/utils/config.py']) carries status "implemented" (closed,
    non-blocking, and this driver never calls compute_reality_cost or
    rollout_with_world in any case). No other open corrupting entry's
    substrate_paths overlaps any module this driver imports or exercises.

SUBSTRATE PROPERTIES held constant across both arms (not part of the
manipulation) -- identical to 956's own list:
  - contextmemory_gated_content_write=True (436d write-path repair).
  - sd016_writepath_mode="sense_only".
  - alpha_world=0.9 (SD-008).
  - use_noise_floor=True (MECH-313/ARC-065).
  - context_memory.memory.requires_grad_(False) after construction (436e/f
    Adam-drift neutralization) -- freezes CONTENT against optimizer drift;
    write_addr_tagger stays fully trainable (it is the object under test).
  - contextmemory_write_selection="gumbel_learned" (constructs
    write_addr_tagger in BOTH arms; H3_UNTRAINED differs from H3_TRAINED
    only in whether H3's OWN loss is added to `total`, never in whether the
    tagger exists).
  - contextmemory_write_addressing_loss_weight=0.0 in BOTH arms (956's own
    write-addressing loss mechanism is disabled entirely -- see ARMS above).
  - use_per_stream_vs=True, use_anchor_sets=True, use_sd039_anchor_payload=True.
  - use_salience_coordinator left at its REEConfig default (False).
  - sd016_enabled left at its REEConfig default (False) -- gates only the
    cue-tagger/context-divergence-loss apparatus, irrelevant here.

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
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
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


EXPERIMENT_TYPE = "v3_exq_971_contextmemory_write_content_h3_task_coupled"
QUEUE_ID = "V3-EXQ-971"
CLAIM_IDS: List[str] = []
EXPERIMENT_PURPOSE = "diagnostic"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"

# --- Arms --------------------------------------------------------------- #
# (label, LAMBDA_H3) -- both arms share write_selection="gumbel_learned" and
# contextmemory_write_addressing_loss_weight=0.0 (see module docstring ARMS).
ARMS: List[Tuple[str, float]] = [
    ("H3_UNTRAINED", 0.0),
    ("H3_TRAINED", 1.0),
]
ARM_NAMES = [a[0] for a in ARMS]

SEEDS: List[int] = [42, 7, 13, 100, 200]  # matches V3-EXQ-956/943 / 436-family

# Substrate properties held constant across both arms.
WRITE_SELECTION = "gumbel_learned"
CONTEXTMEMORY_WRITE_ADDRESSING_LOSS_WEIGHT = 0.0  # 956's own loss disabled
CONTEXTMEMORY_GATED_CONTENT_WRITE = True
SD016_WRITEPATH_MODE = "sense_only"
ALPHA_WORLD = 0.9
USE_NOISE_FLOOR = True
NOISE_FLOOR_ALPHA = 0.1
NOISE_FLOOR_MIN_TEMPERATURE = 1.0
BASELINE_TEMPERATURE = 1.0

TRAINING_EPISODES = 100  # matches V3-EXQ-907/956's own real (non-toy) schedule
STEPS_PER_EPISODE = 150
CONTEXT_SWITCH_EVERY = 5
MAX_HARM_BUF = 4000

NUM_SLOTS = 16  # ContextMemory default
LATENT_DIM = 64  # self_dim(32) + world_dim(32) -- must match the probe's own
                 # LATENT_DIM and the driver's own state construction.
WORLD_DIM = 32  # agent.e1.config.world_dim -- harm_eval_head's input width.
MEMORY_DIM = 128  # E1Config.hidden_dim default -- context_memory.memory's
                  # own row width; NOT equal to WORLD_DIM (see "DESIGN
                  # DEVIATION" in the module docstring).

# H3's own coupling-loss hyperparameters (see module docstring for rationale).
TAU_ST = 1.0
LAMBDA_H3_BY_ARM: Dict[str, float] = dict(ARMS)

# P0 readiness floor -- 956's own.
WRITE_CALLS_FLOOR = 200.0

# H3 readiness / load-bearing margin (mirrors 956's C2 convention).
H3_JACCARD_MARGIN = 0.25

# Post-training synthetic 2-cluster probe (956's own instrument).
PROBE_N = 1500
PROBE_JITTER = 0.0078
PROBE_CLUSTERS = 2

TAGGER_MOVE_EPS = 1e-8  # threshold for "did write_addr_tagger's state_dict move"


# ------------------------------------------------------------------ #
# Env / agent helpers (env params reused verbatim from V3-EXQ-956/943)     #
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


def _make_agent(env: CausalGridWorldV2) -> REEAgent:
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
        contextmemory_write_selection=WRITE_SELECTION,
        contextmemory_write_addressing_loss_weight=CONTEXTMEMORY_WRITE_ADDRESSING_LOSS_WEIGHT,
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
        "the post-training synthetic probe and the H3 state construction "
        "would no longer match; update LATENT_DIM before trusting this run."
    )
    assert agent.e1.config.world_dim == WORLD_DIM, (
        f"agent.e1.config.world_dim={agent.e1.config.world_dim} != "
        f"WORLD_DIM={WORLD_DIM} -- update WORLD_DIM (and read_bridge's "
        "output width) before trusting the h3_loss shape assumptions."
    )
    assert agent.e1.context_memory.memory_dim == MEMORY_DIM, (
        f"context_memory.memory_dim={agent.e1.context_memory.memory_dim} != "
        f"MEMORY_DIM={MEMORY_DIM} -- update MEMORY_DIM (and read_bridge's "
        "input width) before trusting the h3_loss shape assumptions."
    )
    assert agent.e1.context_memory.write_addr_tagger is not None, (
        "write_selection='gumbel_learned' did not construct "
        "write_addr_tagger -- config wiring regression."
    )
    # 436e/f Adam-drift neutralization (see module docstring). write() still
    # mutates memory.data directly under its own no_grad block regardless.
    # write_addr_tagger is deliberately NOT frozen -- it is this experiment's
    # own trainable object under test.
    agent.e1.context_memory.memory.requires_grad_(False)
    return agent


def _make_read_bridge(agent: REEAgent) -> nn.Linear:
    """Dimensionality-matching bridge, memory_dim -> world_dim. See module
    docstring "DESIGN DEVIATION" for why this exists and why it carries no
    content-discrimination semantics of its own. Constructed fresh per cell,
    immediately after _make_agent() within the same seeded RNG stream (see
    _run_cell), so its random init is reproducible per (seed, arm) exactly
    like every other module arm_cell's RNG reset governs.
    """
    return nn.Linear(MEMORY_DIM, WORLD_DIM).to(agent.device)


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
# H3's own coupling mechanism -- see module docstring "THE COUPLING        #
# MECHANISM" and "DESIGN DEVIATION" for the full derivation.               #
# ------------------------------------------------------------------ #

def _compute_h3_soft_read_harm_pred(agent: REEAgent, read_bridge: nn.Linear,
                                     h3_state: torch.Tensor) -> torch.Tensor:
    """Straight-through soft-read simulation of a write-address selection,
    consumed by the REAL downstream task head agent.e3.harm_eval_head.

    `h3_state` MUST already be detached by the caller (mirrors write()'s own
    call-site convention and compute_write_addressing_loss()'s own
    "caller-supplied batch of already-detached states" contract) -- this
    function's graph starts fresh from that leaf and never reaches back into
    the encoder/LSTM, and never touches write()'s own torch.no_grad() block
    or its raw .data mutation (see module docstring for the full hazard
    rationale this is designed to avoid).
    """
    context_memory = agent.e1.context_memory
    scores = context_memory.write_addr_tagger(h3_state)            # (1, num_slots)
    # Same "lower score wins" convention write() and compute_write_
    # addressing_loss() both use: negate before softmax.
    soft_probs = F.softmax(-scores / TAU_ST, dim=-1)                # (1, num_slots)
    soft_read_context = soft_probs @ context_memory.memory          # (1, memory_dim)
    # memory.requires_grad_(False) (see _make_agent): gradient flows ONLY
    # through soft_probs (and, downstream, read_bridge), never into memory
    # CONTENT, regardless of whether .detach() is called on it here.
    bridged = read_bridge(soft_read_context)                        # (1, world_dim)
    return agent.e3.harm_eval_head(bridged)                         # (1, 1)


# ------------------------------------------------------------------ #
# Write-address instrumentation -- INSTRUMENTATION FIELDS ONLY, no          #
# re-derivation of the selection expression (architecture doc mandate).    #
# ------------------------------------------------------------------ #

class _WriteSequenceTracker:
    """Records the ordered sequence of slots ContextMemory.write() actually
    mutated, by polling ContextMemory.last_write_index / .slot_write_counts
    (the authoritative instrumentation, maintained in every selection mode)
    after every agent.sense() call -- never by recomputing write()'s own
    scoring expression. Identical to V3-EXQ-956/943's tracker.
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
    robust -- NOT occ_cos, NOT the synthetic-probe Jaccard (a SEPARATE
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
# Post-training content-discrimination probe. Reuses V3-EXQ-956's own      #
# 2-cluster synthetic instrument (which itself reuses the contract test's  #
# _stream/_jaccard) so the measurement is directly comparable.             #
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


def _run_content_probe(agent: REEAgent, seed: int) -> Tuple[float, Dict[int, Set[int]]]:
    """Post-training, EVAL-MODE probe. eval() makes gumbel_learned selection
    deterministic argmin on write_addr_tagger's own (possibly-trained) scores
    -- exactly the isolation V3-EXQ-956 uses, so noise never contaminates
    what this measures.
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
    read_bridge: nn.Linear,
    lambda_h3: float,
    optimizer,
    harm_eval_opt,
    harm_buf_pos: List[torch.Tensor],
    harm_buf_neg: List[torch.Tensor],
    tracker: _WriteSequenceTracker,
    h3_loss_history: List[float],
) -> None:
    """Run a single training episode: sense -> E1 tick -> baseline action
    selection -> step -> H3 coupling loss -> prediction-loss training ->
    harm_eval training.

    DELIBERATE CONVENTION (matches V3-EXQ-956's own runner, see that file
    for the full incident this fixed): calls agent._e1_tick(latent) after
    agent.clock.advance() so agent.compute_prediction_loss() has a
    populated _world_experience_buffer and does not silently return an
    unconditional zero_loss on every step.
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

        # --- H3's own coupling loss (see module docstring) --------------- #
        h3_state = torch.cat(
            [latent.z_self.detach(), latent.z_world.detach()], dim=-1
        )
        is_harm_label = torch.tensor(
            [[1.0 if is_harm else 0.0]], device=agent.device
        )
        if lambda_h3 > 0.0:
            harm_pred_soft = _compute_h3_soft_read_harm_pred(agent, read_bridge, h3_state)
            # Finding G1 fix (red-team, fable, 2026-09-02): harm_eval_head
            # ends in its own nn.Sigmoid() (e3_selector.py), so its output
            # is already a probability, not a logit. binary_cross_entropy_
            # with_logits here was a double-sigmoid: it confines the
            # effective probability to sigmoid((0,1)) = (0.5, 0.731) with no
            # attainable stationary point, saturating and collapsing the
            # gradient this loss is meant to deliver into write_addr_tagger.
            h3_loss = F.binary_cross_entropy(harm_pred_soft, is_harm_label)
        else:
            with torch.no_grad():
                harm_pred_soft = _compute_h3_soft_read_harm_pred(agent, read_bridge, h3_state)
                h3_loss = F.binary_cross_entropy(harm_pred_soft, is_harm_label)
        h3_loss_history.append(float(h3_loss.item()))

        e1_loss = agent.compute_prediction_loss()
        e2_loss = agent.compute_e2_loss()
        total = e1_loss + e2_loss
        if lambda_h3 > 0.0:
            total = total + lambda_h3 * h3_loss
        if total.requires_grad:
            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(
                list(agent.parameters()) + list(read_bridge.parameters()), 1.0
            )
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
            # Finding G1 fix (red-team, fable, 2026-09-02): same
            # double-sigmoid as above -- harm_eval_head already outputs a
            # probability (nn.Sigmoid() as its last layer).
            h_loss = F.binary_cross_entropy(pred, target_t)
            harm_eval_opt.zero_grad()
            h_loss.backward()
            harm_eval_opt.step()

        if done:
            break


def _run_cell(arm_name: str, lambda_h3: float, seed: int,
              base_config_slice: Dict[str, Any], zg: ZGoalStreamAccumulator,
              training_episodes: int, steps_per_episode: int) -> Dict[str, Any]:
    cell_config_slice = {
        **base_config_slice,
        "arm": arm_name,
        "lambda_h3": lambda_h3,
        "tau_st": TAU_ST,
        "contextmemory_write_selection": WRITE_SELECTION,
        "contextmemory_write_addressing_loss_weight": CONTEXTMEMORY_WRITE_ADDRESSING_LOSS_WEIGHT,
    }
    with arm_cell(seed, config_slice=cell_config_slice, script_path=Path(__file__)) as cell:
        env_safe = _make_env_safe(seed)
        env_dang = _make_env_dangerous(seed)
        agent = _make_agent(env_safe)
        read_bridge = _make_read_bridge(agent)

        tagger_init_state = {
            k: v.clone()
            for k, v in agent.e1.context_memory.write_addr_tagger.state_dict().items()
        }

        standard_params = [
            p for n, p in agent.named_parameters()
            if "harm_eval_head" not in n and "context_memory.memory" not in n
        ] + list(read_bridge.parameters())
        harm_eval_params = list(agent.e3.harm_eval_head.parameters())
        optimizer = optim.Adam(standard_params, lr=1e-3)
        harm_eval_opt = optim.Adam(harm_eval_params, lr=1e-4)

        harm_buf_pos: List[torch.Tensor] = []
        harm_buf_neg: List[torch.Tensor] = []
        tracker = _WriteSequenceTracker()
        h3_loss_history: List[float] = []

        agent.train()
        print(f"Seed {seed} Condition {arm_name}", flush=True)
        for ep in range(training_episodes):
            block = ep // CONTEXT_SWITCH_EVERY
            is_safe_ep = (block % 2 == 0)
            env = env_safe if is_safe_ep else env_dang
            _run_episode(
                agent, env, steps_per_episode, read_bridge, lambda_h3,
                optimizer, harm_eval_opt, harm_buf_pos, harm_buf_neg,
                tracker, h3_loss_history,
            )
            if (ep + 1) % 50 == 0 or (ep + 1) == training_episodes:
                print(
                    f"  [train] arm={arm_name} seed={seed} "
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

        n_h3 = len(h3_loss_history)
        ep_len = max(steps_per_episode, 1)
        h3_loss_mean_first_episode = (
            sum(h3_loss_history[:ep_len]) / min(ep_len, n_h3) if n_h3 else None
        )
        h3_loss_mean_last_episode = (
            sum(h3_loss_history[-ep_len:]) / min(ep_len, n_h3) if n_h3 else None
        )

        row: Dict[str, Any] = {
            "arm": arm_name,
            "seed": seed,
            "lambda_h3": lambda_h3,
            "tau_st": TAU_ST,
            **dvs,
            "tagger_params_moved": tagger_params_moved,
            "tagger_max_abs_param_diff": max_abs_diff,
            "probe_2cluster_jaccard": probe_jaccard,
            "probe_2cluster_occupied": {
                str(cid): sorted(slots) for cid, slots in probe_per_cluster.items()
            },
            "h3_loss_mean_first_episode": h3_loss_mean_first_episode,
            "h3_loss_mean_last_episode": h3_loss_mean_last_episode,
        }
        cell.stamp(row)

    passed = dvs["n_occupied_slots"] >= 2
    print(
        f"verdict: {'PASS' if passed else 'FAIL'} "
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
    seeds = SEEDS[:1] if dry_run else SEEDS

    full_config_slice: Dict[str, Any] = {
        "arms": ARM_NAMES,
        "seeds": seeds,
        "training_episodes": training_episodes,
        "steps_per_episode": steps_per_episode,
        "context_switch_every": CONTEXT_SWITCH_EVERY,
        "contextmemory_gated_content_write": CONTEXTMEMORY_GATED_CONTENT_WRITE,
        "contextmemory_write_selection": WRITE_SELECTION,
        "contextmemory_write_addressing_loss_weight": CONTEXTMEMORY_WRITE_ADDRESSING_LOSS_WEIGHT,
        "sd016_writepath_mode": SD016_WRITEPATH_MODE,
        "alpha_world": ALPHA_WORLD,
        "use_noise_floor": USE_NOISE_FLOOR,
        "num_slots": NUM_SLOTS,
        "tau_st": TAU_ST,
        "lambda_h3_by_arm": LAMBDA_H3_BY_ARM,
        "probe_n": PROBE_N,
        "probe_jitter": PROBE_JITTER,
        "probe_clusters": PROBE_CLUSTERS,
    }

    arm_results: List[Dict[str, Any]] = []
    for arm_name, lambda_h3 in ARMS:
        for seed in seeds:
            row = _run_cell(
                arm_name, lambda_h3, seed, full_config_slice, zg,
                training_episodes, steps_per_episode,
            )
            arm_results.append(row)

    by_arm: Dict[str, List[Dict[str, Any]]] = {name: [] for name in ARM_NAMES}
    for row in arm_results:
        by_arm[row["arm"]].append(row)

    # --- P0 readiness: writepath genuinely engaged in every cell ---
    write_call_counts = [row["n_write_calls"] for row in arm_results]
    min_write_calls = min(write_call_counts) if write_call_counts else 0.0
    try:
        preconditions = p0_readiness_gate([
            {
                "name": "writepath_engaged_every_cell",
                "measured": float(min_write_calls),
                "threshold": WRITE_CALLS_FLOOR,
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

    # --- H3 readiness + load-bearing margin, mean Jaccard across seeds ---
    def _mean_probe_jaccard(arm_name: str) -> float:
        vals = [row["probe_2cluster_jaccard"] for row in by_arm[arm_name]]
        return sum(vals) / len(vals) if vals else float("nan")

    mean_jaccard_untrained = _mean_probe_jaccard("H3_UNTRAINED")
    mean_jaccard_trained = _mean_probe_jaccard("H3_TRAINED")

    h3_headroom_met = mean_jaccard_untrained >= H3_JACCARD_MARGIN
    h3_directional_pass = mean_jaccard_trained <= (mean_jaccard_untrained - H3_JACCARD_MARGIN)
    h3_pass = h3_headroom_met and h3_directional_pass

    overall_pass = h3_pass

    # --- Descriptive, non-gating negative/positive controls ---
    n_untrained_seeds_frozen = sum(
        1 for row in by_arm["H3_UNTRAINED"] if row["tagger_params_moved"] is False
    )
    n_trained_seeds_moved = sum(
        1 for row in by_arm["H3_TRAINED"] if row["tagger_params_moved"] is True
    )

    def _mean_field(arm_name: str, field: str) -> Optional[float]:
        vals = [row[field] for row in by_arm[arm_name] if row.get(field) is not None]
        return sum(vals) / len(vals) if vals else None

    criteria = [
        {
            "name": "H3_readiness_untrained_baseline_headroom",
            "load_bearing": False,
            "kind": "readiness",
            "passed": h3_headroom_met,
            "measured": mean_jaccard_untrained,
            "threshold": H3_JACCARD_MARGIN,
            "direction": "lower",
            "note": (
                "Precondition for the load-bearing criterion's "
                "interpretability, not part of the manipulation itself: "
                "mean_jaccard(H3_UNTRAINED) must be >= 0.25 for 'materially "
                "lower' to be satisfiable at all. If unmet, the load-bearing "
                "criterion below reads precondition_unmet rather than a "
                "plain FAIL -- a floor effect is not evidence the coupling "
                "failed."
            ),
        },
        {
            "name": "H3_task_coupled_objective_succeeds",
            "load_bearing": True,
            "passed": h3_pass,
            "precondition_unmet": not h3_headroom_met,
            "directional_result": h3_directional_pass,
            "mean_jaccard_h3_untrained": mean_jaccard_untrained,
            "mean_jaccard_h3_trained": mean_jaccard_trained,
            "required_margin": H3_JACCARD_MARGIN,
            "note": (
                "PASS iff the readiness precondition is met AND "
                "mean_jaccard(H3_TRAINED) <= mean_jaccard(H3_UNTRAINED) - "
                "0.25. This is H3's own registered open question; a FAIL "
                "here (precondition met, directional_result False) is H3's "
                "own null -- 'coupled training still fails the margin' -- a "
                "real, expected-possible scientific outcome, not a bug. A "
                "FAIL with precondition_unmet=True means the comparison "
                "could not be made at all, not that coupled training was "
                "tried and failed."
            ),
        },
        {
            "name": "H3_UNTRAINED_tagger_received_zero_gradient",
            "load_bearing": False,
            "passed": n_untrained_seeds_frozen == len(by_arm["H3_UNTRAINED"]),
            "n_seeds_frozen": n_untrained_seeds_frozen,
            "n_seeds_total": len(by_arm["H3_UNTRAINED"]),
            "note": (
                "Mechanical negative control: LAMBDA_H3=0.0 should leave "
                "write_addr_tagger's state_dict byte-identical to its random "
                "init across the whole real training run (h3_loss's term is "
                "structurally never added to `total`, and "
                "contextmemory_write_addressing_loss_weight=0.0 rules out "
                "956's own mechanism as an alternate source of movement)."
            ),
        },
        {
            "name": "H3_TRAINED_tagger_received_gradient",
            "load_bearing": False,
            "passed": n_trained_seeds_moved == len(by_arm["H3_TRAINED"]),
            "n_seeds_moved": n_trained_seeds_moved,
            "n_seeds_total": len(by_arm["H3_TRAINED"]),
            "note": (
                "Mechanical positive control: LAMBDA_H3=1.0 should move "
                "write_addr_tagger's parameters away from init over the "
                "real training run, regardless of whether that movement "
                "achieves content-discrimination (the load-bearing "
                "criterion above)."
            ),
        },
        {
            "name": "H3_coupling_loss_trend_descriptive",
            "load_bearing": False,
            "passed": True,
            "h3_loss_mean_first_episode_untrained": _mean_field("H3_UNTRAINED", "h3_loss_mean_first_episode"),
            "h3_loss_mean_last_episode_untrained": _mean_field("H3_UNTRAINED", "h3_loss_mean_last_episode"),
            "h3_loss_mean_first_episode_trained": _mean_field("H3_TRAINED", "h3_loss_mean_first_episode"),
            "h3_loss_mean_last_episode_trained": _mean_field("H3_TRAINED", "h3_loss_mean_last_episode"),
            "note": (
                "Purely descriptive (always 'passed' -- this is a "
                "measurement, not a criterion): mean h3_loss (harm-"
                "prediction BCE via the soft-read path) at the start vs end "
                "of training, per arm. Lets a reader distinguish 'the "
                "coupling loss itself never moved' from 'the coupling loss "
                "improved but that did not translate into content-"
                "discrimination as measured by the synthetic probe' -- the "
                "latter is a more informative FAIL than the former."
            ),
        },
    ]
    criteria_non_degenerate = {
        "H3_task_coupled_objective_succeeds": len(seeds) >= 2 and h3_headroom_met,
    }

    if overall_pass:
        label = "h3_task_coupled_write_addressing_content_discrimination_validated"
    elif not h3_headroom_met:
        label = "h3_readiness_precondition_unmet_untrained_baseline_floor_effect"
    else:
        label = "h3_task_coupled_objective_fails_margin_null_confirmed"
    # claim_ids is empty -- this diagnostic validates a SUBSTRATE mechanism,
    # not a claim hypothesis (matches V3-EXQ-956's own convention).
    evidence_direction = "non_contributory"

    status = "PASS" if overall_pass else "FAIL"

    metrics: Dict[str, Any] = {}
    for arm_name in ARM_NAMES:
        rows = by_arm[arm_name]
        metrics[f"n_occupied_slots_per_seed_{arm_name}"] = [r["n_occupied_slots"] for r in rows]
        metrics[f"n_write_calls_per_seed_{arm_name}"] = [r["n_write_calls"] for r in rows]
        metrics[f"entropy_bits_per_seed_{arm_name}"] = [r["entropy_bits"] for r in rows]
        metrics[f"self_repeat_rate_per_seed_{arm_name}"] = [r["self_repeat_rate"] for r in rows]
        metrics[f"round_robin_agreement_per_seed_{arm_name}"] = [r["round_robin_agreement"] for r in rows]
        metrics[f"probe_2cluster_jaccard_per_seed_{arm_name}"] = [r["probe_2cluster_jaccard"] for r in rows]
        metrics[f"mean_probe_2cluster_jaccard_{arm_name}"] = _mean_probe_jaccard(arm_name)
        metrics[f"tagger_max_abs_param_diff_per_seed_{arm_name}"] = [r["tagger_max_abs_param_diff"] for r in rows]

    summary_markdown = (
        f"# {QUEUE_ID} -- ContextMemory H3 (task-pressure-required) write-"
        f"address content discrimination\n\n"
        f"**Status:** {status}  **Label:** {label}\n"
        f"**Purpose:** diagnostic (claim_ids=[]; validates substrate_queue "
        f"contextmemory-write-path-addressing-degeneracy, H3 leg of "
        f"contextmemory_write_content_discrimination)\n\n"
        f"| Arm | mean probe Jaccard | n_write_calls range | tagger moved (n/N) |\n"
        f"|---|---|---|---|\n"
        + "".join(
            f"| {arm_name} "
            f"| {_mean_probe_jaccard(arm_name):.3f} "
            f"| {min(r['n_write_calls'] for r in by_arm[arm_name])}-"
            f"{max(r['n_write_calls'] for r in by_arm[arm_name])} "
            f"| {sum(1 for r in by_arm[arm_name] if r['tagger_params_moved'])}/"
            f"{len(by_arm[arm_name])} |\n"
            for arm_name in ARM_NAMES
        )
        + f"\nH3 readiness (untrained baseline headroom >= {H3_JACCARD_MARGIN}): "
        f"{'MET' if h3_headroom_met else 'UNMET'}\n"
        f"H3 load-bearing (mean Jaccard trained <= untrained - "
        f"{H3_JACCARD_MARGIN}): {'PASS' if h3_pass else 'FAIL'} "
        f"(untrained={mean_jaccard_untrained:.3f}, trained={mean_jaccard_trained:.3f})\n"
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

    full_config = {
        "arms": ARM_NAMES,
        "seeds": SEEDS,
        "training_episodes": TRAINING_EPISODES,
        "steps_per_episode": STEPS_PER_EPISODE,
        "context_switch_every": CONTEXT_SWITCH_EVERY,
        "contextmemory_gated_content_write": CONTEXTMEMORY_GATED_CONTENT_WRITE,
        "contextmemory_write_selection": WRITE_SELECTION,
        "contextmemory_write_addressing_loss_weight": CONTEXTMEMORY_WRITE_ADDRESSING_LOSS_WEIGHT,
        "sd016_writepath_mode": SD016_WRITEPATH_MODE,
        "alpha_world": ALPHA_WORLD,
        "use_noise_floor": USE_NOISE_FLOOR,
        "num_slots": NUM_SLOTS,
        "tau_st": TAU_ST,
        "lambda_h3_by_arm": LAMBDA_H3_BY_ARM,
        "h3_jaccard_margin": H3_JACCARD_MARGIN,
        "write_calls_floor": WRITE_CALLS_FLOOR,
        "probe_n": PROBE_N,
        "probe_jitter": PROBE_JITTER,
        "probe_clusters": PROBE_CLUSTERS,
        "latent_dim": LATENT_DIM,
        "world_dim": WORLD_DIM,
        "memory_dim": MEMORY_DIM,
    }

    out_path = write_flat_manifest(
        result,
        dry_run=args.dry_run,
        config=full_config,
        seeds=SEEDS,
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
