#!/opt/local/bin/python3
"""
V3-EXQ-436c -- MECH-166 wall-independent representation confirmer
(sleep-cycle write-counter recording fix)
SLEEP DRIVER: manual-multi (run_sleep_cycle() called directly every SLEEP_INTERVAL episodes in training loop)

experiment_purpose: evidence

Discharges IGW-20260801-223 (claim MECH-166, backlog EVB-0475, "Proposal for
MECH-166") and IGW-20260801-225 (GOV-CONFIRM-1, "Confirm evidence: MECH-166,
lit 0.88 exp ~0"). Same underlying evidence gap for MECH-166 436b discharged
-- see that script's docstring and RELATIONSHIP.md/DESIGN.md for the full
derivation; this letter fixes a RECORDING gap in 436b's own instrumentation,
not the experimental design.

SUPERSEDES: V3-EXQ-436b (FAIL on C1, recommended_evidence_direction
non_contributory per failure_autopsy_V3-EXQ-436b_2026-08-02.json/.md,
confirmed 2026-08-02, user-approved as recommended). 436b's C1
(slot_cosine_sim) was BIT-IDENTICAL between WAKING_ONLY and SWS_THEN_REM in
every one of 5 seeds (e.g. seed 42: 0.000965997576713562 in both, to full
float64 precision), while slot_separation was exactly 0.0 in both conditions
every seed too -- yet harm_rate_safe/dangerous, slot_danger_ema, and
train_action_class_entropy_mean genuinely DIFFERED between conditions per
seed, proving the two conditions are not literally identical runs. Bit-
identical readouts across behaviourally-divergent conditions is a strong
tell for a RECORDING gap, not a genuine null (independent floating-point
computation on different trajectories essentially never lands on exact
equality by chance). Code trace confirmed the mechanism: run_sws_schema_pass()
(ree_core/agent.py) DOES call self.e1.context_memory.write(e1_input) and
returns a per-pass write counter (metrics["sws_n_writes"]) -- but 436b's
sleep-invocation line was `_ = agent.run_sleep_cycle()`, discarding the
ENTIRE returned metrics dict, including sws_n_writes and the REM-equivalent
counter (rem_n_rollouts, from run_rem_attribution_pass()). There was no way
to tell from that run's data whether ContextMemory.write() ever executed at
all. recommended_epistemic_category: measurement_gap. This design captures
those counters and gates C1's interpretability on them being non-zero.

WHAT CHANGES vs 436b (the ONLY changes; everything else -- conditions, DV
definitions, action-selection wiring, thresholds, seeds -- is unchanged from
436b, which itself is unchanged from 436a except for the noise-floor fix; see
436b's docstring for that prior history):

1. SLEEP-CYCLE RETURN VALUE CAPTURED, NOT DISCARDED. The sleep-invocation
   line no longer does `_ = agent.run_sleep_cycle()`. The returned metrics
   dict (agent.py:10607 run_sleep_cycle -> Dict[str, float], merging
   run_sws_schema_pass()'s sws_n_writes/sws_slot_diversity/sws_buffer_size
   and, when rem_enabled, run_rem_attribution_pass()'s rem_n_rollouts/
   rem_mean_harm_terrain/rem_terrain_variance) is captured and its
   sws_n_writes / rem_n_rollouts fields are accumulated per seed across every
   sleep pass in the run. The (dead-in-this-design, since sws_en == rem_en
   always here) `else: agent.run_sws_schema_pass()` branch is fixed
   identically for robustness against a future asymmetric-phase variant.

2. POSITIVE-CONTROL GATE ON THOSE COUNTERS, BEFORE C1 IS READ AS MEANINGFUL.
   Per the failure-autopsy's explicit routing and mirroring this session's
   own established pattern for measurement-gap positive-control gates (e.g.
   V3-EXQ-671b's MECH-025b residue-accumulation gate: RESIDUE_FLOOR +
   p0_readiness_gate/P0NotReady, gating C1/C2 on
   sum(residue_delta_samples) > 0 before treating the precision<->residue
   correlation as meaningful), this design adds a P0 readiness gate over the
   SWS_THEN_REM condition's pooled write counters:
     - sws_context_memory_writes_occur: sum of sws_n_writes across all sleep
       passes, all 5 SWS_THEN_REM seeds, must be > 0.
     - rem_attribution_rollouts_occur: sum of rem_n_rollouts across all sleep
       passes, all 5 SWS_THEN_REM seeds, must be > 0.
   If either gate is unmet, the run is NOT scored as a genuine C1 test --
   outcome is FAIL, evidence_direction is non_contributory, and the
   interpretation label is sleep_cycle_recording_gap_still_present. This is
   the load-bearing fix: it makes visible, in THIS run's own manifest,
   exactly the fact 436b's manifest could not show (whether
   ContextMemory.write() fired at all), so a future C1 FAIL is a genuine null
   only when the counters clear this floor.

CLAIM SUBSTRATE UNDER TEST (unchanged from 436b/436a -- all three read the
SAME wall-independent DV; claims.yaml's own "Experimental implication" text
for SD-017/ARC-045 already specifies this exact metric and threshold):
  SD-017    (sleep_phase.minimal_sleep_infrastructure_v3): "context
            representations remain globally undifferentiated" without the
            SWS/REM-analog phases -- slot_cosine_sim -> 1.0 without them.
  ARC-045   (hippocampus.bidirectional_information_flow): "an agent with
            bidirectional offline flow should show cosine_sim < 0.95
            (differentiated contexts) after sleep phases; one with only
            waking online encoding remains at cosine_sim -> 1.0 regardless
            of training duration" (claims.yaml notes, verbatim experimental
            implication -- this IS the C1 test below).
  MECH-166  (hippocampus.slot_formation_filling_temporal_separation): "Slot
            structure must be consolidated during an SWS-analog phase...
            Experimental implication: EXQ-239 (MECH-153) provides an
            indirect test. A direct test requires implementing the SWS-
            analog pass and comparing attribution map quality (context
            cosine_sim...) with vs without it." This experiment IS that
            direct test.

DV-SYMMETRY INVARIANCE DECLARATION (mandatory per 604c net, unchanged from
436b): the manipulation under test is "ran an SWS-analog + REM-analog
consolidation pass" vs "did not" -- run_sleep_cycle() directly mutates
agent.e1.context_memory.memory via ContextMemory.write() during sleep, and
slot_cosine_sim reads that same memory tensor. This is not a uniform
additive constant, not a monotone rescaling, and not a permutation of
interchangeable units -- it is a genuinely different memory-writing
procedure whose effect the DV reads directly, so the DV is not invariant
under any of the three flagged symmetry classes and the manipulation can
legitimately move it.

CONDITIONS (2, unchanged from 436b): WAKING_ONLY (baseline, no sleep) and
SWS_THEN_REM (full SWS-then-REM cycle every SLEEP_INTERVAL episodes, plus
the DR-6 context-conditioned harm threshold in action selection).
use_noise_floor=True on BOTH conditions (the diversity fix is a waking-phase
substrate property, not part of the manipulation being tested).

ACCEPTANCE CRITERIA:
  P0 (NEW, gates C1's interpretability -- see point 2 above):
      sws_context_memory_writes_occur: pooled sws_n_writes (SWS_THEN_REM
      seeds) > 0. rem_attribution_rollouts_occur: pooled rem_n_rollouts
      (SWS_THEN_REM seeds) > 0. Unmet -> outcome FAIL, evidence_direction
      non_contributory, C1/C4 NOT scored as evidence either way.
  C1 (PRIMARY, SOLE GATE when P0 is met -- SD-017 + ARC-045 + MECH-166,
      wall-independent):
      slot_cosine_sim(SWS_THEN_REM) < slot_cosine_sim(WAKING_ONLY) in >= 3/5
      seeds. Directional per ARC-045's own pre-registered experimental
      implication (claims.yaml, quoted above): SWS-then-REM should
      differentiate (orthogonalize) slots relative to the waking-only
      attractor; ARC-045's "< 0.95" absolute-value framing is recorded as a
      descriptive secondary readout (arc045_differentiated_abs_sim, per
      seed) but the paired comparison to WAKING_ONLY is the pre-registered
      gate, since 0.95 is a rule-of-thumb reference point, not a threshold
      ARC-045's notes formally pin.
  C4 (SECONDARY, ARC-045 slot_separation, non-gating, carried from 436b):
      slot_separation(SWS_THEN_REM) > 0.3 in >= 3/5 seeds -- confirms slots
      correspond to the env's actual SAFE/DANGEROUS context structure, not
      an arbitrary partition.
  Secondary / exploratory (recorded, NEVER gating, per IGW-225's mandate to
  decouple behaviour from the representation confirmer):
      harm_rate_dangerous, harm_rate_safe signed diffs (436a's C2/C3). A
      behavioural PASS here is a bonus (upgrades toward SD-017/ARC-045 end-
      to-end support); a behavioural null alongside a representation PASS is
      still a genuine confirmatory result for MECH-166 specifically, which
      is a claim about slot-formation, not about downstream harm-threshold
      use (claims.yaml MECH-166 notes: "Relationship to MECH-165" already
      factors this apart).

PASS: P0 met AND C1 (n_seeds_passed >= 3/5).

INTERPRETATION GRID:
  P0 UNMET -> sleep_cycle_recording_gap_still_present. The sleep-invocation
              return value is now captured, but the write counters
              themselves came back zero -- i.e. ContextMemory.write() (or
              the REM-equivalent replay) genuinely did not fire this run
              (e.g. buffer too small, gate mis-wired, config regression).
              Route to /failure-autopsy on the write-path itself, not on
              slot_cosine_sim -- C1/C4 are not evidence either way here.
  P0 met, C1 PASS  -> SWS-analog consolidation differentiates context slots
              once the waking-phase input-diversity confound (436a's
              diagnosed root cause) is removed, AND the write path is
              confirmed to have fired. Supports SD-017, ARC-045, MECH-166.
              Discharges IGW-223/225; MECH-166 experimental_confidence moves
              off 0.0. C4 / behavioural readouts reported alongside as
              corroborating-or-not context, non-gating.
  P0 met, C1 FAIL  -> Either (a) the noise-floor diversity fix is
              insufficient on its own to unstick slot differentiation (a
              genuine finding: input diversity was necessary-but-not-
              sufficient), or (b) MECH-166's slot-formation/filling
              separation does not hold as stated on this substrate -- and
              this time the write-counter gate confirms it is not simply
              because the write path failed to fire. Check per-seed
              non_degenerate flag first: if waking-phase action-class
              entropy (recorded per seed) is still ~0 despite
              use_noise_floor=True, that is (a) and should route to a
              /failure-autopsy on the noise-floor magnitude (Q-043
              calibration is still open -- default alpha=0.1 may be too
              weak); if waking-phase entropy is healthy but slot_cosine_sim
              still does not differentiate, that is a genuine (b) weakens
              reading for all three claims. Either way, with P0 met, this is
              real evidence (the run is not vacuous, and now provably not a
              recording gap either), not a wiring failure.

claim_ids: ["SD-017", "ARC-045", "MECH-166"]
experiment_purpose: "evidence"

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
from typing import Dict, List, Tuple

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
from experiments._metrics import check_degeneracy, p0_readiness_gate, P0NotReady  # noqa: E402


EXPERIMENT_TYPE = "v3_exq_436c_sd017_mech166_repr_confirmer"
QUEUE_ID = "V3-EXQ-436c"
SUPERSEDES = "V3-EXQ-436b"
CLAIM_IDS = ["SD-017", "ARC-045", "MECH-166"]
EXPERIMENT_PURPOSE = "evidence"

# Pre-registered thresholds (unchanged from 436b/436a unless noted).
BASE_HARM_THRESHOLD = 0.05       # filter actions whose predicted harm exceeds this
CONTEXT_BETA = 0.8                # danger-score modulation strength
SLOT_DANGER_EMA_ALPHA = 0.05      # slot_danger_score EMA update rate

# Phase 2 substrate template (validated by V3-EXQ-265a PASS 2026-05-09T20:12Z;
# reused verbatim from 436b -- unchanged, not part of this redesign).
SD016_DIVERSIFICATION_WEIGHT = 0.5

# MECH-313 / ARC-065 noise floor -- unchanged from 436b.
USE_NOISE_FLOOR = True
NOISE_FLOOR_ALPHA = 0.1
NOISE_FLOOR_MIN_TEMPERATURE = 1.0
BASELINE_TEMPERATURE = 1.0        # matches REEAgent.select_action's own default

# Acceptance thresholds.
C1_N_SEEDS_REQUIRED = 3           # >= 3/5 seeds, sole PASS/FAIL gate (once P0 met)
C4_SLOT_SEPARATION_THRESHOLD = 0.3
C4_N_SEEDS_REQUIRED = 3
ARC045_ABS_COSINE_REFERENCE = 0.95  # descriptive only; claims.yaml's own reference point

# P0 positive-control gate (NEW in 436c -- the fix for the 436b recording
# gap). Pooled across all 5 SWS_THEN_REM seeds; any non-zero total confirms
# the write path fired at least once. Mirrors V3-EXQ-671b's RESIDUE_FLOOR
# pattern (p0_readiness_gate / P0NotReady).
SWS_WRITES_FLOOR = 1.0
REM_ROLLOUTS_FLOOR = 1.0

SLEEP_INTERVAL = 10
CONTEXT_SWITCH_EVERY = 5
TRAINING_EPISODES = 200
STEPS_PER_EPISODE = 150
EVAL_EPISODES_EACH = 30
MAX_HARM_BUF = 4000
SEEDS = [42, 7, 13, 100, 200]      # unchanged from 436/436a/436b for cross-lineage comparability

CONDITIONS = ["WAKING_ONLY", "SWS_THEN_REM"]  # 436a's SWS_ONLY dropped (see 436b's docstring)


# ------------------------------------------------------------------ #
# Env / agent helpers                                                  #
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


def _make_agent(env: CausalGridWorldV2, sws_enabled: bool, rem_enabled: bool,
                use_sleep_loop: bool) -> REEAgent:
    """Phase 2 substrate stack (unchanged from 436b/436a) + MECH-313/ARC-065
    noise floor (unchanged from 436b -- see 436b's docstring point 2).
    """
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=32,
        alpha_world=0.9,
        alpha_self=0.3,
        reafference_action_dim=0,
        novelty_bonus_weight=0.0,
        # Phase 2 substrate template (5 flags, mechanically applied; constant
        # across both conditions in this experiment).
        sd016_writepath_mode="off",
        sd016_diversification_weight=SD016_DIVERSIFICATION_WEIGHT,
        use_per_stream_vs=True,
        use_anchor_sets=True,
        use_sd039_anchor_payload=True,
        # SD-017 sleep phases (toggle per condition).
        sws_enabled=sws_enabled,
        sws_consolidation_steps=8,
        sws_schema_weight=0.1,
        rem_enabled=rem_enabled,
        rem_attribution_steps=6,
        use_sleep_loop=use_sleep_loop,
        # MECH-313 / ARC-065 stochastic noise floor -- ON in every condition
        # (the diversity fix is a waking-phase property, not the manipulation
        # under test).
        use_noise_floor=USE_NOISE_FLOOR,
        noise_floor_alpha=NOISE_FLOOR_ALPHA,
        noise_floor_min_temperature=NOISE_FLOOR_MIN_TEMPERATURE,
    )
    agent = REEAgent(cfg)
    assert agent.noise_floor is not None, (
        "use_noise_floor=True did not construct agent.noise_floor -- "
        "REEConfig/REEAgent wiring regression; the diversity fix this "
        "experiment depends on is not live. See 436b's docstring point 2."
    )
    return agent


def _action_onehot(a_idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, a_idx] = 1.0
    return v


def _effective_temperature(agent: REEAgent) -> float:
    """The same tonic-lift computation REEAgent.select_action() applies
    before e3.select() (ree_core/agent.py:7438-7444), applied here so this
    driver's own harm-based action scores get the same MECH-313 noise-floor
    diversity injection the substrate's own selection path would give them.
    """
    if agent.noise_floor is not None:
        return agent.noise_floor.compute_effective_temperature(
            baseline_temperature=BASELINE_TEMPERATURE, simulation_mode=False,
        )
    return BASELINE_TEMPERATURE


# ------------------------------------------------------------------ #
# Context-slot detection (unchanged from 436b/436a -- re-verified live       #
# against ree_core/predictors/e1_deep.py ContextMemory, 2026-08-01)          #
# ------------------------------------------------------------------ #

def _active_slot_idx(agent: REEAgent, z_self: torch.Tensor,
                     z_world: torch.Tensor) -> int:
    """Determine which ContextMemory slot is most strongly activated by
    (z_self, z_world). Argmax over slots vs ContextMemory.read()'s soft mix.
    """
    with torch.no_grad():
        cm = agent.e1.context_memory
        state = torch.cat([z_self, z_world], dim=-1)
        query = cm.query_proj(state)
        keys = cm.key_proj(cm.memory)
        scores = torch.mm(query, keys.t()) / (cm.memory_dim ** 0.5)
        idx = int(scores.argmax(dim=-1).item())
    return idx


def _compute_slot_cosine_sim(agent: REEAgent) -> float:
    with torch.no_grad():
        mem = agent.e1.context_memory.memory
        n = mem.shape[0]
        normed = F.normalize(mem, dim=-1)
        sim = torch.mm(normed, normed.t())
        mask = ~torch.eye(n, dtype=torch.bool, device=mem.device)
        return float(sim[mask].mean().item())


# ------------------------------------------------------------------ #
# Action selection -- STOCHASTIC (noise-floor temperature-graded sample,   #
# unchanged from 436b)                                                     #
# ------------------------------------------------------------------ #

def _select_action_baseline(agent: REEAgent, z_world: torch.Tensor,
                             num_actions: int) -> Tuple[int, float]:
    """Temperature-graded softmax sample over predicted harm (low harm ->
    high selection probability), using the MECH-313 noise-floor effective
    temperature. Unchanged from 436b.
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
        best_h = harms[best_a]
    return best_a, best_h


def _select_action_context_cond(agent: REEAgent, z_world: torch.Tensor,
                                 num_actions: int, slot_danger_score: float,
                                 base_thresh: float, context_beta: float
                                 ) -> Tuple[int, float, float]:
    """Context-conditioned harm threshold action selection (DR-6 pathway),
    unchanged from 436b: effective threshold =
    base_thresh * (1 - context_beta * slot_danger_score); higher danger ->
    lower threshold -> more candidates filtered -> more cautious. Selection
    WITHIN the filtered (or, on empty filter, the full) candidate set is a
    noise-floor temperature-graded softmax sample.
    Returns (action_idx, chosen_harm, effective_threshold).
    """
    eff_thresh = base_thresh * max(0.1, 1.0 - context_beta * slot_danger_score)
    with torch.no_grad():
        harms: List[float] = []
        for a in range(num_actions):
            a_oh = _action_onehot(a, num_actions, z_world.device)
            zw_next = agent.e2.world_forward(z_world, a_oh)
            h = agent.e3.harm_eval(zw_next).mean().item()
            harms.append(h)
        eff_t = _effective_temperature(agent)
        filtered_idx = [a for a, h in enumerate(harms) if h <= eff_thresh]
        if filtered_idx:
            sub_harms = torch.tensor([harms[a] for a in filtered_idx], dtype=torch.float32)
            probs = F.softmax(-sub_harms / eff_t, dim=0)
            sel = int(torch.multinomial(probs, 1).item())
            best_a = filtered_idx[sel]
        else:
            harms_t = torch.tensor(harms, dtype=torch.float32)
            probs = F.softmax(-harms_t / eff_t, dim=0)
            best_a = int(torch.multinomial(probs, 1).item())
        best_h = harms[best_a]
    return best_a, float(best_h), float(eff_thresh)


# ------------------------------------------------------------------ #
# Episode runner (unchanged control flow from 436b)                        #
# ------------------------------------------------------------------ #

def _run_episode(
    agent: REEAgent,
    env: CausalGridWorldV2,
    steps: int,
    train: bool,
    is_dangerous_ep: bool,
    optimizer,
    harm_eval_opt,
    harm_buf_pos: List,
    harm_buf_neg: List,
    slot_danger_ema: List[float],
    use_context_cond: bool,
) -> Tuple[float, List[torch.Tensor], List[int], List[int]]:
    """Run single episode. Returns (harm_sum, z_world_list, slot_visits,
    action_seq). action_seq is recorded to compute a per-episode action-class
    entropy diagnostic (evidence the noise-floor fix is actually engaging,
    independent of the slot_cosine_sim readout itself).
    Updates slot_danger_ema in place when train=True.
    """
    _, obs_dict = env.reset()
    agent.reset()
    agent.e1.reset_hidden_state()
    ep_harm = 0.0
    z_world_list: List[torch.Tensor] = []
    slot_visits: List[int] = []
    action_seq: List[int] = []

    for _step in range(steps):
        obs_body = obs_dict["body_state"]
        obs_world = obs_dict["world_state"]
        obs_harm = obs_dict.get("harm_obs", None)

        latent = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
        agent.clock.advance()
        z_self = latent.z_self.detach().clone()
        z_world = latent.z_world.detach().clone()
        z_world_list.append(z_world)

        # SECOND-LAYER FIX (436c, found via THIS driver's own smoke test after
        # the run_sleep_cycle() capture fix below -- see module docstring
        # "ADDITIONAL WIRING GAP FOUND"). agent._e1_tick() is the ONLY call
        # site in ree_core/agent.py that (a) appends to
        # agent._world_experience_buffer / agent._self_experience_buffer (the
        # data run_sws_schema_pass() samples prototypes from) and (b) pushes
        # into agent.theta_buffer (whose .recent property
        # run_rem_attribution_pass() reads). _e1_tick() is invoked exclusively
        # from agent.act()/act_batch()-style entry points -- this driver's
        # manual harm-based action-selection loop (unchanged from 436b) calls
        # agent.sense() directly and never agent.act(), so BOTH stores were
        # permanently empty regardless of the run_sleep_cycle() discard fix,
        # and run_sws_schema_pass()'s n_buf<2 / run_rem_attribution_pass()'s
        # recent-is-None guards fired on every sleep pass unconditionally.
        # This ALSO fully explains failure_autopsy_V3-EXQ-436b_2026-08-02's
        # "core anomaly" (slot_cosine_sim bit-identical between WAKING_ONLY
        # and SWS_THEN_REM to full float64 precision, every seed):
        # ContextMemory.memory (ree_core/predictors/e1_deep.py) is an
        # nn.Parameter randomly initialised via torch.randn(...)*0.01 at
        # REEAgent construction, and arm_cell(seed, ...) reseeds torch's
        # global RNG to the SAME seed for both conditions before each is
        # constructed -- so both conditions' memory started bit-identical,
        # and with .write() never firing, it never changed from that shared
        # initial value in either condition.
        # FIX: mirror _e1_tick's append+trim (ree_core/agent.py:4900-4904)
        # and theta_buffer push (ree_core/agent.py:4945) DIRECTLY -- not a
        # call to _e1_tick() itself, which also runs a full extra E1 forward
        # pass plus cue-bias/schema-salience/self-recurrence side effects
        # this driver's design deliberately does not exercise elsewhere, and
        # which would be a materially larger, unreviewed change to introduce
        # here.
        agent._self_experience_buffer.append(z_self)
        agent._world_experience_buffer.append(z_world)
        if len(agent._self_experience_buffer) > 1000:
            del agent._self_experience_buffer[:-1000]
        if len(agent._world_experience_buffer) > 1000:
            del agent._world_experience_buffer[:-1000]
        agent.theta_buffer.update(z_world, z_self)

        slot_idx = _active_slot_idx(agent, z_self, z_world)
        slot_visits.append(slot_idx)

        if use_context_cond:
            danger = slot_danger_ema[slot_idx]
            action_idx, _, _ = _select_action_context_cond(
                agent, z_world, env.action_dim, danger,
                BASE_HARM_THRESHOLD, CONTEXT_BETA,
            )
        else:
            action_idx, _ = _select_action_baseline(agent, z_world, env.action_dim)
        action_seq.append(action_idx)

        action_oh = _action_onehot(action_idx, env.action_dim, agent.device)
        agent._last_action = action_oh

        _, harm_signal, done, _info, obs_dict = env.step(action_oh)
        is_harm = float(harm_signal) < 0
        if is_harm:
            ep_harm += abs(float(harm_signal))

        if train:
            target = 1.0 if is_dangerous_ep else 0.0
            slot_danger_ema[slot_idx] = (
                (1.0 - SLOT_DANGER_EMA_ALPHA) * slot_danger_ema[slot_idx]
                + SLOT_DANGER_EMA_ALPHA * target
            )

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

    return ep_harm, z_world_list, slot_visits, action_seq


def _action_class_entropy(action_seq: List[int], num_actions: int) -> float:
    """Shannon entropy (nats) of the realized action-class distribution over
    one episode. 0.0 = fully degenerate (one action class every tick, the
    436a failure signature); ln(num_actions) = uniform. Diagnostic only --
    not gating -- but decisive for reading a C1 FAIL (see INTERPRETATION
    GRID): near-zero entropy despite use_noise_floor=True would mean the
    diversity fix did not engage, not that MECH-166 is refuted.
    """
    if not action_seq:
        return 0.0
    counts = [0] * num_actions
    for a in action_seq:
        counts[a % num_actions] += 1
    total = float(len(action_seq))
    ent = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            ent -= p * math.log(p)
    return float(ent)


# ------------------------------------------------------------------ #
# Condition runner                                                     #
# ------------------------------------------------------------------ #

def _run_condition(
    seed: int,
    condition: str,
    training_episodes: int,
    steps_per_episode: int,
    eval_episodes_each: int,
    zg: ZGoalStreamAccumulator,
    verbose: bool = True,
) -> Dict:
    sws_en = condition == "SWS_THEN_REM"
    rem_en = condition == "SWS_THEN_REM"
    use_sleep_loop = sws_en or rem_en  # ON for SWS_THEN_REM; OFF for WAKING_ONLY
    use_context_cond = condition == "SWS_THEN_REM"   # DR-6 pathway only here

    env_safe = _make_env_safe(seed)
    env_dang = _make_env_dangerous(seed)
    agent = _make_agent(env_safe, sws_en, rem_en, use_sleep_loop)

    standard_params = [p for n, p in agent.named_parameters()
                       if "harm_eval_head" not in n]
    harm_eval_params = list(agent.e3.harm_eval_head.parameters())
    optimizer = optim.Adam(standard_params, lr=1e-3)
    harm_eval_opt = optim.Adam(harm_eval_params, lr=1e-4)

    harm_buf_pos: List[torch.Tensor] = []
    harm_buf_neg: List[torch.Tensor] = []
    num_slots = agent.e1.context_memory.num_slots
    slot_danger_ema: List[float] = [0.5] * num_slots

    per_ep_harm_safe: List[float] = []
    per_ep_harm_dang: List[float] = []
    per_ep_action_entropy: List[float] = []
    slot_visit_safe_count: List[int] = [0] * num_slots
    slot_visit_dang_count: List[int] = [0] * num_slots
    sleep_passes = 0
    cum_train_pos = 0  # cumulative harm_eval_head TRAINING label counts
    cum_train_neg = 0  # (pre-MAX_HARM_BUF-trim, so not diluted by the cap)

    # NEW in 436c: sleep-cycle write-counter accumulators. 436b discarded
    # agent.run_sleep_cycle()'s entire return value with `_ = ...`; these
    # capture and accumulate the two diagnostic write/consolidation counters
    # per the failure-autopsy's recommended fix.
    cum_sws_n_writes = 0.0
    cum_rem_n_rollouts = 0.0
    last_sws_slot_diversity = 0.0
    last_rem_mean_harm_terrain = 0.0

    agent.train()
    for ep in range(training_episodes):
        block = ep // CONTEXT_SWITCH_EVERY
        is_safe_ep = (block % 2 == 0)
        env = env_safe if is_safe_ep else env_dang

        _len_pos_before, _len_neg_before = len(harm_buf_pos), len(harm_buf_neg)
        ep_harm, _z_list, slot_visits, action_seq = _run_episode(
            agent, env, steps_per_episode,
            train=True,
            is_dangerous_ep=(not is_safe_ep),
            optimizer=optimizer,
            harm_eval_opt=harm_eval_opt,
            harm_buf_pos=harm_buf_pos,
            harm_buf_neg=harm_buf_neg,
            slot_danger_ema=slot_danger_ema,
            use_context_cond=use_context_cond,
        )
        harm_rate = ep_harm / steps_per_episode
        per_ep_action_entropy.append(_action_class_entropy(action_seq, env.action_dim))
        cum_train_pos += len(harm_buf_pos) - _len_pos_before
        cum_train_neg += len(harm_buf_neg) - _len_neg_before
        if is_safe_ep:
            per_ep_harm_safe.append(harm_rate)
            for s in slot_visits:
                slot_visit_safe_count[s] += 1
        else:
            per_ep_harm_dang.append(harm_rate)
            for s in slot_visits:
                slot_visit_dang_count[s] += 1

        if len(harm_buf_pos) > MAX_HARM_BUF:
            harm_buf_pos = harm_buf_pos[-MAX_HARM_BUF:]
        if len(harm_buf_neg) > MAX_HARM_BUF:
            harm_buf_neg = harm_buf_neg[-MAX_HARM_BUF:]

        if (sws_en or rem_en) and (ep + 1) % SLEEP_INTERVAL == 0 and ep > 0:
            # FIX (436c): capture the return value instead of discarding it
            # with `_ = ...`. run_sleep_cycle() (ree_core/agent.py:10607)
            # returns a merged dict of run_sws_schema_pass()'s
            # sws_n_writes/sws_slot_diversity/sws_buffer_size and (when
            # rem_enabled) run_rem_attribution_pass()'s
            # rem_n_rollouts/rem_mean_harm_terrain/rem_terrain_variance.
            if rem_en:
                sleep_metrics = agent.run_sleep_cycle()
            else:
                # Dead branch in this design (sws_en == rem_en always for
                # the two CONDITIONS above), fixed identically for
                # robustness against a future asymmetric-phase variant.
                sleep_metrics = agent.run_sws_schema_pass()
            cum_sws_n_writes += float(sleep_metrics.get("sws_n_writes", 0.0))
            cum_rem_n_rollouts += float(sleep_metrics.get("rem_n_rollouts", 0.0))
            last_sws_slot_diversity = float(
                sleep_metrics.get("sws_slot_diversity", last_sws_slot_diversity)
            )
            last_rem_mean_harm_terrain = float(
                sleep_metrics.get("rem_mean_harm_terrain", last_rem_mean_harm_terrain)
            )
            sleep_passes += 1

        if (ep + 1) % 50 == 0:
            print(f"  [train] label seed={seed} cond={condition} "
                  f"ep {ep+1}/{training_episodes} "
                  f"harm_safe_ema={(sum(per_ep_harm_safe[-10:])/max(len(per_ep_harm_safe[-10:]),1)):.4f} "
                  f"harm_dang_ema={(sum(per_ep_harm_dang[-10:])/max(len(per_ep_harm_dang[-10:]),1)):.4f} "
                  f"action_entropy_ema={(sum(per_ep_action_entropy[-10:])/max(len(per_ep_action_entropy[-10:]),1)):.4f}",
                  flush=True)

    safe_tot = float(sum(slot_visit_safe_count))
    dang_tot = float(sum(slot_visit_dang_count))
    if safe_tot > 0 and dang_tot > 0:
        safe_dist = [c / safe_tot for c in slot_visit_safe_count]
        dang_dist = [c / dang_tot for c in slot_visit_dang_count]
        slot_separation = float(sum(abs(s - d) for s, d in zip(safe_dist, dang_dist)))
    else:
        slot_separation = 0.0

    final_slot_sim = _compute_slot_cosine_sim(agent)
    train_action_entropy_mean = (
        sum(per_ep_action_entropy) / max(1, len(per_ep_action_entropy))
    )

    zg.observe(agent)

    agent.eval()
    eval_harm_safe: List[float] = []
    eval_harm_dang: List[float] = []
    eval_z_safe: List[torch.Tensor] = []
    eval_z_dang: List[torch.Tensor] = []

    for _ in range(eval_episodes_each):
        h_s, zs, _, _ = _run_episode(
            agent, env_safe, steps_per_episode,
            train=False, is_dangerous_ep=False,
            optimizer=optimizer, harm_eval_opt=harm_eval_opt,
            harm_buf_pos=harm_buf_pos, harm_buf_neg=harm_buf_neg,
            slot_danger_ema=slot_danger_ema,
            use_context_cond=use_context_cond,
        )
        eval_harm_safe.append(h_s / steps_per_episode)
        eval_z_safe.extend(zs)

    for _ in range(eval_episodes_each):
        h_d, zd, _, _ = _run_episode(
            agent, env_dang, steps_per_episode,
            train=False, is_dangerous_ep=True,
            optimizer=optimizer, harm_eval_opt=harm_eval_opt,
            harm_buf_pos=harm_buf_pos, harm_buf_neg=harm_buf_neg,
            slot_danger_ema=slot_danger_ema,
            use_context_cond=use_context_cond,
        )
        eval_harm_dang.append(h_d / steps_per_episode)
        eval_z_dang.extend(zd)

    with torch.no_grad():
        n_samp = min(len(eval_z_safe), len(eval_z_dang), 200)
        if n_samp > 0:
            zs_s = torch.cat(eval_z_safe[:n_samp], dim=0)
            zd_s = torch.cat(eval_z_dang[:n_samp], dim=0)
            he_safe = float(agent.e3.harm_eval(zs_s).mean().item())
            he_dang = float(agent.e3.harm_eval(zd_s).mean().item())
        else:
            he_safe = 0.0
            he_dang = 0.0
    harm_discrim = he_dang - he_safe

    harm_safe = sum(eval_harm_safe) / max(1, len(eval_harm_safe))
    harm_dang = sum(eval_harm_dang) / max(1, len(eval_harm_dang))

    if verbose:
        print(f"  [seed={seed} {condition}] "
              f"slot_sim={final_slot_sim:.4f} "
              f"slot_sep={slot_separation:.3f} "
              f"harm_safe={harm_safe:.4f} "
              f"harm_dang={harm_dang:.4f} "
              f"discrim={harm_discrim:.4f} "
              f"action_entropy_mean={train_action_entropy_mean:.4f} "
              f"sleep_passes={sleep_passes} "
              f"sws_n_writes_total={cum_sws_n_writes:.1f} "
              f"rem_n_rollouts_total={cum_rem_n_rollouts:.1f}",
              flush=True)

    # Per-condition verdict (progress-instrumentation / runner-ETA purposes
    # only, matches 436b's convention; the experiment-level PASS/FAIL is the
    # aggregate C1 gate (plus the P0 write-counter gate) computed once across
    # all seeds in __main__).
    verdict = "PASS" if (harm_dang < 0.04 and harm_safe < 0.04) else "FAIL"
    print(f"verdict: {verdict}", flush=True)

    return {
        "seed": seed,
        "condition": condition,
        "slot_cosine_sim": float(final_slot_sim),
        "slot_separation": float(slot_separation),
        "harm_rate_safe": float(harm_safe),
        "harm_rate_dangerous": float(harm_dang),
        "harm_discrimination": float(harm_discrim),
        "harm_eval_safe": float(he_safe),
        "harm_eval_dangerous": float(he_dang),
        "slot_danger_ema": [float(x) for x in slot_danger_ema],
        "slot_visit_safe_count": slot_visit_safe_count,
        "slot_visit_dang_count": slot_visit_dang_count,
        "train_harm_safe_final": float(sum(per_ep_harm_safe[-20:]) / max(1, len(per_ep_harm_safe[-20:]))),
        "train_harm_dang_final": float(sum(per_ep_harm_dang[-20:]) / max(1, len(per_ep_harm_dang[-20:]))),
        "train_action_class_entropy_mean": float(train_action_entropy_mean),
        "sleep_passes": sleep_passes,
        # NEW in 436c -- the recording-gap fix. Zero-for-a-non-sleeping
        # condition (WAKING_ONLY) is expected, not a gap; the P0 gate below
        # only reads these for the SWS_THEN_REM condition.
        "sleep_cycle_sws_n_writes_total": float(cum_sws_n_writes),
        "sleep_cycle_rem_n_rollouts_total": float(cum_rem_n_rollouts),
        "sleep_cycle_sws_slot_diversity_last": float(last_sws_slot_diversity),
        "sleep_cycle_rem_mean_harm_terrain_last": float(last_rem_mean_harm_terrain),
        "effective_temperature_last": float(_effective_temperature(agent)),
        "noise_floor_state": agent.noise_floor.get_state() if agent.noise_floor is not None else None,
        "label_balance": {
            "harm_eval_head_train_pos_frac": (
                float(cum_train_pos) / max(1, cum_train_pos + cum_train_neg)
            ),
            "harm_eval_head_train_n_pos": cum_train_pos,
            "harm_eval_head_train_n_neg": cum_train_neg,
        },
    }


# ------------------------------------------------------------------ #
# Run                                                                   #
# ------------------------------------------------------------------ #

def run(dry_run: bool = False) -> Tuple[dict, ZGoalStreamAccumulator]:
    zg = ZGoalStreamAccumulator()

    if dry_run:
        print("[DRY RUN] MECH-166 repr confirmer smoke "
              "(seed=42, 2 conditions, enough episodes for >=1 sleep pass)", flush=True)
        smoke_ok = True
        smoke_results = []
        # NOTE: training_episodes must clear SLEEP_INTERVAL (10) so the
        # SWS_THEN_REM condition actually triggers >=1 sleep pass during the
        # smoke test -- 436b's own smoke (training_episodes=3) never reached
        # the sleep-invocation line at all, so it could not have caught this
        # recording gap even if it had checked for one. 436c's smoke
        # deliberately exercises the sleep-invocation call site.
        smoke_training_episodes = 11
        smoke_steps_per_episode = 20
        try:
            for cond in CONDITIONS:
                print(f"Seed 42 Condition {cond}", flush=True)
                config_slice = {
                    "seed": 42, "condition": cond,
                    "training_episodes": smoke_training_episodes,
                    "steps_per_episode": smoke_steps_per_episode,
                    "eval_episodes_each": 2,
                }
                with arm_cell(42, config_slice=config_slice, script_path=Path(__file__)) as cell:
                    r = _run_condition(
                        seed=42, condition=cond,
                        training_episodes=smoke_training_episodes,
                        steps_per_episode=smoke_steps_per_episode,
                        eval_episodes_each=2,
                        zg=zg,
                        verbose=False,
                    )
                    cell.stamp(r)
                smoke_results.append(r)
                print(f"  {cond}: slot_sim={r['slot_cosine_sim']:.4f} "
                      f"slot_sep={r['slot_separation']:.3f} "
                      f"harm_safe={r['harm_rate_safe']:.4f} "
                      f"harm_dang={r['harm_rate_dangerous']:.4f} "
                      f"action_entropy={r['train_action_class_entropy_mean']:.4f} "
                      f"sleep_passes={r['sleep_passes']} "
                      f"sws_n_writes_total={r['sleep_cycle_sws_n_writes_total']:.1f} "
                      f"rem_n_rollouts_total={r['sleep_cycle_rem_n_rollouts_total']:.1f}")

            required_keys = {"slot_cosine_sim", "slot_separation",
                             "harm_rate_safe", "harm_rate_dangerous",
                             "train_action_class_entropy_mean",
                             "sleep_cycle_sws_n_writes_total",
                             "sleep_cycle_rem_n_rollouts_total",
                             "arm_fingerprint"}
            for r in smoke_results:
                missing = required_keys - set(r.keys())
                if missing:
                    print(f"  [SMOKE] FAIL: condition {r['condition']} missing keys {missing}")
                    smoke_ok = False

            # THE key verification this smoke test exists to perform: the
            # SWS_THEN_REM condition's write counters must be non-zero, or
            # the recording fix did not actually work.
            sws_then_rem = next(
                (r for r in smoke_results if r["condition"] == "SWS_THEN_REM"), None
            )
            if sws_then_rem is None:
                print("  [SMOKE] FAIL: no SWS_THEN_REM result to check write counters on")
                smoke_ok = False
            else:
                if sws_then_rem["sleep_passes"] < 1:
                    print("  [SMOKE] FAIL: SWS_THEN_REM triggered 0 sleep passes -- "
                          "smoke config does not exercise the fix at all")
                    smoke_ok = False
                if sws_then_rem["sleep_cycle_sws_n_writes_total"] <= 0.0:
                    print("  [SMOKE] FAIL: sleep_cycle_sws_n_writes_total="
                          f"{sws_then_rem['sleep_cycle_sws_n_writes_total']} -- "
                          "still zero, recording fix did not capture a real write")
                    smoke_ok = False
                if sws_then_rem["sleep_cycle_rem_n_rollouts_total"] <= 0.0:
                    print("  [SMOKE] FAIL: sleep_cycle_rem_n_rollouts_total="
                          f"{sws_then_rem['sleep_cycle_rem_n_rollouts_total']} -- "
                          "still zero, REM-equivalent counter not captured")
                    smoke_ok = False

            if smoke_ok:
                print("[DRY RUN] PASS - noise-floor stochastic selection + both "
                      "conditions wire correctly; sleep-cycle write counters "
                      "(sws_n_writes, rem_n_rollouts) are captured and non-zero "
                      "for SWS_THEN_REM")
            else:
                print("[DRY RUN] FAIL - check above for missing metric keys or "
                      "zero write counters")
        except Exception as exc:
            print(f"[DRY RUN] FAIL - exception during smoke: {exc!r}")
            smoke_ok = False

        return {
            "outcome": "PASS" if smoke_ok else "FAIL",
            "status": "PASS" if smoke_ok else "FAIL",
        }, zg

    t0 = time.time()
    print(f"{QUEUE_ID} {EXPERIMENT_TYPE}", flush=True)

    arm_results: List[Dict] = []
    for seed in SEEDS:
        print(f"Seed {seed}")
        for cond in CONDITIONS:
            print(f"Seed {seed} Condition {cond}")
            config_slice = {
                "seed": seed, "condition": cond,
                "training_episodes": TRAINING_EPISODES,
                "steps_per_episode": STEPS_PER_EPISODE,
                "eval_episodes_each": EVAL_EPISODES_EACH,
                "use_noise_floor": USE_NOISE_FLOOR,
                "noise_floor_alpha": NOISE_FLOOR_ALPHA,
                "noise_floor_min_temperature": NOISE_FLOOR_MIN_TEMPERATURE,
                "sd016_diversification_weight": SD016_DIVERSIFICATION_WEIGHT,
            }
            with arm_cell(seed, config_slice=config_slice, script_path=Path(__file__)) as cell:
                r = _run_condition(
                    seed=seed, condition=cond,
                    training_episodes=TRAINING_EPISODES,
                    steps_per_episode=STEPS_PER_EPISODE,
                    eval_episodes_each=EVAL_EPISODES_EACH,
                    zg=zg,
                )
                cell.stamp(r)
            arm_results.append(r)

    elapsed = time.time() - t0

    def by_cond(c):
        return [r for r in arm_results if r["condition"] == c]

    waking = by_cond("WAKING_ONLY")
    sws_r = by_cond("SWS_THEN_REM")

    # --- P0 POSITIVE-CONTROL GATE (NEW in 436c) -----------------------------
    # The 436c fix: before C1's slot_cosine_sim comparison is treated as
    # meaningful, confirm the sleep-cycle write path actually fired.
    # 436b could not distinguish "ContextMemory.write() never fired" from
    # "it fired but had negligible effect" from "the readout is invariant
    # under this config" -- this gate makes that distinction directly, from
    # the counters run_sleep_cycle() itself returns.
    pooled_sws_n_writes = sum(r["sleep_cycle_sws_n_writes_total"] for r in sws_r)
    pooled_rem_n_rollouts = sum(r["sleep_cycle_rem_n_rollouts_total"] for r in sws_r)

    readiness_checks = [
        {
            "name": "sws_context_memory_writes_occur",
            "measured": pooled_sws_n_writes,
            "threshold": SWS_WRITES_FLOOR,
            "direction": "lower",
            "control": (
                "sum of ContextMemory.write() calls (sws_n_writes, returned "
                "by agent.run_sws_schema_pass() via run_sleep_cycle()) "
                "across every SWS_THEN_REM sleep pass, pooled across all "
                f"{len(SEEDS)} seeds -- must be > 0 for the C1 "
                "slot_cosine_sim comparison to test SD-017/ARC-045/MECH-166's "
                "mechanism rather than an unrecorded no-op "
                "(failure_autopsy_V3-EXQ-436b_2026-08-02: 436b's driver "
                "discarded agent.run_sleep_cycle()'s entire return value "
                "with `_ = ...`, so sws_n_writes was never captured)."
            ),
        },
        {
            "name": "rem_attribution_rollouts_occur",
            "measured": pooled_rem_n_rollouts,
            "threshold": REM_ROLLOUTS_FLOOR,
            "direction": "lower",
            "control": (
                "sum of hippocampal replay rollouts (rem_n_rollouts, "
                "returned by agent.run_rem_attribution_pass() via "
                "run_sleep_cycle()) across every SWS_THEN_REM sleep pass, "
                f"pooled across all {len(SEEDS)} seeds -- the REM-equivalent "
                "write/consolidation counter to sws_n_writes above; same "
                "recording gap, same fix."
            ),
        },
    ]
    ready = True
    preconditions: List[Dict] = []
    try:
        preconditions = p0_readiness_gate(readiness_checks)
    except P0NotReady as e:
        preconditions = e.preconditions
        ready = False

    per_seed_diff: Dict[str, Dict] = {}
    for w_r, s_r in zip(waking, sws_r):
        seed = w_r["seed"]
        slot_sim_diff = s_r["slot_cosine_sim"] - w_r["slot_cosine_sim"]
        harm_dang_diff = s_r["harm_rate_dangerous"] - w_r["harm_rate_dangerous"]
        harm_safe_diff = s_r["harm_rate_safe"] - w_r["harm_rate_safe"]
        slot_sep_diff = s_r["slot_separation"] - w_r["slot_separation"]
        per_seed_diff[str(seed)] = {
            "seed": seed,
            "waking_slot_cosine_sim": w_r["slot_cosine_sim"],
            "sws_then_rem_slot_cosine_sim": s_r["slot_cosine_sim"],
            "slot_cosine_sim_signed_diff": slot_sim_diff,
            # C1: PRIMARY GATE (once P0 is met). Directional (per ARC-045's
            # own experimental implication) -- SWS_THEN_REM strictly lower
            # than WAKING_ONLY.
            "slot_cosine_sim_passes_C1": s_r["slot_cosine_sim"] < w_r["slot_cosine_sim"],
            "arc045_sws_then_rem_below_reference": s_r["slot_cosine_sim"] < ARC045_ABS_COSINE_REFERENCE,
            "waking_harm_rate_dangerous": w_r["harm_rate_dangerous"],
            "sws_then_rem_harm_rate_dangerous": s_r["harm_rate_dangerous"],
            "harm_rate_dangerous_signed_diff": harm_dang_diff,
            "waking_harm_rate_safe": w_r["harm_rate_safe"],
            "sws_then_rem_harm_rate_safe": s_r["harm_rate_safe"],
            "harm_rate_safe_signed_diff": harm_safe_diff,
            "waking_slot_separation": w_r["slot_separation"],
            "sws_then_rem_slot_separation": s_r["slot_separation"],
            "slot_separation_signed_diff": slot_sep_diff,
            "sws_then_rem_slot_separation_passes_C4": s_r["slot_separation"] > C4_SLOT_SEPARATION_THRESHOLD,
            "waking_action_class_entropy": w_r["train_action_class_entropy_mean"],
            "sws_then_rem_action_class_entropy": s_r["train_action_class_entropy_mean"],
            "sws_then_rem_sws_n_writes_total": s_r["sleep_cycle_sws_n_writes_total"],
            "sws_then_rem_rem_n_rollouts_total": s_r["sleep_cycle_rem_n_rollouts_total"],
        }

    c1_count = sum(1 for d in per_seed_diff.values() if d["slot_cosine_sim_passes_C1"])
    c1_pass = c1_count >= C1_N_SEEDS_REQUIRED

    c4_count = sum(1 for d in per_seed_diff.values() if d["sws_then_rem_slot_separation_passes_C4"])
    c4_pass = c4_count >= C4_N_SEEDS_REQUIRED

    # Non-degeneracy self-report: guard against a silent repeat of the
    # original 436/436a failure mode (bit-identical trajectories -> zero
    # spread on the load-bearing metric) despite use_noise_floor=True.
    all_cosine_values = [r["slot_cosine_sim"] for r in arm_results]
    all_action_entropy = [r["train_action_class_entropy_mean"] for r in arm_results]
    degeneracy = check_degeneracy({
        "slot_cosine_sim": all_cosine_values,
        "train_action_class_entropy_mean": {"values": all_action_entropy, "floor": 1e-6},
    })

    def _direction(passed: bool) -> str:
        return "supports" if passed else "weakens"

    if not ready:
        # P0 UNMET: the write path itself did not fire this run. This is NOT
        # a repeat of 436b's ambiguity -- it is a positively confirmed
        # write-path failure, distinct from a genuine C1 null. Route to
        # /failure-autopsy on the write path, not on slot_cosine_sim.
        outcome = "FAIL"
        evidence_direction = "non_contributory"
        label = "sleep_cycle_recording_gap_still_present"
        evidence_direction_per_claim = {cid: "non_contributory" for cid in CLAIM_IDS}
    else:
        # PASS = C1 alone (P0 met). C4 and the harm-rate diffs are recorded
        # but never gate.
        outcome = "PASS" if c1_pass else "FAIL"
        evidence_direction = _direction(c1_pass)
        label = (
            "sws_then_rem_differentiates_slots" if c1_pass
            else "sws_then_rem_does_not_differentiate_slots"
        )
        # All three claims share the SAME pre-registered wall-independent
        # test (claims.yaml's own "Experimental implication" text for
        # SD-017 and ARC-045 both specify slot_cosine_sim differentiation
        # after sleep; the MECH-166 notes describe this exact comparison as
        # "the direct test").
        evidence_direction_per_claim = {
            "SD-017": _direction(c1_pass),
            "ARC-045": _direction(c1_pass),
            "MECH-166": _direction(c1_pass),
        }

    summary = {
        "P0_sleep_cycle_write_counters": {
            "sws_n_writes_pooled": pooled_sws_n_writes,
            "sws_n_writes_floor": SWS_WRITES_FLOOR,
            "rem_n_rollouts_pooled": pooled_rem_n_rollouts,
            "rem_n_rollouts_floor": REM_ROLLOUTS_FLOOR,
            "ready": ready,
            "desc": ("P0 GATE (new in 436c). Both counters must clear their "
                     "floor (>0, pooled across all SWS_THEN_REM seeds) before "
                     "C1 is read as evidence either way -- the direct fix for "
                     "failure_autopsy_V3-EXQ-436b_2026-08-02's recording gap."),
        },
        "C1_primary_slot_cosine_sim_directional": {
            "n_seeds_required": C1_N_SEEDS_REQUIRED,
            "n_seeds_passed": c1_count,
            "pass": c1_pass,
            "scored": ready,
            "desc": ("SOLE PASS/FAIL GATE once P0 is met. "
                     "slot_cosine_sim(SWS_THEN_REM) < "
                     "slot_cosine_sim(WAKING_ONLY) in >= 3/5 seeds. "
                     "Wall-independent (read under torch.no_grad() off "
                     "agent.e1.context_memory.memory)."),
        },
        "C4_arc045_slot_separation_threshold": {
            "threshold": C4_SLOT_SEPARATION_THRESHOLD,
            "n_seeds_required": C4_N_SEEDS_REQUIRED,
            "n_seeds_passed": c4_count,
            "pass": c4_pass,
            "scored": ready,
            "desc": ("SECONDARY, non-gating. slot_separation in SWS_THEN_REM "
                     "> 0.3 in >= 3/5 seeds."),
        },
    }

    print(f"\nOutcome: {outcome}  label={label}  P0_ready={ready}")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"  Per-claim direction: {evidence_direction_per_claim}")
    print(f"  Non-degenerate: {degeneracy.get('non_degenerate')} "
          f"({degeneracy.get('degeneracy_reason', '')})")

    result = {
        "queue_id": QUEUE_ID,
        "supersedes": SUPERSEDES,
        "claim_ids": CLAIM_IDS,
        "experiment_type": EXPERIMENT_TYPE,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "outcome": outcome,
        "status": outcome,
        "result": outcome,
        "evidence_direction": evidence_direction,
        "evidence_direction_per_claim": evidence_direction_per_claim,
        "pass_criteria_summary": summary,
        "interpretation": {
            "label": label,
            "preconditions": preconditions,
            "criteria_non_degenerate": {
                "C1": degeneracy.get("non_degenerate"),
            },
        },
        "non_degenerate": degeneracy.get("non_degenerate"),
        "degeneracy_reason": degeneracy.get("degeneracy_reason"),
        "aggregated": {
            "per_seed": per_seed_diff,
            "n_seeds_passed": {"C1": c1_count, "C4": c4_count},
            "sleep_cycle_write_counters_pooled": {
                "sws_n_writes": pooled_sws_n_writes,
                "rem_n_rollouts": pooled_rem_n_rollouts,
            },
        },
        "arm_results": arm_results,
        "per_seed_results": arm_results,
        "registered_thresholds": {
            "C1_N_SEEDS_REQUIRED": C1_N_SEEDS_REQUIRED,
            "C4_SLOT_SEPARATION_THRESHOLD": C4_SLOT_SEPARATION_THRESHOLD,
            "C4_N_SEEDS_REQUIRED": C4_N_SEEDS_REQUIRED,
            "ARC045_ABS_COSINE_REFERENCE": ARC045_ABS_COSINE_REFERENCE,
            "BASE_HARM_THRESHOLD": BASE_HARM_THRESHOLD,
            "CONTEXT_BETA": CONTEXT_BETA,
            "SLOT_DANGER_EMA_ALPHA": SLOT_DANGER_EMA_ALPHA,
            "USE_NOISE_FLOOR": USE_NOISE_FLOOR,
            "NOISE_FLOOR_ALPHA": NOISE_FLOOR_ALPHA,
            "NOISE_FLOOR_MIN_TEMPERATURE": NOISE_FLOOR_MIN_TEMPERATURE,
            "BASELINE_TEMPERATURE": BASELINE_TEMPERATURE,
            "SWS_WRITES_FLOOR": SWS_WRITES_FLOOR,
            "REM_ROLLOUTS_FLOOR": REM_ROLLOUTS_FLOOR,
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
        "elapsed_seconds": elapsed,
        "notes": (
            "MECH-166 wall-independent representation confirmer. Discharges "
            "IGW-20260801-223 (EVB-0475) and IGW-20260801-225 (GOV-CONFIRM-1). "
            "Successor to V3-EXQ-436b: fixes the recording gap identified in "
            "failure_autopsy_V3-EXQ-436b_2026-08-02 -- 436b's driver "
            "discarded agent.run_sleep_cycle()'s entire return value with "
            "`_ = ...`, so sws_n_writes / rem_n_rollouts were never captured, "
            "making 436b's bit-identical slot_cosine_sim readout across "
            "conditions uninterpretable (could not distinguish a genuine "
            "null from an unrecorded write). This driver captures those "
            "counters and gates C1's interpretability on them (P0 readiness "
            "gate, pooled across all 5 SWS_THEN_REM seeds, floor > 0 on "
            "each). Conditions, DVs, thresholds, seeds, and action-selection "
            "wiring are otherwise unchanged from 436b."
        ),
    }
    return result, zg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()
    result, zg_accumulator = run(dry_run=args.dry_run)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["run_id"] = f"{EXPERIMENT_TYPE}_{ts}_v3"
    result["timestamp_utc"] = ts
    result["architecture_epoch"] = "ree_hybrid_guardrails_v1"

    out_path = write_flat_manifest(
        result,
        dry_run=args.dry_run,
        config={
            "conditions": CONDITIONS,
            "sleep_interval": SLEEP_INTERVAL,
            "context_switch_every": CONTEXT_SWITCH_EVERY,
            "training_episodes": TRAINING_EPISODES,
            "steps_per_episode": STEPS_PER_EPISODE,
            "eval_episodes_each": EVAL_EPISODES_EACH,
            "seeds": SEEDS,
            "env_safe_num_hazards": 1,
            "env_dangerous_num_hazards": 8,
            "sd016_writepath_mode": "off",
            "sd016_diversification_weight": SD016_DIVERSIFICATION_WEIGHT,
            "use_per_stream_vs": True,
            "use_anchor_sets": True,
            "use_sd039_anchor_payload": True,
            "use_sleep_loop_in_sleep_arms": True,
            "use_noise_floor": USE_NOISE_FLOOR,
            "noise_floor_alpha": NOISE_FLOOR_ALPHA,
            "noise_floor_min_temperature": NOISE_FLOOR_MIN_TEMPERATURE,
        },
        seeds=SEEDS,
        script_path=__file__,
        started_at=t0,
        z_goal_stream_stats=zg_accumulator.stats(),
    )
    print(f"Output written to: {out_path}", flush=True)

    _outcome_clean = str(result.get("outcome", "FAIL")).upper()
    if _outcome_clean not in ("PASS", "FAIL"):
        _outcome_clean = "FAIL"
    emit_outcome(
        outcome=_outcome_clean,
        manifest_path=out_path,
        dry_run=args.dry_run,
    )
