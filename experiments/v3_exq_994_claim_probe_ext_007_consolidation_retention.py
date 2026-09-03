#!/opt/local/bin/python3
"""
V3-EXQ-994 -- EXT-007 claim probe: does hippocampal offline consolidation
(ARC-007 / MECH-092 / SD-017) cause measurably better cross-episode
retention of world-model structure than no consolidation?

SLEEP DRIVER: K=1 single-fire (SleepLoopManager, fires every episode)

experiment_purpose: evidence
Claims: EXT-007
red-team (fable): CONTESTED -> fixed. Three findings, all addressed before
queuing: (1) dead probe-position override -- _probe_retrieval reused
reset_to()'s obs_dict, computed BEFORE the CONTEXT_A-cue position override,
so every probe read the AGENT_SPAWN cell (shared with CONTEXT_B) instead of
the intended context-specific cue; fixed by re-deriving obs_dict via
env._get_observation_dict() AFTER the override (established idiom, see
v3_exq_002/004/006 etc.). (2) manipulation-check precondition partially
self-referential (mean_without gated the same retention_score values C1
compares against); addressed by adding an independent, retention_score-free
diagnostic (slot_overlap_frac, from ContextMemory.slot_write_counts deltas)
alongside the original check, disclosed as a diagnostic cross-check rather
than a silent replacement. (3) sleep self-perturbation confound -- consolidation's
own INTERFERE/TEST-boundary replay writes B-content into ContextMemory and
retrains E1's encoder between probe moments, so a WITH-loses-to-WITHOUT
result cannot be attributed to "no protective effect" (it is equally
consistent with a real protective effect being outweighed by consolidation's
own noise); a PASS (WITH beats WITHOUT) has no such confound. Fixed by
routing C1-fails-but-non-degenerate to evidence_direction=non_contributory
(never weakens) -- see "FALSIFIABLE CLAIM" below and main()'s routing block.
Also added memory_matrix_retention_score (raw ContextMemory slot-matrix
cosine, immune to encoder query drift) as a non-load-bearing companion DV
for future attribution work. Model: fable.

BACKGROUND / WHAT THIS TESTS
-----------------------------
EXT-007 (external failure-mode claim, claims.yaml) asserts that LLMs have no
offline consolidation phase: the context window is discarded after each
session and nothing integrates experience into a persistent world model
across interactions. The claim's `ree_mechanism` field names REE's own
answer to this: ARC-007 (hippocampal replay over residue-field terrain) and
MECH-092 (quiescent SWR-equivalent replay), operationalised in V3 as SD-017
(SWS-analog schema installation + REM-analog slot-filling,
`REEAgent.run_sleep_cycle()`) plus the SleepLoopManager episode-boundary
driver (`use_sleep_loop`).

This script does NOT test the LLM side of EXT-007 (no LLM is involved). It
tests the REE-mechanism side, as an ablation: does REE's own offline
consolidation machinery actually cause cross-episode persistence of
world-model information, or does the substrate merely carry information
across episode boundaries for free (regardless of sleep), in which case
"consolidation" would not be doing the causal work EXT-007's REE-side
counter-story requires?

SUBSTRATE FACTS ESTABLISHED BEFORE WRITING THIS SCRIPT (empirical probe,
`ree_core/agent.py` read, Step 2.5a):
  - `REEAgent.reset()` (agent.py ~3387-3505) does NOT reset `residue_field`
    or `e1.context_memory` -- both persist across every episode boundary
    REGARDLESS of sleep. Confirmed by a throwaway probe (built + run +
    deleted, per skill Step 2.5a): `context_memory.memory` and
    `residue_field.rbf_field.weights` are bit-identical before/after
    `agent.reset()` with sleep OFF.
  - Therefore raw persistence-across-reset is NOT the thing to measure (it
    happens unconditionally and would confound WITH vs WITHOUT trivially).
    The causal question is whether consolidation makes a stored
    representation MORE ROBUST TO INTERFERENCE from subsequent, unrelated
    experience -- not whether it merely survives an idle reset.
  - `E1DeepPredictor.context_memory.write()` (agent.py:5300-5333) is gated by
    `E1Config.sd016_writepath_mode` (default "off" -- CONFIRMED via source
    read, ree_core/utils/config.py:565/6833). Both arms of this experiment
    explicitly set `sd016_writepath_mode="sense_only"` so ordinary online
    writes happen IDENTICALLY in both arms; the only manipulated variable is
    whether an additional SWS/REM/SHY offline pass also runs at episode
    boundaries. Without this, the WITHOUT_CONSOLIDATION arm would never
    write to context_memory at all and the comparison would be vacuous by
    construction (manipulation-reaches-DV check, red-team Family 1).
  - `run_sws_schema_pass()` (agent.py:12153-12260) writes prototype
    `[z_self, z_world]` vectors SAMPLED FROM `_world_experience_buffer`
    (recent waking experience) directly into `ContextMemory`, explicitly
    documented as "slot-formation" that installs "differentiated context
    attractors" protected from waking overwrite while the offline gate is
    up (`enter_offline_mode()` docstring, agent.py:12071-12094). This is the
    literal mechanism EXT-007's REE-side story claims exists; the script
    below is a direct causal test of whether it does what it says.

DESIGN
------
Two config-only arms (WITH_CONSOLIDATION / WITHOUT_CONSOLIDATION), matched
seeds, one continuous agent life per (arm, seed) cell:

  ENCODE     (N_ENCODE episodes):    env forced to CONTEXT_A (fixed hazard/
                                      resource layout via env.reset_to(),
                                      bypassing RNG placement) via
                                      env.reset_to(). Agent steps with a
                                      random policy (matches the established
                                      V3-EXQ-385/385a/127/691 pattern -- the
                                      DV is about the world-MODEL, not about
                                      a trained policy) and trains E1 online
                                      on its own self-supervised prediction
                                      loss (no downstream head reads z_world,
                                      so the P0->P1->P2 phased-training
                                      hazard does not apply here -- see
                                      "Phased training" below).
  INTERFERE  (N_INTERFERE episodes): env forced to CONTEXT_B (a spatially
                                      DISJOINT fixed layout, same hazard/
                                      resource counts). This is what
                                      competes for ContextMemory's slots via
                                      the SAME online sd016 write path both
                                      arms share -- the retention challenge.
  TEST       (1 probe):              env forced back to CONTEXT_A via
                                      env.reset_to() + agent.reset() (a real
                                      episode boundary -- in the WITH arm
                                      this is also where SleepLoopManager's
                                      K=1 auto-fire runs its last
                                      consolidation cycle before the probe).
                                      A single read-only probe (offline_mode
                                      forced True for the probe's own
                                      sense() call, so the measurement
                                      itself cannot write and contaminate
                                      what it is measuring) re-encodes the
                                      CONTEXT_A resource-adjacent cue and
                                      reads it back from ContextMemory.

PRIMARY DV -- retention_score (per arm, per seed):
  cosine_similarity(
      context_memory.read(probe_query_A) captured at the END of ENCODE
        (before INTERFERE ever runs),
      context_memory.read(probe_query_A) captured at TEST (after
        INTERFERE has had a chance to compete for slots),
  )
  High = the CONTEXT_A retrieval target survived INTERFERE; low = it was
  overwritten/diluted. This mirrors the cosine-based "slot diversity"
  measurement convention already validated in V3-EXQ-385/385a, applied to a
  retrieval-STABILITY question instead of a diversity question.

SECONDARY DV (context only, not load-bearing) -- residue_retention:
  residue_field.evaluate(z_world) at the same two probe moments, for the
  CONTEXT_A resource-adjacent z_world. Convergent-evidence channel: residue
  terrain is architecturally a different persistent store than
  ContextMemory (SD-017's REM pass and MECH-217/290 write here too), so
  agreement or disagreement between the two channels is itself informative
  and is recorded, never used to gate PASS/FAIL.

FALSIFIABLE CLAIM: mean retention_score(WITH_CONSOLIDATION) >
mean retention_score(WITHOUT_CONSOLIDATION) in >= 2/3 seeds.
ASYMMETRIC ROUTING (red-team fable finding, fixed): PASS/supports on C1 met;
but C1 NOT met (given readiness_ok + non_degenerate) routes to
FAIL/non_contributory, never FAIL/weakens -- consolidation's own
cross-context replay + encoder retraining between probe moments can depress
WITH's score for reasons unrelated to whether protection occurred, so only
a WITH-beats-WITHOUT result is cleanly attributable to the claim.

NON-DEGENERACY (mandatory per skill Step 3 -- a null must not default to
"supports", and per this task's own instruction, a null/non-significant
result must route to weaken/non_contributory, not supports):
  - READINESS precondition: context_memory must have been written a
    non-trivial number of times during ENCODE (positive control on the
    shared sd016 write path -- if this is unmet, sd016_writepath_mode
    silently failed to engage and NEITHER arm's memory holds anything,
    which would read as a false "no difference" rather than "nothing was
    ever written"). Self-routes FAIL/non_contributory if unmet, never a
    substrate-verdict label (this is an `evidence`-purpose script, not
    diagnostic, so `substrate_not_ready_requeue` phrasing doesn't apply --
    see the P0 gate below).
  - MANIPULATION-CHECK precondition: INTERFERE must actually have disturbed
    the WITHOUT_CONSOLIDATION arm's retrieval of the CONTEXT_A probe
    (mean retention_score(WITHOUT) < DISTURBANCE_CEILING across seeds). If
    INTERFERE never competes for the same slots (e.g. because 16 slots is
    ample headroom for two contexts), the design has no discriminative
    power regardless of what sleep does, and a "no difference" reading
    would be a starved criterion, not a finding. Unmet -> non_contributory.
  - check_degeneracy() (experiments/_metrics.py) is run over the six
    retention_score cells (2 arms x 3 seeds) as an additional structural
    floor check.

PHASED TRAINING: not applicable in the P0->P1->P2 sense the skill's
MANDATORY rule targets (a downstream head trained on z_world / z_harm
chasing a moving encoder target). No downstream head reads z_world in this
script. E1 itself trains continuously on its own self-supervised prediction
loss through ENCODE and INTERFERE (mirrors the established, validated
V3-EXQ-385/385a/127/691 pattern) so that "world-model" here means a LEARNED
representation of the fixed layouts, not an untrained random projection.
Training is disabled (train=False) during TEST so the retention measurement
is not conflated with new online learning inside the probe episode itself.

ETHICS PREFLIGHT (Step 2.6, condensed): all-false / decision: allow. No
negative valence, no suffering-like state, no self-model, no
inescapability, no offline replay OVER HARM content (replay here is over
ordinary hazard/resource navigation experience, hypothesis_tag=True where
applicable), no social/language content, no human data. SENT-0: V3 is not
claimed sentient; this is pre-ethical instrumentation.

claim_ids: ["EXT-007"]
architecture_epoch: "ree_hybrid_guardrails_v1"
run_id ends: _v3
"""

import os
import sys
import json
import argparse
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig
from experiments.pack_writer import write_flat_manifest, resolve_evidence_experiments_dir
from experiments._lib.arm_fingerprint import arm_cell
from experiments._metrics import check_degeneracy
from experiment_protocol import emit_outcome


# ---------------------------------------------------------------------------
# Experiment constants
# ---------------------------------------------------------------------------

EXPERIMENT_TYPE    = "v3_exq_994_claim_probe_ext_007_consolidation_retention"
CLAIM_IDS          = ["EXT-007"]
EXPERIMENT_PURPOSE = "evidence"
EVIDENCE_CLASS      = "ablation_pair"

SEEDS = [42, 49, 56]

N_ENCODE           = 12   # episodes on CONTEXT_A
N_INTERFERE         = 12   # episodes on CONTEXT_B
STEPS_PER_EPISODE  = 60
GRID_SIZE           = 8
LR                  = 1e-4

# Fixed deterministic layouts (env.reset_to()), spatially disjoint so
# INTERFERE genuinely competes for different grid regions than ENCODE.
AGENT_SPAWN         = (1, 1)
CONTEXT_A_HAZARDS   = [(3, 3)]
CONTEXT_A_RESOURCES = [(6, 6)]
CONTEXT_B_HAZARDS   = [(6, 3)]
CONTEXT_B_RESOURCES = [(3, 6)]

# The probe cue: agent standing at the CONTEXT_A resource cell. Used to
# build a single read-only obs for the retention measurement -- never
# stepped through, so it cannot itself accumulate interference.
PROBE_POS_A = CONTEXT_A_RESOURCES[0]

# Non-degeneracy thresholds (pre-registered, per skill Step 3.5 "Is every
# precondition SATISFIABLE at the values the script itself pre-registers?"
# -- both checked arithmetically before any compute: WRITE_FLOOR_FRAC is a
# fraction of ENCODE ticks, trivially satisfiable if sd016 write-path fires
# at all; DISTURBANCE_CEILING is a cosine value comfortably below the
# perfect-retention value of 1.0).
WRITE_FLOOR_FRAC     = 0.5     # >=50% of ENCODE ticks must write a slot
DISTURBANCE_CEILING  = 0.98    # WITHOUT arm must show measurable drift


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed,
        size=GRID_SIZE,
        num_hazards=1,
        num_resources=1,
        hazard_harm=0.02,
        resource_respawn_on_consume=True,
    )


def _reset_to_context(env: CausalGridWorldV2, hazards, resources):
    return env.reset_to(
        agent_pos=AGENT_SPAWN,
        hazard_positions=hazards,
        resource_positions=resources,
    )


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def _make_agent(env: CausalGridWorldV2, with_consolidation: bool) -> REEAgent:
    """Build REEAgent. Only the offline-consolidation switches differ between
    arms; the online context_memory write path (sd016_writepath_mode) is
    identical in both -- see module docstring "SUBSTRATE FACTS"."""
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        alpha_world=0.9,
        alpha_self=0.3,
        # Shared online write path -- identical in both arms (mandatory; see
        # docstring). Without this the WITHOUT arm never writes at all.
        sd016_enabled=True,
        sd016_writepath_mode="sense_only",
        # SD-017 offline consolidation -- the manipulated variable.
        sws_enabled=with_consolidation,
        sws_consolidation_steps=8,
        sws_schema_weight=0.1,
        rem_enabled=with_consolidation,
        rem_attribution_steps=6,
        shy_enabled=with_consolidation,
        shy_decay_rate=0.98,   # V3-EXQ-385a fix: 0.85 collapses slots at 15 cycles
        # SleepLoopManager: fires run_sleep_cycle() automatically from
        # agent.reset() every sleep_loop_episodes_K episodes when enabled.
        use_sleep_loop=with_consolidation,
        sleep_loop_episodes_K=1,
        sleep_loop_require_passes=True,
    )
    return REEAgent(cfg)


# ---------------------------------------------------------------------------
# Probe helper -- read-only, must not contaminate what it measures
# ---------------------------------------------------------------------------

def _probe_retrieval(
    agent: REEAgent, env: CausalGridWorldV2
) -> Tuple[torch.Tensor, float, torch.Tensor]:
    """Deterministically place the agent at the CONTEXT_A resource cell,
    encode the resulting observation with offline writes suppressed (so the
    probe's own sense() call cannot write to ContextMemory and contaminate
    the very quantity being measured), then read back from ContextMemory and
    evaluate residue at that z_world.

    Also returns a raw snapshot of the full ContextMemory slot matrix
    (`context_memory.memory`), independent of the query-side retrieval read.
    retention_score (computed from the read) conflates two channels -- slot
    CONTENT drift (what we intend to measure) and encoder QUERY drift (E1's
    `query_proj`/z_world encoder moving between probe moments, e.g. via
    `sws_consolidation_steps` gradient steps in the WITH arm) -- because a
    changed query against unchanged memory can still shift the read (red-team
    fable finding, Family 1). The raw matrix cosine (computed by the caller)
    is unaffected by query drift and lets a later reader separate the two
    channels; it is NOT load-bearing for this script's own criterion.

    Returns (context_read_vector [1, latent_dim], residue_value, memory_matrix
    [num_slots, memory_dim]).
    """
    device = agent.device
    was_offline = agent.e1._offline_mode
    agent.e1._offline_mode = True  # suppress context_memory.write() side effect
    try:
        _reset_to_context(env, CONTEXT_A_HAZARDS, CONTEXT_A_RESOURCES)
        # env.reset_to places the agent at AGENT_SPAWN, not PROBE_POS_A;
        # override the observed position deterministically so the probe cue
        # is always "standing at the CONTEXT_A resource cell" regardless of
        # spawn. env exposes agent_x/agent_y directly (see reset_to()).
        # MUST re-derive obs_dict AFTER the override -- reset_to()'s own
        # returned obs_dict was already built (via _get_observation_dict())
        # at the pre-override AGENT_SPAWN position, so reusing it here would
        # silently probe the spawn cell (identical for CONTEXT_A/CONTEXT_B)
        # instead of the intended CONTEXT_A resource-adjacent cue
        # (red-team fable finding, confirmed against _get_observation_dict()
        # call sites in experiments/v3_exq_002/004/006/... -- the established
        # re-observe-after-manual-reposition idiom).
        env.agent_x, env.agent_y = PROBE_POS_A
        obs_dict = env._get_observation_dict()
        obs_body = obs_dict["body_state"]
        obs_world = obs_dict["world_state"]
        if not torch.is_tensor(obs_body):
            obs_body = torch.tensor(obs_body, dtype=torch.float32, device=device)
        else:
            obs_body = obs_body.to(device)
        if obs_body.dim() == 1:
            obs_body = obs_body.unsqueeze(0)
        if not torch.is_tensor(obs_world):
            obs_world = torch.tensor(obs_world, dtype=torch.float32, device=device)
        else:
            obs_world = obs_world.to(device)
        if obs_world.dim() == 1:
            obs_world = obs_world.unsqueeze(0)

        with torch.no_grad():
            latent = agent.sense(obs_body, obs_world)
            query = torch.cat(
                [latent.z_self.detach(), latent.z_world.detach()], dim=-1
            )
            context_read = agent.e1.context_memory.read(query).detach().clone()
            residue_val = float(
                agent.residue_field.evaluate(latent.z_world.detach()).item()
            )
            memory_matrix = agent.e1.context_memory.memory.detach().clone()
    finally:
        agent.e1._offline_mode = was_offline
    return context_read, residue_val, memory_matrix


# ---------------------------------------------------------------------------
# Single-episode step (random policy; matches V3-EXQ-385/385a/127/691)
# ---------------------------------------------------------------------------

def _run_episode(
    agent: REEAgent,
    env: CausalGridWorldV2,
    hazards,
    resources,
    optimizer: Optional[optim.Optimizer],
    steps_per_episode: int,
    train: bool,
) -> Tuple[float, float, int]:
    """Run one episode with a fixed layout (env.reset_to). Returns
    (episode_harm_rate, mean_pred_loss, n_context_memory_writes_this_episode).
    """
    device = agent.device
    _, obs_dict = _reset_to_context(env, hazards, resources)
    agent.reset()
    agent.e1.reset_hidden_state()

    write_counts_before = agent.e1.context_memory.slot_write_counts.sum().item()

    ep_harm = 0.0
    ep_steps = 0
    pred_losses: List[float] = []

    for _step in range(steps_per_episode):
        obs_body = obs_dict["body_state"]
        obs_world = obs_dict["world_state"]
        if not torch.is_tensor(obs_body):
            obs_body = torch.tensor(obs_body, dtype=torch.float32, device=device)
        else:
            obs_body = obs_body.to(device)
        if obs_body.dim() == 1:
            obs_body = obs_body.unsqueeze(0)
        if not torch.is_tensor(obs_world):
            obs_world = torch.tensor(obs_world, dtype=torch.float32, device=device)
        else:
            obs_world = obs_world.to(device)
        if obs_world.dim() == 1:
            obs_world = obs_world.unsqueeze(0)

        latent = agent.sense(obs_body, obs_world)
        agent.clock.advance()

        action_idx = random.randint(0, env.action_dim - 1)
        action = torch.zeros(1, env.action_dim, device=device)
        action[0, action_idx] = 1.0

        if agent._current_latent is not None:
            z_self_prev = agent._current_latent.z_self.detach().clone()
            agent.record_transition(z_self_prev, action, latent.z_self.detach())

        _, harm_signal, done, info, obs_dict = env.step(action)
        ep_harm += max(0.0, float(-harm_signal))
        ep_steps += 1

        if train:
            pred_loss = agent.compute_prediction_loss()
            if pred_loss.requires_grad:
                optimizer.zero_grad()
                pred_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                optimizer.step()
                pred_losses.append(float(pred_loss.item()))

        agent.update_residue(float(harm_signal) if float(harm_signal) < 0 else 0.0)

        if done:
            break

    write_counts_after = agent.e1.context_memory.slot_write_counts.sum().item()
    n_writes = int(write_counts_after - write_counts_before)

    harm_rate = ep_harm / max(1, ep_steps)
    mean_pred_loss = (
        float(sum(pred_losses) / len(pred_losses)) if pred_losses else 0.0
    )
    return harm_rate, mean_pred_loss, n_writes


# ---------------------------------------------------------------------------
# One (arm, seed) cell
# ---------------------------------------------------------------------------

def run_cell(
    condition_name: str,
    with_consolidation: bool,
    seed: int,
    dry_run: bool = False,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    random.seed(seed)

    n_encode = N_ENCODE if not dry_run else 3
    n_interfere = N_INTERFERE if not dry_run else 3
    steps_per_ep = STEPS_PER_EPISODE if not dry_run else 15

    env_a = _make_env(seed)
    env_b = _make_env(seed + 1000)

    agent = _make_agent(env_a, with_consolidation)
    optimizer = optim.Adam(
        list(agent.e1.parameters()) + list(agent.latent_stack.parameters()),
        lr=LR,
    )

    total_eps = n_encode + n_interfere + 1
    ep_num = 0
    encode_writes = 0
    encode_ticks = 0

    print(f"Seed {seed} Condition {condition_name}", flush=True)

    # Per-slot write-count snapshots (independent of retention_score/read
    # path) so INTERFERE-vs-ENCODE slot competition can be measured directly
    # from ContextMemory's own occupancy bookkeeping, not inferred from the
    # same statistic the load-bearing criterion consumes (red-team fable
    # Family 4 finding: gating "did INTERFERE disturb retention" on
    # retention_score itself, then testing WITH>WITHOUT on retention_score,
    # is circular). This is a diagnostic cross-check, not a replacement for
    # the pre-registered readiness preconditions below.
    slots_pre_encode = agent.e1.context_memory.slot_write_counts.detach().clone()

    # ---- ENCODE: CONTEXT_A ----
    for ep in range(n_encode):
        ep_num += 1
        harm_rate, pred_loss, n_writes = _run_episode(
            agent, env_a, CONTEXT_A_HAZARDS, CONTEXT_A_RESOURCES,
            optimizer, steps_per_ep, train=True,
        )
        encode_writes += n_writes
        encode_ticks += steps_per_ep
        if (ep + 1) % max(1, n_encode // 2) == 0 or ep == 0:
            print(
                f"  [train] cond={condition_name} seed={seed}"
                f" ep {ep_num}/{total_eps} phase=ENCODE"
                f" harm_rate={harm_rate:.4f} pred_loss={pred_loss:.4f}"
                f" writes={n_writes}",
                flush=True,
            )

    # Snapshot at end-of-encode, BEFORE interference ever runs.
    end_encode_read, end_encode_residue, end_encode_memory = _probe_retrieval(agent, env_a)
    slots_post_encode = agent.e1.context_memory.slot_write_counts.detach().clone()
    encode_written_slots = set(
        (slots_post_encode - slots_pre_encode).nonzero().flatten().tolist()
    )

    # ---- INTERFERE: CONTEXT_B ----
    for ep in range(n_interfere):
        ep_num += 1
        harm_rate, pred_loss, n_writes = _run_episode(
            agent, env_b, CONTEXT_B_HAZARDS, CONTEXT_B_RESOURCES,
            optimizer, steps_per_ep, train=True,
        )
        if (ep + 1) % max(1, n_interfere // 2) == 0 or ep == 0:
            print(
                f"  [train] cond={condition_name} seed={seed}"
                f" ep {ep_num}/{total_eps} phase=INTERFERE"
                f" harm_rate={harm_rate:.4f} pred_loss={pred_loss:.4f}"
                f" writes={n_writes}",
                flush=True,
            )

    slots_post_interfere = agent.e1.context_memory.slot_write_counts.detach().clone()
    interfere_written_slots = set(
        (slots_post_interfere - slots_post_encode).nonzero().flatten().tolist()
    )
    slot_overlap_frac = (
        len(encode_written_slots & interfere_written_slots) / len(encode_written_slots)
        if encode_written_slots else 0.0
    )

    # ---- TEST: back to CONTEXT_A, a real episode boundary, read-only probe ----
    ep_num += 1
    agent.reset()  # WITH arm: this is where SleepLoopManager's last K=1 fire runs
    test_read, test_residue, test_memory = _probe_retrieval(agent, env_a)
    print(
        f"  [train] cond={condition_name} seed={seed}"
        f" ep {ep_num}/{total_eps} phase=TEST",
        flush=True,
    )

    retention_score = float(
        F.cosine_similarity(end_encode_read, test_read, dim=-1).item()
    )
    # Secondary, non-load-bearing diagnostic: raw slot-matrix cosine, immune
    # to query-side (encoder) drift -- see _probe_retrieval docstring.
    memory_matrix_retention_score = float(
        F.cosine_similarity(
            end_encode_memory.flatten(), test_memory.flatten(), dim=0
        ).item()
    )
    residue_delta = float(test_residue - end_encode_residue)

    # Per-cell completion signal for the runner's progress bar only (NOT the
    # experiment's scientific verdict -- that is the aggregate C1 criterion
    # computed in main() after all cells finish, per per_seed_comparisons).
    # "PASS" here means this cell's data collection succeeded (shared write
    # path engaged, retention_score is a real number), matching the
    # established v3_exq_993/951c per-cell "verdict: PASS/FAIL" convention
    # the runner's RE_RUN_DONE_PATTERNS regex requires.
    cell_ok = (encode_writes > 0) and (retention_score == retention_score)
    print(
        f"  verdict: {'PASS' if cell_ok else 'FAIL'} cond={condition_name} seed={seed}"
        f" retention_score={retention_score:.4f}"
        f" memory_matrix_retention_score={memory_matrix_retention_score:.4f}"
        f" slot_overlap_frac={slot_overlap_frac:.4f}"
        f" residue_delta={residue_delta:.4f}"
        f" encode_writes={encode_writes}/{encode_ticks}",
        flush=True,
    )

    return {
        "condition": condition_name,
        "seed": seed,
        "retention_score": retention_score,
        "memory_matrix_retention_score": memory_matrix_retention_score,
        "slot_overlap_frac": slot_overlap_frac,
        "n_encode_written_slots": len(encode_written_slots),
        "n_interfere_written_slots": len(interfere_written_slots),
        "end_encode_residue": end_encode_residue,
        "test_residue": test_residue,
        "residue_delta": residue_delta,
        "encode_writes": encode_writes,
        "encode_ticks": encode_ticks,
        "n_encode": n_encode,
        "n_interfere": n_interfere,
        "steps_per_episode": steps_per_ep,
        "agent": agent,
    }


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

def _config_slice(with_consolidation: bool) -> Dict[str, Any]:
    """Declares what each arm's computation reads (arm_fingerprint config_slice)."""
    return {
        "grid_size": GRID_SIZE,
        "context_a_hazards": CONTEXT_A_HAZARDS,
        "context_a_resources": CONTEXT_A_RESOURCES,
        "context_b_hazards": CONTEXT_B_HAZARDS,
        "context_b_resources": CONTEXT_B_RESOURCES,
        "n_encode": N_ENCODE,
        "n_interfere": N_INTERFERE,
        "steps_per_episode": STEPS_PER_EPISODE,
        "lr": LR,
        "sd016_writepath_mode": "sense_only",
        "with_consolidation": with_consolidation,
        "sws_consolidation_steps": 8,
        "rem_attribution_steps": 6,
        "shy_decay_rate": 0.98,
        "sleep_loop_episodes_K": 1,
    }


def main(dry_run: bool = False) -> Dict[str, Any]:
    t0 = time.perf_counter()
    conditions = [
        ("WITH_CONSOLIDATION", True),
        ("WITHOUT_CONSOLIDATION", False),
    ]
    arm_results: List[Dict[str, Any]] = []
    all_agents: List[REEAgent] = []

    n_seeds = len(SEEDS)
    total_cells = len(conditions) * n_seeds
    cell_num = 0

    for cond_name, with_consolidation in conditions:
        for seed in SEEDS:
            cell_num += 1
            print(
                f"[train] ep {cell_num}/{total_cells} cond={cond_name} seed={seed}",
                flush=True,
            )
            with arm_cell(
                seed,
                config_slice=_config_slice(with_consolidation),
                script_path=Path(__file__),
                config_slice_declared=True,
            ) as cell:
                row = run_cell(cond_name, with_consolidation, seed, dry_run=dry_run)
                agent_for_row = row.pop("agent")
                all_agents.append(agent_for_row)
                cell.stamp(row)
            arm_results.append(row)

    with_rows = [r for r in arm_results if r["condition"] == "WITH_CONSOLIDATION"]
    without_rows = [r for r in arm_results if r["condition"] == "WITHOUT_CONSOLIDATION"]

    # ---- Readiness precondition: shared online write path actually engaged ----
    min_expected_writes = WRITE_FLOOR_FRAC * (N_ENCODE if not dry_run else 3) * (
        STEPS_PER_EPISODE if not dry_run else 15
    )
    worst_write_frac = min(
        (r["encode_writes"] / max(1, r["encode_ticks"])) for r in arm_results
    )
    write_floor_met = worst_write_frac >= WRITE_FLOOR_FRAC

    # ---- Manipulation-check precondition: INTERFERE actually disturbed WITHOUT ----
    without_scores = [r["retention_score"] for r in without_rows]
    mean_without = sum(without_scores) / len(without_scores) if without_scores else 1.0
    disturbance_confirmed = mean_without < DISTURBANCE_CEILING

    preconditions = [
        {
            "name": "context_memory_write_path_engaged",
            "description": "Shared sd016 online write path fired on >= "
                            f"{WRITE_FLOOR_FRAC:.0%} of ENCODE ticks in every cell "
                            "(positive control on the mechanism both arms share).",
            "measured": worst_write_frac,
            "threshold": WRITE_FLOOR_FRAC,
            "direction": "lower",
            "control": "sd016_writepath_mode=sense_only set identically in both arms",
            "met": write_floor_met,
        },
        {
            "name": "interference_disturbance_confirmed",
            "description": "WITHOUT_CONSOLIDATION arm's mean retention_score is "
                            f"below {DISTURBANCE_CEILING} -- INTERFERE genuinely "
                            "competed for ContextMemory slots.",
            "measured": mean_without,
            "threshold": DISTURBANCE_CEILING,
            "direction": "upper",
            "control": "WITHOUT_CONSOLIDATION arm (no sleep to protect the slot)",
            "met": disturbance_confirmed,
        },
    ]

    degeneracy = check_degeneracy({
        "retention_score": {
            "groups": [
                [r["retention_score"] for r in with_rows],
                [r["retention_score"] for r in without_rows],
            ],
        },
    })

    readiness_ok = write_floor_met and disturbance_confirmed
    non_degenerate = readiness_ok and degeneracy.get("non_degenerate", True)

    per_seed_comparisons = []
    wins = 0
    for with_r, wo_r in zip(with_rows, without_rows):
        assert with_r["seed"] == wo_r["seed"], "Seed mismatch in comparison"
        win = with_r["retention_score"] > wo_r["retention_score"]
        wins += int(win)
        per_seed_comparisons.append({
            "seed": with_r["seed"],
            "with_retention_score": with_r["retention_score"],
            "without_retention_score": wo_r["retention_score"],
            "c1_retention_win": win,
            "with_memory_matrix_retention_score": with_r["memory_matrix_retention_score"],
            "without_memory_matrix_retention_score": wo_r["memory_matrix_retention_score"],
            "with_slot_overlap_frac": with_r["slot_overlap_frac"],
            "without_slot_overlap_frac": wo_r["slot_overlap_frac"],
            "with_residue_delta": with_r["residue_delta"],
            "without_residue_delta": wo_r["residue_delta"],
        })

    threshold = 2  # of 3 seeds, matching V3-EXQ-385/385a convention
    c1_pass = wins >= threshold

    # ATTRIBUTION ASYMMETRY (red-team fable finding, Family 1/3): retention_score
    # is causally reachable not only by "consolidation protects the A slot from
    # B interference" (the intended channel) but also by two side effects that
    # are exclusive to the WITH arm and unrelated to protection -- (a) SWS
    # writing INTERFERE-phase (B) experience into ContextMemory during its
    # INTERFERE-boundary and TEST-boundary fires, actively diluting the A
    # representation being measured, and (b) sws_consolidation_steps training
    # E1's encoder, moving the probe QUERY between the two probe moments even
    # if memory content were untouched. Both push retention_score DOWN in the
    # WITH arm for reasons that have nothing to do with "no consolidation
    # mechanism" -- so a WITH-loses-to-WITHOUT result cannot be read as
    # falsifying consolidation's protective effect; it is equally consistent
    # with the protective effect being real but outweighed, in this design, by
    # consolidation's own cross-context replay noise. A WITH-beats-WITHOUT
    # result has no such confound (the mechanism won despite carrying the
    # extra noise), so PASS/supports stays attributable but FAIL routes to
    # non_contributory, never weakens. See memory_matrix_retention_score /
    # slot_overlap_frac (per_seed_comparisons) as a starting point for a
    # future autopsy wanting to separate query-drift from memory-drift.
    if not non_degenerate:
        outcome = "FAIL"
        evidence_direction = "non_contributory"
        degeneracy_reason = degeneracy.get("degeneracy_reason", "")
        if not readiness_ok:
            reasons = []
            if not write_floor_met:
                reasons.append(
                    f"context_memory write path under floor "
                    f"({worst_write_frac:.2f} < {WRITE_FLOOR_FRAC})"
                )
            if not disturbance_confirmed:
                reasons.append(
                    f"WITHOUT arm showed no measurable interference "
                    f"(mean retention_score {mean_without:.4f} >= {DISTURBANCE_CEILING})"
                )
            degeneracy_reason = "; ".join(reasons) or degeneracy_reason
    elif c1_pass:
        outcome = "PASS"
        evidence_direction = "supports"
        degeneracy_reason = ""
    else:
        outcome = "FAIL"
        evidence_direction = "non_contributory"
        degeneracy_reason = (
            "C1 criterion not met (WITH did not beat WITHOUT in >=2/3 seeds), "
            "but a FAIL here does not license 'weakens': retention_score is "
            "reachable by WITH-exclusive consolidation side effects unrelated "
            "to slot protection (cross-context replay writing B experience "
            "into ContextMemory during INTERFERE/TEST-boundary sleep fires; "
            "encoder query drift from sws_consolidation_steps) that can "
            "depress WITH's score independent of whether protection occurred. "
            "See per_seed_comparisons memory_matrix_retention_score / "
            "slot_overlap_frac for a future autopsy's attribution check."
        )

    print(
        f"verdict: {outcome} C1_retention={c1_pass} ({wins}/{n_seeds} seeds)"
        f" readiness_ok={readiness_ok} non_degenerate={non_degenerate}"
        f" evidence_direction={evidence_direction}",
        flush=True,
    )

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"

    manifest: Dict[str, Any] = {
        "experiment_type": EXPERIMENT_TYPE,
        "run_id": run_id,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "claim_ids_tested": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "evidence_class": EVIDENCE_CLASS,
        "outcome": outcome,
        "timestamp_utc": ts,
        "evidence_direction": evidence_direction,
        "sleep_driver_pattern": (
            "K=1 single-fire (SleepLoopManager, fires every episode)"
        ),
        "non_degenerate": non_degenerate,
        "degeneracy_reason": degeneracy_reason,
        "interpretation": {
            "preconditions": preconditions,
            "criteria_non_degenerate": {
                "C1_retention_score": non_degenerate,
            },
        },
        "criteria": [
            {
                "name": "C1_retention_score_2of3_seeds",
                "load_bearing": True,
                "passed": bool(c1_pass),
            },
        ],
        "acceptance_checks": {
            "C1_retention_2of3_seeds": c1_pass,
            "C1_wins": wins,
            "readiness_ok": readiness_ok,
            "non_degenerate": non_degenerate,
        },
        "per_seed_comparisons": per_seed_comparisons,
        "arm_results": arm_results,
        "reuse_check_note": (
            "GOV-REUSE-1: decisive readout is retention_score, a novel "
            "comparison over a fixed-layout encode/interfere/test protocol "
            "with no precedent in the corpus (grep confirmed no existing "
            "manifest tags claim_ids=[\"EXT-007\"] or computes this DV; "
            "V3-EXQ-385/385a/691/127/792/792a all read for prior art -- none "
            "structure an early-episode-encode -> later-episode-test design; "
            "see module docstring). Not recoverable from existing manifests -> run."
        ),
        "params": {
            "n_encode": N_ENCODE if not dry_run else 3,
            "n_interfere": N_INTERFERE if not dry_run else 3,
            "steps_per_episode": STEPS_PER_EPISODE if not dry_run else 15,
            "seeds": SEEDS,
            "grid_size": GRID_SIZE,
            "context_a_hazards": CONTEXT_A_HAZARDS,
            "context_a_resources": CONTEXT_A_RESOURCES,
            "context_b_hazards": CONTEXT_B_HAZARDS,
            "context_b_resources": CONTEXT_B_RESOURCES,
            "write_floor_frac": WRITE_FLOOR_FRAC,
            "disturbance_ceiling": DISTURBANCE_CEILING,
            "sd016_writepath_mode": "sense_only",
            "dry_run": dry_run,
        },
    }

    if not dry_run:
        out_dir = resolve_evidence_experiments_dir(Path(__file__))
        os.makedirs(out_dir, exist_ok=True)
        out_path = write_flat_manifest(
            manifest,
            out_dir,
            dry_run=False,
            config=manifest.get("params"),
            seeds=SEEDS,
            script_path=Path(__file__),
            started_at=t0,
            agent=all_agents,
        )
        print(f"Results written to {out_path}", flush=True)
    else:
        out_path = write_flat_manifest(
            manifest,
            dry_run=True,
            config=manifest.get("params"),
            seeds=SEEDS,
            script_path=Path(__file__),
            started_at=t0,
            agent=all_agents,
        )
        print(f"[DRY RUN] run_id={run_id} outcome={outcome}", flush=True)

    print(f"Outcome: {outcome}", flush=True)
    manifest["_manifest_path"] = str(out_path)
    return manifest, out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="V3-EXQ-994: EXT-007 claim probe -- offline consolidation "
                     "cross-episode retention ablation (ARC-007/MECH-092)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run minimal episodes (3 encode + 3 interfere, 15 steps/ep) to verify wiring",
    )
    args = parser.parse_args()

    manifest, out_path = main(dry_run=args.dry_run)

    _outcome_raw = str(manifest["outcome"]).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
        run_id=manifest["run_id"],
        dry_run=args.dry_run,
    )
