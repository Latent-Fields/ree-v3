#!/opt/local/bin/python3
"""
V3-EXQ-943 -- ContextMemory write-address selection: real-agent validation of
both landed default-off mechanisms (conscience bias + refractory eligibility
mask) against substrate_queue `contextmemory-write-path-addressing-degeneracy`
(severity: corrupting).

Chip: chip-20260819-queueexp-contextmemory-writesel-validation

experiment_purpose: diagnostic

WHY THIS RUN EXISTS. `ContextMemory.write()` (ree_core/predictors/e1_deep.py)
addresses by a hard argmin over query_proj(state) . memory, which is a
deterministic single-slot fixed point under a near-constant query stream
(V3-EXQ-436e/436f, closed-form sign discriminator, predicted lock-vs-rotate
5/5). Two orthogonal, default-off mechanisms now exist to fix it:
  - E1Config.contextmemory_write_usage_balancing (bool, default False):
    conscience-bias SCORE adjustment, argmin(mean_scores + w*usage_ema*sqrt(d))
    (ree-v3 76cbf844).
  - E1Config.contextmemory_write_selection="refractory" (default "argmin") +
    .contextmemory_write_refractory_k (default 2): ELIGIBILITY-mask mechanism
    excluding the k most-recently-written slots (ree-v3, 2026-08-19).
Both cleared the registered acceptance floor (>=2 occupied slots on >=3/5
seeds) in the ContextMemory-UNIT-level contract tests AND in an independent
synthetic-stream probe (contextmemory_write_selection_comparison_20260819.md,
pre-registration fcfb311e4b, results b7e072ddf0) -- but neither has been run
on a REAL agent under real training dynamics (the 436e/436f harness: a
CausalGridWorldV2 REEAgent training loop), which is what actually produced
the original degeneracy finding. This experiment closes that gap. Both
mechanisms stay at substrate_queue status implemented_pending_validation
until this run is reviewed; this script does NOT flip that status.

TWO HARD DESIGN CONSTRAINTS (from the architecture doc and the pre-registered
probe; violating either reproduces the 436e/436f trap from the other
direction -- see REE_assembly/docs/architecture/
contextmemory_write_address_selection.md):

  1. NEITHER FLAG IS LEFT AT ITS DEFAULT in any arm. Every arm explicitly
     sets contextmemory_write_usage_balancing AND contextmemory_write_selection
     (plus contextmemory_gated_content_write=True, matching 436d/e/f -- without
     it, writes homogenize per the 436c defect regardless of the occupancy
     mask). The LEGACY arm sets both explicitly to their legacy values (rather
     than omitting them) so every arm's config is equally explicit.
  2. NOT POWERED ON OCCUPIED-SLOT COSINE (occ_cos) OR CLUSTER JACCARD. The
     pre-registered probe found occ_cos cannot discriminate these arms at 5
     seeds (every contrast |dz| <= 0.47, |t(4)| <= 1.04, sign-inconsistent;
     required n at 80% power ranges 38-2485 depending on the contrast), and
     Jaccard aliases against the conscience bias's period-16 round-robin
     cycle (exactly 0.0 on 3/5 seeds, 1.0 on 2/5 -- a bimodal artifact whose
     mean looks moderate). This script computes NEITHER. It uses only the
     DETERMINISTIC columns the probe found robust and exactly reproducible:
     n_occupied_slots, self_repeat_rate, entropy_bits, round_robin_agreement.

ARMS (3, all seeds x all arms -- no separate "conditions" dimension):
  LEGACY      contextmemory_write_usage_balancing=False,
              contextmemory_write_selection="argmin"          (bit-identical
              to the pre-fix path; included as a reference/floor, NOT gating
              -- see "Reference arm" below)
  BIAS        contextmemory_write_usage_balancing=True,
              contextmemory_write_selection="argmin"           (conscience bias)
  REFRACTORY  contextmemory_write_usage_balancing=False,
              contextmemory_write_selection="refractory",
              contextmemory_write_refractory_k=2               (eligibility mask)
A fourth "BOTH ON" arm is deliberately NOT queued: at the default bias weight
the usage term dominates content by 2-3 orders of magnitude (sqrt(memory_dim)
scaling), so the composed arm is byte-identical to BIAS alone (pinned by
test_the_conscience_bias_subsumes_the_refractory_mask_at_default_weight) --
running it here would be redundant compute for zero new information.

Reference arm (LEGACY): included so this run's operating point is auditable
against 436e/436f's own finding (near-constant query stream -> single-slot
lock), NOT as a load-bearing acceptance gate. A real training loop is not
guaranteed to reproduce the exact same degree of degeneracy 436e/436f's
particular env/config combination did; LEGACY's occupancy is reported
descriptively, and the PASS/FAIL gate below depends only on BIAS and
REFRACTORY, which is what the substrate_queue entry's own validation_experiment
field asks for ("both arms").

ACCEPTANCE CRITERION (registered verbatim in substrate_queue's failure_record
for contextmemory-write-path-addressing-degeneracy): >=2 occupied slots in
BOTH fix arms on >=3/5 seeds. This experiment's SOLE load-bearing gate is
that criterion, applied independently to BIAS and to REFRACTORY:
  C1_BIAS:       n_occupied_slots(BIAS)       >= 2 on >= 3/5 seeds
  C1_REFRACTORY: n_occupied_slots(REFRACTORY) >= 2 on >= 3/5 seeds
  PASS iff BOTH C1_BIAS and C1_REFRACTORY hold (given P0 met -- see below).

P0 READINESS (gates C1's interpretability -- a below-floor writepath
engagement means "not ready", never a substrate verdict): every one of the 15
cells (3 arms x 5 seeds) must record n_write_calls >= WRITE_CALLS_FLOOR. This
confirms E1Config.sd016_writepath_mode="sense_only" (held constant across all
three arms -- a substrate property, not the manipulation; see agent.py's SD-016
Part B2 per-tick hook, ree_core/agent.py:4860-4894) actually fired ContextMemory
.write() during training, on THIS run, rather than assuming it from the config
alone -- the exact "config-only knob list could not have armed anything"
failure V3-EXQ-436e made with sd016_enabled.

SUBSTRATE PROPERTIES held constant across all three arms (not part of the
manipulation, exactly as 436d/e/f's writepath_mode / requires_grad freeze /
contextmemory_gated_content_write were):
  - contextmemory_gated_content_write=True (436d write-path repair; still
    required so writes are content-bearing rather than homogenizing).
  - sd016_writepath_mode="sense_only" (per-tick ContextMemory writes during
    ordinary waking steps; no sleep pass of any kind is used in this
    experiment -- see "No sleep" below).
  - alpha_world=0.9 (SD-008; z_world fidelity, required for any experiment
    reading query_proj(state) where state includes z_world).
  - use_noise_floor=True (MECH-313/ARC-065; realistic exploration diversity
    in the baseline action-selection policy).
  - context_memory.memory.requires_grad_(False) after construction, on every
    agent (436e/f's Adam-drift-neutralization fix): write() mutates
    memory.data directly under its own torch.no_grad() regardless of this
    flag, so writes are unaffected; only gradient-descent perturbation of
    memory VALUES between writes is suppressed. This matters for THIS
    experiment's DVs (not just for the retired cosine DVs) because write()'s
    own addressing score (torch.mm(query, self.memory.t())) reads the same
    memory tensor Adam's backward pass through read() would otherwise
    perturb -- so leaving it unfrozen could let optimizer drift, not the
    selection mechanism under test, move which slot gets picked.
  - use_per_stream_vs=True, use_anchor_sets=True, use_sd039_anchor_payload=True
    (Phase 2 substrate template, validated by V3-EXQ-265a, reused verbatim
    across the 436 lineage and elsewhere).
  - use_salience_coordinator left at its REEConfig default (False). See
    "Step 2.5c substrate-path overlap" below.
  - sd016_enabled left at its REEConfig default (False). Verified by reading
    ree_core/predictors/e1_deep.py:436/760 and ree_core/agent.py:10136: it
    gates ONLY the cue-tagger/context-divergence-loss apparatus (irrelevant
    here), never the sd016_writepath_mode per-tick write hook this
    experiment actually depends on (ree_core/agent.py:4871, reads
    sd016_writepath_mode directly with no sd016_enabled precondition).

No sleep. This experiment does not set use_sleep_loop, sws_enabled, or
rem_enabled (all left at their REEConfig defaults, False) -- SD-016 sleep-cycle
consolidation is an unrelated, orthogonal question (a DIFFERENT substrate_queue
concern, the SD-017/ARC-045/MECH-166 sleep-differentiation claims) from
whether the WRITE-ADDRESS mechanism itself avoids the single-slot fixed point.
The architecture doc is explicit that "the read-path fix changes write-path
occupancy by ZERO seeds" -- so arming SD-016 or sleep would add compute and
complexity with no bearing on the question this experiment asks. No SLEEP
DRIVER docstring line is required (Step 3 rule) since no sleep flag is set;
sleep_driver_pattern is recorded in the manifest as "N/A (no sleep loop)".

Step 2.5c substrate-path overlap (recorded per the /queue-experiment gate):
open substrate_queue entry `mode-governance-engagement` (severity=corrupting)
lists ree_core/agent.py among its substrate_paths at whole-file granularity,
which this driver necessarily imports (REEAgent). Its actual subject is the
hard affinity-input box clamp / commitment-term default in the mode-governance
/ SalienceCoordinator machinery (use_salience_coordinator gate). This driver
never sets use_salience_coordinator=True (REEConfig's own default, False, is
used throughout) and never reads agent.operating_mode or any
SalienceCoordinator output -- so the overlap is removed by construction, not
merely noted, following the same precedent as V3-EXQ-942's identical overlap
disposition (q081_profile_kwargs() -> forced back to use_salience_coordinator
=False). No other open `corrupting` substrate_queue entry's substrate_paths
overlaps any module this driver imports or exercises (checked against the
2026-08-20 substrate_queue.json: mode-governance-engagement is the only
open-corrupting entry naming agent.py; contextmemory-write-path-addressing-
degeneracy itself, naming e1_deep.py, is the entry under test here, not an
unrelated one).

Step 2.5b re-derive brake: N/A. claim_ids is empty (see below) -- this
experiment tests substrate readiness/mechanism correctness, not a claim
hypothesis, so no claim's autopsy count applies.

Step 2.4 GOV-REUSE-1 (existing-evidence check): the decisive readout
(n_occupied_slots under a REAL agent's training dynamics, for BIAS and
REFRACTORY) is NOT recorded anywhere. The only prior measurement of these
mechanisms is contextmemory_write_selection_comparison_20260819.md's
independent probe, which is explicitly ContextMemory-UNIT-level (a synthetic
degenerate query stream constructed directly, no REEAgent, no environment) --
not a compatible substrate/measurement for this question. Not recoverable;
proceeding to run.

claim_ids: [] -- this experiment does not test SD-017 / ARC-045 / MECH-166 (the
sleep-differentiation claims those substrate items unblock); it validates
that the write-address FIX ITSELF holds under real training dynamics, which
is a substrate-readiness question, not a claim hypothesis test. Diagnostic
purpose experiments with claim_ids=[] are excluded from governance confidence
scoring by design (this run establishes readiness/context, not evidence for
or against any claim).

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
from typing import Any, Dict, List, Optional, Tuple

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


EXPERIMENT_TYPE = "v3_exq_943_contextmemory_write_selection_validation"
QUEUE_ID = "V3-EXQ-943"
CLAIM_IDS: List[str] = []
EXPERIMENT_PURPOSE = "diagnostic"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"

# --- Arms ------------------------------------------------------------------
ARMS: List[Tuple[str, bool, str, int]] = [
    # (label, write_usage_balancing, write_selection, write_refractory_k)
    ("LEGACY", False, "argmin", 2),
    ("BIAS", True, "argmin", 2),
    ("REFRACTORY", False, "refractory", 2),
]
ARM_NAMES = [a[0] for a in ARMS]

SEEDS: List[int] = [42, 7, 13, 100, 200]  # matches 436-family for external comparability

# Substrate properties held constant across all arms (not the manipulation).
CONTEXTMEMORY_GATED_CONTENT_WRITE = True
SD016_WRITEPATH_MODE = "sense_only"
ALPHA_WORLD = 0.9
USE_NOISE_FLOOR = True
NOISE_FLOOR_ALPHA = 0.1
NOISE_FLOOR_MIN_TEMPERATURE = 1.0
BASELINE_TEMPERATURE = 1.0

TRAINING_EPISODES = 200
STEPS_PER_EPISODE = 150
CONTEXT_SWITCH_EVERY = 5
MAX_HARM_BUF = 4000

NUM_SLOTS = 16  # ContextMemory default (E1DeepPredictor constructs it with num_slots=16)

# P0 readiness floor: pooled write() calls per cell must clear this for the
# writepath to count as genuinely engaged this run (436e/436f's
# waking_writepath_engaged check, per-cell rather than pooled since this
# experiment has no separate NO_WRITES negative control to pool against).
WRITE_CALLS_FLOOR = 200.0

# C1 acceptance criterion (registered in substrate_queue's failure_record).
C1_OCCUPIED_SLOTS_FLOOR = 2
C1_N_SEEDS_REQUIRED = 3


# ------------------------------------------------------------------ #
# Env / agent helpers (env params reused from the 436 lineage)             #
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


def _make_agent(env: CausalGridWorldV2, write_usage_balancing: bool,
                 write_selection: str, write_refractory_k: int) -> REEAgent:
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
        contextmemory_write_usage_balancing=write_usage_balancing,
        contextmemory_write_selection=write_selection,
        contextmemory_write_refractory_k=write_refractory_k,
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
    # 436e/f Adam-drift neutralization (see module docstring). write() still
    # mutates memory.data directly under its own no_grad block regardless.
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
    temperature. Reused verbatim from the 436 lineage's baseline policy.
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
    (the authoritative instrumentation, maintained in every selection mode
    including legacy) after every agent.sense() call -- never by recomputing
    write()'s own argmin/scoring expression, which V3-EXQ-436f's tracker did
    and which silently reports the wrong slot once the selection rule is
    configurable (see module docstring / architecture doc "Instrumentation
    change"). Polling once per sense() call is exact (not an approximation):
    the SD-016 Part B2 hook fires at most one write() per sense() call, so
    the pooled write count increases by at most 1 between polls.
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
    """The four robust, deterministic columns identified by the pre-registered
    probe (contextmemory_write_selection_comparison_20260819.md section 5) --
    NOT occ_cos, NOT cluster Jaccard (see module docstring constraint 2).
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

    # Round-robin agreement: fraction of writes landing on the strict
    # least-recently-used slot (never-written slots are maximally "LRU" via
    # last_write_tick=-1; ties broken by lowest index, deterministic).
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
    """Run a single training episode: sense -> baseline action selection ->
    step -> prediction-loss training -> harm_eval training. No sleep, no
    context-conditioned action selection, no SD-016 arming -- this experiment
    isolates the write-address mechanism from every other manipulation.
    """
    _, obs_dict = env.reset()
    agent.reset()
    agent.e1.reset_hidden_state()

    for _step in range(steps):
        obs_body = obs_dict["body_state"]
        obs_world = obs_dict["world_state"]
        obs_harm = obs_dict.get("harm_obs", None)

        latent = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
        agent.clock.advance()
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


def _run_cell(arm_name: str, write_usage_balancing: bool, write_selection: str,
              write_refractory_k: int, seed: int, base_config_slice: Dict[str, Any],
              zg: ZGoalStreamAccumulator, training_episodes: int,
              steps_per_episode: int) -> Dict[str, Any]:
    # Per-cell config slice: the shared base PLUS this cell's own arm flags --
    # without the arm flags, two different arms at the same seed would
    # fingerprint identically (config_slice would not vary by arm).
    cell_config_slice = {
        **base_config_slice,
        "arm": arm_name,
        "contextmemory_write_usage_balancing": write_usage_balancing,
        "contextmemory_write_selection": write_selection,
        "contextmemory_write_refractory_k": write_refractory_k,
    }
    with arm_cell(seed, config_slice=cell_config_slice, script_path=Path(__file__)) as cell:
        env_safe = _make_env_safe(seed)
        env_dang = _make_env_dangerous(seed)
        agent = _make_agent(env_safe, write_usage_balancing, write_selection, write_refractory_k)

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
            if (ep + 1) % 50 == 0 or (ep + 1) == training_episodes:
                print(
                    f"  [train] arm={arm_name} seed={seed} "
                    f"ep {ep + 1}/{training_episodes} "
                    f"n_writes={len(tracker.sequence)}",
                    flush=True,
                )

        zg.observe(agent)
        dvs = _compute_deterministic_dvs(tracker.sequence, NUM_SLOTS)
        row: Dict[str, Any] = {
            "arm": arm_name,
            "seed": seed,
            "write_usage_balancing": write_usage_balancing,
            "write_selection": write_selection,
            "write_refractory_k": write_refractory_k,
            **dvs,
        }
        cell.stamp(row)

    passed = dvs["n_occupied_slots"] >= C1_OCCUPIED_SLOTS_FLOOR
    print(f"verdict: {'PASS' if passed else 'FAIL'}", flush=True)
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
        "sd016_writepath_mode": SD016_WRITEPATH_MODE,
        "alpha_world": ALPHA_WORLD,
        "use_noise_floor": USE_NOISE_FLOOR,
        "num_slots": NUM_SLOTS,
    }

    arm_results: List[Dict[str, Any]] = []
    for arm_name, wub, wsel, wrk in ARMS:
        for seed in seeds:
            row = _run_cell(
                arm_name, wub, wsel, wrk, seed, full_config_slice, zg,
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

    # --- C1: registered acceptance criterion, applied to BIAS and REFRACTORY ---
    def _seeds_clearing_floor(arm_name: str) -> int:
        return sum(
            1 for row in by_arm[arm_name]
            if row["n_occupied_slots"] >= C1_OCCUPIED_SLOTS_FLOOR
        )

    n_pass_bias = _seeds_clearing_floor("BIAS")
    n_pass_refractory = _seeds_clearing_floor("REFRACTORY")
    n_pass_legacy = _seeds_clearing_floor("LEGACY")  # descriptive only, non-gating

    c1_bias_pass = n_pass_bias >= C1_N_SEEDS_REQUIRED
    c1_refractory_pass = n_pass_refractory >= C1_N_SEEDS_REQUIRED
    overall_pass = c1_bias_pass and c1_refractory_pass

    criteria = [
        {
            "name": "C1_BIAS_occupancy_floor",
            "load_bearing": True,
            "passed": c1_bias_pass,
            "n_seeds_passed": n_pass_bias,
            "n_seeds_required": C1_N_SEEDS_REQUIRED,
        },
        {
            "name": "C1_REFRACTORY_occupancy_floor",
            "load_bearing": True,
            "passed": c1_refractory_pass,
            "n_seeds_passed": n_pass_refractory,
            "n_seeds_required": C1_N_SEEDS_REQUIRED,
        },
        {
            "name": "LEGACY_reference_reproduces_known_degeneracy",
            "load_bearing": False,
            "passed": n_pass_legacy < C1_N_SEEDS_REQUIRED,
            "n_seeds_clearing_floor": n_pass_legacy,
            "note": (
                "Descriptive only -- confirms this run's operating point is "
                "comparable to 436e/436f's near-constant-query-stream lock. "
                "Not part of the PASS/FAIL gate (see module docstring "
                "'Reference arm')."
            ),
        },
    ]
    criteria_non_degenerate = {
        "C1_BIAS_occupancy_floor": len(seeds) >= 2,
        "C1_REFRACTORY_occupancy_floor": len(seeds) >= 2,
    }

    if overall_pass:
        label = "write_address_fix_validated_under_real_agent"
    else:
        label = "write_address_fix_not_confirmed_under_real_agent"
    # claim_ids is empty -- this diagnostic validates a SUBSTRATE mechanism,
    # not a claim hypothesis, so it cannot "support" or "weaken" a claim
    # regardless of PASS/FAIL. non_contributory is the correct reading in
    # both directions (matches the convention used by other claim_ids=[]
    # diagnostics, e.g. V3-EXQ-899).
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
        metrics[f"n_seeds_clearing_occupancy_floor_{arm_name}"] = _seeds_clearing_floor(arm_name)

    summary_markdown = (
        f"# {QUEUE_ID} -- ContextMemory write-address selection, real-agent validation\n\n"
        f"**Status:** {status}  **Label:** {label}\n"
        f"**Purpose:** diagnostic (claim_ids=[]; validates substrate_queue "
        f"contextmemory-write-path-addressing-degeneracy)\n\n"
        f"| Arm | seeds >= {C1_OCCUPIED_SLOTS_FLOOR} occupied | n_write_calls range |\n"
        f"|---|---|---|\n"
        + "".join(
            f"| {arm_name} | {_seeds_clearing_floor(arm_name)}/{len(by_arm[arm_name])} "
            f"| {min(r['n_write_calls'] for r in by_arm[arm_name])}-"
            f"{max(r['n_write_calls'] for r in by_arm[arm_name])} |\n"
            for arm_name in ARM_NAMES
        )
        + f"\nC1_BIAS: {'PASS' if c1_bias_pass else 'FAIL'} "
        f"({n_pass_bias}/{len(by_arm['BIAS'])} >= {C1_N_SEEDS_REQUIRED} required)\n"
        f"C1_REFRACTORY: {'PASS' if c1_refractory_pass else 'FAIL'} "
        f"({n_pass_refractory}/{len(by_arm['REFRACTORY'])} >= {C1_N_SEEDS_REQUIRED} required)\n"
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
        "sd016_writepath_mode": SD016_WRITEPATH_MODE,
        "alpha_world": ALPHA_WORLD,
        "use_noise_floor": USE_NOISE_FLOOR,
        "num_slots": NUM_SLOTS,
        "c1_occupied_slots_floor": C1_OCCUPIED_SLOTS_FLOOR,
        "c1_n_seeds_required": C1_N_SEEDS_REQUIRED,
        "write_calls_floor": WRITE_CALLS_FLOOR,
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
