#!/opt/local/bin/python3
"""
V3-EXQ-149b -- Q-004: tau_R Calibration at a Larger Training Budget (instrument fix + budget replication).

Claim:      Q-004 ("How to calibrate tau_R relative to E1/E2?")
Proposal:   EXP-0519 (EVB-0179)
Supersedes: V3-EXQ-149 (v3_exq_149_q004_tau_r_20260329T130911Z_v3, PARTIAL, 2026-03-29)

WHY THIS IS A LETTERED ITERATION, NOT A NEW EXQ NUMBER
-------------------------------------------------------
Q-004's own `evidence_quality_note` and `what_would_answer` (claims.yaml) ask for
exactly one thing: replicate the FAST/DEFAULT/SLOW tau_R sweep "at a training
budget large enough that residue accumulation actually feeds back into
behaviour". Same scientific question, same three conditions, same acceptance
logic (C1-C5) -- so this is a lettered follow-up per the EXQ-versioning policy
("same scientific question -> letter; different mechanism -> new number"),
despite the code changing substantially (see below).

WHAT EXQ-149 ACTUALLY MEASURED (re-derived from re-reading the script, not
assumed): EXQ-149's action selection was UNCONDITIONALLY RANDOM in both
training and eval --

    action_idx = random.randint(0, ACTION_DIM - 1)

-- at every single step, in every condition. `agent.e3.harm_eval(z_world)` was
computed and trained via supervised MSE against the immediate harm signal, but
its OUTPUT was never consulted when choosing an action, and
`residue_field.evaluate_trajectory(...)` (the function that actually reads the
manipulated tau_R field for trajectory scoring, `e3_selector.py:966`) was never
called at all. So `mean_harm` under a uniform-random policy CANNOT depend on
tau_R by construction, for ANY training budget: nothing in the loop reads the
residue field, or E3's harm prediction, before choosing an action. This is the
DV-SYMMETRY class of defect the /queue-experiment skill's "DV-SYMMETRY
INVARIANCE" section warns about -- the manipulation (tau_R condition) is
invariant under the DV (harm from a policy that is a pure function of an
independent RNG stream, not of z_world/residue at all). EXQ-149's own finding
("ALL conditions showed identical mean_harm ~9.085") is not surprising evidence
that tau_R is behaviourally inert -- it is close to an ARITHMETIC IDENTITY,
because harm outcomes under i.i.d. random actions in a stationary-ish env
converge to the same expectation regardless of any agent-internal state that
the policy never reads. Simply re-running the same design at a 10x budget would
have reproduced this non-finding at 10x the compute cost, which is exactly the
failure this rule exists to prevent (see also GOV-REUSE-1 -- there is nothing
to reanalyse here because the original manifest cannot answer this either).

THE FIX (this is what makes the budget increase meaningful, not just bigger)
-----------------------------------------------------------------------------
Action selection now goes through the SUBSTRATE's real decision pathway:

    candidates = agent.hippocampal.propose_trajectories(z_world, z_self, num_candidates=NUM_CANDIDATES)
    action     = agent.select_action(candidates, ticks)

`propose_trajectories`'s CEM inner loop scores candidates against the residue
field terrain directly ("Score via residue field terrain (ARC-007 strict)";
`hippocampal/module.py`), and `agent.select_action` -> E3's scoring additionally
reads `compute_residue_cost` -> `residue_field.evaluate_trajectory(world_seq)`
(`e3_selector.py:966-974`). This is TWO independent causal channels from the
manipulated tau_R field into the executed action, neither of which existed in
EXQ-149. This is the SAME wiring already used and reviewed on this exact
substrate/env by V3-EXQ-120a (ARC-018), adapted here rather than hand-rolled,
to keep the integration risk to code that is already exercised elsewhere.

This is an INSTRUMENT/MEASUREMENT repair, not a redesign of the scientific
question (same claim, same conditions, same thresholds) -- it belongs in the
"instrument defect" carve-out of the /queue-experiment re-derive-brake logic,
not a "substrate_ceiling" verdict. (No failure_autopsy exists yet for Q-004 /
EXQ-149 at all -- re-derive-brake count = 0, well under threshold; this fix is
volunteered by the mandatory Step 3.5 DV-symmetry / substrate-API-correctness
review, not forced by a brake.)

nav_bias (forced hazard-approach on a fraction of TRAINING steps only, never at
eval) is retained from the EXQ-120a pattern so all three tau conditions accumulate
COMPARABLE harm EXPOSURE during training regardless of how quickly the
residue-informed policy learns to avoid hazards -- isolating tau_R's timescale
as the sole manipulated variable, rather than confounding it with "did this
condition even see enough harm events to build a field". Harm-exposure parity
across conditions is recorded as a diagnostic (not a scored criterion).

DV-SYMMETRY DECLARATION (mandatory; one line per condition)
-------------------------------------------------------------
DV = harm accumulated during EVAL (no nav_bias), a function of the realised
state trajectory under E3's residue-informed selection. Symmetry group: none of
the three conditions differ in ANYTHING except accumulation_rate/kernel_bandwidth
in ResidueConfig -- same env, same seeds, same nav_bias schedule, same network
init (torch.manual_seed(seed) reset identically per condition), so any DV
difference is attributable to tau_R alone. This is NOT a broadcast-constant,
monotone-rescaling, or permutation manipulation -- residue evaluated over a
trajectory changes both the SHAPE (kernel_bandwidth) and RATE
(accumulation_rate) of the field the CEM/E3 scoring reads, which is exactly a
manipulation the harm-selection DV is NOT invariant under (unlike EXQ-149's
random-action DV, which was invariant under it by construction).

WHAT WAS DELIBERATELY NOT ADDED (scope decisions, stated explicitly)
-----------------------------------------------------------------------
- E2 world-forward training (`agent.compute_e2_loss()` is called for
  consistency with the EXQ-120a pattern, but its transition buffer is never
  populated via `agent.record_transition(...)`, so it is a harmless zero-loss
  placeholder in both scripts). The CEM's candidate-refinement step scores
  candidates against the RESIDUE FIELD directly (not against E2 rollout
  accuracy), so an untrained E2 world_forward makes the rollout LOCALISATION
  noisier but does not remove the tau_R -> selection channel: the rolled-out
  world_seq is a condition-invariant function of (seed, E2 init, actions)
  while `residue_field.evaluate_trajectory(world_seq)` genuinely differs by
  condition. Adding E2 world-forward training (via
  `world_forward_contrastive_loss`, SD-056) would improve SNR but adds a
  materially larger, less-precedented integration than this follow-up's scope
  warrants; flagged as a candidate refinement if this run reads as
  under-powered rather than genuinely null.
- Coverage/visitation-entropy stats (distinct cells visited) -- the Recording
  Standard's "record generously" list recommends this for exploration/nav
  envs; omitted here to bound the scope of an already-substantial rewrite.
  Residue and harm statistics are recorded richly (see below); this is the one
  generous-recording category knowingly left out.
- z_goal / label_balance -- not applicable (no GoalState / no trained label in
  this design). z_goal_stream_stats is still wired via ZGoalStreamAccumulator
  for completeness; expect active_frac=0/unmeasured (correct, not a defect --
  z_goal is never engaged by this design).

Conditions (values UNCHANGED from EXQ-149 -- same manipulation, same magnitudes)
---------------------------------------------------------------------------------
FAST_TAU:    accumulation_rate=0.5,  kernel_bandwidth=0.5   (matched to E2 ~30-step horizon)
DEFAULT_TAU: accumulation_rate=0.1,  kernel_bandwidth=1.0   (current substrate default)
SLOW_TAU:    accumulation_rate=0.02, kernel_bandwidth=2.0   (matched to E1 ~200-step horizon)

Seeds: [42, 123] (unchanged from EXQ-149, for direct comparability)
Env:   CausalGridWorldV2 size=10, 2 hazards, 3 resources, hazard_harm=0.05 (unchanged)

Pre-registered thresholds (UNCHANGED from EXQ-149 -- same acceptance logic)
------------------------------------------------------------------------------
C1: FAST_TAU reduces harm vs SLOW_TAU by >= 10% (both seeds).
C2: SLOW_TAU reduces harm vs FAST_TAU by >= 10% (both seeds).
C3: DEFAULT_TAU ranks monotonically between FAST and SLOW (both seeds).
C4: FAST_TAU eval residue_contrast > 0.05 (both seeds).
C5: seed consistency of whichever direction C1/C2 (or the no-winner direction) shows.
PASS (fast/slow) => C1|C2 + C4 + C5.  PARTIAL => C4 only.  FAIL => C4 fails.

Non-gating readiness diagnostics (NEW -- guards the exact defect this run fixes)
------------------------------------------------------------------------------
R1 candidate_action_diversity: distinct TRUE first-action classes across a
   post-training positive-control candidate set (>= 1.5, i.e. >= 2 classes) --
   confirms the hippocampal proposer is not degenerate (EXQ-120's own failure
   mode surviving into a differently-wired script).
R2 hippo_fallback_rate: fraction of non-forced ticks where propose_trajectories
   returned no candidates and the driver fell back to uniform random. Recorded
   per condition x seed; a high rate would mean the "fix" silently degraded
   back toward EXQ-149's random policy for a large fraction of ticks.
Both are recorded as `interpretation.preconditions[]` even though
EXPERIMENT_PURPOSE="evidence" does not strictly require it (Step 3.5's
adjudication gate is mandatory only for diagnostic/baseline) -- extra rigor,
essentially free, and targeted directly at the known failure mode.

Estimated runtime: see queue entry note -- calibrated empirically via a timed
partial run before finalising WARMUP_EPISODES (CEM candidate generation is
substantially more expensive per step than EXQ-149's bare env.step() loop).

SLEEP: not used (no sleep flags set) -- no SLEEP DRIVER line required.
"""

import random
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.hippocampal.module import HippocampalModule
from ree_core.utils.config import REEConfig

from experiment_protocol import emit_outcome
from experiments.pack_writer import write_flat_manifest
from experiments._lib.arm_fingerprint import arm_cell
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator
from experiments._metrics import check_degeneracy


EXPERIMENT_TYPE = "v3_exq_149b_q004_tau_r_calibration_largebudget"
CLAIM_IDS = ["Q-004"]
EXPERIMENT_PURPOSE = "evidence"
BACKLOG_ID = "EVB-0179"
SUPERSEDES = "v3_exq_149_q004_tau_r_20260329T130911Z_v3"

# ---------------------------------------------------------------------------
# Pre-registered thresholds -- UNCHANGED from EXQ-149
# ---------------------------------------------------------------------------
THRESH_FAST_WIN = 0.90
THRESH_SLOW_WIN = 0.90
THRESH_CONTRAST = 0.05

# R1 non-degeneracy floor (mirrors V3-EXQ-120a's FLOOR_CANDIDATE_ACTION_CLASSES)
FLOOR_CANDIDATE_ACTION_CLASSES = 1.5

# ---------------------------------------------------------------------------
# Model / substrate constants
# ---------------------------------------------------------------------------
BODY_OBS_DIM = 12
WORLD_OBS_DIM = 250   # CausalGridWorldV2 size=10, use_proxy_fields=True (the default)
ACTION_DIM = 5
WORLD_DIM = 32
SELF_DIM = 16

SEEDS = [42, 123]

# Training budget -- SUBSTANTIALLY larger than EXQ-149 (WARMUP_EPISODES=300,
# EVAL_EPISODES=100), but sized against a REAL empirical timing, not a naive
# multiplier: a timed calibration run (10 warmup + 5 eval eps, this substrate,
# NUM_CANDIDATES=32) measured ~32.2 sec/episode -- the hippocampal CEM proposer
# (the DV-symmetry fix) is the dominant cost and makes each episode ~3x more
# expensive than EXQ-149's bare env.step() loop, so a naive 10x episode
# multiplier would have meant several DAYS of wall-clock across 6 cells.
# 2x warmup / 1.5x eval keeps the increase genuine (both more episodes AND,
# unlike EXQ-149, now genuinely selection-relevant compute) while landing at
# ~6.7 hr/cell x 6 cells ~= 40 hr total -- see the queue entry note.
WARMUP_EPISODES = 600
EVAL_EPISODES = 150
STEPS_PER_EPISODE = 200
LR = 3e-4

# Candidate-generation budget for the hippocampal CEM proposer. 32 is the
# substrate default (HippocampalConfig.num_candidates) and avoids the
# num_elite=1 NaN-std trap documented in V3-EXQ-120a (elite_fraction=0.2 * 32 = 6).
NUM_CANDIDATES = 32

# Fraction of TRAINING steps (never eval) where the action is forced toward a
# hazard via the proxy hazard_field_view gradient, guaranteeing comparable harm
# EXPOSURE across conditions regardless of how fast the residue-informed policy
# learns to avoid hazards. Carried from the V3-EXQ-120a pattern.
NAV_BIAS_TRAIN_FRAC = 0.4

# Post-training positive-control probe length for the R1 diversity precondition.
CONTROL_PROBE_STEPS = 25

# tau_R condition parameters -- UNCHANGED from EXQ-149
CONDITIONS = {
    "FAST_TAU": {
        "accumulation_rate": 0.5,
        "kernel_bandwidth": 0.5,
    },
    "DEFAULT_TAU": {
        "accumulation_rate": 0.1,
        "kernel_bandwidth": 1.0,
    },
    "SLOW_TAU": {
        "accumulation_rate": 0.02,
        "kernel_bandwidth": 2.0,
    },
}


# ---------------------------------------------------------------------------
# Environment / config factories
# ---------------------------------------------------------------------------

def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed,
        size=10,
        num_hazards=2,
        num_resources=3,
        hazard_harm=0.05,
        env_drift_interval=15,
        env_drift_prob=0.2,
        use_proxy_fields=True,   # explicit -- CausalGridWorldV2 defaults this True;
                                 # stated here because _hazard_approach_action depends on it.
    )


def _make_config(accumulation_rate: float, kernel_bandwidth: float) -> REEConfig:
    """Build via REEConfig.from_dims (wires action_dim through e2/hippocampal/e3
    correctly -- EXQ-149 never needed this since it never used select_action),
    then apply the tau_R residue overrides directly (the EXQ-149-proven pattern
    for the one thing actually being manipulated)."""
    cfg = REEConfig.from_dims(
        body_obs_dim=BODY_OBS_DIM,
        world_obs_dim=WORLD_OBS_DIM,
        action_dim=ACTION_DIM,
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
        alpha_world=0.95,   # SD-008
        alpha_self=0.3,
        reafference_action_dim=0,
    )
    cfg.residue.accumulation_rate = accumulation_rate
    cfg.residue.kernel_bandwidth = kernel_bandwidth
    cfg.residue.num_basis_functions = 32
    cfg.residue.decay_rate = 0.0        # invariant: residue cannot be erased
    cfg.residue.benefit_terrain_enabled = False
    return cfg


# ---------------------------------------------------------------------------
# Substrate helpers (selection pathway -- adapted from V3-EXQ-120a)
# ---------------------------------------------------------------------------

def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


def _hazard_approach_action(env: CausalGridWorldV2, n_actions: int,
                            rng: random.Random) -> int:
    """Action moving up the hazard-proximity gradient. Fallback: random.

    Training-phase forcing function only (nav_bias) -- never called during
    eval, so it cannot touch the DV. Requires use_proxy_fields=True (the
    CausalGridWorldV2 default), which supplies world_state[225:250] as the
    5x5 hazard_field_view regardless of grid size.
    """
    obs_dict = env._get_observation_dict()
    world_state = obs_dict.get("world_state", None)
    if world_state is None or not env.use_proxy_fields:
        return rng.randint(0, n_actions - 1)
    field_view = np.asarray(world_state[225:250]).reshape(5, 5)
    deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # matches ACTIONS 0..3
    vals = []
    for dr, dc in deltas:
        r, c = 2 + dr, 2 + dc
        if 0 <= r < 5 and 0 <= c < 5:
            vals.append(float(field_view[r, c]))
        else:
            vals.append(-1.0)
    return int(np.argmax(vals))


def _select_action(
    agent, latent, ticks, n_actions: int, rng: random.Random, num_candidates: int,
) -> Tuple[torch.Tensor, Optional[int], bool]:
    """Select via the canonical E3-over-hippocampal-candidates path.

    `agent.select_action(candidates, ticks)` routes through E3's J(zeta),
    which reads `compute_residue_cost` -> `residue_field.evaluate_trajectory`
    -- the causal channel EXQ-149 lacked entirely. `propose_trajectories`
    itself also scores its CEM refinement against the residue field terrain.

    Returns (action_tensor, n_distinct_true_candidate_action_classes, used_hippocampal).
    """
    with torch.no_grad():
        candidates = agent.hippocampal.propose_trajectories(
            latent.z_world.detach(),
            z_self=latent.z_self.detach(),
            num_candidates=num_candidates,
        )
        if not candidates:
            return (
                _action_to_onehot(rng.randint(0, n_actions - 1), n_actions, agent.device),
                None,
                False,
            )
        classes = set()
        for traj in candidates:
            cls = HippocampalModule.candidate_first_action_class(traj)
            if cls is not None:
                classes.add(int(cls))
        action = agent.select_action(candidates, ticks)
        return action, len(classes), True


def _probe_candidate_action_diversity(
    agent, env, n_actions: int, rng: random.Random, n_steps: int, num_candidates: int,
) -> float:
    """R1 positive control, measured AFTER training and BEFORE the scored eval.

    Reports the MINIMUM observed candidate-class count across the probe (not
    the mean), matching the V3-EXQ-120a convention -- the claim is that the
    candidate set could offer E3 a choice on every tick it mattered, and a
    mean would hide the ticks where it could not.
    """
    counts: List[int] = []
    _, obs_dict = env.reset()
    agent.reset()
    for _ in range(n_steps):
        with torch.no_grad():
            obs_body = torch.tensor(obs_dict["body_state"], dtype=torch.float32)
            obs_world = torch.tensor(obs_dict["world_state"], dtype=torch.float32)
            latent = agent.sense(obs_body, obs_world)
            ticks = agent.clock.advance()
            _action, n_classes, used = _select_action(
                agent, latent, ticks, n_actions, rng, num_candidates
            )
            if used and n_classes is not None:
                counts.append(int(n_classes))
            _, _reward, done, _info, obs_dict = env.step(_action)
            if done:
                _, obs_dict = env.reset()
                agent.reset()
    return float(min(counts)) if counts else 0.0


# ---------------------------------------------------------------------------
# Run one condition x seed
# ---------------------------------------------------------------------------

def _run_condition(
    seed: int,
    condition: str,
    warmup_episodes: int,
    eval_episodes: int,
    steps_per_episode: int,
    lr: float,
    zg_accum: ZGoalStreamAccumulator,
    dry_run: bool,
) -> Dict:
    """Run one condition for one seed. Returns per-cell result dict."""
    print(f"Seed {seed} Condition {condition}", flush=True)

    if dry_run:
        warmup_episodes = min(3, warmup_episodes)
        eval_episodes = min(2, eval_episodes)
        steps_per_episode = min(20, steps_per_episode)

    params = CONDITIONS[condition]
    config_slice = {
        "condition": condition,
        "accumulation_rate": params["accumulation_rate"],
        "kernel_bandwidth": params["kernel_bandwidth"],
        "env": {
            "size": 10, "num_hazards": 2, "num_resources": 3,
            "hazard_harm": 0.05, "env_drift_interval": 15, "env_drift_prob": 0.2,
            "use_proxy_fields": True,
        },
        "dims": {"body_obs_dim": BODY_OBS_DIM, "world_obs_dim": WORLD_OBS_DIM,
                 "action_dim": ACTION_DIM, "world_dim": WORLD_DIM, "self_dim": SELF_DIM},
        "warmup_episodes": warmup_episodes, "eval_episodes": eval_episodes,
        "steps_per_episode": steps_per_episode, "lr": lr,
        "num_candidates": NUM_CANDIDATES, "nav_bias_train_frac": NAV_BIAS_TRAIN_FRAC,
    }

    with arm_cell(
        seed,
        config_slice=config_slice,
        script_path=Path(__file__),
    ) as cell:
        torch.manual_seed(seed)
        random.seed(seed)
        rng = random.Random(seed)

        env = _make_env(seed)
        cfg = _make_config(
            accumulation_rate=params["accumulation_rate"],
            kernel_bandwidth=params["kernel_bandwidth"],
        )
        agent = REEAgent(cfg)
        agent.train()

        optimizer = optim.Adam(list(agent.parameters()), lr=lr)

        # ------------------- TRAIN (warmup) -----------------------------
        harm_events_train = 0
        steps_train = 0
        hippo_fallbacks_train = 0
        nonforced_ticks_train = 0

        for ep in range(warmup_episodes):
            _, obs_dict = env.reset()
            agent.reset()

            for _step in range(steps_per_episode):
                obs_body = torch.tensor(obs_dict["body_state"], dtype=torch.float32)
                obs_world = torch.tensor(obs_dict["world_state"], dtype=torch.float32)

                latent = agent.sense(obs_body, obs_world)
                ticks = agent.clock.advance()
                z_world_cur = latent.z_world.detach()

                forced = rng.random() < NAV_BIAS_TRAIN_FRAC
                if forced:
                    action = _action_to_onehot(
                        _hazard_approach_action(env, ACTION_DIM, rng),
                        ACTION_DIM, agent.device,
                    )
                    agent._last_action = action
                else:
                    nonforced_ticks_train += 1
                    action, _n_classes, used = _select_action(
                        agent, latent, ticks, ACTION_DIM, rng, NUM_CANDIDATES
                    )
                    if not used:
                        hippo_fallbacks_train += 1

                harm_pred = agent.e3.harm_eval(z_world_cur)

                _, reward, done, _info, obs_dict = env.step(action)
                steps_train += 1
                harm_signal = float(min(0.0, reward))

                if harm_signal < 0.0:
                    harm_events_train += 1
                    agent.residue_field.accumulate(
                        z_world_cur,
                        harm_magnitude=abs(harm_signal),
                        hypothesis_tag=False,
                    )

                harm_target = torch.tensor([abs(harm_signal)], dtype=torch.float32)
                e3_loss = nn.functional.mse_loss(harm_pred.view(-1), harm_target)
                total_loss = (
                    agent.compute_prediction_loss()
                    + agent.compute_e2_loss()
                    + e3_loss
                )
                if total_loss.requires_grad:
                    optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                    optimizer.step()

                if done:
                    _, obs_dict = env.reset()
                    agent.reset()

            if (ep + 1) % 50 == 0 or ep == warmup_episodes - 1:
                print(
                    f"  [train] {condition} seed={seed} ep {ep+1}/{warmup_episodes}"
                    f" harm_events={harm_events_train}"
                    f" harm_rate={harm_events_train / max(1, steps_train):.4f}"
                    f" hippo_fallback_rate={hippo_fallbacks_train / max(1, nonforced_ticks_train):.4f}",
                    flush=True,
                )

        # ------------------- R1 readiness probe -------------------------
        agent.eval()
        control_steps = min(5, CONTROL_PROBE_STEPS) if dry_run else CONTROL_PROBE_STEPS
        candidate_diversity_min = _probe_candidate_action_diversity(
            agent, env, ACTION_DIM, rng, control_steps, NUM_CANDIDATES
        )

        # ------------------- EVAL (no nav_bias, no grad) -----------------
        eval_harms: List[float] = []
        eval_residue_at_harm: List[float] = []
        eval_residue_elsewhere: List[float] = []
        hippo_fallbacks_eval = 0
        eval_ticks = 0

        for ep in range(eval_episodes):
            _, obs_dict = env.reset()
            agent.reset()
            ep_harm = 0.0

            for _step in range(steps_per_episode):
                with torch.no_grad():
                    obs_body = torch.tensor(obs_dict["body_state"], dtype=torch.float32)
                    obs_world = torch.tensor(obs_dict["world_state"], dtype=torch.float32)
                    latent = agent.sense(obs_body, obs_world)
                    ticks = agent.clock.advance()

                    eval_ticks += 1
                    action, _n_classes, used = _select_action(
                        agent, latent, ticks, ACTION_DIM, rng, NUM_CANDIDATES
                    )
                    if not used:
                        hippo_fallbacks_eval += 1

                    z_world_cur = latent.z_world.detach()
                    _, reward, done, _info, obs_dict = env.step(action)
                    harm_signal = float(min(0.0, reward))
                    ep_harm += abs(harm_signal)

                    residue_val = float(agent.residue_field.evaluate(z_world_cur).item())
                    if harm_signal < 0.0:
                        eval_residue_at_harm.append(residue_val)
                    else:
                        eval_residue_elsewhere.append(residue_val)

                    if done:
                        _, obs_dict = env.reset()
                        agent.reset()

            eval_harms.append(ep_harm)

        mean_harm = float(sum(eval_harms) / max(len(eval_harms), 1))
        mean_at_harm = float(sum(eval_residue_at_harm) / max(len(eval_residue_at_harm), 1))
        mean_elsewhere = float(sum(eval_residue_elsewhere) / max(len(eval_residue_elsewhere), 1))
        residue_contrast = max(0.0, mean_at_harm - mean_elsewhere)

        residue_stats = agent.residue_field.get_statistics()
        total_residue = float(residue_stats["total_residue"].item())
        num_harm_events_field = int(residue_stats["num_harm_events"].item())
        active_centers = int(residue_stats["active_centers"].item())

        hippo_fallback_rate_eval = hippo_fallbacks_eval / max(1, eval_ticks)
        hippo_fallback_rate_train = hippo_fallbacks_train / max(1, nonforced_ticks_train)

        print(
            f"  [eval] {condition} seed={seed} mean_harm={mean_harm:.4f}"
            f" residue_contrast={residue_contrast:.4f}"
            f" total_residue={total_residue:.3f}"
            f" harm_events_train={harm_events_train}"
            f" candidate_diversity_min={candidate_diversity_min:.1f}"
            f" hippo_fallback_rate_eval={hippo_fallback_rate_eval:.4f}",
            flush=True,
        )
        print(f"verdict: {'PASS' if num_harm_events_field > 0 else 'FAIL'}", flush=True)

        row = {
            "condition": condition,
            "seed": seed,
            "mean_harm": mean_harm,
            "residue_contrast": residue_contrast,
            "mean_residue_at_harm": mean_at_harm,
            "mean_residue_elsewhere": mean_elsewhere,
            "total_residue": total_residue,
            "num_harm_events_field": num_harm_events_field,
            "active_centers": active_centers,
            "n_eval_episodes": len(eval_harms),
            "accumulation_rate": params["accumulation_rate"],
            "kernel_bandwidth": params["kernel_bandwidth"],
            "harm_events_train": harm_events_train,
            "steps_train": steps_train,
            "harm_rate_train": harm_events_train / max(1, steps_train),
            "candidate_diversity_min": candidate_diversity_min,
            "hippo_fallback_rate_train": hippo_fallback_rate_train,
            "hippo_fallback_rate_eval": hippo_fallback_rate_eval,
            "warmup_episodes": warmup_episodes,
            "eval_episodes": eval_episodes,
            "steps_per_episode": steps_per_episode,
        }
        cell.stamp(row)
        zg_accum.observe(agent)  # AFTER both train and eval have stepped this cell's agent
    return row


# ---------------------------------------------------------------------------
# Criterion evaluation -- UNCHANGED from EXQ-149
# ---------------------------------------------------------------------------

def _evaluate_criteria(results_by_condition: Dict[str, List[Dict]]) -> Dict[str, bool]:
    fast_results = results_by_condition["FAST_TAU"]
    slow_results = results_by_condition["SLOW_TAU"]
    def_results = results_by_condition["DEFAULT_TAU"]

    c1_seeds = [
        fast_results[i]["mean_harm"] < slow_results[i]["mean_harm"] * THRESH_FAST_WIN
        for i in range(len(fast_results))
    ]
    c1 = all(c1_seeds)

    c2_seeds = [
        slow_results[i]["mean_harm"] < fast_results[i]["mean_harm"] * THRESH_SLOW_WIN
        for i in range(len(slow_results))
    ]
    c2 = all(c2_seeds)

    c3_seeds = []
    for i in range(len(def_results)):
        fast_h = fast_results[i]["mean_harm"]
        slow_h = slow_results[i]["mean_harm"]
        def_h = def_results[i]["mean_harm"]
        monotone = (fast_h <= def_h <= slow_h) or (slow_h <= def_h <= fast_h)
        c3_seeds.append(monotone)
    c3 = all(c3_seeds)

    c4_seeds = [r["residue_contrast"] > THRESH_CONTRAST for r in fast_results]
    c4 = all(c4_seeds)

    if c1:
        c5_seeds = [fast_results[i]["mean_harm"] < slow_results[i]["mean_harm"] for i in range(len(fast_results))]
        c5 = all(c5_seeds)
    elif c2:
        c5_seeds = [slow_results[i]["mean_harm"] < fast_results[i]["mean_harm"] for i in range(len(slow_results))]
        c5 = all(c5_seeds)
    else:
        c5_seeds = [
            (fast_results[i]["mean_harm"] < slow_results[i]["mean_harm"]) ==
            (fast_results[0]["mean_harm"] < slow_results[0]["mean_harm"])
            for i in range(len(fast_results))
        ]
        c5 = all(c5_seeds)

    return {
        "C1_fast_wins": c1,
        "C2_slow_wins": c2,
        "C3_default_ranked": c3,
        "C4_fast_contrast": c4,
        "C5_seed_consistent": c5,
    }


def _determine_outcome(criteria: Dict[str, bool]) -> str:
    if not criteria["C4_fast_contrast"]:
        return "FAIL"
    if criteria["C1_fast_wins"] and criteria["C5_seed_consistent"]:
        return "PASS"
    if criteria["C2_slow_wins"] and criteria["C5_seed_consistent"]:
        return "PASS"
    return "PARTIAL"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_experiment(dry_run: bool = False) -> Dict:
    t0 = __import__("time").perf_counter()
    print("=== V3-EXQ-149b: Q-004 tau_R Calibration at Larger Budget (selection-fixed) ===", flush=True)
    print(f"Conditions: {list(CONDITIONS.keys())}  Seeds: {SEEDS}", flush=True)
    print(f"WARMUP_EPISODES={WARMUP_EPISODES}  EVAL_EPISODES={EVAL_EPISODES}  "
          f"STEPS_PER_EPISODE={STEPS_PER_EPISODE}  NUM_CANDIDATES={NUM_CANDIDATES}", flush=True)

    zg_accum = ZGoalStreamAccumulator()
    results_by_condition: Dict[str, List[Dict]] = {k: [] for k in CONDITIONS}
    arm_results: List[Dict] = []

    for condition in CONDITIONS:
        print(f"\n--- Condition: {condition} ---", flush=True)
        for seed in SEEDS:
            row = _run_condition(
                seed=seed,
                condition=condition,
                warmup_episodes=WARMUP_EPISODES,
                eval_episodes=EVAL_EPISODES,
                steps_per_episode=STEPS_PER_EPISODE,
                lr=LR,
                zg_accum=zg_accum,
                dry_run=dry_run,
            )
            results_by_condition[condition].append(row)
            arm_results.append(row)

    print("\n=== Evaluating criteria ===", flush=True)
    criteria = _evaluate_criteria(results_by_condition)
    outcome = _determine_outcome(criteria)
    for k, v in criteria.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}", flush=True)
    print(f"Overall outcome: {outcome}", flush=True)

    def _mean_over_seeds(cond: str, key: str) -> float:
        vals = [r[key] for r in results_by_condition[cond]]
        return float(sum(vals) / max(len(vals), 1))

    summary_metrics = {}
    for cond in CONDITIONS:
        prefix = cond.lower()
        summary_metrics[f"{prefix}_mean_harm"] = _mean_over_seeds(cond, "mean_harm")
        summary_metrics[f"{prefix}_residue_contrast"] = _mean_over_seeds(cond, "residue_contrast")
        summary_metrics[f"{prefix}_total_residue"] = _mean_over_seeds(cond, "total_residue")
        summary_metrics[f"{prefix}_harm_events_train"] = _mean_over_seeds(cond, "harm_events_train")
        summary_metrics[f"{prefix}_candidate_diversity_min"] = _mean_over_seeds(cond, "candidate_diversity_min")
        summary_metrics[f"{prefix}_hippo_fallback_rate_eval"] = _mean_over_seeds(cond, "hippo_fallback_rate_eval")

    # Harm-EXPOSURE parity diagnostic across conditions (non-gating): confirms
    # nav_bias produced comparable training-time harm exposure regardless of
    # tau condition, isolating tau_R (not exposure) as the manipulated variable.
    exposure_by_cond = {
        cond: _mean_over_seeds(cond, "harm_events_train") for cond in CONDITIONS
    }
    exposure_vals = list(exposure_by_cond.values())
    exposure_parity_ratio = (
        min(exposure_vals) / max(exposure_vals) if max(exposure_vals) > 0 else 0.0
    )

    # R1/R2 readiness preconditions (informational -- EXPERIMENT_PURPOSE=evidence
    # does not require the hard diagnostic-adjudication gate, but this run's
    # entire point is verifying the selection channel is genuinely engaged).
    all_diversity_mins = [r["candidate_diversity_min"] for r in arm_results]
    all_fallback_rates_eval = [r["hippo_fallback_rate_eval"] for r in arm_results]
    r1_met = min(all_diversity_mins) >= FLOOR_CANDIDATE_ACTION_CLASSES if all_diversity_mins else False
    r2_met = max(all_fallback_rates_eval) < 0.5 if all_fallback_rates_eval else False

    preconditions = [
        {
            "name": "candidate_action_diversity",
            "description": "distinct TRUE first-action classes across a post-training positive-control candidate set, per cell",
            "control": "post-training positive-control probe (CONTROL_PROBE_STEPS ticks, pure hippocampal selection)",
            "measured": min(all_diversity_mins) if all_diversity_mins else 0.0,
            "threshold": FLOOR_CANDIDATE_ACTION_CLASSES,
            "direction": "lower",
            "met": r1_met,
        },
        {
            "name": "hippo_fallback_rate_eval_bounded",
            "description": "fraction of eval ticks where propose_trajectories returned no candidates (fell back to random), worst cell",
            "control": "measured on the same eval pass the DV is read from",
            "measured": max(all_fallback_rates_eval) if all_fallback_rates_eval else 1.0,
            "threshold": 0.5,
            "direction": "upper",
            "met": r2_met,
        },
    ]
    criteria_non_degenerate = {
        "C1_fast_wins": r1_met and r2_met,
        "C2_slow_wins": r1_met and r2_met,
        "C4_fast_contrast": r1_met and r2_met,
    }

    # Non-degeneracy self-report on the load-bearing cross-condition metric
    # (mean_harm) -- guards the EXACT defect this run fixes: if mean_harm is
    # STILL pinned bit-identical across conditions even with real
    # residue-informed selection, that is a genuine (and reportable) finding,
    # not an instrument artifact -- but check_degeneracy flags a zero-spread
    # read either way so it is not silently trusted as a measured null.
    degeneracy = check_degeneracy({
        "mean_harm_across_conditions": [r["mean_harm"] for r in arm_results],
    })

    if outcome == "PASS":
        guidance = "fast_tau_preferred" if criteria["C1_fast_wins"] else "slow_tau_preferred"
        evidence_direction = "supports"
    elif outcome == "PARTIAL":
        guidance = "tau_R_robust_in_range"
        evidence_direction = "mixed"
    else:
        guidance = "accumulation_pathological"
        evidence_direction = "mixed"

    run_id = f"v3_exq_149b_q004_tau_r_largebudget_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_v3"

    full_config = {
        "conditions": CONDITIONS,
        "seeds": SEEDS,
        "warmup_episodes": WARMUP_EPISODES,
        "eval_episodes": EVAL_EPISODES,
        "steps_per_episode": STEPS_PER_EPISODE,
        "num_candidates": NUM_CANDIDATES,
        "nav_bias_train_frac": NAV_BIAS_TRAIN_FRAC,
        "lr": LR,
        "env": {"size": 10, "num_hazards": 2, "num_resources": 3, "hazard_harm": 0.05,
                "env_drift_interval": 15, "env_drift_prob": 0.2, "use_proxy_fields": True},
        "dims": {"body_obs_dim": BODY_OBS_DIM, "world_obs_dim": WORLD_OBS_DIM,
                 "action_dim": ACTION_DIM, "world_dim": WORLD_DIM, "self_dim": SELF_DIM},
    }

    pack: Dict[str, Any] = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "supersedes": SUPERSEDES,
        "outcome": outcome,
        "evidence_direction": evidence_direction,
        "evidence_class": "discriminative_pair",
        "guidance": guidance,
        "criteria": criteria,
        "pre_registered_thresholds": {
            "THRESH_FAST_WIN": THRESH_FAST_WIN,
            "THRESH_SLOW_WIN": THRESH_SLOW_WIN,
            "THRESH_CONTRAST": THRESH_CONTRAST,
        },
        "summary_metrics": summary_metrics,
        "diagnostics": {
            "harm_exposure_by_condition_train": exposure_by_cond,
            "harm_exposure_parity_ratio": exposure_parity_ratio,
        },
        "interpretation": {
            "label": guidance,
            "preconditions": preconditions,
            "criteria_non_degenerate": criteria_non_degenerate,
        },
        "condition_params": CONDITIONS,
        "seeds": SEEDS,
        "scenario": (
            "Three-condition tau_R calibration replication at a substantially larger"
            " training budget, with action selection wired through"
            " agent.hippocampal.propose_trajectories + agent.select_action (E3-informed,"
            " residue-terrain-scored) instead of EXQ-149's uniform-random actions."
            f" 2 seeds x 3 conditions = 6 cells. {WARMUP_EPISODES} warmup +"
            f" {EVAL_EPISODES} eval eps x {STEPS_PER_EPISODE} steps."
            " CausalGridWorldV2 size=10 2 hazards 3 resources."
        ),
        "interpretation_note": (
            f"PASS (fast) => calibrate tau_R to E2 forward-model timescale (~30 steps)."
            f" PASS (slow) => calibrate tau_R to E1 LSTM integration timescale (~200 steps)."
            f" PARTIAL => tau_R is not a critical free parameter in [0.02, 0.5] range,"
            f" MEASURED under real residue-informed selection (not an instrument artifact)."
            f" FAIL => pathological accumulation (contrast < {THRESH_CONTRAST})."
        ),
        "arm_results": arm_results,
        "per_condition_results": {cond: results_by_condition[cond] for cond in CONDITIONS},
        "dry_run": dry_run,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    pack.update(degeneracy)

    out_path = write_flat_manifest(
        pack,
        None,
        dry_run=dry_run,
        config=full_config,
        seeds=SEEDS,
        script_path=Path(__file__),
        started_at=t0,
        z_goal_stream_stats=zg_accum.stats(),
    )
    if not dry_run:
        print(f"\nResult pack written to: {out_path}", flush=True)
    else:
        print("\n[dry_run] Result pack written to scratch (relocated by emit_outcome).", flush=True)

    pack["_manifest_path"] = str(out_path)
    return pack


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    result = run_experiment(dry_run=dry_run)
    print(f"Done. Outcome: {result['outcome']}", flush=True)

    _outcome_raw = str(result.get("outcome", "FAIL")).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=result.get("_manifest_path"),
        dry_run=dry_run,
    )
