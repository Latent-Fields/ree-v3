#!/opt/local/bin/python3
"""
V3-EXQ-801 -- ARC-018 leg B: rollout-depth ablation at VERIFIED E2 fidelity.

Claim: ARC-018 -- "Hippocampus generates explicit rollouts and post-commitment
viability mapping."  Proposal: EXP-0396 (manual_proposals.v1.json, digested
2026-07-21).

WHY THE PASS/FAIL SPLIT IS NOT NOISE
------------------------------------
ARC-018 is CONJUNCTIVE and its evidence separates CLEANLY by leg:

  LEG A "post-commitment viability mapping" -- EXQ-042 PASS (terrain_prior
  learns E3 preferences, hippo_quality_gap 0.393, terrain_loss -> 0), EXQ-053
  PASS (terrain-guided navigation 50x less harm than random, 5/5 criteria),
  EXQ-120 PASS (discriminative pair). Three consistent PASSes. NOT under test.

  LEG B "explicit rollouts" -- EXQ-172 FAIL, EXQ-196 FAIL. UNDER TEST HERE.

Every PASS is leg A; every FAIL is leg B. Carrying this as one conjunction at
exp_conf 0.547 averages a well-supported leg with an unsupported one and hides
both facts.

LEG B IS UNTESTED, NOT REFUTED -- BOTH FAILS ARE CONFOUNDED, DIFFERENTLY
-----------------------------------------------------------------------
  EXQ-172 ran at e2_world_r2 = 0.203. Its own note concludes "rollout planning
  counterproductive at current fidelity ... ARC-018 may be E2-quality-gated".
  A negative advantage from a POOR FORWARD MODEL indicts E2, not the rollout
  architecture. -> trapped here by P1 (E2 fidelity gate, measured IN-RUN).

  EXQ-196 returned harm_advantage_mean EXACTLY 0.0 on all 3 seeds at GOOD
  fidelity (e2_world_r2 0.766, residue populated). An exact-zero-everywhere
  reading is the signature of a DV that was never wired to the manipulation,
  which is why governance classified it non_contributory and deferred the
  verdict. -> trapped here by P3 (unwired-DV trap).

DESIGN -- rollout-depth ablation + matched-noise control
--------------------------------------------------------
Rollout depth is `HippocampalConfig.horizon`, the E2 rollout length inside the
hippocampal CEM (ree_core/hippocampal/module.py -> e2.rollout_with_world).

  A0_DEPTH0   horizon=1  -- no multi-step lookahead (immediate effect only).
                            The pipeline is otherwise IDENTICAL, so this is a
                            depth ablation, not a machinery ablation.
  A1_SHALLOW  horizon=3
  A2_FULL     horizon=10 -- substrate default.
  A3_NOISE    horizon=10, but the CEM's trajectory SCORING is replaced with
                            matched-variance random noise (install_noise_scoring),
                            so the rollout still runs to full depth and the CEM
                            still refits -- on RANDOM elites.

A3_NOISE holds DEPTH and compute fixed while removing the rollout-derived
INFORMATION, separating "rollout CONTENT confers the advantage" from "longer
rollouts merely produce a more diverse candidate set". Its proposals are
rollout-SHAPED but not rollout-INFORMED; E3 selects over them normally.

  Primary DV: harm_advantage(X) = harm_rate(A0_DEPTH0) - harm_rate(X).
  Signal of interest: MONOTONICITY across depth.

DV-SYMMETRY DECLARATION (mandatory; one line per arm)
-----------------------------------------------------
DV = harm_advantage_mean, a function of eval harm_rate, itself a function of the
EXECUTED ACTION STREAM, which E3 produces by ranking the hippocampal candidate
set. Symmetry group of that DV: (i) permutation of candidate index order, (ii) a
uniform additive constant across candidate scores, (iii) any monotone rescaling
of them -- none can move a ranking. Also, critically, (iv) ANY change to the
candidate set at all is invariant if every candidate maps to the same executed
action; P4 measures exactly that.
  A0/A1/A2  the manipulation changes the ROLLOUT HORIZON, so each candidate's
            rolled-out world trajectory changes by a DIFFERENT amount and the CEM
            refits over a different-length action-object sequence. Neither a
            broadcast constant nor a monotone map: NOT invariant.
  A3        replaces CEM scoring with an independent random draw, so the elite
            set is random rather than terrain-selected. Not a monotone map of the
            true scores (it discards them): NOT invariant. Its expected advantage
            is ~0 BY CONSTRUCTION -- it is a control, and its null is informative.
P2 asserts the cross-candidate score RANGE (the V3-EXQ-643 same-statistic rule,
not a magnitude proxy) and P4 asserts EXECUTED-ACTION diversity, which is the
statistic the DV literally routes on.

NON-DEGENERACY PRECONDITIONS (breach -> substrate_not_ready_requeue, NOT a verdict)
----------------------------------------------------------------------------------
  P1 e2_world_r2_adequate     ALL arms: e2_world_r2 >= 0.60 measured IN-RUN
                              against realised next-z_world. Below this the
                              EXQ-172 confound recurs and any result indicts E2
                              rather than the rollout architecture.
  P2 rollout_score_range      A0/A1/A2: cross-candidate rollout score RANGE > 0
                              -- the rollout must actually DISCRIMINATE between
                              candidates. Scoped OUT of A3_NOISE, whose scores
                              are synthetic by construction and so are not
                              evidence about rollout discrimination either way.
  P4 executed_action_diversity  A0/A1/A2: >= 2 distinct EXECUTED action classes.
                              If every tick executes the same class the DV is
                              invariant under the depth manipulation and no
                              result is a measurement. MEASURED: selecting via
                              action_object_decoder argmax collapses to 1 class
                              at every tick (so A2_FULL and A3_NOISE came out
                              BIT-IDENTICAL), while the E3 path spans 2 -- which
                              is why selection routes through E3 here. This is
                              the most plausible reading of the EXQ-196 signature.
  P3 harm_advantage_wired     A2_FULL: number of seeds whose harm_advantage is
                              NOT bit-exactly 0.0 must be >= 1. This is the
                              EXQ-196 trap. NOTE the threshold is on the COUNT
                              OF SEEDS WITH A NONZERO VALUE, never on the
                              MAGNITUDE: a genuine null produces small NONZERO
                              advantages and must be allowed to score as a FAIL,
                              whereas bit-exact 0.0 on every seed is an unwired
                              DV and must self-route substrate_not_ready. A
                              magnitude floor here would silently convert every
                              real null into "substrate not ready".

Per-arm gates are aggregated with experiments/_lib/precondition_gate.py so one
arm's red gate can NEVER vacate another arm's valid result (V3-EXQ-785).

PRE-REGISTERED THRESHOLDS (constants below; never inferred post-hoc)
--------------------------------------------------------------------
  C1 (LOAD-BEARING) monotonic: adv(A1) > 0 AND adv(A2) > adv(A1)
  C2 effect >= 0.80 SD of the cross-seed A2-vs-A0 delta
  C3 direction consistent on >= 2 of 3 seeds
  C4 A2_FULL strictly above A3_NOISE

SLEEP: not used (no sleep flags set) -- no SLEEP DRIVER line required.
"""

import argparse
import math
import random
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import HippocampalConfig, REEConfig

from experiment_protocol import emit_outcome
from experiments.pack_writer import write_flat_manifest
from experiments._lib.arm_fingerprint import arm_cell
from experiments._lib.precondition_gate import (
    PreconditionSpec,
    aggregate_arm_gates,
    arm_criteria_non_degenerate,
    assert_no_structurally_unsatisfiable_gate,
    evaluate_arm_gate,
)

EXPERIMENT_TYPE = "v3_exq_801_arc018_rollout_depth_ablation"
CLAIM_IDS = ["ARC-018"]
EXPERIMENT_PURPOSE = "evidence"
BACKLOG_ID = "EXP-0396"

ARM_DEPTH0 = "A0_DEPTH0"
ARM_SHALLOW = "A1_SHALLOW"
ARM_FULL = "A2_FULL"
ARM_NOISE = "A3_NOISE"
ARMS = [ARM_DEPTH0, ARM_SHALLOW, ARM_FULL, ARM_NOISE]

HORIZON_BY_ARM = {
    ARM_DEPTH0: 1,
    ARM_SHALLOW: 3,
    ARM_FULL: 10,
    ARM_NOISE: 10,
}

# --- pre-registered thresholds -------------------------------------------
THRESH_C2_EFFECT_SD = 0.80
THRESH_C3_MIN_SEEDS = 2

# --- non-degeneracy floors ------------------------------------------------
FLOOR_E2_WORLD_R2 = 0.60          # P1: the EXQ-172 confound floor
FLOOR_ROLLOUT_SCORE_RANGE = 1e-6  # P2: rollout must discriminate candidates
FLOOR_SEEDS_NONZERO_ADV = 0.5     # P3: >= 1 seed with a bit-nonzero advantage

# CEM candidate budget. MUST keep num_elite = int(NUM_CANDIDATES *
# elite_fraction) >= 2. HippocampalModule.propose_trajectories refits with
# `elite_ao_tensor.std(dim=0)`, and the std of a SINGLE elite is NaN, which
# poisons ao_std and NaNs every subsequent candidate rollout -- measured
# directly while smoke-testing the sibling V3-EXQ-800 at NUM_CANDIDATES=8
# (num_elite 1): only 1 of 8 candidates scored finite at 10/40/120 warmup
# episodes alike, identically zeroing the cross-candidate range. 32 is the
# substrate default and gives num_elite = 6.
NUM_CANDIDATES = 32
MIN_REQUIRED_ELITES = 2
R2_PROBE_STEPS = 400  # in-run E2 fidelity measurement


# =========================================================================
# precondition specs
# =========================================================================
PRECONDITION_SPECS = [
    PreconditionSpec(
        name="e2_world_r2_adequate",
        description=(
            "in-run E2 world-forward R^2 against realised next z_world. Below the "
            "floor the EXQ-172 confound recurs: a negative advantage from a poor "
            "forward model indicts E2, not the rollout architecture"
        ),
        control="one-step world_forward predictions vs realised z_world at eval",
        threshold=FLOOR_E2_WORLD_R2,
        direction="lower",
        applies_note="",
    ),
    PreconditionSpec(
        name="rollout_score_range",
        description=(
            "cross-candidate RANGE of rollout-derived scores -- the SAME statistic "
            "the argmin selection routes on. The rollout must actually discriminate "
            "between candidates or the depth manipulation has no path to the DV"
        ),
        control="32 CEM candidates rolled out at the arm's own horizon",
        threshold=FLOOR_ROLLOUT_SCORE_RANGE,
        direction="lower",
        applies_to=lambda ctx: not ctx["synthetic_scores"],
        applies_note=(
            "A3_NOISE scores are synthetic matched noise by construction, so their "
            "spread is evidence about the CONTROL, not about whether the rollout "
            "discriminates -- asserting it there would be structurally meaningless"
        ),
    ),
    PreconditionSpec(
        name="executed_action_diversity",
        description=(
            "distinct EXECUTED action classes -- the statistic the DV literally "
            "routes on. 1 class means the DV is invariant under the depth "
            "manipulation and nothing was measured (the EXQ-196 shape)"
        ),
        control="E3-selected actions over the eval phase",
        threshold=1.5,  # i.e. >= 2 distinct classes
        direction="lower",
        applies_to=lambda ctx: not ctx["synthetic_scores"],
        applies_note=(
            "A3_NOISE selects from randomly-scored elites, so its action "
            "diversity reflects the random control rather than whether the "
            "rollout manipulation can reach behaviour"
        ),
    ),
    PreconditionSpec(
        name="harm_advantage_wired",
        description=(
            "number of seeds whose harm_advantage is NOT bit-exactly 0.0. The "
            "EXQ-196 signature (exactly 0.0 on every seed at good fidelity) means "
            "the rollout output never reached the DV. Thresholded on the COUNT of "
            "nonzero seeds, NEVER on magnitude -- a genuine null has small nonzero "
            "advantages and must remain scoreable as a FAIL"
        ),
        control="per-seed harm_advantage of the full-depth arm vs the depth-0 arm",
        threshold=FLOOR_SEEDS_NONZERO_ADV,
        direction="lower",
        applies_to=lambda ctx: ctx["id"] == ARM_FULL,
        applies_note=(
            "the DV is defined as an advantage OVER A0_DEPTH0, so A0's own "
            "advantage is identically 0 by construction and the trap is only "
            "meaningful on the full-depth arm that carries the claim"
        ),
    ),
]


def arm_contexts() -> List[Dict[str, Any]]:
    return [
        {"id": a, "horizon": HORIZON_BY_ARM[a],
         "synthetic_scores": (a == ARM_NOISE)}
        for a in ARMS
    ]


# =========================================================================
# substrate helpers
# =========================================================================
def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


def _candidate_scores(agent, candidates) -> List[Tuple[int, float]]:
    """FINITE rollout-derived scores as (candidate_index, score), lower = better.

    Non-finite scores are DROPPED, not ranked: `torch.argmin` over a tensor
    containing NaN does not reliably return the true minimum, so ranking them
    would silently degrade the argmin into an arbitrary pick and sever the
    rollout from the DV -- precisely the EXQ-196 unwired-DV failure this
    experiment exists to avoid reproducing.
    """
    out: List[Tuple[int, float]] = []
    for i, traj in enumerate(candidates):
        world_seq = traj.get_world_state_sequence()
        if world_seq is None:
            continue
        val = float(agent.residue_field.evaluate_trajectory(world_seq).sum().item())
        if math.isfinite(val):
            out.append((i, val))
    return out


def _first_action_of(agent, traj, n_actions: int) -> Optional[int]:
    ao_seq = traj.get_action_object_sequence()
    if ao_seq is None or ao_seq.shape[1] == 0:
        return None
    logits = agent.hippocampal.action_object_decoder(ao_seq[:, 0, :])
    return int(torch.argmax(logits, dim=-1).item())


def install_random_proposer(agent, horizon: int) -> None:
    """A3_NOISE: propose candidates from RANDOM action sequences at matched depth.

    The control must hold ROLLOUT DEPTH, candidate count and downstream selection
    fixed while removing the terrain-guided PLANNING that produces the proposals.
    This replaces `propose_trajectories` with rollouts of random action sequences
    at the arm's own horizon, so E3 still ranks 32 fully-rolled-out candidates by
    its normal J(zeta) -- it simply has nothing planned to choose among.

    It answers: does the hippocampal rollout-and-refine PROCESS add anything over
    random proposals evaluated at the same depth? Without this control a monotone
    depth effect could be an artifact of longer rollouts merely producing more
    diverse candidates.

    REJECTED ALTERNATIVE, recorded because it looked right and measured inert:
    patching `_score_trajectory` to return random scalars (so CEM elites are
    chosen at random). Measured in smoke: A2_FULL and A3_NOISE came out equal to
    4 significant figures (harm_rate 0.5115 both). The reason is structural --
    every candidate is `ao_mean + noise`, so the mean of a RANDOM 6 of 32 is
    approximately the mean of the BEST 6, and the refit barely moves. That is an
    inert arm knob (the V3-EXQ-689d D2 defect): an arm DECLARED distinct that
    runs bit-nearly-identically, which silently degrades the conjunctive
    acceptance criteria. Replacing the proposer, not the scorer, is what makes
    the control actually control something.
    """
    e2 = agent.e2
    self_dim = e2.config.self_dim

    def _random_proposals(z_world, z_self=None, num_candidates=None, **_kwargs):
        n = int(num_candidates or NUM_CANDIDATES)
        batch = z_world.shape[0]
        device = z_world.device
        if z_self is None:
            z_self = torch.zeros(batch, self_dim, device=device)
        out = []
        for _ in range(n):
            actions = e2.generate_random_actions(batch, horizon, device)
            out.append(
                e2.rollout_with_world(
                    z_self, z_world, actions, compute_action_objects=True
                )
            )
        return out

    agent.hippocampal.propose_trajectories = _random_proposals  # type: ignore


def _select_action(
    agent, latent, ticks, n_actions: int, rng: random.Random,
) -> Tuple[torch.Tensor, Optional[float], bool]:
    """Select via the CANONICAL E3 path: e3 ranks the hippocampal candidates.

    WHY NOT an argmin over `action_object_decoder(candidate)`. MEASURED on this
    substrate (40 warmup episodes, 32 candidates, seed 42): EVERY candidate
    decodes to the SAME first-action class at EVERY tick (distinct classes =
    1.00 of 5), under the default config AND under both
    `use_support_preserving_cem` and `use_action_class_scaffold_candidates` --
    the decoder is not trained by the E1/E2 objectives here and is a
    near-constant function of its input.

    Under that rule the executed action is INVARIANT under ANY manipulation of
    the candidate score vector, so the depth ablation would be an arithmetically
    forced no-op. That is the most plausible reading of EXQ-196's
    harm_advantage_mean = EXACTLY 0.0 on all three seeds at good E2 fidelity,
    and it reproduced here directly: on the decoder path A2_FULL and A3_NOISE
    came out BIT-IDENTICAL on every recorded field.

    `agent.select_action(candidates, ticks)` routes through E3's J(zeta) and
    returns the action DIRECTLY, never touching the collapsed decoder. Measured:
    executed actions span 2 classes and respond to the terrain (permuting the
    residue field moved 63 of 150 executed ticks). So the manipulation reaches
    the DV on this path and only this path.

    Returns (action_tensor, cross_candidate_score_range, used_e3).
    """
    with torch.no_grad():
        candidates = agent.hippocampal.propose_trajectories(
            latent.z_world.detach(),
            z_self=latent.z_self.detach(),
            num_candidates=NUM_CANDIDATES,
        )
        if not candidates:
            return (
                _action_to_onehot(rng.randint(0, n_actions - 1), n_actions,
                                  agent.device),
                None,
                False,
            )
        scored = _candidate_scores(agent, candidates)
        vals = [v for _i, v in scored]
        score_range = float(max(vals) - min(vals)) if len(vals) >= 2 else None
        action = agent.select_action(candidates, ticks)
        return action, score_range, True


def _r2(preds: List[torch.Tensor], actuals: List[torch.Tensor]) -> float:
    """Coefficient of determination of E2 world_forward vs realised z_world."""
    if len(preds) < 2:
        return float("nan")
    p = torch.cat(preds, dim=0)
    a = torch.cat(actuals, dim=0)
    ss_res = float(((a - p) ** 2).sum().item())
    ss_tot = float(((a - a.mean(dim=0, keepdim=True)) ** 2).sum().item())
    if ss_tot <= 1e-12:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


# =========================================================================
# one (seed, arm) cell
# =========================================================================
def run_cell(
    arm_ctx: Dict[str, Any],
    seed: int,
    full_config: Dict[str, Any],
    warmup_episodes: int,
    eval_episodes: int,
    steps_per_episode: int,
    dry_run: bool,
) -> Dict[str, Any]:
    arm_id = arm_ctx["id"]
    horizon = int(arm_ctx["horizon"])
    print(f"Seed {seed} Condition {arm_id}", flush=True)

    # A0_DEPTH0 is the lineage baseline arm and is minted reuse-ELIGIBLE with the
    # driver script EXCLUDED from the hash, so a later different-driver consumer
    # can match it (mint-as-you-go; terminality is unknowable).
    with arm_cell(
        seed,
        config_slice={**full_config, "horizon": horizon},
        script_path=Path(__file__),
        include_driver_script_in_hash=False,
    ) as cell:
        rng = random.Random(seed)

        env = CausalGridWorldV2(
            seed=seed,
            size=6,
            num_hazards=4,
            num_resources=3,
            hazard_harm=full_config["hazard_harm"],
            env_drift_interval=5,
            env_drift_prob=0.1,
            proximity_harm_scale=full_config["proximity_harm_scale"],
            proximity_benefit_scale=full_config["proximity_harm_scale"] * 0.6,
            proximity_approach_threshold=0.15,
            hazard_field_decay=0.5,
            use_proxy_fields=True,
        )
        n_actions = env.action_dim

        config = REEConfig.from_dims(
            body_obs_dim=env.body_obs_dim,
            world_obs_dim=env.world_obs_dim,
            action_dim=env.action_dim,
            self_dim=full_config["self_dim"],
            world_dim=full_config["world_dim"],
            alpha_world=full_config["alpha_world"],
            alpha_self=full_config["alpha_self"],
            reafference_action_dim=0,
        )
        config.latent.unified_latent_mode = False
        # THE MANIPULATION: rollout depth inside the hippocampal CEM.
        config.hippocampal.horizon = horizon
        agent = REEAgent(config)
        assert int(agent.hippocampal.config.horizon) == horizon, (
            f"horizon manipulation did not reach the module for {arm_id}"
        )
        if arm_ctx["synthetic_scores"]:
            install_random_proposer(agent, horizon)
        optimizer = optim.Adam(list(agent.parameters()), lr=full_config["lr"])

        n_warmup = min(3, warmup_episodes) if dry_run else warmup_episodes
        n_eval = min(2, eval_episodes) if dry_run else eval_episodes
        n_steps = min(20, steps_per_episode) if dry_run else steps_per_episode

        # ---------------- TRAIN ----------------------------------------------
        agent.train()
        harm_train = 0
        steps_train = 0
        fallbacks = 0

        for ep in range(n_warmup):
            _, obs_dict = env.reset()
            agent.reset()
            for _ in range(n_steps):
                latent = agent.sense(obs_dict["body_state"], obs_dict["world_state"])
                ticks = agent.clock.advance()
                z_world_curr = latent.z_world.detach()

                action, _span, used = _select_action(
                    agent, latent, ticks, n_actions, rng
                )
                if not used:
                    fallbacks += 1

                _, harm_signal, done, _info, obs_dict = env.step(action)
                steps_train += 1

                if float(harm_signal) < 0:
                    harm_train += 1
                    agent.residue_field.accumulate(
                        z_world_curr, harm_magnitude=abs(float(harm_signal))
                    )

                loss = agent.compute_prediction_loss() + agent.compute_e2_loss()
                if loss.requires_grad:
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                    optimizer.step()
                if done:
                    break

            if (ep + 1) % 50 == 0 or ep == n_warmup - 1:
                print(
                    f"  [train] seed={seed} arm={arm_id} ep {ep+1}/{n_warmup}"
                    f" harm_events={harm_train}"
                    f" harm_rate={harm_train / max(1, steps_train):.4f}",
                    flush=True,
                )

        # ---------------- EVAL + in-run E2 fidelity --------------------------
        agent.eval()
        harm_eval = 0
        steps_eval = 0
        visited_cells = set()
        score_ranges: List[float] = []
        executed_actions: List[int] = []
        # E3 cadence bookkeeping. heartbeat.e3_steps_per_tick defaults to 10, so
        # agent.select_action returns the HELD action on ~9 of 10 ticks. TRIAGE of
        # the hold-weighted-readout hazard for THIS script: the two quantities that
        # route a verdict are (a) harm_rate, a behavioural rate over EVERY executed
        # env step -- held steps are genuinely executed, so weighting by hold
        # duration is the correct behavioural measure, not a contamination; and
        # (b) executed_action_classes, a SET CARDINALITY, which is invariant to
        # repetition by construction. Neither is a distribution-shape statistic, so
        # neither is reweighted by the cadence. The counters below are emitted so a
        # later reader can audit the true denominator rather than trust len().
        n_fresh_select = 0
        n_latched = 0
        wf_preds: List[torch.Tensor] = []
        wf_actuals: List[torch.Tensor] = []
        pending_pred: Optional[torch.Tensor] = None
        r2_budget = min(40, R2_PROBE_STEPS) if dry_run else R2_PROBE_STEPS

        for _ in range(n_eval):
            _, obs_dict = env.reset()
            agent.reset()
            pending_pred = None
            for _ in range(n_steps):
                with torch.no_grad():
                    latent = agent.sense(
                        obs_dict["body_state"], obs_dict["world_state"]
                    )
                    ticks = agent.clock.advance()
                z_world_curr = latent.z_world.detach()

                # close out the previous step's one-step world_forward prediction
                if pending_pred is not None and len(wf_preds) < r2_budget:
                    wf_preds.append(pending_pred)
                    wf_actuals.append(z_world_curr)
                    pending_pred = None

                with torch.no_grad():
                    action, span, used = _select_action(
                        agent, latent, ticks, n_actions, rng
                    )
                if span is not None:
                    score_ranges.append(span)
                if not used:
                    fallbacks += 1
                if bool(ticks.get("e3_tick", True)):
                    n_fresh_select += 1
                else:
                    n_latched += 1
                executed_actions.append(int(torch.argmax(action, dim=-1).item()))

                if len(wf_preds) < r2_budget:
                    with torch.no_grad():
                        pending_pred = agent.e2.world_forward(z_world_curr, action)

                _, harm_signal, done, _info, obs_dict = env.step(action)
                steps_eval += 1
                if float(harm_signal) < 0:
                    harm_eval += 1
                visited_cells.add((int(env.agent_x), int(env.agent_y)))
                if done:
                    break

        harm_rate_eval = harm_eval / max(1, steps_eval)
        e2_world_r2 = _r2(wf_preds, wf_actuals)
        finite_ranges = [r for r in score_ranges if math.isfinite(r)]

        row: Dict[str, Any] = {
            "arm_id": arm_id,
            "seed": int(seed),
            "rollout_horizon": horizon,
            "synthetic_scores": bool(arm_ctx["synthetic_scores"]),
            "harm_rate_eval": float(harm_rate_eval),
            "harm_events_eval": int(harm_eval),
            "total_steps_eval": int(steps_eval),
            "harm_rate_train": float(harm_train / max(1, steps_train)),
            "harm_events_train": int(harm_train),
            "total_steps_train": int(steps_train),
            "path_efficiency": float(len(visited_cells) / max(1, steps_eval)),
            "unique_cells_visited": int(len(visited_cells)),
            # readiness measurements
            "e2_world_r2": float(e2_world_r2),
            "e2_r2_n_samples": int(len(wf_preds)),
            # worst cell, not the mean: a single non-discriminating tick must not
            # hide behind an in-band average (recomputability rule)
            "rollout_score_range_min": (
                float(min(finite_ranges)) if finite_ranges else 0.0
            ),
            "rollout_score_range_mean": (
                float(statistics.fmean(finite_ranges)) if finite_ranges else 0.0
            ),
            # instrument health: steps where the argmin could not be formed and
            # selection fell back to random
            "rollout_fallback_steps": int(fallbacks),
            # the statistic the DV routes on: 1 class means no manipulation of
            # the candidate set can reach behaviour (see _select_action)
            "executed_action_classes": float(len(set(executed_actions))),
            "n_fresh_select": int(n_fresh_select),
            "n_latched_ticks": int(n_latched),
            "fresh_select_yield": float(
                n_fresh_select / max(1, n_fresh_select + n_latched)
            ),
            "residue_total": float(agent.residue_field.total_residue.item()),
            "residue_coverage": agent.residue_field.get_coverage_telemetry(),
        }
        cell.stamp(row)

    print(
        f"  [eval] seed={seed} arm={arm_id} h={horizon}"
        f" harm_rate={row['harm_rate_eval']:.4f}"
        f" e2_r2={row['e2_world_r2']:.4f}"
        f" score_range_min={row['rollout_score_range_min']:.3e}",
        flush=True,
    )
    print(f"verdict: {'PASS' if row['harm_events_eval'] > 0 else 'FAIL'}", flush=True)
    return row


# =========================================================================
# analysis
# =========================================================================
def _mean(vals: List[float]) -> float:
    return float(statistics.fmean(vals)) if vals else float("nan")


def _by_arm(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {a: [] for a in ARMS}
    for r in rows:
        out[r["arm_id"]].append(r)
    return out


def per_seed_advantage(rows: List[Dict[str, Any]], arm: str,
                       seeds: List[int]) -> List[float]:
    """harm_advantage(arm) per seed = harm_rate(A0_DEPTH0) - harm_rate(arm)."""
    by = _by_arm(rows)
    out: List[float] = []
    for s in seeds:
        base = next((r for r in by[ARM_DEPTH0] if r["seed"] == s), None)
        cur = next((r for r in by[arm] if r["seed"] == s), None)
        if base and cur:
            out.append(base["harm_rate_eval"] - cur["harm_rate_eval"])
    return out


def analyse(rows: List[Dict[str, Any]], seeds: List[int]) -> Dict[str, Any]:
    by = _by_arm(rows)
    harm = {a: _mean([r["harm_rate_eval"] for r in by[a]]) for a in ARMS}
    adv = {a: float(harm[ARM_DEPTH0] - harm[a]) for a in ARMS}

    adv_full_seeds = per_seed_advantage(rows, ARM_FULL, seeds)
    # EXQ-196 trap measurement: bit-exact zeros, NOT a magnitude test.
    n_seeds_nonzero_adv = sum(1 for v in adv_full_seeds if v != 0.0)

    delta_mean = _mean(adv_full_seeds)
    delta_sd = (
        float(statistics.pstdev(adv_full_seeds)) if len(adv_full_seeds) > 1 else 0.0
    )
    effect_sd = (
        float(delta_mean / delta_sd) if delta_sd > 1e-12
        else (float("inf") if delta_mean > 0 else 0.0)
    )
    seeds_consistent = sum(1 for v in adv_full_seeds if v > 0)

    c1 = bool(adv[ARM_SHALLOW] > 0.0 and adv[ARM_FULL] > adv[ARM_SHALLOW])
    c2 = bool(effect_sd >= THRESH_C2_EFFECT_SD)
    c3 = bool(seeds_consistent >= THRESH_C3_MIN_SEEDS)
    c4 = bool(adv[ARM_FULL] > adv[ARM_NOISE])

    return {
        "harm_rate_by_arm": harm,
        "harm_advantage_by_arm": adv,
        "harm_advantage_mean": float(adv[ARM_FULL]),
        "per_seed_advantage_full": adv_full_seeds,
        "n_seeds_nonzero_advantage": int(n_seeds_nonzero_adv),
        "monotonic_depth_response": c1,
        "delta_mean": float(delta_mean),
        "delta_sd": float(delta_sd),
        "effect_sd": float(effect_sd),
        "seeds_consistent": int(seeds_consistent),
        "e2_world_r2_by_arm": {
            a: _mean([r["e2_world_r2"] for r in by[a]]) for a in ARMS
        },
        "c1_monotonic_pass": c1,
        "c2_effect_sd_pass": c2,
        "c3_seed_consistency_pass": c3,
        "c4_above_noise_pass": c4,
        "all_pass": bool(c1 and c2 and c3 and c4),
    }


def build_gate(rows: List[Dict[str, Any]], seeds: List[int]) -> Dict[str, Any]:
    by = _by_arm(rows)
    adv_full_seeds = per_seed_advantage(rows, ARM_FULL, seeds)
    n_nonzero = float(sum(1 for v in adv_full_seeds if v != 0.0))

    gates = []
    for ctx in arm_contexts():
        arm_rows = by[ctx["id"]]
        measured: Dict[str, float] = {
            # worst cell across seeds, matching the quantifier the gate needs
            "e2_world_r2_adequate": min(
                (r["e2_world_r2"] for r in arm_rows
                 if math.isfinite(r["e2_world_r2"])),
                default=0.0,
            ),
        }
        if not ctx["synthetic_scores"]:
            measured["rollout_score_range"] = min(
                r["rollout_score_range_min"] for r in arm_rows
            )
            measured["executed_action_diversity"] = min(
                r["executed_action_classes"] for r in arm_rows
            )
        if ctx["id"] == ARM_FULL:
            measured["harm_advantage_wired"] = n_nonzero
        gates.append(
            evaluate_arm_gate(ctx["id"], ctx, PRECONDITION_SPECS, measured=measured)
        )
    return aggregate_arm_gates(gates)


# =========================================================================
# main
# =========================================================================
def run(
    seeds: Tuple[int, ...] = (42, 123, 456),
    warmup_episodes: int = 300,
    eval_episodes: int = 60,
    steps_per_episode: int = 200,
    dry_run: bool = False,
) -> Dict[str, Any]:
    t0 = time.perf_counter()

    full_config: Dict[str, Any] = {
        "env": "CausalGridWorldV2",
        "size": 6,
        "num_hazards": 4,
        "num_resources": 3,
        "hazard_harm": 0.02,
        "proximity_harm_scale": 0.05,
        "use_proxy_fields": True,
        "self_dim": 32,
        "world_dim": 32,
        "alpha_world": 0.9,
        "alpha_self": 0.3,
        "lr": 1e-3,
        "num_candidates": NUM_CANDIDATES,
        "warmup_episodes": warmup_episodes,
        "eval_episodes": eval_episodes,
        "steps_per_episode": steps_per_episode,
        "unified_latent_mode": False,
        "reafference_action_dim": 0,
        "arms": ARMS,
        "horizon_by_arm": HORIZON_BY_ARM,
        "selection_rule": "argmin_over_candidate_rollout_scores",
        "thresholds": {
            "C2_effect_sd": THRESH_C2_EFFECT_SD,
            "C3_min_seeds": THRESH_C3_MIN_SEEDS,
            "P1_e2_world_r2": FLOOR_E2_WORLD_R2,
            "P2_rollout_score_range": FLOOR_ROLLOUT_SCORE_RANGE,
            "P3_seeds_nonzero_advantage": FLOOR_SEEDS_NONZERO_ADV,
        },
    }

    assert_no_structurally_unsatisfiable_gate(PRECONDITION_SPECS, arm_contexts())

    _elite_fraction = float(HippocampalConfig().elite_fraction)
    _num_elite = max(1, int(NUM_CANDIDATES * _elite_fraction))
    if _num_elite < MIN_REQUIRED_ELITES:
        raise ValueError(
            f"CEM refit would use num_elite={_num_elite} from "
            f"num_candidates={NUM_CANDIDATES} x elite_fraction={_elite_fraction}. "
            f"std() over a single elite is NaN, poisons ao_std and NaNs every "
            f"candidate rollout, identically zeroing the cross-candidate score "
            f"range -- the depth manipulation would have no path to the DV and the "
            f"run would reproduce the EXQ-196 unwired-DV signature. Raise "
            f"NUM_CANDIDATES so num_elite >= {MIN_REQUIRED_ELITES}."
        )
    print(
        f"[V3-EXQ-801] design-audit OK: gate satisfiable, "
        f"num_elite={_num_elite} (>= {MIN_REQUIRED_ELITES})",
        flush=True,
    )

    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        for ctx in arm_contexts():
            rows.append(
                run_cell(
                    arm_ctx=ctx,
                    seed=seed,
                    full_config=full_config,
                    warmup_episodes=warmup_episodes,
                    eval_episodes=eval_episodes,
                    steps_per_episode=steps_per_episode,
                    dry_run=dry_run,
                )
            )

    gate = build_gate(rows, list(seeds))
    analysis = analyse(rows, list(seeds))

    non_degenerate = bool(gate["non_degenerate"])
    if not non_degenerate:
        outcome = "FAIL"
        label = "substrate_not_ready_requeue"
        direction = "non_contributory"
        interpretation_text = (
            "SUBSTRATE NOT READY -- not a verdict on ARC-018. "
            + gate["degeneracy_reason"]
        )
    else:
        outcome = "PASS" if analysis["all_pass"] else "FAIL"
        if analysis["all_pass"]:
            label = "arc018_rollout_leg_supported_depth_monotonic"
            direction = "supports"
            interpretation_text = (
                "ARC-018 rollout leg SUPPORTED: harm_advantage scales monotonically "
                f"with rollout depth (shallow {analysis['harm_advantage_by_arm'][ARM_SHALLOW]:.4f} "
                f"-> full {analysis['harm_advantage_by_arm'][ARM_FULL]:.4f}), strictly "
                "above the matched-noise control, at verified E2 fidelity. ARC-018 "
                "holds CONJUNCTIVELY: leg A (mapping) on EXQ-042/053/120 and leg B "
                "(explicit rollouts) here."
            )
        else:
            label = "arc018_rollout_leg_weakened_mapping_leg_stands"
            direction = "weakens"
            interpretation_text = (
                "ARC-018 rollout leg WEAKENED: no advantage that scales with rollout "
                "depth, despite verified-live rollouts (cross-candidate score range "
                "above floor) and adequate E2 fidelity (r2 >= "
                f"{FLOOR_E2_WORLD_R2}). Both prior FAILs were confounded and are "
                "answered here: this is NOT the EXQ-172 low-fidelity confound and "
                "NOT the EXQ-196 unwired-DV degeneracy. ARC-018 reduces to its "
                "MAPPING leg, which stands on EXQ-042/053/120; MECH-092 and the "
                "replay-consolidation dependents need re-derivation. Decision-"
                "flipping negative, not a null."
            )

    criteria = [
        {"name": "C1_monotonic_depth_response", "load_bearing": True,
         "passed": analysis["c1_monotonic_pass"]},
        {"name": "C2_effect_sd", "load_bearing": False,
         "passed": analysis["c2_effect_sd_pass"]},
        {"name": "C3_seed_consistency", "load_bearing": False,
         "passed": analysis["c3_seed_consistency_pass"]},
        {"name": "C4_above_matched_noise", "load_bearing": False,
         "passed": analysis["c4_above_noise_pass"]},
    ]

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest: Dict[str, Any] = {
        "run_id": f"{EXPERIMENT_TYPE}_{ts}_v3",
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "backlog_id": BACKLOG_ID,
        "outcome": outcome,
        "timestamp_utc": ts,
        "evidence_direction": direction,
        "non_degenerate": non_degenerate,
        "degeneracy_reason": gate["degeneracy_reason"],
        "per_arm_gate": gate["per_arm_gate"],
        "interpretation": {
            "label": label,
            "text": interpretation_text,
            "preconditions": gate["adjudication_preconditions"],
            "preconditions_scope_note": gate["per_arm_gate"].get(
                "preconditions_scope_note", ""
            ),
            "criteria_non_degenerate": arm_criteria_non_degenerate(
                {
                    ARM_FULL: [
                        "C1_monotonic_depth_response",
                        "C2_effect_sd",
                        "C3_seed_consistency",
                    ],
                    ARM_NOISE: ["C4_above_matched_noise"],
                },
                gate,
            ),
        },
        "criteria": criteria,
        "analysis": analysis,
        "arm_results": rows,
        "per_seed_harm_rate": {
            a: [r["harm_rate_eval"] for r in rows if r["arm_id"] == a] for a in ARMS
        },
        "per_seed_e2_world_r2": {
            a: [r["e2_world_r2"] for r in rows if r["arm_id"] == a] for a in ARMS
        },
        "config": full_config,
        "seeds": list(seeds),
        "registered_thresholds": full_config["thresholds"],
        "confound_traps_note": (
            "P1 traps the EXQ-172 confound (e2_world_r2 = 0.203 there; floor "
            f"{FLOOR_E2_WORLD_R2} measured in-run here). P3 traps the EXQ-196 "
            "confound (harm_advantage_mean bit-exactly 0.0 on all 3 seeds at good "
            "fidelity). P3 is thresholded on the COUNT of seeds with a nonzero "
            "advantage, never on magnitude, so a genuine small-effect null still "
            "scores as a FAIL rather than being mislabelled substrate_not_ready."
        ),
    }

    # stamp happens in write_flat_manifest (after arm_results exist, so
    # substrate_hash HOISTS from the per-cell fingerprints)
    print("\n[V3-EXQ-801] Results", flush=True)
    for a in ARMS:
        print(
            f"  {a} (h={HORIZON_BY_ARM[a]}): harm_rate="
            f"{analysis['harm_rate_by_arm'][a]:.4f}"
            f"  advantage={analysis['harm_advantage_by_arm'][a]:+.4f}"
            f"  e2_r2={analysis['e2_world_r2_by_arm'][a]:.4f}",
            flush=True,
        )
    print(
        f"  monotonic={analysis['c1_monotonic_pass']}"
        f"  n_seeds_nonzero_adv={analysis['n_seeds_nonzero_advantage']}"
        f"  non_degenerate={non_degenerate}  outcome={outcome}",
        flush=True,
    )
    manifest["_t0"] = t0
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--warmup", type=int, default=300)
    parser.add_argument("--eval-eps", type=int, default=60)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    _t_start = time.perf_counter()
    manifest = run(
        seeds=tuple(args.seeds),
        warmup_episodes=args.warmup,
        eval_episodes=args.eval_eps,
        steps_per_episode=args.steps,
        dry_run=args.dry_run,
    )
    _t_start = manifest.pop("_t0", _t_start)

    out_dir = (
        Path(__file__).resolve().parents[2]
        / "REE_assembly" / "evidence" / "experiments"
    )
    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=args.dry_run,
        config=manifest.get("config"),
        seeds=list(args.seeds),
        script_path=Path(__file__),
        started_at=_t_start,
    )
    print(f"\nResult written to: {out_path}", flush=True)

    _outcome = str(manifest.get("outcome", "FAIL")).upper()
    emit_outcome(
        outcome=_outcome if _outcome in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
        dry_run=args.dry_run,
    )
