"""
V3-EXQ-876a -- MECH-025: Action-Doing Mode Convergence Redesign
(DV re-operationalization of V3-EXQ-876, per confirmed
failure_autopsy_mech025-cluster-876-671b_2026-08-03)

Claims: MECH-025

Why this redesign (routed by the autopsy, executed here):
  V3-EXQ-876 (2026-08-02) was the first fair V3 test of MECH-025 -- all three
  prior instrument defects (cache-field/torn-down-field commitment read,
  frozen running_variance during eval) were fixed and verified: committed and
  uncommitted regimes were both well-sampled and non-degenerate in every seed
  (committed_step_count 903-9818, uncommitted_step_count 165-216, both far
  above the 20-step floor). Despite the clean instrument, its `doing_mode_delta`
  DV came back NEGATIVE in 4/5 seeds (-0.031, -0.066, -0.081, -0.087, only
  +0.0025 in seed 256) -- the OPPOSITE of C1's predicted sign
  (`doing_mode_delta > 0.002`).

  The confirmed autopsy classified this `non_contributory` /
  `measurement_test_design_defect`, not claim pressure, because the claim's
  OWN cited literature predicts the operationalization's sign convention may
  be backwards:
    - Friston et al. 2013 (this claim's primary grounding, confidence 0.72)
      records its own `failure_signatures`: "The active inference framework
      treats precision as a property of beliefs about policies, not a
      dedicated action-mode regime -- it is continuous, not a discrete mode
      switch." I.e. C1's DISCRETE committed-vs-uncommitted split was never a
      clean fit to this claim's strongest lit support in the first place.
    - Thura & Cisek 2014 (Neuron, confidence 0.81, the closer biological
      reference for a *committed vs. deliberating* operationalization) is
      explicit that premotor/M1 representations CONVERGE (narrow) during the
      deliberation-to-commitment transition, not diverge. A converging
      representation predicts a SMALLER gap between the actual action's
      predicted consequence and a counterfactual alternative's, once
      committed -- i.e. the causal_sig contrast should NARROW under
      commitment, exactly the negative `doing_mode_delta` V3-EXQ-876 observed
      in 4/5 seeds.

  claims.yaml's `what_would_answer` (2026-08-08 digestion) explicitly offers
  two redesign directions: (1) flip C1's predicted sign per the Thura-Cisek
  convergence account, or (2) gate on `precision_committed_mean` vs
  `precision_uncommitted_mean` directly (already recorded in V3-EXQ-876,
  separating by 4-5 orders of magnitude in every seed, but never scored as a
  criterion).

  THIS SCRIPT TAKES DIRECTION (1) AS THE PRIMARY, LOAD-BEARING CRITERION,
  AND EXPLAINS WHY (2) IS DEMOTED TO A NON-GATING, DESCRIPTIVE METRIC:

    Direction (2) tests `precision_committed_mean > precision_uncommitted_mean`
    (or a ratio margin over it). But E3's commit rule is LITERALLY
    `committed = running_variance < commit_threshold` and
    `precision = 1/(running_variance + eps)` (e3_selector.py) -- so
    "committed steps have higher precision" is a near-TAUTOLOGICAL
    consequence of the substrate's own definition of "committed," not an
    independently falsifiable prediction of MECH-025. Gating PASS/FAIL on it
    would be a `vacuous_pass` in exactly the sense the skill's DV-symmetry /
    non-degeneracy discipline warns against: the criterion is structurally
    almost guaranteed to pass regardless of whether the *claim* (action mode
    exercises high-precision control in a way that matters causally) is true.
    ARC-016 (dynamic E3 precision) already independently validates that this
    mechanism exists and is live (`status: stable`) -- re-scoring it under a
    MECH-025 label would mostly re-confirm ARC-016, not test MECH-025's own
    (downstream, dependent) claim.

    Direction (1), by contrast, tests a genuinely independent, computed
    consequence: `causal_sig = E3.harm_eval(E2.world_forward(z_world,
    a_actual)) - E3.harm_eval(E2.world_forward(z_world, a_cf))` involves two
    real world-model rollouts (actual vs. a distinct randomly-sampled
    counterfactual action) and is NOT wired to the commit rule's own
    variance/precision computation at all -- it can come out any sign,
    any magnitude, for reasons entirely orthogonal to how "committed" is
    defined. This is the genuinely falsifiable test.

  DV-SYMMETRY DECLARATION (per Step 3 MANDATORY DECLARATION): causal_sig is
  computed from two independent E2.world_forward rollouts under two distinct
  actions (the actual action and a randomly-sampled distinct counterfactual
  action, `cf_idx != actual_idx` by construction) fed through E3.harm_eval.
  This is not a broadcast constant (both rollouts depend on the actual
  z_world and a genuinely different action one-hot) and not a monotone
  rescaling of a shared upstream quantity (E2.world_forward and E3.harm_eval
  are both nonlinear). So the manipulation (committed vs. uncommitted) is NOT
  structurally invariant under any symmetry that would zero out
  `doing_mode_delta` by construction -- unlike Direction (2)'s precision
  ratio, which IS structurally guaranteed positive by the commit-rule
  definition.

  GOV-REUSE-1 CHECK (Step 2.4, decisive readout = a pre-registered,
  falsifiable criterion on `doing_mode_delta`): `reanalysis_query.py query
  --readout precision_committed_mean --claim MECH-025` found exactly ONE
  manifest carrying MECH-025 precision/causal-signal readouts
  (v3_exq_876_mech025_doing_mode_causal_signal_20260802T214005Z_v3,
  substrate_hash 5bba8a679965b38f...). That manifest's `doing_mode_delta`
  values are ALREADY KNOWN to this authoring session (negative in 4/5 seeds).
  A "reversed sign" criterion re-scored against those SAME already-observed
  values would be circular (choosing the hypothesis to match data already
  seen -- HARKing), regardless of whether it is done via a formal reanalysis
  artifact or by relabeling the same run under a new script. This experiment
  therefore does NOT reuse V3-EXQ-876's manifest as its confirmatory test --
  it runs FRESH SEEDS (11, 17, 29, 53, 71 -- disjoint from V3-EXQ-876's
  [42, 123, 7, 99, 256], and none is 44, the documented reef-config
  instability seed) through the IDENTICAL, already-validated instrument
  (unchanged commitment-field read, unchanged update_residue-every-tick call,
  unchanged BreathOscillator config) so the redesigned criterion is tested
  against genuinely new data, not a relabeling of old data. The magnitude
  threshold (0.002) is carried over UNCHANGED from V3-EXQ-876's own
  pre-registered C1 bar (only the SIGN is flipped) -- not fit to the observed
  effect sizes (-0.03 to -0.09), which were roughly 10-40x larger than this
  floor, leaving real room for the fresh run to fail the reversed criterion
  too.

  RECORDING FIX: V3-EXQ-876's `write_flat_manifest(..., config=None, ...)`
  dropped `config`/`elapsed_seconds` from the always-core (flagged in the
  autopsy, Step 2b). This script passes the real `config` and a `started_at`
  timestamp through.

Env: CausalGridWorldV2(size=6, num_hazards=4, num_resources=5, nav_bias=0.45)
  -- UNCHANGED from V3-EXQ-876 (same tuned env; only the DV/criteria change).
use_event_classifier=True (SD-009) -- UNCHANGED.

EXPERIMENT_PURPOSE = "evidence" (direct test of MECH-025's claim; not a
  diagnostic/probe -- the diagnostic-adjudication `preconditions[]` /
  `criteria_non_degenerate{}` machinery below is populated per the
  "Non-degeneracy scoring net applies to evidence runs too" convention, not
  because this is a diagnostic).

PASS criteria (ALL must hold, per seed):
  C1 (LOAD-BEARING, REVERSED vs V3-EXQ-876): doing_mode_delta < -0.002
     (committed/"doing" steps show a SMALLER |causal-signature| contrast than
     uncommitted steps -- convergence/narrowing during commitment, per
     Thura & Cisek 2014, not the divergence V3-EXQ-876's C1 assumed).
  C2: committed_step_count >= 20     (doing mode genuinely entered)
  C3: uncommitted_step_count >= 20   (exploring mode genuinely entered)
  C4: world_forward_r2 > 0.05        (E2 world model functional)
  C5: harm_pred_std > 0.01           (E3 not collapsed)
  C6: No fatal errors

Reported, NON-GATING (Direction 2, descriptive only -- see rationale above):
  precision_committed_mean, precision_uncommitted_mean, precision_ratio.
  `precision_ratio_note` in the output explains why this is not scored.

Overall verdict: PASS iff ALL seeds PASS; evidence_direction from the
  aggregate criteria-met fraction (same convention as V3-EXQ-876 and
  V3-EXQ-199 before it).

Supersedes (redesign of the same claim/question, per autopsy routing):
  v3_exq_876_mech025_doing_mode_causal_signal_20260802T214005Z_v3
"""

import sys
import random
import dataclasses
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from experiment_protocol import emit_outcome  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from experiments._metrics import (  # noqa: E402
    check_degeneracy,
    p0_readiness_gate,
    P0NotReady,
)
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402


EXPERIMENT_TYPE = "v3_exq_876a_mech025_doing_mode_convergence_redesign"
EXPERIMENT_PURPOSE = "evidence"
CLAIM_IDS = ["MECH-025"]
# Redesign of the SAME claim/question -- unchanged substrate mechanism, unchanged
# instrument, only the DV sign convention (and the precision-ratio metric's role)
# changed, per failure_autopsy_mech025-cluster-876-671b_2026-08-03's routing.
SUPERSEDES = "v3_exq_876_mech025_doing_mode_causal_signal_20260802T214005Z_v3"
COMMITTED_FLOOR = 20    # P0: committed_step_count must clear this (unchanged from 876)
UNCOMMITTED_FLOOR = 20  # P0: uncommitted_step_count must clear this (unchanged from 876)
# C1 bar carried over UNCHANGED from V3-EXQ-876's own pre-registered magnitude
# (0.002) -- only the SIGN flips. NOT fit to the observed effect sizes (see
# GOV-REUSE-1 note in the module docstring).
C1_MAGNITUDE = 0.002

_ZG = ZGoalStreamAccumulator()


def _action_to_onehot(action_idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, action_idx] = 1.0
    return v


def _mean_safe(lst: List[float]) -> float:
    return float(sum(lst) / len(lst)) if lst else 0.0


def _train(
    agent: REEAgent,
    env: CausalGridWorldV2,
    optimizer: optim.Optimizer,
    wf_optimizer: optim.Optimizer,
    harm_eval_optimizer: optim.Optimizer,
    num_episodes: int,
    steps_per_episode: int,
    world_dim: int,
    nav_bias: float,
) -> Dict:
    """Standard full-pipeline training to get a functional E3 + E2.world_forward.

    UNCHANGED from V3-EXQ-876/V3-EXQ-199/V3-EXQ-671a: training does not call
    update_residue (residue/running_variance liveness is only required in
    EVAL, where the commit contrast is actually read).
    """
    agent.train()
    harm_buf_pos: List[torch.Tensor] = []
    harm_buf_neg: List[torch.Tensor] = []
    wf_buf: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    total_harm = 0
    e3_tick_total = 0

    for ep in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        z_world_prev: Optional[torch.Tensor] = None
        action_prev: Optional[torch.Tensor] = None
        z_self_prev: Optional[torch.Tensor] = None

        for _ in range(steps_per_episode):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            latent = agent.sense(obs_body, obs_world)

            if z_self_prev is not None and action_prev is not None:
                agent.record_transition(z_self_prev, action_prev, latent.z_self.detach())

            ticks = agent.clock.advance()
            e1_prior = (
                agent._e1_tick(latent) if ticks.get("e1_tick", False)
                else torch.zeros(1, world_dim, device=agent.device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            theta_z = agent.theta_buffer.summary()
            z_world_curr = latent.z_world.detach()

            if ticks.get("e3_tick", False) and candidates:
                e3_tick_total += 1
                result = agent.e3.select(candidates, temperature=1.0)
                action = result.selected_action.detach()
                agent._last_action = action
            else:
                action = agent._last_action
                if action is None:
                    action = _action_to_onehot(
                        random.randint(0, env.action_dim - 1), env.action_dim, agent.device
                    )
                    agent._last_action = action

            # nav_bias (V3-EXQ-199/876 pattern): with probability nav_bias, override
            # action to move toward the nearest hazard -- increases harm exposure
            # for the E3.harm_eval training signal.
            if random.random() < nav_bias:
                agent_pos = getattr(env, "agent_pos", None)
                if agent_pos is not None and hasattr(env, "hazard_positions") and env.hazard_positions:
                    ax, ay = agent_pos
                    nearest = min(env.hazard_positions, key=lambda h: abs(h[0] - ax) + abs(h[1] - ay))
                    dx, dy = nearest[0] - ax, nearest[1] - ay
                    if abs(dx) >= abs(dy):
                        nav_act = 1 if dx > 0 else 0  # down / up
                    else:
                        nav_act = 3 if dy > 0 else 2  # right / left
                    action = _action_to_onehot(nav_act, env.action_dim, agent.device)
                    agent._last_action = action

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)

            if z_world_prev is not None and action_prev is not None:
                wf_buf.append((z_world_prev.cpu(), action_prev.cpu(), z_world_curr.cpu()))
                if len(wf_buf) > 2000:
                    wf_buf = wf_buf[-2000:]

            if harm_signal < 0:
                total_harm += 1
                harm_buf_pos.append(theta_z.detach())
                if len(harm_buf_pos) > 1000:
                    harm_buf_pos = harm_buf_pos[-1000:]
            else:
                harm_buf_neg.append(theta_z.detach())
                if len(harm_buf_neg) > 1000:
                    harm_buf_neg = harm_buf_neg[-1000:]

            e1_loss = agent.compute_prediction_loss()
            if e1_loss.requires_grad:
                optimizer.zero_grad()
                e1_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.e1.parameters(), 1.0)
                optimizer.step()

            if len(wf_buf) >= 16:
                k = min(32, len(wf_buf))
                idxs = torch.randperm(len(wf_buf))[:k].tolist()
                zw_b = torch.cat([wf_buf[i][0] for i in idxs]).to(agent.device)
                a_b = torch.cat([wf_buf[i][1] for i in idxs]).to(agent.device)
                zw1_b = torch.cat([wf_buf[i][2] for i in idxs]).to(agent.device)
                wf_loss = F.mse_loss(agent.e2.world_forward(zw_b, a_b), zw1_b)
                if wf_loss.requires_grad:
                    wf_optimizer.zero_grad()
                    wf_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(agent.e2.world_transition.parameters())
                        + list(agent.e2.world_action_encoder.parameters()),
                        1.0,
                    )
                    wf_optimizer.step()

            if len(harm_buf_pos) >= 4 and len(harm_buf_neg) >= 4:
                k_p = min(16, len(harm_buf_pos))
                k_n = min(16, len(harm_buf_neg))
                pi = torch.randperm(len(harm_buf_pos))[:k_p].tolist()
                ni = torch.randperm(len(harm_buf_neg))[:k_n].tolist()
                zw_b = torch.cat(
                    [harm_buf_pos[i] for i in pi] + [harm_buf_neg[i] for i in ni], dim=0
                )
                target = torch.cat(
                    [
                        torch.ones(k_p, 1, device=agent.device),
                        torch.zeros(k_n, 1, device=agent.device),
                    ],
                    dim=0,
                )
                pred = agent.e3.harm_eval(zw_b)
                harm_loss = F.mse_loss(pred, target)
                if harm_loss.requires_grad:
                    harm_eval_optimizer.zero_grad()
                    harm_loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.e3.harm_eval_head.parameters(), 0.5)
                    harm_eval_optimizer.step()

            z_world_prev = z_world_curr
            z_self_prev = latent.z_self.detach()
            action_prev = action.detach()
            if done:
                break

        if (ep + 1) % 100 == 0 or ep == num_episodes - 1:
            print(
                f"  [train] ep {ep+1}/{num_episodes}  harm={total_harm}"
                f"  e3_ticks={e3_tick_total}",
                flush=True,
            )

    return {"total_harm": total_harm, "wf_buf": wf_buf, "e3_tick_total": e3_tick_total}


def _compute_world_forward_r2(agent: REEAgent, wf_buf: List, n_test: int = 200) -> float:
    if len(wf_buf) < n_test:
        return 0.0
    idxs = list(range(len(wf_buf) - n_test, len(wf_buf)))
    with torch.no_grad():
        zw = torch.cat([wf_buf[i][0] for i in idxs]).to(agent.device)
        a = torch.cat([wf_buf[i][1] for i in idxs]).to(agent.device)
        zw1 = torch.cat([wf_buf[i][2] for i in idxs]).to(agent.device)
        pred = agent.e2.world_forward(zw, a)
        ss_res = ((zw1 - pred) ** 2).sum()
        ss_tot = ((zw1 - zw1.mean(dim=0, keepdim=True)) ** 2).sum()
    return float((1 - ss_res / (ss_tot + 1e-8)).item())


def _eval_doing_mode(
    agent: REEAgent,
    env: CausalGridWorldV2,
    num_episodes: int,
    steps_per_episode: int,
    world_dim: int,
) -> Dict:
    """
    Probe action-doing mode by comparing |causal signature| during committed
    vs uncommitted steps.

    UNCHANGED instrument vs V3-EXQ-876 (see module docstring -- only the
    downstream PASS criterion built on top of these numbers changes):
      (1) is_committed is sourced from SelectionResult.committed (the real
          BetaGate/E3 commitment-threshold outcome returned directly by
          agent.e3.select(...)), never a candidate cache or a torn-down flag.
      (2) agent.update_residue(...) is called after every env.step() so
          running_variance -- what the commit decision is actually computed
          from -- is genuinely live during eval, not frozen at its
          post-training value.
      (3) sweep_threshold_reduction (BreathOscillator, MECH-108) is passed
          into the SAME agent.e3.select(...) call that produces
          result.committed, guaranteeing periodic genuine uncommitted
          windows even once running_variance converges low.
    """
    agent.eval()
    causal_sigs_committed: List[float] = []
    causal_sigs_uncommitted: List[float] = []
    precision_committed: List[float] = []
    precision_uncommitted: List[float] = []
    all_harm_preds: List[float] = []
    fatal = 0
    sweep_step_count = 0

    for _ in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        z_self_prev: Optional[torch.Tensor] = None
        action_prev: Optional[torch.Tensor] = None
        held_committed: bool = False
        held_precision: float = 0.0

        for _ in range(steps_per_episode):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]

            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world)
                if z_self_prev is not None and action_prev is not None:
                    agent.record_transition(z_self_prev, action_prev, latent.z_self.detach())

                ticks = agent.clock.advance()
                e1_prior = (
                    agent._e1_tick(latent)
                    if ticks.get("e1_tick", False)
                    else torch.zeros(1, world_dim, device=agent.device)
                )
                candidates = agent.generate_trajectories(latent, e1_prior, ticks)

            try:
                # (3) BreathOscillator sweep reduction -- same computation
                # select_action itself performs (agent.py:6084), passed here
                # directly into the e3.select() call whose return value we
                # read result.committed from.
                sweep_reduction = (
                    agent.clock.sweep_amplitude if agent.clock.sweep_active else 0.0
                )
                if agent.clock.sweep_active:
                    sweep_step_count += 1

                if ticks.get("e3_tick", False) and candidates:
                    with torch.no_grad():
                        result = agent.e3.select(
                            candidates,
                            temperature=1.0,
                            sweep_threshold_reduction=sweep_reduction,
                        )
                        action = result.selected_action.detach()
                        agent._last_action = action
                        # (1) the real commitment signal, never a cache or a
                        # torn-down flag.
                        held_committed = bool(result.committed)
                        held_precision = float(result.precision)
                else:
                    action = agent._last_action
                    if action is None:
                        action = _action_to_onehot(
                            random.randint(0, env.action_dim - 1), env.action_dim, agent.device
                        )
                        agent._last_action = action
                    # Held state between e3 ticks: commitment/precision carry
                    # forward from the last e3_tick's read (there is no
                    # fresher signal available between ticks; held_committed
                    # defaults to False until the first e3_tick of the
                    # episode fires).

                is_committed = held_committed
                current_precision = held_precision

                # Compute causal signature via SD-003 counterfactual.
                with torch.no_grad():
                    z_world = latent.z_world  # [1, world_dim]
                    actual_idx = action.argmax(dim=-1).item()
                    cf_idx = (random.randint(0, env.action_dim - 2) + 1 + actual_idx) % env.action_dim
                    a_cf = _action_to_onehot(int(cf_idx), env.action_dim, agent.device)

                    z_actual = agent.e2.world_forward(z_world, action)
                    z_cf = agent.e2.world_forward(z_world, a_cf)
                    h_actual = float(agent.e3.harm_eval(z_actual).item())
                    h_cf = float(agent.e3.harm_eval(z_cf).item())
                    causal_sig = h_actual - h_cf
                    all_harm_preds.append(h_actual)

                # Execute action.
                flat_obs, harm_signal, done, info, obs_dict = env.step(action)

                # (2) THE FIX: feed the harm signal into the canonical
                # post-action path every tick, so running_variance (and
                # hence the commit decision) is genuinely live for the whole
                # eval run. Ownership does not matter for this claim (we are
                # not testing responsibility attribution), so owned=True
                # unconditionally -- matching StepHarness's canonical default.
                agent.update_residue(
                    harm_signal=float(harm_signal),
                    world_delta=None,
                    hypothesis_tag=False,
                    owned=True,
                )

                if is_committed:
                    causal_sigs_committed.append(abs(causal_sig))
                    precision_committed.append(current_precision)
                else:
                    causal_sigs_uncommitted.append(abs(causal_sig))
                    precision_uncommitted.append(current_precision)

            except Exception:
                fatal += 1
                action = _action_to_onehot(
                    random.randint(0, env.action_dim - 1), env.action_dim, agent.device
                )
                agent._last_action = action
                flat_obs, harm_signal, done, info, obs_dict = env.step(action)
                try:
                    agent.update_residue(
                        harm_signal=float(harm_signal),
                        world_delta=None,
                        hypothesis_tag=False,
                        owned=True,
                    )
                except Exception:
                    pass

            z_self_prev = latent.z_self.detach()
            action_prev = action.detach()
            if done:
                break

    mean_committed = _mean_safe(causal_sigs_committed)
    mean_uncommitted = _mean_safe(causal_sigs_uncommitted)
    doing_mode_delta = mean_committed - mean_uncommitted
    harm_pred_std = (
        float(torch.tensor(all_harm_preds).std().item()) if len(all_harm_preds) > 1 else 0.0
    )
    precision_committed_mean = _mean_safe(precision_committed)
    precision_uncommitted_mean = _mean_safe(precision_uncommitted)
    precision_ratio = (
        precision_committed_mean / precision_uncommitted_mean
        if precision_uncommitted_mean > 1e-12
        else 0.0
    )

    print(
        f"  |causal_sig| committed={mean_committed:.4f}  uncommitted={mean_uncommitted:.4f}"
        f"  doing_mode_delta={doing_mode_delta:+.4f}"
        f"  n_committed={len(causal_sigs_committed)}  n_uncommitted={len(causal_sigs_uncommitted)}"
        f"  sweep_steps={sweep_step_count}"
        f"  precision_ratio={precision_ratio:.2f}",
        flush=True,
    )

    return {
        "mean_abs_causal_sig_committed": mean_committed,
        "mean_abs_causal_sig_uncommitted": mean_uncommitted,
        "doing_mode_delta": doing_mode_delta,
        "committed_step_count": len(causal_sigs_committed),
        "uncommitted_step_count": len(causal_sigs_uncommitted),
        "harm_pred_std": harm_pred_std,
        "fatal_errors": fatal,
        "sweep_step_count": sweep_step_count,
        "precision_committed_mean": precision_committed_mean,
        "precision_uncommitted_mean": precision_uncommitted_mean,
        "precision_ratio": precision_ratio,
        "causal_sigs_committed": causal_sigs_committed,
        "causal_sigs_uncommitted": causal_sigs_uncommitted,
    }


def run(
    seed: int = 0,
    warmup_episodes: int = 500,
    eval_episodes: int = 50,
    steps_per_episode: int = 200,
    alpha_world: float = 0.9,
    alpha_self: float = 0.3,
    harm_scale: float = 0.02,
    proximity_scale: float = 0.05,
    nav_bias: float = 0.45,
    lr: float = 1e-3,
    self_dim: int = 32,
    world_dim: int = 32,
    breath_period: int = 50,
    breath_sweep_amplitude: float = 0.30,
    breath_sweep_duration: int = 10,
    **kwargs,
) -> dict:
    torch.manual_seed(seed)
    random.seed(seed)

    env = CausalGridWorldV2(
        seed=seed,
        size=6,
        num_hazards=4,
        num_resources=5,
        hazard_harm=harm_scale,
        env_drift_interval=5,
        env_drift_prob=0.1,
        proximity_harm_scale=proximity_scale,
        proximity_benefit_scale=proximity_scale * 0.6,
        proximity_approach_threshold=0.15,
        hazard_field_decay=0.5,
    )
    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=self_dim,
        world_dim=world_dim,
        alpha_world=alpha_world,
        alpha_self=alpha_self,
        reafference_action_dim=env.action_dim,
        use_event_classifier=True,  # SD-009 (V3-EXQ-199/876 precedent)
        breath_period=breath_period,
        breath_sweep_amplitude=breath_sweep_amplitude,
        breath_sweep_duration=breath_sweep_duration,
    )
    agent = REEAgent(config)

    optimizer = optim.Adam(list(agent.e1.parameters()), lr=lr)
    wf_optimizer = optim.Adam(
        list(agent.e2.world_transition.parameters())
        + list(agent.e2.world_action_encoder.parameters()),
        lr=1e-3,
    )
    harm_eval_optimizer = optim.Adam(
        list(agent.e3.harm_eval_head.parameters()),
        lr=1e-4,
    )

    print(
        f"[V3-EXQ-876a] MECH-025: Action-Doing Mode Convergence Redesign (seed={seed})\n"
        f"  warmup={warmup_episodes}  eval={eval_episodes}  alpha_world={alpha_world}\n"
        f"  breath_period={breath_period}  sweep_amplitude={breath_sweep_amplitude}"
        f"  sweep_duration={breath_sweep_duration}\n"
        f"  nav_bias={nav_bias}  size=6  n_hazards=4",
        flush=True,
    )

    train_out = _train(
        agent,
        env,
        optimizer,
        wf_optimizer,
        harm_eval_optimizer,
        warmup_episodes,
        steps_per_episode,
        world_dim,
        nav_bias,
    )
    world_forward_r2 = _compute_world_forward_r2(agent, train_out["wf_buf"])
    print(f"  world_forward_r2: {world_forward_r2:.4f}", flush=True)

    print(f"\n[V3-EXQ-876a] Eval -- probing action-doing mode convergence...", flush=True)
    eval_out = _eval_doing_mode(agent, env, eval_episodes, steps_per_episode, world_dim)

    # z_goal stream liveness recording (Step 3 output-contract requirement).
    # A fresh agent is built inside run() per seed and falls out of scope on
    # return -- pool counters via the accumulator rather than keeping every
    # seed's agent alive.
    _ZG.observe(agent)

    # --- POSITIVE-CONTROL GATE ------------------------------------------
    # P0: both regimes must be genuinely sampled this run before C1 is read
    # as evidence -- unchanged floors from V3-EXQ-876 (proven robust: cleared
    # cleanly in every seed of that run).
    readiness_checks = [
        {
            "name": "committed_regime_sampled",
            "measured": eval_out["committed_step_count"],
            "threshold": COMMITTED_FLOOR,
            "direction": "lower",
            "control": (
                "committed_step_count from SelectionResult.committed, must "
                f">= {COMMITTED_FLOOR} for the doing-mode contrast to be "
                "non-vacuous (V3-EXQ-050/050b root cause: cache field read "
                "as always-committed)."
            ),
        },
        {
            "name": "uncommitted_regime_sampled",
            "measured": eval_out["uncommitted_step_count"],
            "threshold": UNCOMMITTED_FLOOR,
            "direction": "lower",
            "control": (
                "uncommitted_step_count from SelectionResult.committed, must "
                f">= {UNCOMMITTED_FLOOR} for the doing-mode contrast to be "
                "non-vacuous (V3-EXQ-199 root cause: torn-down field read as "
                "never-committed)."
            ),
        },
    ]
    ready = True
    preconditions = []
    try:
        preconditions = p0_readiness_gate(readiness_checks)
    except P0NotReady as e:
        preconditions = e.preconditions
        ready = False

    degeneracy = check_degeneracy(
        {
            "C1_causal_sig_committed": eval_out["causal_sigs_committed"],
            "C1_causal_sig_uncommitted": eval_out["causal_sigs_uncommitted"],
        }
    )
    non_degenerate = degeneracy["non_degenerate"]

    # PASS / FAIL. C1 REVERSED vs V3-EXQ-876 (see module docstring): predicts
    # convergence (SMALLER causal-sig contrast under commitment), per
    # Thura & Cisek 2014, not the divergence V3-EXQ-876 assumed.
    c1_pass = eval_out["doing_mode_delta"] < -C1_MAGNITUDE
    c2_pass = eval_out["committed_step_count"] >= COMMITTED_FLOOR
    c3_pass = eval_out["uncommitted_step_count"] >= UNCOMMITTED_FLOOR
    c4_pass = world_forward_r2 > 0.05
    c5_pass = eval_out["harm_pred_std"] > 0.01
    c6_pass = eval_out["fatal_errors"] == 0

    all_pass = c1_pass and c2_pass and c3_pass and c4_pass and c5_pass and c6_pass
    criteria_met = sum([c1_pass, c2_pass, c3_pass, c4_pass, c5_pass, c6_pass])

    if not ready:
        status = "FAIL"
        evidence_direction = "non_contributory"
        label = "substrate_not_ready_requeue"
    elif not non_degenerate:
        status = "FAIL"
        evidence_direction = "non_contributory"
        label = "c1_causal_sig_degenerate_vacuous_test"
    elif all_pass:
        status = "PASS"
        evidence_direction = "supports"
        label = "doing_mode_produces_convergent_causal_signature"
    else:
        status = "FAIL"
        evidence_direction = "mixed" if criteria_met >= 4 else "weakens"
        label = "doing_mode_does_not_produce_convergent_causal_signature"

    failure_notes = []
    if not ready:
        failure_notes.append(
            "P0 FAIL: readiness precondition unmet -- "
            f"committed={eval_out['committed_step_count']} (floor {COMMITTED_FLOOR}), "
            f"uncommitted={eval_out['uncommitted_step_count']} (floor {UNCOMMITTED_FLOOR}) "
            "(substrate_not_ready_requeue)"
        )
    if ready and not non_degenerate:
        failure_notes.append(f"DEGENERATE: {degeneracy['degeneracy_reason']}")
    if not c1_pass:
        failure_notes.append(
            f"C1 FAIL: doing_mode_delta={eval_out['doing_mode_delta']:.4f} "
            f">= -{C1_MAGNITUDE} (predicted convergence not observed)"
        )
    if not c2_pass:
        failure_notes.append(
            f"C2 FAIL: committed_step_count={eval_out['committed_step_count']} < {COMMITTED_FLOOR}"
        )
    if not c3_pass:
        failure_notes.append(
            f"C3 FAIL: uncommitted_step_count={eval_out['uncommitted_step_count']} < {UNCOMMITTED_FLOOR}"
        )
    if not c4_pass:
        failure_notes.append(f"C4 FAIL: world_forward_r2={world_forward_r2:.4f} <= 0.05")
    if not c5_pass:
        failure_notes.append(f"C5 FAIL: harm_pred_std={eval_out['harm_pred_std']:.4f} <= 0.01")
    if not c6_pass:
        failure_notes.append(f"C6 FAIL: fatal_errors={eval_out['fatal_errors']}")

    print(f"\nV3-EXQ-876a seed={seed} verdict: {status}  label={label}  ({criteria_met}/6)", flush=True)
    for note in failure_notes:
        print(f"  {note}", flush=True)

    precision_ratio_note = (
        "precision_committed_mean/precision_uncommitted_mean is reported for "
        "descriptive completeness (Friston 2013's precision-elevation "
        "prediction) but is NOT scored as a PASS/FAIL criterion: E3's commit "
        "rule (committed = running_variance < commit_threshold, "
        "precision = 1/(running_variance+eps)) makes 'committed steps have "
        "higher precision' a near-tautological consequence of the substrate's "
        "own definition of 'committed', not an independently falsifiable "
        "prediction of MECH-025 -- gating on it would risk a structurally "
        "guaranteed (vacuous) PASS. See module docstring for the full "
        "rationale."
    )

    metrics = {
        "mean_abs_causal_sig_committed": float(eval_out["mean_abs_causal_sig_committed"]),
        "mean_abs_causal_sig_uncommitted": float(eval_out["mean_abs_causal_sig_uncommitted"]),
        "doing_mode_delta": float(eval_out["doing_mode_delta"]),
        "committed_step_count": float(eval_out["committed_step_count"]),
        "uncommitted_step_count": float(eval_out["uncommitted_step_count"]),
        "harm_pred_std": float(eval_out["harm_pred_std"]),
        "world_forward_r2": float(world_forward_r2),
        "e3_tick_total": float(train_out["e3_tick_total"]),
        "total_harm_train": float(train_out["total_harm"]),
        "fatal_error_count": float(eval_out["fatal_errors"]),
        "sweep_step_count": float(eval_out["sweep_step_count"]),
        "precision_committed_mean": float(eval_out["precision_committed_mean"]),
        "precision_uncommitted_mean": float(eval_out["precision_uncommitted_mean"]),
        "precision_ratio": float(eval_out["precision_ratio"]),
        "breath_period": float(breath_period),
        "breath_sweep_amplitude": float(breath_sweep_amplitude),
        "breath_sweep_duration": float(breath_sweep_duration),
        "crit1_pass": 1.0 if c1_pass else 0.0,
        "crit2_pass": 1.0 if c2_pass else 0.0,
        "crit3_pass": 1.0 if c3_pass else 0.0,
        "crit4_pass": 1.0 if c4_pass else 0.0,
        "crit5_pass": 1.0 if c5_pass else 0.0,
        "crit6_pass": 1.0 if c6_pass else 0.0,
        "criteria_met": float(criteria_met),
    }

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(f"- {n}" for n in failure_notes)

    summary_markdown = f"""# V3-EXQ-876a (seed={seed}) -- MECH-025: Action-Doing Mode Convergence Redesign

**Status:** {status}
**Label:** {label}
**Claim:** MECH-025 -- action-doing mode produces a distinct (convergent) internal causal signature
**Supersedes:** {SUPERSEDES} (DV re-operationalization -- C1 sign reversed per
  Thura & Cisek 2014; precision-ratio demoted to non-gating descriptive metric;
  `config`/`elapsed_seconds` recording gap fixed)
**alpha_world:** {alpha_world}
**Warmup:** {warmup_episodes} eps  |  Eval: {eval_episodes} eps
**Seed:** {seed}

## Positive-Control Gate

| Check | Measured | Floor | Met |
|---|---|---|---|
| committed_regime_sampled | {eval_out['committed_step_count']} | {COMMITTED_FLOOR} | {eval_out['committed_step_count'] >= COMMITTED_FLOOR} |
| uncommitted_regime_sampled | {eval_out['uncommitted_step_count']} | {UNCOMMITTED_FLOOR} | {eval_out['uncommitted_step_count'] >= UNCOMMITTED_FLOOR} |

C1 non-degeneracy (causal_sig sample spread, both regimes): {non_degenerate} ({degeneracy['degeneracy_reason'] or 'ok'})

## Results

| Metric | Value |
|--------|-------|
| |causal_sig| Committed (mean) | {eval_out['mean_abs_causal_sig_committed']:.4f} |
| |causal_sig| Uncommitted (mean) | {eval_out['mean_abs_causal_sig_uncommitted']:.4f} |
| doing_mode_delta | {eval_out['doing_mode_delta']:+.4f} |
| Committed steps sampled | {eval_out['committed_step_count']} |
| Uncommitted steps sampled | {eval_out['uncommitted_step_count']} |
| Sweep steps (BreathOscillator) | {eval_out['sweep_step_count']} |
| precision (committed mean) | {eval_out['precision_committed_mean']:.4f} |
| precision (uncommitted mean) | {eval_out['precision_uncommitted_mean']:.4f} |
| precision_ratio (NON-GATING, see note) | {eval_out['precision_ratio']:.2f} |
| World Forward R2 | {world_forward_r2:.4f} |
| Harm Pred Std | {eval_out['harm_pred_std']:.4f} |

## PASS Criteria

| Criterion | Result | Value |
|---|---|---|
| C1 (LOAD-BEARING, REVERSED): doing_mode_delta < -{C1_MAGNITUDE} | {"PASS" if c1_pass else "FAIL"} | {eval_out['doing_mode_delta']:+.4f} |
| C2: committed_step_count >= {COMMITTED_FLOOR} | {"PASS" if c2_pass else "FAIL"} | {eval_out['committed_step_count']} |
| C3: uncommitted_step_count >= {UNCOMMITTED_FLOOR} | {"PASS" if c3_pass else "FAIL"} | {eval_out['uncommitted_step_count']} |
| C4: world_forward_r2 > 0.05 | {"PASS" if c4_pass else "FAIL"} | {world_forward_r2:.4f} |
| C5: harm_pred_std > 0.01 | {"PASS" if c5_pass else "FAIL"} | {eval_out['harm_pred_std']:.4f} |
| C6: No fatal errors | {"PASS" if c6_pass else "FAIL"} | {eval_out['fatal_errors']} |

Combination rule: PASS iff ALL SIX criteria hold (plain AND; unchanged from V3-EXQ-876).
Only C1 is load-bearing for evidence_direction; C2-C6 are positive-control /
substrate-health gates.

precision_ratio note: {precision_ratio_note}

Criteria met: {criteria_met}/6 -> **{status}** (label: {label})
{failure_section}
"""

    return {
        "status": status,
        "metrics": metrics,
        "summary_markdown": summary_markdown,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": evidence_direction,
        "experiment_type": EXPERIMENT_TYPE,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "supersedes": SUPERSEDES,
        "fatal_error_count": eval_out["fatal_errors"],
        "precision_ratio_note": precision_ratio_note,
        "interpretation": {
            "label": label,
            "preconditions": preconditions,
            "criteria_non_degenerate": {"C1": non_degenerate},
            "criteria": [
                {"name": "C1_doing_mode_delta_reversed", "load_bearing": True, "passed": bool(c1_pass)},
                {"name": "C2_committed_step_count", "load_bearing": False, "passed": bool(c2_pass)},
                {"name": "C3_uncommitted_step_count", "load_bearing": False, "passed": bool(c3_pass)},
                {"name": "C4_world_forward_r2", "load_bearing": False, "passed": bool(c4_pass)},
                {"name": "C5_harm_pred_std", "load_bearing": False, "passed": bool(c5_pass)},
                {"name": "C6_no_fatal_errors", "load_bearing": False, "passed": bool(c6_pass)},
            ],
            "combination_rule": "PASS iff ALL SIX criteria hold (plain AND); only C1 load-bearing for evidence_direction.",
        },
        "non_degenerate": non_degenerate,
        "degeneracy_reason": degeneracy["degeneracy_reason"],
        # dataclasses.asdict: REEConfig (and its nested sub-configs) are plain
        # @dataclass trees of scalars -- asdict() gives a JSON-serializable
        # Mapping, which is what stamp_recording_core's `config` param expects.
        # (V3-EXQ-876's bug was passing `config=None` here instead.)
        "config": dataclasses.asdict(config),
    }


def _run_multi_seed(seeds, **run_kwargs) -> dict:
    """Run across multiple seeds, aggregate metrics, produce combined result."""
    per_seed = {}
    all_metrics_keys = None
    last_config = None

    for seed in seeds:
        print(f"\n{'='*60}", flush=True)
        print(f"  Seed {seed}", flush=True)
        print(f"{'='*60}", flush=True)
        result = run(seed=seed, **run_kwargs)
        per_seed[seed] = result
        last_config = result.get("config")
        if all_metrics_keys is None:
            all_metrics_keys = list(result["metrics"].keys())

    agg_metrics = {}
    for key in all_metrics_keys:
        vals = [per_seed[s]["metrics"][key] for s in seeds]
        agg_metrics[key] = _mean_safe(vals)

    all_pass = all(per_seed[s]["status"] == "PASS" for s in seeds)
    overall_status = "PASS" if all_pass else "FAIL"
    total_criteria = sum(int(per_seed[s]["metrics"]["criteria_met"]) for s in seeds)
    max_criteria = 6 * len(seeds)

    if all_pass:
        evidence_direction = "supports"
    elif total_criteria >= 4 * len(seeds):
        evidence_direction = "mixed"
    else:
        evidence_direction = "weakens"

    # If EVERY seed self-routed non_contributory (P0 unmet or degenerate),
    # the aggregate should read as non_contributory too, not "weakens" --
    # that would misreport an unready substrate as claim pressure.
    labels = {per_seed[s]["interpretation"]["label"] for s in seeds}
    if labels <= {"substrate_not_ready_requeue", "c1_causal_sig_degenerate_vacuous_test"}:
        evidence_direction = "non_contributory"
        overall_status = "FAIL"

    seed_lines = []
    for s in seeds:
        m = per_seed[s]["metrics"]
        seed_lines.append(
            f"| {s} | {per_seed[s]['status']} | {int(m['criteria_met'])}/6 |"
            f" {m['doing_mode_delta']:+.4f} |"
            f" {int(m['committed_step_count'])} |"
            f" {int(m['uncommitted_step_count'])} |"
            f" {m['world_forward_r2']:.4f} |"
            f" {m['harm_pred_std']:.4f} |"
            f" {m['precision_ratio']:.2f} |"
        )
    seed_table = "\n".join(seed_lines)

    failure_notes = []
    for s in seeds:
        if per_seed[s]["status"] != "PASS":
            for line in per_seed[s]["summary_markdown"].split("\n"):
                if line.startswith("- C") or line.startswith("- P0") or line.startswith("- DEGENERATE"):
                    if "FAIL" in line or "P0 FAIL" in line or "DEGENERATE" in line:
                        failure_notes.append(f"seed {s}: {line.strip('- ')}")

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(f"- {n}" for n in failure_notes)

    bp = run_kwargs.get("breath_period", 50)
    bsa = run_kwargs.get("breath_sweep_amplitude", 0.30)
    bsd = run_kwargs.get("breath_sweep_duration", 10)

    summary_markdown = f"""# V3-EXQ-876a -- MECH-025: Action-Doing Mode Convergence Redesign

**Overall Status:** {overall_status}  ({total_criteria}/{max_criteria} criteria across {len(seeds)} seeds)
**Claim:** MECH-025 -- action-doing mode produces a distinct (convergent) internal causal signature
**Redesign of:** V3-EXQ-876 (measurement_test_design_defect per
  failure_autopsy_mech025-cluster-876-671b_2026-08-03 -- C1's predicted sign
  contradicted the claim's own cited literature)
**Fresh seeds (disjoint from V3-EXQ-876's [42,123,7,99,256], per GOV-REUSE-1 anti-circularity):** {seeds}
**BreathOscillator:** period={bp}, amplitude={bsa}, duration={bsd}

## Per-Seed Results

| Seed | Status | Criteria | doing_mode_delta | n_committed | n_uncommitted | wf_r2 | harm_std | precision_ratio |
|------|--------|----------|-------------------|-------------|-----------------|-------|----------|-----------------|
{seed_table}

## Aggregate Metrics

| Metric | Value |
|--------|-------|
| doing_mode_delta (mean) | {agg_metrics['doing_mode_delta']:+.4f} |
| committed_step_count (mean) | {agg_metrics['committed_step_count']:.0f} |
| uncommitted_step_count (mean) | {agg_metrics['uncommitted_step_count']:.0f} |
| world_forward_r2 (mean) | {agg_metrics['world_forward_r2']:.4f} |
| harm_pred_std (mean) | {agg_metrics['harm_pred_std']:.4f} |
| sweep_step_count (mean) | {agg_metrics['sweep_step_count']:.0f} |
| precision_ratio (mean, NON-GATING) | {agg_metrics['precision_ratio']:.2f} |
{failure_section}
"""

    return {
        "status": overall_status,
        "metrics": agg_metrics,
        "per_seed_metrics": {str(s): per_seed[s]["metrics"] for s in seeds},
        "per_seed_status": {str(s): per_seed[s]["status"] for s in seeds},
        "per_seed_interpretation": {str(s): per_seed[s]["interpretation"] for s in seeds},
        "summary_markdown": summary_markdown,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": evidence_direction,
        "experiment_type": EXPERIMENT_TYPE,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "supersedes": SUPERSEDES,
        "fatal_error_count": int(agg_metrics.get("fatal_error_count", 0)),
        "interpretation": {
            "label": (
                "doing_mode_produces_convergent_causal_signature"
                if overall_status == "PASS"
                else "doing_mode_does_not_produce_convergent_causal_signature"
            ),
        },
        "_full_config": last_config,
    }


if __name__ == "__main__":
    import argparse
    import time
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=str, default="11,17,29,53,71")
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--eval-eps", type=int, default=50)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--alpha-world", type=float, default=0.9)
    parser.add_argument("--harm-scale", type=float, default=0.02)
    parser.add_argument("--nav-bias", type=float, default=0.45)
    parser.add_argument("--breath-period", type=int, default=50)
    parser.add_argument("--breath-amplitude", type=float, default=0.30)
    parser.add_argument("--breath-duration", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true", help="Quick validation run")
    args = parser.parse_args()

    t0 = time.perf_counter()
    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    run_kwargs = dict(
        warmup_episodes=args.warmup,
        eval_episodes=args.eval_eps,
        steps_per_episode=args.steps,
        alpha_world=args.alpha_world,
        harm_scale=args.harm_scale,
        nav_bias=args.nav_bias,
        breath_period=args.breath_period,
        breath_sweep_amplitude=args.breath_amplitude,
        breath_sweep_duration=args.breath_duration,
    )

    if args.dry_run:
        run_kwargs["warmup_episodes"] = 5
        run_kwargs["eval_episodes"] = 3
        run_kwargs["steps_per_episode"] = 30
        seeds = [11]
        print("[DRY RUN] Minimal config for smoke test", flush=True)

    result = _run_multi_seed(seeds, **run_kwargs)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"] = ts
    result["claim"] = CLAIM_IDS[0]
    result["verdict"] = result["status"]
    result["run_id"] = f"{EXPERIMENT_TYPE}_{ts}_v3"
    result["architecture_epoch"] = "ree_hybrid_guardrails_v1"

    full_config = result.pop("_full_config", None)

    out_path = None
    if not args.dry_run:
        out_path = write_flat_manifest(
            result,
            out_dir=None,
            dry_run=False,
            config=full_config,
            seeds=seeds,
            script_path=Path(__file__),
            started_at=t0,
            z_goal_stream_stats=_ZG.stats(),
        )
        print(f"\nResult written to: {out_path}", flush=True)

    print(f"Status: {result['status']}", flush=True)
    for k, v in result["metrics"].items():
        print(f"  {k}: {v}", flush=True)

    if not args.dry_run:
        _outcome_raw = str(result.get("status", "FAIL")).upper()
        emit_outcome(
            outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
            manifest_path=out_path,
        )
    else:
        # Even in --dry-run, emit so the runner-conformance sentinel machinery
        # (and any harness checking for it) sees a clean exit signal, matching
        # V3-EXQ-199/876's dry-run convention (no manifest written at all in
        # dry-run mode -- manifest_path=None).
        _outcome_raw = str(result.get("status", "FAIL")).upper()
        emit_outcome(
            outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
            manifest_path=None,
            dry_run=True,
        )
