"""
V3-EXQ-671b -- MECH-025b: Precision-Responsibility Attribution Linkage
(CORRECTED -- precision-variance positive-control gate + multi-seed;
 supersedes V3-EXQ-671a)

Claims: MECH-025b

Motivation (2026-06-11, corrected 2026-08-02, re-corrected 2026-08-02):
  MECH-025b: "High-precision action mode carries responsibility attribution:
  the precision level at which an action was committed determines the degree
  of ethical accountability assigned to its consequences."

  Decomposed from MECH-025 (2026-04-02). MECH-025 established that doing mode
  produces distinct internal signatures (EXQ-050 PASS). MECH-025b extends this:
  not just binary committed/uncommitted, but DEGREE of precision should modulate
  responsibility weight.

WHY THIS RETEST (confirmed failure autopsy, not a re-run of the same code):
  V3-EXQ-671a (failure_autopsy_V3-EXQ-671a_2026-08-02, confirmed, user-routed
  to /queue-experiment) FAILed discrimination cleanly (C1 correlation=0.0505,
  C2 ratio=0.9214, n=29, single seed) AFTER root-causing and fixing 671's
  degenerate defect (agent.update_residue() never called -- residue field
  mathematically pinned at 0). The residue-accumulation positive-control gate
  671a added CLEARED (1.86 vs 1e-6 floor), confirming that fix genuinely
  worked. But the autopsy identified an ASYMMETRIC GAP in 671a's own
  measurement logic: a readiness gate was added for residue (the dependent
  variable) but NOT for precision (the independent variable), despite this
  being the EXACT historical failure signature for this substrate --
  MECH-025's own V2 record: "E3 precision hardcoded, no dynamic channel."
  Whether precision varies enough WITHIN a single eval run to give the C1/C2
  correlation/ratio tests any power was never checked. Separately, n=29 from
  a single seed is underpowered for a Pearson correlation and a median-split
  ratio (autopsy Section 4, "scale: underpowered").

  Substrate-level check performed before writing this script (per the
  autopsy's own instruction not to assume "still hardcoded" without reading
  the code): E3TrajectorySelector.current_precision (e3_selector.py:600-603)
  is `1.0 / (self._running_variance + 1e-6)` -- a genuinely DYNAMIC,
  running-prediction-error-derived signal (ARC-016), not a config constant.
  ARC-016 itself is `stable` with real V3 confirmation (V3-EXQ-018b PASS:
  precision 718 in a stable env vs 426 in a perturbed env). So the V2-era
  "hardcoded, no dynamic channel" defect does NOT still hold at the substrate
  level -- but ARC-016's evidence is CROSS-ENVIRONMENT-CONDITION variance,
  not the WITHIN-a-single-29-committed-step-eval-run variance this
  correlation test actually needs. That is exactly the gap this script closes
  with a measurement, not an assumption.

TWO CORRECTIONS OVER 671a (same design, same env, same 6 PASS criteria,
same thresholds -- this is a lettered fix per CLAUDE.md's EXQ convention,
not a new question):

  (a) PRECISION-VARIANCE POSITIVE-CONTROL GATE (new in 671b, mirrors the
      residue-accumulation gate 671a already has). Before C1/C2 are read as
      evidence:
        - P0 scalar check: pooled precision_samples range (max-min across
          ALL committed samples, ALL seeds) must clear a floor. Floor set
          empirically (see PRECISION_VARIANCE_FLOOR comment below) from a
          real smoke measurement of this exact substrate/environment, not
          guessed.
        - Secondary, GROUP-mode check_degeneracy net on precision_samples
          GROUPED BY SEED (not pooled). This is the one 671a's flat
          check_degeneracy on residue_delta_samples could not catch even if
          added flat: a flat/pooled spread check can pass purely from
          BETWEEN-seed differences in average precision (e.g. seed 0 always
          ~700, seed 1 always ~420) while precision is still pinned WITHIN
          every individual seed's own committed-step population -- which
          would defeat the correlation test's actual premise (that precision
          varies across events WITHIN a run) exactly as thoroughly as a
          flat-zero-spread case, just less obviously. This is the identical
          "cross-seed variance masks within-seed pinning" failure class
          metric_groups_are_degenerate's own docstring names (the V3-EXQ-603
          / 543e bit-identical-arms family) -- applied here to the
          INDEPENDENT variable of a correlation test rather than to a
          multi-arm comparison.

  (b) MULTI-SEED (new in 671b): 4 seeds (SEEDS = [0, 1, 2, 3], full
      warmup+eval pipeline per seed, fresh agent each seed) instead of 671a's
      single seed=0. Primary C1/C2 statistics are computed on samples POOLED
      across all 4 seeds (n~4x single-seed n, giving the correlation/ratio
      tests real statistical power per the autopsy's routing note). Per-seed
      C1/C2 values are ALSO computed and reported as non-gating diagnostics,
      so a reviewer can see whether the pooled read is consistent across
      seeds or driven by one seed (the same transparency the corpus's
      existing majority-of-seeds pattern, e.g. v3_exq_431, provides -- this
      script additionally reports it rather than gating PASS on it, because
      the claim under test is about a within-run relationship, not an
      across-condition contrast).

Experimental design (methodologically UNCHANGED from 671a otherwise -- same
env, same train loop, same 6 PASS criteria, same thresholds):
  - Train agent with functional E2 (world_forward) + E3 (harm_eval + precision)
  - During eval, track precision at each committed step (E3.current_precision)
  - Measure residue accumulation per step (ResidueField changes)
  - Compare residue-per-harm in high-precision vs low-precision regimes

Key metrics (computed on POOLED samples across all seeds):
  precision_residue_correlation: Pearson r(precision, residue_accumulated)
  high_precision_residue_ratio: mean residue/harm when precision > median
                                 vs mean residue/harm when precision <= median

PASS criteria (ALL must hold, computed on pooled samples):
  C1: precision_residue_correlation > 0.15  (positive correlation exists)
  C2: high_precision_residue_ratio > 1.1    (high-precision steps accumulate
                                              proportionally more residue)
  C3: committed_step_count >= 20             (sufficient samples)
  C4: world_forward_r2 > 0.05                (E2 attribution functional,
                                               mean across seeds)
  C5: harm_pred_std > 0.01                   (E3 not collapsed, mean across
                                               seeds)
  C6: No fatal errors (summed across seeds)

Same 4 instrument fixes as 671a are retained unchanged (see 671a's own
docstring / git history for the full root-cause trace):
  (1) agent.update_residue(...) called after every env.step().
  (2) is_committed sourced from SelectionResult.committed, not
      agent._committed_candidates (a candidate cache flag).
  (3) owned= sourced from info["transition_type"]=="agent_caused_hazard".
  (4) eval-sample filter is harm_signal < 0 AND owned (genuine, agent-caused
      harm only), not abs(harm_signal) > 1e-6 (which pooled in benefit
      ticks that can never produce a residue delta).
"""

import sys
import random
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


EXPERIMENT_TYPE = "v3_exq_671b_mech025b_precision_responsibility"
CLAIM_IDS = ["MECH-025b"]
SUPERSEDES = "v3_exq_671a_mech025b_precision_responsibility_20260802T035128Z_v3"
SEEDS = [0, 1, 2, 3]
MIN_SEEDS_PASS = 3  # majority-of-4, matching corpus convention (e.g. v3_exq_431) -- diagnostic only, not gating

RESIDUE_FLOOR = 1e-6  # P0: sum(residue_delta_samples) must clear this to be non-vacuous

# P0: pooled precision_samples range (max-min across ALL committed samples,
# ALL seeds) must clear this floor. Empirically calibrated (2026-08-02) from a
# real smoke measurement of THIS substrate/environment/config at
# warmup=50/eval=10/steps=200 (the same smoke scale used to validate the gate
# logic below): observed pooled range ~120-200 (current_precision units,
# 1/running_variance scale) across 4 seeds with committed, agent-owned,
# genuine-harm samples. Set the floor an order of magnitude below the
# smallest observed value to leave headroom for a smaller committed-sample
# count at full scale while still cleanly separating "some real spread" from
# "pinned/near-constant" (671a's own RESIDUE_FLOOR uses the same
# floor-well-below-observed convention: 1e-6 vs an observed 1.86).
PRECISION_VARIANCE_FLOOR = 1.0


def _action_to_onehot(action_idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, action_idx] = 1.0
    return v


def _mean_safe(lst: List[float]) -> float:
    return float(sum(lst) / len(lst)) if lst else 0.0


def _pearson_correlation(x: List[float], y: List[float]) -> float:
    """Compute Pearson correlation coefficient between two lists."""
    if len(x) < 2 or len(y) < 2 or len(x) != len(y):
        return 0.0
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    denom_x = sum((x[i] - mean_x) ** 2 for i in range(n))
    denom_y = sum((y[i] - mean_y) ** 2 for i in range(n))
    if denom_x < 1e-9 or denom_y < 1e-9:
        return 0.0
    return numerator / ((denom_x * denom_y) ** 0.5)


def _high_low_ratio(
    precision_samples: List[float],
    residue_delta_samples: List[float],
    harm_magnitude_samples: List[float],
) -> float:
    if len(precision_samples) < 4:
        return 0.0
    median_precision = sorted(precision_samples)[len(precision_samples) // 2]
    high_prec_residue = []
    low_prec_residue = []
    for i in range(len(precision_samples)):
        ratio = residue_delta_samples[i] / max(harm_magnitude_samples[i], 1e-6)
        if precision_samples[i] > median_precision:
            high_prec_residue.append(ratio)
        else:
            low_prec_residue.append(ratio)
    mean_high = _mean_safe(high_prec_residue)
    mean_low = _mean_safe(low_prec_residue)
    return mean_high / max(mean_low, 1e-6)


def _train(
    agent: REEAgent,
    env: CausalGridWorldV2,
    optimizer: optim.Optimizer,
    wf_optimizer: optim.Optimizer,
    harm_eval_optimizer: optim.Optimizer,
    num_episodes: int,
    steps_per_episode: int,
    world_dim: int,
) -> Dict:
    """Standard full-pipeline training to get functional E3 + E2.world_forward.

    UNCHANGED from V3-EXQ-671/671a: C4 (world_forward_r2) and C5
    (harm_pred_std) were NOT degenerate in either prior attempt -- the defect
    was isolated to the eval loop's residue wiring, not training.
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


def _eval_precision_responsibility(
    agent: REEAgent,
    env: CausalGridWorldV2,
    num_episodes: int,
    steps_per_episode: int,
    world_dim: int,
) -> Dict:
    """
    Probe precision-responsibility linkage by tracking:
      - precision at each committed step (E3.current_precision)
      - residue accumulated per step (ResidueField.total_residue delta)
      - harm magnitude per step

    Tests MECH-025b: high-precision actions should accumulate proportionally
    more residue (responsibility weight) than low-precision actions.

    UNCHANGED from 671a (see module docstring for the full root-cause trace
    of the 4 instrument fixes over the original 671):
      (1) calls agent.update_residue(...) after every env.step().
      (2) is_committed is sourced from SelectionResult.committed.
      (3) owned= is sourced from info["transition_type"]=="agent_caused_hazard".
      (4) the eval-sample filter is harm_signal < 0 AND owned.
    """
    agent.eval()
    precision_samples: List[float] = []
    residue_delta_samples: List[float] = []
    harm_magnitude_samples: List[float] = []
    all_harm_preds: List[float] = []
    fatal = 0
    committed_step_count = 0
    total_residue_start = float(agent.residue_field.total_residue.item())

    for _ in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        z_self_prev: Optional[torch.Tensor] = None
        action_prev: Optional[torch.Tensor] = None
        held_committed = False
        residue_prev = float(agent.residue_field.total_residue.item())

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
                if ticks.get("e3_tick", False) and candidates:
                    with torch.no_grad():
                        result = agent.e3.select(candidates, temperature=1.0)
                        action = result.selected_action.detach()
                        agent._last_action = action
                        held_committed = bool(result.committed)
                else:
                    action = agent._last_action
                    if action is None:
                        action = _action_to_onehot(
                            random.randint(0, env.action_dim - 1), env.action_dim, agent.device
                        )
                        agent._last_action = action

                is_committed = held_committed

                if is_committed:
                    with torch.no_grad():
                        current_precision = agent.e3.current_precision
                        h_pred = float(agent.e3.harm_eval(latent.z_world).item())
                        all_harm_preds.append(h_pred)

                flat_obs, harm_signal, done, info, obs_dict = env.step(action)

                owned = info.get("transition_type") == "agent_caused_hazard"

                agent.update_residue(
                    harm_signal=float(harm_signal),
                    world_delta=None,
                    hypothesis_tag=False,
                    owned=owned,
                )
                residue_curr = float(agent.residue_field.total_residue.item())
                residue_delta = residue_curr - residue_prev
                residue_prev = residue_curr

                if is_committed and owned and harm_signal < -1e-6:
                    committed_step_count += 1
                    precision_samples.append(current_precision)
                    residue_delta_samples.append(residue_delta)
                    harm_magnitude_samples.append(abs(harm_signal))

            except Exception:
                fatal += 1
                action = _action_to_onehot(
                    random.randint(0, env.action_dim - 1), env.action_dim, agent.device
                )
                agent._last_action = action
                flat_obs, harm_signal, done, info, obs_dict = env.step(action)
                residue_prev = float(agent.residue_field.total_residue.item())

            z_self_prev = latent.z_self.detach()
            action_prev = action.detach()
            if done:
                break

    total_residue_end = float(agent.residue_field.total_residue.item())
    total_residue_accumulated = total_residue_end - total_residue_start

    harm_pred_std = (
        float(torch.tensor(all_harm_preds).std().item()) if len(all_harm_preds) > 1 else 0.0
    )

    return {
        "precision_samples": precision_samples,
        "residue_delta_samples": residue_delta_samples,
        "harm_magnitude_samples": harm_magnitude_samples,
        "committed_step_count": committed_step_count,
        "harm_pred_std": harm_pred_std,
        "fatal_errors": fatal,
        "total_residue_accumulated": total_residue_accumulated,
    }


def _run_one_seed(
    seed: int,
    warmup_episodes: int,
    eval_episodes: int,
    steps_per_episode: int,
    alpha_world: float,
    alpha_self: float,
    harm_scale: float,
    proximity_scale: float,
    lr: float,
    self_dim: int,
    world_dim: int,
    zg_accum: Optional["ZGoalStreamAccumulator"] = None,
) -> Dict:
    torch.manual_seed(seed)
    random.seed(seed)

    env = CausalGridWorldV2(
        seed=seed,
        size=12,
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

    # Runner boundary line (RE_SEED_CONDITION): resets episodes_in_run at the
    # start of each seed's run. Single condition ("single") since this driver
    # is not multi-arm.
    print(f"Seed {seed} Condition single", flush=True)

    train_out = _train(
        agent,
        env,
        optimizer,
        wf_optimizer,
        harm_eval_optimizer,
        warmup_episodes,
        steps_per_episode,
        world_dim,
    )
    world_forward_r2 = _compute_world_forward_r2(agent, train_out["wf_buf"])

    eval_out = _eval_precision_responsibility(
        agent, env, eval_episodes, steps_per_episode, world_dim
    )

    seed_correlation = _pearson_correlation(
        eval_out["precision_samples"], eval_out["residue_delta_samples"]
    )
    seed_ratio = _high_low_ratio(
        eval_out["precision_samples"],
        eval_out["residue_delta_samples"],
        eval_out["harm_magnitude_samples"],
    )
    precision_spread = (
        max(eval_out["precision_samples"]) - min(eval_out["precision_samples"])
        if eval_out["precision_samples"]
        else 0.0
    )

    print(
        f"  [seed {seed}] world_forward_r2={world_forward_r2:.4f}"
        f"  harm_pred_std={eval_out['harm_pred_std']:.4f}"
        f"  committed={eval_out['committed_step_count']}"
        f"  precision_spread={precision_spread:.4f}"
        f"  corr={seed_correlation:.4f}  ratio={seed_ratio:.4f}"
        f"  residue_accum={eval_out['total_residue_accumulated']:.6f}"
        f"  fatal={eval_out['fatal_errors']}",
        flush=True,
    )
    # Per-seed runner completion signal (diagnostic only -- the SCIENTIFIC
    # verdict is the pooled one printed at the end of run(); this is the
    # per-seed proxy the runner's progress bar / ETA needs one of per
    # seed x condition unit, per the progress-instrumentation contract).
    seed_verdict_pass = (
        seed_correlation > 0.15
        and seed_ratio > 1.1
        and eval_out["committed_step_count"] >= 5
        and eval_out["fatal_errors"] == 0
    )
    print(f"verdict: {'PASS' if seed_verdict_pass else 'FAIL'}", flush=True)

    # z_goal stream liveness (Experimental Recording Standard, always-core
    # adjacent) -- observe AFTER stepping, before the agent goes out of scope.
    # MECH-025b does not itself exercise GoalState; recorded for completeness
    # and so an absent block never has to be diagnosed as "unmeasured" later.
    if zg_accum is not None:
        zg_accum.observe(agent)

    return {
        "seed": seed,
        "world_forward_r2": world_forward_r2,
        "harm_pred_std": eval_out["harm_pred_std"],
        "committed_step_count": eval_out["committed_step_count"],
        "precision_samples": eval_out["precision_samples"],
        "residue_delta_samples": eval_out["residue_delta_samples"],
        "harm_magnitude_samples": eval_out["harm_magnitude_samples"],
        "precision_spread": precision_spread,
        "seed_correlation": seed_correlation,
        "seed_ratio": seed_ratio,
        "total_residue_accumulated": eval_out["total_residue_accumulated"],
        "fatal_errors": eval_out["fatal_errors"],
    }


def run(
    seeds: Optional[List[int]] = None,
    warmup_episodes: int = 500,
    eval_episodes: int = 50,
    steps_per_episode: int = 200,
    alpha_world: float = 0.9,
    alpha_self: float = 0.3,
    harm_scale: float = 0.02,
    proximity_scale: float = 0.05,
    lr: float = 1e-3,
    self_dim: int = 32,
    world_dim: int = 32,
    **kwargs,
) -> dict:
    seeds = list(SEEDS) if seeds is None else list(seeds)
    full_config = {
        "seeds": seeds,
        "warmup_episodes": warmup_episodes,
        "eval_episodes": eval_episodes,
        "steps_per_episode": steps_per_episode,
        "alpha_world": alpha_world,
        "alpha_self": alpha_self,
        "harm_scale": harm_scale,
        "proximity_scale": proximity_scale,
        "lr": lr,
        "self_dim": self_dim,
        "world_dim": world_dim,
        "env": "CausalGridWorldV2",
        "env_size": 12,
        "env_num_hazards": 4,
        "env_num_resources": 5,
        "residue_floor": RESIDUE_FLOOR,
        "precision_variance_floor": PRECISION_VARIANCE_FLOOR,
    }

    print(
        f"[V3-EXQ-671b] MECH-025b: Precision-Responsibility Attribution "
        f"(precision-variance positive-control gate + {len(seeds)}-seed pooling)\n"
        f"  seeds={seeds}  warmup={warmup_episodes}  eval={eval_episodes}"
        f"  alpha_world={alpha_world}",
        flush=True,
    )

    _zg_accum = ZGoalStreamAccumulator()
    per_seed_results = []
    for seed in seeds:
        print(f"\n[V3-EXQ-671b] Running seed={seed}...", flush=True)
        per_seed_results.append(
            _run_one_seed(
                seed,
                warmup_episodes,
                eval_episodes,
                steps_per_episode,
                alpha_world,
                alpha_self,
                harm_scale,
                proximity_scale,
                lr,
                self_dim,
                world_dim,
                zg_accum=_zg_accum,
            )
        )

    # --- Pool across seeds for the primary (gating) C1/C2 statistics -------
    pooled_precision: List[float] = []
    pooled_residue_delta: List[float] = []
    pooled_harm_magnitude: List[float] = []
    per_seed_precision_groups: List[List[float]] = []
    per_seed_residue_groups: List[List[float]] = []
    for r in per_seed_results:
        pooled_precision.extend(r["precision_samples"])
        pooled_residue_delta.extend(r["residue_delta_samples"])
        pooled_harm_magnitude.extend(r["harm_magnitude_samples"])
        per_seed_precision_groups.append(r["precision_samples"])
        per_seed_residue_groups.append(r["residue_delta_samples"])

    precision_residue_correlation = _pearson_correlation(pooled_precision, pooled_residue_delta)
    high_precision_residue_ratio = _high_low_ratio(
        pooled_precision, pooled_residue_delta, pooled_harm_magnitude
    )
    committed_step_count = len(pooled_precision)
    world_forward_r2 = _mean_safe([r["world_forward_r2"] for r in per_seed_results])
    harm_pred_std = _mean_safe([r["harm_pred_std"] for r in per_seed_results])
    fatal_errors = sum(r["fatal_errors"] for r in per_seed_results)
    total_residue_accumulated = sum(r["total_residue_accumulated"] for r in per_seed_results)

    per_seed_diagnostics = [
        {
            "seed": r["seed"],
            "committed_step_count": r["committed_step_count"],
            "precision_spread": r["precision_spread"],
            "correlation": r["seed_correlation"],
            "ratio": r["seed_ratio"],
            "world_forward_r2": r["world_forward_r2"],
            "harm_pred_std": r["harm_pred_std"],
        }
        for r in per_seed_results
    ]
    c1_seeds_positive = sum(1 for r in per_seed_results if r["seed_correlation"] > 0.15)
    c2_seeds_positive = sum(1 for r in per_seed_results if r["seed_ratio"] > 1.1)

    print(
        f"\n[V3-EXQ-671b] POOLED (n={committed_step_count} across {len(seeds)} seeds): "
        f"precision_residue_correlation={precision_residue_correlation:.4f}"
        f"  high_precision_residue_ratio={high_precision_residue_ratio:.4f}"
        f"  total_residue_accumulated={total_residue_accumulated:.6f}",
        flush=True,
    )
    print(
        f"  per-seed diagnostic: {c1_seeds_positive}/{len(seeds)} seeds C1>0.15, "
        f"{c2_seeds_positive}/{len(seeds)} seeds C2>1.1 (non-gating)",
        flush=True,
    )

    # --- POSITIVE-CONTROL GATES ---------------------------------------------
    # P0-1 (unchanged from 671a): residue must actually have moved under
    # committed, agent-owned harm, pooled across all seeds.
    # P0-2 (NEW in 671b): precision must show adequate spread, pooled across
    # all seeds, before C1/C2 are read as evidence at all.
    pooled_precision_spread = (
        max(pooled_precision) - min(pooled_precision) if pooled_precision else 0.0
    )
    readiness_checks = [
        {
            "name": "residue_accumulates_under_committed_harm",
            "measured": sum(pooled_residue_delta),
            "threshold": RESIDUE_FLOOR,
            "direction": "lower",
            "control": (
                "sum of ResidueField.total_residue deltas across committed, "
                "agent-owned harm-events during eval, pooled across all seeds "
                "-- must be > 0 for the C1/C2 precision<->residue readout to "
                "be non-vacuous (failure_autopsy_batch9_2026-06-12 root cause)."
            ),
        },
        {
            "name": "precision_shows_adequate_variance",
            "measured": pooled_precision_spread,
            "threshold": PRECISION_VARIANCE_FLOOR,
            "direction": "lower",
            "control": (
                "max-min spread of E3.current_precision across committed, "
                "agent-owned, genuine-harm samples, pooled across all seeds "
                "-- must clear a floor for the C1/C2 test to have power over "
                "its independent variable (failure_autopsy_V3-EXQ-671a_"
                "2026-08-02 root cause: 'asymmetric positive-control gap', "
                "the exact historical failure signature per MECH-025's own "
                "V2 record: 'E3 precision hardcoded, no dynamic channel')."
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

    # Secondary nets: even if the pooled scalars clear their floors, the
    # discriminative metrics C1 actually reads must have usable spread --
    # checked BOTH flat (pooled) and GROUPED BY SEED. The grouped check on
    # precision_samples is the one 671a's flat-only design could not provide:
    # it catches within-seed pinning that a pooled/flat spread check can miss
    # when it is driven entirely by between-seed mean differences.
    degeneracy = check_degeneracy(
        {
            "C1_residue_delta_samples": pooled_residue_delta,
            "C1_precision_samples_within_seed": {"groups": per_seed_precision_groups},
        }
    )
    non_degenerate = degeneracy["non_degenerate"]

    # PASS / FAIL (criteria unchanged from 671/671a, computed on pooled samples)
    c1_pass = precision_residue_correlation > 0.15
    c2_pass = high_precision_residue_ratio > 1.1
    c3_pass = committed_step_count >= 20
    c4_pass = world_forward_r2 > 0.05
    c5_pass = harm_pred_std > 0.01
    c6_pass = fatal_errors == 0

    all_pass = c1_pass and c2_pass and c3_pass and c4_pass and c5_pass and c6_pass
    criteria_met = sum([c1_pass, c2_pass, c3_pass, c4_pass, c5_pass, c6_pass])

    if not ready:
        status = "FAIL"
        evidence_direction = "non_contributory"
        label = "substrate_not_ready_requeue"
    elif not non_degenerate:
        status = "FAIL"
        evidence_direction = "non_contributory"
        label = "c1_degenerate_vacuous_test"
    elif all_pass:
        status = "PASS"
        evidence_direction = "supports"
        label = "precision_modulates_residue_responsibility_weight"
    else:
        status = "FAIL"
        evidence_direction = "mixed" if criteria_met >= 4 else "weakens"
        label = "precision_does_not_modulate_residue_responsibility_weight"

    failure_notes = []
    if not ready:
        for p in preconditions:
            if not p.get("met", True):
                failure_notes.append(
                    f"P0 FAIL: {p['name']} unmet -- measured={p.get('measured')} "
                    f"vs threshold={p.get('threshold')} (substrate_not_ready_requeue)"
                )
    if ready and not non_degenerate:
        failure_notes.append(f"DEGENERATE: {degeneracy['degeneracy_reason']}")
    if not c1_pass:
        failure_notes.append(
            f"C1 FAIL: precision_residue_correlation={precision_residue_correlation:.4f} <= 0.15"
        )
    if not c2_pass:
        failure_notes.append(
            f"C2 FAIL: high_precision_residue_ratio={high_precision_residue_ratio:.4f} <= 1.1"
        )
    if not c3_pass:
        failure_notes.append(f"C3 FAIL: committed_step_count={committed_step_count} < 20")
    if not c4_pass:
        failure_notes.append(f"C4 FAIL: world_forward_r2={world_forward_r2:.4f} <= 0.05")
    if not c5_pass:
        failure_notes.append(f"C5 FAIL: harm_pred_std={harm_pred_std:.4f} <= 0.01")
    if not c6_pass:
        failure_notes.append(f"C6 FAIL: fatal_errors={fatal_errors}")

    print(f"\nV3-EXQ-671b verdict: {status}  label={label}  ({criteria_met}/6)", flush=True)
    for note in failure_notes:
        print(f"  {note}", flush=True)

    metrics = {
        "precision_residue_correlation": float(precision_residue_correlation),
        "high_precision_residue_ratio": float(high_precision_residue_ratio),
        "committed_step_count": float(committed_step_count),
        "harm_pred_std": float(harm_pred_std),
        "world_forward_r2": float(world_forward_r2),
        "fatal_error_count": float(fatal_errors),
        "n_samples": float(committed_step_count),
        "total_residue_accumulated": float(total_residue_accumulated),
        "pooled_precision_spread": float(pooled_precision_spread),
        "n_seeds": float(len(seeds)),
        "c1_seeds_positive": float(c1_seeds_positive),
        "c2_seeds_positive": float(c2_seeds_positive),
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

    per_seed_table = "\n".join(
        f"| {d['seed']} | {d['committed_step_count']} | {d['precision_spread']:.4f} | "
        f"{d['correlation']:.4f} | {d['ratio']:.4f} | {d['world_forward_r2']:.4f} | "
        f"{d['harm_pred_std']:.4f} |"
        for d in per_seed_diagnostics
    )

    summary_markdown = f"""# V3-EXQ-671b -- MECH-025b: Precision-Responsibility Attribution (corrected)

**Status:** {status}
**Label:** {label}
**Claim:** MECH-025b -- high-precision action mode carries responsibility attribution
**Prerequisite:** MECH-025 (doing mode produces internal signature)
**Supersedes:** V3-EXQ-671a (asymmetric positive-control gap on the independent
variable + underpowered single-seed n=29, per failure_autopsy_V3-EXQ-671a_2026-08-02)
**alpha_world:** {alpha_world}
**Warmup:** {warmup_episodes} eps/seed  |  Eval: {eval_episodes} eps/seed
**Seeds:** {seeds}

## Motivation

MECH-025b tests the philosophical bridge: does precision level modulate
responsibility weight? Actions committed at higher precision should accumulate
proportionally more residue (ethical accountability) than low-precision actions,
because high-precision implies the agent had finer discrimination capacity.

## Positive-Control Gates

| Check | Measured | Floor | Met |
|---|---|---|---|
| residue_accumulates_under_committed_harm | {sum(pooled_residue_delta):.6g} | {RESIDUE_FLOOR:g} | {ready} |
| precision_shows_adequate_variance (pooled) | {pooled_precision_spread:.6g} | {PRECISION_VARIANCE_FLOOR:g} | {pooled_precision_spread > PRECISION_VARIANCE_FLOOR} |

C1 non-degeneracy (pooled residue_delta_samples spread + per-seed precision_samples groups):
{non_degenerate} ({degeneracy['degeneracy_reason'] or 'ok'})

## Pooled Results (n={committed_step_count}, {len(seeds)} seeds)

| Metric | Value |
|--------|-------|
| Precision-Residue Correlation | {precision_residue_correlation:.4f} |
| High/Low Precision Residue Ratio | {high_precision_residue_ratio:.4f} |
| Committed Steps Sampled | {committed_step_count} |
| World Forward R2 (mean across seeds) | {world_forward_r2:.4f} |
| Harm Pred Std (mean across seeds) | {harm_pred_std:.4f} |
| Total Residue Accumulated | {total_residue_accumulated:.6f} |

## Per-Seed Diagnostics (non-gating)

| Seed | Committed | Precision Spread | Correlation | Ratio | WF R2 | Harm Std |
|---|---|---|---|---|---|---|
{per_seed_table}

{c1_seeds_positive}/{len(seeds)} seeds individually show C1>0.15; {c2_seeds_positive}/{len(seeds)} show C2>1.1.

## PASS Criteria (computed on pooled samples)

| Criterion | Result | Value |
|---|---|---|
| C1: precision_residue_correlation > 0.15 | {"PASS" if c1_pass else "FAIL"} | {precision_residue_correlation:.4f} |
| C2: high_precision_residue_ratio > 1.1 | {"PASS" if c2_pass else "FAIL"} | {high_precision_residue_ratio:.4f} |
| C3: committed_step_count >= 20 | {"PASS" if c3_pass else "FAIL"} | {committed_step_count} |
| C4: world_forward_r2 > 0.05 | {"PASS" if c4_pass else "FAIL"} | {world_forward_r2:.4f} |
| C5: harm_pred_std > 0.01 | {"PASS" if c5_pass else "FAIL"} | {harm_pred_std:.4f} |
| C6: No fatal errors | {"PASS" if c6_pass else "FAIL"} | {fatal_errors} |

Criteria met: {criteria_met}/6 -> **{status}** (label: {label})
{failure_section}
"""

    return {
        "status": status,
        "metrics": metrics,
        "summary_markdown": summary_markdown,
        "claim_ids": CLAIM_IDS,
        "supersedes": SUPERSEDES,
        "evidence_direction": evidence_direction,
        "experiment_type": EXPERIMENT_TYPE,
        "fatal_error_count": fatal_errors,
        "interpretation": {
            "label": label,
            "preconditions": preconditions,
            "criteria_non_degenerate": {"C1": non_degenerate},
            "per_seed_diagnostics": per_seed_diagnostics,
            "criteria": [
                {"name": "C1_precision_residue_correlation", "load_bearing": True, "passed": bool(c1_pass)},
                {"name": "C2_high_precision_residue_ratio", "load_bearing": True, "passed": bool(c2_pass)},
                {"name": "C3_committed_step_count", "load_bearing": False, "passed": bool(c3_pass)},
                {"name": "C4_world_forward_r2", "load_bearing": False, "passed": bool(c4_pass)},
                {"name": "C5_harm_pred_std", "load_bearing": False, "passed": bool(c5_pass)},
                {"name": "C6_no_fatal_errors", "load_bearing": False, "passed": bool(c6_pass)},
            ],
        },
        "non_degenerate": non_degenerate,
        "degeneracy_reason": degeneracy["degeneracy_reason"],
        "config": full_config,
        "seeds_used": seeds,
        "z_goal_stream_stats": _zg_accum.stats(),
    }


if __name__ == "__main__":
    import argparse
    import time
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--eval-eps", type=int, default=50)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--alpha-world", type=float, default=0.9)
    parser.add_argument("--harm-scale", type=float, default=0.02)
    parser.add_argument("--dry-run", action="store_true", help="Quick validation run")
    args = parser.parse_args()

    if args.dry_run:
        print("[DRY RUN] Quick validation mode", flush=True)
        args.warmup = 5
        args.eval_eps = 2
        args.steps = 50
        args.seeds = args.seeds or SEEDS[:2]

    _t0 = time.perf_counter()
    result = run(
        seeds=args.seeds,
        warmup_episodes=args.warmup,
        eval_episodes=args.eval_eps,
        steps_per_episode=args.steps,
        alpha_world=args.alpha_world,
        harm_scale=args.harm_scale,
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"] = ts
    result["claim"] = CLAIM_IDS[0]
    result["verdict"] = result["status"]
    result["run_id"] = f"{EXPERIMENT_TYPE}_{ts}_v3"
    result["architecture_epoch"] = "ree_hybrid_guardrails_v1"

    out_path = None
    if not args.dry_run:
        out_path = write_flat_manifest(
            result,
            out_dir=None,
            dry_run=False,
            config=result.get("config"),
            seeds=result.get("seeds_used", SEEDS),
            script_path=Path(__file__),
            started_at=_t0,
            z_goal_stream_stats=result.get("z_goal_stream_stats"),
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
